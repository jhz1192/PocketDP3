from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from pocket_diffusion_policy_3d.model.diffusion.positional_embedding import SinusoidalPosEmb

def modulate(x, shift, scale):
    if len(x.shape) == 3 and len(shift.shape) == 2:
        # [B, K, D] + [B, D]
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    elif len(x.shape) == len(shift.shape) == 3:
        # [B, K, D] + [B, K, D]
        return x * (1 + scale) + shift
    elif len(x.shape) == 4 and len(shift.shape) == 2:
        # [B, K, A, D] + [B, D]
        return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
    elif len(x.shape) == len(shift.shape) == 4:
        # [B, K, A, D] + [B, K, A, D]
        return x * (1 + scale) + shift
    else:
        raise ValueError("Invalid shapes to modulate")


class DiMBlock(nn.Module):
    def __init__(self, spatial_dim, hidden_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp_spatial = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim*4),
            nn.SiLU(),
            nn.Linear(spatial_dim*4, spatial_dim),
        )
        self.mlp_channel = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, hidden_dim),
        )
        
        # AdaLN
        self.t_modulator = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, 3*hidden_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.squeeze(1)

        y = self.norm1(x)
        y = rearrange(y, 'b t d -> b d t')
        y = self.mlp_spatial(y)
        y = rearrange(y, 'b d t -> b t d')
        x = x + y
        
        y = self.norm2(x)
        y = self.mlp_channel(y)
        x = x + y

        gamma, scale, shift = torch.chunk(self.t_modulator(t), 3, dim=-1) # [B, D] * 3
        y = modulate(x, shift, scale)
        
        x = x + y.mul_(gamma[:, None, :])
        
        return x

class DiffusionMixer(nn.Module):
    """Pure-MLP backbone that keeps the diffusion interface consistent."""

    def __init__(
        self,
        input_dim: int,
        horizon: int = 8,
        hidden_dim: int = 1024,
        depth: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        diffusion_step_embed_dim: int = 256,
        global_cond_dim: Optional[int] = None,
        local_cond_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        time_embed_dim = diffusion_step_embed_dim
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim)

        self.global_cond_proj = (
            nn.Linear(global_cond_dim, hidden_dim) if global_cond_dim is not None else None
        )
        self.local_cond_proj = (
            nn.Linear(local_cond_dim, hidden_dim) if local_cond_dim is not None else None
        )

        self.blocks = nn.ModuleList([
            DiMBlock(horizon, hidden_dim, hidden_dim * mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        bias = self._collect_bias(sample, timestep, local_cond, global_cond)
        x = self.input_proj(sample)

        for block in self.blocks:
            x = block(x, bias)

        x = self.output_proj(x)

        return x

    def _collect_bias(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        local_cond: Optional[torch.Tensor],
        global_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = sample.shape[0]
        device = sample.device
        timed_bias = self._time_bias(timestep, batch_size, device)
        bias = timed_bias

        if self.global_cond_proj is not None and global_cond is not None:
            cond = global_cond
            if cond.dim() == 3:
                cond = cond.reshape(cond.shape[0], -1)
            bias = bias + self.global_cond_proj(cond).unsqueeze(1)

        if self.local_cond_proj is not None and local_cond is not None:
            bias = bias + self.local_cond_proj(local_cond)

        return bias

    def _time_bias(self, timestep, batch_size: int, device: torch.device) -> torch.Tensor:
        timesteps = self._format_timesteps(timestep, batch_size, device)
        time_embed = self.time_embed(timesteps)
        return self.time_proj(time_embed).unsqueeze(1)

    @staticmethod
    def _format_timesteps(timestep, batch_size: int, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timesteps = torch.tensor([timestep], dtype=torch.long, device=device)
        else:
            timesteps = timestep
            if timesteps.dim() == 0:
                timesteps = timesteps.unsqueeze(0)
            timesteps = timesteps.to(device)
        return timesteps.expand(batch_size)
    