"""导出 replay buffer 为 SFP 轨迹 npz.

输出 npz 格式：
  breaks: object 数组，每个元素是 1D 时间戳数组（归一化 0~1）
  values: object 数组，每个元素形状 (T, D) 的动作序列
  prior: 1D 浮点数组，可选；默认均匀

用法示例：
  python scripts/replay_to_sfp.py \
    --zarr-path diffusion_policy_3d/dataset/data/your.zarr \
    --key action \
    --out trajectories_sfp.npz
"""

import argparse
import os
import numpy as np

from diffusion_policy_3d.common.replay_buffer import ReplayBuffer


def export(zarr_path: str, key: str, out: str, max_episodes: int = None):
    rb = ReplayBuffer.copy_from_path(zarr_path, keys=[key])
    n = rb.n_episodes if max_episodes is None else min(max_episodes, rb.n_episodes)

    breaks_list = []
    values_list = []
    for idx in range(n):
        arr = rb[key][idx]  # shape (T, D)
        T = arr.shape[0]
        breaks = np.linspace(0.0, 1.0, T)
        breaks_list.append(breaks.astype(np.float64))
        values_list.append(arr.astype(np.float64))

    np.savez(out, breaks=np.array(breaks_list, dtype=object), values=np.array(values_list, dtype=object))
    print(f"saved {n} trajectories to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", required=True, help="replay buffer zarr 路径")
    parser.add_argument("--key", default="action", help="要导出的字段，如 action")
    parser.add_argument("--out", required=True, help="输出 npz 文件")
    parser.add_argument("--max-episodes", type=int, default=None, help="最多导出多少条 episode")
    args = parser.parse_args()

    if not os.path.exists(args.zarr_path):
        raise FileNotFoundError(args.zarr_path)

    export(args.zarr_path, args.key, args.out, args.max_episodes)


if __name__ == "__main__":
    main()
