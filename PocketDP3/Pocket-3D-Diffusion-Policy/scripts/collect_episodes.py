"""Rollout runner that writes replay data into a zarr for DP3/SFP."""

from __future__ import annotations

import argparse
import os
from typing import Callable

import gym
import numpy as np

from pocket_diffusion_policy_3d.common.replay_buffer import ReplayBuffer


def random_policy(_: np.ndarray, action_space: gym.Space) -> np.ndarray:
    return action_space.sample()


def collect_episode(
    env: gym.Env,
    step_fn: Callable[[np.ndarray], np.ndarray],
    max_steps: int,
) -> np.ndarray:
    obs = env.reset()
    actions = []
    done = False

    while not done and len(actions) < max_steps:
        action = step_fn(obs)
        obs, _, done, _ = env.step(action)
        actions.append(np.asarray(action, dtype=np.float32))

    if not actions:
        raise RuntimeError("no action recorded for the episode")

    return np.stack(actions, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect action-only episodes into a zarr.")
    parser.add_argument("--env", required=True, help="gym environment id (e.g., 'CartPole-v1')")
    parser.add_argument("--zarr-path", required=True, help="Destination zarr directory")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to record")
    parser.add_argument("--max-steps", type=int, default=500, help="Cap for each episode length")
    parser.add_argument("--policy", choices=["random"], default="random", help="Action policy")
    parser.add_argument("--seed", type=int, default=0, help="Environment/numpy seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    env = gym.make(args.env)
    env.seed(args.seed)

    if args.policy == "random":
        step_fn = lambda obs: random_policy(obs, env.action_space)
    else:  # pragma: no cover
        raise NotImplementedError(f"policy {args.policy} is not implemented")

    buffer = ReplayBuffer.create_empty_zarr()
    for episode in range(args.episodes):
        actions = collect_episode(env, step_fn, args.max_steps)
        buffer.add_episode({"action": actions})
        print(f"episode {episode + 1}/{args.episodes} length={actions.shape[0]}")

    buffer.save_to_path(args.zarr_path)
    print("saved replay buffer to", args.zarr_path)


if __name__ == "__main__":
    main()