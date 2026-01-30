"""Convert RoboTwin hdf5 episodes into SFP-ready npz (breaks, values, prior)."""

import argparse
import glob
import os
from typing import List, Tuple

import h5py
import numpy as np


def load_action(path: str, dataset: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if dataset not in f:
            if "/" in dataset:
                grp, key = dataset.split("/", 1)
                data = f[grp][key][...]
            else:
                data = f[dataset][...]
        else:
            data = f[dataset][...]
    return np.asarray(data, dtype=np.float32)


def build_trajectories(files: List[str], dataset: str, max_episodes: int | None) -> Tuple[np.ndarray, np.ndarray]:
    chosen = files if max_episodes is None else files[:max_episodes]
    breaks_list, values_list = [], []
    for idx, fp in enumerate(chosen):
        values = load_action(fp, dataset)
        T = values.shape[0]
        breaks = np.linspace(0.0, 1.0, T, dtype=np.float64)
        breaks_list.append(breaks)
        values_list.append(values.astype(np.float64))
        print(f"[{idx + 1}/{len(chosen)}] {os.path.basename(fp)} -> T={T}, D={values.shape[1] if values.ndim>1 else 1}")
    return np.array(breaks_list, dtype=object), np.array(values_list, dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboTwin hdf5 -> SFP npz")
    parser.add_argument("--data-dir", required=True, help="Directory containing episode*.hdf5")
    parser.add_argument("--pattern", default="episode*.hdf5", help="Glob pattern under data-dir")
    parser.add_argument("--dataset", default="joint_action/vector", help="Dataset path inside hdf5")
    parser.add_argument("--out", required=True, help="Output npz path")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit number of episodes")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    if not files:
        raise FileNotFoundError(f"no hdf5 matched in {args.data_dir} with {args.pattern}")

    breaks, values = build_trajectories(files, args.dataset, args.max_episodes)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(args.out, breaks=breaks, values=values)
    print(f"saved {len(breaks)} trajectories to {args.out}")


if __name__ == "__main__":
    main()