# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flag generated OpenArm Mimic episodes whose motion no source demo has, and print the command that removes them.

Run it on generate_dataset.py's output before converting it to LeRobot:

    ./isaaclab.sh -p scripts/tools/check_mimic_demos.py --dataset_file logs/demos/pringles_sim_mimic_v00.hdf5

Each check is a jolt or a jump the hand-recorded source demos never show (thresholds are about twice the worst
source value, measured on logs/demos/openarm_pringles_lerobot_sim_10_v00_annotated.hdf5):

  joint windup   a joint command more than 0.3 rad from where the joint is: the IK commanding a joint past its
                 limit (before openarm_mimic_actions: elbow flips that swung the right hand up ~25 cm)
  hand jolt      a hand faster than 0.6 m/s (source demos: at most 0.48)
  hand lag       a hand more than 5 cm from its commanded pose
  can jolt       the can faster than 1.0 m/s (source demos: at most 0.57). Seen right after the giving hand lets go,
                 when the can settles into the receiving grip with a lurch

Reads only the HDF5 (no simulator). Episodes are reported as remove_demos_hdf5.py's --episodes spec.
"""

import argparse

import h5py
import numpy as np

FPS = 30
CHECKS = {
    "joint windup": 0.3,
    "hand jolt": 0.6,
    "hand lag": 0.05,
    "can jolt": 1.0,
}
_ARMS = {  # arm: (eef position obs, its 14D delta-pose slice)
    "right": ("obs/right_eef_pos", slice(7, 10)),
    "left": ("obs/eef_pos", slice(0, 3)),
}


def speed(pos: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(pos, axis=0), axis=1) * FPS


def check_episode(ep) -> dict[str, float]:
    """Worst value of each check in one episode."""
    worst = dict.fromkeys(CHECKS, 0.0)
    for arm, (pos_key, delta) in _ARMS.items():
        names = [f"openarm_{arm}_joint{i}" for i in range(1, 8)]
        target = np.stack([ep[f"lerobot/joint_pos_target/{n}"][()] for n in names], 1)
        measured = np.stack([ep[f"lerobot/joint_pos/{n}"][()] for n in names], 1)
        worst["joint windup"] = max(worst["joint windup"], float(np.abs(target - measured).max()))
        worst["hand jolt"] = max(worst["hand jolt"], float(speed(ep[pos_key][()]).max()))
        worst["hand lag"] = max(worst["hand lag"], float(np.linalg.norm(ep["actions"][:, delta], axis=1).max()))
    worst["can jolt"] = float(speed(ep["states/rigid_object/can/root_pose"][:, :3]).max())
    return worst


def episode_spec(indices: list[int]) -> str:
    """[3, 7, 8, 9] -> '3 7-9'."""
    runs, start = [], None
    for i, n in enumerate(indices):
        if start is None:
            start = n
        if i + 1 == len(indices) or indices[i + 1] != n + 1:
            runs.append(str(start) if start == n else f"{start}-{n}")
            start = None
    return " ".join(runs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset_file", required=True, help="generate_dataset.py output (HDF5)")
    args = parser.parse_args()

    with h5py.File(args.dataset_file, "r") as f:
        data = f["data"]
        names = sorted(data, key=lambda n: int(n.split("_")[-1]))
        if names and "lerobot/joint_pos_target" not in data[names[0]]:
            raise SystemExit(f"{args.dataset_file} has no lerobot/joint_pos_target: not a generate_dataset.py output.")
        flagged: dict[str, list[int]] = {c: [] for c in CHECKS}
        for name in names:
            worst = check_episode(data[name])
            for check, limit in CHECKS.items():
                if worst[check] > limit:
                    flagged[check].append(int(name.split("_")[-1]))

    bad = sorted(set(sum(flagged.values(), [])))
    print(f"{args.dataset_file}: {len(names)} episodes, {len(bad)} flagged ({100 * len(bad) / max(len(names), 1):.0f}%)")
    for check, eps in flagged.items():
        print(f"  {check:13s} > {CHECKS[check]:<5} {len(eps):4d}   {episode_spec(eps)[:100]}")
    if bad:
        out = args.dataset_file.removesuffix(".hdf5") + "_clean.hdf5"
        print("\nRemove them (writes a new file, the original stays):\n")
        print(f"./isaaclab.sh -p scripts/tools/remove_demos_hdf5.py --dataset_file {args.dataset_file} \\\n"
              f"  --episodes {episode_spec(bad)} \\\n  --output {out}")


if __name__ == "__main__":
    main()
