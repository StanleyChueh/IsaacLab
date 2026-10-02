# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Gate a generated Isaac Lab HDF5 on wrist-camera aim, before converting / training on it.

Measures where the table edge sits in the early frames of each episode (first row, from the top, where the
median brightness drops below 0.5) for wrist_cam and right_wrist_cam, and compares it with the nominal-mount
values. openarm_pringles_DR_v00 was generated WITHOUT domain randomization yet 498 of its 500 episodes had both
wrist cameras pitched ~3-6 degrees down (table edge row ~342-344 instead of ~379 / ~422 of 480), which no loss
curve or file size shows. Runs in ~seconds. Plain python: h5py + numpy.

  python check_camera_aim.py logs/demos/my_generated.hdf5 [--tol 12] [--frame 2]
Exit code 1 if either camera's mean is off nominal by more than --tol rows, or any episode is.
A set generated WITH camera domain randomization is meant to spread; use this on no-DR sets, or only to read the numbers.
"""

import argparse
import re
import sys

import h5py
import numpy as np

NOMINAL = {"wrist_cam": 379.0, "right_wrist_cam": 422.0}   # openarm_visuomotor_stanley_GR00T_500, nominal mounts


def edge_row(rgb: np.ndarray) -> int:
    med = np.median(rgb.astype(np.float32).mean(-1) / 255.0, axis=1)
    return int(np.argmax(med < 0.5)) if (med < 0.5).any() else -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hdf5")
    ap.add_argument("--tol", type=float, default=12.0, help="Allowed deviation from nominal, in image rows (of 480).")
    ap.add_argument("--frame", type=int, default=2, help="Frame to measure (arms are still at the reset pose early on).")
    args = ap.parse_args()

    bad = False
    with h5py.File(args.hdf5, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(re.sub(r"\D", "", k)))
        print(f"{args.hdf5}: {len(keys)} episodes")
        for cam, nominal in NOMINAL.items():
            rows = []
            for k in keys:
                obs = f["data"][k]["obs"]
                if cam not in obs:
                    print(f"  {cam}: not in this file's obs -- skipped")
                    rows = None
                    break
                rows.append(edge_row(obs[cam][min(args.frame, len(obs[cam]) - 1)]))
            if rows is None:
                continue
            rows = np.array(rows)
            off = np.abs(rows - nominal) > args.tol
            ok = abs(rows.mean() - nominal) <= args.tol and not off.any()
            bad |= not ok
            print(f"  {cam:16s} mean {rows.mean():6.1f} (nominal {nominal:.0f})  std {rows.std():4.1f}  "
                  f"{int(off.sum())}/{len(rows)} episodes off by > {args.tol:g} rows  -> {'OK' if ok else 'CHECK THE CAMERA MOUNT'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
