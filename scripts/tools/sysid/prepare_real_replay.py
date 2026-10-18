# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Prepare a paired real/sim replay of a real dataset: every episode's joint trajectory and where its can started.

Writes <out>/trajectories.npz (per-episode ``state_E`` / ``action_E`` (T,16), ``names``, ``fps``) and
<out>/can_starts.json ({episode: [x, y]} in the sim env frame, metres). The can's start is found in the body camera's
first frames: the can is segmented (Grounding DINO + SAM 2.1), the lowest point of its mask is the near edge of its
base, that pixel's ray is intersected with the pad plane ``--table_z`` and moved back one can radius along the ray.
Replaying the real actions in sim with the can placed there gives sim episodes whose actions, states and scene layout
equal the real ones, so any difference between a sim and a real frame is purely how it looks.

Plain python in lerobot_openarm's venv:
  python scripts/tools/sysid/prepare_real_replay.py ~/.cache/huggingface/lerobot/ethanCSL/openarm_pringles_lerobot_real_v00 \\
      --mount logs/real2sim/body_cam_fit_v2/camera_fit.json --out logs/real2sim/replay
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_states import decode_episodes, load_episodes  # noqa: E402
from segment_robot_real import Segmenter  # noqa: E402

CAN_RADIUS = 0.0293  # m, openarm_task_modes.CAN_RADIUS
SIM_K = (607.22, 607.22, 320.0, 240.0)  # the lens the body-cam mount was fitted with (centred principal point)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--mount", required=True, help="camera_fit.json of the body camera.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--table_z", type=float, default=0.28)
    ap.add_argument("--frame", type=int, default=3, help="Frame index whose can mask is used.")
    args = ap.parse_args()

    episodes, names, fps = load_episodes(args.root, "body_cam")
    os.makedirs(args.out, exist_ok=True)
    traj = {"names": np.array(names), "fps": fps}
    import pandas as pd

    data = pd.concat(pd.read_parquet(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(args.root, "data"))
                     for f in sorted(fs) if f.endswith(".parquet"))
    for e in episodes:
        rows = data[data.episode_index == e["episode"]].sort_values("frame_index")
        traj[f"state_{e['episode']}"] = np.stack(rows["observation.state"].to_numpy()).astype(np.float32)
        traj[f"action_{e['episode']}"] = np.stack(rows["action"].to_numpy()).astype(np.float32)
    np.savez_compressed(os.path.join(args.out, "trajectories.npz"), **traj)

    mount = json.load(open(args.mount))
    q = mount["rot_wxyz_ros"]
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t = np.array(mount["pos"])
    seg = Segmenter("cuda")
    starts = {}
    for e, frames in decode_episodes(episodes, fps, gray=False):
        img = frames[min(args.frame, len(frames) - 1)]
        m = seg(img, "pringles can.", 0.3, 0.15)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)) > 0
        if m.sum() < 800:
            print(f"episode {e['episode']:2d}: no can found")
            continue
        ys, xs = np.nonzero(m)
        vb = ys.max()
        ub = xs[ys > vb - 6].mean()
        ray = R @ np.array([(ub - SIM_K[2]) / SIM_K[0], (vb - SIM_K[3]) / SIM_K[1], 1.0])
        hit = t + ray * (args.table_z - t[2]) / ray[2]
        horiz = ray[:2] / np.linalg.norm(ray[:2])
        centre = hit[:2] + horiz * CAN_RADIUS
        starts[str(e["episode"])] = [float(centre[0]), float(centre[1])]
        print(f"episode {e['episode']:2d}: can at x={centre[0]:.3f} y={centre[1]:.3f}", flush=True)
    json.dump(starts, open(os.path.join(args.out, "can_starts.json"), "w"), indent=1)
    xy = np.array(list(starts.values()))
    print(f"{len(starts)} / {len(episodes)} episodes; x {xy[:, 0].min():.2f}..{xy[:, 0].max():.2f}, y {xy[:, 1].min():.2f}..{xy[:, 1].max():.2f}")


if __name__ == "__main__":
    main()
