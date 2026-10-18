# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Prepare a paired real/sim replay of the battery-insertion dataset: joint trajectories and where each object started.

Writes <out>/trajectories.npz (per-episode ``state_E`` / ``action_E``, ``names``, ``fps``) and <out>/object_starts.json
({episode: {"charger": [x, y, yaw_deg], "battery": [x, y, yaw_deg]}}, sim env frame, metres / degrees). Each object is
segmented (Grounding DINO + SAM 2.1) in an early frame, where the arms are at home and cannot hide it; the mask outline is
projected through the calibrated body camera onto the plane of the object's top (``--charger_top`` / battery axis height),
and the minimum-area rectangle of that outline gives centre, long-axis direction and size. The battery's known 5.05 cm
length is the check on the whole chain.

Plain python in lerobot_openarm's venv.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_states import decode_episodes, load_episodes  # noqa: E402
from segment_robot_real import Segmenter  # noqa: E402

SIM_K = (607.22, 607.22, 320.0, 240.0)
PAD_TOP = 0.28


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--mount", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--charger_top", type=float, default=0.0375, help="m, height of the charger's top face above the mat.")
    ap.add_argument("--battery_radius", type=float, default=0.00725)
    ap.add_argument("--frames", default="2,6,12,20", help="Early frames tried in order until an object is found.")
    args = ap.parse_args()

    episodes, names, fps = load_episodes(args.root, "body_cam")
    os.makedirs(args.out, exist_ok=True)
    data = pd.concat(pd.read_parquet(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(args.root, "data"))
                     for f in sorted(fs) if f.endswith(".parquet"))
    traj = {"names": np.array(names), "fps": fps}
    for e in episodes:
        rows = data[data.episode_index == e["episode"]].sort_values("frame_index")
        traj[f"state_{e['episode']}"] = np.stack(rows["observation.state"].to_numpy()).astype(np.float32)
        traj[f"action_{e['episode']}"] = np.stack(rows["action"].to_numpy()).astype(np.float32)
    np.savez_compressed(os.path.join(args.out, "trajectories.npz"), **traj)

    mount = json.load(open(args.mount))
    q = mount["rot_wxyz_ros"]
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t = np.array(mount["pos"])

    def outline_to_plane(mask, z):
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float64)
        rays = np.stack([(c[:, 0] - SIM_K[2]) / SIM_K[0], (c[:, 1] - SIM_K[3]) / SIM_K[1], np.ones(len(c))], 1) @ R.T
        P = t + rays * ((PAD_TOP + z - t[2]) / rays[:, 2:3])
        (cx, cy), (w, h), ang = cv2.minAreaRect(P[:, :2].astype(np.float32))
        if w < h:
            w, h, ang = h, w, ang + 90.0
        return np.array([cx, cy]), w, h, ((ang + 90.0) % 180.0) - 90.0  # long-axis angle in [-90, 90)

    seg = Segmenter("cuda")
    wanted = [int(k) for k in args.frames.split(",")]
    starts, sizes = {}, {"charger": [], "battery": []}
    for e, frames in decode_episodes(episodes, fps, gray=False):
        entry = {}
        for k in wanted:
            img = frames[min(k, len(frames) - 1)]
            if "charger" not in entry:
                m = seg(img, "white battery charger.", 0.3, 0.08)
                m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
                if m.sum() > 500:
                    c, w, h, ang = outline_to_plane(m, args.charger_top)
                    entry["charger"] = [float(c[0]), float(c[1]), float(ang)]
                    sizes["charger"].append([w, h])
            if "battery" not in entry:
                for prompt, thr in (("aa battery.", 0.25), ("white battery.", 0.2), ("silver cylinder.", 0.2), ("battery.", 0.2)):
                    m = seg(img, prompt, thr, 0.012)
                    m = m & ~entry.get("_c", np.zeros_like(m))
                    if m.sum() > 40:
                        c, w, h, ang = outline_to_plane(m, args.battery_radius)
                        if 0.035 < w < 0.07:  # sanity: an AA battery is 5.05 cm long
                            entry["battery"] = [float(c[0]), float(c[1]), float(ang)]
                            sizes["battery"].append([w, h])
                            break
            if len(entry) == 2:
                break
        print(f"episode {e['episode']}: " + ", ".join(f"{k} x={v[0]:.3f} y={v[1]:.3f} yaw={v[2]:+.0f}" for k, v in entry.items())
              + ("" if len(entry) == 2 else "  <-- INCOMPLETE"), flush=True)
        starts[str(e["episode"])] = entry
    json.dump(starts, open(os.path.join(args.out, "object_starts.json"), "w"), indent=1)
    for k, v in sizes.items():
        v = np.array(v) * 100
        print(f"{k}: {len(v)} detections, size median {np.median(v[:, 0]):.2f} x {np.median(v[:, 1]):.2f} cm (std {v[:, 0].std():.2f}, {v[:, 1].std():.2f})")


if __name__ == "__main__":
    main()
