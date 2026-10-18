# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Measure how the RTX renderer maps an emissive backdrop colour to an output pixel, and write tone.json.

The backdrop is built from the clean plate through the same camera it is rendered by, so wherever the robot
is not in the way the sim image should equal the plate. They do not: emissive values go through the
renderer's exposure and tone mapping. This compares sim and plate pixel by pixel over the static pixels of
a render_real_states.py run (backdrop built with the plain sRGB decode, i.e. without tone.json) and writes
the per-channel curve  emissive value -> rendered pixel, which openarm_real_look.emissive_from_srgb inverts.

  python scripts/tools/sysid/fit_tone.py logs/real2sim/look/body_iter1 --plate logs/real2sim/look/body_plate/clean_plate.png \\
      --frames logs/real2sim/body_cam_frames_v2/frames.npz --masks logs/real2sim/body_cam_frames_v2/robot_masks.npz \\
      --out source/isaaclab_assets/data/real2sim/pringles_v00/tone.json
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_robot_colors import label_maps, srgb_to_linear  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("render_dir", help="render_real_states.py output folder (sim_XX.png, poses.npz)")
    ap.add_argument("--plate", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins", type=int, default=32)
    args = ap.parse_args()

    poses = np.load(os.path.join(args.render_dir, "poses.npz"))
    masks = np.load(args.masks)["masks"]
    plate = cv2.cvtColor(cv2.imread(args.plate), cv2.COLOR_BGR2RGB)
    H, W = plate.shape[:2]
    inp, out = [], []
    for k in range(0, len(masks), 3):
        sim = cv2.cvtColor(cv2.imread(os.path.join(args.render_dir, f"sim_{k:02d}.png")), cv2.COLOR_BGR2RGB)
        near = cv2.dilate(((label_maps(poses, k, H, W) >= 0) | masks[k]).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
        keep = ~near
        inp.append(plate[keep][::11])
        out.append(sim[keep][::11])
    inp, out = np.concatenate(inp), np.concatenate(out)
    edges = np.linspace(0, 255, args.bins + 1)
    centers, curve = [], [[], [], []]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (inp.mean(1) >= lo) & (inp.mean(1) < hi)  # bin by brightness, then per-channel medians
        if sel.sum() < 200:
            continue
        centers.append(np.median(inp[sel], 0))
        for c in range(3):
            curve[c].append(np.median(out[sel][:, c]))
    centers = np.array(centers)
    emissive = srgb_to_linear(np.median(centers, 1) if False else centers.mean(1))  # what the plain decode fed the renderer
    result = {"emissive": emissive.tolist(), "output": []}
    for c in range(3):
        # per-channel: the emissive value fed in was decode(plate channel), so sample that channel's own decode
        e_c = srgb_to_linear(centers[:, c])
        o = np.maximum.accumulate(np.array(curve[c]))
        result["output"].append(np.interp(emissive, e_c, o).tolist())
    json.dump(result, open(args.out, "w"), indent=1)
    print("emissive -> output (R,G,B):")
    for i in range(0, len(emissive), 4):
        print(f"  {emissive[i]:.4f} -> {result['output'][0][i]:6.1f} {result['output'][1][i]:6.1f} {result['output'][2][i]:6.1f}")
    print(f"mean |sim - plate| before: {np.abs(inp.astype(float) - out).mean():.1f} grey levels -> {args.out}")


if __name__ == "__main__":
    main()
