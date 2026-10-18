# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Rectify the charger's top face from the real body-camera frames into a flat texture for the sim's charger.

For every episode, the early frame (arms at home) shows the charger from above. The charger's footprint -- centre, yaw
and size from prepare_battery_replay.py -- defines a plane at the height of its top face; each texel of the output is a
point on that plane, projected through the calibrated body camera into the frame and sampled there. The nine episodes put
the charger in slightly different places and orientations, so averaging them lowers the noise. The charger's local +x
(its long axis) is image-up for the camera, local +y is image-left (the robot's left arm is on the image's left).

  python scripts/tools/sysid/build_charger_texture.py --frames logs/real2sim/battery/home_frames/frames.npz \\
      --starts logs/real2sim/battery/replay/object_starts.json --mount logs/real2sim/body_cam_fit_v2/camera_fit.json \\
      --out source/isaaclab_assets/data/real2sim/battery_v01/charger_top.png
"""

import argparse
import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

SIM_K = (607.22, 607.22, 320.0, 240.0)
PAD_TOP = 0.28


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--starts", required=True)
    ap.add_argument("--mount", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="0.115,0.0652,0.0375", help="charger length, width, height (m)")
    ap.add_argument("--px_per_mm", type=float, default=8.0)
    args = ap.parse_args()

    L, Wd, T = (float(v) for v in args.size.split(","))
    d = np.load(args.frames)
    images, episodes = d["images"], d["episode"]
    starts = json.load(open(args.starts))
    mount = json.load(open(args.mount))
    q = mount["rot_wxyz_ros"]
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t = np.array(mount["pos"])
    w, h = int(Wd * 1000 * args.px_per_mm), int(L * 1000 * args.px_per_mm)  # columns run along local y, rows along local x
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    lx = (0.5 - (ii + 0.5) / h) * L   # local +x runs up the image (texture row 0 = +x end)
    ly = (0.5 - (jj + 0.5) / w) * Wd   # local +y runs left
    acc, n = np.zeros((h, w, 3), np.float64), 0
    for img, ep in zip(images, episodes):
        s = starts[str(int(ep))].get("charger")
        if s is None:
            continue
        cx, cy, yaw = s[0], s[1], np.radians(s[2])
        X = cx + np.cos(yaw) * lx - np.sin(yaw) * ly
        Y = cy + np.sin(yaw) * lx + np.cos(yaw) * ly
        P = np.stack([X, Y, np.full_like(X, PAD_TOP + T)], -1).reshape(-1, 3)
        cam = (P - t) @ R
        u = SIM_K[0] * cam[:, 0] / cam[:, 2] + SIM_K[2]
        v = SIM_K[1] * cam[:, 1] / cam[:, 2] + SIM_K[3]
        tex = cv2.remap(img.astype(np.float32), u.reshape(h, w).astype(np.float32), v.reshape(h, w).astype(np.float32), cv2.INTER_CUBIC)
        acc += tex
        n += 1
    tex = (acc / max(n, 1)).clip(0, 255).astype(np.uint8)
    cv2.imwrite(args.out, cv2.cvtColor(tex, cv2.COLOR_RGB2BGR))
    print(f"{n} episodes averaged -> {args.out} ({w}x{h}); mean colour {tex.reshape(-1, 3).mean(0).round(0)}")


if __name__ == "__main__":
    main()
