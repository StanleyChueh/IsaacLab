# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Recover what the fixed body camera sees with the robot and the can taken away (a "clean plate").

The body camera never moves, so every pixel the arms uncover at some point in the recordings shows the same
static scene. Each frame's robot is masked by projecting the sim robot's surface (posed by forward
kinematics at that frame's joint state, through the calibrated camera) and dilating it to cover cables and
calibration error; the per-pixel median over the unmasked frames then removes the can too, which never
stands on one spot for long. Pixels the robot never uncovers are inpainted -- the robot covers them in
every image anyway.

Plain python (numpy, opencv, pandas, av):
  python scripts/tools/sysid/build_clean_plate.py ~/.cache/huggingface/lerobot/ethanCSL/openarm_pringles_lerobot_real_v00 \\
      --surface logs/real2sim/look/body_iter0/poses.npz --mount logs/real2sim/body_cam_fit_v2/camera_fit.json \\
      --out logs/real2sim/look/body_plate
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_states import decode_episodes, load_episodes, real_state_to_sim  # noqa: E402
from urdf_fk import URDFChain  # noqa: E402


def quat_wxyz_to_matrix(q):
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--surface", required=True, help="Any poses.npz from render_real_states.py (for its surface samples).")
    ap.add_argument("--mount", required=True, help="camera_fit.json of the body camera.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", default="body_cam")
    ap.add_argument("--every", type=int, default=6, help="Use every n-th frame.")
    ap.add_argument("--dilate", type=int, default=25, help="px added around the projected robot (cables, error).")
    ap.add_argument("--calibration", default=os.path.expanduser("~/Stanley_ws/lerobot_openarm/calibration.json"))
    args = ap.parse_args()

    calib = json.load(open(args.calibration))
    chain = URDFChain()
    surf = np.load(args.surface)
    bodies = [str(b) for b in surf["body_names"]]
    link_of, xyz = surf["surface_link"], surf["surface_xyz"].astype(np.float64)
    used_links = [i for i in np.unique(link_of) if bodies[i] in chain.joints or bodies[i] == "openarm_body_link0"]
    mount = json.load(open(args.mount))
    R_pc, t_pc = quat_wxyz_to_matrix(mount["rot_wxyz_ros"]), np.array(mount["pos"])
    K = surf["intrinsics"]  # the renderer's effective lens the mount was fitted with (centred principal point)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.dilate + 1,) * 2)

    episodes, names, fps = load_episodes(args.root, args.camera)
    stack, masks = [], []
    for e, frames in decode_episodes(episodes, fps, gray=False, every=args.every):
        for k, img in frames:
            q = real_state_to_sim(e["state"][min(k, len(e["state"]) - 1)], names, calib)
            pts = []
            for li in used_links:
                T = chain.pose(bodies[li], q, base=mount["parent_link"])
                p = xyz[link_of == li]
                pts.append(p @ T[:3, :3].T + T[:3, 3])
            cam = (np.concatenate(pts) - t_pc) @ R_pc
            z = cam[:, 2]
            ok = z > 0.05
            u = (K[0, 0] * cam[ok, 0] / z[ok] + K[0, 2]).astype(int)
            v = (K[1, 1] * cam[ok, 1] / z[ok] + K[1, 2]).astype(int)
            m = np.zeros(img.shape[:2], np.uint8)
            inb = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
            m[v[inb], u[inb]] = 1
            m = cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)), kernel) > 0
            stack.append(img)
            masks.append(m)
        print(f"  episode {e['episode']:2d}: {len(stack)} frames", flush=True)

    stack, masks = np.stack(stack), np.stack(masks)  # uint8 / bool: ~2 GB for 2000 frames
    plate = np.zeros(stack.shape[1:], np.float32)
    for r in range(0, stack.shape[1], 24):  # nanmedian in row bands: one band's float copy at a time
        band = stack[:, r:r + 24].astype(np.float32)
        band[masks[:, r:r + 24]] = np.nan
        plate[r:r + 24] = np.nanmedian(band, axis=0)
    seen = (~masks).sum(0)
    holes = (seen < 5).astype(np.uint8)
    plate = np.nan_to_num(plate).clip(0, 255).astype(np.uint8)
    plate = cv2.inpaint(plate, holes, 7, cv2.INPAINT_TELEA)
    os.makedirs(args.out, exist_ok=True)
    cv2.imwrite(os.path.join(args.out, "clean_plate.png"), cv2.cvtColor(plate, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.out, "coverage.png"), np.clip(seen * 255.0 / max(seen.max(), 1), 0, 255).astype(np.uint8))
    np.save(os.path.join(args.out, "never_seen.npy"), holes.astype(bool))
    print(f"{len(stack)} frames; {holes.mean() * 100:.1f}% of pixels never uncovered (inpainted) -> {args.out}/clean_plate.png")


if __name__ == "__main__":
    main()
