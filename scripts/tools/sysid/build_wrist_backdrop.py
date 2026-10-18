# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Bake what a wrist camera sees at the arms' rest pose into a backdrop plane for the sim.

Every real episode starts from the same rest pose, where each wrist camera looks at the same static piece of
room (a window, the lab's black cabinet). The median of those first frames is that view with the can, the
operator's cables and noise removed. It is hung as a coloured quad on the camera's optical axis, placed with the
camera pose the SIM will use (forward kinematics at the rest state times the env cfg's mount) -- so in sim at
the rest pose the wrist camera reproduces the real frame pixel for pixel, whatever the real mount's true pose
is. As the arm moves away the plane is a rough stand-in for the room (depth ``--distance``), which is far
better than the empty sky the sim shows otherwise.

The foam fingertips at the left/right edges of a wrist view move with the camera and the sim renders its own
fingers there, so they are painted out of the plate first.

  python scripts/tools/sysid/build_wrist_backdrop.py ~/.cache/huggingface/lerobot/ethanCSL/openarm_pringles_lerobot_real_v00 \\
      --camera wrist_cam --out source/isaaclab_assets/data/real2sim/pringles_v00/backdrop_wrist_cam.npz
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_backdrop import depth_mesh  # noqa: E402
from real_states import decode_episodes, load_episodes, real_state_to_sim  # noqa: E402
from urdf_fk import URDFChain  # noqa: E402

LINK = {"wrist_cam": "openarm_left_ee_tcp", "right_wrist_cam": "openarm_right_ee_tcp"}
SIDE = {"wrist_cam": "L", "right_wrist_cam": "R"}
# What the env cfg gives the cameras (stack_ik_abs_visuomotor_env_cfg.py): lens fx,fy at the centred principal point
# the renderer uses, and the mount in ee_tcp (ROS convention, wxyz).
SIM_K = {"wrist_cam": (605.44, 605.13), "right_wrist_cam": (607.79, 607.61)}
CFG_MOUNT = (0.06, 0.001, 0.11125, 0.68057, -0.19188, -0.19188, 0.68057)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--camera", required=True, choices=sorted(LINK))
    ap.add_argument("--out", required=True)
    ap.add_argument("--distance", type=float, default=1.5, help="m; depth of the plane in front of the camera.")
    ap.add_argument("--first", type=int, default=6, help="Use the first n frames of every episode.")
    ap.add_argument("--rest_tol", type=float, default=0.03, help="rad; episodes starting further than this from the "
                    "median start pose of this arm are skipped.")
    ap.add_argument("--border", type=int, default=50, help="px painted out on the left/right edges (fingertips).")
    ap.add_argument("--calibration", default=os.path.expanduser("~/Stanley_ws/lerobot_openarm/calibration.json"))
    args = ap.parse_args()

    calib = json.load(open(args.calibration))
    chain = URDFChain()
    episodes, names, fps = load_episodes(args.root, args.camera)
    side = SIDE[args.camera]
    arm_cols = [names.index(f"{side}J{n}.pos") for n in range(1, 8)]
    start = np.stack([e["state"][0, arm_cols] for e in episodes])
    rest = np.median(start, 0)
    good = {e["episode"] for e, s in zip(episodes, start) if np.abs(s - rest).max() < args.rest_tol}
    print(f"{len(good)} / {len(episodes)} episodes start within {args.rest_tol} rad of the median rest pose")

    imgs = []
    for e, frames in decode_episodes([e for e in episodes if e["episode"] in good], fps, gray=False):
        imgs += frames[:args.first]
    plate = np.median(np.stack(imgs), 0).astype(np.uint8)
    H, W = plate.shape[:2]
    strip = np.zeros((H, W), np.uint8)
    strip[:, :args.border] = strip[:, W - args.border:] = 255
    plate = cv2.inpaint(plate, strip, 9, cv2.INPAINT_TELEA)

    # Camera pose in the env frame at the rest state, with the cfg's mount.
    state = np.median(np.stack([e["state"][0] for e in episodes if e["episode"] in good]), 0)
    link = chain.pose(LINK[args.camera], real_state_to_sim(state, names, calib))
    mount = np.eye(4)
    mount[:3, :3] = Rotation.from_quat([CFG_MOUNT[4], CFG_MOUNT[5], CFG_MOUNT[6], CFG_MOUNT[3]]).as_matrix()
    mount[:3, 3] = CFG_MOUNT[:3]
    cam = link @ mount

    K = np.array([[SIM_K[args.camera][0], 0, W / 2], [0, SIM_K[args.camera][1], H / 2], [0, 0, 1]])
    pts, faces, col = depth_mesh(np.full((H, W), args.distance), plate, K, 4, 1.0)
    world = pts @ cam[:3, :3].T + cam[:3, 3]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, points=world.astype(np.float32), faces=faces, colors=col.astype(np.uint8))
    cv2.imwrite(args.out.replace(".npz", "_plate.png"), cv2.cvtColor(plate, cv2.COLOR_RGB2BGR))
    print(f"camera at {np.round(cam[:3, 3], 3)}, looking along {np.round(cam[:3, 2], 2)}; plane extent "
          f"{np.round(world.min(0), 2)} .. {np.round(world.max(0), 2)} -> {args.out}")


if __name__ == "__main__":
    main()
