# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Metric depth of the static scene the body camera sees, and the table plane in the robot's frame.

The dataset has no depth, so a monocular model (Depth Anything V2, metric indoor) predicts it per real
frame. Its scale is only roughly right, but each frame also shows the robot, whose true depth is known: the
sim robot posed at that frame's joint state, rendered through the calibrated camera (render_real_states.py's
depth). Fitting depth_true = a * predicted + b on the pixels where the real and the sim robot agree turns each
prediction metric; the static pixels of all frames are then combined (median) into one depth map for the
clean plate. A RANSAC plane through the mat pixels gives the table height and tilt in the camera's parent
frame, to compare with the sim's workspace pad.

Needs transformers (lerobot_openarm's venv):
  python scripts/tools/sysid/body_cam_depth.py logs/real2sim/look/body_iter0/poses.npz \\
      logs/real2sim/body_cam_frames_v2/frames.npz --masks logs/real2sim/body_cam_frames_v2/robot_masks.npz \\
      --mount logs/real2sim/body_cam_fit_v2/camera_fit.json --out logs/real2sim/look/body_depth
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_robot_colors import label_maps  # noqa: E402

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


class DepthModel:
    def __init__(self, device="cuda"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.proc = AutoImageProcessor.from_pretrained(MODEL)
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        out = self.model(**self.proc(images=rgb, return_tensors="pt").to(self.device)).predicted_depth
        return torch.nn.functional.interpolate(out[:, None], rgb.shape[:2], mode="bicubic")[0, 0].cpu().numpy()


def fit_plane(P: np.ndarray, iters: int = 2000, tol: float = 0.01, seed: int = 0):
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < tol
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    Q = P[best]
    mu = Q.mean(0)
    n = np.linalg.svd(Q - mu, full_matrices=False)[2][2]
    return n if n[2] > 0 else -n, mu, best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("poses")
    ap.add_argument("frames")
    ap.add_argument("--masks", required=True)
    ap.add_argument("--mount", required=True)
    ap.add_argument("--plate", default="logs/real2sim/look/body_plate/clean_plate.png")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    poses, frames = np.load(args.poses), np.load(args.frames)
    masks = np.load(args.masks)["masks"]
    images = frames["images"]
    H, W = images.shape[1:3]
    K = poses["intrinsics"]
    model = DepthModel()

    # One scale/offset for the whole sequence: a single frame's robot pixels span too narrow a depth range to
    # pin down both, but pooled over 100+ poses they do.
    erode = np.ones((9, 9), np.uint8)
    preds, xs, ys = [], [], []
    for k, img in enumerate(images):
        pred = model(img)
        preds.append(pred)
        robot_sim = label_maps(poses, k, H, W) >= 0
        both = cv2.erode((robot_sim & masks[k]).astype(np.uint8), erode) > 0
        if both.sum() >= 2000:
            sel = np.flatnonzero(both.ravel())[::7]
            xs.append(pred.ravel()[sel])
            ys.append(poses["depth"][k].ravel()[sel])
    x, y = np.concatenate(xs), np.concatenate(ys)
    A = np.c_[x, np.ones_like(x)]
    ab = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(4):  # trim the worst 20% (segmentation / calibration misses) and refit
        r = np.abs(A @ ab - y)
        keep = r < np.percentile(r, 80)
        ab = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
    print(f"global depth alignment: scale {ab[0]:.3f}, offset {ab[1]:.3f} m, median robot-pixel error "
          f"{np.median(np.abs(A[keep] @ ab - y[keep])) * 100:.1f} cm over {keep.sum()} pixels")
    metric = []
    for k in range(len(images)):
        robot_sim = label_maps(poses, k, H, W) >= 0
        static = ~(cv2.dilate((masks[k] | robot_sim).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)
        metric.append(np.where(static, ab[0] * preds[k] + ab[1], np.nan))
    depth = np.nanmedian(np.stack(metric), axis=0)
    # Pixels no frame saw as static: take the clean plate's own prediction, scaled the median way.
    plate = cv2.cvtColor(cv2.imread(args.plate), cv2.COLOR_BGR2RGB)
    plate_pred = model(plate)
    ok = np.isfinite(depth)
    A = np.c_[plate_pred[ok], np.ones(ok.sum())]
    ab = np.linalg.lstsq(A, depth[ok], rcond=None)[0]
    depth_plate = ab[0] * plate_pred + ab[1]
    print(f"clean-plate prediction vs per-frame median: {np.median(np.abs(depth_plate[ok] - depth[ok])) * 100:.1f} cm median diff")
    depth_full = np.where(ok, depth, depth_plate)

    # Back-project into the camera's parent frame and fit the table plane through the lower image half.
    mount = json.load(open(args.mount))
    q = mount["rot_wxyz_ros"]
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t = np.array(mount["pos"])
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    P = (rays * depth_full[..., None]) @ R.T + t
    region = np.zeros((H, W), bool)
    region[200:470, 120:620] = True  # the mats, clear of the left arm's rest pose
    n, mu, inl = fit_plane(P[region & ok].reshape(-1, 3))
    tilt = np.degrees(np.arccos(min(1.0, abs(n[2]))))
    print(f"table plane in {mount['parent_link']}: height z = {mu[2]:.3f} m at ({mu[0]:.2f}, {mu[1]:.2f}), "
          f"tilt {tilt:.1f} deg, normal {np.round(n, 3)}, {inl.mean() * 100:.0f}% inliers")
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "body_depth.npz"), depth=depth_full.astype(np.float32), seen_static=ok,
                        points=P.astype(np.float32), plane_normal=n, plane_point=mu, K=K,
                        mount_pos=t, mount_rot=np.array(q), parent_link=mount["parent_link"])
    vis = cv2.applyColorMap((255 * np.clip((depth_full - 0.3) / 2.0, 0, 1)).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(args.out, "body_depth.png"), vis)
    print(f"-> {args.out}/body_depth.npz")


if __name__ == "__main__":
    main()
