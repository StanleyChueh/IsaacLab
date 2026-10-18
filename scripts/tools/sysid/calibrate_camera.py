# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Solve a sim camera's mount pose by fitting the sim robot's silhouette to the real one.

The real and sim images look nothing alike (black arm on a black mat vs a white arm on a white pad), so no
appearance is compared. What both share is WHERE the robot is: render_real_states.py posed the sim robot
at each real frame's joint state and saved points sampled over its visual meshes plus every link's pose;
segment_robot_real.py masked the robot in the real frames. For a candidate camera mount (pose in its
parent link, which is fixed to the torso) the points are projected through the REAL intrinsics into a
sim silhouette, and the mount is moved until the silhouettes agree across all frames at once.

The score is a truncated two-sided chamfer distance (sim pixels outside the real mask, real pixels the sim
does not cover, each weighted by how far it is to the other mask), which has a wide basin unlike plain
IoU. It is minimised coarse-to-fine, from the mount poses.npz was rendered with and from random restarts
around it.

  python scripts/tools/sysid/calibrate_camera.py logs/real2sim/body_cam_render_K/poses.npz \\
      logs/real2sim/body_cam_frames/robot_masks.npz --intrinsics 607.24,607.21,326.43,238.68 \\
      --frames logs/real2sim/body_cam_frames/frames.npz --out logs/real2sim/body_cam_fit

Prints IoU per frame before/after and the fitted CameraCfg.OffsetCfg; writes camera_fit.json and
fit_XX.png (real frame; outlines: real robot red, fitted sim green, starting mount magenta). The starting
mount is whatever poses.npz was rendered with, so re-rendering with a fit and re-running refines it.
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def quat_wxyz_to_matrix(q) -> np.ndarray:
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def matrix_to_quat_wxyz(R) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if w >= 0 else -q


class SilhouetteFit:
    def __init__(self, poses, masks: np.ndarray, K: np.ndarray, device: str):
        self.device = device
        n, self.H, self.W = masks.shape
        link, xyz = poses["surface_link"], poses["surface_xyz"]
        R_wp = quat_wxyz_to_matrix(poses["parent_quat_w"][0])
        t_wp = poses["parent_pos_w"][0]
        pts = []
        for k in range(n):  # sampled surface, posed at frame k, in the camera's parent-link frame
            R_b = Rotation.from_quat(poses["body_quat_w"][k][:, [1, 2, 3, 0]]).as_matrix()
            world = np.einsum("nij,nj->ni", R_b[link], xyz) + poses["body_pos_w"][k][link]
            pts.append((world - t_wp) @ R_wp)
        self.pts = torch.tensor(np.stack(pts), dtype=torch.float32, device=device)  # (n, M, 3)
        self.K = torch.tensor(K, dtype=torch.float32, device=device)
        self.real = masks
        self.real_dist = np.stack([cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5) for m in masks])

    def silhouettes(self, R_pc: np.ndarray, t_pc: np.ndarray, scale: int = 1, chunk: int = 16) -> np.ndarray:
        """Sim robot masks (n, H/scale, W/scale) for the camera at (R_pc, t_pc) in the parent link."""
        R = torch.tensor(R_pc, dtype=torch.float32, device=self.device)
        t = torch.tensor(t_pc, dtype=torch.float32, device=self.device)
        H, W = self.H // scale, self.W // scale
        masks = []
        for pts in self.pts.split(chunk):  # bounded GPU memory however many frames
            cam = (pts - t) @ R  # parent -> camera (ROS optical: x right, y down, z forward)
            z = cam[..., 2].clamp(min=1e-3)
            u = (self.K[0, 0] * cam[..., 0] / z + self.K[0, 2]) / scale
            v = (self.K[1, 1] * cam[..., 1] / z + self.K[1, 2]) / scale
            ok = (cam[..., 2] > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            frame = torch.arange(len(pts), device=self.device)[:, None].expand_as(u)
            out = torch.zeros(len(pts), H * W, device=self.device)
            out[frame[ok], v[ok].long() * W + u[ok].long()] = 1.0
            out = torch.nn.functional.max_pool2d(out.view(-1, 1, H, W), 3, 1, 1)
            out = -torch.nn.functional.max_pool2d(-torch.nn.functional.max_pool2d(out, 5, 1, 2), 5, 1, 2)  # close holes
            masks.append(out[:, 0].bool().cpu().numpy())
        return np.concatenate(masks)

    def cost(self, R_pc, t_pc, scale: int, tau: float) -> float:
        sim = self.silhouettes(R_pc, t_pc, scale)
        total = 0.0
        for k, s in enumerate(sim):
            real = self.real[k][::scale, ::scale]
            d_real = self.real_dist[k][::scale, ::scale] / scale
            d_sim = cv2.distanceTransform((~s).astype(np.uint8), cv2.DIST_L2, 5)
            total += np.minimum(d_real[s], tau).sum() + np.minimum(d_sim[real], tau).sum()
        return total * scale * scale / (len(sim) * self.H * self.W)

    def iou(self, R_pc, t_pc) -> np.ndarray:
        sim = self.silhouettes(R_pc, t_pc)
        return np.array([(s & r).sum() / max((s | r).sum(), 1) for s, r in zip(sim, self.real)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("poses", help="poses.npz from render_real_states.py")
    ap.add_argument("masks", help="robot_masks.npz from segment_robot_real.py")
    ap.add_argument("--intrinsics", required=True, help="Real camera fx,fy,cx,cy (rs-enumerate-devices -c).")
    ap.add_argument("--frames", default=None, help="frames.npz, for the fit_XX.png previews.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--restarts", type=int, default=6, help="Random starts besides the cfg's mount.")
    ap.add_argument("--spread_cm", type=float, default=8.0)
    ap.add_argument("--spread_deg", type=float, default=8.0)
    args = ap.parse_args()

    poses = np.load(args.poses)
    masks = np.load(args.masks)["masks"]
    if len(masks) != len(poses["body_pos_w"]):
        raise SystemExit(f"{len(masks)} masks but {len(poses['body_pos_w'])} rendered frames.")
    fx, fy, cx, cy = (float(v) for v in args.intrinsics.split(","))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    fit = SilhouetteFit(poses, masks, K, "cuda" if torch.cuda.is_available() else "cpu")

    R0 = quat_wxyz_to_matrix(poses["cam_offset_rot"])
    t0 = np.array(poses["cam_offset_pos"], dtype=np.float64)

    def unpack(x):  # x = [dx, dy, dz (cm, parent frame), rx, ry, rz (deg, camera frame)]
        return R0 @ Rotation.from_rotvec(np.radians(x[3:])).as_matrix(), t0 + x[:3] / 100.0

    def solve(x, scale, tau, iters):
        simplex = np.vstack([x, x + np.eye(6) * scale])  # first steps: `scale` cm / deg per axis
        res = minimize(lambda x: fit.cost(*unpack(x), scale, tau), x, method="Nelder-Mead",
                       options={"maxfev": iters, "xatol": 0.05, "fatol": 1e-5, "initial_simplex": simplex})
        return res.x, res.fun

    rng = np.random.default_rng(0)
    starts = [np.zeros(6)] + [np.r_[rng.normal(0, args.spread_cm, 3), rng.normal(0, args.spread_deg, 3)]
                              for _ in range(args.restarts)]
    coarse = []
    for i, x in enumerate(starts):
        x, f = solve(x, 4, 40.0, 600)
        coarse.append((f, x))
        print(f"  start {i}: coarse cost {f:.4f}  offset {np.round(x[:3], 1)} cm  {np.round(x[3:], 1)} deg")
    coarse.sort(key=lambda c: c[0])
    best = None
    for f, x in coarse[:3]:  # refine the three best at full resolution
        x, f = solve(x, 2, 20.0, 800)
        x, f = solve(x, 1, 10.0, 800)
        if best is None or f < best[0]:
            best = (f, x)
    x = best[1]
    R_pc, t_pc = unpack(x)
    q = matrix_to_quat_wxyz(R_pc)

    iou0, iou1 = fit.iou(R0, t0), fit.iou(R_pc, t_pc)
    print(f"\nIoU, starting mount -> fitted mount (same intrinsics in both), over {len(iou0)} frames:")
    for name, f in (("mean", np.mean), ("median", np.median), ("10th pct", lambda a: np.percentile(a, 10))):
        print(f"  {name:8s}: {f(iou0):.3f} -> {f(iou1):.3f}")
    worst = np.argsort(iou1)[:8]
    print("  worst frames: " + ", ".join(f"{k} ({iou1[k]:.2f})" for k in worst))
    turned = np.degrees(Rotation.from_matrix(R0.T @ R_pc).magnitude())
    print(f"\nmount change vs starting mount: {np.round(x[:3], 1)} cm (parent frame, |{np.linalg.norm(x[:3]):.1f}| cm), {turned:.1f} deg")
    print(f"offset=CameraCfg.OffsetCfg(pos=({t_pc[0]:.5f}, {t_pc[1]:.5f}, {t_pc[2]:.5f}), "
          f"rot=({q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}, {q[3]:.5f}), convention=\"ros\")")
    print(f"render check: --intrinsics {fx},{fy},{cx},{cy} --cam_pos {t_pc[0]:.5f},{t_pc[1]:.5f},{t_pc[2]:.5f} "
          f"--cam_rot {q[0]:.5f},{q[1]:.5f},{q[2]:.5f},{q[3]:.5f}")

    os.makedirs(args.out, exist_ok=True)
    json.dump({"parent_link": str(poses["parent_link"]), "pos": t_pc.tolist(), "rot_wxyz_ros": q.tolist(),
               "intrinsics": [fx, fy, cx, cy], "iou_before": iou0.tolist(), "iou_after": iou1.tolist()},
              open(os.path.join(args.out, "camera_fit.json"), "w"), indent=2)
    if args.frames:
        images = np.load(args.frames)["images"]
        sim = fit.silhouettes(R_pc, t_pc)
        sim0 = fit.silhouettes(R0, t0)
        for k, img in enumerate(images):
            vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for m, color in ((sim0[k], (255, 0, 255)), (masks[k], (0, 0, 255)), (sim[k], (0, 255, 0))):
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(vis, contours, -1, color, 1)
            cv2.imwrite(os.path.join(args.out, f"fit_{k:02d}.png"), vis)
        print(f"previews: {args.out}/fit_XX.png  (red = real robot, green = fitted sim, magenta = starting mount)")


if __name__ == "__main__":
    main()
