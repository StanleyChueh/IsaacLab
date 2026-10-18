# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Fit the sim robot's material colours to the real robot, part by part.

render_real_states.py posed the sim robot at real frames' joint states and saved, besides the sim images and
depth, points sampled over every visual mesh labelled by part (one part = one material on one link). Those
points are projected into each frame and depth-tested against the sim depth to find which part covers which
pixel; eroding each part's region keeps only pixels well inside it, so the few-pixel calibration error and
the real robot's outline do not leak the background in. For every part the real and the sim pixel colours
there are compared (median, in linear light), and the part's albedo is scaled by real / sim -- a closed loop:
re-render with the new albedos and run again to absorb how the sim's lighting shades each part.

Plain python (numpy, opencv):
  python scripts/tools/sysid/fit_robot_colors.py logs/real2sim/look/body_render/poses.npz \\
      logs/real2sim/body_cam_frames_v2/frames.npz --masks logs/real2sim/body_cam_frames_v2/robot_masks.npz \\
      --albedo source/isaaclab_tasks/.../openarm_real_look_albedo.json --out logs/real2sim/look/fit_1

Writes <out>/albedo.json (the next albedo per part, same layout as --albedo) and <out>/parts_XX.png.
"""

import argparse
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def srgb_to_linear(c):
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def label_maps(poses, k: int, H: int, W: int, depth_tol: float = 0.01) -> np.ndarray:
    """(H, W) part index of the sim robot surface seen at each pixel of frame k, -1 where none."""
    R_b = Rotation.from_quat(poses["body_quat_w"][k][:, [1, 2, 3, 0]]).as_matrix()
    link, xyz = poses["surface_link"], poses["surface_xyz"]
    world = np.einsum("nij,nj->ni", R_b[link], xyz) + poses["body_pos_w"][k][link]
    q = poses["cam_quat_w_ros"][k]
    R_c = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    cam = (world - poses["cam_pos_w"][k]) @ R_c
    K = poses["intrinsics"]
    z = cam[:, 2]
    ok = z > 0.05
    u = np.floor(K[0, 0] * cam[:, 0] / np.where(ok, z, 1) + K[0, 2]).astype(int)
    v = np.floor(K[1, 1] * cam[:, 1] / np.where(ok, z, 1) + K[1, 2]).astype(int)
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    depth = poses["depth"][k]
    ok[ok] &= np.abs(depth[v[ok], u[ok]] - z[ok]) < depth_tol  # the surface the renderer actually shows there
    labels = np.full((H, W), -1, np.int32)
    labels[v[ok], u[ok]] = poses["surface_part"][ok]
    # Fill the gaps between samples with the nearest label (3x3 majority is enough at 1M points).
    for _ in range(2):
        holes = labels < 0
        grown = cv2.dilate((labels + 1).astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32) - 1
        labels[holes] = grown[holes]
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("poses", help="poses.npz from render_real_states.py (with surface_part)")
    ap.add_argument("frames", help="frames.npz of the real frames the render was made from")
    ap.add_argument("--masks", default=None, help="robot_masks.npz (segment_robot_real.py): only use pixels inside it.")
    ap.add_argument("--albedo", required=True, help="Current per-part albedo JSON (sRGB 0..1, as authored).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--erode", type=int, default=4, help="px trimmed off each part's region.")
    ap.add_argument("--min_px", type=int, default=300, help="Parts seen in fewer pixels keep their albedo.")
    ap.add_argument("--gain_limit", type=float, default=10.0, help="Max change per iteration (either way).")
    args = ap.parse_args()

    poses = np.load(args.poses)
    images = np.load(args.frames)["images"]
    masks = np.load(args.masks)["masks"] if args.masks else None
    sim_dir = os.path.dirname(os.path.abspath(args.poses))
    parts = [str(p) for p in poses["part_names"]]
    albedo = json.load(open(args.albedo))
    n, H, W = len(images), *images.shape[1:3]

    real_px = {p: [] for p in range(len(parts))}
    sim_px = {p: [] for p in range(len(parts))}
    kernel = np.ones((2 * args.erode + 1,) * 2, np.uint8)
    os.makedirs(args.out, exist_ok=True)
    palette = np.random.default_rng(0).integers(40, 255, (len(parts), 3)).astype(np.uint8)
    for k in range(n):
        labels = label_maps(poses, k, H, W)
        sim = cv2.cvtColor(cv2.imread(os.path.join(sim_dir, f"sim_{k:02d}.png")), cv2.COLOR_BGR2RGB)
        for p in np.unique(labels[labels >= 0]):
            region = cv2.erode((labels == p).astype(np.uint8), kernel) > 0
            if masks is not None:
                region &= masks[k]
            if region.any():
                real_px[p].append(images[k][region])
                sim_px[p].append(sim[region])
        if k < 12:
            vis = palette[np.maximum(labels, 0)] * (labels >= 0)[..., None]
            cv2.imwrite(os.path.join(args.out, f"parts_{k:02d}.png"),
                        cv2.cvtColor(np.hstack([images[k], (0.5 * images[k] + 0.5 * vis).astype(np.uint8)]), cv2.COLOR_RGB2BGR))

    new = dict(albedo)
    print(f"{'part':44s} {'px':>7s}  real sRGB        sim sRGB         albedo -> new")
    for p, name in enumerate(parts):
        if not real_px[p] or name not in albedo:
            continue
        r, s = np.concatenate(real_px[p]), np.concatenate(sim_px[p])
        if len(r) < args.min_px:
            continue
        r_lin, s_lin = srgb_to_linear(np.median(r, 0)), srgb_to_linear(np.median(s, 0))
        # One brightness gain per part (luminance ratio), never per colour channel: with a few thousand pixels a
        # per-channel fit chases the room's colour cast into the albedo (parts turn teal), whereas the real parts
        # are neutral grey and black and the cast is the light's -- see lighting.json.
        lum = np.array([0.2126, 0.7152, 0.0722])
        gain = np.clip((lum @ r_lin + 1e-3) / (lum @ s_lin + 1e-3), 1 / args.gain_limit, args.gain_limit) * np.ones(3)
        a = np.array(albedo[name]["diffuse"])
        if lum @ a < 0.02:
            # A near-black albedo cannot be scaled into a lighter one. Add the missing linear luminance instead,
            # through the shading factor L (~0.6: a 0.98 albedo renders at ~0.6 linear under the dome light).
            a_new = np.clip(a + max(lum @ r_lin - lum @ s_lin, 0.0) / 0.6, 0.005, 0.95) * np.ones(3)
        else:
            a_new = np.clip(a * gain, 0.005, 0.95)
        new[name] = dict(albedo[name], diffuse=[round(float(v), 4) for v in a_new])
        print(f"{name:44s} {len(r):7d}  {np.median(r, 0).astype(int)!s:16s} {np.median(s, 0).astype(int)!s:16s} "
              f"{np.round(a, 3)} -> {np.round(a_new, 3)}")
    json.dump(new, open(os.path.join(args.out, "albedo.json"), "w"), indent=1)
    print(f"-> {args.out}/albedo.json")


if __name__ == "__main__":
    main()
