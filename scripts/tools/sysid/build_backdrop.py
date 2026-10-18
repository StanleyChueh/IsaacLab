# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Turn a camera's clean plate + metric depth into a coloured mesh of the real scene, in the sim env frame.

Every few pixels becomes a vertex at its back-projected 3D position, coloured with that pixel; neighbouring
vertices are joined into triangles except across depth jumps (an object's silhouette against the wall
behind it), which would otherwise be bridged by long stretched faces. Seen from the camera it was made from
the mesh reproduces the clean plate pixel for pixel; seen from nearby cameras (the wrist cameras) it is a
2.5D approximation of the room, correct where the surfaces really are.

openarm_real_look.py spawns every ``backdrop_*.npz`` in its look folder as an emissive mesh.

  python scripts/tools/sysid/build_backdrop.py --depth logs/real2sim/look/body_depth/body_depth.npz \\
      --image logs/real2sim/look/body_plate/clean_plate.png --poses logs/real2sim/look/body_iter0/poses.npz \\
      --out source/isaaclab_assets/data/real2sim/pringles_v00/backdrop_body.npz
"""

import argparse
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def depth_mesh(depth: np.ndarray, image: np.ndarray, K: np.ndarray, step: int, max_jump: float, valid=None):
    """Vertices (camera frame), faces, colours of a grid mesh over ``depth`` sampled every ``step`` px."""
    H, W = depth.shape
    vs, us = np.arange(0, H, step), np.arange(0, W, step)
    if vs[-1] != H - 1:
        vs = np.r_[vs, H - 1]
    if us[-1] != W - 1:
        us = np.r_[us, W - 1]
    V, U = np.meshgrid(vs, us, indexing="ij")
    z = depth[V, U]
    ok = np.isfinite(z) & (z > 0.05)
    if valid is not None:
        ok &= valid[V, U]
    X = (U + 0.5 - K[0, 2]) / K[0, 0] * z
    Y = (V + 0.5 - K[1, 2]) / K[1, 1] * z
    pts = np.stack([X, Y, z], -1).reshape(-1, 3)
    col = image[V, U].reshape(-1, 3)
    idx = np.arange(V.size).reshape(V.shape)
    faces = []
    for a, b, c in (((0, 0), (1, 0), (0, 1)), ((1, 0), (1, 1), (0, 1))):
        ia = idx[a[0]:V.shape[0] - 1 + a[0], a[1]:V.shape[1] - 1 + a[1]].ravel()
        ib = idx[b[0]:V.shape[0] - 1 + b[0], b[1]:V.shape[1] - 1 + b[1]].ravel()
        ic = idx[c[0]:V.shape[0] - 1 + c[0], c[1]:V.shape[1] - 1 + c[1]].ravel()
        f = np.stack([ia, ib, ic], 1)
        zf = z.ravel()[f]
        keep = ok.ravel()[f].all(1) & (zf.max(1) / np.maximum(zf.min(1), 1e-6) < 1 + max_jump)
        faces.append(f[keep])
    faces = np.concatenate(faces)
    used = np.unique(faces)
    remap = np.full(len(pts), -1)
    remap[used] = np.arange(len(used))
    return pts[used], remap[faces].astype(np.int32), col[used]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", required=True, help="npz with depth (H,W), K, mount_pos, mount_rot (wxyz ROS), parent_link")
    ap.add_argument("--image", required=True)
    ap.add_argument("--poses", required=True, help="poses.npz (render_real_states.py) for the parent link's env pose")
    ap.add_argument("--frame", type=int, default=0, help="Which frame of poses.npz gives the parent pose (moving parents).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--max_jump", type=float, default=0.06, help="Max relative depth change inside one triangle.")
    ap.add_argument("--max_depth", type=float, default=6.0)
    ap.add_argument("--table_z", type=float, default=0.28, help="Env-frame height the table's pixels are re-projected onto.")
    ap.add_argument("--wall_x", type=float, default=0.9, help="Env-frame x of the far wall everything off the table is flattened onto.")
    ap.add_argument("--plane_tol", type=float, default=0.03, help="m; depth-model points this close to the fitted table plane are 'table'.")
    args = ap.parse_args()

    d = np.load(args.depth)
    depth = d["depth"].astype(np.float64)
    depth[depth > args.max_depth] = np.nan
    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    pts, faces, col = depth_mesh(depth, image, d["K"], args.step, args.max_jump)

    q = d["mount_rot"]
    R_pc = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    poses = np.load(args.poses)
    bodies = [str(b) for b in poses["body_names"]]
    i = bodies.index(str(d["parent_link"]))
    qp = poses["body_quat_w"][args.frame][i]
    R_wp = Rotation.from_quat([qp[1], qp[2], qp[3], qp[0]]).as_matrix()
    t_wp = poses["body_pos_w"][args.frame][i]  # already relative to the env origin
    origin = t_wp + R_wp @ d["mount_pos"]
    world = (pts @ R_pc.T + d["mount_pos"]) @ R_wp.T + t_wp

    # Monocular depth puts the table ~15 cm higher than the sim pad (and than the grasp heights in the real
    # episodes), so it cannot be used as geometry: a mesh at that height would swallow the can and the lower arms.
    # Pixels on the table plane are re-projected along their own camera ray onto the physical pad height instead.
    # Seen from this camera nothing changes; from any other viewpoint the table is where the collider is.
    on_table = np.abs((world - (R_wp @ d["plane_point"] + t_wp)) @ (R_wp @ d["plane_normal"])) < args.plane_tol
    ray = world - origin
    s_hit = (args.table_z - origin[2]) / ray[:, 2]
    ok = on_table & (ray[:, 2] < -1e-3) & (s_hit > 0)
    ok &= world[:, 0] < args.wall_x
    world[ok] = origin + ray[ok] * s_hit[ok, None]
    # Everything else (the room beyond and beside the table) is flattened onto a vertical wall. The depth model's
    # room geometry is only right from this camera's own viewpoint -- seen from the wrist cameras it shears into
    # stretched slivers -- while a plain wall carrying the same picture is a stable, plausible background. From this
    # camera the result is unchanged.
    rest = ~ok & (ray[:, 0] > 1e-3)
    s_wall = (args.wall_x - origin[0]) / ray[rest, 0]
    world[rest] = origin + ray[rest] * s_wall[:, None]
    print(f"{ok.mean() * 100:.0f}% of vertices are table (re-projected onto z={args.table_z} m), "
          f"{rest.mean() * 100:.0f}% flattened onto the wall x={args.wall_x} m")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, points=world.astype(np.float32), faces=faces, colors=col.astype(np.uint8))
    print(f"{len(world)} vertices, {len(faces)} faces; extent {np.round(world.min(0), 2)} .. {np.round(world.max(0), 2)} -> {args.out}")


if __name__ == "__main__":
    main()
