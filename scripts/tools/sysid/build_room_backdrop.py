# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Build the room around the rig as three photo-textured walls, fused from many real frames seen from known poses.

build_wrist_backdrop.py hangs each wrist camera's rest-pose view on one plane in front of it: exact at the rest pose,
but as soon as the arm moves the picture slides, tilts and leaves sky around it. Here the room is a box instead -- a
back wall beyond the table, a wall either side and the floor -- and every texel takes the MEDIAN colour of the real pixels it
projects to, over thousands of real frames from the three cameras (wrist cameras at every recorded arm pose, posed
with forward kinematics at the frame's joint state and the env cfg's mount, as the sim will pose them). Each wall's
distance (and the floor's depth) is chosen by a plane sweep: the depth at which the real frames from different poses agree best.

Pixels that cannot be the room are left out: the wrist cameras' foam fingertips (fixed corners), the can (its red /
orange / yellow), anything the ray meets on the table first. People, the other arm and the operator's cables move
between frames and drop out in the median.

  python scripts/tools/sysid/build_room_backdrop.py ~/.cache/huggingface/lerobot/ethanCSL/openarm_pringles_lerobot_real_v00 \\
      --out source/isaaclab_assets/data/real2sim/pringles_v00/backdrop_room.npz

Writes the npz (points, faces, colors: same format as the other backdrops) and <out>_walls.png (the textures).

--body_backdrop trims build_backdrop.py's body-camera backdrop to the table itself (its faces on the table plane) and
writes it to --body_out: beyond the table it holds what body_cam saw flattened onto stand-in planes, right for body_cam
and wrong from the wrist cameras -- the room replaces it.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_states import decode_episodes, load_episodes, real_state_to_sim  # noqa: E402
from urdf_fk import URDFChain  # noqa: E402

# The env cfg's cameras (stack_ik_abs_visuomotor_env_cfg.py): parent link, mount (pos, ROS wxyz), fx, fy at 640x480
# with the principal point centred as the renderer does.
CAMERAS = {
    "wrist_cam": ("openarm_left_ee_tcp", (0.05342, 0.001, 0.12199), (0.68057, -0.19188, -0.19188, 0.68057), 605.44, 605.13),
    "right_wrist_cam": ("openarm_right_ee_tcp", (0.05342, 0.001, 0.12199), (0.68057, -0.19188, -0.19188, 0.68057), 607.79, 607.61),
    "body_cam": ("openarm_body_link0", (0.05621, -0.00305, 0.83062), (0.18720, -0.68244, 0.69543, -0.12490), 607.22, 607.22),
}
TABLE = (0.03, 0.56, -0.30, 0.48, 0.28)  # x0, x1, y0, y1, top z: the rig's table with its far rail (backdrop_body's table part)
Z_RANGE = (-0.3, 2.2)  # m: wall height range while the walls are swept (then from the fitted floor up)
SCALE = 0.5  # frames are used at half resolution


def mount_matrix(pos, quat_wxyz) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
    M[:3, 3] = pos
    return M


def room_mask(img: np.ndarray, camera: str) -> np.ndarray:
    """True where the pixel may be the room: not the fingertips' corners, not the can's colours."""
    h, w = img.shape[:2]
    ok = np.ones((h, w), bool)
    if camera != "body_cam":
        ys, xs = np.mgrid[0:h, 0:w]
        ok &= ~(((xs < 0.14 * w) | (xs > 0.86 * w)) & (ys > 0.3 * h))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    can = ((hsv[..., 0] < 35) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 60)
    can = cv2.dilate(can.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return ok & ~can


def load_views(root: str, every: int, calib: dict) -> list[dict]:
    chain = URDFChain()
    views = []
    for camera, (link, pos, quat, fx, fy) in CAMERAS.items():
        episodes, names, fps = load_episodes(root, camera)
        step = every * 6 if camera == "body_cam" else every  # body_cam never moves: a few frames per episode will do
        mount = mount_matrix(pos, quat)
        for e, frames in decode_episodes(episodes, fps, gray=False, every=step):
            for k, img in frames:
                q = real_state_to_sim(e["state"][min(k, len(e["state"]) - 1)], names, calib)
                pose = chain.pose(link, q) @ mount if link != "openarm_body_link0" else mount
                small = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
                views.append(dict(camera=camera, img=small, ok=room_mask(small, camera), pose=pose,
                                  K=(fx * SCALE, fy * SCALE, small.shape[1] / 2, small.shape[0] / 2)))
        print(f"[build_room_backdrop] {camera}: {sum(v['camera'] == camera for v in views)} frames", flush=True)
    return views


def wall_grid(wall: str, d: float, box: dict, res: float):
    """Texel centres (H, W, 3) of a wall: 'back' is x = d, 'left' y = d, 'right' y = -d, 'floor' z = -d; rows run top to
    bottom (floor: far to near)."""
    m = 0.05  # m each surface runs past the box's edges, so neighbouring surfaces overlap: no sky at the seams
    if wall == "floor":
        X, Y = np.meshgrid(np.arange(box["left"] + m, -box["right"] - m, -res), np.arange(box["back"] + m, -0.3, -res))
        return np.stack([Y, X, np.full_like(X, -d)], -1)
    zs = np.arange(Z_RANGE[1], -box["floor"] - m, -res)
    if wall == "back":
        us = np.arange(box["left"] + m, -box["right"] - m, -res)  # left (+y) to right, as seen from the robot
        Y, Z = np.meshgrid(us, zs)
        return np.stack([np.full_like(Y, d), Y, Z], -1)
    xs = np.arange(0.0, box["back"] + m, res) if wall == "left" else np.arange(box["back"] + m, 0.0, -res)
    X, Z = np.meshgrid(xs, zs)
    return np.stack([X, np.full_like(X, d if wall == "left" else -d), Z], -1)


def table_blocks(cam: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """True where the segment from the camera to the point crosses the table top inside its footprint (plus a 2 cm margin
    all round: rays that graze the table's white far edge must not paint it onto the wall behind)."""
    x0, x1, y0, y1, top = TABLE
    x1, y0, y1, top = x1 + 0.02, y0 - 0.02, y1 + 0.02, top + 0.02
    dz = pts[:, 2] - cam[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (top - cam[2]) / dz
    hit = cam[None] + t[:, None] * (pts - cam[None])
    return (t > 0) & (t < 1) & (hit[:, 0] > x0) & (hit[:, 0] < x1) & (hit[:, 1] > y0) & (hit[:, 1] < y1)


def sample(views: list[dict], pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Colours (V, N, 3) uint8 and validity (V, N) of world points pts (N, 3) in every view."""
    cols = np.zeros((len(views), len(pts), 3), np.uint8)
    valid = np.zeros((len(views), len(pts)), bool)
    for i, v in enumerate(views):
        R, t = v["pose"][:3, :3], v["pose"][:3, 3]
        c = (pts - t) @ R
        fx, fy, cx, cy = v["K"]
        z = c[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(fx * c[:, 0] / z + cx).astype(np.int64)
            w = np.round(fy * c[:, 1] / z + cy).astype(np.int64)
        h, wd = v["img"].shape[:2]
        inside = (z > 0.05) & (u >= 0) & (u < wd) & (w >= 0) & (w < h)
        idx = np.flatnonzero(inside)
        idx = idx[v["ok"][w[idx], u[idx]]]
        idx = idx[~table_blocks(t, pts[idx])]
        cols[i, idx] = v["img"][w[idx], u[idx]]
        valid[i, idx] = True
    return cols, valid


def fuse(views, pts, min_views: int, chunk: int = 4000):
    """Per-point median colour, median absolute deviation and view count."""
    med = np.zeros((len(pts), 3))
    mad = np.full(len(pts), np.nan)
    count = np.zeros(len(pts), int)
    for s in range(0, len(pts), chunk):
        cols, valid = sample(views, pts[s:s + chunk])
        f = np.where(valid[..., None], cols.astype(np.float32), np.nan)
        m = np.nanmedian(f, 0)
        count[s:s + chunk] = valid.sum(0)
        good = count[s:s + chunk] >= min_views
        med[s:s + chunk][good] = m[good]
        mad[s:s + chunk][good] = np.nanmedian(np.abs(f - m[None]), 0).mean(-1)[good]
    return med, mad, count


def sweep(views, wall: str, box: dict, candidates, res: float, min_views: int) -> float:
    """The wall distance at which the views agree best (lowest mean colour deviation over textured, well-seen texels)."""
    scores, seen = [], []
    for d in candidates:
        g = wall_grid(wall, d, box, res).reshape(-1, 3)
        med, mad, count = fuse(views, g, min_views)
        good = count >= min_views
        scores.append(np.nanmean(mad[good]) if good.sum() > 50 else np.inf)
        seen.append(good.sum())
        print(f"[build_room_backdrop]   {wall} wall at {d:.2f} m: deviation {scores[-1]:.2f} over {good.sum()} texels", flush=True)
    # a depth where almost nothing lands on the plane agrees trivially: only compare depths that see a fair share of it
    scores = np.where(np.array(seen) >= 0.25 * max(seen), scores, np.inf)
    return float(candidates[int(np.argmin(scores))])


def grid_mesh(grid: np.ndarray, colors: np.ndarray, offset: int):
    h, w = grid.shape[:2]
    i = np.arange(h * w).reshape(h, w) + offset
    a, b, c, d = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, :-1].ravel(), i[1:, 1:].ravel()
    faces = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
    return grid.reshape(-1, 3), faces, colors.reshape(-1, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=10, help="Use every n-th frame of each episode (wrist cameras).")
    ap.add_argument("--res", type=float, default=0.01, help="m per texel of the final walls.")
    ap.add_argument("--min_views", type=int, default=4)
    ap.add_argument("--calibration", default=os.path.expanduser("~/Stanley_ws/lerobot_openarm/calibration.json"))
    ap.add_argument("--box", default=None, help="back,left,right,floor distances in m: skip the plane sweeps.")
    ap.add_argument("--floor", type=float, default=None, help="m below the robot base; default: plane sweep. The cameras barely"
                    " see the floor, so the sweep is weak -- set it below everything the walls show.")
    ap.add_argument("--body_backdrop", default=None, help="build_backdrop.py's backdrop_body.npz, to trim to the table.")
    ap.add_argument("--body_out", default=None)
    args = ap.parse_args()

    if args.body_backdrop:
        d = np.load(args.body_backdrop)
        on_table = d["points"][:, 2] > TABLE[4] - 0.005
        faces = d["faces"][on_table[d["faces"]].all(1)]
        used = np.unique(faces)
        remap = np.full(len(on_table), -1)
        remap[used] = np.arange(len(used))
        np.savez_compressed(args.body_out, points=d["points"][used], faces=remap[faces].astype(np.int32), colors=d["colors"][used])
        print(f"[build_room_backdrop] body backdrop trimmed to the table: {len(faces)} of {len(d['faces'])} faces -> {args.body_out}")

    calib = json.load(open(args.calibration))
    views = load_views(args.root, args.every, calib)
    rng = np.random.default_rng(0)
    coarse = [views[i] for i in rng.choice(len(views), min(400, len(views)), replace=False)]

    box = {"back": 3.0, "left": 2.0, "right": 2.0, "floor": -Z_RANGE[0]}
    if args.box:
        box = dict(zip(("back", "left", "right", "floor"), (float(v) for v in args.box.split(","))))
    else:
        box["back"] = sweep(coarse, "back", box, np.arange(0.8, 2.61, 0.1), 0.03, args.min_views)
        box["left"] = sweep(coarse, "left", box, np.arange(0.5, 2.01, 0.1), 0.03, args.min_views)
        box["right"] = sweep(coarse, "right", box, np.arange(0.5, 2.01, 0.1), 0.03, args.min_views)
        box["floor"] = args.floor if args.floor else sweep(coarse, "floor", box, np.arange(0.2, 1.21, 0.1), 0.03, args.min_views)
    print(f"[build_room_backdrop] room: back wall x = {box['back']:.2f}, left wall y = {box['left']:.2f}, "
          f"right wall y = {-box['right']:.2f}, floor z = {-box['floor']:.2f}", flush=True)

    pts, faces, cols, sheets = [], [], [], []
    n = 0
    foot = []  # what the cameras saw at the foot of the walls: the colour for a surface they never saw at all
    for wall in ("back", "left", "right", "floor"):
        g = wall_grid(wall, box[wall], box, args.res)
        med, _, count = fuse(views, g.reshape(-1, 3), args.min_views)
        tex = med.reshape(*g.shape[:2], 3).astype(np.uint8)
        hole = (count < args.min_views).reshape(g.shape[:2]).astype(np.uint8)
        print(f"[build_room_backdrop] {wall} wall: {g.shape[1]} x {g.shape[0]} texels, {100 * (1 - hole.mean()):.0f}% seen", flush=True)
        if wall != "floor":
            seen_rows = np.flatnonzero((hole == 0).any(1))
            if len(seen_rows):
                low = tex[seen_rows[-10:]][hole[seen_rows[-10:]] == 0]
                foot.append(low)
        if hole.mean() > 0.98:  # (next to) nothing seen: one plain colour, not an inpainting of nothing
            tex[:] = np.median(np.concatenate(foot), 0).astype(np.uint8) if foot else 128
        elif hole.any():  # never seen by any camera: continue the nearby colours rather than leave sky
            tex = cv2.inpaint(tex, hole, 15, cv2.INPAINT_TELEA)
        p, f, c = grid_mesh(g, tex, n)
        pts.append(p), faces.append(f), cols.append(c)
        n += len(p)
        sheets.append(cv2.resize(tex, (tex.shape[1] * 2, tex.shape[0] * 2), interpolation=cv2.INTER_NEAREST))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, points=np.concatenate(pts).astype(np.float32), faces=np.concatenate(faces).astype(np.int32),
                        colors=np.concatenate(cols).astype(np.uint8))
    hmax = max(s.shape[0] for s in sheets)
    sheet = np.concatenate([np.pad(s, ((0, hmax - s.shape[0]), (0, 8), (0, 0))) for s in sheets], 1)
    cv2.imwrite(args.out.replace(".npz", "_walls.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"[build_room_backdrop] {n} vertices -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
