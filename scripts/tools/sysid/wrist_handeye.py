# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Solve a wrist camera's mount on its link by hand-eye calibration over real episodes (AX = XB).

The wrist cameras barely see the robot, so the silhouette fit used for the body camera has nothing to hold
on to, and a full structure-from-motion of the room fails (blank walls, soft frames, a few distinct views).
Two frames a few tenths of a second apart in the same episode, though, match reliably: their relative
camera motion B comes out of the essential matrix (rotation exactly, translation up to scale), and the
arm's forward kinematics gives the link's motion A over the same interval. A fixed mount X must satisfy
A X = X B for every such pair. Rotation is solved first (robust least squares on the rotation residual),
then the translation together with each pair's unknown scale.

A pair is only used when the camera and the link turned by the same angle (rotation angle is invariant under
the unknown mount, so this gate does not depend on X): that throws out pairs dominated by something moving
with the camera -- a held can fills the view and looks static -- and pairs whose state and frame drifted
apart in time. The same angle test, scored over a range of shifts between the state and video streams,
also measures that offset before solving.

Plain python (numpy, scipy, opencv, pandas, av -- lerobot_openarm's venv has them):
  python scripts/tools/sysid/wrist_handeye.py ~/.cache/huggingface/lerobot/ethanCSL/openarm_pringles_lerobot_real_v00 \\
      --camera wrist_cam --out logs/real2sim/wrist_handeye
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_states import decode_episodes, load_episodes, real_state_to_sim  # noqa: E402
from urdf_fk import URDFChain  # noqa: E402

# Real colour intrinsics at 640x480 (`rs-enumerate-devices -c`; serial -> role from 99-realsense-rgb.rules).
INTRINSICS = {
    "body_cam": (607.24, 607.21, 326.43, 238.68),
    "wrist_cam": (605.44, 605.13, 322.85, 246.44),
    "right_wrist_cam": (607.79, 607.61, 322.58, 248.52),
}
LINK = {"wrist_cam": "openarm_left_ee_tcp", "right_wrist_cam": "openarm_right_ee_tcp"}
CFG_MOUNT = (0.06, 0.001, 0.11125, 0.68057, -0.19188, -0.19188, 0.68057)  # the cfg's, pos + wxyz (ROS)


def quat_wxyz_to_matrix(q) -> np.ndarray:
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def matrix_to_quat_wxyz(R) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if w >= 0 else -q


def angle(R) -> float:
    return float(np.degrees(Rotation.from_matrix(R).magnitude()))


def track(frames, mask=None):
    """KLT-track corners from frames[0] through every intermediate frame to frames[-1] -> (p0, p1).

    The wrist frames are too soft for descriptors (a few hundred SIFT keypoints at best), but tracking
    through consecutive frames only has to follow a few pixels at a time, which blurry corners survive.
    """
    p = cv2.goodFeaturesToTrack(frames[0], maxCorners=1500, qualityLevel=0.003, minDistance=7, mask=mask)
    if p is None:
        return None, None
    p0 = p.reshape(-1, 2)
    cur, alive = p0.copy(), np.ones(len(p0), bool)
    lk = dict(winSize=(25, 25), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    for a, b in zip(frames[:-1], frames[1:]):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, cur.reshape(-1, 1, 2), None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, None, **lk)  # forward-backward check
        alive &= (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back.reshape(-1, 2) - cur, axis=1) < 0.7)
        cur = nxt.reshape(-1, 2)
    return p0[alive], cur[alive]


def relative_camera_motion(frames, K, mask, min_inliers):
    """(R, t_unit, inliers) of the last frame's camera in the first one's frame, or None."""
    p1, p2 = track(frames, mask)
    if p1 is None or len(p1) < min_inliers:
        return None
    E, inl = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 0.75)
    if E is None or E.shape != (3, 3):
        return None
    n, R, t, inl = cv2.recoverPose(E, p1, p2, K, mask=inl)
    if n < min_inliers:
        return None
    # recoverPose: x2 = R x1 + t  ->  camera 2's pose in camera 1's frame is (R^T, -R^T t).
    return R.T, (-R.T @ t).ravel(), int(n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="Local LeRobot dataset root.")
    ap.add_argument("--camera", required=True, choices=sorted(LINK))
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibration", default=os.path.expanduser("~/Stanley_ws/lerobot_openarm/calibration.json"))
    ap.add_argument("--gaps", default="4,8,14", help="Frame gaps tried for pairs.")
    ap.add_argument("--min_turn", type=float, default=4.0, help="deg; smaller link turns carry little rotation info.")
    ap.add_argument("--max_turn", type=float, default=30.0)
    ap.add_argument("--min_inliers", type=int, default=60)
    ap.add_argument("--gate_deg", type=float, default=1.5, help="Max |camera turn - link turn| for a pair to count.")
    ap.add_argument("--max_shift", type=int, default=4, help="Frames of state/video offset to scan.")
    ap.add_argument("--retrack", action="store_true", help="Ignore the cached pairs.")
    args = ap.parse_args()

    calib = json.load(open(args.calibration))
    chain = URDFChain()
    K = np.array([[INTRINSICS[args.camera][0], 0, INTRINSICS[args.camera][2]],
                  [0, INTRINSICS[args.camera][1], INTRINSICS[args.camera][3]], [0, 0, 1]])
    episodes, names, fps = load_episodes(args.root, args.camera)
    gaps = [int(g) for g in args.gaps.split(",")]
    # Corners only away from the edges, where the fingertips swing in and out.
    mask = np.zeros((480, 640), np.uint8)
    mask[10:470, 50:590] = 255

    # For every pair: camera motion from the images, and the link motion at each candidate state shift.
    # Tracking is the slow part (~10 min a camera), so the pairs are cached next to the result.
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, f"{args.camera}_pairs.npz")
    shifts = list(range(-args.max_shift, args.max_shift + 1))
    if os.path.isfile(cache) and not args.retrack:
        c = np.load(cache)
        pairs = [(c["RB"][i], c["tB"][i], int(c["inliers"][i]), {s: c["A"][i, si] for si, s in enumerate(shifts) if c["A_ok"][i, si]})
                 for i in range(len(c["RB"]))]
        print(f"[handeye] {len(pairs)} cached pairs from {cache}")
    else:
        pairs = []  # (B_R, B_t_unit, inliers, {shift: A (4x4)})
        for e, frames in decode_episodes(episodes, fps):
            n = min(len(frames), len(e["state"]))
            links = [chain.pose(LINK[args.camera], real_state_to_sim(s, names, calib)) for s in e["state"][:n]]
            sharp = np.array([cv2.Laplacian(f, cv2.CV_32F).var() for f in frames[:n]])
            ok = sharp > 0.3 * np.median(sharp)
            for k in range(n):
                for g in gaps:
                    j = k + g
                    if j >= n or not (ok[k] and ok[j]):
                        continue
                    turn = angle(links[k][:3, :3].T @ links[j][:3, :3])
                    if not args.min_turn <= turn <= args.max_turn:
                        continue
                    rel = relative_camera_motion(frames[k:j + 1], K, mask, args.min_inliers)
                    if rel is None:
                        continue
                    A = {s: np.linalg.inv(links[k + s]) @ links[j + s] for s in shifts if 0 <= k + s and j + s < n}
                    pairs.append((rel[0], rel[1], rel[2], A))
            print(f"  episode {e['episode']:2d}: {n} frames, {len(pairs)} pairs so far", flush=True)
        A_all = np.zeros((len(pairs), len(shifts), 4, 4))
        A_ok = np.zeros((len(pairs), len(shifts)), bool)
        for i, p in enumerate(pairs):
            for si, s in enumerate(shifts):
                if s in p[3]:
                    A_all[i, si], A_ok[i, si] = p[3][s], True
        np.savez_compressed(cache, RB=np.array([p[0] for p in pairs]), tB=np.array([p[1] for p in pairs]),
                            inliers=np.array([p[2] for p in pairs]), A=A_all, A_ok=A_ok)

    # State/video offset: the shift under which camera and link turn by the same angle most often.
    best_shift, best_score = 0, -1
    for s in range(-args.max_shift, args.max_shift + 1):
        err = np.array([abs(angle(p[0]) - angle(p[3][s][:3, :3])) for p in pairs if s in p[3]])
        score = int((err < args.gate_deg).sum())
        print(f"  shift {s:+d} frames ({s / fps * 1000:+.0f} ms): {score} pairs agree within {args.gate_deg} deg, median diff {np.median(err):.2f} deg")
        if score > best_score:
            best_shift, best_score = s, score
    used = [(p[0], p[1], p[2], p[3][best_shift]) for p in pairs
            if best_shift in p[3] and abs(angle(p[0]) - angle(p[3][best_shift][:3, :3])) < args.gate_deg]
    print(f"[handeye] {len(used)} / {len(pairs)} pairs used, state lags video by {best_shift} frames")
    if len(used) < 20:
        raise SystemExit("Too few consistent pairs to solve the mount.")

    RA = np.array([u[3][:3, :3] for u in used])
    tA = np.array([u[3][:3, 3] for u in used])
    RB = np.array([u[0] for u in used])
    tB = np.array([u[1] for u in used])

    # Rotation: start from the cfg's mount, refine  R_A R_X = R_X R_B  robustly.
    R0 = quat_wxyz_to_matrix(CFG_MOUNT[3:])

    def rot_res(x):
        RX = Rotation.from_rotvec(x).as_matrix() @ R0
        return Rotation.from_matrix(np.einsum("nij,jk,nlk,ml->nim", RA, RX, RB, RX)).as_rotvec().ravel()

    starts = [np.zeros(3)] + [np.radians(r) for r in ((90, 0, 0), (0, 90, 0), (0, 0, 90), (180, 0, 0), (0, 180, 0), (0, 0, 180))]
    sols = [least_squares(rot_res, x, loss="soft_l1", f_scale=np.radians(1.0)) for x in starts]
    sol = min(sols, key=lambda s: s.cost)
    RX = Rotation.from_rotvec(sol.x).as_matrix() @ R0
    r_err = np.degrees(np.linalg.norm(rot_res(sol.x).reshape(-1, 3), axis=1))

    # Translation: (R_A - I) t_X + t_A = lambda_i R_X t_B,i, with an unknown scale lambda_i per pair. Projecting
    # each equation onto the plane normal to v_i = R_X t_B,i removes lambda_i, leaving a 3x3 linear system in
    # t_X, solved with Huber reweighting (IRLS).
    v = tB @ RX.T
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    P = np.eye(3) - v[:, :, None] * v[:, None, :]
    M = np.einsum("nij,njk->nik", P, RA - np.eye(3))
    c = -np.einsum("nij,nj->ni", P, tA)

    def solve_t(idx, iters=10, delta=0.003):
        w = np.ones(len(idx))
        for _ in range(iters):
            Mi, ci = M[idx] * np.sqrt(w)[:, None, None], c[idx] * np.sqrt(w)[:, None]
            t = np.linalg.lstsq(Mi.reshape(-1, 3), ci.ravel(), rcond=None)[0]
            r = np.linalg.norm(np.einsum("nij,j->ni", M[idx], t) - c[idx], axis=1)
            w = np.where(r < delta, 1.0, delta / np.maximum(r, 1e-9))
        return t

    all_idx = np.arange(len(used))
    tX = solve_t(all_idx)
    t_err = np.linalg.norm(np.einsum("nij,j->ni", M, tX) - c, axis=1) * 1000
    lam = np.einsum("ni,ni->n", np.einsum("nij,j->ni", RA - np.eye(3), tX) + tA, v)
    rng = np.random.default_rng(0)
    t_std = np.std([solve_t(rng.choice(len(used), len(used))) for _ in range(50)], axis=0) * 100  # bootstrap
    print(f"pairs whose implied camera baseline points backwards: {(lam < 0).mean() * 100:.0f}% (should be ~0)")

    q = matrix_to_quat_wxyz(RX)
    print(f"\nrotation residual: median {np.median(r_err):.2f} deg, 90th pct {np.percentile(r_err, 90):.2f} deg")
    print(f"translation residual (projected): median {np.median(t_err):.1f} mm; bootstrap std {np.round(t_std, 2)} cm")
    print(f"vs cfg mount: moved {np.linalg.norm(tX - CFG_MOUNT[:3]) * 100:.1f} cm, turned {angle(R0.T @ RX):.1f} deg")
    print(f"offset=CameraCfg.OffsetCfg(pos=({tX[0]:.5f}, {tX[1]:.5f}, {tX[2]:.5f}), "
          f"rot=({q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}, {q[3]:.5f}), convention=\"ros\")")
    os.makedirs(args.out, exist_ok=True)
    json.dump({"camera": args.camera, "parent_link": LINK[args.camera], "pos": tX.tolist(), "rot_wxyz_ros": q.tolist(),
               "intrinsics": INTRINSICS[args.camera], "state_shift_frames": best_shift, "pairs": len(used),
               "rot_residual_median_deg": float(np.median(r_err)), "pos_bootstrap_std_cm": t_std.tolist()},
              open(os.path.join(args.out, f"{args.camera}_mount.json"), "w"), indent=2)
    print(f"-> {args.out}/{args.camera}_mount.json")


if __name__ == "__main__":
    main()
