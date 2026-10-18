# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Reconstruct the real room from dataset frames and solve the wrist cameras' mounts by hand-eye calibration.

The wrist cameras barely see the robot, so the silhouette fit used for the body camera has nothing to hold
on to. What they do see is the room, which does not move: structure-from-motion (COLMAP) over frames from
all three cameras recovers every camera's pose in one arbitrary, unscaled frame, and the arm's forward
kinematics (render_real_states.py's poses.npz) says where each wrist link was. A wrist camera's mount X is
the fixed transform that makes  link_k * X = S * camera_k  hold for every frame k at once, with S the
similarity from the SfM frame into the robot frame; the body camera (mount already calibrated) pins S down
together with the wrists. Moving things (robot, can, plate) are masked out of the features, and so are the
two fingertips that sit in the corners of each wrist view.

Needs pycolmap (a scratch venv with ``pycolmap opencv-python-headless scipy`` is enough):
  python scripts/tools/sysid/sfm_handeye.py logs/real2sim/sfm --body_mount logs/real2sim/body_cam_fit_v2/camera_fit.json

<root> holds one folder per camera, each with frames.npz (extract_real_frames.py, the same frame indices for
every camera) and dyn_masks.npz (moving objects, True = masked), plus <root>/poses.npz: render_real_states.py
run on any one camera's frames.npz, for the link poses at those states. Writes
<root>/colmap (database, sparse model), <root>/handeye.json (the wrist mounts as CameraCfg.OffsetCfg values)
and <root>/points_robot.npz (the room's sparse points and colours in the robot's base frame).
"""

import argparse
import json
import os

import cv2
import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# Real colour intrinsics at 640x480 (`rs-enumerate-devices -c`; serial -> role from 99-realsense-rgb.rules).
INTRINSICS = {
    "body_cam": (607.24, 607.21, 326.43, 238.68),
    "wrist_cam": (605.44, 605.13, 322.85, 246.44),
    "right_wrist_cam": (607.79, 607.61, 322.58, 248.52),
}
WRIST_CAMS = ("wrist_cam", "right_wrist_cam")


def quat_wxyz_to_matrix(q) -> np.ndarray:
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def matrix_to_quat_wxyz(R) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if w >= 0 else -q


def se3(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def fingertip_mask(h: int, w: int, border: int = 45) -> np.ndarray:
    """The foam fingertips swing in and out along the left/right edges of a wrist view as the hand opens and
    closes; they move with the camera, so they must not be matched. A plain border strip covers them."""
    m = np.zeros((h, w), bool)
    m[:, :border] = m[:, w - border:] = True
    return m


def write_images(root: str, cams: list[str], body_every: int) -> tuple[str, str, dict]:
    img_dir, mask_dir = os.path.join(root, "colmap", "images"), os.path.join(root, "colmap", "masks")
    index = {}  # image name -> (camera, frame index)
    for cam in cams:
        os.makedirs(os.path.join(img_dir, cam), exist_ok=True)
        os.makedirs(os.path.join(mask_dir, cam), exist_ok=True)
        images = np.load(os.path.join(root, cam, "frames.npz"))["images"]
        dyn = np.load(os.path.join(root, cam, "dyn_masks.npz"))["masks"]
        for k, (im, m) in enumerate(zip(images, dyn)):
            if cam == "body_cam" and k % body_every:
                continue  # the body camera never moves: a few copies anchor it, more only add identical views
            if cam in WRIST_CAMS:
                m = m | fingertip_mask(*m.shape)
            name = f"{cam}/{k:04d}.png"
            cv2.imwrite(os.path.join(img_dir, name), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(mask_dir, name + ".png"), (~m).astype(np.uint8) * 255)  # COLMAP: 0 = ignore
            index[name] = (cam, k)
    return img_dir, mask_dir, index


def approx_poses(root: str, index: dict, mounts: dict) -> dict:
    """Camera-to-robot poses from forward kinematics and the current (approximate) mounts, for pair selection."""
    out = {}
    for name, (cam, k) in index.items():
        poses = np.load(os.path.join(root, "poses.npz"))
        body = [str(b) for b in poses["body_names"]]
        li = body.index(mounts[cam]["parent"])
        L = se3(quat_wxyz_to_matrix(poses["body_quat_w"][k][li]), poses["body_pos_w"][k][li])
        out[name] = L @ mounts[cam]["X"]
    return out


def select_pairs(poses: dict, n_neighbours: int) -> list[tuple[str, str]]:
    names = list(poses)
    C = np.array([poses[n][:3, 3] for n in names])
    Z = np.array([poses[n][:3, 2] for n in names])  # optical axes
    cos = Z @ Z.T
    dist = np.linalg.norm(C[:, None] - C[None], axis=-1)
    score = dist + 0.5 * (1 - cos)  # ~1 m of separation weighs like looking 90 deg apart
    score[cos < 0.3] = np.inf  # views more than ~70 deg apart share little
    pairs = set()
    for i in range(len(names)):
        for j in np.argsort(score[i])[1:n_neighbours + 1]:
            if np.isfinite(score[i, j]):
                pairs.add(tuple(sorted((names[i], names[j]))))
    return sorted(pairs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--body_mount", required=True, help="camera_fit.json of the body camera (calibrate_camera.py).")
    ap.add_argument("--wrist_mount", default="0.06,0.001,0.11125,0.68057,-0.19188,-0.19188,0.68057",
                    help="Starting wrist mount in ee_tcp, x,y,z,qw,qx,qy,qz (ROS), i.e. the cfg's.")
    ap.add_argument("--neighbours", type=int, default=30)
    ap.add_argument("--body_every", type=int, default=30)
    ap.add_argument("--features", type=int, default=8000)
    ap.add_argument("--peak_threshold", type=float, default=0.0015)
    ap.add_argument("--reuse", action="store_true", help="Reuse the existing database and sparse model.")
    args = ap.parse_args()

    body = json.load(open(args.body_mount))
    w = [float(v) for v in args.wrist_mount.split(",")]
    mounts = {"body_cam": {"parent": body["parent_link"], "X": se3(quat_wxyz_to_matrix(body["rot_wxyz_ros"]), body["pos"])}}
    for cam, side in (("wrist_cam", "left"), ("right_wrist_cam", "right")):
        mounts[cam] = {"parent": f"openarm_{side}_ee_tcp", "X": se3(quat_wxyz_to_matrix(w[3:]), w[:3])}

    cams = ["wrist_cam", "right_wrist_cam", "body_cam"]
    work = os.path.join(args.root, "colmap")
    db, sparse = os.path.join(work, "database.db"), os.path.join(work, "sparse")
    img_dir, mask_dir, index = write_images(args.root, cams, args.body_every)
    guess = approx_poses(args.root, index, mounts)

    if not args.reuse:
        if os.path.exists(db):
            os.remove(db)
        for cam in cams:  # one fixed pinhole per physical camera; the real lens is distortion-free (coeffs 0)
            reader = pycolmap.ImageReaderOptions(camera_model="PINHOLE", camera_params=",".join(map(str, INTRINSICS[cam])),
                                                 mask_path=mask_dir)
            ext = pycolmap.FeatureExtractionOptions()
            ext.sift.max_num_features = args.features
            # The room is mostly blank wall and the frames are soft: the default contrast threshold keeps ~150
            # features per image, too few to register most views. Lower it, and pool over scales for blur.
            ext.sift.peak_threshold = args.peak_threshold
            ext.sift.domain_size_pooling = True
            ext.sift.estimate_affine_shape = True
            names = sorted(n for n in index if n.startswith(cam + "/"))
            pycolmap.extract_features(db, img_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                                      reader_options=reader, extraction_options=ext)
        pairs = select_pairs(guess, args.neighbours)
        pair_file = os.path.join(work, "pairs.txt")
        open(pair_file, "w").write("\n".join(f"{a} {b}" for a, b in pairs))
        print(f"[sfm] {len(index)} images, {len(pairs)} pairs")
        pycolmap.match_image_pairs(db, pairing_options=pycolmap.ImportedPairingOptions(match_list_path=pair_file))
        opts = pycolmap.IncrementalPipelineOptions()
        opts.ba_refine_focal_length = opts.ba_refine_principal_point = opts.ba_refine_extra_params = False
        os.makedirs(sparse, exist_ok=True)
        recs = pycolmap.incremental_mapping(db, img_dir, sparse, options=opts)
        best = max(recs.values(), key=lambda r: r.num_reg_images())
        best.write(sparse)
    rec = pycolmap.Reconstruction(sparse)
    print(f"[sfm] model: {rec.num_reg_images()} registered images, {rec.num_points3D()} points")

    # Camera-to-SfM poses of the registered images.
    sfm = {}
    for img in rec.images.values():
        if img.name in index:
            T = img.cam_from_world().inverse() if callable(img.cam_from_world) else img.cam_from_world.inverse()
            sfm[img.name] = T.matrix() if T.matrix().shape == (4, 4) else np.vstack([T.matrix(), [0, 0, 0, 1]])
    for cam in cams:
        print(f"  {cam}: {sum(index[n][0] == cam for n in sfm)} / {sum(v[0] == cam for v in index.values())} registered")

    # Link poses of the registered images.
    links = {}
    for name in sfm:
        cam, k = index[name]
        poses = np.load(os.path.join(args.root, "poses.npz"))
        li = [str(b) for b in poses["body_names"]].index(mounts[cam]["parent"])
        links[name] = se3(quat_wxyz_to_matrix(poses["body_quat_w"][k][li]), poses["body_pos_w"][k][li])

    # Initial similarity: align SfM camera centres to where the current mounts put them (Umeyama).
    names = list(sfm)
    A = np.array([sfm[n][:3, 3] for n in names])
    B = np.array([(links[n] @ mounts[index[n][0]]["X"])[:3, 3] for n in names])
    mu_a, mu_b = A.mean(0), B.mean(0)
    U, D, Vt = np.linalg.svd((B - mu_b).T @ (A - mu_a) / len(A))
    Sg = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R_s = U @ Sg @ Vt
    s = np.trace(np.diag(D) @ Sg) / ((A - mu_a) ** 2).sum(1).mean()
    t_s = mu_b - s * R_s @ mu_a

    def unpack(x):
        S = (Rotation.from_rotvec(x[0:3]).as_matrix() @ R_s, t_s + x[3:6], s * np.exp(x[6]))
        X = {c: mounts[c]["X"] @ se3(Rotation.from_rotvec(x[7 + 6 * i:10 + 6 * i]).as_matrix(), x[10 + 6 * i:13 + 6 * i])
             for i, c in enumerate(WRIST_CAMS)}
        X["body_cam"] = mounts["body_cam"]["X"]
        return S, X

    def residuals(x, w_rot=0.05):
        (R, t, sc), X = unpack(x)
        res = []
        for n in names:
            cam = index[n][0]
            pred = links[n] @ X[cam]  # camera in robot frame from kinematics
            meas_R, meas_t = R @ sfm[n][:3, :3], sc * R @ sfm[n][:3, 3] + t  # same camera from SfM
            res.append(pred[:3, 3] - meas_t)
            res.append(w_rot * Rotation.from_matrix(pred[:3, :3].T @ meas_R).as_rotvec())  # 0.05 m per rad
        return np.concatenate(res)

    x0 = np.zeros(7 + 6 * len(WRIST_CAMS))
    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=0.02)
    (R, t, sc), X = unpack(sol.x)
    r = residuals(sol.x).reshape(-1, 2, 3)
    pos_err, rot_err = np.linalg.norm(r[:, 0], axis=1), np.degrees(np.linalg.norm(r[:, 1], axis=1) / 0.05)
    print(f"\n[handeye] SfM scale {sc:.4f} m/unit; residual per camera (median / 90th pct):")
    out = {"scale": sc, "R_sfm_to_robot": R.tolist(), "t_sfm_to_robot": t.tolist()}
    for cam in cams:
        sel = np.array([index[n][0] == cam for n in names])
        if sel.any():
            print(f"  {cam:16s} pos {np.median(pos_err[sel]) * 100:.1f} / {np.percentile(pos_err[sel], 90) * 100:.1f} cm   "
                  f"rot {np.median(rot_err[sel]):.2f} / {np.percentile(rot_err[sel], 90):.2f} deg")
    for cam in WRIST_CAMS:
        T, T0 = X[cam], mounts[cam]["X"]
        q = matrix_to_quat_wxyz(T[:3, :3])
        moved = np.linalg.norm(T[:3, 3] - T0[:3, 3]) * 100
        turned = np.degrees(Rotation.from_matrix(T0[:3, :3].T @ T[:3, :3]).magnitude())
        print(f"\n{cam} (in {mounts[cam]['parent']}): moved {moved:.1f} cm, turned {turned:.1f} deg from the start")
        print(f"  offset=CameraCfg.OffsetCfg(pos=({T[0, 3]:.5f}, {T[1, 3]:.5f}, {T[2, 3]:.5f}), "
              f"rot=({q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}, {q[3]:.5f}), convention=\"ros\")")
        out[cam] = {"parent_link": mounts[cam]["parent"], "pos": T[:3, 3].tolist(), "rot_wxyz_ros": q.tolist(),
                    "intrinsics": INTRINSICS[cam]}
    json.dump(out, open(os.path.join(args.root, "handeye.json"), "w"), indent=2)

    xyz = np.array([p.xyz for p in rec.points3D.values()])
    rgb = np.array([p.color for p in rec.points3D.values()])
    err = np.array([p.error for p in rec.points3D.values()])
    track = np.array([p.track.length() for p in rec.points3D.values()])
    np.savez_compressed(os.path.join(args.root, "points_robot.npz"), xyz=(sc * xyz @ R.T + t).astype(np.float32),
                        rgb=rgb.astype(np.uint8), error=err.astype(np.float32), track=track.astype(np.int32))
    print(f"-> {args.root}/handeye.json, points_robot.npz ({len(xyz)} points)")


if __name__ == "__main__":
    main()
