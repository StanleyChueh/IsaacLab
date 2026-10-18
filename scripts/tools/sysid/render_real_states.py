# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Render one sim camera with the robot posed at each real frame's joint state, next to the real image.

Input is extract_real_frames.py's frames.npz. For every frame the arm joints are written straight into
the sim (calibration.json maps real motor radians to sim joints) and held there, the manipulated object is
parked out of view (its real position is unknown), and the camera is rendered. Output, in --out:

  sim_XX.png / overlay_XX.png   the sim render, and a 50/50 blend with the real frame
  poses.npz                     per frame: the depth image, every robot body's world pose, the camera's
                                world pose (ROS convention) and its parent link's world pose, plus the
                                intrinsic matrix; and points sampled over every link's visual meshes, in
                                that link's frame -- everything calibrate_camera.py needs without Isaac.

--intrinsics / --cam_pos / --cam_rot override the camera cfg, so the same script both shows the current
mismatch and verifies a calibrated result:

  ./isaaclab.sh -p scripts/tools/sysid/render_real_states.py --enable_cameras --headless \\
      --frames logs/real2sim/body_cam_frames/frames.npz --out logs/real2sim/body_cam_render
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0")
parser.add_argument("--task_mode", default="handover")
parser.add_argument("--frames", required=True, help="frames.npz from extract_real_frames.py")
parser.add_argument("--out", required=True)
parser.add_argument("--camera", default="body_cam")
parser.add_argument(
    "--calibration", default=os.path.expanduser("~/Stanley_ws/lerobot_openarm/calibration.json"),
    help="lerobot_openarm calibration.json (real motor <-> sim joint mapping).",
)
parser.add_argument("--intrinsics", default=None, help="fx,fy,cx,cy in pixels, replacing the cfg's lens.")
parser.add_argument("--cam_pos", default=None, help="x,y,z of the camera in its parent link's frame.")
parser.add_argument("--cam_rot", default=None, help="w,x,y,z of the camera in its parent link's frame, ROS convention.")
parser.add_argument("--look_dir", default=None, help="Apply openarm_real_look from this folder ('default' = LOOK_DIR).")
parser.add_argument("--settle_steps", type=int, default=3)
parser.add_argument("--surface_points", type=int, default=200000, help="Points sampled over the robot's visual meshes.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_joint_actions import (  # noqa: E402
    swap_to_openarm_joint_actions,
)
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_task_modes import apply_task_mode  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from real_states import real_state_to_sim  # noqa: E402


def _floats(text: str, n: int) -> tuple[float, ...]:
    values = tuple(float(v) for v in text.split(","))
    if len(values) != n:
        raise SystemExit(f"Expected {n} comma-separated numbers, got {text!r}")
    return values


def sample_robot_surface(robot_path: str, body_names: list[str], n_total: int, seed: int = 0):
    """Area-weighted points on each body's visual meshes, in that body's frame.

    Returns (link index (M,), xyz (M,3), part index (M,), part names): a part is one (link, material) pair,
    e.g. "openarm_left_link7/mat_3_001", resolved per face through the meshes' GeomSubsets, so the colour of
    each part can be compared against the real robot's (fit_robot_colors.py).

    Mesh-to-link transforms come from the authored USD, which is consistent within one stage even though
    physics never writes poses back to it; per-frame link poses then come from the articulation's data.
    """
    import omni.usd
    from pxr import Usd, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    xc = UsdGeom.XformCache()
    tris_link, tris_xyz, tris_part, parts, skipped = [], [], [], [], []

    def part_id(body: str, material) -> int:
        name = f"{body}/{material.GetPath().name if material else 'none'}"
        if name not in parts:
            parts.append(name)
        return parts.index(name)
    for li, body in enumerate(body_names):
        link = stage.GetPrimAtPath(f"{robot_path}/{body}")
        if not link.IsValid():
            skipped.append(body)
            continue
        to_link = np.array(xc.GetLocalToWorldTransform(link).GetInverse())
        it = iter(Usd.PrimRange(link, Usd.TraverseInstanceProxies()))
        for prim in it:
            if prim != link and prim.GetName() in body_names:
                it.PruneChildren()  # a nested link: sampled under its own name
                continue
            if "collision" in str(prim.GetPath()).lower():
                it.PruneChildren()
                continue
            if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide:
                continue
            mesh = UsdGeom.Mesh(prim)
            pts, counts, idx = mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
            if not pts or not counts:
                continue
            pts, idx = np.array(pts, dtype=np.float64), np.array(idx)
            rel = np.array(xc.GetLocalToWorldTransform(prim)) @ to_link  # USD row-vector convention
            pts = np.c_[pts, np.ones(len(pts))] @ rel
            counts = np.array(counts)
            if (counts == 3).all():
                faces, face_of = idx.reshape(-1, 3), np.arange(len(counts))
            else:
                start, faces, face_of = 0, [], []
                for f, c in enumerate(counts):  # fan triangulation
                    faces += [(idx[start], idx[start + j], idx[start + j + 1]) for j in range(1, c - 1)]
                    face_of += [f] * (c - 2)
                    start += c
                faces, face_of = np.array(faces), np.array(face_of)
            face_part = np.full(len(counts), part_id(body, UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]))
            for subset in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(prim)):
                bound = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
                face_part[np.array(subset.GetIndicesAttr().Get(), dtype=np.int64)] = part_id(body, bound)
            tri = pts[faces][..., :3]
            tris_xyz.append(tri)
            tris_link.append(np.full(len(tri), li))
            tris_part.append(face_part[face_of])
    tri, link_of, part_of = np.concatenate(tris_xyz), np.concatenate(tris_link), np.concatenate(tris_part)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tri), n_total, p=area / area.sum())
    a, b = rng.random((2, n_total, 1))
    flip = (a + b) > 1
    a, b = np.where(flip, 1 - a, a), np.where(flip, 1 - b, b)
    t = tri[pick]
    xyz = t[:, 0] + a * (t[:, 1] - t[:, 0]) + b * (t[:, 2] - t[:, 0])
    print(f"[render_real_states] sampled {n_total} surface points from {len(tri)} triangles in {len(parts)} parts; "
          f"no prim for {skipped}")
    return link_of[pick].astype(np.int32), xyz.astype(np.float32), part_of[pick].astype(np.int32), parts


def main() -> None:
    frames = np.load(args_cli.frames)
    names = [str(n) for n in frames["names"]]
    calib = json.load(open(args_cli.calibration))

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    apply_task_mode(env_cfg, args_cli.task_mode)
    if args_cli.look_dir:
        from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_real_look import LOOK_DIR, apply_real_look

        print(f"[render_real_states] real look: {apply_real_look(env_cfg, LOOK_DIR if args_cli.look_dir == 'default' else args_cli.look_dir)}")
    swap_to_openarm_joint_actions(env_cfg, absolute=True)
    for name in list(vars(env_cfg.terminations)):
        if not name.startswith("_"):
            setattr(env_cfg.terminations, name, None)
    env_cfg.recorders = None
    env_cfg.observations.policy.concatenate_terms = False

    cam_cfg = getattr(env_cfg.scene, args_cli.camera)
    height, width = frames["images"].shape[1:3]
    if (cam_cfg.height, cam_cfg.width) != (height, width):
        raise SystemExit(f"{args_cli.camera} renders {cam_cfg.height}x{cam_cfg.width}, the real frames are {height}x{width}.")
    if args_cli.intrinsics:
        fx, fy, cx, cy = _floats(args_cli.intrinsics, 4)
        old = cam_cfg.spawn
        cam_cfg.spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0], width, height,
            clipping_range=old.clipping_range, focal_length=old.focal_length, focus_distance=old.focus_distance,
        )
    if args_cli.cam_pos:
        cam_cfg.offset.pos = _floats(args_cli.cam_pos, 3)
    if args_cli.cam_rot:
        cam_cfg.offset.rot = _floats(args_cli.cam_rot, 4)
    if "distance_to_image_plane" not in cam_cfg.data_types:
        cam_cfg.data_types = list(cam_cfg.data_types) + ["distance_to_image_plane"]
    if cam_cfg.offset.convention != "ros":
        raise SystemExit(f"{args_cli.camera}'s offset is in the {cam_cfg.offset.convention!r} convention; this script assumes 'ros'.")
    parent_link = cam_cfg.prim_path.split("/Robot/")[1].split("/")[0]
    print(f"[render_real_states] {args_cli.camera} on {parent_link}: pos {tuple(cam_cfg.offset.pos)} rot {tuple(cam_cfg.offset.rot)}")

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    camera = env.scene[args_cli.camera]
    obj = env.scene["can"]
    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)
    parent_idx = body_names.index(parent_link)

    # Action slots, read off the live terms (a regex-resolved term orders joints as the articulation does).
    slots, offset = {}, 0
    for term_name in env.action_manager.active_terms:
        term = env.action_manager.get_term(term_name)
        if term_name.endswith("gripper_action"):
            slots[f"openarm_{'right' if term_name.startswith('right') else 'left'}_finger_joint1"] = offset
        else:
            for j, name in enumerate(term._joint_names):
                slots[name] = offset + j
        offset += term.action_dim
    action = torch.zeros(1, offset, device=env.device)
    surface_link, surface_xyz, surface_part, part_names = sample_robot_surface(robot.cfg.prim_path.replace("env_.*", "env_0"), body_names, args_cli.surface_points)
    parked = obj.data.default_root_state.clone()
    parked[:, :3] = env.scene.env_origins + torch.tensor([0.0, 3.0, 0.0], device=env.device)  # off the table, out of view

    os.makedirs(args_cli.out, exist_ok=True)
    record = {k: [] for k in ("depth", "body_pos_w", "body_quat_w", "cam_pos_w", "cam_quat_w_ros", "parent_pos_w", "parent_quat_w")}
    with torch.inference_mode():
        for k, state in enumerate(frames["state"]):
            sim = real_state_to_sim(state, names, calib)
            pos = robot.data.joint_pos.clone()
            for name, value in sim.items():
                pos[0, joint_names.index(name)] = value
                if name in slots:
                    action[0, slots[name]] = value
            robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
            robot.set_joint_position_target(pos)
            obj.write_root_state_to_sim(parked)
            for _ in range(args_cli.settle_steps):
                obs, *_ = env.step(action)
            err = (robot.data.joint_pos - pos).abs().max().item()
            if err > 1e-2:
                print(f"[render_real_states] frame {k}: robot settled {err:.3f} rad from the real state (joint limit?)")

            sim_img = obs["policy"][args_cli.camera][0, ..., :3].to(torch.uint8).cpu().numpy()
            real_img = frames["images"][k]
            Image.fromarray(sim_img).save(os.path.join(args_cli.out, f"sim_{k:02d}.png"))
            Image.fromarray(((sim_img.astype(np.uint16) + real_img) // 2).astype(np.uint8)).save(
                os.path.join(args_cli.out, f"overlay_{k:02d}.png"))
            record["depth"].append(camera.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy().astype(np.float32))
            record["body_pos_w"].append((robot.data.body_pos_w[0] - env.scene.env_origins[0]).cpu().numpy())
            record["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy())
            record["cam_pos_w"].append((camera.data.pos_w[0] - env.scene.env_origins[0]).cpu().numpy())
            record["cam_quat_w_ros"].append(camera.data.quat_w_ros[0].cpu().numpy())
            record["parent_pos_w"].append(record["body_pos_w"][-1][parent_idx])
            record["parent_quat_w"].append(record["body_quat_w"][-1][parent_idx])

    np.savez(
        os.path.join(args_cli.out, "poses.npz"),
        body_names=np.array(body_names), parent_link=parent_link, surface_link=surface_link, surface_xyz=surface_xyz,
        surface_part=surface_part, part_names=np.array(part_names),
        intrinsics=camera.data.intrinsic_matrices[0].cpu().numpy(),
        cam_offset_pos=np.array(cam_cfg.offset.pos), cam_offset_rot=np.array(cam_cfg.offset.rot),
        **{k: np.stack(v) for k, v in record.items()},
    )
    print(f"[render_real_states] {len(frames['state'])} frames -> {args_cli.out}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:  # Kit keeps the process alive after an uncaught exception otherwise
        simulation_app.close()
