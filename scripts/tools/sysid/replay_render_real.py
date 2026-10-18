# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Replay real episodes' actions in sim and save the three cameras' frames, under a chosen look.

For paired real/sim studies (prepare_real_replay.py writes the inputs): each real episode's recorded joint
targets drive the sim robot, with the can placed where it started in the real episode, so sim and real frames at the
same index share action, joint state and scene layout, and differ in how they look. Looks:

  legacy -- the camera setup this repo had before the real2sim work (hand-placed body camera, 733 px wrist lenses,
            white robot, grey pad)
  plain  -- the calibrated cameras (current cfg) with the sim's own materials, light and table
  look   -- plain + openarm_real_look (colours, light, photo backdrop) + the fitted sensor model

Frames are resized to --size (square, area-averaged) and written as JPEG: <out>/<look>/ep<E>/<camera>_<k:04d>.jpg.
Also writes <out>/<look>/summary.json: where the can ended up per episode (did the replay carry it?).

  ./isaaclab.sh -p scripts/tools/sysid/replay_render_real.py --headless --look look --replay logs/real2sim/replay \\
      --out logs/real2sim/replay_frames --episodes 0-49
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0")
parser.add_argument("--task_mode", default="handover")
parser.add_argument("--look", choices=["legacy", "plain", "look"], required=True)
parser.add_argument("--replay", required=True, help="Folder with trajectories.npz and can_starts.json.")
parser.add_argument("--can_starts", default=None, help="Override <replay>/can_starts.json.")
parser.add_argument("--out", required=True)
parser.add_argument("--episodes", default="0-49", help="e.g. 0-49 or 3,5,8")
parser.add_argument("--size", type=int, default=224)
parser.add_argument("--calibration", default=os.path.expanduser("~/Stanley_ws/lerobot_openarm/calibration.json"))
parser.add_argument("--settle_steps", type=int, default=20, help="Steps the robot is held at the first state, so the can settles.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_joint_actions import (  # noqa: E402
    swap_to_openarm_joint_actions,
)
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_task_modes import apply_task_mode  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_states import real_state_to_sim  # noqa: E402

CAMERAS = ("body_cam", "wrist_cam", "right_wrist_cam")
CAN_REST_Z = 0.28 + 0.1 + 0.003  # pad top + half the can's length + a hair


def episode_list(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def apply_legacy_cameras(env_cfg) -> None:
    """The camera cfg before the real2sim calibration (git history of stack_ik_abs_visuomotor_env_cfg.py)."""
    for name in CAMERAS:
        getattr(env_cfg.scene, name).spawn.horizontal_aperture = 20.955
    body = env_cfg.scene.body_cam.offset
    body.pos, body.rot = (0.04563, 0.0, 0.97524), (0.1203, -0.6968, 0.6968, -0.1203)


def main() -> None:
    traj = np.load(os.path.join(args_cli.replay, "trajectories.npz"))
    can_starts = json.load(open(args_cli.can_starts or os.path.join(args_cli.replay, "can_starts.json")))
    names = [str(n) for n in traj["names"]]
    calib = json.load(open(args_cli.calibration))

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    apply_task_mode(env_cfg, args_cli.task_mode)
    sensor = None
    if args_cli.look == "legacy":
        apply_legacy_cameras(env_cfg)
    elif args_cli.look == "look":
        from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_real_look import (
            LOOK_DIR,
            SensorModel,
            apply_real_look,
        )

        print(f"[replay_render_real] real look: {apply_real_look(env_cfg, LOOK_DIR)}")
        sensor = SensorModel(LOOK_DIR)
    swap_to_openarm_joint_actions(env_cfg, absolute=True)
    for name in list(vars(env_cfg.terminations)):
        if not name.startswith("_"):
            setattr(env_cfg.terminations, name, None)
    env_cfg.recorders = None
    env_cfg.observations.policy.concatenate_terms = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    robot, can = env.scene["robot"], env.scene["can"]
    joint_names = list(robot.data.joint_names)
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

    def to_action(real_row: np.ndarray) -> dict:
        sim = real_state_to_sim(real_row, names, calib)
        return sim

    def set_action(sim: dict) -> None:
        for name, value in sim.items():
            if name in slots:
                action[0, slots[name]] = value

    out_root = os.path.join(args_cli.out, args_cli.look)
    os.makedirs(out_root, exist_ok=True)
    summary = {}
    with torch.inference_mode():
        for ep in episode_list(args_cli.episodes):
            if str(ep) not in can_starts:
                print(f"[replay_render_real] episode {ep}: no can start, skipped")
                continue
            states, actions = traj[f"state_{ep}"], traj[f"action_{ep}"]
            os.makedirs(os.path.join(out_root, f"ep{ep}"), exist_ok=True)
            env.reset()
            first = real_state_to_sim(states[0], names, calib)
            pos = robot.data.joint_pos.clone()
            for name, value in first.items():
                pos[0, joint_names.index(name)] = value
            robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
            robot.set_joint_position_target(pos)
            set_action(first)
            x, y = can_starts[str(ep)]
            root = can.data.default_root_state.clone()
            root[:, :3] = env.scene.env_origins + torch.tensor([x, y, CAN_REST_Z], device=env.device)
            root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            root[:, 7:] = 0.0
            can.write_root_state_to_sim(root)
            for _ in range(args_cli.settle_steps):
                obs, *_ = env.step(action)
            z_track = []
            for k in range(len(actions)):
                set_action(to_action(actions[k]))
                obs, *_ = env.step(action)
                z_track.append(float(can.data.root_pos_w[0, 2] - env.scene.env_origins[0, 2]))
                for cam in CAMERAS:
                    img = obs["policy"][cam][0, ..., :3].to(torch.uint8)
                    if sensor is not None:
                        img = sensor(cam, img)
                    small = torch.nn.functional.interpolate(
                        img.permute(2, 0, 1)[None].float(), size=(args_cli.size, args_cli.size), mode="area")[0]
                    Image.fromarray(small.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()).save(
                        os.path.join(out_root, f"ep{ep}", f"{cam}_{k:04d}.jpg"), quality=90)
            summary[str(ep)] = {"frames": len(actions), "can_start_z": z_track[0], "can_max_z": max(z_track), "can_end_z": z_track[-1],
                                "can_end_xy": [float(v) for v in (can.data.root_pos_w[0, :2] - env.scene.env_origins[0, :2])]}
            print(f"[replay_render_real] episode {ep}: {len(actions)} frames, can z {z_track[0]:.3f} -> max {max(z_track):.3f} -> {z_track[-1]:.3f}", flush=True)
            json.dump(summary, open(os.path.join(out_root, "summary.json"), "w"), indent=1)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
