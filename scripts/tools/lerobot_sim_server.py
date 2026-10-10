# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Serve an OpenArm task to lerobot's official CLIs: the simulated robot behind --robot.type=openarm_isaac.

With this running, lerobot-record / lerobot-teleoperate / lerobot-rollout drive Isaac Sim through the SAME
pipeline as the real arm (lerobot_openarm, branch official-lerobot-quest): the openarm_quest teleop and its
IK, the robot plugin's start pose, step limit, safety guard and slow returns, lerobot-record's 30 Hz loop
and its dataset format. Only the bottom layer differs -- this process instead of the CAN bus and the
RealSense cameras -- so a sim dataset has exactly the keys, units, action meaning and episode structure of
a real one.

  ./isaaclab.sh -p scripts/tools/lerobot_sim_server.py --enable_cameras \\
      --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 --task_mode handover

then, in lerobot_openarm's venv: lerobot-record --robot.type=openarm_isaac ... (see its README).

Lockstep: the simulation advances one control step (1 / 30 s of sim time) per `step` request and is
paused otherwise, so each recorded frame is exactly one control period of sim time even when rendering
three cameras is slower than real time. While nothing is asking, the viewport keeps rendering (no
physics), so the window stays alive.

The action space is replaced by absolute joint targets (openarm_joint_actions.swap_to_openarm_joint_actions):
7 arm joints + one finger position per arm, the same joint convention calibration.json maps to the motors.
Every termination is removed: the scene resets only when the robot asks (between episodes), as a real
table does -- a dropped can stays dropped. The task's success term is still evaluated and reported.

The wire protocol is lerobot_openarm's plugins/.../sim_link.py, loaded from --lerobot_openarm.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0")
parser.add_argument("--task_mode", default="handover", help="left | right | handover (openarm_task_modes.py)")
parser.add_argument(
    "--domain_randomization",
    default="none",
    choices=["none", "visual", "full"],
    help="Randomization profile applied at every scene reset (pickup_ik_abs_env_cfg.DOMAIN_RANDOMIZATION_PROFILES).",
)
parser.add_argument("--enable_camera_shake", action="store_true", help="With --domain_randomization: also shake the cameras.")
parser.add_argument("--host", default="127.0.0.1", help="Pickle over TCP: keep it on localhost.")
parser.add_argument("--port", type=int, default=5710)
parser.add_argument(
    "--lerobot_openarm",
    default=os.path.expanduser("~/Stanley_ws/lerobot_openarm"),
    help="The lerobot_openarm checkout whose sim_link.py defines the wire protocol.",
)
parser.add_argument("--settle_steps", type=int, default=3, help="Steps run after a reset before answering it.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # the robot's observations ARE the cameras

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.util  # noqa: E402
import select  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_joint_actions import (  # noqa: E402
    swap_to_openarm_joint_actions,
)
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_sim_timing import CONTROL_HZ  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_task_modes import apply_task_mode  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.pickup_ik_abs_env_cfg import (  # noqa: E402
    attach_domain_randomization,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def _load_sim_link(checkout: str):
    path = os.path.join(
        checkout, "plugins", "lerobot_robot_openarm_umeow", "lerobot_robot_openarm_umeow", "sim_link.py"
    )
    if not os.path.isfile(path):
        raise SystemExit(f"sim_link.py not found at {path}: pass --lerobot_openarm <checkout>.")
    spec = importlib.util.spec_from_file_location("openarm_sim_link", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


link = _load_sim_link(args_cli.lerobot_openarm)

# The action terms swap_to_openarm_joint_actions installs, in the cfg's field order (= action vector order).
ACTION_TERMS = ("arm_action", "gripper_action", "right_arm_action", "right_gripper_action")
REPORTED_JOINTS = [f"openarm_{s}_joint{n}" for s in ("left", "right") for n in range(1, 8)] + [
    "openarm_left_finger_joint1",
    "openarm_right_finger_joint1",
]


class SimRobotServer:
    def __init__(self, env, success_term):
        self.env = env
        self.robot = env.scene["robot"]
        self.success_term = success_term
        names = list(self.robot.data.joint_names)
        missing = [j for j in REPORTED_JOINTS if j not in names]
        if missing:
            raise SystemExit(f"The task's robot has no joint(s) {missing}; is --task an OpenArm task?")
        self.joint_idx = [names.index(j) for j in REPORTED_JOINTS]

        # Where each joint's target goes in the 16D action vector: read off the live terms rather than
        # assumed, since a regex-resolved term orders its joints as the articulation does.
        self.action_slots: dict[str, int] = {}
        offset = 0
        for term_name in ACTION_TERMS:
            term = env.action_manager.get_term(term_name)
            if term_name.endswith("gripper_action"):
                side = "right" if term_name.startswith("right") else "left"
                self.action_slots[f"openarm_{side}_finger_joint1"] = offset
            else:
                for j, name in enumerate(term._joint_names):
                    self.action_slots[name] = offset + j
            offset += term.action_dim
        if offset != env.action_manager.total_action_dim or set(self.action_slots) != set(REPORTED_JOINTS):
            raise SystemExit(
                f"Unexpected action layout {self.action_slots} ({env.action_manager.total_action_dim}D):"
                " swap_to_openarm_joint_actions changed?"
            )
        self.action = torch.zeros(1, offset, device=env.device)

        self.obs = None
        self.success = False
        self.step_count = 0
        self.step_s = 0.0  # time inside env.step (physics + rendering), for the rate line
        policy_obs = env.observation_manager.compute()["policy"]
        self.cameras = {
            k: list(v.shape[1:3]) for k, v in policy_obs.items() if v.ndim == 4 and v.shape[-1] in (3, 4)
        }
        if not self.cameras:
            raise SystemExit("The task's policy observations hold no camera images.")

    # -- simulation -------------------------------------------------------------------------------------
    def _set_targets(self, joints: dict) -> None:
        for name, value in joints.items():
            slot = self.action_slots.get(name)
            if slot is not None:
                self.action[0, slot] = float(value)

    def _step(self) -> None:
        self.obs, *_ = self.env.step(self.action)
        self.step_count += 1
        if self.success_term is not None:
            done = self.success_term.func(self.env, **self.success_term.params)
            self.success = bool(done[0])

    def _observation(self) -> dict:
        pos = self.robot.data.joint_pos[0, self.joint_idx].cpu().numpy()
        images = {}
        for cam in self.cameras:
            img = self.obs["policy"][cam][0]
            images[cam] = img[..., :3].to(torch.uint8).cpu().numpy()
        return {
            "joints": {name: float(v) for name, v in zip(REPORTED_JOINTS, pos)},
            "images": images,
            "success": self.success,
            "sim_time": self.step_count / CONTROL_HZ,
            "step": self.step_count,
        }

    def reset(self, joints: dict | None) -> dict:
        """Re-randomize the scene. With `joints`, put the robot there; without, keep it exactly where it is
        (pose, velocity, targets), as when a person resets the table between episodes."""
        kept = None
        if not joints:
            kept = (self.robot.data.joint_pos.clone(), self.robot.data.joint_vel.clone(), self.action.clone())
        self.obs, _ = self.env.reset()
        self.success = False
        if kept is not None:
            self.robot.write_joint_state_to_sim(kept[0], kept[1])
            self.action = kept[2]
            for _ in range(max(1, args_cli.settle_steps)):  # also renders the cameras at the new scene
                self._step()
            self.step_count = 0
            return self._observation()
        if joints:
            names = list(self.robot.data.joint_names)
            pos = self.robot.data.joint_pos.clone()
            for name, value in joints.items():
                pos[0, names.index(name)] = float(value)
                mimic = name.replace("finger_joint1", "finger_joint2")
                if mimic != name and mimic in names:
                    pos[0, names.index(mimic)] = float(value)
            self.robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
            self.robot.set_joint_position_target(pos)
        current = self.robot.data.joint_pos[0, self.joint_idx].cpu().numpy()
        self._set_targets(dict(zip(REPORTED_JOINTS, current)))
        if joints:
            self._set_targets(joints)
        for _ in range(max(1, args_cli.settle_steps)):  # also renders the cameras at the new scene
            self._step()
        self.step_count = 0
        return self._observation()

    # -- protocol ---------------------------------------------------------------------------------------
    def handle(self, msg: dict) -> dict:
        cmd = msg.get("cmd")
        if cmd == "hello":
            return {
                "protocol": link.PROTOCOL,
                "joint_names": REPORTED_JOINTS,
                "cameras": self.cameras,
                "control_hz": CONTROL_HZ,
                "task": args_cli.task,
                "task_mode": args_cli.task_mode,
                "domain_randomization": args_cli.domain_randomization,
            }
        if cmd == "reset":
            return self.reset(msg.get("joints"))
        if cmd == "step":
            self._set_targets(msg["joints"])
            t0 = time.perf_counter()
            self._step()
            self.step_s += time.perf_counter() - t0
            return self._observation()
        if cmd == "observe":
            return self._observation()
        return {"error": f"unknown command {cmd!r}"}

    def serve(self, host: str, port: int) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        server.settimeout(0.05)
        print(f"[lerobot_sim_server] ready on {host}:{port} -- {args_cli.task} ({args_cli.task_mode}),"
              f" {CONTROL_HZ} Hz, cameras {self.cameras}, randomization: {args_cli.domain_randomization}", flush=True)
        gui = self.env.sim.has_gui()
        while simulation_app.is_running():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                if gui:
                    self.env.sim.render()
                continue
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(None)  # frames are read whole and blocking; idling is select()'s job
            print(f"[lerobot_sim_server] robot connected from {addr[0]}:{addr[1]}", flush=True)
            n, t0 = 0, time.perf_counter()
            try:
                while simulation_app.is_running():
                    if not select.select([conn], [], [], 0.05)[0]:
                        if gui:
                            self.env.sim.render()  # keep the viewport alive while nobody asks; no physics
                        continue
                    msg = link.recv_msg(conn)
                    try:
                        reply = self.handle(msg)
                    except Exception as e:  # report to the robot, keep serving
                        reply = {"error": f"{type(e).__name__}: {e}"}
                    link.send_msg(conn, reply)
                    if msg.get("cmd") == "step":
                        n += 1
                        if time.perf_counter() - t0 >= 10.0:
                            rate = n / (time.perf_counter() - t0)
                            slow = "" if rate >= 0.95 * CONTROL_HZ else "  <-- slower than real time"
                            print(f"[lerobot_sim_server] {rate:.1f} steps/s, env.step {1e3 * self.step_s / n:.0f} ms"
                                  f" of {1e3 / rate:.0f} ms per step{slow}", flush=True)
                            n, t0, self.step_s = 0, time.perf_counter(), 0.0
            except (ConnectionError, OSError) as e:
                print(f"[lerobot_sim_server] robot disconnected ({e}); waiting for the next one.", flush=True)
            except Exception as e:  # an unreadable frame: drop this connection, keep the simulator up
                print(f"[lerobot_sim_server] dropped the connection: {type(e).__name__}: {e}", flush=True)
            finally:
                conn.close()
        server.close()


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task.split(":")[-1]
    apply_task_mode(env_cfg, args_cli.task_mode)
    if args_cli.domain_randomization != "none":
        attached = attach_domain_randomization(env_cfg, args_cli.domain_randomization, args_cli.enable_camera_shake)
        print(f"[lerobot_sim_server] domain randomization '{args_cli.domain_randomization}': {', '.join(attached)}")
    swap_to_openarm_joint_actions(env_cfg, absolute=True)

    success_term = getattr(env_cfg.terminations, "success", None)
    for name in list(vars(env_cfg.terminations)):
        if not name.startswith("_"):
            setattr(env_cfg.terminations, name, None)
    env_cfg.recorders = None
    env_cfg.observations.policy.concatenate_terms = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    server = SimRobotServer(env, success_term)
    try:
        server.serve(args_cli.host, args_cli.port)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
