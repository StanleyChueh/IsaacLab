# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Fit object placement and contact settings so replayed real battery episodes grasp what the real arms grasped.

The real left arm lifts the charger and the real right arm lifts the battery. Replaying their joint commands in sim with the
objects at the positions estimated from the camera (prepare_battery_replay.py, good to about a centimetre) does not lift
either: the grasp depends on where the finger pads land to a few millimetres. This searches, for all 9 episodes at once and
many candidates in parallel (one env per candidate x episode, no cameras), a common offset for each object's start pose
(x, y, yaw), the gripper's squeeze force, and the friction of fingers and objects, scoring each candidate by how high the
charger and the battery are lifted (capped at 5 and 2.5 cm).

  ./isaaclab.sh -p scripts/tools/sysid/fit_battery_replay.py --headless --replay logs/real2sim/battery/replay \\
      --out logs/real2sim/battery/fit
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-InsertBattery-OpenArm-IK-Abs-v0")
parser.add_argument("--replay", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--candidates", type=int, default=8, help="Candidates evaluated in parallel per batch.")
parser.add_argument("--batches", type=int, default=14)
parser.add_argument("--random_batches", type=int, default=4)
parser.add_argument("--calibration", default=os.path.expanduser("~/Stanley_ws/lerobot_openarm/calibration.json"))
parser.add_argument("--settle_steps", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402, F401
from isaaclab.envs.mdp import joint_pos_rel  # noqa: E402
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_joint_actions import (  # noqa: E402
    swap_to_openarm_joint_actions,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_states import real_state_to_sim  # noqa: E402

# name: (low, high, scale)
SPACE = {
    "charger_dx": (-0.10, 0.10, "lin"), "charger_dy": (-0.06, 0.06, "lin"), "charger_yaw": (-25.0, 25.0, "lin"),
    "battery_dx": (-0.12, 0.12, "lin"), "battery_dy": (-0.08, 0.08, "lin"), "battery_yaw": (-30.0, 30.0, "lin"),
    "grip_effort": (5.0, 80.0, "log"), "finger_friction": (0.2, 2.5, "log"), "object_friction": (0.3, 2.5, "log"),
}
NAMES = list(SPACE)


def to_params(u: np.ndarray) -> dict:
    out = {}
    for x, name in zip(np.clip(u, 0, 1), NAMES):
        lo, hi, kind = SPACE[name]
        out[name] = float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo)))) if kind == "log" else float(lo + x * (hi - lo))
    return out


@configclass
class MinimalObs:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=joint_pos_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


def main() -> None:
    traj = np.load(os.path.join(args_cli.replay, "trajectories.npz"))
    starts = json.load(open(os.path.join(args_cli.replay, "object_starts.json")))
    names = [str(n) for n in traj["names"]]
    calib = json.load(open(args_cli.calibration))
    episodes = sorted(int(k.split("_")[1]) for k in traj.files if k.startswith("state_"))
    n_ep, n_cand = len(episodes), args_cli.candidates
    n_env = n_ep * n_cand
    have = [(v["battery"][0] - v["charger"][0], v["battery"][1] - v["charger"][1]) for v in starts.values() if "battery" in v]
    off = np.median(np.array(have), 0)
    for v in starts.values():
        if "battery" not in v:
            v["battery"] = [v["charger"][0] + off[0], v["charger"][1] + off[1], 0.0]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n_env)
    swap_to_openarm_joint_actions(env_cfg, absolute=True)
    for cam in ("front_cam", "body_cam", "wrist_cam", "right_wrist_cam"):
        if hasattr(env_cfg.scene, cam):
            setattr(env_cfg.scene, cam, None)
    for attr in [a for a in vars(env_cfg.scene) if a.startswith("real_")]:
        setattr(env_cfg.scene, attr, None)  # the photo backdrop is for cameras
    env_cfg.observations.policy = MinimalObs.PolicyCfg()
    for name in list(vars(env_cfg.terminations)):
        if not name.startswith("_"):
            setattr(env_cfg.terminations, name, None)
    env_cfg.recorders = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    dev = env.device
    robot, charger, battery = env.scene["robot"], env.scene["charger"], env.scene["battery"]
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
    lengths = [len(traj[f"action_{e}"]) for e in episodes]
    T = max(lengths)
    actions = torch.zeros(n_ep, T, offset)
    first_pos = torch.zeros(n_ep, len(joint_names))
    for i, e in enumerate(episodes):
        acts, states = traj[f"action_{e}"], traj[f"state_{e}"]
        for k in range(T):
            for name, value in real_state_to_sim(acts[min(k, len(acts) - 1)], names, calib).items():
                if name in slots:
                    actions[i, k, slots[name]] = value
        for name, value in real_state_to_sim(states[0], names, calib).items():
            first_pos[i, joint_names.index(name)] = value
    ep_of_env = torch.arange(n_env, device=dev) % n_ep
    cand_of_env = torch.arange(n_env, device=dev) // n_ep
    actions_all, first_all = actions.to(dev)[ep_of_env], first_pos.to(dev)[ep_of_env]
    len_all = torch.tensor(lengths, device=dev)[ep_of_env]
    s_c = torch.tensor([starts[str(e)]["charger"] for e in episodes], device=dev)[ep_of_env]
    s_b = torch.tensor([starts[str(e)]["battery"] for e in episodes], device=dev)[ep_of_env]
    ids = robot.find_joints(["openarm_.*_finger_joint.*"])[0]
    print(f"[fit_battery_replay] {n_ep} episodes x {n_cand} candidates = {n_env} envs, {T} steps", flush=True)

    @torch.inference_mode()
    def evaluate(P: list[dict]) -> list[dict]:
        pc = {k: torch.tensor([p[k] for p in P], device=dev, dtype=torch.float32)[cand_of_env] for k in NAMES}
        eff = pc["grip_effort"][:, None].expand(-1, len(ids)).contiguous()
        robot.write_joint_effort_limit_to_sim(eff, joint_ids=ids)
        for asset, key in ((robot, "finger_friction"), (charger, "object_friction"), (battery, "object_friction")):
            view = asset.root_physx_view
            mats = view.get_material_properties().clone()
            mu = pc[key].cpu()[:, None]
            mats[..., 0], mats[..., 1] = mu, 0.9 * mu
            view.set_material_properties(mats, torch.arange(mats.shape[0], dtype=torch.int32))
        env.reset()
        pos = robot.data.joint_pos.clone()
        pos[:] = first_all
        robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
        robot.set_joint_position_target(pos)
        z0 = {}
        for asset, s, key in ((charger, s_c, "charger"), (battery, s_b, "battery")):
            root = asset.data.default_root_state.clone()
            root[:, 0] = env.scene.env_origins[:, 0] + s[:, 0] + pc[f"{key}_dx"]
            root[:, 1] = env.scene.env_origins[:, 1] + s[:, 1] + pc[f"{key}_dy"]
            half = torch.deg2rad(s[:, 2] + pc[f"{key}_yaw"]) / 2
            root[:, 3] = torch.cos(half)
            root[:, 4:6] = 0.0
            root[:, 6] = torch.sin(half)
            root[:, 7:] = 0.0
            asset.write_root_state_to_sim(root)
            z0[key] = root[:, 2] - env.scene.env_origins[:, 2]
        act = actions_all[:, 0].clone()
        for _ in range(args_cli.settle_steps):
            env.step(act)
        zmax = {k: torch.full((n_env,), -1.0, device=dev) for k in z0}
        for k in range(T):
            env.step(actions_all[:, k])
            live = k < len_all
            for key, asset in (("charger", charger), ("battery", battery)):
                z = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
                zmax[key] = torch.where(live, torch.maximum(zmax[key], z), zmax[key])
        lift = {key: (zmax[key] - z0[key]) * 100 for key in z0}
        out = []
        for c in range(n_cand):
            sel = cand_of_env == c
            lc, lb = lift["charger"][sel].clamp(min=0), lift["battery"][sel].clamp(min=0)
            score = ((lc.clamp(max=5.0) / 5.0) + (lb.clamp(max=2.5) / 2.5)).mean().item() / 2
            out.append({"score": score, "charger_lifted": int((lc > 4.0).sum()), "battery_lifted": int((lb > 2.0).sum()),
                        "both": int(((lc > 4.0) & (lb > 2.0)).sum()), "charger_lift_cm": lc.mean().item(), "battery_lift_cm": lb.mean().item()})
        return out

    os.makedirs(args_cli.out, exist_ok=True)
    rng = np.random.default_rng(args_cli.seed)
    log, best_u, best_s = [], None, -1.0
    zero_u = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, np.nan, np.nan, np.nan])
    sigma = 0.25
    for b in range(args_cli.batches):
        if b < args_cli.random_batches or best_u is None:
            U = rng.random((n_cand, len(NAMES)))
        else:
            U = np.clip(best_u + sigma * rng.standard_normal((n_cand, len(NAMES))), 0, 1)
            U[0] = best_u  # re-score the incumbent in the new batch (physics noise)
        if b == 0:  # the cfg's own contact settings, objects where the camera put them
            U[0] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, (np.log(25.0) - np.log(5.0)) / (np.log(80.0) - np.log(5.0)),
                    (np.log(0.5) - np.log(0.2)) / (np.log(2.5) - np.log(0.2)), (np.log(1.0) - np.log(0.3)) / (np.log(2.5) - np.log(0.3))]
        t0 = time.perf_counter()
        P = [to_params(u) for u in U]
        res = evaluate(P)
        improved = False
        for u, p, r in zip(U, P, res):
            log.append({"batch": b, "params": p, **r})
            if r["score"] > best_s:
                best_u, best_s, improved = u, r["score"], True
        sigma *= 0.85 if improved else 0.7
        top = max(res, key=lambda r: r["score"])
        print(f"[fit_battery_replay] batch {b:2d}: best of batch score {top['score']:.3f} (charger lifted {top['charger_lifted']}/{n_ep}, "
              f"battery {top['battery_lifted']}/{n_ep}, both {top['both']}); overall best {best_s:.3f}  ({time.perf_counter() - t0:.0f} s)", flush=True)
        json.dump({"best": to_params(best_u), "best_score": best_s, "log": log}, open(os.path.join(args_cli.out, "fit_log.json"), "w"), indent=1)
    print(f"[fit_battery_replay] BEST {to_params(best_u)} score {best_s:.3f}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
