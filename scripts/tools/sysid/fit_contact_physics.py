# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Fit the sim's contact physics against real episodes: make replayed real motions carry the can like the real ones did.

Every real episode in the dataset picks the can up, hands it over, and ends with it held. Replaying the recorded joint
commands in sim, with the can placed where the real one started, carries it in under half of them: the can is knocked,
squeezed out, or dropped, because friction, the gripper's squeeze and the can's mass in the sim are not the real
ones. This script searches those parameters for the set that makes the sim replays behave like the real episodes.

Evaluation: all episodes are replayed at once, one per parallel env (no camera rendering, so ~1 s per 100 steps for all of
them). For each episode the score is the fraction of the carry phase -- from shortly after the right gripper first closes
to the end -- during which one of the two jaws is still on the can (within the can's radius of its axis, along its
length). That is smooth in the parameters, unlike "the can ended up high". It also records how many episodes lift the
can above 0.46 m and end with it above 0.36 m.

Search: random global samples, then an evolution strategy around the best, scored on --train episodes only; the best
parameter set is reported on the held-out --test episodes, so a fit that only memorised the training episodes shows up.

  ./isaaclab.sh -p scripts/tools/sysid/fit_contact_physics.py --headless --replay logs/real2sim/replay \\
      --can_starts logs/real2sim/replay/can_starts_use.json --out logs/real2sim/contact_fit
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0")
parser.add_argument("--task_mode", default="handover")
parser.add_argument("--replay", required=True)
parser.add_argument("--can_starts", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--train", default="0-34")
parser.add_argument("--test", default="35-49")
parser.add_argument("--random_samples", type=int, default=24)
parser.add_argument("--generations", type=int, default=8)
parser.add_argument("--population", type=int, default=8)
parser.add_argument("--only_default", action="store_true", help="Just score the cfg's own physics and exit.")
parser.add_argument("--probe", action="store_true", help="Apply each parameter group once, alone, and time it (debugging).")
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
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_joint_actions import (  # noqa: E402
    swap_to_openarm_joint_actions,
)
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_real_look import apply_contact_physics  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_task_modes import apply_task_mode  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_states import real_state_to_sim  # noqa: E402

CAN_REST_Z = 0.28 + 0.1 + 0.003
JAW_DEPTH = 0.15  # m along the tool z axis from the ee_tcp to the middle of the finger pads
CAN_HALF_LENGTH, CAN_RADIUS = 0.10, 0.0293

# name: (low, high, scale, default marker). Searched in a normalised cube; "log" parameters in log space.
SPACE = {
    "can_friction": (0.2, 4.0, "log"),
    "finger_friction": (0.2, 4.0, "log"),
    "can_mass": (0.10, 0.40, "lin"),
    "grip_stiffness": (100.0, 4000.0, "log"),
    "grip_damping": (1.0, 200.0, "log"),
    "grip_effort": (2.0, 80.0, "log"),
    "can_dx": (-0.025, 0.025, "lin"),
    "can_dy": (-0.025, 0.025, "lin"),
}
NAMES = list(SPACE)


def to_params(u: np.ndarray) -> dict:
    out = {}
    for x, name in zip(np.clip(u, 0, 1), NAMES):
        lo, hi, kind = SPACE[name]
        out[name] = float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo)))) if kind == "log" else float(lo + x * (hi - lo))
    return out


def to_unit(params: dict) -> np.ndarray:
    u = []
    for name in NAMES:
        lo, hi, kind = SPACE[name]
        v = params[name]
        u.append((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if kind == "log" else (v - lo) / (hi - lo))
    return np.array(u)


def span(s):
    lo, _, hi = s.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


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
    can_starts = json.load(open(args_cli.can_starts))
    names = [str(n) for n in traj["names"]]
    calib = json.load(open(args_cli.calibration))
    train, test = span(args_cli.train), span(args_cli.test)
    episodes = [e for e in train + test if str(e) in can_starts]
    n_env = len(episodes)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n_env)
    apply_task_mode(env_cfg, args_cli.task_mode)
    swap_to_openarm_joint_actions(env_cfg, absolute=True)
    for cam in ("front_cam", "body_cam", "wrist_cam", "right_wrist_cam"):
        if hasattr(env_cfg.scene, cam):
            setattr(env_cfg.scene, cam, None)  # replay needs no images: this is what makes 50 envs at once cheap
    env_cfg.observations.policy = MinimalObs.PolicyCfg()
    for name in list(vars(env_cfg.terminations)):
        if not name.startswith("_"):
            setattr(env_cfg.terminations, name, None)
    env_cfg.recorders = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    dev = env.device
    robot, can = env.scene["robot"], env.scene["can"]
    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)
    tcp_ids = [body_names.index("openarm_left_ee_tcp"), body_names.index("openarm_right_ee_tcp")]

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
    actions = torch.zeros(n_env, T, offset)
    first_pos = torch.zeros(n_env, len(joint_names))
    carry_from = np.zeros(n_env, int)
    gr = calib["right"]["gripper"]
    for i, e in enumerate(episodes):
        acts, states = traj[f"action_{e}"], traj[f"state_{e}"]
        for k in range(T):
            sim = real_state_to_sim(acts[min(k, len(acts) - 1)], names, calib)
            for name, value in sim.items():
                if name in slots:
                    actions[i, k, slots[name]] = value
        for name, value in real_state_to_sim(states[0], names, calib).items():
            first_pos[i, joint_names.index(name)] = value
        frac = np.clip((states[:, names.index("RJ8.pos")] - gr["closed_raw"]) / (gr["open_raw"] - gr["closed_raw"]), 0, 1)
        op = np.flatnonzero(frac > 0.85)
        k_close = next((j for j in range(op[0], len(frac)) if frac[j] < 0.6), len(frac) // 2) if len(op) else len(frac) // 2
        carry_from[i] = min(k_close + 8, lengths[i] - 1)
    actions, first_pos = actions.to(dev), first_pos.to(dev)
    start_xy = torch.tensor([can_starts[str(e)] for e in episodes], device=dev)
    lengths_t = torch.tensor(lengths, device=dev)
    k_idx = torch.arange(T, device=dev)[None]
    in_episode = k_idx < lengths_t[:, None]
    in_carry = (k_idx >= torch.tensor(carry_from, device=dev)[:, None]) & in_episode
    default_params = apply_contact_physics(env, {})  # read-only: what the cfg gives
    print(f"[fit_contact_physics] {n_env} episodes, {T} steps; cfg physics: {default_params}", flush=True)

    @torch.inference_mode()
    def evaluate(params: dict) -> dict:
        apply_contact_physics(env, {k: v for k, v in params.items() if not k.startswith("can_d") and v is not None})
        env.reset()
        pos = robot.data.joint_pos.clone()
        pos[:] = first_pos
        robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
        robot.set_joint_position_target(pos)
        root = can.data.default_root_state.clone()
        root[:, :2] = env.scene.env_origins[:, :2] + start_xy + torch.tensor([params.get("can_dx", 0.0), params.get("can_dy", 0.0)], device=dev)
        root[:, 2] = env.scene.env_origins[:, 2] + CAN_REST_Z
        root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        root[:, 7:] = 0.0
        can.write_root_state_to_sim(root)
        act = actions[:, 0].clone()
        for _ in range(args_cli.settle_steps):
            env.step(act)
        near = torch.zeros(n_env, T, dtype=torch.bool, device=dev)
        z = torch.zeros(n_env, T, device=dev)
        for k in range(T):
            env.step(actions[:, k])
            c = can.data.root_pos_w - env.scene.env_origins
            axis = quat_apply(can.data.root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n_env, 3))
            hit = torch.zeros(n_env, dtype=torch.bool, device=dev)
            for tid in tcp_ids:
                q = robot.data.body_quat_w[:, tid]
                jaw = robot.data.body_pos_w[:, tid] - env.scene.env_origins + JAW_DEPTH * quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n_env, 3))
                rel = jaw - c
                along = (rel * axis).sum(-1)
                perp = (rel - along[:, None] * axis).norm(dim=-1)
                hit |= (along.abs() < CAN_HALF_LENGTH + 0.01) & (perp < CAN_RADIUS + 0.015)
            near[:, k] = hit
            z[:, k] = c[:, 2]
        follow = ((near & in_carry).sum(1).float() / in_carry.sum(1).clamp(min=1)).cpu().numpy()
        zz = z.masked_fill(~in_episode, -1.0)
        lifted = (zz.max(1).values > 0.46).cpu().numpy()
        end_z = z[torch.arange(n_env, device=dev), lengths_t - 1].cpu().numpy()
        return {"follow": follow, "lifted": lifted, "up_end": end_z > 0.36}

    def summarise(res: dict, eps: list[int]) -> dict:
        idx = [episodes.index(e) for e in eps if e in episodes]
        return {"follow": float(res["follow"][idx].mean()), "lifted": int(res["lifted"][idx].sum()),
                "up_end": int(res["up_end"][idx].sum()), "n": len(idx), "held_both": int((res["lifted"][idx] & res["up_end"][idx]).sum())}

    os.makedirs(args_cli.out, exist_ok=True)
    log = []

    def run(label: str, params: dict) -> float:
        t0 = time.perf_counter()
        res = evaluate(params)
        tr, te = summarise(res, train), summarise(res, test)
        log.append({"label": label, "params": params, "train": tr, "test": te})
        json.dump(log, open(os.path.join(args_cli.out, "fit_log.json"), "w"), indent=1)
        print(f"[fit_contact_physics] {label:12s} train follow {tr['follow']:.3f} held {tr['held_both']}/{tr['n']}   "
              f"test follow {te['follow']:.3f} held {te['held_both']}/{te['n']}   ({time.perf_counter() - t0:.0f} s)", flush=True)
        return tr["follow"]

    baseline = {"can_friction": None, "finger_friction": None, "can_mass": None, "grip_stiffness": None,
                "grip_damping": None, "grip_effort": None, "can_dx": 0.0, "can_dy": 0.0}
    run("cfg default", baseline)
    if args_cli.probe:
        for name, value in (("can_friction", 1.5), ("finger_friction", 1.5), ("can_mass", 0.25), ("grip_stiffness", 1500.0),
                            ("grip_damping", 30.0), ("grip_effort", 20.0)):
            print(f"[fit_contact_physics] probe {name}={value} ...", flush=True)
            t0 = time.perf_counter()
            apply_contact_physics(env, {name: value})
            print(f"[fit_contact_physics]   set in {time.perf_counter() - t0:.2f} s", flush=True)
            run(f"probe {name}", {**baseline, name: value})
        return
    if args_cli.only_default:
        return
    rng = np.random.default_rng(args_cli.seed)
    best_u, best_f = None, -1.0
    for i in range(args_cli.random_samples):
        u = rng.random(len(NAMES))
        for name in ("can_dx", "can_dy"):
            u[NAMES.index(name)] = 0.5 + 0.2 * (rng.random() - 0.5)  # offsets: start near 0
        f = run(f"random {i}", to_params(u))
        if f > best_f:
            best_u, best_f = u, f
    sigma = 0.2
    for g in range(args_cli.generations):
        cands = [np.clip(best_u + sigma * rng.standard_normal(len(NAMES)), 0, 1) for _ in range(args_cli.population)]
        improved = False
        for j, u in enumerate(cands):
            f = run(f"gen {g}.{j}", to_params(u))
            if f > best_f:
                best_u, best_f, improved = u, f, True
        sigma *= 0.85 if improved else 0.7
    final = to_params(best_u)
    json.dump({"params": final, "train_follow": best_f, "log": log}, open(os.path.join(args_cli.out, "best.json"), "w"), indent=1)
    print(f"[fit_contact_physics] best: {final}", flush=True)
    run("BEST", final)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()  # printed here, because Kit can hang in close() before Python reports the exception
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
