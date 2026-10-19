# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Run the scripted battery expert (battery_expert.py) in many envs at once, no cameras, and report what happened.

The development loop for the expert: every env gets its own scattered charger and battery (the task's reset events), its own
``BatteryExpert``, and the joint targets it asks for are the env's action, exactly as the lerobot sim server would apply them.
Prints per env the stage it reached, whether the battery ended in the charger's bay, and how high each prop was lifted.

  ./isaaclab.sh -p scripts/tools/sysid/run_battery_expert.py --headless --num_envs 16 --out logs/real2sim/battery/expert
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="Isaac-InsertBattery-OpenArm-IK-Abs-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--max_steps", type=int, default=1500)
parser.add_argument("--out", default="logs/real2sim/battery/expert")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--debug_env", type=int, default=-1, help="Print this env's hands, objects and stage every --debug_every steps.")
parser.add_argument("--hide_battery", action="store_true", help="Put the battery 3 m away (is the charger what blocks the left hand?).")
parser.add_argument("--sweep", default="", help="joint name,lo,hi: sweep one joint over the range, print measured (no expert).")
parser.add_argument("--goto", default="", help="side:q1,...,q7 : drive one arm there from home over 3 s and print measured joints (no expert).")
parser.add_argument("--place", default="", help="cx,cy,cyaw_deg:bx,by,byaw_deg  every env starts like this (instead of the task's scatter).")
parser.add_argument("--scatter", action="store_true", help="every env starts at a random pose from the region record_expert_episodes.py draws from.")
parser.add_argument("--hide_charger", action="store_true")
parser.add_argument("--hide_charger_at", type=int, default=-1, help="Teleport the charger away at this step (after the expert planned from it).")
parser.add_argument("--debug_every", type=int, default=15)
parser.add_argument("--set", nargs="*", default=[], help="BatteryExpert overrides, name=value (angles in degrees for *_pitch).")
parser.add_argument("--contact", nargs="*", default=[], help="contact overrides, name=value (grip_effort, finger_friction, object_friction).")
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
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_battery_assets import battery_in_charger  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_real_look import apply_contact_physics  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery_expert import BatteryExpert, Failed, base_transform  # noqa: E402

REPORTED = [f"openarm_{s}_joint{n}" for s in ("left", "right") for n in range(1, 8)] + ["openarm_left_finger_joint1", "openarm_right_finger_joint1"]


@configclass
class MinimalObs:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=joint_pos_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


def parse_overrides(items, degrees=True):
    out = {}
    for it in items:
        k, v = it.split("=")
        v = float(v)
        out[k] = np.radians(v) if (degrees and k.endswith("_pitch")) else v
    return out


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    swap_to_openarm_joint_actions(env_cfg, absolute=True)
    for cam in ("front_cam", "body_cam", "wrist_cam", "right_wrist_cam"):
        if hasattr(env_cfg.scene, cam):
            setattr(env_cfg.scene, cam, None)
    for attr in [a for a in vars(env_cfg.scene) if a.startswith("real_")]:
        setattr(env_cfg.scene, attr, None)
    env_cfg.observations.policy = MinimalObs.PolicyCfg()
    for name in list(vars(env_cfg.terminations)):
        if not name.startswith("_"):
            setattr(env_cfg.terminations, name, None)
    env_cfg.recorders = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    dev = env.device
    look_dir = env.cfg.real_look_dir
    contact = json.load(open(os.path.join(look_dir, "contact_physics.json")))["params"]
    contact.update(parse_overrides(args_cli.contact, degrees=False))
    print("[run_battery_expert] contact physics", apply_contact_physics(env, contact), flush=True)
    robot, charger, battery = env.scene["robot"], env.scene["charger"], env.scene["battery"]
    if os.environ.get("DUMP_FINGER"):
        import isaaclab.sim as sim_utils_
        from pxr import Usd, UsdPhysics

        link = sim_utils_.get_current_stage().GetPrimAtPath("/World/envs/env_0/Robot/openarm_left_left_finger")
        for p_ in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            print("  [finger]", p_.GetPath(), p_.GetTypeName(), "inst" if p_.IsInstance() else "", "proxy" if p_.IsInstanceProxy() else "", "COLL" if p_.HasAPI(UsdPhysics.CollisionAPI) else "", flush=True)
    print("[run_battery_expert] arm gravity disabled in cfg:", env.cfg.scene.robot.spawn.rigid_props.disable_gravity, "| physx view:", [bool(v) for v in robot.root_physx_view.get_disable_gravities()[0][:3].cpu().numpy()] if hasattr(robot.root_physx_view, "get_disable_gravities") else "n/a", flush=True)
    names = list(robot.data.joint_names)
    idx = [names.index(j) for j in REPORTED]
    slots, offset = {}, 0
    for term_name in env.action_manager.active_terms:
        term = env.action_manager.get_term(term_name)
        if term_name.endswith("gripper_action"):
            slots[f"openarm_{'right' if term_name.startswith('right') else 'left'}_finger_joint1"] = offset
        else:
            for j, name in enumerate(term._joint_names):
                slots[name] = offset + j
        offset += term.action_dim
    N = args_cli.num_envs
    overrides = parse_overrides(args_cli.set)
    torch.manual_seed(args_cli.seed)

    def observe(i):
        pos = robot.data.joint_pos[i, idx].cpu().numpy()
        o = env.scene.env_origins[i]
        objects = {}
        for key, asset in (("charger", charger), ("battery", battery)):
            objects[key] = [float(v) for v in torch.cat([asset.data.root_pos_w[i] - o, asset.data.root_quat_w[i]]).cpu()]
        base = torch.cat([robot.data.root_pos_w[i] - o, robot.data.root_quat_w[i]])
        return {"joints": dict(zip(REPORTED, map(float, pos))), "objects": objects, "robot_base": [float(v) for v in base.cpu()]}

    sys.path.insert(0, os.path.expanduser("~/Stanley_ws/lerobot_openarm"))
    from sim_bridge_common import load_calibration, motor_action_to_sim_joints

    traj = np.load("logs/real2sim/battery/replay/trajectories.npz")
    tnames = [str(n) for n in traj["names"]]
    home = motor_action_to_sim_joints(dict(zip(tnames, traj["state_0"][0].tolist())),
                                      load_calibration(os.path.expanduser("~/Stanley_ws/lerobot_openarm/calibration.json")))  # the real arms' first pose
    env.reset()
    pos = robot.data.joint_pos.clone()
    for n, v in home.items():
        for nn in (n, n.replace("finger_joint1", "finger_joint2")):
            if nn in names:
                pos[:, names.index(nn)] = float(v)
    robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
    robot.set_joint_position_target(pos)
    hold = torch.zeros(N, offset, device=dev)
    for n in REPORTED:
        hold[:, slots[n]] = float(home[n])
    if args_cli.place:
        import math as _m

        for part, asset in zip(args_cli.place.split(":"), (charger, battery)):
            x_, y_, yaw_ = [float(v) for v in part.split(",")]
            root = asset.data.default_root_state.clone()
            half = _m.radians(yaw_) / 2.0
            root[:, :2] = env.scene.env_origins[:, :2] + torch.tensor([x_, y_], device=dev)
            root[:, 3:7] = torch.tensor([_m.cos(half), 0.0, 0.0, _m.sin(half)], device=dev)
            root[:, 7:] = 0.0
            asset.write_root_state_to_sim(root)
    if args_cli.scatter:
        import math as _m

        rng_ = np.random.default_rng(args_cli.seed)
        ranges = {"charger": ((0.300, 0.355), (0.070, 0.100), (-14.0, 6.0)), "battery": ((0.260, 0.390), (-0.115, -0.075), (-14.0, 5.0))}
        for key, asset in (("charger", charger), ("battery", battery)):
            root = asset.data.default_root_state.clone()
            for i_ in range(N):
                x_, y_, yaw_ = [rng_.uniform(*r) for r in ranges[key]]
                half = _m.radians(yaw_) / 2.0
                root[i_, :2] = env.scene.env_origins[i_, :2] + torch.tensor([x_, y_], device=dev)
                root[i_, 3:7] = torch.tensor([_m.cos(half), 0.0, 0.0, _m.sin(half)], device=dev)
            root[:, 7:] = 0.0
            asset.write_root_state_to_sim(root)
    for _ in range(25):  # settle the props on the mat
        env.step(hold)
    for flag, asset in ((args_cli.hide_battery, battery), (args_cli.hide_charger, charger)):
        if flag:
            keep = asset.data.root_state_w.clone()
            keep[:, 0] += 3.0
            keep[:, 7:] = 0.0
            asset.write_root_state_to_sim(keep)
    if args_cli.goto:
        side, qs = args_cli.goto.split(":")
        tgt = np.array([float(v) for v in qs.split(",")])
        jn = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        q0 = np.array([home[n] for n in jn])
        for k in range(200):
            q = q0 + (tgt - q0) * min(1.0, k / 90)
            for n, v in zip(jn, q):
                hold[:, slots[n]] = float(v)
            env.step(hold)
        meas = robot.data.joint_pos[0, [names.index(n) for n in jn]].cpu().numpy()
        print("  goto target", np.round(tgt, 3), "measured", np.round(meas, 3), flush=True)
        env.close()
        return
    if args_cli.sweep:
        jn, lo, hi = args_cli.sweep.split(",")
        lo, hi = float(lo), float(hi)
        for k in range(400):
            tgt = lo + (hi - lo) * min(1.0, k / 300)
            hold[:, slots[jn]] = tgt
            env.step(hold)
            if k % 25 == 0:
                print(f"  sweep {jn} cmd {tgt:+.3f} meas {float(robot.data.joint_pos[0, names.index(jn)]):+.3f}", flush=True)
        env.close()
        return
    experts = [BatteryExpert(**overrides) for _ in range(N)]
    status = ["running"] * N
    stage_of = [""] * N
    ok = np.zeros(N, bool)
    first_ok = [None] * N
    zmax = {k: np.full(N, -1.0) for k in ("charger", "battery")}
    z0 = {k: np.array([observe(i)["objects"][k][2] for i in range(N)]) for k in zmax}
    obs0 = [observe(i) for i in range(N)]
    for i in range(N):
        experts[i].reset(obs0[i])
    lim = robot.data.joint_pos_limits[0].cpu().numpy()
    print("[run_battery_expert] sim joint limits", {n: (round(float(lim[names.index(n), 0]), 3), round(float(lim[names.index(n), 1]), 3)) for n in REPORTED}, flush=True)
    print("[run_battery_expert] robot base", np.round(obs0[0]["robot_base"], 4), "joints", {k: round(v, 3) for k, v in obs0[0]["joints"].items()}, flush=True)
    for i in range(N):
        c_, b_ = obs0[i]["objects"]["charger"], obs0[i]["objects"]["battery"]
        from battery_expert import yaw_of

        print(f"  [start] env {i}: charger ({c_[0]:.3f}, {c_[1]:.3f}, {np.degrees(yaw_of(c_[3:])):.1f} deg) battery ({b_[0]:.3f}, {b_[1]:.3f}, {np.degrees(yaw_of(b_[3:])):.1f} deg)", flush=True)
    action = torch.zeros(N, offset, device=dev)
    for i in range(N):
        for n, v in obs0[i]["joints"].items():
            action[i, slots[n]] = v
    t0 = time.perf_counter()
    trace = [[] for _ in range(N)]
    for step in range(args_cli.max_steps):
        for i in range(N):
            if status[i] != "running":
                continue
            ob = observe(i)
            try:
                joints, done = experts[i].step(ob)
            except Failed as e:
                status[i] = f"FAILED {e}"
                continue
            stage_of[i] = experts[i].cur["label"] if experts[i].cur else ""
            for n, v in joints.items():
                action[i, slots[n]] = v
            trace[i].append([joints[n] for n in REPORTED])
            if i == args_cli.debug_env and step % args_cli.debug_every == 0:
                Tb = base_transform(ob["robot_base"])
                parts = []
                for side, s_ in (("left", experts[i].p["l_s"]), ("right", experts[i].p["r_s"])):
                    arm = experts[i].arms[side]
                    q = np.array([ob["joints"][n] for n in arm.names])
                    T = Tb @ arm.fk(q)
                    g = T[:3, 3] + T[:3, 2] * s_
                    qc = np.array([joints[n] for n in arm.names])
                    gc = (Tb @ arm.fk(qc))[:3, 3] + (Tb @ arm.fk(qc))[:3, 2] * s_
                    parts.append(f"{side[0]} grasp ({g[0]:.3f},{g[1]:.3f},{g[2]:.3f}) cmd ({gc[0]:.3f},{gc[1]:.3f},{gc[2]:.3f}) fing {ob['joints'][f'openarm_{side}_finger_joint1']:.4f}")
                if os.environ.get("DEBUG_Q"):
                    tq = robot.data.applied_torque[i, [names.index(n) for n in experts[i].arms[os.environ.get("DEBUG_SIDE", "left")].names]].cpu().numpy()
                    print("    torque", np.round(tq, 1), "limits", np.round(robot.data.joint_effort_limits[i, [names.index(n) for n in experts[i].arms[os.environ.get("DEBUG_SIDE", "left")].names]].cpu().numpy(), 1))
                    print("    q_meas", np.round([ob["joints"][n] for n in experts[i].arms[os.environ.get("DEBUG_SIDE", "left")].names], 3), "q_cmd", np.round([joints[n] for n in experts[i].arms[os.environ.get("DEBUG_SIDE", "left")].names], 3), flush=True)
                c, b = ob["objects"]["charger"], ob["objects"]["battery"]
                print(f"  [{step:4d}] {stage_of[i]:8s} {' | '.join(parts)} | charger ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) battery ({b[0]:.3f},{b[1]:.3f},{b[2]:.3f})", flush=True)
            if done:
                status[i] = "done"
        if step == args_cli.hide_charger_at:
            keep = charger.data.root_state_w.clone()
            keep[:, 0] += 3.0
            keep[:, 7:] = 0.0
            charger.write_root_state_to_sim(keep)
        env.step(action)
        seated = battery_in_charger(env).cpu().numpy()
        for k, asset in (("charger", charger), ("battery", battery)):
            zmax[k] = np.maximum(zmax[k], (asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).cpu().numpy())
        for i in range(N):
            if seated[i] and first_ok[i] is None:
                first_ok[i] = step
        ok = seated
        if all(s != "running" for s in status):
            break
    print(f"[run_battery_expert] {step + 1} steps in {time.perf_counter() - t0:.0f} s", flush=True)
    for i in range(N):
        c = env.scene["charger"].data.root_pos_w[i] - env.scene.env_origins[i]
        b = env.scene["battery"].data.root_pos_w[i] - env.scene.env_origins[i]
        print(f"env {i:2d}: {status[i][:150]:150s} stage {stage_of[i]:8s} in bay NOW {bool(ok[i])} (first at step {first_ok[i]}); "
              f"lift charger {100 * (zmax['charger'][i] - z0['charger'][i]):5.1f} cm battery {100 * (zmax['battery'][i] - z0['battery'][i]):5.1f} cm; "
              f"battery-charger ({100 * (b[0] - c[0]):+5.1f}, {100 * (b[1] - c[1]):+5.1f}, {100 * (b[2] - c[2]):+5.1f}) cm", flush=True)
    print(f"[run_battery_expert] SUCCESS (battery in the bay at the end): {int(ok.sum())}/{N}; ever seated: {sum(f is not None for f in first_ok)}/{N}", flush=True)
    os.makedirs(args_cli.out, exist_ok=True)
    np.savez(os.path.join(args_cli.out, "trace.npz"), **{f"joints_{i}": np.array(trace[i]) for i in range(N)})
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
