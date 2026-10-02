# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Fit the SIM arm's actuator parameters so it responds to a command the way the REAL arm did.

Input is the output of analyze_real_dataset.py (or of lerobot_openarm/sysid_probe_real.py): per
episode, the command the real motors were sent (``cmd``) and the joint positions they produced
(``meas``). Each real episode is replayed open-loop in the sim as joint-position targets, and the
sim's joint trajectory is scored against the real one.

HOW. N copies of the robot run the SAME real episode at once, each with different stiffness / damping
/ armature, written per environment at run time. A cross-entropy search then keeps the values that
track the real arm best. The joints are separable to a good approximation, so every joint has its own
search distribution and is scored on its own tracking error -- one population fits all joints at once.
Left and right arms are the same hardware, so joint n of both arms shares one parameter set (use
--no-share-arms to fit them separately). The two fingers of a gripper are mimicked and share too.

THE REAL ARM'S GRAVITY FEED-FORWARD. OpenArmFollower adds a model-based gravity torque to every command,
so at its low gains (kp 20-60) it holds position nearly exactly. The task's robot cfg already has gravity
disabled on the robot links (openarm.py: OPENARM_BI_HIGH_PD_CFG ... disable_gravity = True), the sim
equivalent, and that is the default here. --gravity on shows what a sim WITHOUT that compensation would do
(it sags by tau_gravity / kp) -- useful only to confirm the setting matters.

GRIPPERS. A real gripper stalls on the can; this replay has no can between the fingers, so the command
is floored at the measured hold width (``w_hold_m`` from real_arm_response.json) -- otherwise the sim
fingers would close fully and the error would be about geometry, not about the actuator.

WHAT IT CAN IDENTIFY. Only what the input excites. Commands from a dataset recorded under the bridge's
speed cap move slowly (<= ~0.4 rad/s), so stiffness and damping are only loosely pinned down and the
search is regularised toward the current values. The tool reports a baseline-vs-fitted error on
held-out episodes -- trust a parameter only if it improves THAT. Add a probe recording for the rest.

Usage:
  ./isaaclab.sh -p scripts/tools/sysid/fit_sim_actuators.py --headless \\
      --traces logs/sysid/openarm_pringles_real_v00/traces.npz \\
      --response logs/sysid/openarm_pringles_real_v00/real_arm_response.json
  (writes sim_actuator_profile.json next to the traces; apply it with OPENARM_ACTUATOR_PROFILE=<path>)

Report goes to --report as well as the console (Kit can swallow stdout in headless runs).
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--traces", type=str, required=True, help="traces.npz from analyze_real_dataset.py / the probe.")
parser.add_argument("--response", type=str, default=None, help="real_arm_response.json (for the gripper hold width).")
parser.add_argument("--out", type=str, default=None, help="Profile path. Default: sim_actuator_profile.json beside --traces.")
parser.add_argument("--report", type=str, default=None, help="Report file. Default: fit_report.txt beside --traces.")
parser.add_argument("--num-envs", type=int, default=64, help="Candidate parameter sets evaluated per rollout.")
parser.add_argument("--generations", type=int, default=8, help="Search generations.")
parser.add_argument("--train-episodes", type=int, default=2, help="Episodes used to fit (cycled, one per generation).")
parser.add_argument("--test-episodes", type=int, default=2, help="Held-out episodes for the baseline-vs-fitted comparison.")
parser.add_argument("--eval-list", type=int, nargs="+", default=None,
                    help="Score exactly these segments (baseline-only, one line each) instead of the held-out split.")
parser.add_argument("--max-seconds", type=float, default=10.0, help="Replay at most this much of each episode.")
parser.add_argument("--gravity", choices=("on", "off"), default="off", help="Robot gravity during the replay; default off = what the task cfg does (see docstring).")
parser.add_argument("--no-share-arms", action="store_true", help="Fit left and right arm joints separately.")
parser.add_argument("--elite-frac", type=float, default=0.2)
parser.add_argument("--reg", type=float, default=0.0005, help="Per-unit-of-log-scale penalty pulling parameters to the baseline.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--min-improvement", type=float, default=0.10,
                    help="A fitted group replaces the baseline only if its HELD-OUT rmse is at least this fraction lower. "
                         "Below that the difference is noise the data cannot distinguish (default 10%%).")
parser.add_argument("--baseline-only", action="store_true", help="Only score the current sim parameters; no search.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
app = AppLauncher(args).app

import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import _sim_common  # noqa: E402

TRACES = os.path.expanduser(args.traces)
OUT_DIR = os.path.dirname(os.path.abspath(TRACES))
REPORT = open(os.path.expanduser(args.report or os.path.join(OUT_DIR, "fit_report.txt")), "w")


def say(msg=""):
    print(msg, flush=True)
    REPORT.write(msg + "\n")
    REPORT.flush()


# parameter groups: (n=1..7 arm joint, 8 gripper). With shared arms a group spans left+right.
GROUPS = list(range(1, 9))
SIDES = ("left", "right")
LOG_BOUNDS = {"kp": (np.log(0.1), np.log(10.0)), "kd": (np.log(0.05), np.log(20.0))}
ARMATURE_BOUNDS = (0.0, 1.0)


def joint_names_for(n: int, side: str) -> list[str]:
    if n == 8:
        return [f"openarm_{side}_finger_joint1", f"openarm_{side}_finger_joint2"]
    return [f"openarm_{side}_joint{n}"]


def load_traces():
    z = np.load(TRACES, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    eps = sorted({k.split("_")[0] for k in z.files if k.startswith("ep")})
    data = [(z[f"{e}_cmd"], z[f"{e}_meas"]) for e in eps]
    return names, float(z["dt"]), data


def main():
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    names, dt_trace, episodes = load_traces()
    w_hold = {"LJ8.pos": 0.0, "RJ8.pos": 0.0}
    if args.response:
        resp = json.load(open(os.path.expanduser(args.response)))
        for k, v in resp.get("grippers", {}).items():
            w_hold[k] = float(v["w_hold_m"])
    say(f"[DATA] {len(episodes)} episodes in {TRACES}, dt {dt_trace * 1000:.1f} ms, gripper hold widths {w_hold}")

    P = args.num_envs
    env, cfg = _sim_common.build_env(P, args.device, disable_robot_gravity=(args.gravity == "off"))
    robot = env.scene["robot"]
    physics_dt = env.sim.get_physics_dt()
    decimation = env.cfg.decimation
    step_dt = physics_dt * decimation
    if abs(step_dt - dt_trace) > 1e-4:
        raise SystemExit(f"[FATAL] env steps at {1 / step_dt:.2f} Hz but the traces are {1 / dt_trace:.2f} Hz -- "
                         f"set OPENARM_CONTROL_HZ={round(1 / dt_trace)} so the replay timing matches the recording.")
    say(f"[SIM] {P} envs, physics {1 / physics_dt:.0f} Hz x {decimation} = control {1 / step_dt:.1f} Hz, gravity {args.gravity}")

    env.sim.reset()
    env.reset()

    # joint id bookkeeping -------------------------------------------------------------------------
    ds_index = {n: i for i, n in enumerate(names)}          # 'LJ3.pos' -> column in cmd/meas
    all_ids, ds_cols, group_of = [], [], []                 # per driven sim joint
    for side, pre in (("left", "L"), ("right", "R")):
        for n in GROUPS:
            for jn in joint_names_for(n, side):
                ids, _ = robot.find_joints(jn)
                all_ids.append(ids[0])
                ds_cols.append(ds_index[f"{pre}J{n}.pos"])
                group_of.append(n)
    all_ids_t = torch.tensor(all_ids, device=env.device)
    side_of = [s for s in SIDES for n in GROUPS for _ in joint_names_for(n, s)]
    J = len(all_ids)
    # reading column per (side, n): the first (primary) joint of each
    primary = [i for i in range(J) if not (group_of[i] == 8 and all_ids[i] != min(
        all_ids[k] for k in range(J) if group_of[k] == 8 and side_of[k] == side_of[i]))]
    primary_cols = [ds_cols[i] for i in primary]

    base = {
        "kp": robot.data.joint_stiffness[0, all_ids_t].clone(),
        "kd": robot.data.joint_damping[0, all_ids_t].clone(),
        "arm": robot.data.joint_armature[0, all_ids_t].clone(),
    }
    say("[SIM] baseline actuator parameters (env 0):")
    for i in primary:
        say(f"   {side_of[i]:5s} {robot.joint_names[all_ids[i]]:28s} kp {base['kp'][i].item():8.2f}  kd {base['kd'][i].item():7.3f}  armature {base['arm'][i].item():.4f}")

    cube = env.scene.rigid_objects.get("cube_2") if hasattr(env.scene, "rigid_objects") else None

    # mapping from search variables to per-sim-joint tensors --------------------------------------
    # x: (P, G, 3) with G = len(GROUPS) shared groups [or 2*len(GROUPS) when arms are fitted apart]
    share = not args.no_share_arms
    G = len(GROUPS) if share else 2 * len(GROUPS)

    def group_index(i):
        n = group_of[i] - 1
        return n if share else n + (0 if side_of[i] == "left" else len(GROUPS))

    gidx = torch.tensor([group_index(i) for i in range(J)], device=env.device)

    def apply_params(x: torch.Tensor):
        """x: (P, G, 3) = (log kp scale, log kd scale, armature) -> written to the sim for every env."""
        xg = x[:, gidx, :]                                     # (P, J, 3)
        kp = base["kp"][None] * torch.exp(xg[..., 0])
        kd = base["kd"][None] * torch.exp(xg[..., 1])
        arm = xg[..., 2]
        robot.write_joint_stiffness_to_sim(kp, joint_ids=all_ids)
        robot.write_joint_damping_to_sim(kd, joint_ids=all_ids)
        robot.write_joint_armature_to_sim(arm, joint_ids=all_ids)

    # one rollout: replay a real episode in every env ---------------------------------------------
    def rollout(ep: int, x: torch.Tensor):
        cmd, meas = episodes[ep]
        T = min(len(cmd), int(args.max_seconds / step_dt))
        cmd_t = torch.tensor(cmd[:T], dtype=torch.float32, device=env.device)
        # grippers: floor the command at the measured hold width (no can between the fingers here)
        for col_name, hold in w_hold.items():
            c = ds_index[col_name]
            cmd_t[:, c] = torch.clamp(cmd_t[:, c], min=hold)
        env.reset()
        apply_params(x)
        # park the cube out of the way, start every joint where the real arm started
        if cube is not None:
            pose = cube.data.root_state_w[:, :7].clone()
            pose[:, 0] = env.scene.env_origins[:, 0] + 3.0
            pose[:, 2] = 0.2
            cube.write_root_pose_to_sim(pose)
        s0 = torch.tensor(meas[0], dtype=torch.float32, device=env.device)
        q0 = robot.data.joint_pos.clone()
        q0[:, all_ids_t] = s0[ds_cols][None]
        robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        robot.reset()
        tgt0 = q0[:, all_ids_t]
        robot.set_joint_position_target(tgt0, joint_ids=all_ids)
        for _ in range(8 * decimation):                        # settle
            robot.write_data_to_sim()
            env.sim.step(render=False)
            robot.update(physics_dt)
        out = torch.empty((T, P, J), device=env.device)
        for t in range(T):
            out[t] = robot.data.joint_pos[:, all_ids_t]       # state BEFORE this step's command, like the dataset
            robot.set_joint_position_target(cmd_t[t, ds_cols][None].expand(P, -1), joint_ids=all_ids)
            for _ in range(decimation):
                robot.write_data_to_sim()
                env.sim.step(render=False)
                robot.update(physics_dt)
        real = torch.tensor(meas[:T], dtype=torch.float32, device=env.device)[:, ds_cols]   # (T, J)
        err = (out - real[:, None, :]) ** 2                    # (T, P, J)
        err = err[1:]                                          # frame 0 is the (settled) start pose
        rmse_j = err.mean(0).sqrt()                            # (P, J)
        # collapse joints -> groups (mean of the MSE over the joints in a group)
        mse_j = err.mean(0)
        mse_g = torch.zeros((P, G), device=env.device)
        cnt = torch.zeros(G, device=env.device)
        for i in range(J):
            mse_g[:, gidx[i]] += mse_j[:, i]
            cnt[gidx[i]] += 1
        return (mse_g / cnt).sqrt(), rmse_j                    # (P, G), (P, J)

    def x_from(means: np.ndarray) -> torch.Tensor:
        return torch.tensor(means, dtype=torch.float32, device=env.device)[None].expand(P, -1, -1).clone()

    def default_means():
        m = np.zeros((G, 3))
        for g in range(G):
            members = [i for i in range(J) if group_index(i) == g]
            m[g, 2] = float(base["arm"][members[0]])
        return m

    train_eps = list(range(0, len(episodes), 2))[: args.train_episodes]
    test_eps = list(range(1, len(episodes), 2))[: args.test_episodes]
    if not test_eps:   # a single segment: nothing is held out, so the verdict is optimistic -- say so
        test_eps = train_eps[:1]
        say("[WARN] only one trace segment: it is used for BOTH fitting and testing, so 'improved' is optimistic. "
            "Record at least 4 segments (e.g. several speed caps / episodes).")
    group_label = [f"J{n}" if n < 8 else "grip" for n in GROUPS] if share else \
        [f"L{n}" if n < 8 else "Lgrip" for n in GROUPS] + [f"R{n}" if n < 8 else "Rgrip" for n in GROUPS]

    def score(means: np.ndarray, eps: list[int]) -> np.ndarray:
        """Mean rmse per group over episodes for ONE parameter set (all envs identical)."""
        acc = []
        for e in eps:
            r, _ = rollout(e, x_from(means))
            acc.append(r[0].cpu().numpy())
        return np.mean(acc, axis=0)

    t_start = time.time()
    base_means = default_means()
    if args.eval_list is not None:
        say("\n[PER-SEGMENT] current sim parameters, one line per trace segment (rmse rad; gripper m)")
        say("   seg   " + "  ".join(f"{l:>6s}" for l in group_label))
        for e in args.eval_list:
            v = score(base_means, [e])
            say(f"   {e:3d}   " + "  ".join(f"{x:6.4f}" for x in v))
        env.close()
        return
    base_test = score(base_means, test_eps)
    say(f"\n[BASELINE] current sim parameters, held-out episodes {test_eps}  ({time.time() - t_start:.0f} s)")
    say("   rmse (rad; gripper in m):  " + "  ".join(f"{l}:{v:.4f}" for l, v in zip(group_label, base_test)))
    if args.baseline_only:
        env.close()
        return

    # cross-entropy search ----------------------------------------------------------------------------
    mu = base_means.copy()
    sigma = np.tile(np.array([0.6, 0.6, 0.05]), (G, 1))
    n_elite = max(3, int(args.elite_frac * P))
    best_mu, best_cost = mu.copy(), np.full(G, np.inf)
    lo = np.array([LOG_BOUNDS["kp"][0], LOG_BOUNDS["kd"][0], ARMATURE_BOUNDS[0]])
    hi = np.array([LOG_BOUNDS["kp"][1], LOG_BOUNDS["kd"][1], ARMATURE_BOUNDS[1]])
    for gen in range(args.generations):
        ep = train_eps[gen % len(train_eps)]
        samples = rng.normal(mu[None], sigma[None], size=(P, G, 3))
        samples = np.clip(samples, lo, hi)
        samples[0] = mu                                           # the incumbent, measured on the same episode
        r, _ = rollout(ep, torch.tensor(samples, dtype=torch.float32, device=env.device))
        r = r.cpu().numpy()                                       # (P, G)
        reg = args.reg * (np.abs(samples[..., 0]) + np.abs(samples[..., 1]))
        cost = r + reg
        for g in range(G):
            order = np.argsort(cost[:, g])
            elite = samples[order[:n_elite], g]
            mu[g] = 0.5 * mu[g] + 0.5 * elite.mean(0)
            sigma[g] = np.maximum(0.6 * sigma[g] + 0.4 * elite.std(0), np.array([0.03, 0.03, 0.002]))
            if cost[order[0], g] < best_cost[g]:
                best_cost[g], best_mu[g] = cost[order[0], g], samples[order[0], g]
        say(f"[GEN {gen + 1}/{args.generations}] ep {ep}  best rmse " + "  ".join(f"{l}:{r[:, g].min():.4f}" for g, l in enumerate(group_label))
            + f"   ({time.time() - t_start:.0f} s)")

    fitted = np.clip(0.5 * mu + 0.5 * best_mu, lo, hi)            # blend the distribution mean with the best sample
    fit_test = score(fitted, test_eps)
    say(f"\n[RESULT] held-out episodes {test_eps}: baseline -> fitted rmse")
    profile = {"control_hz": round(1 / step_dt), "traces": TRACES, "gravity_in_fit": args.gravity,
               "shared_arms": share, "train_episodes": train_eps, "test_episodes": test_eps, "groups": {}}
    for g, label in enumerate(group_label):
        members = [i for i in range(J) if group_index(i) == g]
        kp = float(base["kp"][members[0]]) * float(np.exp(fitted[g, 0]))
        kd = float(base["kd"][members[0]]) * float(np.exp(fitted[g, 1]))
        better = fit_test[g] < base_test[g] * (1.0 - args.min_improvement)
        say(f"   {label:6s} {base_test[g]:.4f} -> {fit_test[g]:.4f} ({'better' if better else 'no real gain -> baseline kept'})"
            f"   kp {float(base['kp'][members[0]]):7.2f} -> {kp:7.2f}   kd {float(base['kd'][members[0]]):6.2f} -> {kd:6.2f}   armature {fitted[g, 2]:.3f}")
        use = fitted[g] if better else base_means[g]
        profile["groups"][label] = {
            "joint_names": sorted({robot.joint_names[all_ids[i]] for i in members}),
            "stiffness": float(base["kp"][members[0]]) * float(np.exp(use[0])),
            "damping": float(base["kd"][members[0]]) * float(np.exp(use[1])),
            "armature": float(use[2]),
            "rmse_baseline": float(base_test[g]), "rmse_fitted_heldout": float(fit_test[g]), "improved": bool(better),
        }
    n_better = sum(g["improved"] for g in profile["groups"].values())
    say(f"\n[VERDICT] {n_better}/{len(group_label)} groups improved by >= {args.min_improvement:.0%} on held-out episodes.")
    if n_better == 0:
        say("   The current sim parameters already track the real arm as well as this data can tell. The profile below"
            " therefore equals the baseline -- the data does not exercise the arm hard enough to distinguish"
            " parameters (see the docstring). Record faster / larger-amplitude motion and refit.")
    out = os.path.expanduser(args.out or os.path.join(OUT_DIR, "sim_actuator_profile.json"))
    json.dump(profile, open(out, "w"), indent=2)
    say(f"\n[OUT] {out}   (apply with OPENARM_ACTUATOR_PROFILE={out})")
    env.close()


try:
    main()
except SystemExit as e:
    say(str(e))
except Exception:
    say(traceback.format_exc())
finally:
    REPORT.close()
    app.close()
