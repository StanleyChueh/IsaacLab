#!/usr/bin/env python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Identify how the REAL OpenArm follower responds to commands, from a real-robot LeRobot dataset.

Plain python (numpy / scipy / pandas / matplotlib) -- no Isaac Sim needed, so it also runs in the
lerobot venv. Feed its output to fit_sim_actuators.py, which makes the SIM arm respond the same way.

WHAT THE DATASET CONTAINS. A dataset recorded with mirror_bridge.py + --real_arm_dataset (e.g.
ethanCSL/openarm_pringles_real_v00) holds, per frame, ``action`` = the sim arm's joint targets that
were mirrored to the real arm and ``observation.state`` = what the real arm then measured. That is
a paired command -> response recording of the real arm, 30 fps, for free.

THE MODEL (per arm joint), fitted by least squares over whole open-loop episodes:

    c[t]     = c[t-1] + clip(u[t] - c[t-1], +-cap*dt)     bridge speed cap on the COMMAND
    s[t+1]   = s[t]   + k * (c[t-d] + off - s[t])         motor response: delay d frames, 1st-order
                                                          lag k per frame, constant offset (gravity)

  u = recorded action, s = measured state, c = what the motors were actually commanded.
  mirror_bridge.py clamps every command to --max-joint-speed before it reaches the motors; that cap,
  not the motors, is what limits how fast the real arm moves in these recordings. The cap is ONE
  number for the whole bridge, so it is fitted per joint (to show it agrees) and then fixed to the
  median while the per-joint motor parameters are refitted.

GRIPPERS get their own model (the command is a 0 / 0.044 toggle, and the fingers stall on the can):
  s[t+1] = s[t] + k_dir * (max(u[t-d], w_hold) - s[t]), k_dir = k_close or k_open.

WHAT THIS CAN AND CANNOT TELL YOU. The recorded commands move at <= the bridge cap (~0.4 rad/s) and
are sampled at 30 fps, so this dataset identifies the real arm's SMALL-SIGNAL response and its
steady-state offsets well. It says nothing about behaviour at higher speed or torque (saturation,
overshoot, torque limits) -- the fit reports ``resolvable: false`` when a time constant is shorter
than ~1 frame, i.e. faster than 30 fps can see. For that, record a probe with
lerobot_openarm/sysid_probe_real.py and pass it to fit_sim_actuators.py as well.

Outputs (in --out-dir):
  real_arm_response.json   fitted parameters + fit quality (train / held-out episodes)
  traces.npz               per-episode reconstructed command + measured state, for the sim fit
  response_fit.png         measured vs model for the worst and best joints
  [compare.json]           with --compare-root: real vs sim state statistics (gripper hold width, ...)

Usage:
  python analyze_real_dataset.py --repo-id ethanCSL/openarm_pringles_real_v00
  python analyze_real_dataset.py --root ~/.cache/huggingface/lerobot/ethanCSL/openarm_pringles_real_v00 \\
      --compare-root /path/to/sim_lerobot_dataset
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

GRIPPER_IDX = (7, 15)
"""LJ8 / RJ8 -- the 16D layout is LJ1..LJ8 then RJ1..RJ8 (see deploy_smolvla_pickup_jointspace.py)."""
CAP_BOUNDS = (0.2, 6.0)
K_BOUNDS = (0.02, 1.0)
OFFSET_BOUND = 0.3
DELAYS = tuple(range(0, 7))


# ── data ─────────────────────────────────────────────────────────────────────────────────────────


def resolve_root(args) -> str:
    """Local dataset dir. --repo-id downloads meta/ + data/ of revision ``main`` (not the v3.0 tag --
    LeRobot's default revision -- which can point at an older upload; see the episode count printed)."""
    if args.root:
        return os.path.expanduser(args.root)
    from huggingface_hub import snapshot_download

    dest = os.path.expanduser(os.path.join("~/.cache/openarm_sysid", args.repo_id.replace("/", "--")))
    snapshot_download(
        args.repo_id, repo_type="dataset", revision=args.revision, local_dir=dest,
        allow_patterns=["meta/**", "data/**"],
    )
    return dest


def load_episodes(root: str):
    """-> (names, fps, [(action (T,16), state (T,16)) per episode], total_episodes_claimed)."""
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files])
    eps = [
        (np.stack(g.action.values).astype(np.float64), np.stack(g["observation.state"].values).astype(np.float64))
        for _, g in df.groupby("episode_index")
    ]
    return info["features"]["action"]["names"], float(info["fps"]), eps, int(info["total_episodes"])


def pad_batch(series: list[np.ndarray]):
    """Stack ragged (T_i,) arrays into (E, Tmax) + validity mask (padded with each episode's last value)."""
    tmax = max(len(x) for x in series)
    out = np.empty((len(series), tmax))
    mask = np.zeros((len(series), tmax), dtype=bool)
    for i, x in enumerate(series):
        out[i, : len(x)] = x
        out[i, len(x):] = x[-1]
        mask[i, : len(x)] = True
    return out, mask


# ── model ────────────────────────────────────────────────────────────────────────────────────────


def rate_limit(u: np.ndarray, s0: np.ndarray, cap: float, dt: float) -> np.ndarray:
    """Command after the bridge's speed cap. u: (E, T), s0: (E,) -> (E, T). Vectorised over episodes."""
    c = np.empty_like(u)
    c[:, 0] = s0
    step = cap * dt
    for t in range(1, u.shape[1]):
        c[:, t] = c[:, t - 1] + np.clip(u[:, t] - c[:, t - 1], -step, step)
    return c


def lag_response(c: np.ndarray, s0: np.ndarray, k: float, d: int, off: float) -> np.ndarray:
    """First-order tracking of the (delayed) command. c: (E, T) -> predicted state (E, T)."""
    s = np.empty_like(c)
    s[:, 0] = s0
    for t in range(1, c.shape[1]):
        s[:, t] = s[:, t - 1] + k * (c[:, max(t - 1 - d, 0)] + off - s[:, t - 1])
    return s


def tau_ms(k: float, dt: float) -> float:
    """Continuous-time constant equivalent to a per-frame lag coefficient k."""
    return 1000.0 * (-dt / np.log(max(1.0 - k, 1e-9)))


def rmse(pred, truth, mask) -> float:
    return float(np.sqrt(np.mean((pred - truth)[mask] ** 2)))


def fit_arm_joint(u, s, mask, dt, cap=None):
    """Fit (cap?, k, off) for each delay; return the best. cap=None fits it too."""
    best = None
    for d in DELAYS:
        if cap is None:
            def res(p):
                c = rate_limit(u, s[:, 0], p[0], dt)
                return (lag_response(c, s[:, 0], p[1], d, p[2]) - s)[mask]

            x0, lo, hi = [1.0, 0.4, 0.0], [CAP_BOUNDS[0], K_BOUNDS[0], -OFFSET_BOUND], [CAP_BOUNDS[1], K_BOUNDS[1], OFFSET_BOUND]
        else:
            c_fixed = rate_limit(u, s[:, 0], cap, dt)

            def res(p):
                return (lag_response(c_fixed, s[:, 0], p[0], d, p[1]) - s)[mask]

            x0, lo, hi = [0.4, 0.0], [K_BOUNDS[0], -OFFSET_BOUND], [K_BOUNDS[1], OFFSET_BOUND]
        r = least_squares(res, x0, bounds=(lo, hi))
        score = float(np.sqrt(np.mean(r.fun**2)))
        if best is None or score < best[0]:
            best = (score, d, r.x)
    score, d, x = best
    if cap is None:
        return dict(rmse=score, delay=d, cap=float(x[0]), k=float(x[1]), off=float(x[2]))
    return dict(rmse=score, delay=d, cap=cap, k=float(x[0]), off=float(x[1]))


def gripper_response(u, s0, k_close, k_open, w_hold, d):
    s = np.empty_like(u)
    s[:, 0] = s0
    mid = 0.5 * (u.max() + u.min())
    for t in range(1, u.shape[1]):
        cmd = u[:, max(t - 1 - d, 0)]
        closing = cmd < mid
        tgt = np.where(closing, np.maximum(cmd, w_hold), cmd)
        s[:, t] = s[:, t - 1] + np.where(closing, k_close, k_open) * (tgt - s[:, t - 1])
    return s


def fit_gripper(u, s, mask, dt):
    best = None
    for d in DELAYS:
        def res(p):
            return (gripper_response(u, s[:, 0], p[0], p[1], p[2], d) - s)[mask]

        r = least_squares(res, [0.4, 0.3, 0.02], bounds=([K_BOUNDS[0], K_BOUNDS[0], 0.0], [1.0, 1.0, 0.044]))
        score = float(np.sqrt(np.mean(r.fun**2)))
        if best is None or score < best[0]:
            best = (score, d, r.x)
    score, d, x = best
    return dict(rmse=score, delay=d, k_close=float(x[0]), k_open=float(x[1]), w_hold=float(x[2]))


# ── report ───────────────────────────────────────────────────────────────────────────────────────


def dataset_stats(eps, names):
    """Gripper open / holding-the-can widths, from the MEASURED state alone.

    State-only so a real dataset (binary 0/0.044 command) and a sim dataset (action = next measured
    state, never binary) are summarised by exactly the same rule: "holding" = the finger sits below
    0.038 and has not moved more than 0.5 mm over the last 10 frames.
    """
    out = {}
    for j in GRIPPER_IDX:
        holds, opens = [], []
        for _, s in eps:
            g = s[:, j]
            settled = np.zeros(len(g), dtype=bool)
            for t in range(10, len(g)):
                w = g[t - 10: t + 1]
                settled[t] = (w < 0.038).all() and (w.max() - w.min()) < 0.0005
            if settled.any():
                holds.append(float(np.median(g[settled])))
            opens.append(float(np.median(g[:20])))
        out[names[j]] = dict(
            hold_width_median=float(np.median(holds)) if holds else None,
            hold_width_p10_p90=[float(np.percentile(holds, 10)), float(np.percentile(holds, 90))] if holds else None,
            open_width_median=float(np.median(opens)) if opens else None,
            n_episodes_with_hold=len(holds),
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo-id", type=str, help="HF dataset repo id (downloads meta/ + data/ of --revision).")
    src.add_argument("--root", type=str, help="Local LeRobot dataset directory.")
    ap.add_argument("--revision", type=str, default="main", help="Hub revision for --repo-id (default main).")
    ap.add_argument("--out-dir", type=str, default=None, help="Default: logs/sysid/<dataset name>.")
    ap.add_argument("--compare-root", type=str, default=None, help="A SIM LeRobot dataset to compare state statistics against.")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    root = resolve_root(args)
    names, fps, eps, claimed = load_episodes(root)
    dt = 1.0 / fps
    name = (args.repo_id or os.path.basename(root.rstrip("/"))).replace("/", "--")
    out_dir = os.path.expanduser(args.out_dir or os.path.join("logs", "sysid", name))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[DATA] {root}: {len(eps)} episodes found ({claimed} in info.json), {sum(len(u) for u, _ in eps)} frames @ {fps:g} fps")
    if len(eps) != claimed:
        print("[WARN] episode count differs from info.json -- the local copy may be stale or partial.")

    # How often did the recorded command actually change? mirror_bridge records one frame per tick, but a new
    # command only exists when the sim finished a step. A sim that cannot keep up leaves repeated actions --
    # real_v00 updated at ~17 Hz inside a 30 fps recording, i.e. 4 of every 10 frames were stale copies.
    arm_cols = [j for j in range(16) if j not in GRIPPER_IDX]
    same = np.mean([(np.abs(np.diff(u[:, arm_cols], axis=0)).sum(1) < 1e-9).mean() for u, _ in eps])
    upd = fps * (1.0 - same)
    print(f"[DATA] command update rate ~{upd:.1f} Hz inside a {fps:g} fps recording ({same:.0%} of frames repeat the previous command)"
          + ("" if upd >= 0.9 * fps else "  <-- the sim step could not keep up with the recording rate"))

    arm_idx = [j for j in range(16) if j not in GRIPPER_IDX]
    train = list(range(0, len(eps), 2))
    test = list(range(1, len(eps), 2))
    U = {j: pad_batch([e[0][:, j] for e in eps]) for j in range(16)}
    S = {j: pad_batch([e[1][:, j] for e in eps]) for j in range(16)}

    def sub(j, idx):
        u, m = U[j]
        s, _ = S[j]
        return u[idx], s[idx], m[idx]

    # stage 1: free cap per joint -- shows the bridge cap is one number, not a joint property
    print("\n[FIT] stage 1: speed cap fitted independently per joint (train episodes)")
    free = {j: fit_arm_joint(*sub(j, train), dt) for j in arm_idx}
    caps = np.array([free[j]["cap"] for j in arm_idx])
    cap = float(np.median(caps))
    for j in arm_idx:
        print(f"   {names[j]:8s} cap {free[j]['cap']:.2f} rad/s   k {free[j]['k']:.2f}   delay {free[j]['delay']}f   rmse {free[j]['rmse']:.4f} rad")
    print(f"   -> cap spread across {len(arm_idx)} joints: {caps.min():.2f}..{caps.max():.2f}; using the median {cap:.3f} rad/s")

    # stage 2: cap fixed; per-joint motor response on train, scored on held-out episodes
    print(f"\n[FIT] stage 2: cap fixed at {cap:.3f} rad/s, motor response per joint (train) / held-out (test)")
    joints = {}
    for j in arm_idx:
        f = fit_arm_joint(*sub(j, train), dt, cap=cap)
        u_te, s_te, m_te = sub(j, test)
        pred = lag_response(rate_limit(u_te, s_te[:, 0], cap, dt), s_te[:, 0], f["k"], f["delay"], f["off"])
        u_all, s_all, m_all = sub(j, list(range(len(eps))))
        raw = rmse(u_all, s_all, m_all)
        joints[names[j]] = dict(
            delay_frames=int(f["delay"]), k_per_frame=f["k"], tau_ms=tau_ms(f["k"], dt), offset_rad=f["off"],
            rmse_train_rad=f["rmse"], rmse_heldout_rad=rmse(pred, s_te, m_te), rmse_raw_u_vs_s_rad=raw,
            resolvable=bool(f["k"] < 0.63),   # tau shorter than ~1 frame is below what 30 fps can resolve
        )
        j_ = joints[names[j]]
        print(f"   {names[j]:8s} tau {j_['tau_ms']:6.1f} ms (k {f['k']:.2f}{'' if j_['resolvable'] else ', >=1 frame: unresolved'})"
              f"  delay {f['delay']}f  offset {f['off']:+.4f}  rmse train {f['rmse']:.4f} / held-out {j_['rmse_heldout_rad']:.4f}  (raw u-s {raw:.3f})")

    print("\n[FIT] grippers")
    grippers = {}
    for j in GRIPPER_IDX:
        f = fit_gripper(*sub(j, train), dt)
        u_te, s_te, m_te = sub(j, test)
        pred = gripper_response(u_te, s_te[:, 0], f["k_close"], f["k_open"], f["w_hold"], f["delay"])
        grippers[names[j]] = dict(
            delay_frames=int(f["delay"]), k_close=f["k_close"], k_open=f["k_open"],
            tau_close_ms=tau_ms(f["k_close"], dt), tau_open_ms=tau_ms(f["k_open"], dt), w_hold_m=f["w_hold"],
            rmse_train_m=f["rmse"], rmse_heldout_m=rmse(pred, s_te, m_te),
        )
        g = grippers[names[j]]
        du = np.concatenate([np.diff(e[0][:, j]) for e in eps])
        g["n_close_events"], g["n_open_events"] = int((du < -0.02).sum()), int((du > 0.02).sum())
        if g["n_open_events"] < 5:   # e.g. the receiving hand never lets go of the can in these episodes
            g["k_open"], g["tau_open_ms"] = None, None
        open_txt = "open tau not identifiable (%d open events)" % g["n_open_events"] if g["tau_open_ms"] is None else f"open tau {g['tau_open_ms']:.0f} ms"
        print(f"   {names[j]:8s} close tau {g['tau_close_ms']:.0f} ms, {open_txt}, "
              f"holds the can at {g['w_hold_m']:.4f}, delay {g['delay_frames']}f, rmse {g['rmse_train_m']:.4f} / {g['rmse_heldout_m']:.4f} m")

    stats = dataset_stats(eps, names)
    result = dict(
        dataset=args.repo_id or root, revision=args.revision if args.repo_id else None, fps=fps, n_episodes=len(eps),
        speed_cap_rad_s=cap, speed_cap_per_joint_rad_s={names[j]: free[j]["cap"] for j in arm_idx},
        arm_joints=joints, grippers=grippers, state_stats=stats,
        notes=("Small-signal identification: commands in this dataset move at <= the bridge speed cap. "
               "Joints with resolvable=false respond faster than 30 fps can resolve. Large-signal behaviour "
               "(saturation, overshoot, torque limits) needs a probe recording."),
    )

    # traces for the sim fit: what the motors were actually commanded (u after the cap) and what they did
    traces = {"names": np.array(names), "dt": dt, "cap": cap}
    for i, (u, s) in enumerate(eps):
        c = u.copy()
        for j in arm_idx:
            c[:, j] = rate_limit(u[None, :, j], s[None, :1, j][:, 0], cap, dt)[0]
        traces[f"ep{i:03d}_cmd"], traces[f"ep{i:03d}_meas"] = c, s
    np.savez_compressed(os.path.join(out_dir, "traces.npz"), **traces)

    if args.compare_root:
        names_s, fps_s, eps_s, _ = load_episodes(os.path.expanduser(args.compare_root))
        sim_stats = dataset_stats(eps_s, names_s)
        result["compare"] = dict(root=args.compare_root, fps=fps_s, n_episodes=len(eps_s), state_stats=sim_stats)
        print(f"\n[COMPARE] real vs sim ({args.compare_root}, {len(eps_s)} episodes @ {fps_s:g} fps)")
        for g in stats:
            r, s_ = stats[g], sim_stats[g]
            if r["hold_width_median"] and s_["hold_width_median"]:
                print(f"   {g}: width while holding the can  real {r['hold_width_median']:.4f}   sim {s_['hold_width_median']:.4f}"
                      f"   (diff {s_['hold_width_median'] - r['hold_width_median']:+.4f} m)")
        sa = np.concatenate([e[1] for e in eps_s]); ra = np.concatenate([e[1] for e in eps])
        print("   state std (sim / real) per joint -- a large ratio means the sim visits a different range:")
        print("   " + "  ".join(f"{names[j]}:{sa[:, j].std() / max(ra[:, j].std(), 1e-6):.2f}" for j in arm_idx))

    json.dump(result, open(os.path.join(out_dir, "real_arm_response.json"), "w"), indent=2)
    print(f"\n[OUT] {out_dir}/real_arm_response.json, traces.npz")

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = sorted(arm_idx, key=lambda j: -joints[names[j]]["rmse_raw_u_vs_s_rad"])
        pick = [order[0], order[1], order[-1]]
        fig, axes = plt.subplots(len(pick), 1, figsize=(11, 3.2 * len(pick)), sharex=True)
        ep = test[0]
        for ax, j in zip(axes, pick):
            u, s = eps[ep][0][:, j], eps[ep][1][:, j]
            p = joints[names[j]]
            c = rate_limit(u[None], s[:1], cap, dt)[0]
            m = lag_response(c[None], s[:1], p["k_per_frame"], p["delay_frames"], p["offset_rad"])[0]
            t = np.arange(len(u)) * dt
            ax.plot(t, u, color="0.6", label="recorded action u (sim target)")
            ax.plot(t, c, "--", color="tab:orange", label=f"command after {cap:.2f} rad/s cap")
            ax.plot(t, s, color="tab:blue", label="measured (real)")
            ax.plot(t, m, ":", color="k", label="model")
            ax.set_ylabel(f"{names[j]} (rad)")
            ax.set_title(f"{names[j]}: held-out episode {ep}, rmse model {rmse(m[None], s[None], np.ones((1, len(s)), bool)):.3f} rad "
                         f"vs raw u-s {p['rmse_raw_u_vs_s_rad']:.3f} rad", fontsize=9)
        axes[0].legend(fontsize=8, ncol=2)
        axes[-1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "response_fit.png"), dpi=110)
        print(f"[OUT] {out_dir}/response_fit.png")


if __name__ == "__main__":
    main()
