# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Vision-only transfer probe: locate the can from pre-grasp frames, trained on a sim look, tested on real frames.

probe_transfer.py predicts the next move from camera features AND joint state, and the state alone already explains
most of it (R^2 0.84 with no images), so it mostly measures how badly sim features disturb a head. Here the target
only a camera can supply: the can's start position (x, y). The sim value is the position replay_render_real.py put the
can at; the real value is the right jaw's position at its first closure (forward kinematics), the same quantity the
sim can was placed from, so labels agree across domains. Inputs are the frozen DINOv2 features of the three cameras for
frames before the right gripper first closes (the can has not been touched yet), no joint state. Heads are trained on
a sim look (or on real frames, the ceiling) and scored on held-out real episodes: mean position error in cm, against
the error of always guessing the mean position.

Uses the feature caches written by probe_transfer.py. Plain python in lerobot_openarm's venv.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_transfer as P  # noqa: E402
from real_states import real_state_to_sim  # noqa: E402


def pre_grasp_frames(traj, names, calib, ep, margin=15):
    S = traj[f"state_{ep}"]
    gr = calib["right"]["gripper"]
    frac = np.clip((S[:, names.index("RJ8.pos")] - gr["closed_raw"]) / (gr["open_raw"] - gr["closed_raw"]), 0, 1)
    op = np.flatnonzero(frac > 0.85)
    k = next((i for i in range(op[0], len(frac)) if frac[i] < 0.6), None) if len(op) else None
    return None if k is None else np.arange(0, max(k - margin, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--looks", default="legacy,plain,look")
    ap.add_argument("--train", default="0-39")
    ap.add_argument("--test", default="40-49")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cams", default="body_cam,wrist_cam,right_wrist_cam", help="Which cameras' features to use.")
    ap.add_argument("--calibration", default=os.path.expanduser("~/Stanley_ws/lerobot_openarm/calibration.json"))
    args = ap.parse_args()

    span = lambda s: list(range(int(s.split("-")[0]), int(s.split("-")[1]) + 1))  # noqa: E731
    tr, te = span(args.train), span(args.test)
    traj = np.load(os.path.join(args.replay, "trajectories.npz"))
    names = [str(n) for n in traj["names"]]
    calib = json.load(open(args.calibration))
    starts = json.load(open(os.path.join(args.replay, "can_starts_use.json")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    frames = {e: pre_grasp_frames(traj, names, calib, e) for e in tr + te}

    order = ["body_cam", "wrist_cam", "right_wrist_cam"]
    cols = np.concatenate([np.arange(order.index(c) * 1536, (order.index(c) + 1) * 1536) for c in args.cams.split(",")])

    def load(path):
        z = np.load(path)
        return {int(k): z[k][:, cols] for k in z.files}

    def xy(feat, eps):
        X, Y = [], []
        for e in eps:
            idx = frames[e]
            if idx is None or str(e) not in starts:
                continue
            idx = idx[idx < len(feat[e])]
            X.append(feat[e][idx])
            Y.append(np.tile(np.array(starts[str(e)]) * 100.0, (len(idx), 1)))  # cm
        return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)

    sets = {"real": load(os.path.join(args.replay, "real_dino_features.npz"))}
    for look in args.looks.split(","):
        sets[look] = load(os.path.join(args.frames, look, "dino_features.npz"))
    X_te, Y_te = xy(sets["real"], te)
    base = np.abs(Y_te - xy(sets["real"], tr)[1].mean(0)).mean()
    print(f"test: {len(X_te)} real pre-grasp frames from {len(te)} episodes; guessing the mean position errs by {base:.1f} cm")
    for name, feat in sets.items():
        X, Y = xy(feat, tr)
        mu, sd, ym, ys = X.mean(0), X.std(0) + 1e-6, Y.mean(0), Y.std(0)
        errs = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            net = P.fit_head((X - mu) / sd, (Y - ym) / ys, dev, epochs=150)
            with torch.no_grad():
                pred = net(torch.tensor((X_te - mu) / sd, device=dev)).cpu().numpy() * ys + ym
            errs.append(np.abs(pred - Y_te).mean())
        print(f"trained on {name:7s}: position error on held-out REAL frames {np.mean(errs):5.2f} cm +- {np.std(errs):.2f}", flush=True)


if __name__ == "__main__":
    main()
