# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Does training on a sim look help predict the real robot's actions? A controlled sim-to-real transfer probe.

replay_render_real.py re-renders real episodes in sim, so a sim frame and a real frame at the same index share the
joint state, the action and the scene layout. Here a small head is trained to predict the action from
(frozen DINOv2 features of the three cameras, joint state) on episodes 0..N-1 rendered in each sim look -- and, as a
ceiling, on the real frames -- and every head is scored on the real frames of the held-out episodes. What is
predicted is the action minus the current state (the move to make next, averaged over the next ``--horizon`` frames),
because the action itself is nearly the state; the score is R^2 of that move, per look. A look that closes the
vision gap should get a head trained on it close to the one trained on real frames.

Plain python in lerobot_openarm's venv (decodes real videos, DINOv2 via transformers):
  python scripts/tools/sysid/probe_transfer.py --replay logs/real2sim/replay --frames logs/real2sim/replay_frames \\
      --real ~/.cache/huggingface/lerobot/ethanCSL/openarm_pringles_lerobot_real_v00 --looks legacy,plain,look
"""

import argparse
import io
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_states import decode_episodes, load_episodes  # noqa: E402

CAMERAS = ("body_cam", "wrist_cam", "right_wrist_cam")
MODEL = "facebook/dinov2-base"


def load_model(device):
    from transformers import AutoImageProcessor, AutoModel

    return AutoImageProcessor.from_pretrained(MODEL), AutoModel.from_pretrained(MODEL).to(device).eval()


@torch.no_grad()
def features(images: np.ndarray, proc, model, device) -> np.ndarray:
    """(N,224,224,3) uint8 -> (N, 1536) CLS + mean-patch features, no resize / crop (frames are already 224 square)."""
    out = []
    for i in range(0, len(images), 64):
        batch = proc(images=list(images[i:i + 64]), return_tensors="pt", do_resize=False, do_center_crop=False).to(device)
        h = model(**batch).last_hidden_state
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def jpeg_roundtrip(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Real frame -> the same resize + JPEG treatment the sim frames got."""
    import cv2

    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(small).save(buf, format="JPEG", quality=90)
    return np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def real_features(root: str, episodes: list[int], proc, model, device, cache: str) -> dict:
    if os.path.isfile(cache):
        z = np.load(cache)
        return {int(k): z[k] for k in z.files}
    out = {}
    for cam in CAMERAS:
        eps, _, fps = load_episodes(root, cam)
        for e, frames in decode_episodes([x for x in eps if x["episode"] in episodes], fps, gray=False):
            imgs = np.stack([jpeg_roundtrip(f) for f in frames[:len(e["state"])]])
            out.setdefault(e["episode"], {})[cam] = features(imgs, proc, model, device)
            print(f"  real {cam} episode {e['episode']}: {len(imgs)} frames", flush=True)
    flat = {str(e): np.concatenate([out[e][c] for c in CAMERAS], 1) for e in out}
    np.savez(cache, **flat)
    return {int(k): v for k, v in flat.items()}


def sim_features(folder: str, episode: int, n: int, proc, model, device) -> np.ndarray:
    parts = []
    for cam in CAMERAS:
        imgs = np.stack([np.array(Image.open(os.path.join(folder, f"ep{episode}", f"{cam}_{k:04d}.jpg")).convert("RGB"))
                         for k in range(n)])
        parts.append(features(imgs, proc, model, device))
    return np.concatenate(parts, 1)


def targets(state: np.ndarray, action: np.ndarray, horizon: int) -> np.ndarray:
    """Mean over the next `horizon` actions minus the current state: the move to make."""
    n = len(state)
    idx = np.minimum(np.arange(n)[:, None] + np.arange(horizon)[None], n - 1)
    return action[idx].mean(1) - state


def fit_head(X: np.ndarray, Y: np.ndarray, device, epochs: int = 300, wd: float = 1e-2, hidden: int = 512) -> torch.nn.Module:
    X, Y = torch.tensor(X, device=device), torch.tensor(Y, device=device)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(0.2),
                              torch.nn.Linear(hidden, Y.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    g = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g).to(device)
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(X[b]), Y[b])
            loss.backward()
            opt.step()
    return net.eval()


def r2(net, X, Y, device) -> float:
    with torch.no_grad():
        P = net(torch.tensor(X, device=device)).cpu().numpy()
    return float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--frames", required=True, help="replay_render_real.py --out folder (holds one folder per look)")
    ap.add_argument("--real", required=True)
    ap.add_argument("--looks", default="legacy,plain,look")
    ap.add_argument("--train", default="0-39")
    ap.add_argument("--test", default="40-49")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    def span(s):
        lo, _, hi = s.partition("-")
        return list(range(int(lo), int(hi) + 1))

    train_eps, test_eps = span(args.train), span(args.test)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    traj = np.load(os.path.join(args.replay, "trajectories.npz"))
    proc, model = load_model(device)
    feats_real = real_features(args.real, train_eps + test_eps, proc, model, device,
                               os.path.join(args.replay, "real_dino_features.npz"))

    def xy(feat_by_ep, eps):
        X, Y = [], []
        for e in eps:
            st, ac = traj[f"state_{e}"], traj[f"action_{e}"]
            n = min(len(st), len(feat_by_ep[e]))
            X.append(np.concatenate([feat_by_ep[e][:n], st[:n]], 1))
            Y.append(targets(st, ac, args.horizon)[:n])
        return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)

    X_te, Y_te = xy(feats_real, test_eps)
    # standardise with the real test statistics of Y only through the R^2 definition; inputs z-scored per training set
    results = {}
    train_sets = {"real": xy(feats_real, train_eps)}
    for look in [l for l in args.looks.split(",") if l]:
        folder = os.path.join(args.frames, look)
        cache = os.path.join(args.frames, look, "dino_features.npz")
        if os.path.isfile(cache):
            z = np.load(cache)
            fb = {int(k): z[k] for k in z.files}
        else:
            fb = {}
            for e in train_eps + test_eps:
                n = len(traj[f"state_{e}"])
                if os.path.isdir(os.path.join(folder, f"ep{e}")):
                    fb[e] = sim_features(folder, e, n, proc, model, device)
            np.savez(cache, **{str(k): v for k, v in fb.items()})
        have = [e for e in train_eps if e in fb]
        print(f"{look}: {len(have)} / {len(train_eps)} training episodes rendered", flush=True)
        train_sets[look] = xy(fb, have)
        if look == "real":
            continue
    for name, (X, Y) in train_sets.items():
        mu, sd = X.mean(0), X.std(0) + 1e-6
        scores = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            net = fit_head((X - mu) / sd, Y, device)
            scores.append(r2(net, (X_te - mu) / sd, Y_te, device))
        results[name] = (float(np.mean(scores)), float(np.std(scores)))
        print(f"trained on {name:7s}: R^2 on held-out REAL episodes {np.mean(scores):+.3f} +- {np.std(scores):.3f}", flush=True)
    json.dump(results, open(os.path.join(args.frames, "probe_transfer.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
