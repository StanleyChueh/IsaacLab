# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""How far are sim camera frames from real ones, measured in a pretrained vision model's feature space?

For each camera, real frames (extract_real_frames.py) are compared with sim frames rendered at the same joint
states (render_real_states.py), for any number of sim variants (e.g. before / after a change):

* paired distance -- mean cosine distance between the DINOv2 embeddings of a real frame and the sim frame at its
  state (what a policy's frozen vision encoder would see change), and
* separability -- cross-validated ROC-AUC of a linear classifier trained to tell real from sim embeddings
  (1.0: trivially told apart; 0.5: indistinguishable to this probe). Chance level is 0.5 +- ~0.05 at ~120 frames.

It cannot say a policy will transfer; it says whether the vision gap a policy's encoder sees went down.

Needs transformers (lerobot_openarm's venv):
  python scripts/tools/sysid/eval_realism.py --camera body_cam --real logs/real2sim/body_cam_frames_v2/frames.npz \\
      --sim old=logs/real2sim/body_cam_render_fitted --sim new=logs/real2sim/look/body_iter4
"""

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image


def embed(images, model, proc, device):
    out = []
    for i in range(0, len(images), 16):
        batch = proc(images=[Image.fromarray(im) for im in images[i:i + 16]], return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**batch).last_hidden_state
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu())  # CLS + mean patch token
    e = torch.cat(out).numpy()
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def logistic_scores(X_tr, y_tr, X_te, l2: float = 1e-2) -> np.ndarray:
    """Decision values of an L2-regularised logistic regression (full-batch LBFGS), in plain torch."""
    X, y = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)
    w, b = torch.zeros(X.shape[1], requires_grad=True), torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return (torch.tensor(X_te, dtype=torch.float32) @ w + b).detach().numpy()


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), via ranks."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos = int(y.sum())
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(y) - n_pos)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--real", required=True, help="frames.npz")
    ap.add_argument("--sim", action="append", required=True, help="name=folder with sim_XX.png (repeatable)")
    ap.add_argument("--model", default="facebook/dinov2-base")
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    real = np.load(args.real)["images"]
    e_real = embed(real, model, proc, device)
    print(f"{args.camera}: {len(real)} real frames")
    for spec in args.sim:
        name, folder = spec.split("=", 1)
        files = sorted(glob.glob(os.path.join(folder, "sim_*.png")))
        sim = np.stack([np.array(Image.open(f).convert("RGB")) for f in files])[:len(real)]
        e_sim = embed(sim, model, proc, device)
        paired = 1 - (e_real[:len(sim)] * e_sim).sum(1)
        X = np.concatenate([e_real[:len(sim)], e_sim])
        y = np.r_[np.zeros(len(sim)), np.ones(len(sim))]
        # Pair-aware folds: a real frame and the sim frame at its state go to the same fold.
        folds = np.tile(np.arange(len(sim)) % 5, 2)
        scores = np.zeros(len(y))
        for f in range(5):
            tr, te = folds != f, folds == f
            scores[te] = logistic_scores(X[tr], y[tr], X[te])
        print(f"  {name:10s} paired cosine distance {paired.mean():.3f}   real-vs-sim AUC {roc_auc(y, scores):.3f}")


if __name__ == "__main__":
    main()
