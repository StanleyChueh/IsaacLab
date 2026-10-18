# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Real LeRobot dataset access shared by the real2sim tools (no Isaac imports): episodes, frames, joint states."""

import json
import os

import numpy as np

GRIPPER_SIM_OPEN = 0.044  # sim finger joint at fully open, as in lerobot_openarm/sim_bridge_common.py


def real_state_to_sim(state: np.ndarray, names: list[str], calib: dict) -> dict:
    """16 real motor values -> {sim joint name: rad}. Same mapping as sim_bridge_common.motor_action_to_sim_joints."""
    real = dict(zip(names, state.tolist()))
    sim = {}
    for side, prefix in (("left", "L"), ("right", "R")):
        sec = calib[side]
        for n in range(1, 8):
            sim[f"openarm_{side}_joint{n}"] = (real[f"{prefix}J{n}.pos"] - sec["offset_rad"][f"joint{n}"]) / sec["sign"][f"joint{n}"]
        grip = sec["gripper"]
        frac = np.clip((real[f"{prefix}J8.pos"] - grip["closed_raw"]) / (grip["open_raw"] - grip["closed_raw"]), 0.0, 1.0)
        sim[f"openarm_{side}_finger_joint1"] = sim[f"openarm_{side}_finger_joint2"] = float(frac) * GRIPPER_SIM_OPEN
    return sim


def load_episodes(root: str, camera: str):
    import pandas as pd

    info = json.load(open(os.path.join(root, "meta/info.json")))
    key = f"observation.images.{camera}"
    data = pd.concat(pd.read_parquet(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(root, "data"))
                     for f in sorted(fs) if f.endswith(".parquet"))
    episodes = pd.concat(pd.read_parquet(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(root, "meta/episodes"))
                         for f in sorted(fs) if f.endswith(".parquet")).set_index("episode_index")
    out = []
    for ep, meta in episodes.iterrows():
        rows = data[data.episode_index == ep].sort_values("frame_index")
        video = os.path.join(root, info["video_path"].format(
            video_key=key, chunk_index=int(meta[f"videos/{key}/chunk_index"]), file_index=int(meta[f"videos/{key}/file_index"])))
        out.append(dict(episode=int(ep), video=video, t0=float(meta[f"videos/{key}/from_timestamp"]),
                        state=np.stack(rows["observation.state"].to_numpy()).astype(np.float64)))
    return out, info["features"]["observation.state"]["names"], info["fps"]


def decode_episodes(episodes, fps: float, gray: bool = True, every: int = 1):
    """Yield (episode dict, frames) in file order, one episode buffered at a time.

    frames is a list of images (grayscale if ``gray``) indexed by frame number when ``every`` is 1, else a
    list of (frame number, RGB image) for every ``every``-th frame.
    """
    import av  # lerobot_openarm's venv has it; the Isaac env does not, and only needs real_state_to_sim
    import cv2

    by_video = {}
    for e in episodes:
        by_video.setdefault(e["video"], []).append(e)
    for video, eps in by_video.items():
        eps = sorted(eps, key=lambda e: e["t0"])
        with av.open(video) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            i, frames = 0, []
            for frame in container.decode(stream):
                t = float(frame.pts * stream.time_base)
                e = eps[i]
                k = int(round((t - e["t0"]) * fps))
                if k >= len(e["state"]):
                    yield e, frames
                    i, frames = i + 1, []
                    if i == len(eps):
                        break
                    e = eps[i]
                    k = int(round((t - e["t0"]) * fps))
                if k >= 0 and k % every == 0:
                    img = frame.to_ndarray(format="rgb24")
                    if every == 1:
                        frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if gray else img)
                    else:
                        frames.append((k, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if gray else img))
            if i < len(eps) and frames:
                yield eps[i], frames
