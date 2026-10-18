# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Pull (camera frame, joint state) pairs out of a real LeRobot v3 dataset, for camera calibration.

Writes <out>/frames.npz with ``images`` (N,H,W,3 uint8), ``state`` (N,16), ``names`` (the 16 state keys),
``episode`` and ``frame`` (N,), plus one PNG per frame to look at. Frames are spread over episodes and over
each episode's length so the arms are in many different poses -- the camera is solved from where the robot
appears, so near-identical poses add nothing. Frames where the arm is moving fast are skipped: the state
and the video frame are stamped by different devices, and a moving arm turns any small offset between them
into a silhouette that does not match its own joint reading.

Plain python with pandas + pyarrow + av (lerobot_openarm's venv has them; this IsaacLab env has no av):
  ~/Stanley_ws/lerobot_openarm/.venv/bin/python3 scripts/tools/sysid/extract_real_frames.py \\
      ~/.cache/huggingface/lerobot/ethanCSL/openarm_pringles_lerobot_real_v00 logs/real2sim/body_cam_frames
"""

import argparse
import json
import os

import av
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="Local LeRobot dataset root (contains meta/info.json).")
    ap.add_argument("out")
    ap.add_argument("--camera", default="body_cam")
    ap.add_argument("--episodes", type=int, default=16, help="How many episodes to sample (evenly spaced).")
    ap.add_argument("--per_episode", type=int, default=3, help="Frames per episode (evenly spaced in time).")
    ap.add_argument("--max_speed", type=float, default=0.15, help="Skip frames whose fastest joint moves faster (rad/s).")
    args = ap.parse_args()

    info = json.load(open(os.path.join(args.root, "meta/info.json")))
    fps = info["fps"]
    key = f"observation.images.{args.camera}"
    names = info["features"]["observation.state"]["names"]
    data = pd.concat(
        pd.read_parquet(os.path.join(dp, f))
        for dp, _, fs in os.walk(os.path.join(args.root, "data")) for f in sorted(fs) if f.endswith(".parquet")
    )
    episodes = pd.concat(
        pd.read_parquet(os.path.join(dp, f))
        for dp, _, fs in os.walk(os.path.join(args.root, "meta/episodes")) for f in sorted(fs) if f.endswith(".parquet")
    ).set_index("episode_index")

    chosen = np.linspace(0, len(episodes) - 1, min(args.episodes, len(episodes))).round().astype(int)
    picks = []  # (video file, timestamp in that file, episode, frame, state)
    for ep in chosen:
        rows = data[data.episode_index == ep].sort_values("frame_index")
        state = np.stack(rows["observation.state"].to_numpy()).astype(np.float32)
        speed = np.abs(np.gradient(state[:, [i for i, n in enumerate(names) if not n.endswith("8.pos")]], axis=0)).max(1) * fps
        meta = episodes.loc[ep]
        video = os.path.join(args.root, info["video_path"].format(
            video_key=key, chunk_index=int(meta[f"videos/{key}/chunk_index"]), file_index=int(meta[f"videos/{key}/file_index"])))
        for target in np.linspace(0.1, 0.9, args.per_episode) * (len(rows) - 1):
            # nearest slow frame to the evenly spaced target
            order = np.argsort(np.abs(np.arange(len(rows)) - target))
            slow = [i for i in order[:60] if speed[i] <= args.max_speed]
            if slow:
                i = slow[0]
                picks.append((video, float(meta[f"videos/{key}/from_timestamp"]) + i / fps, int(ep), i, state[i]))

    images = {}
    for video in sorted({p[0] for p in picks}):
        wanted = sorted(p[1] for p in picks if p[0] == video)
        with av.open(video) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                t = float(frame.pts * stream.time_base)
                for w in wanted:
                    if (video, w) not in images and abs(t - w) < 0.5 / fps:
                        images[(video, w)] = frame.to_ndarray(format="rgb24")
                if t > wanted[-1] + 1.0:
                    break

    picks = [p for p in picks if (p[0], p[1]) in images]
    os.makedirs(args.out, exist_ok=True)
    out = dict(
        images=np.stack([images[(p[0], p[1])] for p in picks]),
        state=np.stack([p[4] for p in picks]),
        names=np.array(names),
        episode=np.array([p[2] for p in picks]),
        frame=np.array([p[3] for p in picks]),
    )
    np.savez_compressed(os.path.join(args.out, "frames.npz"), **out)
    for k, p in enumerate(picks):
        av.VideoFrame.from_ndarray(out["images"][k], format="rgb24").to_image().save(
            os.path.join(args.out, f"real_{k:02d}_ep{p[2]}_f{p[3]}.png"))
    print(f"{len(picks)} frames from {len(chosen)} episodes -> {args.out}/frames.npz")


if __name__ == "__main__":
    main()
