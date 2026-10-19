# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Record battery-insertion episodes from the scripted expert (battery_expert.py) through the running lerobot sim server.

Same wire protocol as ``lerobot-record --robot.type=openarm_isaac`` and ``record_replayed_episodes.py`` (observe, act, one
30 Hz control step each; the dataset has the real battery dataset's feature schema and motor-radian units), but the action
is the expert's: it reads the charger, the battery and the hands from the observation and plans each stage from there, so an
episode ends with the battery IN the charger's bay, not merely carried near it. Only episodes whose success condition
(``battery_in_charger``, held until the end of the episode) was met are kept.

  # terminal 1:
  ./isaaclab.sh -p scripts/tools/lerobot_sim_server.py --enable_cameras --headless --task Isaac-InsertBattery-OpenArm-IK-Abs-v0
  # terminal 2 (lerobot_openarm's venv):
  python scripts/tools/sysid/record_expert_episodes.py --repo_id ethanCSL/openarm_insertbattery_lerobot_sim_v02_expert --num_episodes 20
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

LEROBOT_OPENARM = os.path.expanduser("~/Stanley_ws/lerobot_openarm")
sys.path.insert(0, LEROBOT_OPENARM)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_bridge_common import load_calibration, motor_action_to_sim_joints, sim_joints_to_motor_action  # noqa: E402

from battery_expert import BatteryExpert, Failed  # noqa: E402

CAMERAS = ("body_cam", "wrist_cam", "right_wrist_cam")


def load_sim_link():
    path = os.path.join(LEROBOT_OPENARM, "plugins", "lerobot_robot_openarm_umeow", "lerobot_robot_openarm_umeow", "sim_link.py")
    spec = importlib.util.spec_from_file_location("openarm_sim_link", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", default=None, help="Dataset to write (omit to only run and report).")
    ap.add_argument("--num_episodes", type=int, default=5)
    ap.add_argument("--max_tries", type=int, default=40)
    ap.add_argument("--real_repo", default=os.path.expanduser("~/.cache/huggingface/lerobot/ethanCSL/openarm_insertbattery_lerobot_real_v01"))
    ap.add_argument("--replay", default="logs/real2sim/battery/replay")
    ap.add_argument("--task", default="Insert the battery into the charger")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5710)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--tail_s", type=float, default=1.0, help="Seconds recorded after the expert finishes.")
    ap.add_argument("--debug_dir", default=None, help="Dump the three camera images every --debug_every steps here.")
    ap.add_argument("--debug_every", type=int, default=30)
    ap.add_argument("--charger", default=None, help="x,y,yaw_deg: place the charger here (and --battery) instead of drawing a start.")
    ap.add_argument("--battery", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="BatteryExpert overrides, name=value.")
    ap.add_argument("--calibration", default=os.path.join(LEROBOT_OPENARM, "calibration.json"))
    args = ap.parse_args()

    link = load_sim_link()
    calib = load_calibration(args.calibration)
    traj = np.load(os.path.join(args.replay, "trajectories.npz"))
    names = [str(n) for n in traj["names"]]
    starts = json.load(open(os.path.join(args.replay, "object_starts_fitted.json")))
    ep_ids = sorted(int(k.split("_")[1]) for k in traj.files if k.startswith("state_"))
    rng = np.random.default_rng(args.seed)
    overrides = {}
    for it in args.set:
        k, v = it.split("=")
        overrides[k] = np.radians(float(v)) if k.endswith("_pitch") else float(v)

    dataset = None
    if args.repo_id:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        info = json.load(open(os.path.join(args.real_repo, "meta", "info.json")))
        features = {k: {a: b for a, b in ft.items() if a != "info"} for k, ft in info["features"].items()
                    if k in ("action", "observation.state") or k.startswith("observation.images.")}
        dataset = LeRobotDataset.create(args.repo_id, fps=info["fps"], features=features, robot_type=info["robot_type"],
                                        use_videos=True, streaming_encoding=True, encoder_threads=2)
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    client = link.SimClient(args.host, args.port, timeout_s=600)
    hello = client.connect()
    print(f"connected: {hello['task']}, cameras {list(hello['cameras'])}, {hello['control_hz']} Hz", flush=True)
    saved = tries = 0
    while saved < args.num_episodes and tries < args.max_tries:
        tries += 1
        # a real episode's first pose, and objects scattered over the region the 9 real episodes (and the arms' reach) cover
        src = int(rng.choice(ep_ids))
        first = dict(zip(names, traj[f"state_{src}"][0].tolist()))
        place = {"charger": [rng.uniform(0.300, 0.355), rng.uniform(0.070, 0.100), rng.uniform(-14.0, 6.0)],
                 "battery": [rng.uniform(0.260, 0.390), rng.uniform(-0.115, -0.075), rng.uniform(-14.0, 5.0)]}
        if args.charger and args.battery:
            place = {"charger": [float(v) for v in args.charger.split(",")], "battery": [float(v) for v in args.battery.split(",")]}
        obs = client.request("reset", joints=motor_action_to_sim_joints(first, calib), objects=place)
        jitter = dict(overrides)
        jitter.setdefault("speed", float(rng.uniform(1.35, 1.65)))  # not every episode at the same pace
        expert = BatteryExpert(**jitter)
        expert.reset(obs)
        frames, done_at, ok_steps = [], None, 0
        t0 = time.perf_counter()
        status = "ok"
        for k in range(args.max_steps):
            try:
                joints, done = expert.step(obs)
            except Failed as e:
                status = f"FAILED {e}"
                break
            state = sim_joints_to_motor_action(obs["joints"], calib)
            action = sim_joints_to_motor_action(joints, calib)
            frame = {"observation.state": np.array([state[n] for n in names], np.float32),
                     "action": np.array([action[n] for n in names], np.float32), "task": args.task}
            for cam in CAMERAS:
                frame[f"observation.images.{cam}"] = obs["images"][cam]
            frames.append(frame)
            if args.debug_dir and k % args.debug_every == 0:
                from PIL import Image

                row = np.concatenate([obs["images"][c] for c in CAMERAS], axis=1)
                Image.fromarray(row).save(os.path.join(args.debug_dir, f"try{tries:02d}_{k:04d}_{expert.cur['label'] if expert.cur else 'x'}.jpg"))
            obs = client.request("step", joints=joints)
            ok_steps = ok_steps + 1 if obs.get("success") else 0
            if done and done_at is None:
                done_at = k
            if done_at is not None and k - done_at >= args.tail_s * hello["control_hz"]:
                break
        success = status == "ok" and done_at is not None and bool(obs.get("success"))
        b, c = obs["objects"]["battery"], obs["objects"]["charger"]
        print(f"try {tries}: start charger ({place['charger'][0]:.3f}, {place['charger'][1]:.3f}, {place['charger'][2]:.0f} deg) battery ({place['battery'][0]:.3f}, {place['battery'][1]:.3f}, {place['battery'][2]:.0f} deg); {len(frames)} steps in {time.perf_counter() - t0:.0f} s, {status}; success at end {success} (seated for the last {ok_steps} steps); "
              f"battery-charger ({100 * (b[0] - c[0]):+.1f}, {100 * (b[1] - c[1]):+.1f}, {100 * (b[2] - c[2]):+.1f}) cm", flush=True)
        if success and dataset is not None:
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            saved += 1
            print(f"  saved as sim episode {saved - 1}", flush=True)
        elif success:
            saved += 1
    if dataset is not None:
        dataset.finalize()
        print(f"{saved} episodes ({tries} tries) -> {dataset.root}")
    client.close()


if __name__ == "__main__":
    main()
