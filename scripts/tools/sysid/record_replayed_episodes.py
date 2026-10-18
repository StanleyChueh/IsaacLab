# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Record sim episodes without a teleoperator: replay real episodes' actions through lerobot_sim_server.py.

Drives the running sim server the way lerobot-record drives it through --robot.type=openarm_isaac -- read the
observation, send the action, one control step each, 30 Hz lockstep -- and writes the frames to a LeRobot dataset with
the real dataset's feature schema (motor-radian state and action, three 480x640 cameras). The actions are a real
episode's recorded teleop commands, and the can is put where that episode's can started (prepare_real_replay.py), so
the episode is a plausible handover. Whatever look the server runs with (--real_look) is what gets recorded.

The sim does not always carry the can (finger geometry and friction differ), so by default only episodes in which the task's
own success condition (the hand-over) was met at some step are kept; the others are discarded and the next real episode is
tried. --keep_all saves every replay.

  # terminal 1: ./isaaclab.sh -p scripts/tools/lerobot_sim_server.py --enable_cameras --headless --real_look \\
  #                 --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 --task_mode handover --domain_randomization none
  # terminal 2 (lerobot_openarm's venv):
  python scripts/tools/sysid/record_replayed_episodes.py --repo_id ethanCSL/openarm_pringles_lerobot_sim_v01_reallook \\
      --num_episodes 5
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
from sim_bridge_common import load_calibration, motor_action_to_sim_joints, sim_joints_to_motor_action  # noqa: E402

CAMERAS = ("body_cam", "wrist_cam", "right_wrist_cam")


def load_sim_link():
    path = os.path.join(LEROBOT_OPENARM, "plugins", "lerobot_robot_openarm_umeow", "lerobot_robot_openarm_umeow", "sim_link.py")
    spec = importlib.util.spec_from_file_location("openarm_sim_link", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--num_episodes", type=int, default=5)
    ap.add_argument("--real_repo", default=os.path.expanduser("~/.cache/huggingface/lerobot/ethanCSL/openarm_pringles_lerobot_real_v00"))
    ap.add_argument("--replay", default="logs/real2sim/replay", help="prepare_real_replay.py output (trajectories, can starts).")
    ap.add_argument("--can_starts", default="can_starts_use.json")
    ap.add_argument("--first_episode", type=int, default=0, help="Real episode to start from.")
    ap.add_argument("--keep_all", action="store_true", help="Save every replay, not only the ones that met the task's success condition.")
    ap.add_argument("--task", default="Pick up the Pringles can with the right arm, hand it to the left arm")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5710)
    ap.add_argument("--calibration", default=os.path.join(LEROBOT_OPENARM, "calibration.json"))
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    link = load_sim_link()
    calib = load_calibration(args.calibration)
    traj = np.load(os.path.join(args.replay, "trajectories.npz"))
    names = [str(n) for n in traj["names"]]
    starts = json.load(open(os.path.join(args.replay, args.can_starts)))
    props = any(isinstance(v, dict) for v in starts.values())  # object_starts.json (battery task) rather than can xy
    if props:  # an object the detector missed: put it where it usually sits relative to the charger
        have = [(v["battery"][0] - v["charger"][0], v["battery"][1] - v["charger"][1]) for v in starts.values() if "battery" in v and "charger" in v]
        off = np.median(np.array(have), 0)
        for k, v in starts.items():
            if "battery" not in v and "charger" in v:
                v["battery"] = [v["charger"][0] + off[0], v["charger"][1] + off[1], 0.0]
                print(f"  episode {k}: battery start not detected, placed at the median offset from the charger")
    info = json.load(open(os.path.join(args.real_repo, "meta", "info.json")))
    features = {}
    for key, ft in info["features"].items():
        if key in ("action", "observation.state") or key.startswith("observation.images."):
            features[key] = {k: v for k, v in ft.items() if k != "info"}
    dataset = LeRobotDataset.create(args.repo_id, fps=info["fps"], features=features, robot_type=info["robot_type"],
                                    use_videos=True, streaming_encoding=True, encoder_threads=2)

    client = link.SimClient(args.host, args.port, timeout_s=300)
    hello = client.connect()
    print(f"connected: {hello['task']} ({hello['task_mode']}), cameras {list(hello['cameras'])}, {hello['control_hz']} Hz")
    saved, ep = 0, args.first_episode
    while saved < args.num_episodes and ep < len(episodes_available := [int(k.split("_")[1]) for k in traj.files if k.startswith("state_")]):
        if str(ep) not in starts:
            ep += 1
            continue
        actions = traj[f"action_{ep}"]
        states = traj[f"state_{ep}"]
        first = dict(zip(names, states[0].tolist()))
        placement = {"objects": starts[str(ep)]} if props else {"object_xy": starts[str(ep)]}
        obs = client.request("reset", joints=motor_action_to_sim_joints(first, calib), **placement)
        z0 = {k: v[2] for k, v in obs.get("objects", {}).items()}
        zmax = dict(z0)
        t0 = time.perf_counter()
        frames, success = [], bool(obs.get("success"))
        for k in range(len(actions)):
            state = sim_joints_to_motor_action(obs["joints"], calib)
            frame = {"observation.state": np.array([state[n] for n in names], np.float32),
                     "action": actions[k].astype(np.float32), "task": args.task}
            for cam in CAMERAS:
                frame[f"observation.images.{cam}"] = obs["images"][cam]
            frames.append(frame)
            obs = client.request("step", joints=motor_action_to_sim_joints(dict(zip(names, actions[k].tolist())), calib))
            success = success or bool(obs.get("success"))
            for name, pose in obs.get("objects", {}).items():
                zmax[name] = max(zmax[name], pose[2])
        if props:  # grasp-level outcome: did each arm really lift its object? (no insertion: the charger's slots are only drawn)
            lifts = {k: 100 * (zmax[k] - z0[k]) for k in z0}
            c, b = obs["objects"]["charger"], obs["objects"]["battery"]
            success = lifts["charger"] > 4.0 and lifts["battery"] > 2.5
            print(f"real episode {ep}: {len(actions)} steps in {time.perf_counter() - t0:.0f} s; lifted charger {lifts['charger']:.1f} cm, "
                  f"battery {lifts['battery']:.1f} cm; final battery-to-charger offset ({100 * (b[0] - c[0]):+.1f}, {100 * (b[1] - c[1]):+.1f}, "
                  f"{100 * (b[2] - c[2]):+.1f}) cm -> grasp-level success: {success}", flush=True)
        else:
            print(f"real episode {ep}: {len(actions)} steps in {time.perf_counter() - t0:.0f} s, hand-over success condition met: {success}", flush=True)
        if not args.keep_all and not success:
            print("  not a successful handover in sim: discarded")
            ep += 1
            continue
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        saved += 1
        print(f"  saved as sim episode {saved - 1} ({saved}/{args.num_episodes})", flush=True)
        ep += 1
    dataset.finalize()
    client.close()
    print(f"{saved} episodes -> {dataset.root}")


if __name__ == "__main__":
    main()
