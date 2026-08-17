# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Dual-arm recording script for OpenArm tasks.

Extends the standard record_demos.py with TAB key arm switching so the operator
can control either the left or the right arm during data collection.

Uses OpenArmKeyboard instead of Se3Keyboard to avoid Isaac Sim viewport conflicts.
W/A/S/D/Q/E are viewport gizmo shortcuts — this script uses arrow keys + I/O instead.

Keyboard controls:
  W/S        forward / backward  (EE +x/-x)
  A/D        left / right        (EE +y/-y)
  PgUp/PgDn  up / down           (EE +z/-z)
  ↑/↓        pitch ±
  ←/→        yaw ±
  [/]        roll ±
  K          toggle gripper open/close
  TAB        switch active arm (left ↔ right)
  R          reset / discard current episode
  N          save current episode as successful

Action space, --teleop_device keyboard/vr_ros2 (14D flat):
  [0:6]   left arm IK delta pose (dx dy dz drx dry drz)
  [6:7]   left gripper command (±1.0)
  [7:13]  right arm IK delta pose
  [13:14] right gripper command (±1.0)

Action space, --teleop_device vr_joint_ros2 (16D flat -- arm_action/right_arm_action are
reassigned from IK to JointPositionActionCfg at load time, see swap_to_joint_position_actions):
  [0:7]   left arm joint-position delta-from-default (7 joints)
  [7:8]   left gripper command (±1.0)
  [8:15]  right arm joint-position delta-from-default
  [15:16] right gripper command (±1.0)

Action space, --teleop_device vr_joint_ros2_native (16D flat -- all four action fields are
reassigned to ROS2JointCommandActionCfg at load time, see swap_to_native_ros2_joint_actions).
Every column is the ABSOLUTE joint target decoded from the ROS2 message that step, i.e. the
command actually applied, not the pose that resulted from it (see
ROS2NativeJointTeleop.build_dual_action):
  [0:7]   left arm joint positions (rad)
  [7:8]   left gripper joint1 position (m), quantised to 0.0 closed / 0.044 open by
          snap_gripper_command -- never an intermediate trigger position
  [8:15]  right arm joint positions
  [15:16] right gripper joint1 position

Usage:

 ./isaaclab.sh -p scripts/tools/record_demos_openarm.py  --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0  --dataset_file logs/demos/pickup.hdf5 --enable_cameras  --num_demos 1    --teleop_device vr_ros2  --vr_udp_host 127.0.0.1  --vr_udp_port 5800

processed:vr_joint_ros2 udp
 ./isaaclab.sh -p scripts/tools/record_demos_openarm.py \
 --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0  \
 --dataset_file logs/demos/pickup.hdf5  \
 --enable_cameras  \
 --num_demos 1   \
 --teleop_device vr_joint_ros2  \
 --vr_joint_udp_host 127.0.0.1 \
 --vr_joint_udp_port 5801


./isaaclab.sh -p scripts/tools/record_demos_openarm.py \
  --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \
  --dataset_file logs/demos/pickup.hdf5 \
  --enable_cameras \
  --num_demos 1 \
  --teleop_device vr_joint_ros2_native \
  --ros2_domain_id 1

Resuming a session:

  An existing --dataset_file is never silently overwritten; the run refuses to start instead. Add
  --resume to keep what is in it and append until it holds --num_demos in total, or --overwrite to
  start again from empty. So a session that was stopped at 12 of 20 demos is finished with the SAME
  command plus --resume:

  ./isaaclab.sh -p scripts/tools/record_demos_openarm.py \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \
    --dataset_file logs/demos/pickup_pringles_VR_V7.hdf5 \
    --enable_cameras --num_demos 20 --teleop_device vr_joint_ros2_native \
    --ros2_domain_id 1 --task_mode handover --manual_save --resume

  To drop bad demos first (and renumber the survivors, which --resume then continues after), see
  scripts/tools/remove_demos_hdf5.py.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import atexit
import signal
import contextlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record dual-arm OpenArm demonstrations with arm switching.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--task_mode",
    type=str,
    default=None,
    choices=["left", "right", "handover"],
    help=(
        "Which variant of the OpenArm pick-up task to record, i.e. which arm(s) the demo is"
        " about, which arm(s) you may drive, and therefore what ends it. Any mode replaces the"
        " task's default cube_2 with a can (see openarm_task_modes.py). 'left'/'right':"
        " single-arm pick-up -- ONLY that arm is teleoperable (the other is held at its rest"
        " pose), the can spawns anywhere on the pad, and the demo auto-ends once"
        " THAT arm is holding it in the air. 'handover': both arms are teleoperable, the can"
        " spawns in a narrow band on the midline between the two hands (y within +-1.2 cm of 0),"
        " the right arm picks it up and passes it to the left, and the"
        " demo auto-ends once the LEFT arm is holding it in the air with the right gripper open."
        " Every mode records BOTH arms' joints either way -- the gating is on control, not on"
        " what lands in the dataset. Omit to keep the task cfg's own settings (cube_2 anywhere on"
        " the pad, both arms live, auto-end on cube height alone, which accepts episodes where"
        " nothing was ever grasped -- fine for raw teleop, not for Mimic source demos). Whatever"
        " is chosen here must be passed to annotate_demos.py too, since it also selects which"
        " subtask term signals exist. See"
        " isaaclab_tasks/.../config/openarm/openarm_task_modes.py."
    ),
)
parser.add_argument(
    "--manual_save",
    action="store_true",
    default=False,
    help=(
        "Never auto-export on the task's success condition -- an episode ends only when you say"
        " so (Quest button X a second time, or the N key). Use this when the success condition is"
        " not reliably recognised and you would rather judge the demo yourself than have episodes"
        " that refuse to end: --task_mode handover in particular has to recognise a whole"
        " right-grasp -> pass -> release -> land sequence, and any step it misses leaves the"
        " episode running. Recording still STARTS the same way (button X). Without this flag the"
        " success condition can also end the episode, and button X remains available either way."
    ),
)
parser.add_argument(
    "--return_to_rest_secs",
    type=float,
    default=1.5,
    help=(
        "How long the closing return-to-rest motion takes, in seconds. On the second Quest button"
        " X press the robot is driven back to the task's rest pose along a straight line in joint"
        " space and the episode is exported only once it arrives -- so the retreat is RECORDED as"
        " part of the demo instead of happening off-camera during the reset. The grippers are held"
        " at whatever they were commanded to, so a held object is carried back rather than dropped."
        " The trajectory picks up from the pose and speed the arms were last commanded at and eases"
        " to a stop (see ReturnToRestTrajectory), so this buys a shorter tail on every episode"
        " without the joints being asked to jump or to stop dead -- lower it"
        " further if the retreat still drags, raise it if a carried object slips or swings."
        " Set to 0 to export immediately on the second press (the old behavior). Only applies to"
        " the joint-space Quest-button devices (vr_joint_ros2 / vr_joint_ros2_native); the N key"
        " always exports immediately, whatever this is set to."
    ),
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=["keyboard", "vr_ros2", "vr_joint_ros2", "vr_joint_ros2_native"],
    help=(
        "Teleop device. 'keyboard' uses OpenArmKeyboard (single active arm, TAB to switch)."
        " 'vr_ros2' drives BOTH arms simultaneously from a UDP JSON side-channel fed by"
        " nodes/dora-openarm-ros2-bridge/bridge.py's --vr-udp-port (see --vr_udp_host/"
        " --vr_udp_port below) -- only valid on dual-arm (14D action) tasks. 'vr_joint_ros2'"
        " also drives both arms, but from target JOINT ANGLES (not a Cartesian pose) via a UDP"
        " JSON side-channel fed by dora-openarm-ros2-bridge/joint_command_processor.py's"
        " --vr-joint-udp-port (see --vr_joint_udp_host/--vr_joint_udp_port below); it reassigns"
        " the task's arm_action/right_arm_action from IK to JointPositionActionCfg at load time,"
        " so the joint targets apply directly with no Cartesian frame/calibration step (unlike"
        " 'vr_ros2', it doesn't need --vr_quat_offset). 'vr_joint_ros2_native' is the same"
        " joint-space control as 'vr_joint_ros2', but skips the UDP JSON hop entirely: it builds a"
        " ROS2 OmniGraph (isaacsim.ros2.bridge.ROS2SubscribeJointState) that decodes"
        " /openarm/vr_joint_command_processed over native ROS2 DDS every frame, and"
        " arm_action/gripper_action/right_arm_action/right_gripper_action are reassigned (see"
        " swap_to_native_ros2_joint_actions) to read that decoded message directly in"
        " apply_actions() instead of using whatever env.step() was called with -- see"
        " ROS2JointCommandAction's docstring for why the joint targets still go through"
        " IsaacLab's normal action/actuator pipeline rather than a graph node writing PhysX"
        " directly (the latter races with IsaacLab's own actuator model and was tried first; it"
        " silently lost that race and looked like the topic wasn't connected at all). See"
        " --ros2_topic/--ros2_domain_id below. Much lower latency than the UDP path, at the cost"
        " of needing isaacsim.ros2.bridge and the same ROS_DOMAIN_ID as the dora side. R/N/T"
        " keyboard shortcuts (reset/save/ramp-test) still work in all VR modes; TAB/arm-switch"
        " does not (both arms are always live)."
    ),
)
parser.add_argument(
    "--vr_udp_host",
    type=str,
    default="127.0.0.1",
    help="Host to bind for --teleop_device vr_ros2's UDP JSON listener.",
)
parser.add_argument(
    "--vr_udp_port",
    type=int,
    default=5800,
    help="Port to bind for --teleop_device vr_ros2's UDP JSON listener.",
)
parser.add_argument(
    "--vr_max_pos_step",
    type=float,
    default=0.01,
    help=(
        "--teleop_device vr_ros2 only: per-step clamp (meters) on the position delta sent to"
        " the IK controller. Since the delta is recomputed from the live VR target every step"
        " (not accumulated), clamping just makes the arm converge to the target over a few"
        " steps instead of possibly snapping -- e.g. right after a reset, or if a UDP packet"
        " is stale/lost for a moment."
    ),
)
parser.add_argument(
    "--vr_max_rot_step",
    type=float,
    default=0.05,
    help="--teleop_device vr_ros2 only: per-step clamp (radians) on the axis-angle rotation delta. See --vr_max_pos_step.",
)
parser.add_argument(
    "--vr_quat_offset",
    type=float,
    nargs=4,
    default=[1.0, 0.0, 0.0, 0.0],
    metavar=("QW", "QX", "QY", "QZ"),
    help=(
        "--teleop_device vr_ros2 only: fixed wxyz rotation left-multiplied onto every incoming"
        " VR target orientation (in the robot base frame) before it's used for IK. dora-openarm-vr's"
        " quest_receiver.py maps raw controller poses into the robot 'arm_origin' frame using a"
        " _FRAME_ROT/r_fix pair that was hand-tuned against the MuJoCo viewer -- if Isaac Sim's robot"
        " base frame uses a different axis convention than MuJoCo's arm_origin, the same UDP pose"
        " stream will produce a rotated-looking arm here. Default is identity (no correction, matches"
        " prior behavior). To determine the actual offset: hold the VR controller still, let the arm"
        " converge in both sims, read off openarm_*_ee_tcp's orientation here vs. MuJoCo's for that"
        " same held pose, and solve for the rotation that maps one to the other."
    ),
)
parser.add_argument(
    "--vr_joint_udp_host",
    type=str,
    default="127.0.0.1",
    help="Host to bind for --teleop_device vr_joint_ros2's UDP JSON listener.",
)
parser.add_argument(
    "--vr_joint_udp_port",
    type=int,
    default=5801,
    help="Port to bind for --teleop_device vr_joint_ros2's UDP JSON listener.",
)
parser.add_argument(
    "--ros2_topic",
    type=str,
    default="/openarm/vr_joint_command_processed",
    help=(
        "--teleop_device vr_joint_ros2_native only: ROS2 JointState topic the OmniGraph"
        " subscribes to. Must match dora-openarm-ros2-bridge/joint_command_processor.py's"
        " output topic (its --vr-joint-udp-port UDP broadcast carries the same data over a"
        " different transport, for vr_joint_ros2 instead)."
    ),
)
parser.add_argument(
    "--ros2_domain_id",
    type=int,
    default=1,
    help=(
        "--teleop_device vr_joint_ros2_native only: ROS_DOMAIN_ID for the OmniGraph's"
        " ROS2Context node. Must match the dora side's ROS_DOMAIN_ID (see"
        " dataflow-vr-mujoco-ros2.yaml's ros2-bridge/joint-command-processor node env) or this"
        " process simply never sees any packets -- DDS domains that don't match don't discover"
        " each other, there's no error, just silence."
    ),
)
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos",
    type=int,
    default=0,
    help=(
        "Number of demonstrations the dataset should end up holding (0 = record until stopped)."
        " With --resume this is the TOTAL, counting the demos already in the file, so the same"
        " --num_demos can be used to finish an interrupted session."
    ),
)
parser.add_argument(
    "--resume",
    action="store_true",
    help=(
        "Continue an interrupted recording session: keep the episodes already in --dataset_file and"
        " append new ones after them, stopping once the file holds --num_demos in total. Without"
        " this flag an existing dataset file is not overwritten -- the run refuses to start."
    ),
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Discard an existing --dataset_file and record from scratch. Mutually exclusive with --resume.",
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of consecutive success steps to conclude a demo.",
)
parser.add_argument(
    "--mirror_udp_port",
    type=int,
    default=0,
    help=(
        "If nonzero, broadcast the robot's current joint positions (by name, radians) as a UDP JSON packet"
        " to 127.0.0.1:<port> after every env.step(). Off by default. Intended to feed a separate,"
        " out-of-process real-robot bridge (see lerobot_openarm/mirror_bridge.py) -- this process never"
        " talks to hardware directly."
    ),
)
parser.add_argument(
    "--mirror_udp_host",
    type=str,
    default="127.0.0.1",
    help="Destination host for --mirror_udp_port. Defaults to loopback; only change this if you know why.",
)
parser.add_argument(
    "--mirror_feedback_port",
    type=int,
    default=0,
    help=(
        "If nonzero, listen on 127.0.0.1:<port> for UDP JSON feedback packets from the real-robot"
        " bridge (see lerobot_openarm/mirror_bridge.py's --feedback-port) carrying its ACTUAL current"
        " joint positions (already inverse-mapped back to sim joint names/radians). If set, a"
        " sim-vs-real comparison plot is saved to the current directory when this script exits."
        " Off by default -- this process still never talks to hardware directly, it only listens for"
        " numbers the bridge process chooses to send back."
    ),
)
parser.add_argument(
    "--dump_joint_order",
    type=str,
    default=None,
    help=(
        "If set, write the OpenArm joint name ordering (matching the column order of"
        " states/articulation/robot/joint_position in the exported HDF5) to this JSON path once at"
        " startup. Needed to replay a recorded episode on real hardware without requiring IsaacLab"
        " installed in the hardware-control environment -- see lerobot_openarm/replay_sim_dataset.py."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# ─── Existing-dataset pre-flight (--resume / --overwrite) ─────────────────────
# Deliberately before AppLauncher: everything below only reads the output file and the parsed
# arguments, and getting "that file already holds 12 demos" back in a second beats getting it after
# a minute of Isaac Sim startup. Nothing here writes; the file is not opened for writing until the
# RecorderManager is built, long after this has had its say.
import json as _json  # noqa: E402 -- see above; the post-launch import block re-imports these
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402


def next_demo_index(data_group) -> int:
    """The index a new episode should be written under, given the ones already in the file.

    One past the HIGHEST existing index rather than the episode count, so appending is safe even
    for a dataset whose numbering has gaps -- writing demo_{count} into such a file would collide
    with an episode already there and abort the export.
    """
    indices = [
        int(name[len("demo_") :])
        for name in data_group
        if name.startswith("demo_") and name[len("demo_") :].isdigit()
    ]
    return max(indices) + 1 if indices else 0


def inspect_dataset(file_path: str) -> dict:
    """Read back what an existing dataset holds, without opening it for writing."""
    import h5py

    with h5py.File(file_path, "r") as handle:
        if "data" not in handle:
            raise ValueError("file has no 'data' group -- it is not an IsaacLab dataset")
        data_group = handle["data"]
        env_args = _json.loads(data_group.attrs.get("env_args", "{}"))
        return {
            "count": len(data_group),
            "next_index": next_demo_index(data_group),
            "env_name": env_args.get("env_name"),
        }


def preflight_dataset(args) -> tuple[str, int]:
    """Decide what to do about an existing ``--dataset_file``, or exit explaining why we won't.

    Returns the path the recorder will write and how many episodes are already in it (0 unless
    resuming). Exits the process rather than returning an error, because every outcome here that
    is not "carry on" means the run must not start at all.
    """
    output_dir = _os.path.dirname(args.dataset_file)
    stem = _os.path.splitext(_os.path.basename(args.dataset_file))[0]
    # Reassembled the way the recorder's file handler does it (directory + stem + .hdf5) rather
    # than taken from --dataset_file, so a path given without the extension is still checked
    # against the file that will actually be written.
    dataset_path = _os.path.join(output_dir if output_dir else ".", f"{stem}.hdf5")

    def fail(message: str):
        print(f"ERROR: {message}")
        _sys.exit(1)

    if args.resume and args.overwrite:
        fail("--resume and --overwrite do the opposite of each other; pass at most one.")

    existing = None
    if _os.path.isfile(dataset_path):
        try:
            existing = inspect_dataset(dataset_path)
        except Exception as error:
            fail(
                f"could not read the existing dataset at {dataset_path}: {error}. If it is a"
                " leftover from a crashed run and holds nothing worth keeping, delete it or pass"
                " --overwrite."
            )

    if existing is None:
        if args.resume:
            print(f"[RESUME] {dataset_path} does not exist yet -- starting a fresh recording.")
        return dataset_path, 0

    if not (args.resume or args.overwrite):
        # Refusing rather than warning: the recorder opens its output for writing from scratch, so
        # starting this run would destroy every demo in that file -- irreversibly, and before the
        # operator has recorded anything to show for it.
        fail(
            f"{dataset_path} already exists and holds {existing['count']} episode(s), which"
            " recording into it would DELETE. Pass --resume to keep them and record the rest into"
            " the same file, --overwrite to start again from empty, or point --dataset_file"
            " somewhere else."
        )

    if args.overwrite:
        print(f"[OVERWRITE] Discarding the {existing['count']} episode(s) already in {dataset_path}.")
        return dataset_path, 0

    task_name = args.task.split(":")[-1]
    if existing["env_name"] and existing["env_name"] != task_name:
        fail(
            f"{dataset_path} was recorded for task '{existing['env_name']}', but this run is"
            f" '{task_name}'. Resuming would mix two tasks in one dataset. Use a different"
            " --dataset_file."
        )

    resume_offset = existing["count"]
    if args.num_demos > 0 and resume_offset >= args.num_demos:
        print(
            f"[RESUME] {dataset_path} already holds {resume_offset} episode(s), which meets"
            f" --num_demos {args.num_demos}. Nothing to record."
        )
        _sys.exit(0)

    remaining = args.num_demos - resume_offset if args.num_demos > 0 else None
    print(
        f"[RESUME] {dataset_path} holds {resume_offset} episode(s); appending from"
        f" demo_{existing['next_index']}."
        + (f" {remaining} more to reach --num_demos {args.num_demos}." if remaining else "")
    )
    # The file records nothing about --task_mode, so this is the one half of "same settings as last
    # time" that cannot be checked automatically.
    print("[RESUME] Check yourself that --task_mode matches the session being resumed.")
    return dataset_path, resume_offset


DATASET_PATH, RESUME_OFFSET = preflight_dataset(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import logging
import os
import re
import socket
import threading
import time
from dataclasses import MISSING

import gymnasium as gym
import torch

import matplotlib
matplotlib.use("Agg")  # headless -- this process only ever saves a PNG, never shows a window
import matplotlib.pyplot as plt

import omni.ui as ui

import isaaclab.envs.mdp as mdp
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import ActionTerm, ActionTermCfg, DatasetExportMode
from isaaclab.utils import configclass

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm import openarm_task_modes
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_task_modes import (
    CONTROLLED_ARMS,
    apply_task_mode,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


# ─── Keyboard controller ──────────────────────────────────────────────────────
# Linux XIM fires a secondary character-input RELEASE (no .name, str='w') for
# every letter key.  The try/except AttributeError in _on_event skips these so
# the delta is only modified once per physical key press/release.
class OpenArmKeyboard:
    """Robot keyboard controller for OpenArm recording.

    Translation:  W/S = ±X  |  A/D = ±Y  |  PgUp/PgDn = ±Z
    Rotation:     ↑/↓ = pitch ±Y  |  ←/→ = yaw ±Z  |  [/] = roll ±X
    Gripper:      K (toggle)
    Save/Reset:   N / R
    Arm switch:   TAB  (used by record_demos_openarm.py)
    """

    _POS_KEYS = {
        "W":           ( 1.0,  0.0,  0.0),   # EE forward
        "S":           (-1.0,  0.0,  0.0),   # EE backward
        "A":           ( 0.0,  1.0,  0.0),   # EE left
        "D":           ( 0.0, -1.0,  0.0),   # EE right
        "PAGE_UP":     ( 0.0,  0.0,  1.0),   # EE up
        "PAGE_DOWN":   ( 0.0,  0.0, -1.0),   # EE down
    }
    _ROT_KEYS = {
        "UP":            ( 0.0,  1.0,  0.0),   # pitch +
        "DOWN":          ( 0.0, -1.0,  0.0),   # pitch -
        "LEFT":          ( 0.0,  0.0,  1.0),   # yaw +
        "RIGHT":         ( 0.0,  0.0, -1.0),   # yaw -
        "LEFT_BRACKET":  ( 1.0,  0.0,  0.0),   # roll +
        "RIGHT_BRACKET": (-1.0,  0.0,  0.0),   # roll -
    }

    def __init__(self, pos_sensitivity: float = 0.05, rot_sensitivity: float = 0.1, sim_device: str = "cpu"):
        import carb.input as ci
        import omni.appwindow
        import numpy as np

        self._pos_sensitivity = pos_sensitivity
        self._rot_sensitivity = rot_sensitivity
        self._sim_device = sim_device
        self._np = np

        self._delta_pos = np.zeros(3, dtype=np.float64)
        self._delta_rot = np.zeros(3, dtype=np.float64)
        self._close_gripper = False
        self._additional_callbacks: dict = {}

        import weakref
        appwindow = omni.appwindow.get_default_app_window()
        self._keyboard = appwindow.get_keyboard()
        self._ci = ci.acquire_input_interface()
        self._sub = self._ci.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *_, obj=weakref.proxy(self): obj._on_event(event),
        )

    def __del__(self):
        try:
            self._ci.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
        except Exception:
            pass

    def reset(self):
        self._delta_pos[:] = 0.0
        self._delta_rot[:] = 0.0
        self._close_gripper = False

    def clear_deltas(self):
        """Like `reset()` but leaves `_close_gripper` untouched -- for clearing stray
        position/rotation deltas accumulated while keys were held during a non-teleop
        interlude (e.g. the ramp-to-rest test), without forcing the gripper open."""
        self._delta_pos[:] = 0.0
        self._delta_rot[:] = 0.0

    def add_callback(self, key: str, func):
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        from scipy.spatial.transform import Rotation
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        cmd = self._np.concatenate([self._delta_pos, rot_vec])
        cmd = self._np.append(cmd, -1.0 if self._close_gripper else 1.0)
        return torch.tensor(cmd, dtype=torch.float32, device=self._sim_device)

    def _on_event(self, event) -> bool:
        import carb.input as ci

        # Use event.input.name directly — same as Se3Keyboard.
        # carb.input.KeyboardInput enum members have .name == "UP", "PAGE_UP", etc.
        try:
            name = event.input.name
        except AttributeError:
            return True

        if event.type == ci.KeyboardEventType.KEY_PRESS:
            if name == "K":
                self._close_gripper = not self._close_gripper
            elif name in self._POS_KEYS:
                self._delta_pos += self._np.array(self._POS_KEYS[name]) * self._pos_sensitivity
            elif name in self._ROT_KEYS:
                self._delta_rot += self._np.array(self._ROT_KEYS[name]) * self._rot_sensitivity

            if name in self._additional_callbacks:
                self._additional_callbacks[name]()

        elif event.type == ci.KeyboardEventType.KEY_RELEASE:
            if name in self._POS_KEYS:
                self._delta_pos -= self._np.array(self._POS_KEYS[name]) * self._pos_sensitivity
            elif name in self._ROT_KEYS:
                self._delta_rot -= self._np.array(self._ROT_KEYS[name]) * self._rot_sensitivity

        return True

try:
    import isaaclab_mimic.envs  # noqa: F401
    from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions
    HAS_MIMIC = True
except ImportError:
    HAS_MIMIC = False

    class InstructionDisplay:
        def __init__(self, xr=False):
            pass
        def show_demo(self, text):
            pass
        def set_labels(self, *args):
            pass

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler

logger = logging.getLogger(__name__)


# ─── Resuming an interrupted recording session (--resume) ─────────────────────
# The pre-flight half of --resume (next_demo_index, inspect_dataset, preflight_dataset) lives
# further up, before AppLauncher, so a refusal costs a second rather than a sim startup. This half
# is what lets the recorder write into a dataset that already exists.
class ResumeExistingDatasetMixin(HDF5DatasetFileHandler):
    """Makes a dataset file handler APPEND to an existing file instead of truncating it.

    RecorderManager only ever calls ``create()`` on its handler, and create means h5py mode ``"w"``
    -- it wipes whatever was at that path. Right for a fresh session, and exactly wrong for
    ``--resume``, which exists because the file already holds demos worth keeping. Nothing else
    about the handler changes (same file layout, same episode contents); only the "file already
    exists" case differs: it is opened rather than replaced, and numbering continues after the
    highest demo already in it.

    A mixin rather than a subclass of one specific handler so it composes with whichever handler
    the recorder cfg carries -- see :func:`make_resumable`.
    """

    def create(self, file_path: str, env_name: str | None = None):
        import h5py

        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"
        if not _os.path.isfile(file_path):
            # "" rather than None for a missing name, which is what the base class turns None into
            # anyway -- it just declares the parameter as str.
            super().create(file_path, env_name=env_name if env_name is not None else "")
            return
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")

        # A handler that writes on a background thread (the async variant of this handler starts
        # one in its own create()) needs that thread up before anything can be queued, and starting
        # it is the only setup any of these handlers' create() does beyond opening the file. No-op
        # for the plain synchronous handler, which has no worker.
        worker_main = getattr(self, "_worker_main", None)
        if worker_main is not None and getattr(self, "_worker", None) is None:
            worker = threading.Thread(target=worker_main, name="hdf5-episode-writer", daemon=True)
            worker.start()
            setattr(self, "_worker", worker)  # noqa: B010 -- not an attribute of the base class

        self._hdf5_file_stream = h5py.File(file_path, "a")
        self._hdf5_data_group = self._hdf5_file_stream["data"]
        self._demo_count = next_demo_index(self._hdf5_data_group)
        # Seed the in-memory env_args from the file's, because add_env_args UPDATES this dict and
        # rewrites the whole attribute from it. Starting empty would drop any key the resumed run
        # does not happen to set again.
        self._env_args = json.loads(self._hdf5_data_group.attrs.get("env_args", "{}"))
        if env_name is not None:
            self.add_env_args({"env_name": env_name})


def make_resumable(handler_class: type) -> type:
    """The given dataset file handler, with appending bolted on (see ResumeExistingDatasetMixin).

    Derived from whatever class the recorder cfg is carrying rather than hard-coded, so this keeps
    working if the export path is later switched to a different handler (e.g. the async one).
    """
    return type(f"Resumable{handler_class.__name__}", (ResumeExistingDatasetMixin, handler_class), {})


# ─── Tasks whose real target object is the can ────────────────────────────────
# These tasks' own cfgs still describe the ORIGINAL cube (cube_2) -- apply_task_mode() is what
# swaps in the can, along with the per-arm signals and success condition. Recording one of them
# without --task_mode therefore silently produces cube demos: the run looks completely normal,
# and the mistake only surfaces later as a dataset whose object is the wrong shape. Cheaper to
# refuse up front than to discover it after a recording session (see main()).
CAN_TARGET_TASKS = (
    "Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0",
    "Isaac-PickUp-RedCube-OpenArm-CamMount-IK-Abs-v0",
)

# ─── Motion gate: no robot motion before the episode is armed ─────────────────
MOTION_GATE = {"enabled": True}
"""Whether teleop is allowed to move the robot at all, right now.

For the two Quest-button modes (see requires_manual_arm) this is False from every reset until
button X arms the episode, so the robot sits at its rest pose instead of tracking the headset.
Recording and motion are then the same switch: nothing moves before X, and everything that moves
after it is in the episode.

A dict rather than a module-level bool because ROS2JointCommandAction.apply_actions() reads it
from inside the action pipeline, and rebinding a bare global would not be seen through the
`from ... import` style binding this module's own helpers use.

Deliberately replaces the older "let the arm track before X, then throw those steps away" design.
That kept the robot continuously following the operator so arming never caused a jump, but it also
meant the robot moved -- and could knock the can over -- during setup, before any episode existed
to hold that motion. NOTE the trade this makes: because the robot no longer follows the headset
before X, whatever pose the operator's real arms are in at the moment X is pressed is applied in
one step. Align to the rest pose before arming, or the episode opens with a jump.
"""

# ─── Return-to-rest override: the episode's own closing motion ────────────────
RETURN_TO_REST = {"targets": None}
"""Joint position targets that override the live teleop command, or None when not returning.

Set by :class:`ReturnToRestTrajectory` for the closing phase of an episode: after the operator's
second button-X press the robot is driven back to the task's rest pose along a straight line in
joint space, and only then is the episode exported. Those steps go through ``env.step()`` like any
other, so the retreat is part of the recorded demo rather than something that happens to the robot
between episodes (a dataset whose demos all end mid-reach teaches a policy that never retreats).

Shape is the robot's full ``(num_envs, num_joints)`` target vector, so each action term can slice
out the joints it owns. Same dict-not-bare-global reason as MOTION_GATE: it is read from inside
ROS2JointCommandAction.apply_actions().

Only the ROS2 native path needs this side channel -- for every other device the returned action
vector IS the command, so ReturnToRestTrajectory.advance()'s return value is enough on its own.
"""

# ─── Action space layout ──────────────────────────────────────────────────────
# Must match the order fields are inserted into the env's ActionsCfg:
#   arm_action        (left IK,  6D) → indices [0:6]
#   gripper_action    (left bin, 1D) → index  [6]
#   right_arm_action  (right IK, 6D) → indices [7:13]
#   right_gripper_action (right bin, 1D) → index [13]
LEFT_IK_SLICE    = slice(0, 6)
LEFT_GRP_IDX     = 6
RIGHT_IK_SLICE   = slice(7, 13)
RIGHT_GRP_IDX    = 13
TOTAL_ACTION_DIM = 14

# ─── Joint-space action layout (--teleop_device vr_joint_ros2 only) ───────────────
# Same field order as above (arm_action, gripper_action, right_arm_action,
# right_gripper_action), but arm_action/right_arm_action are reassigned to a 7D
# JointPositionActionCfg instead of a 6D IK delta -- see swap_to_joint_position_actions.
LEFT_JOINT_SLICE       = slice(0, 7)
LEFT_JOINT_GRP_IDX     = 7
RIGHT_JOINT_SLICE      = slice(8, 15)
RIGHT_JOINT_GRP_IDX    = 15
TOTAL_ACTION_DIM_JOINT = 16


class VRDualArmTeleop:
    """Bimanual VR teleop device fed by a UDP JSON side-channel from the Dora ROS 2
    bridge (nodes/dora-openarm-ros2-bridge/bridge.py's --vr-udp-port in the
    dora-openarm-data-collection repo).

    Isaac Lab's rclpy can't be imported here directly: this conda env is Python 3.11,
    but ROS 2 Humble's rclpy C extension is only built for the system Python 3.10, so
    the bridge process (which already runs under the matching 3.10 venv to publish the
    ROS topics) also fire-and-forget UDP-broadcasts the same data as JSON -- this class
    is the receiving end, not a ROS node itself.

    Wire format per packet (see bridge.py's VrUdpBroadcaster):
        {"t": float,
         "pose_right": [x,y,z,qw,qx,qy,qz] | null, "pose_left": [...] | null,
         "pose_reference": [...] | null,
         "gripper_right": float | null, "gripper_left": float | null}
    A null field means "no VR data received yet for this field" -- distinct from a
    real zero pose -- so a stale/never-populated arm/gripper is left at its previous
    commanded state rather than snapping to the origin.

    Unlike OpenArmKeyboard (which accumulates small deltas from held keys), the task's
    DifferentialIKControllerCfg is relative-mode (use_relative_mode=True) but the VR
    source gives an ABSOLUTE target pose. So every step this recomputes
    delta = target_pose_from_VR - current_live_EE_pose (in the robot base frame, via
    isaaclab.utils.math.compute_pose_error -- the exact inverse of the IK controller's
    own apply_delta_pose) rather than accumulating anything itself. That makes the arm
    continuously track the VR controller's live absolute pose.

    `quat_offset` (wxyz, default identity) is left-multiplied onto every incoming target
    orientation, in the robot base frame, before the pose error is computed. It exists
    because dora-openarm-vr's quest_receiver.py maps raw controller poses into the robot
    "arm_origin" frame using a _FRAME_ROT/r_fix pair that was hand-tuned against the
    MuJoCo viewer's axis convention -- if Isaac Sim's robot base frame differs from
    MuJoCo's arm_origin convention, this corrects the residual fixed rotation without
    touching the shared dora pipeline (which must keep working for MuJoCo too).

    Both arms are driven every step (true bimanual) -- there is no "active arm"/TAB
    here, unlike OpenArmKeyboard's single-arm scheme.
    """

    # Raw gripper values arrive as dora-openarm-kinematics' trigger-mapped joint command
    # (_map_trigger_to_gripper), read directly off whichever MuJoCo model --xml points the
    # dora `ik` node at (see ik.py's _gripper_endpoints). For the v1_camera model (the one
    # both dora and this v1_camera_isaac robot now use) that's a prismatic finger joint,
    # closed=0.0 m .. open=0.044 m on BOTH sides (unlike the older v2 hinge models, which
    # were signed and in radians -- right [-0.785, 0], left [0, 0.785] -- hence the old
    # GRIPPER_RAW_RANGE=0.785/abs() scheme this replaced; that range is stale for
    # v1_camera and collapsed every raw value into the same "closed" half of cmd's [-1, 1],
    # i.e. the gripper no longer responded to the trigger at all).
    # trigger=0 (released) -> open (0.044), trigger=1 (fully squeezed) -> closed (0.0).
    GRIPPER_OPEN_VAL = 0.044
    GRIPPER_CLOSED_VAL = 0.0

    def __init__(
        self,
        robot,
        udp_host: str,
        udp_port: int,
        sim_device: str,
        max_pos_step: float = 0.01,
        max_rot_step: float = 0.05,
        quat_offset: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ):
        self._robot = robot
        self._sim_device = sim_device
        self._max_pos_step = max_pos_step
        self._max_rot_step = max_rot_step
        self._quat_offset = torch.tensor([list(quat_offset)], dtype=torch.float32, device=sim_device)

        right_ids, right_names = robot.find_bodies("openarm_right_ee_tcp")
        left_ids, left_names = robot.find_bodies("openarm_left_ee_tcp")
        if len(right_ids) != 1 or len(left_ids) != 1:
            raise RuntimeError(
                "VRDualArmTeleop expected exactly one match each for"
                f" 'openarm_right_ee_tcp'/'openarm_left_ee_tcp', got right={right_names},"
                f" left={left_names}."
            )
        self._right_body_idx = right_ids[0]
        self._left_body_idx = left_ids[0]

        self._lock = threading.Lock()
        self._latest: dict = {}
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((udp_host, udp_port))
        self._sock.settimeout(0.5)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        self._additional_callbacks: dict = {}
        self._setup_keyboard()  # R/N/T only -- no WASD, no arm-switch

        print(f"[VR TELEOP] Listening for Dora bridge UDP JSON on {udp_host}:{udp_port}")
        if not torch.allclose(self._quat_offset, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim_device)):
            print(f"[VR TELEOP] Applying quat_offset (wxyz) = {quat_offset} to every incoming target orientation")

    def _run(self):
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                packet = json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            with self._lock:
                self._latest = packet

    def _setup_keyboard(self):
        import carb.input as ci
        import omni.appwindow
        import weakref

        appwindow = omni.appwindow.get_default_app_window()
        self._keyboard = appwindow.get_keyboard()
        self._ci = ci.acquire_input_interface()
        self._sub = self._ci.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *_, obj=weakref.proxy(self): obj._on_key_event(event),
        )

    def _on_key_event(self, event) -> bool:
        import carb.input as ci

        try:
            name = event.input.name
        except AttributeError:
            return True
        if event.type == ci.KeyboardEventType.KEY_PRESS and name in self._additional_callbacks:
            self._additional_callbacks[name]()
        return True

    def __del__(self):
        self._stop.set()
        try:
            self._sock.close()
        except Exception:
            pass
        try:
            self._ci.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
        except Exception:
            pass

    def add_callback(self, key: str, func):
        self._additional_callbacks[key] = func

    def reset(self):
        pass

    def clear_deltas(self):
        pass

    def _gripper_raw_to_cmd(self, raw: float) -> float:
        span = self.GRIPPER_OPEN_VAL - self.GRIPPER_CLOSED_VAL
        cmd = 2.0 * (raw - self.GRIPPER_CLOSED_VAL) / span - 1.0
        return max(-1.0, min(1.0, cmd))

    def _current_ee_pose_b(self, body_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Live EE (pos, quat wxyz) in the robot base/root frame, batch size 1."""
        from isaaclab.utils.math import subtract_frame_transforms

        ee_pos_w = self._robot.data.body_pos_w[:1, body_idx]
        ee_quat_w = self._robot.data.body_quat_w[:1, body_idx]
        root_pos_w = self._robot.data.root_pos_w[:1]
        root_quat_w = self._robot.data.root_quat_w[:1]
        return subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    def _arm_delta(self, body_idx: int, target_pose: list | None) -> tuple[torch.Tensor, bool]:
        """Returns (delta6, valid) for one arm -- delta6 is [dx,dy,dz, axis-angle rx,ry,rz]."""
        from isaaclab.utils.math import compute_pose_error, quat_mul

        if target_pose is None:
            return torch.zeros(6, dtype=torch.float32, device=self._sim_device), False

        ee_pos_b, ee_quat_b = self._current_ee_pose_b(body_idx)
        target_pos_b = torch.tensor([target_pose[0:3]], dtype=torch.float32, device=self._sim_device)
        target_quat_b = torch.tensor(
            [[target_pose[3], target_pose[4], target_pose[5], target_pose[6]]],
            dtype=torch.float32,
            device=self._sim_device,
        )
        target_quat_b = quat_mul(self._quat_offset, target_quat_b)

        pos_err, rot_err = compute_pose_error(
            ee_pos_b, ee_quat_b, target_pos_b, target_quat_b, rot_error_type="axis_angle"
        )
        pos_err = torch.clamp(pos_err[0], -self._max_pos_step, self._max_pos_step)
        rot_err = torch.clamp(rot_err[0], -self._max_rot_step, self._max_rot_step)

        return torch.cat([pos_err, rot_err]), True

    def build_dual_action(
        self, left_gripper_state: float, right_gripper_state: float
    ) -> tuple[torch.Tensor, float, float]:
        """Build the full 14D action vector directly from the latest VR packet, driving
        both arms simultaneously. Any arm/gripper missing fresh VR data keeps its
        previous commanded state (zero IK delta -- i.e. hold position -- for a missing
        arm; unchanged binary command for a missing gripper).
        """
        with self._lock:
            packet = dict(self._latest)

        left_delta, left_valid = self._arm_delta(self._left_body_idx, packet.get("pose_left"))
        right_delta, right_valid = self._arm_delta(self._right_body_idx, packet.get("pose_right"))

        gripper_left_raw = packet.get("gripper_left")
        gripper_right_raw = packet.get("gripper_right")
        if gripper_left_raw is not None:
            left_gripper_state = self._gripper_raw_to_cmd(gripper_left_raw)
        if gripper_right_raw is not None:
            right_gripper_state = self._gripper_raw_to_cmd(gripper_right_raw)

        full = torch.zeros(TOTAL_ACTION_DIM, dtype=torch.float32, device=self._sim_device)
        if left_valid:
            full[LEFT_IK_SLICE] = left_delta
        full[LEFT_GRP_IDX] = left_gripper_state
        if right_valid:
            full[RIGHT_IK_SLICE] = right_delta
        full[RIGHT_GRP_IDX] = right_gripper_state

        return full, left_gripper_state, right_gripper_state


# Regex joint_names passed to BOTH the JointPositionActionCfg swap (see
# swap_to_joint_position_actions) and VRDualArmJointTeleop's own robot.find_joints call --
# using the exact same list in both places (with find_joints' shared default
# preserve_order=False) guarantees identical joint ordering without depending on any
# assumption about how resolve_matching_names breaks ties.
LEFT_ARM_JOINT_REGEX = ["openarm_left_joint[1-7]"]
RIGHT_ARM_JOINT_REGEX = ["openarm_right_joint[1-7]"]
JOINT_ACTION_SCALE = 1.0

# Real robot gripper joint names (isaaclab_assets/robots/openarm.py) -- must match
# dora-openarm-ros2-bridge/joint_command_processor.py's LEFT_GRIPPER_JOINT_NAME /
# RIGHT_GRIPPER_JOINT_NAME on the dora side, since that's what labels the entries in the
# UDP packet this teleop device receives.
LEFT_GRIPPER_JOINT_NAME = "openarm_left_finger_joint1"
RIGHT_GRIPPER_JOINT_NAME = "openarm_right_finger_joint1"

# The second jaw of each gripper. Never published over ROS2 (on hardware it is coupled to
# joint1 mechanically), but in sim it is a real actuated joint that has to be commanded
# alongside joint1 or it stays shut -- see openarm.py's "openarm_gripper" actuator comment
# and ROS2JointCommandActionCfg.mimic_joints.
LEFT_GRIPPER_FOLLOWER_JOINT_NAME = "openarm_left_finger_joint2"
RIGHT_GRIPPER_FOLLOWER_JOINT_NAME = "openarm_right_finger_joint2"

# ── Gripper command quantisation (vr_joint_ros2_native only) ──────────────────
# The value dora publishes for a finger joint is the VR trigger's ANALOG travel
# (ik.py's _map_trigger_to_gripper: trigger 0 -> 0.044 m open, 1 -> 0.0 m closed).
# Everything downstream of this script treats the gripper as a two-state command
# instead: the task's own BinaryJointPositionActionCfg, Mimic's +-1 convention
# (openarm_pickup_ik_abs_mimic_env.actions_to_gripper_actions), and
# JointMirrorBroadcaster's open/close target. An operator who squeezes the trigger
# all the way (as ours does) already produces a clean 0.0/0.044 column apart from
# the one or two frames each transition passes through -- but a squeeze held short
# of the end stop records a half-closed target that replay_demos.py faithfully
# reproduces as a gripper that visibly never shuts, and that Mimic maps to a
# non-canonical value (0.0425 -> +0.93) instead of a clean +-1.
#
# Snapping to the two endpoints happens on the way IN, at every point that reads a
# gripper value out of the decoded ROS2 message -- both the term that ACTUATES the
# joint (ROS2JointCommandAction.apply_actions) and the one that LOGS it
# (ROS2NativeJointTeleop.build_dual_action), through this one function so the two
# cannot disagree. Snapping only the logged value would write an action the robot
# was never given, contradicting the recorded (measured) states.
#
# Hysteresis rather than a single midpoint threshold: a trigger held near the
# midpoint would otherwise flip the fingers fully open/closed from step to step.
#
# Same two endpoints as VRDualArmTeleop.GRIPPER_OPEN_VAL/GRIPPER_CLOSED_VAL (which
# document where the raw values come from) and JointMirrorBroadcaster's -- keep in sync.
GRIPPER_OPEN_VAL = 0.044
GRIPPER_CLOSED_VAL = 0.0
GRIPPER_SNAP_CLOSE_BELOW = 0.35 * GRIPPER_OPEN_VAL
GRIPPER_SNAP_OPEN_ABOVE = 0.65 * GRIPPER_OPEN_VAL

# Last snapped value per gripper joint name, i.e. the hysteresis state. Module-level
# because the two call sites live in different objects and run at different rates
# (apply_actions every physics substep, build_dual_action once per env step) but must
# share one state machine. Keyed by the LEADER joint name, so a follower jaw resolved
# from its leader lands on the same state.
_GRIPPER_SNAP_STATE: dict[str, float] = {}

GRIPPER_JOINT_NAMES = (LEFT_GRIPPER_JOINT_NAME, RIGHT_GRIPPER_JOINT_NAME)


def snap_gripper_command(joint_name: str, raw: float) -> float:
    """Quantise one raw gripper command (m) to fully open / fully closed.

    Returns the previous state for a value inside the hysteresis band, defaulting to
    open -- the safe state for an unarmed episode, since a gripper that starts closed
    can clamp onto whatever it is lowered around.
    """
    if raw <= GRIPPER_SNAP_CLOSE_BELOW:
        state = GRIPPER_CLOSED_VAL
    elif raw >= GRIPPER_SNAP_OPEN_ABOVE:
        state = GRIPPER_OPEN_VAL
    else:
        state = _GRIPPER_SNAP_STATE.get(joint_name, GRIPPER_OPEN_VAL)
    _GRIPPER_SNAP_STATE[joint_name] = state
    return state


def swap_to_joint_position_actions(env_cfg) -> None:
    """Reassign env_cfg.actions.arm_action/right_arm_action from the task's default IK
    (DifferentialInverseKinematicsActionCfg, 6D Cartesian delta) to a direct
    JointPositionActionCfg (7D joint target), for --teleop_device vr_joint_ros2. Must run
    on the parsed cfg BEFORE gym.make() -- ActionsCfg field ORDER (arm_action,
    gripper_action, right_arm_action, right_gripper_action) is fixed by the class's
    declared field order and unaffected by reassigning a field's value, only each
    reassigned field's WIDTH changes (6D -> 7D per arm), which is why the resulting
    action vector grows from TOTAL_ACTION_DIM (14) to TOTAL_ACTION_DIM_JOINT (16).
    gripper_action/right_gripper_action are left untouched -- already
    BinaryJointPositionActionCfg, not IK-based, so nothing to swap there.

    Trades away needing a Cartesian frame correction (--vr_quat_offset, see
    VRDualArmTeleop) -- the incoming data is already in joint space, so there's no
    MuJoCo-vs-Isaac-Sim base-frame convention to get wrong.
    """
    if not hasattr(env_cfg.actions, "right_arm_action"):
        raise RuntimeError(
            "--teleop_device vr_joint_ros2 requires a dual-arm task (env_cfg.actions has no"
            " right_arm_action to swap)."
        )
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEFT_ARM_JOINT_REGEX, scale=JOINT_ACTION_SCALE, use_default_offset=True
    )
    env_cfg.actions.right_arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=RIGHT_ARM_JOINT_REGEX, scale=JOINT_ACTION_SCALE, use_default_offset=True
    )


class ROS2JointCommandAction(ActionTerm):
    """Action term that applies joint position targets read straight out of a ROS2
    OmniGraph's ROS2SubscribeJointState node (see build_ros2_joint_command_graph),
    instead of whatever env.step() was called with.

    This exists because a true no-op action term doesn't actually free these joints up
    for something else (e.g. an IsaacArticulationController OG node) to drive: IsaacLab's
    implicit actuator model recomputes and re-applies a PD target for EVERY joint on
    EVERY Articulation.write_data_to_sim() call regardless of which (if any) action term
    "owns" it -- actuators are wired up per-asset (ArticulationCfg.actuators), completely
    independent of the env's action pipeline. With a real no-op, that per-step
    write_data_to_sim() call keeps re-asserting these joints' STALE cached
    joint_pos_target (left at whatever it was after the last reset, e.g. the default
    pose) via set_dof_position_targets(..., ALL_INDICES) -- fighting an
    IsaacArticulationController node writing the same joints from the other side, with
    whichever happens to run later in a given physics substep winning. In practice this
    made the arms hold near their default pose rather than track ROS2 commands at all.

    Reading the OmniGraph's subscriber output directly in apply_actions() and pushing it
    through the normal set_joint_position_target()/actuator/write_data_to_sim() path
    instead makes IsaacLab's own pipeline the SOLE writer of these joints' targets again
    -- eliminating the race -- while the OmniGraph is still only used for what it's
    needed for: decoding ROS2 messages without importing rclpy into this Python 3.11
    process (see VRDualArmTeleop's docstring for why that doesn't work directly).

    process_actions() ignores whatever env.step() was called with entirely -- the actual
    source of truth is read fresh from the OmniGraph in apply_actions() (called once per
    *simulation* step, i.e. every physics substep, so it's always using the latest
    decoded ROS2 message).
    """

    cfg: "ROS2JointCommandActionCfg"

    def __init__(self, cfg: "ROS2JointCommandActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        self._raw_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._default_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._subscriber_path = f"{cfg.graph_path}/SubscriberJointState"

        # Followers are driven but excluded from action_dim -- see the cfg's docstring.
        follower_names = list(cfg.follower_joints)
        if follower_names:
            self._follower_ids, resolved = self._asset.find_joints(follower_names, preserve_order=True)
            self._follower_sources = [cfg.follower_joints[n] for n in resolved]
            self._follower_default_pos = self._asset.data.default_joint_pos[:, self._follower_ids].clone()
        else:
            self._follower_ids, self._follower_sources = [], []

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        # Not used for control (see class docstring) -- stashed only so it still shows up
        # as the recorded episode's "action" the same way every other action term's does.
        self._raw_actions[:] = actions

    def apply_actions(self):
        import omni.graph.core as og

        # The episode's closing return-to-rest ramp outranks everything below, including `locked`:
        # the trajectory is computed over the WHOLE robot (a locked arm's segment is simply the
        # constant rest pose), so honouring it here needs no per-arm special case. The operator's
        # headset keeps publishing throughout -- this is what stops it being listened to.
        override = RETURN_TO_REST["targets"]
        if override is not None:
            self._asset.set_joint_position_target(override[:, self._joint_ids], joint_ids=self._joint_ids)
            if self._follower_ids:
                self._asset.set_joint_position_target(
                    override[:, self._follower_ids], joint_ids=self._follower_ids
                )
            return

        # Two independent reasons to ignore the ROS2 command and pin these joints to their rest
        # pose: the arm is locked out for the whole run by --task_mode (see CONTROLLED_ARMS), or
        # the episode has not been armed yet (button X -- see MOTION_GATE).
        #
        # Actively re-asserting the default target each step, rather than just returning early, is
        # what actually holds the joints still: the operator's headset keeps publishing targets for
        # both arms regardless, and an early return would leave the last commanded target standing.
        if self.cfg.locked or not MOTION_GATE["enabled"]:
            self._asset.set_joint_position_target(self._default_joint_pos, joint_ids=self._joint_ids)
            if self._follower_ids:
                self._asset.set_joint_position_target(self._follower_default_pos, joint_ids=self._follower_ids)
            return

        try:
            names = og.Controller.attribute(f"{self._subscriber_path}.outputs:jointNames").get()
            positions = og.Controller.attribute(f"{self._subscriber_path}.outputs:positionCommand").get()
        except Exception:
            return  # graph/attribute not ready yet (e.g. very first frame) -- try again next step
        if not names or positions is None or len(positions) == 0:
            return  # no ROS2 message decoded yet -- leave joints at whatever they're already at

        by_name = dict(zip(names, positions))
        target = self._default_joint_pos.clone()
        for i, joint_name in enumerate(self._joint_names):
            if joint_name in by_name:
                # A gripper is driven to one of its two endpoints, never to the trigger's
                # analog position -- see snap_gripper_command.
                if joint_name in GRIPPER_JOINT_NAMES:
                    target[:, i] = snap_gripper_command(joint_name, float(by_name[joint_name]))
                else:
                    target[:, i] = by_name[joint_name]
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)

        # Follower joints (the grippers' *_finger_joint2). The dora/ROS2 side never
        # publishes them -- on hardware that jaw is mechanically coupled to joint1 -- but
        # in sim it is a genuine actuated joint (see openarm.py's "openarm_gripper"
        # comment). Left uncommanded it would sit at its default target, i.e. clamped
        # shut, while joint1 opens. Copy the leader's commanded value onto it.
        if self._follower_ids:
            f_target = self._follower_default_pos.clone()
            for i, source in enumerate(self._follower_sources):
                if source in by_name:
                    # Same snap as the leader (and keyed by the leader's name), or the two
                    # jaws would be commanded to different positions.
                    f_target[:, i] = (
                        snap_gripper_command(source, float(by_name[source]))
                        if source in GRIPPER_JOINT_NAMES
                        else by_name[source]
                    )
            self._asset.set_joint_position_target(f_target, joint_ids=self._follower_ids)


@configclass
class ROS2JointCommandActionCfg(ActionTermCfg):
    """Configuration for :class:`ROS2JointCommandAction`."""

    class_type: type[ActionTerm] = ROS2JointCommandAction
    joint_names: list[str] = MISSING
    """Regex(es) selecting which of the ROS2 message's joints this field owns (e.g. the
    left arm's 7 joints, or just the left gripper's 1) -- matched the same way as any
    other joint action term's joint_names."""
    follower_joints: dict[str, str] = {}
    """Extra joints to drive alongside :attr:`joint_names`, mapped to the joint whose
    commanded value they copy, e.g.
    ``{"openarm_left_finger_joint2": "openarm_left_finger_joint1"}``.

    These are deliberately NOT part of :attr:`joint_names`: they must not widen
    :attr:`action_dim`, because the recorded action vector is fixed at
    TOTAL_ACTION_DIM_JOINT (16). See :meth:`ROS2JointCommandAction.apply_actions`."""
    graph_path: str = "/Graph/ROS_JointCommand"
    """Must match build_ros2_joint_command_graph's graph_path."""
    locked: bool = False
    """Ignore the incoming ROS2 command and hold these joints at their rest pose.

    Set by --task_mode for the arm the mode says must not move (see CONTROLLED_ARMS). The width
    of this term -- and therefore of the recorded action vector -- is unaffected: a locked arm
    still contributes its columns to every recorded row, it just contributes a constant."""


def swap_to_native_ros2_joint_actions(env_cfg) -> None:
    """Reassign all four of env_cfg.actions' fields (arm_action, gripper_action,
    right_arm_action, right_gripper_action) to :class:`ROS2JointCommandActionCfg`, for
    --teleop_device vr_joint_ros2_native. Must run on the parsed cfg BEFORE gym.make(),
    same as swap_to_joint_position_actions -- ActionsCfg field order/widths are fixed at
    that point, and the resulting action vector is TOTAL_ACTION_DIM_JOINT (16) wide, same
    as vr_joint_ros2, purely so downstream code (recorder, dataset shape) doesn't need a
    third case.

    Unlike swap_to_joint_position_actions, the values env.step() is called with are
    irrelevant to control here -- see ROS2JointCommandAction's docstring for where the
    actual targets come from and why.
    """
    if not hasattr(env_cfg.actions, "right_arm_action"):
        raise RuntimeError(
            "--teleop_device vr_joint_ros2_native requires a dual-arm task (env_cfg.actions has"
            " no right_arm_action to swap)."
        )
    env_cfg.actions.arm_action = ROS2JointCommandActionCfg(asset_name="robot", joint_names=LEFT_ARM_JOINT_REGEX)
    env_cfg.actions.gripper_action = ROS2JointCommandActionCfg(
        asset_name="robot",
        joint_names=[LEFT_GRIPPER_JOINT_NAME],
        follower_joints={LEFT_GRIPPER_FOLLOWER_JOINT_NAME: LEFT_GRIPPER_JOINT_NAME},
    )
    env_cfg.actions.right_arm_action = ROS2JointCommandActionCfg(
        asset_name="robot", joint_names=RIGHT_ARM_JOINT_REGEX
    )
    env_cfg.actions.right_gripper_action = ROS2JointCommandActionCfg(
        asset_name="robot",
        joint_names=[RIGHT_GRIPPER_JOINT_NAME],
        follower_joints={RIGHT_GRIPPER_FOLLOWER_JOINT_NAME: RIGHT_GRIPPER_JOINT_NAME},
    )


def build_ros2_joint_command_graph(
    topic_name: str,
    domain_id: int,
    graph_path: str = "/Graph/ROS_JointCommand",
) -> None:
    """Build a ROS2 OmniGraph that decodes `topic_name` (a sensor_msgs/JointState, e.g.
    dora-openarm-ros2-bridge/joint_command_processor.py's
    /openarm/vr_joint_command_processed) over native ROS2 DDS every simulation frame.

    Deliberately does NOT include an IsaacArticulationController node -- see
    ROS2JointCommandAction's docstring for why writing PhysX joint targets directly from
    the graph races with IsaacLab's own actuator model. This graph only decodes the
    message into OmniGraph attributes (outputs:jointNames/positionCommand on
    SubscriberJointState); ROS2JointCommandAction.apply_actions() reads those back out
    each simulation step and applies them the normal IsaacLab way.

        OnPlaybackTick ─tick─┬──────────────────────────┐
                             ▼                           ▼
        ROS2Context ──context──▶ ROS2SubscribeJointState (outputs read directly by
                                  topicName=`topic_name`   ROS2JointCommandAction, not
                                                            wired to anything downstream
                                                            in this graph)

    Must run AFTER the robot has been spawned (i.e. after gym.make()/env creation) purely
    for consistency with when the rest of this script sets things up -- unlike the
    IsaacArticulationController version this replaced, nothing here actually depends on
    the robot prim existing yet.
    """
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.ros2.bridge")

    import omni.graph.core as og

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("SubscriberJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "SubscriberJointState.inputs:execIn"),
                ("Context.outputs:context", "SubscriberJointState.inputs:context"),
            ],
            keys.SET_VALUES: [
                ("Context.inputs:domain_id", domain_id),
                # ROS2Context's useDomainIDEnvVar defaults to True ("use ROS_DOMAIN_ID env var
                # if set, ignoring domain_id"). This process's shell/conda env has no reason to
                # have ROS_DOMAIN_ID set to anything in particular, so left at its default this
                # would non-deterministically either match or silently diverge from whatever
                # domain_id happens to be picked -- turned off so --ros2_domain_id is always the
                # one source of truth, matching dora's ROS_DOMAIN_ID (dataflow-vr-mujoco-ros2.yaml
                # sets it to "1" for ros2-bridge/joint-command-processor).
                ("Context.inputs:useDomainIDEnvVar", False),
                ("SubscriberJointState.inputs:topicName", topic_name),
            ],
        },
    )
    print(f"[ROS2 NATIVE TELEOP] Built {graph_path}: decoding '{topic_name}' (domain_id={domain_id}).")


class VRDualArmJointTeleop:
    """Bimanual JOINT-SPACE VR teleop device fed by a UDP JSON side-channel from
    dora-openarm-ros2-bridge/joint_command_processor.py's --vr-joint-udp-port (in the
    dora-openarm-data-collection repo).

    Unlike VRDualArmTeleop (which targets a Cartesian EE pose and drives the task's IK
    action), this targets JOINT ANGLES directly -- see swap_to_joint_position_actions,
    which main() calls before gym.make() to reassign arm_action/right_arm_action to a
    JointPositionActionCfg so those targets apply with no IK/Jacobian step and no
    Cartesian base-frame convention to calibrate against MuJoCo (contrast
    --vr_quat_offset in the pose-based path).

    Wire format per packet (see joint_command_processor.py's VrJointUdpBroadcaster):
        {"t": float, "name": [str, ...], "position": [float, ...],
         "button_x": bool, "button_y": bool}
    `name`/`position` are the same 16-entry arrays published on the ROS 2 topic
    /openarm/vr_joint_command_processed: left_joint1..7, left gripper joint,
    right_joint1..7, right gripper joint -- passed through unchanged by that node (no
    joint6/joint7 swap anymore: the dora `ik` node now solves directly against the
    v1_camera MuJoCo model, whose joint6/joint7 axes already match this v1_camera_isaac
    robot's, so the old per-arm swap+negate hack -- needed only when `ik` solved against
    the v2 model -- was removed on the dora side and must NOT be reintroduced here). A
    joint name missing from the latest packet holds its last commanded target
    (initialized to the robot's default_joint_pos) rather than snapping to zero; a
    missing gripper name keeps its previous binary command.

    `button_x`/`button_y` are the Quest controller's X/Y buttons, forwarded from
    dora's quest_receiver.py through joint_command_processor.py. build_dual_action()
    watches for a rising edge (False -> True) on each and fires a callback -- X calls
    main()'s start_recording() (arms recording; nothing is saved before this press,
    see requires_manual_arm), Y calls reset_episode() (same as the R key: discard &
    reset). See main()'s teleop.add_callback("X_START", ...) / ("R", ...) wiring, which
    populates self._additional_callbacks the exact same way a keypress would.
    """

    def __init__(
        self,
        robot,
        udp_host: str,
        udp_port: int,
        sim_device: str,
    ):
        self._robot = robot
        self._sim_device = sim_device

        self._left_joint_ids, self._left_joint_names = robot.find_joints(LEFT_ARM_JOINT_REGEX)
        self._right_joint_ids, self._right_joint_names = robot.find_joints(RIGHT_ARM_JOINT_REGEX)
        if len(self._left_joint_ids) != 7 or len(self._right_joint_ids) != 7:
            raise RuntimeError(
                "VRDualArmJointTeleop expected 7 joints per arm, got"
                f" left={self._left_joint_names}, right={self._right_joint_names}."
            )

        self._left_default = robot.data.default_joint_pos[0, self._left_joint_ids].clone()
        self._right_default = robot.data.default_joint_pos[0, self._right_joint_ids].clone()
        # Last commanded ABSOLUTE joint target per arm -- what a missing/stale packet holds at.
        self._left_target = self._left_default.clone()
        self._right_target = self._right_default.clone()

        self._lock = threading.Lock()
        self._latest: dict = {}
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((udp_host, udp_port))
        self._sock.settimeout(0.5)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        self._additional_callbacks: dict = {}
        self._prev_button_x = False
        self._prev_button_y = False
        self._setup_keyboard()  # R/N/T only -- no WASD, no arm-switch

        print(f"[VR JOINT TELEOP] Listening for Dora bridge UDP JSON on {udp_host}:{udp_port}")

    def _run(self):
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                packet = json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            with self._lock:
                self._latest = packet

    def _setup_keyboard(self):
        import carb.input as ci
        import omni.appwindow
        import weakref

        appwindow = omni.appwindow.get_default_app_window()
        self._keyboard = appwindow.get_keyboard()
        self._ci = ci.acquire_input_interface()
        self._sub = self._ci.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *_, obj=weakref.proxy(self): obj._on_key_event(event),
        )

    def _on_key_event(self, event) -> bool:
        import carb.input as ci

        try:
            name = event.input.name
        except AttributeError:
            return True
        if event.type == ci.KeyboardEventType.KEY_PRESS and name in self._additional_callbacks:
            self._additional_callbacks[name]()
        return True

    def __del__(self):
        self._stop.set()
        try:
            self._sock.close()
        except Exception:
            pass
        try:
            self._ci.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
        except Exception:
            pass

    def add_callback(self, key: str, func):
        self._additional_callbacks[key] = func

    def _gripper_raw_to_cmd(self, raw: float) -> float:
        # Same raw-value convention as VRDualArmTeleop -- see its GRIPPER_OPEN_VAL/
        # GRIPPER_CLOSED_VAL docstring.
        span = VRDualArmTeleop.GRIPPER_OPEN_VAL - VRDualArmTeleop.GRIPPER_CLOSED_VAL
        cmd = 2.0 * (raw - VRDualArmTeleop.GRIPPER_CLOSED_VAL) / span - 1.0
        return max(-1.0, min(1.0, cmd))

    def reset(self):
        # Re-sync targets to whatever the env just reset the robot to, so the first
        # post-reset action doesn't drag the arm from the *previous* episode's pose.
        self._left_target = self._robot.data.joint_pos[0, self._left_joint_ids].clone()
        self._right_target = self._robot.data.joint_pos[0, self._right_joint_ids].clone()

    def clear_deltas(self):
        pass

    def build_dual_action(
        self, left_gripper_state: float, right_gripper_state: float
    ) -> tuple[torch.Tensor, float, float]:
        """Build the full 16D action vector directly from the latest VR joint-command
        packet. A joint missing from the packet holds its last commanded target; a
        missing gripper keeps its previous binary command.
        """
        with self._lock:
            packet = dict(self._latest)

        button_x = bool(packet.get("button_x", False))
        button_y = bool(packet.get("button_y", False))
        if button_x and not self._prev_button_x:
            start_cb = self._additional_callbacks.get("X_START")
            if start_cb is not None:
                print("[VR JOINT TELEOP] X pressed -> start recording")
                start_cb()
        if button_y and not self._prev_button_y:
            reset_cb = self._additional_callbacks.get("R")
            if reset_cb is not None:
                print("[VR JOINT TELEOP] Y pressed -> discard & reset episode")
                reset_cb()
        self._prev_button_x = button_x
        self._prev_button_y = button_y

        by_name = dict(zip(packet.get("name", []), packet.get("position", [])))

        for i, name in enumerate(self._left_joint_names):
            if name in by_name:
                self._left_target[i] = by_name[name]
        for i, name in enumerate(self._right_joint_names):
            if name in by_name:
                self._right_target[i] = by_name[name]

        if LEFT_GRIPPER_JOINT_NAME in by_name:
            left_gripper_state = self._gripper_raw_to_cmd(by_name[LEFT_GRIPPER_JOINT_NAME])
        if RIGHT_GRIPPER_JOINT_NAME in by_name:
            right_gripper_state = self._gripper_raw_to_cmd(by_name[RIGHT_GRIPPER_JOINT_NAME])

        full = torch.zeros(TOTAL_ACTION_DIM_JOINT, dtype=torch.float32, device=self._sim_device)
        full[LEFT_JOINT_SLICE] = (self._left_target - self._left_default) / JOINT_ACTION_SCALE
        full[LEFT_JOINT_GRP_IDX] = left_gripper_state
        full[RIGHT_JOINT_SLICE] = (self._right_target - self._right_default) / JOINT_ACTION_SCALE
        full[RIGHT_JOINT_GRP_IDX] = right_gripper_state

        return full, left_gripper_state, right_gripper_state


class ROS2NativeJointTeleop:
    """Bimanual JOINT-SPACE teleop for --teleop_device vr_joint_ros2_native.

    Unlike VRDualArmJointTeleop, this class does not drive the robot at all -- joint
    targets are written directly by the ROS2 OmniGraph built in
    build_ros2_joint_command_graph, which subscribes to
    dora-openarm-ros2-bridge/joint_command_processor.py's
    /openarm/vr_joint_command_processed topic over native ROS2 DDS (no UDP JSON hop, no
    Python-side socket polling). env_cfg.actions' four fields are reassigned to
    NoOpActionCfg by swap_to_native_ros2_joint_actions so IsaacLab's own action pipeline
    never writes a competing target for the same joints.

    This class's jobs: keep the R/N/T keyboard shortcuts working (main() wires them the
    same way as every other teleop device), report each step's COMMANDED joint positions
    as build_dual_action's return value (that's what the recorder logs as the episode's
    "action" -- see _read_command_map/build_dual_action for why it must be the command
    and not the resulting measured pose), and watch for the Quest X/Y buttons.

    The X/Y buttons ride along on the SAME /openarm/vr_joint_command_processed message
    the OmniGraph already decodes for joint targets -- joint_command_processor.py
    appends two extra "joint" entries named "button_x"/"button_y" (position 1.0/0.0 for
    pressed/released) after the real arm/gripper joints. ROS2JointCommandAction's action
    terms only ever look up their OWN joint_names in that name/position map (see its
    apply_actions()), so these two extra entries are silently ignored there -- but this
    class reads the same graph_path/SubscriberJointState outputs independently, purely
    to pull "button_x"/"button_y" back out and fire a callback on a rising edge -- X
    calls main()'s start_recording() (arms recording; nothing is saved before this
    press, see requires_manual_arm), Y calls reset_episode() (same as the R key:
    discard & reset). No second OmniGraph subscriber needed.
    """

    def __init__(
        self,
        robot,
        sim_device: str,
        graph_path: str = "/Graph/ROS_JointCommand",
        locked_arms: tuple[str, ...] = (),
    ):
        self._robot = robot
        self._sim_device = sim_device
        self._subscriber_path = f"{graph_path}/SubscriberJointState"
        self._prev_button_x = False
        self._prev_button_y = False
        self._locked_arms = locked_arms

        self._left_joint_ids, self._left_joint_names = robot.find_joints(LEFT_ARM_JOINT_REGEX)
        self._right_joint_ids, self._right_joint_names = robot.find_joints(RIGHT_ARM_JOINT_REGEX)
        if len(self._left_joint_ids) != 7 or len(self._right_joint_ids) != 7:
            raise RuntimeError(
                "ROS2NativeJointTeleop expected 7 joints per arm, got"
                f" left={self._left_joint_names}, right={self._right_joint_names}."
            )
        left_gripper_ids, _ = robot.find_joints([LEFT_GRIPPER_JOINT_NAME])
        right_gripper_ids, _ = robot.find_joints([RIGHT_GRIPPER_JOINT_NAME])
        if len(left_gripper_ids) != 1 or len(right_gripper_ids) != 1:
            raise RuntimeError(
                "ROS2NativeJointTeleop expected exactly one match each for"
                f" '{LEFT_GRIPPER_JOINT_NAME}'/'{RIGHT_GRIPPER_JOINT_NAME}', got"
                f" left={left_gripper_ids}, right={right_gripper_ids}."
            )
        self._left_gripper_id = left_gripper_ids[0]
        self._right_gripper_id = right_gripper_ids[0]

        self._additional_callbacks: dict = {}
        self._setup_keyboard()  # R/N/T only -- no WASD, no arm-switch

        print("[ROS2 NATIVE TELEOP] Joint targets are driven directly by a ROS2 OmniGraph -- this"
              " process only logs state and services R/N/T keyboard shortcuts.")

    def _setup_keyboard(self):
        import carb.input as ci
        import omni.appwindow
        import weakref

        appwindow = omni.appwindow.get_default_app_window()
        self._keyboard = appwindow.get_keyboard()
        self._ci = ci.acquire_input_interface()
        self._sub = self._ci.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *_, obj=weakref.proxy(self): obj._on_key_event(event),
        )

    def _on_key_event(self, event) -> bool:
        import carb.input as ci

        try:
            name = event.input.name
        except AttributeError:
            return True
        if event.type == ci.KeyboardEventType.KEY_PRESS and name in self._additional_callbacks:
            self._additional_callbacks[name]()
        return True

    def __del__(self):
        try:
            self._ci.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
        except Exception:
            pass

    def add_callback(self, key: str, func):
        self._additional_callbacks[key] = func

    def reset(self):
        pass

    def clear_deltas(self):
        pass

    def _read_command_map(self) -> dict | None:
        """The latest ROS2 joint command the OmniGraph decoded, as {joint name: position}.

        These are the exact same two attributes ROS2JointCommandAction.apply_actions()
        reads to actually drive the joints; reading them a second time here is what lets
        build_dual_action log the COMMAND that was applied rather than the pose that
        resulted from it.

        Returns None whenever there is nothing to read (graph not built yet, or no ROS2
        message decoded so far) -- same "leave everything as it is and try again next
        step" contract apply_actions() uses.
        """
        import omni.graph.core as og

        try:
            names = og.Controller.attribute(f"{self._subscriber_path}.outputs:jointNames").get()
            positions = og.Controller.attribute(f"{self._subscriber_path}.outputs:positionCommand").get()
        except Exception:
            return None  # graph/attribute not ready yet -- try again next step
        if not names or positions is None or len(positions) == 0:
            return None  # no ROS2 message decoded yet
        return dict(zip(names, positions))

    def _poll_buttons(self, by_name: dict | None) -> None:
        if by_name is None:
            return

        button_x = by_name.get("button_x", 0.0) > 0.5
        button_y = by_name.get("button_y", 0.0) > 0.5

        if button_x and not self._prev_button_x:
            start_cb = self._additional_callbacks.get("X_START")
            if start_cb is not None:
                print("[ROS2 NATIVE TELEOP] X pressed -> start recording")
                start_cb()
        if button_y and not self._prev_button_y:
            reset_cb = self._additional_callbacks.get("R")
            if reset_cb is not None:
                print("[ROS2 NATIVE TELEOP] Y pressed -> discard & reset episode")
                reset_cb()
        self._prev_button_x = button_x
        self._prev_button_y = button_y

    def build_dual_action(
        self, left_gripper_state: float, right_gripper_state: float
    ) -> tuple[torch.Tensor, float, float]:
        """Ignores left_gripper_state/right_gripper_state (kept only so the call
        signature matches VRDualArmTeleop/VRDualArmJointTeleop) and instead returns the
        joint targets the OmniGraph is driving the robot with this step -- real actuation
        for this mode happens out-of-band through the graph, not through anything
        computed here. Also polls the same decoded message for the button_x/button_y
        "joint" entries and fires the save/discard callbacks on a rising edge -- see
        class docstring.

        Logging the COMMAND rather than the robot's measured joint positions is what
        makes these episodes replayable (replay_demos.py --action_mode openarm_joint_abs
        feeds each recorded row straight back in as an absolute joint target). Measured
        positions were recorded here originally, and for the arms that only cost a step
        of tracking lag -- but for the grippers it silently destroyed every grasp in the
        dataset. Closing on the cube blocks the fingers at the cube's surface, so the
        measured value stalls there (e.g. 0.0325 m) while the command keeps going to
        ~0 m; it is precisely that command/measured gap that the implicit actuator turns
        into grip force. Replaying the stalled measured value commands zero error, hence
        zero squeeze, and the cube slips out of a gripper that never visibly closes.

        The gripper columns are additionally quantised to fully open / fully closed by
        snap_gripper_command -- the same call ROS2JointCommandAction.apply_actions makes
        on the same shared state before driving the joint, so what is logged stays what
        was applied. Without it a half-squeezed trigger recorded a half-closed target
        (see that function's comment).

        Any joint the decoded message doesn't carry -- and every step before the first
        message arrives -- falls back to the measured position, which is the closest
        thing to "what it was told to do" available at that point. In practice recording
        can only be armed by the button_x entry of that same message, so the fallback
        only ever covers pre-arming steps that are discarded anyway.
        """
        by_name = self._read_command_map()
        self._poll_buttons(by_name)

        full = torch.zeros(TOTAL_ACTION_DIM_JOINT, dtype=torch.float32, device=self._sim_device)
        full[LEFT_JOINT_SLICE] = self._robot.data.joint_pos[0, self._left_joint_ids]
        full[LEFT_JOINT_GRP_IDX] = self._robot.data.joint_pos[0, self._left_gripper_id]
        full[RIGHT_JOINT_SLICE] = self._robot.data.joint_pos[0, self._right_joint_ids]
        full[RIGHT_JOINT_GRP_IDX] = self._robot.data.joint_pos[0, self._right_gripper_id]

        if by_name is not None:
            # Column order within each arm's slice is this class's find_joints() order,
            # which replay_demos.py's JointPositionActionCfg over the same regex resolves
            # identically -- so index by name, never by the message's own ordering.
            for offset, joint_name in enumerate(self._left_joint_names):
                if joint_name in by_name:
                    full[LEFT_JOINT_SLICE.start + offset] = float(by_name[joint_name])
            for offset, joint_name in enumerate(self._right_joint_names):
                if joint_name in by_name:
                    full[RIGHT_JOINT_SLICE.start + offset] = float(by_name[joint_name])
            # Only each gripper's joint1 is ever published (joint2 is the follower that
            # ROS2JointCommandAction copies it onto -- see its apply_actions()). Quantised
            # by the same shared state machine that term applies, so the logged action is
            # the endpoint the joint was actually driven to -- see snap_gripper_command.
            if LEFT_GRIPPER_JOINT_NAME in by_name:
                full[LEFT_JOINT_GRP_IDX] = snap_gripper_command(
                    LEFT_GRIPPER_JOINT_NAME, float(by_name[LEFT_GRIPPER_JOINT_NAME])
                )
            if RIGHT_GRIPPER_JOINT_NAME in by_name:
                full[RIGHT_JOINT_GRP_IDX] = snap_gripper_command(
                    RIGHT_GRIPPER_JOINT_NAME, float(by_name[RIGHT_GRIPPER_JOINT_NAME])
                )

        # An arm locked out by --task_mode is held at its rest pose by ROS2JointCommandAction,
        # which never reads the message above. Log THAT, not the command the operator's headset
        # is still publishing for it: the recorded action has to be what the robot was actually
        # told to do, or replay_demos.py would drive the locked arm through motions this episode
        # never contained (and the recorded state, which is measured, would contradict it).
        default_pos = self._robot.data.default_joint_pos
        for arm in self._locked_arms:
            if arm == "left":
                full[LEFT_JOINT_SLICE] = default_pos[0, self._left_joint_ids]
                full[LEFT_JOINT_GRP_IDX] = default_pos[0, self._left_gripper_id]
            else:
                full[RIGHT_JOINT_SLICE] = default_pos[0, self._right_joint_ids]
                full[RIGHT_JOINT_GRP_IDX] = default_pos[0, self._right_gripper_id]

        return full, full[LEFT_JOINT_GRP_IDX].item(), full[RIGHT_JOINT_GRP_IDX].item()


class JointMirrorBroadcaster:
    """Best-effort UDP broadcaster of the robot's current joint positions.

    Fire-and-forget by design: never blocks and never raises into the sim loop.
    A separate, out-of-process bridge (e.g. lerobot_openarm/mirror_bridge.py) is
    responsible for everything hardware-related, including deciding what to do
    about stale/missing packets. This class does not know a real robot exists.
    """

    JOINT_NAME_PATTERNS = [
        r"openarm_left_joint[1-7]",
        r"openarm_right_joint[1-7]",
        r"openarm_left_finger_joint.*",
        r"openarm_right_finger_joint.*",
    ]

    @classmethod
    def resolve_mirror_joint_indices(cls, robot) -> tuple[list[int], list[str]]:
        """Return (indices, names) of the OpenArm joints within robot.data.joint_names,
        in the SAME order they appear in robot.data.joint_pos -- i.e. the same column
        order recorded into the HDF5 dataset's states/articulation/robot/joint_position.
        """
        all_names = robot.data.joint_names
        pattern = re.compile("|".join(f"(?:{p})" for p in cls.JOINT_NAME_PATTERNS))
        indices = [i for i, name in enumerate(all_names) if pattern.fullmatch(name)]
        names = [all_names[i] for i in indices]
        return indices, names

    WIDTH_PRINT_PERIOD_S = 0.5  # throttle -- printing every step at 30Hz would flood the console

    # Matches BinaryJointPositionActionCfg's open_command_expr/close_command_expr for the
    # finger joints in stack_joint_pos_env_cfg.py (both arms use the same values).
    GRIPPER_OPEN_VAL = 0.044
    GRIPPER_CLOSED_VAL = 0.0

    def __init__(self, robot, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)
        self._indices, self._names = self.resolve_mirror_joint_indices(robot)
        self._robot = robot
        self._seq = 0
        self._history: list[tuple[float, dict]] = []
        self._last_width_print_t = 0.0
        print(f"[MIRROR] Broadcasting {len(self._names)} joints to {host}:{port} -> {self._names}")

    def broadcast(self, left_gripper_state: float | None = None, right_gripper_state: float | None = None):
        """Broadcast the robot's mirrored joint state.

        Arm joints use the actual/measured position (`robot.data.joint_pos`) -- that's the real
        kinematic pose and should be mirrored as-is. Finger joints, if a `*_gripper_state` is
        given, are overridden to the fixed commanded open/close target instead of the measured
        position.

        Why: sim's cube and gripper fingers are rigid bodies, so once they contact each other the
        measured finger joint position stalls almost exactly at the object's geometric surface --
        that's correct rigid-body physics, not a bug. The real gripper's fingertip pads are
        compliant and keep squeezing past that same contact point to reach a firm grip (observed:
        sim/real agreed closely at first contact, ~52.6/53.4mm, but the real gripper needed to
        reach ~43.4mm for a grip that actually holds). Mirroring the *measured* sim position was
        capping the real robot's target at sim's rigid stopping point, never letting it command
        the real gripper to keep squeezing further. Mirroring the fixed open/close *target*
        instead lets the real hardware's own compliant pads decide how far they actually close,
        independent of whatever sim's specific rigid object happened to stop the fingers at.

        The `*_gripper_state` argument carries a +-1 binary command in the IK/keyboard modes but
        a position in metres in vr_joint_ros2_native (build_dual_action returns the gripper
        column it logged). The `> 0` test below reads both correctly only because that column is
        quantised to exactly 0.0 / 0.044 -- see snap_gripper_command. Feeding it an unquantised
        trigger position would classify every value but an exact 0.0 as "open", i.e. the real
        gripper would never be told to close.
        """
        joint_pos = self._robot.data.joint_pos[0, self._indices].tolist()
        joints = dict(zip(self._names, joint_pos))

        if left_gripper_state is not None:
            left_val = self.GRIPPER_OPEN_VAL if left_gripper_state > 0 else self.GRIPPER_CLOSED_VAL
            for name in joints:
                if name.startswith("openarm_left_finger_joint"):
                    joints[name] = left_val
        if right_gripper_state is not None:
            right_val = self.GRIPPER_OPEN_VAL if right_gripper_state > 0 else self.GRIPPER_CLOSED_VAL
            for name in joints:
                if name.startswith("openarm_right_finger_joint"):
                    joints[name] = right_val

        packet = {
            "seq": self._seq,
            "t": time.time(),
            "joints": joints,
        }
        self._seq += 1
        self._history.append((packet["t"], joints))
        try:
            self._sock.sendto(json.dumps(packet).encode("utf-8"), self._addr)
        except OSError:
            pass  # best-effort only -- never let a networking hiccup break recording

        if packet["t"] - self._last_width_print_t >= self.WIDTH_PRINT_PERIOD_S:
            self._last_width_print_t = packet["t"]
            # Both finger joints are prismatic, 0.0 (closed) .. 0.044m (open), moving symmetrically
            # outward -- see openarm_description.urdf finger_joint1/2 limits and mimic tag. Total
            # gripper opening width is the sum of both fingers' travel from the closed position.
            left_mm = joints.get("openarm_left_finger_joint1", 0.0) * 2000.0
            right_mm = joints.get("openarm_right_finger_joint1", 0.0) * 2000.0
            print(f"[SIM GRIPPER]  left={left_mm:5.1f}mm  right={right_mm:5.1f}mm")

    def history(self) -> list[tuple[float, dict]]:
        return self._history


class JointFeedbackReceiver:
    """Best-effort UDP listener for ACTUAL joint feedback broadcast back by a
    real-robot bridge (e.g. lerobot_openarm/mirror_bridge.py's --feedback-port).
    Logs the full history (not just the latest packet) for a sim-vs-real comparison
    plot when this script exits. Runs in a background thread; never blocks the sim
    loop. This process still never talks to hardware -- it only listens for numbers
    a separate, out-of-process bridge chooses to send back.
    """

    def __init__(self, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.settimeout(0.5)
        self._lock = threading.Lock()
        self._log: list[tuple[float, dict]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[MIRROR] Listening for real-robot feedback on {host}:{port}")

    def _run(self):
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                packet = json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            with self._lock:
                self._log.append((packet["t"], packet["joints"]))

    def history(self) -> list[tuple[float, dict]]:
        with self._lock:
            return list(self._log)

    def stop(self):
        self._stop.set()
        self._sock.close()


class RateLimiter:
    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time += self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def run_ramp_to_rest_test(env, duration: float = 4.0, rate_hz: float = 50.0, stop_requested: dict | None = None):
    """Replay reset_to_rest_pose.py's exact real-robot ramp (straight-line joint-space
    interpolation from the current pose to the default rest pose) inside Isaac Sim, so you can
    watch the viewport for the arm/gripper intersecting the pad -- without needing the real
    hardware to test it.

    This bypasses the env's normal action interface (IK deltas + binary gripper) entirely and
    drives the robot's joints directly, matching lerobot_openarm/sim_bridge_common.py's
    `ramp_to()`:  cmd = start + alpha * (end - start), alpha going 0->1 over `duration` seconds.
    `env.step()` is not called during the ramp (nothing is recorded, no observations/rewards/
    terminations run) -- this is a visual diagnostic only. Whatever pose the arm is in when you
    trigger this becomes the ramp's start; the target is the task's own default joint pose
    (`robot.data.default_joint_pos`), i.e. what a normal `env.reset()` would put it in.
    """
    robot = env.unwrapped.scene["robot"]
    joint_ids, names = JointMirrorBroadcaster.resolve_mirror_joint_indices(robot)

    start_pos = robot.data.joint_pos[0, joint_ids].clone()
    end_pos = robot.data.default_joint_pos[0, joint_ids].clone()

    print(f"[RAMP TEST] Ramping {len(names)} joints to the default rest pose over {duration}s...")
    print("[RAMP TEST] Watch the viewport -- Ctrl+C aborts and leaves the arm wherever it stopped.")

    steps = max(1, int(duration * rate_hz))
    dt = 1.0 / rate_hz
    sim_dt = env.unwrapped.sim.get_physics_dt()
    for i in range(1, steps + 1):
        if stop_requested is not None and stop_requested.get("flag"):
            print("[RAMP TEST] Aborted.")
            break
        step_start = time.time()
        alpha = i / steps
        target = start_pos + alpha * (end_pos - start_pos)
        target_batched = target.unsqueeze(0).expand(env.unwrapped.num_envs, -1)
        robot.set_joint_position_target(target_batched, joint_ids=joint_ids)
        robot.write_data_to_sim()
        env.unwrapped.sim.step()
        robot.update(sim_dt)
        env.unwrapped.sim.render()
        remaining = dt - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)
    else:
        print("[RAMP TEST] Done -- arm should now be at the default rest pose.")


class ReturnToRestTrajectory:
    """The closing motion of a recorded episode: a joint-space trajectory from wherever the arms
    were last commanded to back to the task's rest pose, executed one control step at a time THROUGH
    the main loop's ``env.step()`` so every step of it lands in the episode.

    The same straight line through joint space as ``run_ramp_to_rest_test`` (and, on hardware,
    reset_to_rest_pose.py's ``ramp_to()``) -- ``cmd = start + alpha * (rest - start)`` -- but with
    three differences that matter. That one drives PhysX directly with ``env.step()`` bypassed, which
    is exactly why it records nothing; here the ramp only *produces* an action vector and the main
    loop steps the env with it. Its ``alpha`` is linear in time, where this one is a quintic
    minimum-jerk profile (see :meth:`advance`) so the retreat can be short without arriving abruptly.
    And its start state is the arms' *measured* pose at rest, where this one starts from the pose and
    velocity the arms were last **commanded** to and were actually moving at (see :meth:`start`) --
    which is what makes the handover from live teleop to this trajectory seamless from any pose the
    operator happens to stop in, mid-motion or not.

    Only meaningful for the two joint-layout devices (see requires_manual_arm), and the two encode
    that 16D vector differently, hence ``is_native``:

    * ``vr_joint_ros2`` -- JointPositionActionCfg with ``use_default_offset=True``, so the arm
      columns are ``(target - rest) / JOINT_ACTION_SCALE`` and the gripper column is the binary
      +-1.0 command.
    * ``vr_joint_ros2_native`` -- absolute joint targets and an absolute gripper position, matching
      ROS2NativeJointTeleop.build_dual_action. That device ignores the action vector for control
      (see ROS2JointCommandAction), so :attr:`RETURN_TO_REST` is populated alongside it as the
      channel that actually moves the robot; the vector is still what gets recorded, and the two
      are built from the same numbers so they cannot disagree.

    The grippers do NOT ramp, and are the one part of the robot the rest pose is NOT applied to:
    both are held at exactly the command they were last given, all the way to the rest pose. Their
    rest value is fully open, so ramping them alongside the arms would make every episode end by
    dropping the object the demo just spent itself picking up.

    Holding the *command* is also what keeps the object gripped, not just un-released. A closed
    gripper's jaws stall on the object's surface well short of where they were told to go (e.g.
    0.0325 m measured against a ~0 m command), and it is precisely that command/measured gap the
    implicit actuator turns into squeeze force -- see ROS2NativeJointTeleop.build_dual_action, which
    records commands rather than measurements for the same reason. Re-issuing the held command
    unchanged every step reproduces that gap exactly, so grip force during the return is the same
    force that was holding the object when the operator pressed X. Latching the *measured* position
    instead would command zero error, i.e. zero squeeze, and the object would slide out of a hand
    that never visibly opened.
    """

    #: Cap on how far the latched start pose is allowed to sit from the measured pose, in rad. The
    #: start comes from the last *commanded* target (see :meth:`start`), and under the implicit
    #: actuator that command only ever leads the measurement by a fraction of a radian. A larger gap
    #: than this means the command buffer is not describing this robot -- the realistic case being
    #: that no action term has written these joints since the articulation was created, leaving
    #: ``joint_pos_target`` at its zero-fill -- so clamp toward the measurement rather than ramp from
    #: a pose the arm has never been in.
    MAX_COMMAND_LAG = 0.5

    #: Cap on the departure velocity the profile is allowed to carry, expressed as the distance it
    #: would cover at that velocity over the ramp's whole duration (rad). The velocity term peaks at
    #: ~0.2x this much of extra excursion beyond the direct path, so 0.75 rad buys the smooth
    #: deceleration while bounding any overshoot at roughly 0.15 rad (~9 deg) per joint.
    MAX_START_LEAD = 0.75

    def __init__(self, robot, sim_device: str, duration: float, rate_hz: float, is_native: bool):
        self._robot = robot
        self._device = sim_device
        self._is_native = is_native
        self._steps = max(1, int(round(duration * rate_hz)))
        # The duration actually flown, which is the step count rounded above rather than the
        # requested `duration` -- the velocity term is scaled by it and has to agree with the `s`
        # advance() computes, or the profile would depart at a velocity slightly off the arm's.
        self._duration = self._steps / rate_hz
        # Same find_joints() calls, in the same order, that ROS2NativeJointTeleop/
        # swap_to_joint_position_actions resolve the 16D layout's columns with -- index by the
        # resolved ids, never by assuming the regex enumerates joint1..joint7 in order.
        self._left_ids, _ = robot.find_joints(LEFT_ARM_JOINT_REGEX)
        self._right_ids, _ = robot.find_joints(RIGHT_ARM_JOINT_REGEX)
        self._left_grp_ids, _ = robot.find_joints(
            [LEFT_GRIPPER_JOINT_NAME, LEFT_GRIPPER_FOLLOWER_JOINT_NAME]
        )
        self._right_grp_ids, _ = robot.find_joints(
            [RIGHT_GRIPPER_JOINT_NAME, RIGHT_GRIPPER_FOLLOWER_JOINT_NAME]
        )
        self.active = False
        self.left_gripper_cmd = 0.0
        self.right_gripper_cmd = 0.0
        self._grippers_latched = False
        self._step = 0
        # Placeholders only -- start() overwrites all four, and `active` is what gates advance().
        self._left_start = torch.zeros(len(self._left_ids), device=sim_device)
        self._right_start = torch.zeros(len(self._right_ids), device=sim_device)
        self._left_lead = torch.zeros_like(self._left_start)
        self._right_lead = torch.zeros_like(self._right_start)

    def start(self) -> None:
        """Latch the trajectory's start state -- the arms' last COMMANDED joint targets and the
        velocity they are actually travelling at -- and begin.

        The start pose is read from ``robot.data.joint_pos_target``, NOT ``robot.data.joint_pos``,
        because this trajectory *commands* positions rather than measuring them. Under the implicit
        actuator the measured pose always sits a steady-state error behind the command -- the very
        same command/measurement gap this class leans on to keep grip force up (see the class
        docstring), only spread over seven arm joints holding themselves and a payload against
        gravity, which is where it is largest. Starting from the measurement therefore steps the
        command *backwards* by that entire error on the trajectory's first sample, and a step input
        into a stiff PD loop is a jolt: the arm snaps, and only then follows the rest of the profile
        smoothly. Starting from the last command makes the handover from teleop to trajectory
        continuous in position, so there is nothing to snap. That it also makes the *recorded* action
        sequence continuous matters just as much: a jump baked into the same place in every demo is a
        jump the policy learns to reproduce.

        Velocities are latched for the same reason one derivative up. The operator's hands are often
        still moving when they press X, and a profile that departs at zero velocity is a demand for
        an instantaneous stop -- so :meth:`advance` departs at this velocity and bleeds it off
        instead. Between the two, the trajectory is smooth out of any pose *and* any motion the
        episode happens to end in.

        The grippers are deliberately NOT latched here: this runs from the button callback, which
        fires from inside the teleop device's own build_dual_action() before that method has
        computed this step's gripper commands -- so anything read now is one step stale. The first
        :meth:`advance` call latches them instead, from the values the main loop has by then
        received. One step is usually nothing, but "usually" is not what should stand between a
        grasped object and the floor. The arm state above has no such problem: no ``env.step()``
        runs between here and that first :meth:`advance`, so the commanded targets read now are the
        ones the simulation is still converging to.
        """
        self._step = 0
        self._left_start, self._left_lead = self._latch_arm_start(self._left_ids)
        self._right_start, self._right_lead = self._latch_arm_start(self._right_ids)
        self._grippers_latched = False
        self.active = True

    def _latch_arm_start(self, joint_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """One arm's start pose and departure lead -- see :meth:`start` for why these come from the
        commanded targets and the measured velocities.

        Returns:
            ``(start_pos, lead)``. ``lead`` is the start velocity pre-multiplied by the ramp's
            duration, i.e. already in the units :meth:`advance`'s normalised-time basis wants, and
            clamped to :attr:`MAX_START_LEAD`.
        """
        measured = self._robot.data.joint_pos[0, joint_ids]
        commanded = self._robot.data.joint_pos_target[0, joint_ids]
        start = torch.clamp(commanded, measured - self.MAX_COMMAND_LAG, measured + self.MAX_COMMAND_LAG)
        lead = (self._robot.data.joint_vel[0, joint_ids] * self._duration).clamp(
            -self.MAX_START_LEAD, self.MAX_START_LEAD
        )
        return start.clone(), lead.clone()

    def stop(self) -> None:
        """End the ramp and hand control back to the teleop device."""
        self.active = False
        RETURN_TO_REST["targets"] = None

    def advance(self, left_gripper_cmd: float, right_gripper_cmd: float) -> tuple[torch.Tensor, bool]:
        """Produce this control step's action vector.

        Args:
            left_gripper_cmd: The gripper command the teleop device produced for this step, in
                whatever encoding this device uses (absolute position for the native path, binary
                +-1.0 otherwise). Latched on the FIRST call -- so the hold value is the live
                command from the step the operator pressed X, not a stale one (see :meth:`start`)
                -- and ignored on every call after that, which is what makes it a hold.
            right_gripper_cmd: As above, for the right gripper.

        Returns:
            ``(action_16d, finished)``. ``finished`` is True on the step that reaches the rest
            pose -- the caller should still step the env with this action (so the final,
            at-rest sample is in the episode) before saving.
        """
        if not self._grippers_latched:
            self.left_gripper_cmd = left_gripper_cmd
            self.right_gripper_cmd = right_gripper_cmd
            self._grippers_latched = True
        self._step += 1
        # A quintic minimum-jerk profile rather than the linear `s` run_ramp_to_rest_test uses: it
        # arrives at zero joint velocity AND zero acceleration, which is what makes a SHORT return
        # usable. A linear ramp reaches its full speed on step one and drops to zero on the last, and
        # at these durations that pair of velocity steps is what shakes a carried object loose -- so
        # the same peak effort buys a noticeably quicker retreat here. Peak rate is 1.875x the
        # average, i.e. --return_to_rest_secs 1.5 moves no faster at its quickest than a linear 0.8 s
        # ramp would throughout.
        #
        # Two bases, summed, because the boundary conditions at the two ends are not symmetric:
        #
        # * `pos_basis` (10s^3 - 15s^4 + 6s^5) carries the actual travel, 0 -> 1, flat in both
        #   velocity and acceleration at each end.
        # * `vel_basis` (s - 6s^3 + 8s^4 - 3s^5) starts at zero with unit slope and decays to zero
        #   value, slope and curvature by s = 1. Scaled by the latched lead, it is what lets the
        #   trajectory *depart at the speed the arm is already moving* and bleed that motion off,
        #   instead of demanding it stop dead on the first sample. This is the half the older
        #   smoothstep could not express: smoothstep eases in from zero velocity, which reads as
        #   smooth only if the arm was already stationary when the operator pressed X.
        #
        # Both are exact at the endpoints, so s = 1 still lands on the rest pose to the bit.
        s = min(1.0, self._step / self._steps)
        pos_basis = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)
        vel_basis = s * (1.0 - s * s * (6.0 - 8.0 * s + 3.0 * s * s))
        default = self._robot.data.default_joint_pos
        left_rest = default[0, self._left_ids]
        right_rest = default[0, self._right_ids]
        left_target = self._left_start + pos_basis * (left_rest - self._left_start) + vel_basis * self._left_lead
        right_target = self._right_start + pos_basis * (right_rest - self._right_start) + vel_basis * self._right_lead

        if self._is_native:
            full_target = default.clone()
            full_target[:, self._left_ids] = left_target
            full_target[:, self._right_ids] = right_target
            # Both jaws (leader and its follower) get the one held command -- see class docstring.
            full_target[:, self._left_grp_ids] = self.left_gripper_cmd
            full_target[:, self._right_grp_ids] = self.right_gripper_cmd
            RETURN_TO_REST["targets"] = full_target

        full = torch.zeros(TOTAL_ACTION_DIM_JOINT, dtype=torch.float32, device=self._device)
        if self._is_native:
            full[LEFT_JOINT_SLICE] = left_target
            full[RIGHT_JOINT_SLICE] = right_target
        else:
            full[LEFT_JOINT_SLICE] = (left_target - left_rest) / JOINT_ACTION_SCALE
            full[RIGHT_JOINT_SLICE] = (right_target - right_rest) / JOINT_ACTION_SCALE
        full[LEFT_JOINT_GRP_IDX] = self.left_gripper_cmd
        full[RIGHT_JOINT_GRP_IDX] = self.right_gripper_cmd

        return full, self._step >= self._steps


def build_single_arm_action(
    teleop_7d: torch.Tensor,
    gripper_state: float,
    device: str,
) -> tuple[torch.Tensor, float]:
    """Pass-through for 7D single-arm tasks (IK 6D + gripper 1D)."""
    full = torch.zeros(7, dtype=torch.float32, device=device)
    full[:6] = teleop_7d[:6]
    full[6] = teleop_7d[6].item()
    return full, teleop_7d[6].item()


def build_dual_arm_action(
    teleop_7d: torch.Tensor,
    active_arm: str,
    left_gripper_state: float,
    right_gripper_state: float,
    device: str,
) -> tuple[torch.Tensor, float, float]:
    """Route 7D teleop output to the correct arm in the 14D action vector.

    Args:
        teleop_7d: Shape (7,) tensor [dx,dy,dz,drx,dry,drz,gripper].
        active_arm: "left" or "right".
        left_gripper_state: Last gripper command for left arm (±1.0).
        right_gripper_state: Last gripper command for right arm (±1.0).
        device: Torch device string.

    Returns:
        (full_action_14d, updated_left_gripper_state, updated_right_gripper_state)
    """
    full = torch.zeros(TOTAL_ACTION_DIM, dtype=torch.float32, device=device)
    gripper_cmd = teleop_7d[6].item()

    if active_arm == "left":
        full[LEFT_IK_SLICE] = teleop_7d[:6]
        full[LEFT_GRP_IDX] = gripper_cmd
        full[RIGHT_GRP_IDX] = right_gripper_state  # keep right gripper unchanged
        return full, gripper_cmd, right_gripper_state
    else:
        full[LEFT_GRP_IDX] = left_gripper_state    # keep left gripper unchanged
        full[RIGHT_IK_SLICE] = teleop_7d[:6]
        full[RIGHT_GRP_IDX] = gripper_cmd
        return full, left_gripper_state, gripper_cmd


def lock_arms_in_native_action_cfg(env_cfg, locked_arms: tuple[str, ...]) -> None:
    """Mark the ROS2JointCommandActionCfg fields of *locked_arms* as ``locked``.

    Only meaningful for --teleop_device vr_joint_ros2_native, where control does NOT flow
    through the action vector env.step() is called with (the targets are read from the ROS2
    OmniGraph inside apply_actions), so masking the action vector -- what
    mask_locked_arms_in_action does for every other device -- would gate nothing at all.
    Must run AFTER swap_to_native_ros2_joint_actions has installed those cfg fields.
    """
    for arm in locked_arms:
        prefix = "" if arm == "left" else "right_"
        for field in (f"{prefix}arm_action", f"{prefix}gripper_action"):
            getattr(env_cfg.actions, field).locked = True


def mask_locked_arms_in_action(
    full_action: torch.Tensor, locked_arms: tuple[str, ...], is_joint_layout: bool
) -> torch.Tensor:
    """Zero out the action columns of every arm in *locked_arms*, in place.

    Zero is "hold still" in both layouts this covers, which is why one mask serves both:
    the IK layout's arm terms are relative (``use_relative_mode=True``, so a zero delta pose
    means stay put) and the joint layout's are JointPositionActionCfg with
    ``use_default_offset=True`` (so a zero target resolves to the arm's default pose).

    The gripper column is masked to +1.0 rather than 0.0 -- it feeds a BinaryJointPositionAction,
    where the sign is the command and 0.0 would read as "close". A locked arm must sit there with
    its hand open, not slowly crush whatever is in front of it.
    """
    arm_slices = {
        "left": (LEFT_JOINT_SLICE if is_joint_layout else LEFT_IK_SLICE,
                 LEFT_JOINT_GRP_IDX if is_joint_layout else LEFT_GRP_IDX),
        "right": (RIGHT_JOINT_SLICE if is_joint_layout else RIGHT_IK_SLICE,
                  RIGHT_JOINT_GRP_IDX if is_joint_layout else RIGHT_GRP_IDX),
    }
    for arm in locked_arms:
        arm_slice, grp_idx = arm_slices[arm]
        full_action[arm_slice] = 0.0
        full_action[grp_idx] = 1.0
    return full_action


def main():
    rate_limiter = RateLimiter(args_cli.step_hz)

    # ── Output dirs ──────────────────────────────────────────────────────────
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Settled before Isaac Sim was even launched -- see preflight_dataset().
    dataset_path, resume_offset = DATASET_PATH, RESUME_OFFSET

    # ── Env config ────────────────────────────────────────────────────────────
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as e:
        logger.error(f"Failed to parse env config: {e}")
        return

    # Refuse the silent-cube trap rather than recording a session's worth of wrong-object demos.
    if args_cli.task_mode is None and args_cli.task.split(":")[-1] in CAN_TARGET_TASKS:
        logger.error(
            f"'{args_cli.task}' targets the can, but that swap only happens via --task_mode, which"
            " was not passed -- this run would have recorded the base task's CUBE instead."
            " Re-run with --task_mode left|right|handover (and pass the SAME one to"
            " annotate_demos.py)."
        )
        return

    # Task mode first: it rewrites terminations.success, so it has to land before the
    # success_term is lifted off the cfg below.
    locked_arms: tuple[str, ...] = ()
    if args_cli.task_mode is not None:
        try:
            apply_task_mode(env_cfg, args_cli.task_mode)
        except ValueError as e:
            logger.error(str(e))
            return
        controlled = CONTROLLED_ARMS[args_cli.task_mode]
        locked_arms = tuple(arm for arm in ("left", "right") if arm not in controlled)
        print(f"[TASK MODE] {args_cli.task_mode}")
        print(f"[TASK MODE] Teleoperable arm(s): {', '.join(controlled).upper()}")
        if locked_arms:
            print(
                f"[TASK MODE] Locked at rest pose: {', '.join(locked_arms).upper()}"
                " (still recorded, just not driveable)"
            )

    if args_cli.teleop_device == "vr_joint_ros2":
        try:
            swap_to_joint_position_actions(env_cfg)
        except RuntimeError as e:
            logger.error(str(e))
            return
    elif args_cli.teleop_device == "vr_joint_ros2_native":
        try:
            swap_to_native_ros2_joint_actions(env_cfg)
        except RuntimeError as e:
            logger.error(str(e))
            return
        # Must follow the swap -- it is the cfg fields the swap just installed that carry the flag.
        lock_arms_in_native_action_cfg(env_cfg, locked_arms)

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir if output_dir else "."
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    if args_cli.resume:
        # Same handler the export path would otherwise use, minus the truncation -- see
        # ResumeExistingDatasetMixin.
        env_cfg.recorders.dataset_file_handler_class_type = make_resumable(
            env_cfg.recorders.dataset_file_handler_class_type
        )

    # ── Create env ────────────────────────────────────────────────────────────
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return

    sim_device = args_cli.device

    # ── Real-robot mirror broadcaster (opt-in, off by default) ─────────────────
    mirror_broadcaster = None
    if args_cli.mirror_udp_port:
        mirror_broadcaster = JointMirrorBroadcaster(
            robot=env.scene["robot"], host=args_cli.mirror_udp_host, port=args_cli.mirror_udp_port
        )

    # ── Real-robot feedback listener, for a sim-vs-real plot on exit (opt-in) ──
    feedback_receiver = None
    if args_cli.mirror_feedback_port:
        feedback_receiver = JointFeedbackReceiver(host=args_cli.mirror_udp_host, port=args_cli.mirror_feedback_port)

        # Ctrl+C in this app appears to tear the process down directly rather than
        # raising a normal Python KeyboardInterrupt that falls through to code after
        # the main loop -- contextlib.suppress(KeyboardInterrupt) below never got a
        # chance to matter, the code after env.close() never ran, and no plot was
        # ever saved. atexit hooks Python's actual interpreter shutdown instead, which
        # survives that. Guarded so it can't double-run if normal flow ALSO reaches
        # the equivalent call later.
        plot_state = {"saved": False}

        def _save_plot_once():
            if plot_state["saved"]:
                return
            plot_state["saved"] = True
            feedback_receiver.stop()
            save_sim_vs_real_plot(mirror_broadcaster, feedback_receiver)

        atexit.register(_save_plot_once)

    # ── Joint-order manifest for offline replay-on-hardware (opt-in, off by default) ──
    if args_cli.dump_joint_order:
        _, joint_order_names = JointMirrorBroadcaster.resolve_mirror_joint_indices(env.scene["robot"])
        with open(args_cli.dump_joint_order, "w") as f:
            json.dump({"joint_order": joint_order_names}, f, indent=2)
        print(f"[INFO] Wrote joint order ({len(joint_order_names)} joints) to {args_cli.dump_joint_order}")

    # ── Detect action space ────────────────────────────────────────────────────
    total_action_dim = env.action_manager.total_action_dim
    is_dual_arm = total_action_dim in (TOTAL_ACTION_DIM, TOTAL_ACTION_DIM_JOINT)
    print(f"[INFO] Action dim: {total_action_dim}  ({'dual-arm' if is_dual_arm else 'single-arm'})")

    use_vr_teleop = args_cli.teleop_device == "vr_ros2"
    use_vr_joint_teleop = args_cli.teleop_device == "vr_joint_ros2"
    use_vr_joint_teleop_native = args_cli.teleop_device == "vr_joint_ros2_native"
    use_any_vr_teleop = use_vr_teleop or use_vr_joint_teleop or use_vr_joint_teleop_native
    # vr_joint_ros2/vr_joint_ros2_native have a Quest X button wired to "arm" recording
    # (see VRDualArmJointTeleop/ROS2NativeJointTeleop's button_x handling) -- for those two modes
    # BOTH recording and robot motion start OFF after every reset and stay off until X is pressed
    # (see MOTION_GATE): the robot holds its rest pose, so there is no pre-X teleop to save and
    # nothing that can disturb the scene before the episode begins. Keyboard and vr_ros2 have no
    # such button, so they keep the old always-on behavior.
    requires_manual_arm = use_vr_joint_teleop or use_vr_joint_teleop_native
    if use_vr_teleop and total_action_dim != TOTAL_ACTION_DIM:
        logger.error("--teleop_device vr_ros2 requires a dual-arm (14D action) task.")
        return
    if use_vr_joint_teleop and total_action_dim != TOTAL_ACTION_DIM_JOINT:
        logger.error(
            "--teleop_device vr_joint_ros2 requires a dual-arm task with the arm actions"
            f" swapped to JointPositionActionCfg (16D action, got {total_action_dim})."
        )
        return
    if use_vr_joint_teleop_native and total_action_dim != TOTAL_ACTION_DIM_JOINT:
        logger.error(
            "--teleop_device vr_joint_ros2_native requires a dual-arm task with all four action"
            f" fields swapped to NoOpActionCfg (16D action, got {total_action_dim})."
        )
        return

    # ── Teleop device ─────────────────────────────────────────────────────────
    if use_vr_teleop:
        teleop = VRDualArmTeleop(
            robot=env.scene["robot"],
            udp_host=args_cli.vr_udp_host,
            udp_port=args_cli.vr_udp_port,
            sim_device=sim_device,
            max_pos_step=args_cli.vr_max_pos_step,
            max_rot_step=args_cli.vr_max_rot_step,
            quat_offset=tuple(args_cli.vr_quat_offset),
        )
    elif use_vr_joint_teleop:
        teleop = VRDualArmJointTeleop(
            robot=env.scene["robot"],
            udp_host=args_cli.vr_joint_udp_host,
            udp_port=args_cli.vr_joint_udp_port,
            sim_device=sim_device,
        )
    elif use_vr_joint_teleop_native:
        build_ros2_joint_command_graph(
            topic_name=args_cli.ros2_topic,
            domain_id=args_cli.ros2_domain_id,
        )
        teleop = ROS2NativeJointTeleop(
            robot=env.scene["robot"], sim_device=sim_device, locked_arms=locked_arms
        )
    else:
        # Use OpenArmKeyboard (arrow keys + I/O) to avoid Isaac Sim viewport
        # gizmo conflicts with W/A/S/D/Q/E.
        teleop = OpenArmKeyboard(pos_sensitivity=0.05, rot_sensitivity=0.1, sim_device=sim_device)

    # ── State ─────────────────────────────────────────────────────────────────
    # A single-arm task mode pins the keyboard's active arm to the arm that mode is about (and
    # disables TAB below) -- starting on "left" in --task_mode right would otherwise hand the
    # operator a locked arm to drive until they noticed.
    active_arm = "left" if "left" not in locked_arms else "right"
    left_gripper_state = 1.0   # 1.0 = open, -1.0 = close
    right_gripper_state = 1.0
    should_reset = False
    running = True
    # Demos IN THE DATASET, not demos recorded by this process -- under --resume those differ by
    # resume_offset, and it is the former that --num_demos is about and that the on-screen counter
    # should show.
    demo_count = resume_offset
    success_step_count = 0
    ramp_test_requested = False
    # Whether env.step()'s data is meant to be kept AND the robot is allowed to move -- for the
    # button modes these are deliberately the same switch. Starts True for modes with no arm
    # button (keyboard, vr_ros2) -- unchanged old behavior, always live, always recording. Starts
    # False for requires_manual_arm modes: env.step() still runs (the sim must keep ticking to
    # service ROS2 and render), but MOTION_GATE holds the robot at its rest pose and nothing is
    # recorded until start_recording() (button X) fires.
    recording_armed = not requires_manual_arm
    MOTION_GATE["enabled"] = recording_armed
    # The episode's closing motion (second button X) -- None when disabled or not applicable, in
    # which case that press exports immediately, as it always used to.
    return_to_rest = (
        ReturnToRestTrajectory(
            robot=env.scene["robot"],
            sim_device=sim_device,
            duration=args_cli.return_to_rest_secs,
            rate_hz=args_cli.step_hz,
            is_native=use_vr_joint_teleop_native,
        )
        if requires_manual_arm and args_cli.return_to_rest_secs > 0
        else None
    )

    def reset_episode():
        nonlocal should_reset
        should_reset = True
        print("Reset requested")

    def request_ramp_test():
        nonlocal ramp_test_requested
        ramp_test_requested = True

    # ── Hand-over progress reporting ──────────────────────────────────────────
    # Only meaningful in --task_mode handover, whose success condition is a 4-stage sequence rather
    # than a single test. Printing each transition turns "the episode never ended" into "it never
    # got past stage 1", which is the difference between a guess and a diagnosis.
    _HANDOVER_STAGE_LABELS = {
        0: "waiting for the RIGHT arm to grasp the can",
        1: "right arm has it -- waiting for the LEFT arm to take it while the right still holds",
        # NOT "and the can to land": handover_success dropped that requirement, because with manual
        # saving the operator routinely stops the episode still holding the can. What 2 -> 3 does
        # need besides the release is the LEFT hand keeping hold of the can for a stretch -- see
        # HANDOVER_RECEIVER_HOLD_STEPS -- so keep carrying it for ~1 s after letting go.
        2: "passed to the left arm -- waiting for the RIGHT arm to let go while the LEFT keeps hold",
        3: "complete",
    }
    _HANDOVER_STALL_REPORT_PERIOD = 40
    """Steps between diagnostics while a stage does not advance -- ~2 s at this task's 20 Hz control
    rate. Every step would be 20 near-identical lines a second; a stage CHANGE prints immediately
    regardless, so nothing is delayed by this."""

    handover_stage_seen: dict[str, int | None] = {"value": None}
    handover_stall_steps: dict[str, int] = {"value": 0}

    def _handover_conditions_line(stage: int) -> str:
        """Per-condition readout of what the stage machine is still waiting for.

        Every VERDICT here comes from the same functions the machine itself uses -- the booleans
        are openarm_task_modes.object_grasped_by / handover_latches, invoked with the success
        term's own params. Only the raw numbers alongside them are read separately, and those are
        exactly the quantities those functions threshold (hand_to_object_offsets,
        gripper_jaw_positions). A diagnostic that computed its own verdict could disagree with the
        machine, which is worse than no diagnostic at all.
        """
        from isaaclab.managers import SceneEntityCfg

        if success_term is None:
            return ""
        params = success_term.params
        object_cfg = params["object_cfg"]
        diff_threshold = params["diff_threshold"]

        def _arm(frame_name: str, fingers: list[str]) -> str:
            radial, axial = openarm_task_modes.hand_to_object_offsets(
                env, SceneEntityCfg(frame_name), object_cfg
            )
            jaws = openarm_task_modes.gripper_jaw_positions(env, fingers)[0]
            holds = bool(
                openarm_task_modes.object_grasped_by(
                    env,
                    ee_frame_cfg=SceneEntityCfg(frame_name),
                    gripper_joint_names=fingers,
                    object_cfg=object_cfg,
                    diff_threshold=diff_threshold,
                )[0]
            )
            closed_below = openarm_task_modes.GRIPPER_OPEN_VAL - openarm_task_modes.GRIPPER_THRESHOLD
            return (
                f"holds={str(holds):5} radial={float(radial[0]):.3f}/{diff_threshold:.2f}"
                f" axial={float(axial[0]):.3f}/{openarm_task_modes.GRASP_AXIAL_TOLERANCE:.2f}"
                f" jaws={'/'.join(f'{float(j):.3f}' for j in jaws)}<{closed_below:.3f}"
            )

        can_z = float(env.scene[object_cfg.name].data.root_pos_w[0, 2])
        lines = []
        if stage == 0:
            lines.append(f"  right {_arm('right_ee_frame', openarm_task_modes.RIGHT_FINGER_JOINTS)}")
        elif stage == 1:
            latches = openarm_task_modes.handover_latches(env)
            lines.append(f"  left  {_arm('ee_frame', openarm_task_modes.LEFT_FINGER_JOINTS)}")
            lines.append(
                f"  was_lifted={bool(latches['was_lifted'][0])} (can_z={can_z:.3f} needs"
                f" >{params['min_height']:.3f} WHILE the right hand holds it)"
                f"  right_holding_since_pickup={bool(latches['right_holding_since_pickup'][0])}"
            )
        elif stage == 2:
            lines.append(f"  right {_arm('right_ee_frame', openarm_task_modes.RIGHT_FINGER_JOINTS)} -> must go False")
            # The other half of 2->3, and the one an operator cannot see: the left hand has to be
            # confirmed still holding the can (jaw aperture inside the band -- shut on nothing
            # fails it) for HANDOVER_RECEIVER_HOLD_STEPS steps running, or the counter restarts.
            aperture = float(
                openarm_task_modes.gripper_aperture(env, openarm_task_modes.LEFT_FINGER_JOINTS)[0]
            )
            confirmed = bool(
                openarm_task_modes.receiver_grip_confirmed(
                    env,
                    ee_frame_cfg=SceneEntityCfg("ee_frame"),
                    gripper_joint_names=openarm_task_modes.LEFT_FINGER_JOINTS,
                    object_cfg=object_cfg,
                    diff_threshold=diff_threshold,
                )[0]
            )
            lo, hi = openarm_task_modes.HANDOVER_RECEIVER_APERTURE_RANGE
            held_steps = int(openarm_task_modes.handover_latches(env)["receiver_grip_steps"][0])
            lines.append(
                f"  left  grip_confirmed={str(confirmed):5} aperture={aperture:.4f} in"
                f" [{lo:.3f}, {hi:.3f}]  held {held_steps}/"
                f"{openarm_task_modes.HANDOVER_RECEIVER_HOLD_STEPS} steps running"
            )
        elif stage >= 3:
            # Stage 3 is reached and stays reached, but handover_success also re-asserts the grip
            # live, so an operator who opens the receiving hand after the pass is back to NOT
            # meeting the success condition -- with the stage line still reading "complete". Say
            # so, or that reads as the auto-save being broken.
            lines.append(
                "  the pass is done, but the LEFT hand has let go of the can -- close it again"
                " (or save manually); success needs the can still in that hand"
            )
        return "\n".join(lines)

    def _report_handover_stage(succeeded: bool = False):
        if args_cli.task_mode != "handover":
            return
        try:
            stage = int(openarm_task_modes.handover_stage(env)[0])
        except Exception:
            return
        if stage != handover_stage_seen["value"]:
            handover_stage_seen["value"] = stage
            handover_stall_steps["value"] = 0
            print(f"[HANDOVER {stage}/3] {_HANDOVER_STAGE_LABELS.get(stage, '?')}")
        handover_stall_steps["value"] += 1

        # Nothing left to wait for once complete AND the success condition is actually being met --
        # which past stage 3 is no longer the same thing, since handover_success re-checks the
        # receiving hand's grip live. Keep reporting while stage 3 is not converting into success.
        # No point reporting a stage the same step it was entered either: the conditions that block
        # it have not had a chance to change yet.
        if stage >= 3 and succeeded:
            return
        if handover_stall_steps["value"] < _HANDOVER_STALL_REPORT_PERIOD:
            return
        handover_stall_steps["value"] = 0
        try:
            detail = _handover_conditions_line(stage)
        except Exception as exc:  # a diagnostic must never take the recording session down
            detail = f"  (condition readout unavailable: {exc})"
        if detail:
            print(f"[HANDOVER {stage}/3] still waiting:\n{detail}")

    def on_button_x():
        """button_x handler -- a toggle: first press starts the episode, second press ends it.

        Manual saving exists because the hand-over's auto-success condition has to recognise a
        whole multi-step sequence (see handover_success) and any one of its steps going
        unrecognised leaves the operator with an episode that will not end. A button the operator
        controls does not depend on the sequence being detected at all.

        "Ends it" is not the same as "saves it": unless --return_to_rest_secs 0 disables it, the
        second press hands the robot to ReturnToRestTrajectory, and the export happens when THAT
        finishes (see the main loop). Further presses while it is on its way home are ignored --
        the ramp is short, and re-triggering it from a half-returned pose would only truncate the
        motion the press was asking for in the first place.
        """
        if return_to_rest is not None and return_to_rest.active:
            return
        if not recording_armed:
            start_recording()
        elif return_to_rest is not None:
            return_to_rest.start()
            print(
                f"Returning to rest pose over {args_cli.return_to_rest_secs:.1f}s"
                " -- grippers stay as they are; the episode is saved on arrival."
            )
        else:
            save_episode()

    def start_recording():
        """button_x handler (vr_joint_ros2 / vr_joint_ros2_native only, see
        requires_manual_arm) -- releases the robot AND arms recording, in that order.

        Also wipes whatever env.step() data has piled up in the recorder_manager's buffer since
        the last reset (IsaacLab's env.step() unconditionally feeds every step into that buffer;
        there is no separate "recording enabled" switch inside RecorderManager itself), so the
        episode that eventually gets exported (by save_episode() or the auto-success check below)
        only contains steps taken AFTER this button press. Those discarded steps are now all
        rest-pose ones, since MOTION_GATE kept the robot still until this fired.
        """
        nonlocal recording_armed
        if recording_armed:
            return  # already armed -- ignore repeat X presses within the same episode
        env.recorder_manager.reset([0])
        recording_armed = True
        MOTION_GATE["enabled"] = True
        print("Recording started (button X) -- robot released")

    def save_episode():
        nonlocal should_reset
        if requires_manual_arm and not recording_armed:
            print("Not recording yet -- press X to start recording before saving.")
            return
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.set_success_to_episodes(
            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
        )
        env.recorder_manager.export_episodes([0])
        print("Episode saved!")
        should_reset = True

    def toggle_arm():
        nonlocal active_arm
        active_arm = "right" if active_arm == "left" else "left"
        print(f"[ARM SWITCH] Active arm: {active_arm.upper()}")
        # Update the on-screen label
        _refresh_label()

    def _refresh_label():
        if use_any_vr_teleop:
            label_text = f"Bimanual VR teleop  |  Demos: {demo_count}"
        else:
            arm_indicator = "◄ LEFT" if active_arm == "left" else "RIGHT ►"
            label_text = f"Active arm: {arm_indicator}  |  Demos: {demo_count}"
        try:
            instruction_display.show_demo(label_text)
        except Exception:
            pass

    teleop.add_callback("R", reset_episode)
    teleop.add_callback("N", save_episode)
    # No TAB in a single-arm task mode: the other arm is locked, so switching to it would just
    # look like the teleop had died.
    if not use_any_vr_teleop and not locked_arms:
        teleop.add_callback("TAB", toggle_arm)
    teleop.add_callback("T", request_ramp_test)
    if requires_manual_arm:
        teleop.add_callback("X_START", on_button_x)

    # ── UI ────────────────────────────────────────────────────────────────────
    instruction_display = InstructionDisplay(xr=False)
    if HAS_MIMIC:
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(f"Active arm: LEFT  |  Demos: {demo_count}")
            arm_label = ui.Label("")
            instruction_display.set_labels(arm_label, demo_label)

    # ── Initial reset ─────────────────────────────────────────────────────────
    env.sim.reset()
    env.reset()
    teleop.reset()

    mode_str = "Dual-Arm" if is_dual_arm else "Single-Arm"
    print(f"\n=== OpenArm {mode_str} Recording ===")
    if use_vr_teleop:
        print("  VR (bimanual, pose/IK) — both arms + grippers driven live from the Dora UDP bridge")
    elif use_vr_joint_teleop:
        print("  VR (bimanual, joint-space) — both arms + grippers driven live from the Dora UDP bridge")
    elif use_vr_joint_teleop_native:
        print(
            "  VR (bimanual, joint-space, native ROS2) — both arms + grippers driven live by a ROS2"
            f" OmniGraph subscribed to '{args_cli.ros2_topic}' (domain_id={args_cli.ros2_domain_id})"
        )
    else:
        if is_dual_arm:
            print("  TAB        — switch active arm (left ↔ right)")
        print("  K          — toggle gripper open/close")
        print("  W/S        — EE forward / backward  (+x/-x)")
        print("  A/D        — EE left / right         (+y/-y)")
        print("  PgUp/PgDn  — EE up / down            (+z/-z)")
        print("  ↑/↓        — pitch ±  |  ←/→ — yaw ±  |  [/] — roll ±")
    if requires_manual_arm:
        print("  X (Quest)  — press ONCE to release the robot and start recording; press AGAIN to")
        print("               end the episode. Until the first press the robot stays at its rest")
        print("               pose and ignores your controllers, and nothing is saved. Line your")
        print("               arms up with the rest pose before the first press -- the robot takes")
        print("               your current pose in one step.")
        if return_to_rest is not None:
            print("               On the second press the robot drives itself back to the rest pose")
            print(f"               over {args_cli.return_to_rest_secs:.1f}s (grippers held as-is) and the episode is")
            print("               saved on arrival -- that return motion IS part of the demo. Let")
            print("               go of your controllers; they are ignored until it finishes.")
        print("  Y (Quest)  — discard & reset episode (re-freezes the robot until X)")
        if args_cli.manual_save:
            print("  (--manual_save: the task's success condition will NOT end an episode; only")
            print("   button X or the N key will.)")
    print("  N          — save episode as success (immediately, with no return-to-rest)")
    print("  R          — discard & reset episode")
    print("  T          — ramp current pose to rest (matches reset_to_rest_pose.py) and watch")
    print("               the viewport for a pad collision -- doesn't record, doesn't reset")
    if is_dual_arm and not use_any_vr_teleop:
        print(f"\nActive arm: LEFT\n")
    else:
        print()

    # Installed here (after AppLauncher/simulation_app setup, which may install its own
    # SIGINT handler) so this one takes effect for Ctrl+C from here on. It only sets a
    # flag rather than doing any work itself -- signal handlers can fire between any two
    # bytecode instructions, so anything more (file I/O, plotting) belongs in the main
    # loop's normal execution, not here. This exists because Ctrl+C was observed to tear
    # this app down before normal Python cleanup code (even atexit) got a chance to run.
    stop_requested = {"flag": False}
    signal.signal(signal.SIGINT, lambda signum, frame: stop_requested.__setitem__("flag", True))

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running() and not stop_requested["flag"]:
            if ramp_test_requested:
                ramp_test_requested = False
                run_ramp_to_rest_test(env, stop_requested=stop_requested)
                teleop.clear_deltas()  # clear deltas accumulated while keys were held during the ramp
                continue

            # Build action vector sized to match the task's action space
            if use_any_vr_teleop:
                full_action, left_gripper_state, right_gripper_state = teleop.build_dual_action(
                    left_gripper_state=left_gripper_state,
                    right_gripper_state=right_gripper_state,
                )
            elif is_dual_arm:
                teleop_7d = teleop.advance()
                full_action, left_gripper_state, right_gripper_state = build_dual_arm_action(
                    teleop_7d=teleop_7d,
                    active_arm=active_arm,
                    left_gripper_state=left_gripper_state,
                    right_gripper_state=right_gripper_state,
                    device=sim_device,
                )
            else:
                teleop_7d = teleop.advance()
                full_action, left_gripper_state = build_single_arm_action(
                    teleop_7d=teleop_7d,
                    gripper_state=left_gripper_state,
                    device=sim_device,
                )
            # The closing return-to-rest ramp overrides whatever the operator's controllers just
            # asked for -- but only AFTER build_dual_action() has been called above, because that
            # is also what polls the Quest buttons: skipping it for the duration of the ramp would
            # miss the release of the very X press that started it, and the next press would then
            # not read as a rising edge at all.
            rtr_active = False
            rtr_done = False
            if return_to_rest is not None and return_to_rest.active:
                rtr_active = True
                # The gripper commands go IN so the first call can latch this step's live values
                # (the button callback fired too early to read them -- see start()), and come back
                # out below as the held ones every call after.
                full_action, rtr_done = return_to_rest.advance(left_gripper_state, right_gripper_state)
                # Keep the mirror broadcaster and the post-reset state in step with what is
                # actually being commanded, rather than with the headset it is ignoring.
                left_gripper_state = return_to_rest.left_gripper_cmd
                right_gripper_state = return_to_rest.right_gripper_cmd

            # Enforce the arm gating. Both cases below are skipped for vr_joint_ros2_native, where
            # the action vector isn't what drives the robot -- that path is gated inside
            # ROS2JointCommandAction instead (--task_mode via cfg.locked, button X via
            # MOTION_GATE). Applied to the OTHER devices here rather than inside each teleop class
            # so there is exactly one place a gated arm can leak through.
            if is_dual_arm and not use_vr_joint_teleop_native:
                # Not armed yet (button X pending): freeze BOTH arms, not just the locked one.
                gated = ("left", "right") if not MOTION_GATE["enabled"] else locked_arms
                if gated:
                    full_action = mask_locked_arms_in_action(
                        full_action, gated, is_joint_layout=use_vr_joint_teleop
                    )
            actions = full_action.unsqueeze(0).expand(env.num_envs, -1)

            if running:
                obs, *_ = env.step(actions)
                if mirror_broadcaster is not None:
                    mirror_broadcaster.broadcast(
                        left_gripper_state=left_gripper_state,
                        right_gripper_state=right_gripper_state if is_dual_arm else None,
                    )

                # ── Debug: right arm left-finger contact force vs cube_2 ────────
                # Diagnoses the "right arm left finger doesn't grip" report -- prints
                # only while contact force is nonzero, so a silent terminal during a
                # grasp attempt means that finger truly isn't touching the cube.
                # Remove once the underlying cause is confirmed/fixed.
                if "contact_right_left_finger" in env.scene.sensors:
                    force_matrix = env.scene["contact_right_left_finger"].data.force_matrix_w
                    if force_matrix is not None:
                        force_norm = force_matrix[0].norm(dim=-1).max().item()
                        if force_norm > 1e-4:
                            print(f"[contact] openarm_right_left_finger vs cube_2 force: {force_norm:.3f} N")

            # The ramp's last step has now been recorded, so the episode is complete -- export it.
            # Deliberately after env.step(): saving on the step that merely *computes* the final
            # target would cut the arrival itself out of every demo.
            if rtr_done and return_to_rest is not None:
                return_to_rest.stop()
                print("Back at rest pose.")
                save_episode()

            # Success check -- gated on recording_armed so a not-yet-armed episode
            # (requires_manual_arm modes, before X is pressed) can never auto-export.
            #
            # Still EVALUATED under --manual_save and during the return-to-rest ramp, just never
            # acted on in either case: this call is what ticks the hand-over stage machine forward
            # (see handover_success), and the progress print below is the only way to see which
            # step of the sequence a stuck episode never reached. Not acting during the ramp is the
            # point -- that episode is already committed to being exported on arrival, and an
            # auto-export mid-ramp would truncate the very retreat it is recording.
            if success_term is not None and recording_armed:
                succeeded = bool(success_term.func(env, **success_term.params)[0])
                _report_handover_stage(succeeded)
                if succeeded and not args_cli.manual_save and not rtr_active:
                    success_step_count += 1
                    if success_step_count >= args_cli.num_success_steps:
                        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(
                            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                        )
                        env.recorder_manager.export_episodes([0])
                        print("Auto-success condition met!")
                        should_reset = True
                else:
                    success_step_count = 0

            # Update demo counter label
            total_in_dataset = resume_offset + env.recorder_manager.exported_successful_episode_count
            if total_in_dataset > demo_count:
                demo_count = total_in_dataset
                if resume_offset:
                    print(f"Total demos in dataset: {demo_count} ({demo_count - resume_offset} this session)")
                else:
                    print(f"Total demos recorded: {demo_count}")
                _refresh_label()

            # Check exit condition
            if args_cli.num_demos > 0 and demo_count >= args_cli.num_demos:
                print(f"All {demo_count} demos recorded. Exiting.")
                break

            # Handle reset
            if should_reset:
                print("Resetting environment...")
                # Covers the discard path too (button Y / R mid-ramp): the override must be gone
                # before env.reset(), or the next episode would open with the teleop still ignored.
                if return_to_rest is not None:
                    return_to_rest.stop()
                env.sim.reset()
                env.recorder_manager.reset()
                env.reset()
                teleop.reset()
                success_step_count = 0
                should_reset = False
                # requires_manual_arm modes go back to un-armed after every reset -- the next
                # episode isn't recorded, and the robot won't move, until X is pressed again.
                # Other modes stay always-armed (old behavior).
                recording_armed = not requires_manual_arm
                MOTION_GATE["enabled"] = recording_armed
                handover_stage_seen["value"] = None
                # Reset gripper states to open
                left_gripper_state = 1.0
                right_gripper_state = 1.0
                _refresh_label()
                print(f"Ready. Active arm: {active_arm.upper()}")
                if requires_manual_arm:
                    print("Robot held at rest pose. Press X to release it and start recording.")

            if env.sim.is_stopped():
                break

            rate_limiter.sleep(env)

    env.close()
    session_count = demo_count - resume_offset
    print(
        f"\nRecording done. {demo_count} successful demos in {dataset_path}"
        + (f" ({session_count} added this session, {resume_offset} were already there)." if resume_offset else ".")
    )
    if args_cli.num_demos > 0 and demo_count < args_cli.num_demos:
        print(
            f"That is short of --num_demos {args_cli.num_demos}. Re-run the same command with"
            " --resume to add the remaining"
            f" {args_cli.num_demos - demo_count} without losing what is already recorded."
        )

    if feedback_receiver is not None:
        _save_plot_once()


def save_sim_vs_real_plot(mirror_broadcaster, feedback_receiver) -> None:
    """Compare the sim joint positions this process broadcast against the real
    robot's actual joint feedback received back from mirror_bridge.py, and save a
    per-joint time-series plot (PNG) to the current working directory."""
    sim_history = mirror_broadcaster.history() if mirror_broadcaster is not None else []
    real_history = feedback_receiver.history()
    if not sim_history or not real_history:
        print("[MIRROR] No data collected for a sim-vs-real comparison plot (one or both histories are"
              " empty) -- skipping. Did mirror_bridge.py have --feedback-port set to match?")
        return

    # Minimum y-axis half-range per joint category -- prevents matplotlib's auto-scaling
    # from zooming into sensor noise or a small steady-state offset and making it look
    # like a large sim-vs-real gap. Chosen relative to each category's own meaningful
    # scale, not one-size-fits-all: 0.1 rad matches the arm joints' own handshake
    # tolerance elsewhere in this pipeline (the threshold already established as "worth
    # taking seriously"), while the gripper's entire physical range is only ~0.044 rad,
    # so it gets a proportionally smaller floor rather than being flattened by the arm's.
    ARM_AXIS_TOLERANCE = 0.1
    GRIPPER_AXIS_TOLERANCE = 0.005

    t0 = sim_history[0][0]
    joint_names = list(sim_history[0][1].keys())
    ncols = 4
    nrows = (len(joint_names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, name in enumerate(joint_names):
        ax = axes[idx // ncols][idx % ncols]
        sim_t = [t - t0 for t, j in sim_history if name in j]
        sim_v = [j[name] for t, j in sim_history if name in j]
        real_t = [t - t0 for t, j in real_history if name in j]
        real_v = [j[name] for t, j in real_history if name in j]
        ax.plot(sim_t, sim_v, label="sim", linewidth=1)
        ax.plot(real_t, real_v, label="real", linewidth=1, linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("s", fontsize=7)
        ax.set_ylabel("rad", fontsize=7)

        all_v = sim_v + real_v
        if all_v:
            tolerance = GRIPPER_AXIS_TOLERANCE if "finger" in name else ARM_AXIS_TOLERANCE
            data_min, data_max = min(all_v), max(all_v)
            center = (data_min + data_max) / 2
            half_range = max((data_max - data_min) / 2 * 1.1, tolerance)
            ax.set_ylim(center - half_range, center + half_range)

    for idx in range(len(joint_names), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Sim vs real joint positions (live mirroring session)")
    fig.tight_layout()
    out_path = os.path.join(os.getcwd(), f"sim_vs_real_{time.strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(out_path, dpi=120)
    print(f"[MIRROR] Saved sim-vs-real comparison plot to {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()