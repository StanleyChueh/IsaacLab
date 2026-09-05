"""
Evaluate a JOINT-SPACE-trained GR00T N1.7 policy inside Isaac Sim.

Sibling of eval_smolvla_jointspace.py, adapted for GR00T checkpoints instead of SmolVLA. This
script is otherwise POLICY-AGNOSTIC: everything Isaac-Sim-side (joint-space state extraction,
direct joint-target stepping, gripper binarization, task-mode handling, real-robot mirroring) is
identical to eval_smolvla_jointspace.py and untouched by which policy answers on the other end of
the wire -- it only ever speaks the small TCP step/reset/close protocol both *_server.py scripts
implement identically. Only gr00t_server.py itself (run separately, in its own conda env) knows
anything about GR00T. See gr00t_server.py's own docstring for why GR00T needs different serving
logic than smolvla_server.py despite the identical wire protocol (this checkpoint's actions are
relative deltas, not absolute targets, and GR00T normalizes/decodes them in its own processor
pipeline rather than the plain mean/std buffers SmolVLA uses).

The one thing that IS different here vs eval_smolvla_jointspace.py is _get_cameras(): this
checkpoint's own input_features were never renamed to slot names ("camera1"/"camera2"/...) at
train time -- its config.json's input_features are keyed by the actual env camera names
(observation.images.right_wrist_cam / .wrist_cam / .body_cam) -- so the dict this script sends
over the wire is keyed by those same real names instead of camera1/camera2/camera3. Everything
else about --cameras (comma-separated, in the order the env exposes them) is unchanged.

Run gr00t_server.py FIRST, in the 'lerobot-latest' conda env (transformers>=5, needed for GR00T's
Qwen3-VL backbone -- NOT the same env smolvla_server.py runs in). It stays up across client
disconnects, so one server serves any number of eval runs; only Ctrl-C stops it. Then this:

  Terminal 1 (lerobot-latest env) -- --task must be the dataset's task string VERBATIM (read it
  from the dataset's own meta/tasks.parquet; GR00T is language-conditioned, an unseen prompt such
  as "." silently degrades the policy). This checkpoint was fine-tuned from the SAME
  ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz dataset the sibling SmolVLA
  checkpoint uses, so the task string is the same one you would pass smolvla_server.py:
    cd ~/Stanley_ws/IsaacLab
    conda run -n lerobot-latest python \\
        scripts/imitation_learning/lerobot/gr00t_server.py \\
        --checkpoint ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz_gr00t \\
        --task "Pick up the Pringles can with the right arm, hand it to the left arm."

  Terminal 2 (Isaac Sim) -- --task is the GYM ENV and --task_mode the variant applied on top
  of it; BOTH must match what the demos were recorded with, because the env id alone does not
  determine the scene (see below):
    cd ~/Stanley_ws/IsaacLab
    ./isaaclab.sh -p scripts/imitation_learning/lerobot/eval_groot_jointspace.py \\
        --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \\
        --task_mode handover \\
        --num_rollouts 5 \\
        --horizon 300 \\
        --enable_cameras \\
        --cameras right_wrist_cam,wrist_cam,body_cam

--task_mode: on the CAN_TARGET_TASKS the manipulated object in the cfg is the base task's red
cube_2, and apply_task_mode() is the only thing that replaces it with the can -- along with the
per-mode object spawn range and success condition. Recording goes through
record_demos_openarm.py --task_mode, so a dataset's env id does NOT by itself describe the scene
it was recorded in; evaluating the same env id without the mode silently puts the policy in
front of a cube. This script refuses that combination rather than running it, the same way the
recording script does. Which mode a dataset used is readable from its own annotated HDF5:
data/demo_0/obs/datagen_info/subtask_term_signals -- grasp_right/handover/presented is
``handover``, grasp/lift is ``left`` or ``right`` (see openarm_task_modes.py's table).

State (16D by default): LJ1..LJ8 then RJ1..RJ8, i.e. openarm_left_joint1..7 + left gripper,
then openarm_right_joint1..7 + right gripper, extracted via BOTH_IDX below. The physics
Articulation's joint order is INTERLEAVED dual-arm (even indices 0,2,...,12 = left arm 1-7, odd
1,3,...,13 = right arm 1-7, then 14 / 16 = the left / right gripper's actuated finger),
confirmed against a live Isaac Sim robot.data.joint_names printout and matching
convert_hdf5_to_lerobot.py's own LJ_IDX / RJ_IDX / BOTH_IDX exactly. This deliberately does NOT
reuse eval_smolvla.py's _get_state() (joint_pos[:7]) -- a plain leading slice mixes left- and
right-arm values under this interleaved ordering and would feed the model a wrong state.

Action (16D by default): applied DIRECTLY as target joint positions for those same joints -- no
IK, no delta interpretation on THIS side of the wire (gr00t_server.py's postprocessor already
converts the checkpoint's native relative joint deltas back into these same absolute targets
before an action ever reaches this script -- see that server's docstring). The two gripper
columns are the one exception: they are binarized to fully open (0.044) or fully closed (0.0) at
--gripper_threshold_ratio of the open position, because the policy regresses that column
continuously and a part-closed jaw neither holds the can nor lets it through, while the teleop
demos it learned from only ever commanded those two values. Everything else is passed through
untouched. Each gripper's finger_joint2 gets the same target as its finger_joint1 (see
FOLLOWER_OF_LEADER): joint2 is a PhysX mimic of joint1 but is ALSO independently actuated by this
robot's "openarm_gripper" group, so leaving it out does not let the mimic carry it -- it pegs
joint2 at the untouched buffer value of 0.0, which is fully CLOSED, and shuts both hands for the
entire rollout.

For the same reason the whole joint-position-target buffer is seeded from the reset pose at the
top of every rollout: it starts as zeros and env.reset() does not reseed it, so any joint this
script does not command is driven to 0.0 rather than left alone.

The layout MUST match the --arms convert_hdf5_to_lerobot.py exported this checkpoint's dataset
with. Pass --arms left for an older 8D left-arm-only checkpoint; then only LJ_IDX is driven and
the right arm holds wherever env.reset() put it. A mismatch raises in _step_direct on the first
step rather than being silently truncated.

--cameras: comma-separated env camera names, IN THE SAME ORDER AND UNDER THE SAME NAMES this
checkpoint's dataset used -- unlike eval_smolvla_jointspace.py's --cameras (which maps env names
to policy slot names camera1/camera2/...), gr00t_server.py's protocol expects the dict keyed by
these exact names because that is how this checkpoint's own input_features are keyed (see
config.json on the HF repo). The default is this checkpoint's own three cameras.
MUST match this exact checkpoint's training camera keys. Sending a name gr00t_server.py's
checkpoint does not recognize is silent (no error, no crash) -- see _get_cameras().

MIRRORING THE ROLLOUT ONTO THE REAL ROBOT (--mirror_udp_port, off by default)
----------------------------------------------------------------------------
The same policy that is driving the sim can drive the real dual-arm follower at the same time,
over the identical two-process bridge a teleop recording session uses (scripts/tools/
record_demos_openarm.py --mirror_udp_port + lerobot_openarm/mirror_bridge.py). This process still
never talks to hardware: it only broadcasts the sim robot's joint positions as UDP JSON, and
mirror_bridge.py owns calibration, speed clamping, the startup handshake and the kill switch.
The shared broadcaster/listener/plotter live in scripts/tools/sim_mirror.py.

This is NOT the same thing as a script that runs the policy directly against the real robot's own
cameras and joint feedback. Here the policy sees ONLY the sim: sim cameras in, sim joints out, and
the real arm is a follower of sim's resulting pose. Every real-world difference (lighting, can
pose, contact) is therefore invisible to the policy, so treat this as a way to watch a sim rollout
play out on hardware, not as a real-world evaluation.

  Terminal 2 (Isaac Sim), same command as above plus:
        --mirror_udp_port 5557 --mirror_feedback_port 5558 --mirror_rate_hz 20

  Terminal 3 (lerobot-openarm env, real hardware) -- start this only ONCE the sim has printed its
  "[MIRROR] Holding this pose on the wire" prompt, NOT before. The bridge gives up if no packet
  arrives within 10s of connecting to the robot, and this script does not broadcast anything until
  it reaches that first hold, which is minutes after launch. Started at the right moment it sees
  packets immediately, prints a real-vs-sim pose comparison, and refuses to move until you type
  YES. Then answer the sim's Enter prompt once the real arm has settled:
    cd ~/Stanley_ws/lerobot_openarm
    python mirror_bridge.py --calibration calibration.json --udp-port 5557 \
        --model-path <path>/urdf --right-port can0 --left-port can1 \
        --max-joint-speed 0.3 --feedback-port 5558

--mirror_rate_hz paces the whole eval loop when mirroring. Without it this script runs as fast as
the policy server answers, which is fine for a sim-only eval but hands the follower a command
stream whose timing bears no relation to the ~20-30 Hz cadence the demos were recorded at; the
bridge's speed clamp would then be absorbing all of that discrepancy on its own.

Between rollouts the script broadcasts the post-reset pose and waits for Enter, because env.reset()
teleports the sim robot in one frame and the real arm needs seconds to follow at a safe speed. It
keeps broadcasting through that wait rather than going quiet, since silence past mirror_bridge.py's
--timeout-ms makes the bridge ramp down and disable the motors.

Gripper: what gets mirrored is the BINARIZED open/closed decision (see --gripper_threshold_ratio),
not sim's measured finger position -- sim's fingers stall against the rigid can at first contact
while the real compliant pads must keep closing past that point to actually hold it. mirror_bridge
maps those 0.0 / 0.044 values through calibration.json's open_raw/closed_raw.
"""

"""Launch Isaac Sim first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate a joint-space-action GR00T N1.7 policy in Isaac Lab via gr00t_server.py"
)
parser.add_argument("--task",        type=str, required=True, help="Gym env id")
parser.add_argument("--num_rollouts",type=int, default=5,   help="Number of evaluation episodes")
parser.add_argument("--horizon",     type=int, default=300,  help="Max steps per episode")
parser.add_argument("--port",        type=int, default=5556, help="Policy server TCP port")
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument(
    "--policy_server_timeout", type=float, default=600.0,
    help=(
        "Seconds to keep retrying the connection to gr00t_server.py before giving up; 0 waits"
        " indefinitely. The server binds its port only after the checkpoint finishes loading, so"
        " launching it and this script together used to fail outright with ConnectionRefusedError."
        " GR00T's 3B backbone makes a cold HF-cache first load slower than SmolVLA's, so raise this"
        " (or set it to 0) on a first-time download."
    ),
)
parser.add_argument(
    "--task_mode", type=str, default=None, choices=["left", "right", "handover"],
    help=(
        "Task-mode variant to apply to the env cfg -- MUST be the same one the demos were"
        " recorded and annotated with (record_demos_openarm.py --task_mode). The can-target"
        " tasks' own cfgs still describe the base task's red CUBE; apply_task_mode() is what"
        " swaps in the can, and also sets the per-mode object spawn range and success"
        " condition. Omitting it on such a task evaluates the policy against the wrong object"
        " entirely, so it is refused rather than run (see CAN_TARGET_TASKS)."
    ),
)
parser.add_argument(
    "--gripper_threshold_ratio", type=float, default=0.95,
    help=(
        "Binarize each gripper command at this fraction of the fully-open position"
        " (GRIPPER_OPEN_VAL = 0.044 m): a commanded value at or above the threshold becomes"
        " fully open, anything below becomes fully closed. The policy's gripper column is"
        " regressed continuously and lands mid-range whenever it is unsure, which reads as a"
        " half-shut jaw that neither grasps nor clears the object. 0 disables binarization and"
        " passes the raw commanded position through."
    ),
)
parser.add_argument(
    "--arms", type=str, default="both", choices=["both", "left"],
    help=(
        "Which joint layout the checkpoint was trained on -- MUST match the --arms used for "
        "convert_hdf5_to_lerobot.py on this checkpoint's dataset. 'both' (default) = 16D "
        "LJ1..LJ8+RJ1..RJ8; 'left' = 8D LJ1..LJ8 only, for older left-arm-only checkpoints. "
        "A mismatch is caught at the first step (see _step_direct), not silently ignored."
    ),
)
parser.add_argument(
    "--cameras", type=str, default="right_wrist_cam,wrist_cam,body_cam",
    help=(
        "Comma-separated env camera names, sent to gr00t_server.py UNDER THEIR OWN NAMES (unlike"
        " eval_smolvla_jointspace.py's --cameras, this is NOT remapped to camera1/camera2/... --"
        " see module docstring). The default is this checkpoint's own three training cameras."
        " MUST match this exact checkpoint's --rename_map (if any) at train time. Sending the"
        " wrong env camera is silent (no error, no crash)."
    ),
)
parser.add_argument(
    "--mirror_udp_port", type=int, default=0,
    help=(
        "If nonzero, broadcast the sim robot's joint positions (by name, radians) as a UDP JSON"
        " packet to <--mirror_udp_host>:<port> after every step, so a separate out-of-process"
        " bridge (lerobot_openarm/mirror_bridge.py, run with a matching --udp-port) can drive the"
        " REAL robot along with the rollout. Off by default. This process never talks to hardware."
        " See the MIRRORING section of the module docstring -- in particular that the policy still"
        " sees only the sim, so this is not a real-world evaluation."
    ),
)
parser.add_argument(
    "--mirror_udp_host", type=str, default="127.0.0.1",
    help="Destination host for --mirror_udp_port (and bind address for --mirror_feedback_port)."
    " Defaults to loopback; only change this if you know why.",
)
parser.add_argument(
    "--mirror_feedback_port", type=int, default=0,
    help=(
        "If nonzero, listen for UDP JSON packets from mirror_bridge.py's --feedback-port carrying"
        " the real robot's ACTUAL joint positions (already inverse-mapped to sim joint names), and"
        " save a per-joint sim-vs-real comparison plot on exit. Requires --mirror_udp_port. Still"
        " no hardware access here -- this only listens for numbers the bridge sends back."
    ),
)
parser.add_argument(
    "--mirror_rate_hz", type=float, default=20.0,
    help=(
        "Pace the eval loop to this rate while mirroring (ignored without --mirror_udp_port)."
        " Ungated, this script steps as fast as the policy server answers, which gives the real"
        " follower a command stream unrelated to the cadence the demos were recorded at. Match it"
        " to the --inference-hz you would use for a real-robot deployment script."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything else after Isaac Sim is up."""

import atexit
import os
import pickle
import random
import socket
import struct
import sys
import threading
import time

import carb
import carb.input
import omni.appwindow
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.stack  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_task_modes import (
    CAN_TARGET_TASKS,
    GRIPPER_OPEN_VAL,
    apply_task_mode,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# The sim-side half of the real-robot mirror bridge (--mirror_udp_port), shared verbatim with
# scripts/tools/record_demos_openarm.py so one mirror_bridge.py process serves either sim-side
# script over one wire format. Reached by path because scripts/ is a directory of standalone
# scripts, not an installed package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
from sim_mirror import JointFeedbackReceiver, JointMirrorBroadcaster, save_sim_vs_real_plot  # noqa: E402

# ── TCP client helpers (identical to eval_smolvla_jointspace.py, must match server) ────

def _recv_exactly(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Policy server closed connection")
        buf.extend(chunk)
    return bytes(buf)

def _send(sock, payload: bytes):
    sock.sendall(struct.pack(">I", len(payload)) + payload)

def _recv(sock) -> dict:
    raw_len = _recv_exactly(sock, 4)
    (n,) = struct.unpack(">I", raw_len)
    return pickle.loads(_recv_exactly(sock, n))

def policy_reset(sock):
    _send(sock, pickle.dumps({"cmd": "reset"}))
    _recv(sock)

def policy_step(sock, state: np.ndarray, cameras: dict) -> np.ndarray:
    _send(sock, pickle.dumps({"cmd": "step", "state": state, "cameras": cameras}))
    resp = _recv(sock)
    if "error" in resp:
        raise RuntimeError(f"Policy server error: {resp['error']}")
    return np.asarray(resp["action"], dtype=np.float32)

def policy_close(sock):
    _send(sock, pickle.dumps({"cmd": "close"}))
    _recv(sock)


# ── joint-space state/action (identical to eval_smolvla_jointspace.py) ─────────

# Matches convert_hdf5_to_lerobot.py's LJ_IDX / RJ_IDX / BOTH_IDX exactly. The physics
# Articulation's joint order is INTERLEAVED dual-arm: even indices 0,2,...,12 = left arm 1-7,
# odd indices 1,3,...,13 = right arm 1-7, then 14 = left gripper's actuated finger and 16 = the
# right one (15 / 17 are their PhysX-mimic partners, which follow automatically).
LJ_IDX = list(range(0, 14, 2)) + [14]
RJ_IDX = list(range(1, 14, 2)) + [16]
# Concatenation order is LJ1..LJ8 then RJ1..RJ8 -- the exact column order of the dataset's
# observation.state / action, and of the real robot's action dict.
BOTH_IDX = LJ_IDX + RJ_IDX

# The joints the POLICY speaks for, and the exact column order of the state it sends.
DRIVEN_IDX = BOTH_IDX if args_cli.arms == "both" else LJ_IDX

# Each hand's finger_joint2 is a PhysX mimic of joint1 AND an independently actuated member of
# the "openarm_gripper" group (stiffness 3000) -- see OPENARM_BI_CFG in
# isaaclab_assets/robots/openarm.py, which spells out that "driving joint1 alone while joint2 is
# actuated would peg joint2 at its default and genuinely fight the mimic". The dataset carries
# only the leader joint per hand (convert_hdf5_to_lerobot.py exports LJ8/RJ8 = joint1, since the
# follower holds no independent information), so the follower has no action column of its own and
# has to be commanded the leader's value -- exactly what the tasks' BinaryJointPositionActionCfg
# does by naming both finger joints. Without this the follower keeps the target every joint this
# script never writes keeps, which is 0.0 (Articulation._initialize_impl zeroes the buffer and
# nothing reseeds it), and 0.0 is a gripper's FULLY CLOSED position -- so both hands sit shut for
# the whole rollout no matter what the policy commands.
FOLLOWER_OF_LEADER = {
    "openarm_left_finger_joint1": "openarm_left_finger_joint2",
    "openarm_right_finger_joint1": "openarm_right_finger_joint2",
}


# Commanded gripper positions at or above this are snapped fully open, below it fully closed --
# the same two values the tasks' own BinaryJointPositionActionCfg commands (open_command_expr
# 0.044 / close_command_expr 0.0), so a binarized rollout drives the jaws exactly where a teleop
# demo drove them. 0.0 turns binarization off (see --gripper_threshold_ratio).
GRIPPER_BINARY_THRESHOLD = args_cli.gripper_threshold_ratio * GRIPPER_OPEN_VAL
GRIPPER_CLOSED_VAL = 0.0


def _binarize_grippers(target_q: np.ndarray, gripper_cols: list[int]) -> np.ndarray:
    """Snap every gripper column of *target_q* to fully open or fully closed.

    The dataset's gripper column is an absolute finger position in metres that the policy
    regresses continuously, so its output drifts through the middle of the 0..0.044 range -- a
    part-closed jaw, which on this gripper is the one command that neither holds the can nor
    lets it through. The source demos themselves are effectively two-valued (teleop drives the
    task's BinaryJointPositionActionCfg), so snapping to those two values recovers the intent
    rather than discarding information.

    The default 0.8 ratio is NOT a midpoint: only a command within 20% of fully open is read as
    "open", so any partial retraction counts as a close. That asymmetry is deliberate -- an
    under-committed close drops the can, whereas an over-eager close on this task's 60 mm can
    still leaves the jaws well short of each other.
    """
    if not gripper_cols or args_cli.gripper_threshold_ratio <= 0.0:
        return target_q
    out = target_q.copy()
    cols = np.asarray(gripper_cols)
    out[cols] = np.where(out[cols] >= GRIPPER_BINARY_THRESHOLD, GRIPPER_OPEN_VAL, GRIPPER_CLOSED_VAL)
    return out


def _resolve_grippers(env) -> tuple[list[int], list[int]]:
    """Map each gripper column of the action vector to the articulation joint index of that
    gripper's mimic follower. Resolved BY NAME off the live articulation rather than hardcoded
    alongside LJ_IDX, so a USD whose joint order differs fails loudly here instead of quietly
    driving some unrelated joint. Returns (action columns, follower joint indices)."""
    names = list(env.scene["robot"].data.joint_names)
    cols, followers = [], []
    for col, joint_idx in enumerate(DRIVEN_IDX):
        follower = FOLLOWER_OF_LEADER.get(names[joint_idx])
        if follower is None:
            continue
        if follower not in names:
            raise RuntimeError(
                f"'{names[joint_idx]}' is driven but its mimic follower '{follower}' is not a"
                f" joint of this robot. Joints: {names}"
            )
        cols.append(col)
        followers.append(names.index(follower))
    return cols, followers


def _get_state(env) -> np.ndarray:
    """Return the joint-space state in DRIVEN_IDX order (16D LJ1..LJ8+RJ1..RJ8 by default,
    8D LJ-only under --arms left), matching the exact index order the training dataset used --
    see module docstring for why this is NOT a plain joint_pos[:n] slice."""
    joint_pos = env.scene["robot"].data.joint_pos[0]   # (num_joints,)
    return joint_pos[DRIVEN_IDX].cpu().numpy().astype(np.float32)


def _get_cameras(obs_dict: dict, camera_names: list[str]) -> dict:
    """Extract camera images as uint8 HWC numpy arrays, keyed by their OWN env camera name.

    Unlike eval_smolvla_jointspace.py's _get_cameras() (which remaps to policy slot names
    camera1/camera2/...), gr00t_server.py's checkpoint keeps the real camera names as its
    input_features keys (see module docstring), so no remapping happens here -- the env key
    IS the policy key.
    """
    policy_obs = obs_dict["policy"]
    cameras = {}
    for env_key in camera_names:
        if env_key in policy_obs:
            img = policy_obs[env_key]        # (1, H, W, 3) uint8
            cameras[env_key] = img.squeeze(0).cpu().numpy().astype(np.uint8)
        else:
            print(f"[warn] Camera '{env_key}' not found in policy observations")
    return cameras


def _step_direct(env, target_q: np.ndarray, apply_idx: list[int], gripper_cols: list[int]) -> dict:
    """Advance the sim by one env "step" worth of physics (env.cfg.decimation substeps), driving
    the DRIVEN_IDX joints directly instead of going through the env's IK action term -- see
    module docstring for the exact ManagerBasedRLEnv.step() sequence this replicates. Returns a
    fresh observation dict in the same {"policy": {...}} shape env.step()/env.reset() return.
    """
    if target_q.shape[0] != len(DRIVEN_IDX):
        raise ValueError(
            f"Policy returned a {target_q.shape[0]}D action but --arms {args_cli.arms} drives"
            f" {len(DRIVEN_IDX)} joints. The checkpoint's action dim must match the layout"
            " convert_hdf5_to_lerobot.py exported its dataset with: 16 = --arms both"
            " (LJ1..LJ8+RJ1..RJ8), 8 = --arms left. Re-run with the matching --arms."
        )
    robot = env.scene["robot"]
    target_q = _binarize_grippers(target_q, gripper_cols)
    # Append each gripper leader's value again, once per follower joint -- see FOLLOWER_OF_LEADER.
    # Done AFTER binarizing so both jaws of a hand get the same snapped value.
    full_q = np.concatenate([target_q, target_q[gripper_cols]]) if gripper_cols else target_q
    target_t = torch.as_tensor(full_q, dtype=torch.float32, device=env.device).unsqueeze(0)

    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        robot.set_joint_position_target(target_t, joint_ids=apply_idx)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if (env._sim_step_counter % env.cfg.sim.render_interval == 0) and is_rendering:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)

    # ManagerBasedRLEnv.step()'s post-physics counter bump, which this function is standing in for.
    # Not cosmetic: the hand-over stage machine (openarm_task_modes._handover_tick) reads
    # common_step_counter to make itself idempotent within a step -- both the subtask_terms
    # observations and the success condition call it every step, and only the first call may
    # advance the bookkeeping. With the counter frozen, that guard reads every call for the rest of
    # the process as a repeat of the same step: the machine ticks ONCE, during the very first
    # reset's observation compute with nothing grasped, and stays at stage 0 forever. Every
    # --task_mode handover rollout then reports "failed" no matter how well the pass goes, because
    # handover_success is asking a counter that stopped moving. episode_length_buf is bumped
    # alongside it for the same reason step() does -- so anything reading "how far into the episode
    # are we" sees the truth.
    env.common_step_counter += 1
    env.episode_length_buf += 1

    return env.observation_manager.compute()


# ── real-robot mirroring helpers (only used with --mirror_udp_port) ────────────

class RateLimiter:
    """Pace the eval loop to a fixed rate, rendering the viewport while waiting.

    Only used while mirroring. On its own this script steps as fast as the policy server answers,
    which is exactly right for a sim-only eval but hands the real follower a command stream whose
    timing has nothing to do with the cadence the demos were recorded at -- mirror_bridge.py's
    speed clamp would then be absorbing the entire discrepancy by itself. Renders rather than
    dead-sleeping through the wait so the viewport stays live (same reason record_demos_openarm.py's
    own RateLimiter does).
    """

    def __init__(self, hz: float):
        self._period = 1.0 / hz
        self._next = time.time() + self._period

    def sleep(self, env):
        is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
        while True:
            remaining = self._next - time.time()
            if remaining <= 0.0:
                break
            time.sleep(min(0.005, remaining))
            if is_rendering:
                env.sim.render()
        self._next += self._period
        # Fell behind (a slow policy step, a long render): re-base off now instead of trying to
        # catch up with a burst of un-paced steps, which is the one thing the real arm must not see.
        if self._next < time.time():
            self._next = time.time() + self._period


class EnterWaiter:
    """Background 'press Enter' watcher, so the caller can keep broadcasting while it waits.

    A failed read is NOT treated as a press. When stdin is closed or redirected (nohup, a
    launcher that hands the process /dev/null, a pipe that has already ended) input() returns
    EOF immediately and forever; counting that as "the operator pressed Enter" would silently
    skip the very wait it exists to enforce and start the rollout with the real arm still
    seconds behind sim. Instead the waiter marks itself unusable and simply never fires, leaving
    the Isaac-window key as the way in -- see _mirror_hold_until_enter, which says so out loud
    rather than letting the hold look like it is waiting on a key that can never arrive.
    """

    def __init__(self):
        self._done = threading.Event()
        self._usable = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            input()
        except (EOFError, OSError):
            self._usable = False
            return
        self._done.set()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def stdin_unusable(self) -> bool:
        return not self._usable


def _mirror_hold_until_enter(env, mirror, rate_hz: float, message: str, kb: "KeyboardHandler") -> bool:
    """Broadcast the robot's current, static pose until the operator presses Enter.

    env.reset() teleports the sim robot to its reset pose in a single frame; the real follower can
    only get there at mirror_bridge.py's --max-joint-speed, which takes seconds. Starting the
    policy loop immediately would run the opening of every rollout against an arm still crawling
    toward the start pose. Keeping the broadcast going through the wait (rather than simply not
    sending) is what lets the bridge close that gap at all -- silence for longer than its
    --timeout-ms is its signal to ramp down and disable the motors.

    No gripper state is passed, so the broadcaster mirrors the measured finger position: during a
    hold there is no commanded open/close decision to forward, only the pose actually being held.

    Returns False if the operator asked to quit instead of starting (Q in the Isaac window, which
    is also watched here -- otherwise the only way out of a hold would be the Enter it is waiting
    for, i.e. starting the very rollout you wanted to abandon).
    """
    print(f"\n[MIRROR] {message}")
    print("[MIRROR] Holding this pose on the wire. Once the real arm has settled on it, start the"
          " rollout with EITHER Enter in this terminal OR Enter/Space in the Isaac Sim window"
          " (whichever has your keyboard). Q in the Isaac window quits.")
    waiter = EnterWaiter()
    limiter = RateLimiter(rate_hz)
    warned_stdin = False
    while not (waiter.done or kb.consume_start()):
        if kb.quit_requested:
            print("[MIRROR] Quit requested during the hold -- not starting this rollout.")
            return False
        if waiter.stdin_unusable and not warned_stdin:
            warned_stdin = True
            print("[MIRROR] stdin is not readable in this process (redirected or closed), so the"
                  " terminal's Enter cannot reach it. Use Enter/Space in the Isaac Sim window.")
        mirror.broadcast()
        limiter.sleep(env)
    print("[MIRROR] Starting rollout.")
    return True


# ── keyboard reset (identical to eval_smolvla_jointspace.py) ───────────────────

class NullKeyboardHandler:
    """Stand-in for KeyboardHandler under --headless.

    omni.appwindow.get_default_app_window() still returns an object headless (a WindowType.VIRTUAL
    window rather than raising or returning None), but that virtual window has no real OS-backed
    input device behind it -- get_keyboard()/subscribe_to_keyboard_events is at best inert (no key
    ever fires) and on some Kit versions hands back a null/invalid keyboard handle instead. Rather
    than rely on that failing loudly, follow the same guard scripts/imitation_learning/
    isaaclab_mimic/annotate_demos.py uses: skip constructing the real keyboard device entirely when
    headless. There is no window to read R/Q/Enter/Space from anyway; stop a headless run between
    trials with Ctrl-C instead.
    """

    quit_requested = False

    def consume_reset(self) -> bool:
        return False

    def consume_start(self) -> bool:
        return False


class KeyboardHandler:
    """Listens for R = manual reset, Q = quit."""

    def __init__(self):
        self.reset_requested = False
        self.quit_requested  = False
        self.start_requested = False
        app_window = omni.appwindow.get_default_app_window()
        self._keyboard = app_window.get_keyboard()
        self._input    = carb.input.acquire_input_interface()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_key
        )
        print("[keyboard] Press  R  to reset episode early,  Q  to quit,"
              "  Enter/Space to start a held rollout (--mirror_udp_port only).")

    def _on_key(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.R:
                self.reset_requested = True
                print("[keyboard] Manual reset requested.")
            elif event.input == carb.input.KeyboardInput.Q:
                self.quit_requested = True
                print("[keyboard] Quit requested.")
            elif event.input in (carb.input.KeyboardInput.ENTER, carb.input.KeyboardInput.SPACE):
                # Second way to answer the mirror hold, for when the terminal's stdin is not
                # reaching this process (the Isaac Sim app owns the foreground window, and whether
                # Enter typed in the launching terminal arrives here depends on how the session was
                # started). Ignored outside a hold -- consume_start() is only polled there.
                self.start_requested = True
                print("[keyboard] Start requested.")
        return True

    def consume_reset(self) -> bool:
        if self.reset_requested:
            self.reset_requested = False
            return True
        return False

    def consume_start(self) -> bool:
        if self.start_requested:
            self.start_requested = False
            return True
        return False


# ── rollout ────────────────────────────────────────────────────────────────────

def rollout(env, sock, success_term, horizon: int, kb: "KeyboardHandler", camera_names: list[str],
            apply_idx: list[int], gripper_cols: list[int], gripper_labels: list[str],
            mirror=None, limiter: "RateLimiter | None" = None,
            feedback_receiver=None, trial: int | None = None) -> bool | None:
    obs_dict, _ = env.reset()
    policy_reset(sock)

    # Seed the FULL joint-position-target buffer from the pose reset() just produced. That buffer
    # is zeros until something writes it (Articulation._initialize_impl) and env.reset() does not
    # reseed it, so every joint this script never commands is actively driven to 0.0 rather than
    # left where it is -- an arbitrary pose for an arm, and fully CLOSED for a gripper. The joints
    # we do command are overwritten every substep below, so this only affects the rest.
    robot = env.scene["robot"]
    robot.set_joint_position_target(robot.data.joint_pos.clone())

    # Let the real arm walk to the pose reset() just teleported the sim one to, before any policy
    # action is sent. See _mirror_hold_until_enter for why this waits rather than simply starting.
    if mirror is not None and not _mirror_hold_until_enter(
        env, mirror, args_cli.mirror_rate_hz,
        "Sim has been reset to the rollout start pose.", kb,
    ):
        # None, not False: this rollout never ran a single policy step, so scoring it as a failure
        # would put a trial the operator deliberately abandoned into the success rate.
        return None

    # Previous open/closed decision per hand, so the report can flag the transitions -- over a
    # 300-step rollout the two or three steps where a jaw actually changes state are the whole
    # story and are otherwise invisible in a wall of identical lines.
    prev_open: list[bool] | None = None

    # Timed from here, after the hold: the hold is the real arm catching up to a pose sim reached
    # instantly, so including it would open every plot with a multi-second interval where the two
    # legitimately disagree and squash the rollout itself into the right-hand edge.
    rollout_t0 = time.time()
    try:
        return _rollout_steps(env, sock, success_term, horizon, kb, camera_names, apply_idx,
                              gripper_cols, gripper_labels, mirror, limiter, prev_open, obs_dict)
    finally:
        if feedback_receiver is not None and mirror is not None:
            save_sim_vs_real_plot(
                mirror, feedback_receiver,
                t_start=rollout_t0, t_end=time.time(),
                title=f"Sim vs real joint positions -- rollout {trial}",
                out_path=os.path.join(
                    os.getcwd(),
                    f"sim_vs_real_rollout{trial}_{time.strftime('%Y%m%d_%H%M%S')}.png",
                ),
            )


def _rollout_steps(env, sock, success_term, horizon: int, kb: "KeyboardHandler",
                   camera_names: list[str], apply_idx: list[int], gripper_cols: list[int],
                   gripper_labels: list[str], mirror, limiter, prev_open, obs_dict) -> bool:
    """The policy loop itself. Split out of rollout() only so the per-rollout plot can be saved
    from a finally block covering every way the loop exits -- success, horizon, manual reset --
    without repeating the call at each return."""
    for step in range(horizon):
        if kb.consume_reset():
            print(f"  [step {step+1:3d}] manually reset")
            return False

        state   = _get_state(env)
        cameras = _get_cameras(obs_dict, camera_names)

        if not cameras:
            print("[warn] No camera images found in obs — check image_obs_list config")
            break

        policy_action = policy_step(sock, state, cameras)   # (n,) absolute joint targets

        # Report what the gripper columns were commanded BEFORE _step_direct binarizes them, so
        # both the policy's raw regressed value and the open/closed call it lands on are visible.
        # Compared against the same GRIPPER_BINARY_THRESHOLD _binarize_grippers uses, so the
        # printed decision cannot disagree with the one actually driven into the sim.
        if gripper_cols:
            binarizing = args_cli.gripper_threshold_ratio > 0.0
            open_now = [bool(policy_action[c] >= GRIPPER_BINARY_THRESHOLD) for c in gripper_cols]
            fields = [
                f"{label} cmd={float(policy_action[col]):+.4f}"
                + (f" -> {'OPEN ' if is_open else 'CLOSE'}" if binarizing else " (raw, not binarized)")
                for label, col, is_open in zip(gripper_labels, gripper_cols, open_now)
            ]
            changed = binarizing and prev_open is not None and open_now != prev_open
            print(
                f"  [step {step+1:3d}] gripper  " + "  |  ".join(fields)
                + ("   <-- CHANGED" if changed else "")
            )
            prev_open = open_now

        obs_dict = _step_direct(env, policy_action, apply_idx, gripper_cols)

        if mirror is not None:
            # Forward the SAME binarized open/closed decision _step_direct just drove the sim
            # with, not the measured finger position: sim's fingers stall against the rigid can
            # at first contact (correct rigid-body physics), while the real gripper's compliant
            # pads have to keep closing well past that point to actually hold it -- mirroring the
            # measured value would cap the real gripper at sim's stopping point. See
            # JointMirrorBroadcaster.broadcast, which this argument exists for. With
            # --gripper_threshold_ratio 0 there is no binary decision to forward, so None falls
            # back to mirroring the measured position, which is then what sim is really doing.
            grip_state = {"left_gripper_state": None, "right_gripper_state": None}
            if gripper_cols and args_cli.gripper_threshold_ratio > 0.0:
                for label, is_open in zip(gripper_labels, open_now):
                    grip_state[f"{label}_gripper_state"] = (
                        GRIPPER_OPEN_VAL if is_open else GRIPPER_CLOSED_VAL
                    )
            mirror.broadcast(**grip_state)
            limiter.sleep(env)

        if bool(success_term.func(env, **success_term.params)[0]):
            print(f"  [step {step+1:3d}] SUCCESS")
            return True

    print(f"  [step {step+1:3d}] failed")
    return False


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )

    # Same refusal record_demos_openarm.py makes, for the same reason: on these tasks the object
    # is a red cube until apply_task_mode() replaces it with the can, and a rollout against the
    # wrong object looks completely normal apart from failing every episode.
    bare_task = args_cli.task.split(":")[-1]
    if args_cli.task_mode is None and bare_task in CAN_TARGET_TASKS:
        raise SystemExit(
            f"'{args_cli.task}' targets the can, but that swap only happens via --task_mode,"
            " which was not passed -- this run would have evaluated against the base task's"
            " CUBE. Re-run with the SAME --task_mode the demos were recorded with"
            " (left|right|handover)."
        )

    # Task mode first: it rewrites terminations.success, so it has to land before the
    # success_term is lifted off the cfg below.
    if args_cli.task_mode is not None:
        apply_task_mode(env_cfg, args_cli.task_mode)

    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None

    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    print(f"\nEnv obs keys   : {list(env.observation_space.spaces.keys())}")
    print("Action mode    : DIRECT joint-space (bypassing the env's own IK action term)")
    print(f"Joint layout   : --arms {args_cli.arms} -> {len(DRIVEN_IDX)}D state/action")
    print(f"Task mode      : {args_cli.task_mode or 'none (base task cfg as-is)'}")

    gripper_cols, follower_joints = _resolve_grippers(env)
    apply_idx = list(DRIVEN_IDX) + follower_joints
    joint_names = list(env.scene["robot"].data.joint_names)
    print(
        "Gripper binary : "
        + (
            f"open>={GRIPPER_BINARY_THRESHOLD:.4f} m"
            f" ({args_cli.gripper_threshold_ratio:g} x {GRIPPER_OPEN_VAL} open),"
            f" else {GRIPPER_CLOSED_VAL} closed"
            if args_cli.gripper_threshold_ratio > 0.0
            else "disabled (raw commanded position)"
        )
    )
    # "openarm_left_finger_joint1" -> "left": the hand each gripper column speaks for, for the
    # per-step report in rollout().
    gripper_labels = [
        joint_names[DRIVEN_IDX[c]].removeprefix("openarm_").removesuffix("_finger_joint1")
        for c in gripper_cols
    ]
    print(
        "Gripper mimic  : "
        + ", ".join(
            f"{joint_names[DRIVEN_IDX[c]]} -> {joint_names[j]}"
            for c, j in zip(gripper_cols, follower_joints)
        )
    )

    # ── Real-robot mirroring (opt-in, off by default) ──────────────────────────
    mirror, limiter = None, None
    if args_cli.mirror_udp_port:
        # rollout() forwards each hand's binarized command as broadcast()'s
        # <side>_gripper_state keyword, so the labels resolved off the live articulation above
        # have to be exactly the two sides that keyword exists for. Checked here, before the
        # policy server is even connected, rather than as a TypeError on the first step of a
        # rollout that is already moving the real arm.
        if set(gripper_labels) - {"left", "right"}:
            raise SystemExit(
                f"--mirror_udp_port cannot forward gripper commands for {gripper_labels}: this"
                " robot's gripper joints did not resolve to the 'left'/'right' sides"
                " JointMirrorBroadcaster.broadcast() takes. Re-run without mirroring, or fix the"
                " joint naming."
            )
        mirror = JointMirrorBroadcaster(
            robot=env.scene["robot"], host=args_cli.mirror_udp_host, port=args_cli.mirror_udp_port
        )
        limiter = RateLimiter(args_cli.mirror_rate_hz)
        print(
            f"[MIRROR] Eval loop paced to {args_cli.mirror_rate_hz:g} Hz while mirroring.\n"
            "[MIRROR] SAFETY: this moves the REAL robot along with the rollout. mirror_bridge.py"
            " owns the calibration, the speed clamp, the startup handshake and the 'q'+Enter kill"
            " switch -- keep emergency_disable.py within reach anyway.\n"
            "[MIRROR] Start mirror_bridge.py now if it isn't already running; it needs packets"
            " from here before it will do anything."
        )

    feedback_receiver = None
    save_plot_once = None
    if args_cli.mirror_feedback_port:
        if mirror is None:
            raise SystemExit(
                "--mirror_feedback_port needs --mirror_udp_port: the plot compares what THIS"
                " process broadcast against what the bridge sent back, and without the broadcast"
                " there is nothing to compare (and no bridge running to send anything)."
            )
        feedback_receiver = JointFeedbackReceiver(
            host=args_cli.mirror_udp_host, port=args_cli.mirror_feedback_port
        )

        # Ctrl+C under Isaac Sim tears the process down without raising a KeyboardInterrupt that
        # reaches code after the main loop, so the plot has to be hung off interpreter shutdown --
        # same reasoning, and the same guard against double-running, as record_demos_openarm.py.
        plot_state = {"saved": False}

        def save_plot_once():  # noqa: F811
            if plot_state["saved"]:
                return
            plot_state["saved"] = True
            feedback_receiver.stop()
            save_sim_vs_real_plot(mirror, feedback_receiver)

        atexit.register(save_plot_once)

    # Retry rather than die on the first refusal. gr00t_server.py binds its port only AFTER
    # GrootPolicy.from_pretrained() returns, which on a cold HF cache is minutes (a 3B backbone,
    # slower than SmolVLA's) -- so "the server is running" and "the server is accepting
    # connections" are two very different moments, and a single connect() attempt turns that gap
    # into a crash that also strands whatever else was started alongside this run (the mirror
    # bridge waiting for packets that now never come).
    print(f"\nConnecting to policy server on port {args_cli.port} …")
    deadline = None if args_cli.policy_server_timeout <= 0 else time.time() + args_cli.policy_server_timeout
    attempt = 0
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(("127.0.0.1", args_cli.port))
            break
        except ConnectionRefusedError:
            sock.close()
            attempt += 1
            if deadline is not None and time.time() > deadline:
                raise SystemExit(
                    f"Policy server did not start accepting connections on port {args_cli.port}"
                    f" within {args_cli.policy_server_timeout:g}s. Start gr00t_server.py first and"
                    " wait for its '[server] Listening on 127.0.0.1:<port>' line -- it prints that"
                    " only once the checkpoint has finished loading. Raise"
                    " --policy_server_timeout (or set it to 0 to wait indefinitely) for a slow"
                    " first-time checkpoint download."
                )
            if attempt == 1 or attempt % 10 == 0:
                print(f"  … not accepting connections yet (attempt {attempt}). Waiting for"
                      " gr00t_server.py to finish loading the checkpoint.")
            time.sleep(1.0)
    print("Connected.\n")

    # --headless has no OS-backed window/keyboard behind it (see NullKeyboardHandler) -- only
    # construct the real omni.appwindow/carb.input-backed handler in GUI mode.
    if args_cli.headless:
        print("[keyboard] --headless: R/Q/Enter/Space hotkeys are unavailable. Use Ctrl-C to stop"
              " between trials.")
        kb = NullKeyboardHandler()
    else:
        kb = KeyboardHandler()

    camera_names = [name.strip() for name in args_cli.cameras.split(",") if name.strip()]
    print(f"[INFO] Camera keys (sent to gr00t_server.py verbatim, no slot remapping): {camera_names}")

    results = []
    for trial in range(args_cli.num_rollouts):
        if kb.quit_requested:
            print("[keyboard] Quitting early.")
            break
        print(f"── Trial {trial + 1}/{args_cli.num_rollouts} ──────────────────")
        success = rollout(env, sock, success_term, args_cli.horizon, kb, camera_names,
                          apply_idx, gripper_cols, gripper_labels, mirror, limiter,
                          feedback_receiver, trial + 1)
        if success is None:  # aborted before it started -- not a trial, see rollout()
            break
        results.append(success)

    n_ok = results.count(True)
    print(f"\n{'='*50}")
    print(f"Success: {n_ok} / {len(results)}")
    if results:
        print(f"Rate   : {n_ok / len(results):.1%}")
    print(f"Results: {results}")

    if mirror is not None:
        print(
            "\n[MIRROR] No longer broadcasting. mirror_bridge.py will hit its --timeout-ms and"
            " ramp the real arm down to a hold on its own; watch it do so before walking away."
        )
    if save_plot_once is not None:
        save_plot_once()

    policy_close(sock)
    sock.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
