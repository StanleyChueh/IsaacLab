# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to replay demonstrations with Isaac Lab environments.

Besides the datasets produced by ``record_demos.py``, this also replays the dual-arm OpenArm
datasets produced by ``record_demos_openarm.py``. Those need extra handling because that script
reshapes the task's action space before recording, so the recorded action vector no longer matches
the action space the task defines by default -- see ``--action_mode`` and
``swap_to_openarm_joint_actions`` below.

Usage:

  # standard single-arm dataset (task's own action space)
  ./isaaclab.sh -p scripts/tools/replay_demos.py --dataset_file datasets/dataset.hdf5

  # dual-arm OpenArm dataset recorded by record_demos_openarm.py (--enable_cameras is needed
  # because the OpenArm pick-up tasks have camera observation terms)
  ./isaaclab.sh -p scripts/tools/replay_demos.py \
    --dataset_file logs/demos/pickup.hdf5 \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \
    --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)
parser.add_argument("--dataset_file", type=str, default="datasets/dataset.hdf5", help="Dataset file to be replayed.")
parser.add_argument(
    "--task_mode",
    type=str,
    default=None,
    choices=["left", "right", "handover"],
    help=(
        "OpenArm pick-up tasks only: which task variant the dataset was recorded under. The mode"
        " swaps the task's default cube_2 for the can, so replaying a mode-recorded dataset"
        " against the plain task fails with KeyError: 'cube_2' -- the scene contains an object the"
        " recording has no state for. Normally you do NOT need to pass this: the object is"
        " detected from the dataset and the scene is fixed up automatically. Pass it to be"
        " explicit, or if detection guesses wrong."
    ),
)
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help=(
        "Validate if the states, if available, match between loaded from datasets and replayed. Only valid if"
        " --num_envs is 1."
    ),
)
parser.add_argument(
    "--validate_success_rate",
    action="store_true",
    default=False,
    help="Validate the replay success rate using the task environment termination criteria",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--action_mode",
    type=str,
    default="auto",
    choices=["auto", "task_default", "openarm_joint_abs", "openarm_joint_delta"],
    help=(
        "How to interpret the recorded action vector. 'task_default' feeds it straight into the"
        " action space the task config defines (correct for record_demos.py datasets, and for"
        " record_demos_openarm.py's 14D keyboard/vr_ros2 datasets). The two 'openarm_*' modes are"
        " for record_demos_openarm.py's 16D dual-arm datasets, whose arm actions are joint-space"
        " and therefore need the task's IK action terms swapped out before the env is created:"
        " 'openarm_joint_delta' for --teleop_device vr_joint_ros2 (per-arm joint offsets from the"
        " default pose + binary +-1 grippers), 'openarm_joint_abs' for --teleop_device"
        " vr_joint_ros2_native (absolute joint positions + absolute finger positions in meters)."
        " 'auto' (default) picks between them from the recorded action width and gripper column"
        " value range -- see detect_openarm_action_mode."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# args_cli.headless = True

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version
    # installed by IsaacLab and not the one installed by Isaac Sim.
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import h5py
import numpy as np
import os

import gymnasium as gym
import torch

import isaaclab.envs.mdp as mdp
from isaaclab.assets.articulation import Articulation
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

is_paused = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> (bool, str):
    """Compare states from dataset and runtime.

    Args:
        state_from_dataset: State from dataset.
        runtime_state: State from runtime.
        runtime_env_index: Index of the environment in the runtime states to be compared.

    Returns:
        bool: True if states match, False otherwise.
        str: Log message if states don't match.
    """
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                # EpisodeData.get_state keeps the (single-env) leading dimension the states were
                # recorded with, while the runtime state has already been indexed down to one env
                # above -- drop it so the two are compared value-by-value rather than by shape.
                if dataset_asset_state.ndim > runtime_asset_state.ndim:
                    dataset_asset_state = dataset_asset_state[0]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(f"State shape of {state_name} for asset {asset_name} don't match")
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log


# ─── OpenArm dual-arm dataset support ─────────────────────────────────────────
# record_demos_openarm.py rewrites the task's action space before recording when its
# --teleop_device is one of the joint-space VR modes, so a dataset recorded that way can only be
# replayed against the same rewritten action space. The constants/regexes below are deliberate
# duplicates of that script's (record_demos_openarm.py can't be imported here -- it parses argv
# and launches its own AppLauncher at import time); keep them in sync with it.
OPENARM_LEFT_ARM_JOINT_REGEX = ["openarm_left_joint[1-7]"]
OPENARM_RIGHT_ARM_JOINT_REGEX = ["openarm_right_joint[1-7]"]
OPENARM_LEFT_FINGER_JOINT_NAMES = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
OPENARM_RIGHT_FINGER_JOINT_NAMES = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]
OPENARM_JOINT_ACTION_SCALE = 1.0
# Layout of the 16D joint-space action vector (record_demos_openarm.py's TOTAL_ACTION_DIM_JOINT):
#   [0:7] left arm | [7] left gripper | [8:15] right arm | [15] right gripper
OPENARM_JOINT_ACTION_DIM = 16
OPENARM_LEFT_GRIPPER_IDX = 7
OPENARM_RIGHT_GRIPPER_IDX = 15
# Fully-open finger position (meters) -- matches the task's BinaryJointPositionActionCfg
# open_command_expr, and is the upper end of the absolute gripper values recorded in
# vr_joint_ros2_native mode.
OPENARM_GRIPPER_OPEN_VAL = 0.044


class OpenArmGripperJointPositionAction(ActionTerm):
    """1D absolute finger-position action driving both jaws of one OpenArm gripper.

    Needed to replay --teleop_device vr_joint_ros2_native datasets: there the recorded gripper
    column is the *measured* position of ``*_finger_joint1`` in meters (0 closed .. 0.044 open),
    not the +-1 binary command the task's own BinaryJointPositionActionCfg expects, so that term
    would read every value in that range as "close". A plain JointPositionActionCfg over both
    finger joints can't be used instead because it would be 2D and widen the action vector past
    the recorded 16D -- hence this term: one action value, written to both jaws (joint2 is a PhysX
    mimic of joint1 *and* independently actuated, so leaving it uncommanded pins it at its default
    while joint1 moves -- see the openarm task cfg's gripper action comment).
    """

    cfg: "OpenArmGripperJointPositionActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "OpenArmGripperJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._raw_actions.repeat(1, len(self._joint_ids)), joint_ids=self._joint_ids
        )


@configclass
class OpenArmGripperJointPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`OpenArmGripperJointPositionAction`."""

    class_type: type[ActionTerm] = OpenArmGripperJointPositionAction
    joint_names: list[str] = []
    """Both finger joints of one gripper; all of them receive the single action value."""


def peek_episode_actions(dataset_file: str, episode_name: str) -> np.ndarray | None:
    """Read one episode's recorded action array, and nothing else.

    Used to pick the action mode *before* the environment is created (the action-space swap has to
    happen on the parsed cfg, before gym.make()). Goes straight to h5py rather than through
    :meth:`HDF5DatasetFileHandler.load_episode` so the episode's image observations -- hundreds of
    MB for a camera-equipped task -- aren't loaded just to read the action width.
    """
    with h5py.File(dataset_file, "r") as file_stream:
        episode_group = file_stream["data"][episode_name]
        if "actions" not in episode_group:
            return None
        return episode_group["actions"][:]


def peek_episode_rigid_objects(dataset_file: str, episode_name: str) -> list[str]:
    """Names of the rigid objects an episode recorded state for, and nothing else.

    Same reason as :func:`peek_episode_actions` for going straight to h5py: this has to be known
    before gym.make(), and loading the episode properly would drag in every camera frame.

    The scene the env is built with has to contain exactly the objects the recording did, or
    ManagerBasedEnv.reset_to() raises KeyError on the first missing one -- it looks up each of the
    env's rigid objects in the recorded state.
    """
    with h5py.File(dataset_file, "r") as file_stream:
        episode_group = file_stream["data"][episode_name]
        states = episode_group.get("initial_state") or episode_group.get("states")
        if states is None or "rigid_object" not in states:
            return []
        return list(states["rigid_object"].keys())


def detect_openarm_action_mode(actions: np.ndarray | None) -> str:
    """Infer which action mode a recorded action array was produced with.

    Anything that isn't record_demos_openarm.py's 16D joint-space layout replays against the task's
    own action space, including that script's 14D dual-arm IK datasets (keyboard / vr_ros2), which
    record exactly what the unmodified task expects.

    The two 16D flavors are told apart by their gripper columns, which is the one place they can't
    overlap: vr_joint_ros2 records a binary +-1 command, while vr_joint_ros2_native records a
    measured finger position in meters, which is physically confined to [0, 0.044].
    """
    if actions is None or actions.ndim != 2 or actions.shape[1] != OPENARM_JOINT_ACTION_DIM:
        return "task_default"
    grippers = actions[:, [OPENARM_LEFT_GRIPPER_IDX, OPENARM_RIGHT_GRIPPER_IDX]]
    if grippers.min() >= -0.01 and grippers.max() <= 2 * OPENARM_GRIPPER_OPEN_VAL:
        return "openarm_joint_abs"
    return "openarm_joint_delta"


def swap_to_openarm_joint_actions(env_cfg, absolute: bool) -> None:
    """Rebuild a dual-arm OpenArm task's action space to match a joint-space recording.

    Mirrors record_demos_openarm.py's ``swap_to_joint_position_actions`` (``absolute=False``) and
    ``swap_to_native_ros2_joint_actions`` (``absolute=True``) so the replayed action vector is
    interpreted exactly as it was recorded. Must be called on the parsed cfg BEFORE gym.make():
    the field order (arm_action, gripper_action, right_arm_action, right_gripper_action) is fixed
    by the cfg class and unaffected by reassigning a field, but each field's *width* is not, and
    that's what turns the task's 14D IK action vector into the recorded 16D joint one.

    With ``absolute=False`` the arm columns are offsets from the default joint pose and the
    grippers stay on the task's own binary terms. With ``absolute=True`` both are absolute
    positions, so the default offset is dropped and the binary gripper terms are replaced (see
    :class:`OpenArmGripperJointPositionAction`).
    """
    if not hasattr(env_cfg.actions, "right_arm_action"):
        raise ValueError(
            "The dataset was recorded with a dual-arm joint-space action space, but the task's"
            " action cfg has no 'right_arm_action' field to swap. Is --task pointing at the"
            " dual-arm task the demos were recorded on?"
        )
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=OPENARM_LEFT_ARM_JOINT_REGEX,
        scale=OPENARM_JOINT_ACTION_SCALE,
        use_default_offset=not absolute,
    )
    env_cfg.actions.right_arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=OPENARM_RIGHT_ARM_JOINT_REGEX,
        scale=OPENARM_JOINT_ACTION_SCALE,
        use_default_offset=not absolute,
    )
    if absolute:
        env_cfg.actions.gripper_action = OpenArmGripperJointPositionActionCfg(
            asset_name="robot", joint_names=OPENARM_LEFT_FINGER_JOINT_NAMES
        )
        env_cfg.actions.right_gripper_action = OpenArmGripperJointPositionActionCfg(
            asset_name="robot", joint_names=OPENARM_RIGHT_FINGER_JOINT_NAMES
        )


def main():
    """Replay episodes loaded from a file."""
    global is_paused

    # Load dataset
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))

    episode_names = list(dataset_file_handler.get_episode_names())

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")
    # The gym id to instantiate: the fully qualified --task if given (it may carry a namespace
    # prefix that gym.make needs), otherwise the env name recorded in the dataset.
    task_id = args_cli.task if args_cli.task is not None else env_name

    num_envs = args_cli.num_envs

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)

    # Resolve how the recorded actions should be interpreted, and reshape the task's action space
    # if the dataset needs it. Both have to happen before the env is created.
    first_episode_index = next((index for index in episode_indices_to_replay if index < episode_count), None)
    if first_episode_index is None:
        raise ValueError(f"None of the selected episode indices exist in a dataset of {episode_count} episodes.")
    dataset_actions = peek_episode_actions(args_cli.dataset_file, episode_names[first_episode_index])
    dataset_action_dim = dataset_actions.shape[1] if dataset_actions is not None and dataset_actions.ndim == 2 else None

    # ── Match the scene to the objects the dataset actually recorded ──────────
    # A dataset recorded with record_demos_openarm.py --task_mode has the can in place of the
    # task's cube_2. Rebuilding the plain task for it puts a cube_2 in the scene that the recording
    # never saw, and reset_to() dies on the first frame with KeyError: 'cube_2'.
    recorded_objects = peek_episode_rigid_objects(args_cli.dataset_file, episode_names[first_episode_index])
    task_mode = args_cli.task_mode
    if task_mode is None and "can" in recorded_objects and "cube_2" not in recorded_objects:
        # Every mode builds the SAME scene -- they differ only in which arm may be driven, the
        # success condition and the spawn range, none of which affect a replay of recorded actions
        # against recorded states. So any mode reconstructs the right scene; say which was assumed.
        task_mode = "right"
        print(
            f"[INFO] Dataset records objects {recorded_objects} -- applying --task_mode {task_mode}"
            " so the scene matches. Pass --task_mode explicitly to override."
        )
    if task_mode is not None:
        from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_task_modes import (
            apply_task_mode,
        )

        apply_task_mode(env_cfg, task_mode)
        print(f"[TASK MODE] {task_mode}")

    action_mode = args_cli.action_mode
    if action_mode == "auto":
        action_mode = detect_openarm_action_mode(dataset_actions)
        print(f"Auto-detected action mode: {action_mode} (recorded action dim: {dataset_action_dim})")
    else:
        print(f"Using action mode: {action_mode} (recorded action dim: {dataset_action_dim})")
    if action_mode == "openarm_joint_abs":
        # vr_joint_ros2_native datasets recorded BEFORE record_demos_openarm.py's
        # ROS2NativeJointTeleop.build_dual_action was fixed hold the robot's *measured* joint
        # positions rather than the ROS2 command that produced them. Those replay as a pose chased
        # one step behind (harmless for the arms) and as grasps that never squeeze: a gripper
        # blocked by the cube records the position it stalled at, and re-commanding exactly that
        # position produces zero position error and hence zero grip force. A gripper column that
        # never goes far below the 0.044 open value is the signature -- re-record such demos.
        grippers = None
        if dataset_actions is not None and dataset_action_dim == OPENARM_JOINT_ACTION_DIM:
            grippers = dataset_actions[:, [OPENARM_LEFT_GRIPPER_IDX, OPENARM_RIGHT_GRIPPER_IDX]]
        if grippers is not None and grippers.min() > 0.5 * OPENARM_GRIPPER_OPEN_VAL:
            print(
                f"Warning: no gripper column in episode #{first_episode_index} ever closes past"
                f" {grippers.min():.4f} m (fully open is {OPENARM_GRIPPER_OPEN_VAL} m). If this was"
                " recorded before the measured-vs-commanded fix in record_demos_openarm.py, its"
                " grasps cannot be reproduced -- the recording holds where the fingers stalled"
                " against the object, not the command that squeezed them."
            )
    if action_mode != "task_default":
        swap_to_openarm_joint_actions(env_cfg, absolute=(action_mode == "openarm_joint_abs"))

    # extract success checking function to invoke in the main loop
    success_term = None
    if args_cli.validate_success_rate:
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        else:
            print(
                "No success termination term was found in the environment."
                " Will not be able to mark recorded demos as successful."
            )

    # Disable all recorders and terminations
    env_cfg.recorders = {}
    env_cfg.terminations = {}

    # create environment from loaded config
    env = gym.make(task_id, cfg=env_cfg).unwrapped

    # A dataset whose action width doesn't match the env's is unreplayable -- every step would be
    # silently mis-sliced across the action terms, so fail here with the numbers rather than
    # letting it run and produce nonsense motion.
    if dataset_action_dim is not None and dataset_action_dim != env.action_manager.total_action_dim:
        raise ValueError(
            f"The dataset's recorded actions are {dataset_action_dim}D but the environment's action"
            f" space is {env.action_manager.total_action_dim}D (action mode: {action_mode}). If these"
            " demos came from record_demos_openarm.py, pass the matching --action_mode"
            " (openarm_joint_abs / openarm_joint_delta / task_default)."
        )

    teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    teleop_interface.add_callback("N", play_cb)
    teleop_interface.add_callback("B", pause_cb)
    print('Press "B" to pause and "N" to resume the replayed actions.')

    # Determine if state validation should be conducted
    state_validation_enabled = False
    if args_cli.validate_states and num_envs == 1:
        state_validation_enabled = True
    elif args_cli.validate_states and num_envs > 1:
        print("Warning: State validation is only supported with a single environment. Skipping state validation.")

    # Get idle action (idle actions are applied to envs without next action)
    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(num_envs, 1).to(env.device)
    else:
        idle_action = torch.zeros(num_envs, env.action_manager.total_action_dim, device=env.device)

    # For a joint-space action space, a zeroed action is not "do nothing": it commands the default
    # joint pose (openarm_joint_delta) or the zero pose (openarm_joint_abs), i.e. an env that has
    # run out of actions would yank its arms away instead of standing still. Hold that env's last
    # action instead -- with absolute/offset joint targets, re-commanding the last target is what
    # actually holds the pose. Relative IK deltas are the opposite (repeating one keeps drifting),
    # so task_default keeps using the idle action.
    hold_last_action = action_mode != "task_default"
    actions = idle_action.clone()

    # reset before starting
    env.reset()
    teleop_interface.reset()

    # simulate environment -- run everything in inference mode
    replayed_episode_count = 0
    recorded_episode_count = 0

    # Track current episode indices for each environment
    current_episode_indices = [None] * num_envs

    # Track failed demo IDs
    failed_demo_ids = []

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            first_loop = True
            has_next_action = True
            episode_ended = [False] * num_envs
            while has_next_action:
                # initialize actions with idle action so those without next action will not move
                # (in hold_last_action modes, leave the buffer alone so those envs keep commanding
                # their last target, which is what holds them still there)
                if not hold_last_action:
                    actions = idle_action.clone()
                has_next_action = False
                for env_id in range(num_envs):
                    env_next_action = env_episode_data_map[env_id].get_next_action()
                    if env_next_action is None:
                        # check if the episode is successful after the whole episode_data is
                        if (
                            (success_term is not None)
                            and (current_episode_indices[env_id]) is not None
                            and (not episode_ended[env_id])
                        ):
                            if bool(success_term.func(env, **success_term.params)[env_id]):
                                recorded_episode_count += 1
                                plural_trailing_s = "s" if recorded_episode_count > 1 else ""

                                print(
                                    f"Successfully replayed {recorded_episode_count} episode{plural_trailing_s} out"
                                    f" of {replayed_episode_count} demos."
                                )
                            else:
                                # if not successful, add to failed demo IDs list
                                if (
                                    current_episode_indices[env_id] is not None
                                    and current_episode_indices[env_id] not in failed_demo_ids
                                ):
                                    failed_demo_ids.append(current_episode_indices[env_id])

                            episode_ended[env_id] = True

                        next_episode_index = None
                        while episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)

                            if next_episode_index < episode_count:
                                episode_ended[env_id] = False
                                break
                            next_episode_index = None

                        if next_episode_index is not None:
                            replayed_episode_count += 1
                            current_episode_indices[env_id] = next_episode_index
                            print(f"{replayed_episode_count:4}: Loading #{next_episode_index} episode to env_{env_id}")
                            episode_data = dataset_file_handler.load_episode(
                                episode_names[next_episode_index], env.device
                            )
                            if episode_data is None:
                                raise ValueError(f"Failed to load episode #{next_episode_index} from the dataset.")
                            env_episode_data_map[env_id] = episode_data
                            # Set initial state for the new episode
                            initial_state = episode_data.get_initial_state()
                            if initial_state is None:
                                # record_demos_openarm.py's VR modes arm recording on a button
                                # press, which calls recorder_manager.reset() mid-episode -- that
                                # drops the initial_state recorded at env reset along with the
                                # pre-arming steps, so those datasets only have per-step states.
                                # states[i] is the state *after* actions[i] was applied, so
                                # resetting to states[0] means replay has to resume at actions[1]
                                # to keep the action/state pairing intact.
                                initial_state = episode_data.get_state(0)
                                if initial_state is None:
                                    raise ValueError(
                                        f"Episode #{next_episode_index} has neither an initial state nor any"
                                        " recorded states to reset to."
                                    )
                                episode_data.next_action_index = 1
                                episode_data.next_state_index = 1
                                print(
                                    "      no initial_state recorded; resetting to the first recorded state and"
                                    " replaying from action index 1"
                                )
                            env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)
                            # Get the first action for the new episode
                            env_next_action = env_episode_data_map[env_id].get_next_action()
                            if env_next_action is None:
                                print(f"      episode #{next_episode_index} has no actions to replay; skipping.")
                                continue
                            has_next_action = True
                        else:
                            continue
                    else:
                        has_next_action = True
                    actions[env_id] = env_next_action
                if first_loop:
                    first_loop = False
                else:
                    while is_paused:
                        env.sim.render()
                        continue
                env.step(actions)

                if state_validation_enabled:
                    state_from_dataset = env_episode_data_map[0].get_next_state()
                    if state_from_dataset is not None:
                        print(
                            f"Validating states at action-index: {env_episode_data_map[0].next_state_index - 1:4}",
                            end="",
                        )
                        current_runtime_state = env.scene.get_state(is_relative=True)
                        states_matched, comparison_log = compare_states(state_from_dataset, current_runtime_state, 0)
                        if states_matched:
                            print("\t- matched.")
                        else:
                            print("\t- mismatched.")
                            print(comparison_log)
            break
    # Close environment after replay in complete
    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")

    # Print success statistics only if validation was enabled
    if success_term is not None:
        print(f"Successfully replayed: {recorded_episode_count}/{replayed_episode_count}")

        # Print failed demo IDs if any
        if failed_demo_ids:
            print(f"\nFailed demo IDs ({len(failed_demo_ids)} total):")
            print(f"  {sorted(failed_demo_ids)}")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
