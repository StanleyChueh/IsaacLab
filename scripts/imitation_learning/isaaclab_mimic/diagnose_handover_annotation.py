# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Diagnose why annotate_demos.py (--task_mode handover --auto) rejects specific episodes.

annotate_demos.py only ever reports "did not detect completion for subtask X" -- it does not say
how close the episode came. This replays the same episodes the same way (optionally
--from_states, matching annotate_demos.py's own replay path) and reports, per episode:

* the hand-over stage reached (openarm_task_modes.handover_stage) and when each stage was entered
* whether grasp_right / presented / handover ever fired, and if not, the closest the underlying
  continuous quantities (hand-to-can radial/axial offset, gripper jaw travel, receiver aperture)
  ever got to their thresholds

Usage (same --task/--task_mode/--input_file/--from_states as the annotate_demos.py run that
rejected these episodes):

    ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/diagnose_handover_annotation.py \\
        --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0 --task_mode handover --from_states \\
        --enable_cameras --input_file logs/demos/pickup_pringles_stanley.hdf5 \\
        --episodes 0,3,6,13,14,15,20,26
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose hand-over annotation rejections.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--input_file", type=str, required=True, help="Dataset to diagnose.")
parser.add_argument(
    "--episodes", type=str, required=True, help="Comma-separated episode indices to diagnose, e.g. '0,3,6'."
)
parser.add_argument("--task_mode", type=str, default="handover", choices=["left", "right", "handover"])
parser.add_argument(
    "--from_states",
    action="store_true",
    default=False,
    help="Same meaning as annotate_demos.py's --from_states -- pass it if that is what the failing"
    " annotate_demos.py run used, so this replays the episodes identically.",
)
parser.add_argument("--enable_pinocchio", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import h5py
import torch

import gymnasium as gym

import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm import openarm_task_modes
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_joint_actions import (
    match_action_space_to_dataset,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

_STAGE_LABELS = {
    0: "waiting for the RIGHT arm to grasp the can",
    1: "right holds it -- waiting for the LEFT arm to take it while the right still holds",
    2: "passed to the left -- waiting for the RIGHT arm to let go while the LEFT keeps hold",
    3: "complete",
}


def _restore_state(env: ManagerBasedRLMimicEnv, episode: EpisodeData, state_index: int) -> None:
    state = episode.get_state(state_index)
    if state is None:
        return
    env.scene.reset_to(state, None, is_relative=True)
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)


def diagnose_episode(
    env: ManagerBasedRLMimicEnv, episode: EpisodeData, object_cfg: SceneEntityCfg, diff_threshold: float
) -> None:
    actions = episode.data["actions"]
    if "initial_state" in episode.data:
        initial_state = episode.data["initial_state"]
        first_action_index = 0
    else:
        initial_state = episode.get_state(0)
        if initial_state is None:
            print("  no initial_state and no recorded states -- cannot replay this episode.")
            return
        first_action_index = 1

    env.sim.reset()
    env.recorder_manager.reset()
    env.reset_to(initial_state, None, is_relative=True)

    signal_first_fire: dict[str, int] = {}
    stage_first_step: dict[int, int] = {}
    stage_seen = -1
    min_right_radial = float("inf")
    min_right_axial = float("inf")
    max_right_jaw_travel = 0.0
    min_left_radial_after_pass = float("inf")
    aperture_after_pass: list[float] = []
    can_z_after_release: list[float] = []  # only once stage 2 AND the right hand is no longer holding

    def record(step_index: int) -> None:
        nonlocal stage_seen, min_right_radial, min_right_axial, max_right_jaw_travel, min_left_radial_after_pass

        for name, flags in env.get_subtask_term_signals().items():
            if bool(flags[0] > 0.5) and name not in signal_first_fire:
                signal_first_fire[name] = step_index

        stage = int(openarm_task_modes.handover_stage(env)[0])
        if stage != stage_seen:
            stage_seen = stage
            stage_first_step[stage] = step_index

        r_radial, r_axial = openarm_task_modes.hand_to_object_offsets(
            env, SceneEntityCfg("right_ee_frame"), object_cfg
        )
        min_right_radial = min(min_right_radial, float(r_radial[0]))
        min_right_axial = min(min_right_axial, float(r_axial[0]))
        right_jaws = openarm_task_modes.gripper_jaw_positions(env, openarm_task_modes.RIGHT_FINGER_JOINTS)[0]
        travel = min(abs(float(j) - openarm_task_modes.GRIPPER_OPEN_VAL) for j in right_jaws)
        max_right_jaw_travel = max(max_right_jaw_travel, travel)

        if stage >= 1:
            l_radial, _ = openarm_task_modes.hand_to_object_offsets(env, SceneEntityCfg("ee_frame"), object_cfg)
            min_left_radial_after_pass = min(min_left_radial_after_pass, float(l_radial[0]))
        if stage >= 2:
            aperture_after_pass.append(
                float(openarm_task_modes.gripper_aperture(env, openarm_task_modes.LEFT_FINGER_JOINTS)[0])
            )
            right_holds = bool(
                openarm_task_modes.object_grasped_by(
                    env,
                    ee_frame_cfg=SceneEntityCfg("right_ee_frame"),
                    gripper_joint_names=openarm_task_modes.RIGHT_FINGER_JOINTS,
                    object_cfg=object_cfg,
                    diff_threshold=diff_threshold,
                )[0]
            )
            if not right_holds:
                can_z = float(env.scene[object_cfg.name].data.root_pos_w[0, 2])
                can_z_after_release.append(can_z)

    for action_index, action in enumerate(actions[first_action_index:], start=first_action_index):
        if args_cli.from_states and action_index > first_action_index:
            _restore_state(env, episode, action_index - 1)
        action_tensor = torch.Tensor(action).reshape([1, action.shape[0]])
        env.step(action_tensor)
        record(action_index)

    if args_cli.from_states:
        _restore_state(env, episode, len(actions) - 1)
    record(len(actions) - 1)

    print(f"  final stage reached: {stage_seen} ({_STAGE_LABELS.get(stage_seen, '?')})")
    for stage, step in sorted(stage_first_step.items()):
        print(f"    stage {stage} first reached at action index {step}")
    for name in ("grasp_right", "presented", "handover"):
        fired_at = signal_first_fire.get(name)
        print(f"  {name}: {'fired at step ' + str(fired_at) if fired_at is not None else 'NEVER FIRED'}")

    gripper_threshold = openarm_task_modes.GRIPPER_THRESHOLD
    print(
        f"  right hand vs can: closest radial={min_right_radial:.4f} m (threshold {diff_threshold:.3f}),"
        f" closest axial={min_right_axial:.4f} m (threshold {openarm_task_modes.GRASP_AXIAL_TOLERANCE:.3f}),"
        f" deepest jaw travel={max_right_jaw_travel:.4f} m (threshold {gripper_threshold:.3f})"
    )
    if stage_seen >= 1:
        print(
            f"  left hand vs can after the right lifted it: closest radial="
            f"{min_left_radial_after_pass:.4f} m (threshold {diff_threshold:.3f})"
        )
    if aperture_after_pass:
        lo, hi = openarm_task_modes.HANDOVER_RECEIVER_APERTURE_RANGE
        print(
            f"  left aperture once the right let go: min={min(aperture_after_pass):.4f} m,"
            f" max={max(aperture_after_pass):.4f} m (band [{lo:.3f}, {hi:.3f}])"
        )
    if can_z_after_release:
        rest_z = float(openarm_task_modes._object_rest_z(env))
        put_down_z = rest_z + openarm_task_modes.HANDOVER_PUT_DOWN_MARGIN
        min_z = min(can_z_after_release)
        # Gravity is the ground truth here: a can NOT actually held by the left hand falls back to
        # the pad the instant the right hand lets go. One held the whole time never gets near
        # put_down_z; one that was really just resting on an open/empty hand does.
        verdict = "STAYED ELEVATED (physically consistent with a real hold)" if min_z > put_down_z + 0.01 else (
            "DROPPED close to resting height (looks like it was NOT actually held)"
        )
        print(
            f"  can height once the right hand let go: min={min_z:.4f} m vs. resting"
            f" height {rest_z:.4f} m (put-down cutoff {put_down_z:.4f} m) -> {verdict}"
        )


def main() -> None:
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)
    env_name = args_cli.task.split(":")[-1]

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
    env_cfg.env_name = env_name
    openarm_task_modes.apply_task_mode(env_cfg, args_cli.task_mode)

    with h5py.File(args_cli.input_file, "r") as f:
        first_ep = next(iter(f["data"].keys()), None)
        actions = f["data"][first_ep]["actions"][:] if first_ep and "actions" in f["data"][first_ep] else None
    match_action_space_to_dataset(env_cfg, actions)

    if not hasattr(env_cfg.terminations, "success"):
        raise NotImplementedError("No success termination term was found in the environment.")
    success_term = env_cfg.terminations.success
    env_cfg.terminations = None
    env_cfg.recorders = ActionStateRecorderManagerCfg()

    env: ManagerBasedRLMimicEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    object_cfg = success_term.params["object_cfg"]
    diff_threshold = success_term.params["diff_threshold"]

    episode_names = list(dataset_file_handler.get_episode_names())
    requested = [int(x) for x in args_cli.episodes.split(",") if x.strip() != ""]

    with torch.inference_mode():
        for idx in requested:
            if idx < 0 or idx >= len(episode_names):
                print(f"\n=== episode #{idx}: out of range (dataset has {len(episode_names)} episodes) ===")
                continue
            name = episode_names[idx]
            print(f"\n=== episode #{idx} ({name}) ===")
            episode = dataset_file_handler.load_episode(name, env.device)
            diagnose_episode(env, episode, object_cfg, diff_threshold)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
