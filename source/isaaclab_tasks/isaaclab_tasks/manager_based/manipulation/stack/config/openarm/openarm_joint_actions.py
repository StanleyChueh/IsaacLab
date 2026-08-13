# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Re-interpreting an OpenArm task's action space as the joint-space one its demos were recorded in.

``record_demos_openarm.py`` rewrites the task's 14D IK action space before recording whenever its
``--teleop_device`` is one of the joint-space VR modes, producing a 16D joint-space action vector:

    [0:7] left arm | [7] left gripper | [8:15] right arm | [15] right gripper

Any script that later feeds those recorded actions back into the task -- replaying them, or
annotating them -- has to rebuild the same action space first, or ``env.step()`` rejects the row
outright (``Invalid action shape, expected: 14, received: 16``).

This module is where that lives, rather than in any one script, because more than one script needs
it and none of them can import each other: ``record_demos_openarm.py`` and ``replay_demos.py`` both
parse argv and launch their own AppLauncher at import time. Importing this module has no side
effects beyond pulling in isaaclab, so any of them can use it.
"""

import numpy as np
import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

# Deliberate duplicates of record_demos_openarm.py's constants -- keep them in sync with it.
OPENARM_LEFT_ARM_JOINT_REGEX = ["openarm_left_joint[1-7]"]
OPENARM_RIGHT_ARM_JOINT_REGEX = ["openarm_right_joint[1-7]"]
OPENARM_LEFT_FINGER_JOINT_NAMES = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
OPENARM_RIGHT_FINGER_JOINT_NAMES = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]
OPENARM_JOINT_ACTION_SCALE = 1.0
OPENARM_JOINT_ACTION_DIM = 16
OPENARM_LEFT_GRIPPER_IDX = 7
OPENARM_RIGHT_GRIPPER_IDX = 15
OPENARM_GRIPPER_OPEN_VAL = 0.044
"""Fully-open finger position (m) -- matches the task's BinaryJointPositionActionCfg
open_command_expr, and is the upper end of the absolute gripper values recorded in
vr_joint_ros2_native mode."""


class OpenArmGripperJointPositionAction(ActionTerm):
    """1D absolute finger-position action driving both jaws of one OpenArm gripper.

    Needed for --teleop_device vr_joint_ros2_native datasets: there the recorded gripper column is
    an absolute position of ``*_finger_joint1`` in meters (0 closed .. 0.044 open), not the +-1
    binary command the task's own BinaryJointPositionActionCfg expects -- that term reads every
    value in this range as one saturated command, so grasps never register. A plain
    JointPositionActionCfg over both finger joints can't be used instead because it would be 2D and
    widen the action vector past the recorded 16D -- hence this term: one action value, written to
    both jaws (joint2 is a PhysX mimic of joint1 *and* independently actuated, so leaving it
    uncommanded pins it at its default while joint1 moves).
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


def detect_openarm_action_mode(actions) -> str:
    """Infer which action mode a recorded action array was produced with.

    Anything that isn't the 16D joint-space layout runs against the task's own action space,
    including record_demos_openarm.py's 14D dual-arm IK datasets (keyboard / vr_ros2), which record
    exactly what the unmodified task expects.

    The two 16D flavors are told apart by their gripper columns, the one place they cannot overlap:
    vr_joint_ros2 records a binary +-1 command, while vr_joint_ros2_native records an absolute
    finger position in meters, physically confined to [0, 0.044].
    """
    actions = np.asarray(actions) if actions is not None else None
    if actions is None or actions.ndim != 2 or actions.shape[1] != OPENARM_JOINT_ACTION_DIM:
        return "task_default"
    grippers = actions[:, [OPENARM_LEFT_GRIPPER_IDX, OPENARM_RIGHT_GRIPPER_IDX]]
    if grippers.min() >= -0.01 and grippers.max() <= 2 * OPENARM_GRIPPER_OPEN_VAL:
        return "openarm_joint_abs"
    return "openarm_joint_delta"


def swap_to_openarm_joint_actions(env_cfg, absolute: bool) -> None:
    """Rebuild a dual-arm OpenArm task's action space to match a joint-space recording.

    Mirrors record_demos_openarm.py's ``swap_to_joint_position_actions`` (``absolute=False``) and
    ``swap_to_native_ros2_joint_actions`` (``absolute=True``) so a recorded action vector is
    interpreted exactly as it was produced. Must be called on the parsed cfg BEFORE gym.make():
    the field order (arm_action, gripper_action, right_arm_action, right_gripper_action) is fixed
    by the cfg class and unaffected by reassigning a field, but each field's *width* is not, and
    that is what turns the task's 14D IK action vector into the recorded 16D joint one.

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


def match_action_space_to_dataset(env_cfg, actions) -> str:
    """Swap *env_cfg*'s action space to whatever *actions* were recorded in. Returns the mode.

    The one call an annotation/replay script needs: detect, swap if required, and report. A
    "task_default" result means the recording already matches the task's own action space and
    nothing was changed.
    """
    mode = detect_openarm_action_mode(actions)
    if mode != "task_default":
        swap_to_openarm_joint_actions(env_cfg, absolute=(mode == "openarm_joint_abs"))
    return mode
