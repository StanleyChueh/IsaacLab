# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab Mimic environment wrapper for the OpenArm pick-up task.

The OpenArm "IK-Abs" task uses use_relative_mode=True in its differential IK controllers, so
the action format is a RELATIVE delta pose per arm (same pattern as Franka IK-Rel):

  action (14D):
    [0:3]   left arm delta position  (dx, dy, dz)
    [3:6]   left arm delta rotation  (axis-angle, rad)
    [6]     left gripper command     (+1.0 = open, -1.0 = close)
    [7:10]  right arm delta position
    [10:13] right arm delta rotation
    [13]    right gripper command

Both arms are exposed as Mimic end-effectors ("left_eef" / "right_eef"). Which of them
actually does the work -- and with what subtask sequence -- is decided by the task mode
applied to the cfg (see openarm_task_modes.apply_task_mode): in the single-arm modes one arm
gets the grasp/lift sequence and the other gets a single idle subtask; in handover mode both
carry real sequences tied together by a SEQUENTIAL constraint. This class stays mode-agnostic
and simply serves whatever end-effector keys the cfg declares.
"""

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

# 14D IK action layout, per arm: (delta slice, gripper index, eef pose obs key prefix)
_ARM_LAYOUT = {
    "left_eef": (slice(0, 6), 6, "eef"),
    "right_eef": (slice(7, 13), 13, "right_eef"),
}
_ACTION_DIM = 14

# 16D joint-space layout produced by record_demos_openarm.py's vr_joint_ros2* devices:
#   [0:7] left arm | [7] left gripper | [8:15] right arm | [15] right gripper
# Only the gripper columns are ever read from it here -- see actions_to_gripper_actions.
_JOINT_ACTION_DIM = 16
_JOINT_GRIPPER_IDX = {"left_eef": 7, "right_eef": 15}
_GRIPPER_OPEN_VAL = 0.044
"""Fully-open finger position (m) in the joint-space recordings, i.e. the value that maps to
the +1 "open" command of the task's own BinaryJointPositionActionCfg."""


class OpenArmPickUpIKAbsMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic wrapper for Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 (both arms)."""

    def _eef_names(self) -> list[str]:
        return list(self.cfg.subtask_configs.keys())

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Return one arm's EEF pose as a (N, 4, 4) homogeneous matrix.

        Reads the pose out of the policy observation buffer, where the left arm is published as
        eef_pos/eef_quat by the base task and the right arm as right_eef_pos/right_eef_quat by
        apply_task_mode(). Quaternion convention: w, x, y, z.
        """
        if env_ids is None:
            env_ids = slice(None)
        if eef_name not in _ARM_LAYOUT:
            raise ValueError(f"Unknown end-effector '{eef_name}'. Expected one of {list(_ARM_LAYOUT)}.")
        _, _, obs_prefix = _ARM_LAYOUT[eef_name]
        eef_pos = self.obs_buf["policy"][f"{obs_prefix}_pos"][env_ids]
        eef_quat = self.obs_buf["policy"][f"{obs_prefix}_quat"][env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert per-arm target EEF poses -> one 14D environment action.

        Each arm's slot is filled with the delta from its CURRENT pose to its target, so an arm
        whose target equals its current pose contributes a zero delta and simply holds still --
        which is what an idle arm's subtask produces.
        """
        full = torch.zeros(_ACTION_DIM, device=self.device)

        for eef_name, target_eef_pose in target_eef_pose_dict.items():
            delta_slice, gripper_idx, _ = _ARM_LAYOUT[eef_name]
            target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

            curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
            curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

            delta_position = target_pos - curr_pos
            delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
            delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
            delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

            pose_action = torch.cat([delta_position, delta_rotation], dim=0)
            if action_noise_dict is not None and eef_name in action_noise_dict:
                noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
                pose_action = torch.clamp(pose_action + noise, -1.0, 1.0)

            full[delta_slice] = pose_action.to(full.dtype)
            full[gripper_idx] = gripper_action_dict[eef_name].squeeze().to(full.dtype)

        # An arm with no entry in target_eef_pose_dict keeps a zero delta (hold pose), but its
        # gripper must not default to 0.0 -- that is neither open nor close for a binary term.
        for eef_name, (_, gripper_idx, _) in _ARM_LAYOUT.items():
            if eef_name not in target_eef_pose_dict:
                full[gripper_idx] = 1.0  # stay open, so an unused arm can't clamp onto anything

        return full

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert 14D env actions -> per-arm target EEF poses (N, 4, 4).

        For a relative controller, target = current_pose (+) delta.
        """
        target_poses = {}
        for eef_name in self._eef_names():
            delta_slice, _, _ = _ARM_LAYOUT[eef_name]
            delta_position = action[:, delta_slice.start : delta_slice.start + 3]
            delta_rotation = action[:, delta_slice.start + 3 : delta_slice.stop]

            curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
            curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

            target_pos = curr_pos + delta_position

            delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
            delta_rotation_axis = delta_rotation / delta_rotation_angle
            is_close_to_zero_angle = torch.isclose(
                delta_rotation_angle, torch.zeros_like(delta_rotation_angle)
            ).squeeze(1)
            delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[
                is_close_to_zero_angle
            ]

            delta_quat = PoseUtils.quat_from_angle_axis(
                delta_rotation_angle.squeeze(1), delta_rotation_axis
            ).squeeze(0)
            delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
            target_rot = torch.matmul(delta_rot_mat, curr_rot)

            target_poses[eef_name] = PoseUtils.make_pose(target_pos, target_rot).clone()
        return target_poses

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract each arm's gripper command from a batch of actions.

        Handles both action layouts this task is recorded in, because this is the one Mimic API
        that is called on the RAW actions of a source dataset (see
        datagen_info_pool.DataGenInfoPool) rather than on actions this env produced:

        * 14D IK datasets (--teleop_device keyboard / vr_ros2) already store the +-1 binary
          command the task's BinaryJointPositionActionCfg expects.
        * 16D joint-space datasets (--teleop_device vr_joint_ros2*) store an absolute finger
          position in metres, 0.0 closed .. 0.044 open, which has to be mapped back onto that
          same +-1 convention or every value in the range would read as "close".
        """
        if actions.shape[-1] == _JOINT_ACTION_DIM:
            return {
                eef_name: 2.0 * (actions[:, idx : idx + 1] / _GRIPPER_OPEN_VAL) - 1.0
                for eef_name, idx in _JOINT_GRIPPER_IDX.items()
                if eef_name in self._eef_names()
            }
        return {
            eef_name: actions[:, _ARM_LAYOUT[eef_name][1] : _ARM_LAYOUT[eef_name][1] + 1]
            for eef_name in self._eef_names()
        }

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return every subtask termination signal the task mode published.

        Deliberately reads the group rather than naming signals: which ones exist depends on the
        task mode ('grasp'/'lift' for the single-arm modes, 'grasp_left'/'handover'/'grasp_right'
        for handover), and annotate_demos.py rejects an episode in which any RETURNED signal
        never fires -- so hardcoding a name here would fail every episode recorded under a
        different mode.
        """
        if env_ids is None:
            env_ids = slice(None)
        subtask_terms = self.obs_buf["subtask_terms"]
        return {name: signal[env_ids] for name, signal in subtask_terms.items()}
