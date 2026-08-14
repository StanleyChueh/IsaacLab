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
# Nothing here decodes an eef pose out of it -- these are joint targets, not IK deltas. Only the
# gripper columns carry directly usable information (see actions_to_gripper_actions); the arm
# columns are recognised solely so action_to_target_eef_pose does not mis-slice them.
_JOINT_ACTION_DIM = 16
_JOINT_GRIPPER_IDX = {"left_eef": 7, "right_eef": 15}

_TAKEN_STEP_ATTR = "_openarm_handover_taken_step"
"""Attribute _hold_giving_hand_until_taken keeps its per-env latch step under."""
_GRIPPER_OPEN_VAL = 0.044
"""Fully-open finger position (m) in the joint-space recordings, i.e. the value that maps to
the +1 "open" command of the task's own BinaryJointPositionActionCfg."""

_GIVING_EEF = "right_eef"
"""Which arm hands the can over in ``handover`` mode. Fixed by the mode itself (openarm_task_modes'
handover_success ends with the can in the LEFT hand), not something a demo can vary."""

_HANDOVER_RELEASE_OVERLAP_STEPS = 5
"""Steps the giving hand keeps holding after the receiving hand has taken the can.

0.25 s at this task's 20 Hz control rate, matching the 0-4 step overlap the operator actually leaves
in the recorded demos. Not raised further on purpose: both jaws close on a rigid can at kp=500, so a
long two-handed hold is squeezing it between two position-controlled grippers, which PhysX resolves
as growing internal force rather than as a firmer grip."""


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

        self._hold_giving_hand_until_taken(full, env_id)
        return full

    def _hold_giving_hand_until_taken(self, action: torch.Tensor, env_id: int) -> None:
        """Clamp the giving hand shut, in place, until the receiving hand has really taken the can.

        Two jobs, one load-bearing and one backstop, both for openarm_task_modes'
        ``_handover_subtask_configs``:

        * While the giving arm's release subtask WAITS on its sequential constraint (holding at its
          first waypoint until the receiving hand has closed), the gripper command replayed at that
          waypoint comes from the source demo -- and in half of the recorded demos the source jaws
          are already open at that boundary. Without this clamp the hand would open the moment the
          wait began. The constraint orders the ARM segments; only this orders the JAWS.
        * Backstop against drift: generation replays segments by length, so tracking drift between
          the arms can move the release relative to the take. This reads the actual scene, so no
          amount of drift can produce a release before a take.

        The condition is built from two subtask term signals the task mode already publishes, rather
        than from a fresh distance/jaw test, so there is exactly one definition of "this hand is
        holding the can" in play (an earlier hand-over bug came from two descriptions of one event
        disagreeing):

        * ``presented`` -- the giving hand has the can UP. Until then there is nothing to drop and
          the gate stays out of the way entirely, so a trial where the right arm never picked the
          can up is not left with a hand welded shut.
        * ``handover`` -- the receiving hand has closed on the can while the giving hand still had
          it. Latched by the stage machine, which matters: the raw per-step grasp detection flickers
          badly right after contact (one of the ten demos drops out on 13 of the 31 frames after its
          first detected grip, flipping 10 times), and an unlatched gate would let the giving hand
          re-close mid-release.

        After ``handover`` the hand is held for :data:`_HANDOVER_RELEASE_OVERLAP_STEPS` more steps.
        Both signals are absent outside hand-over mode, which is what scopes this to it.

        Deliberately NOT a reproduction of the demos: measured over the ten recorded hand-overs the
        operator opens the giving hand 0-4 steps after closing the receiving one, and once 1 step
        BEFORE. The overlap is a task invariant imposed on generation, not a property of the source.
        """
        terms = self.obs_buf.get("subtask_terms") if isinstance(self.obs_buf, dict) else None
        if not terms or "presented" not in terms or "handover" not in terms:
            return
        if float(terms["presented"][env_id]) <= 0.5:
            return  # the giving hand has not lifted the can; nothing to protect

        gripper_idx = _ARM_LAYOUT[_GIVING_EEF][1]
        # Keyed by env_id and keyed on the step counter rather than on call count, so it survives
        # this method being called any number of times per step, and self-clears on reset: the stage
        # machine resets to 0 there, so `handover` reads 0 again and the entry is dropped.
        first_taken_step = getattr(self, _TAKEN_STEP_ATTR, None)
        if first_taken_step is None:
            first_taken_step = {}
            setattr(self, _TAKEN_STEP_ATTR, first_taken_step)

        if float(terms["handover"][env_id]) <= 0.5:
            first_taken_step.pop(env_id, None)
            action[gripper_idx] = -1.0
            return

        step = int(getattr(self, "common_step_counter", 0))
        if step - first_taken_step.setdefault(env_id, step) < _HANDOVER_RELEASE_OVERLAP_STEPS:
            action[gripper_idx] = -1.0

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert env actions -> per-arm target EEF poses (N, 4, 4).

        For the 14D IK layout this is the relative controller's target = current_pose (+) delta.

        The 16D joint-space layout has to be handled separately, for the same reason
        actions_to_gripper_actions does: annotate_demos.py rebuilds the env's action space to match
        the dataset (see openarm_joint_actions.match_action_space_to_dataset), so the action reaching
        this method is a row of JOINT targets whenever the demos came from a --teleop_device
        vr_joint_ros2* recording. Slicing it as if it were an IK delta reads joint1-3 as a delta
        position and joint4-6 as a delta rotation -- which is what previously wrote a
        `target_eef_pose` averaging 0.30 m away from the eef pose of the very same timestep into the
        annotated dataset, i.e. the waypoints Mimic replays (generation_transform_first_robot_pose
        takes essentially all of them from here) were garbage and every generated trial knocked the
        object over within ~30 steps.

        There is no eef delta to decode from joint targets, so the target reported is the pose the
        arm actually reached. That lags the true commanded pose by one step, which is worth ~7 mm on
        these demos (measured mean per-step eef motion; p95 ~2 cm) -- two orders of magnitude below
        the error it replaces, and small relative to the delta the IK controller resolves per step.
        """
        if action.shape[-1] == _JOINT_ACTION_DIM:
            return {
                eef_name: self.get_robot_eef_pose(eef_name, env_ids=None).clone()
                for eef_name in self._eef_names()
            }

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
