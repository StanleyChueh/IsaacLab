# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for Mimic generation that move the arms the way the recorded source demos do.

The source demos are recorded through lerobot_sim_server.py, whose joint targets come from the Quest teleop's IK
(joint-limit aware, speed limited) and whose grippers are slewed by the robot config. Generation instead drives the
task's 14D IK action: Isaac Lab's DifferentialInverseKinematicsAction with plain damped least squares and binary
grippers. Measured on logs/demos/openarm_pringles_lerobot_sim_10_v00_generated.hdf5 (500 episodes, 10 source demos)
that difference, not the source demos, made the two artefacts that remain in the generated motion:

* **Elbow flips.** DLS knows no joint limits. In 73 of 500 episodes the right arm's reach drove the elbow (joint4,
  limits 0 .. 2.44) to a target of up to -6.4 rad: the measured elbow stops at 0, the arm locks straight and the
  hand swings up ~25 cm before it recovers (episode 2). The wrist (joint7) and joint1 wind up the same way, by up to
  3 rad. The source demos never straighten the elbow below 0.78 rad.
* **Gripper snaps.** A binary gripper jumps between 0 and 0.044 in one step; the source demos move it at most
  0.0199 m per 30 Hz step. Opening the giving hand in one step, while both hands squeeze the can, kicked the receiving
  arm to 0.64 m/s where its source moved at 0.35 (39 episodes, all from source demo 7).

:class:`OpenArmLimitedIKAction` keeps DLS but solves with joint limits: joints that would cross a limit are locked
at it and the pose error is redistributed over the others; the elbow keeps a lower bound away from the straight-arm
singularity; and the redundant direction is pulled gently toward the rest posture (as the teleop's IK does). It does
not cap the step: the step is how far the target leads the measured joint, which is what the joint drive pushes with,
so capping it slows the arm (tried at 2.5 rad/s: the hand lagged its target 2.2 cm instead of 0.6 and generation
success fell from 54% to 26%). :class:`OpenArmSlewedGripperAction` moves the binary gripper target at the source
demos' speed. :func:`use_source_like_motion` installs both on an env cfg; the Mimic env cfgs call it, so only
generation changes -- the sim server replaces these actions with joint targets anyway.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import compute_pose_error
from isaaclab.utils.string import resolve_matching_names

ARM_ACTIONS = ("arm_action", "right_arm_action")
GRIPPER_ACTIONS = ("gripper_action", "right_gripper_action")


class OpenArmLimitedIKAction(DifferentialInverseKinematicsAction):
    """Damped least squares IK that respects joint limits and keeps the elbow bent."""

    cfg: OpenArmLimitedIKActionCfg

    def __init__(self, cfg: OpenArmLimitedIKActionCfg, env):
        super().__init__(cfg, env)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids].clone()
        self._lower, self._upper = limits[..., 0], limits[..., 1]
        for expr, value in cfg.min_joint_pos.items():
            ids, _ = resolve_matching_names(expr, self._joint_names)
            self._lower[:, ids] = self._lower[:, ids].clamp(min=value)
        self._posture = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._null_step = cfg.nullspace_gain * env.physics_dt
        self._damping = cfg.controller.ik_params["lambda_val"] ** 2

    def apply_actions(self):
        ee_pos, ee_quat = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if ee_quat.norm() == 0:
            target = joint_pos.clone()
        else:
            pos_err, rot_err = compute_pose_error(
                ee_pos, ee_quat, self._ik_controller.ee_pos_des, self._ik_controller.ee_quat_des,
                rot_error_type="axis_angle",
            )
            target = joint_pos + self._limited_step(self._compute_frame_jacobian(), torch.cat((pos_err, rot_err), 1),
                                                    joint_pos)
        self._asset.set_joint_position_target(target, self._joint_ids)

    def _limited_step(self, jacobian: torch.Tensor, error: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """One IK step: DLS over the joints that stay inside their limits, plus a null-space pull to the posture.

        A joint whose step would cross a limit is locked at that limit and solved out of the task: its motion is
        taken off the error and the remaining joints are solved again (at most once per joint).
        """
        eye_task = torch.eye(jacobian.shape[1], device=q.device)
        eye_joint = torch.eye(q.shape[1], device=q.device)
        free = torch.ones_like(q, dtype=torch.bool)
        locked_dq = torch.zeros_like(q)
        for _ in range(q.shape[1]):
            jac = jacobian * free.unsqueeze(1)
            err = error - (jacobian @ locked_dq.unsqueeze(-1)).squeeze(-1)
            jac_t = jac.transpose(1, 2)
            pinv = jac_t @ torch.linalg.inv(jac @ jac_t + self._damping * eye_task)
            pull = ((self._posture - q) * self._null_step * free).unsqueeze(-1)
            dq = (pinv @ err.unsqueeze(-1) + (eye_joint - pinv @ jac) @ pull).squeeze(-1)
            dq = torch.where(free, dq, locked_dq)
            crossing = free & ((q + dq < self._lower) | (q + dq > self._upper))
            if not crossing.any():
                break
            locked_dq = torch.where(crossing, (q + dq).clamp(self._lower, self._upper) - q, locked_dq)
            free &= ~crossing
        return (q + dq).clamp(self._lower, self._upper) - q


@configclass
class OpenArmLimitedIKActionCfg(mdp.DifferentialInverseKinematicsActionCfg):
    class_type: type[ActionTerm] = OpenArmLimitedIKAction

    min_joint_pos: dict[str, float] = {r"openarm_(left|right)_joint4": 0.3}
    """Lower bounds tighter than the joint limits. The elbow's limit, 0, is the straight-arm singularity; the source
    demos bend it at least 0.78 rad."""
    nullspace_gain: float = 1.0
    """1/s: how fast the redundant direction drifts back to the rest posture. Does not move the hand."""


class OpenArmSlewedGripperAction(BinaryJointPositionAction):
    """Binary gripper whose finger target moves toward open / closed at a limited speed."""

    cfg: OpenArmSlewedGripperActionCfg

    def __init__(self, cfg: OpenArmSlewedGripperActionCfg, env):
        super().__init__(cfg, env)
        self._max_step = cfg.max_speed * env.physics_dt
        self._target = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def apply_actions(self):
        self._target += (self._processed_actions - self._target).clamp(-self._max_step, self._max_step)
        self._asset.set_joint_position_target(self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        env_ids = slice(None) if env_ids is None else env_ids
        self._target[env_ids] = self._asset.data.default_joint_pos[env_ids][:, self._joint_ids]


@configclass
class OpenArmSlewedGripperActionCfg(mdp.BinaryJointPositionActionCfg):
    class_type: type[ActionTerm] = OpenArmSlewedGripperAction

    max_speed: float = 0.6
    """m/s per finger: the source demos' 0.0199 m per 30 Hz step."""


def use_source_like_motion(env_cfg) -> None:
    """Swap an OpenArm IK env cfg's arm and gripper actions for the limited / slewed ones, same settings otherwise."""
    for name in ARM_ACTIONS:
        old = getattr(env_cfg.actions, name)
        setattr(env_cfg.actions, name, OpenArmLimitedIKActionCfg(
            asset_name=old.asset_name, joint_names=old.joint_names, body_name=old.body_name,
            body_offset=old.body_offset, scale=old.scale, controller=old.controller,
        ))
    for name in GRIPPER_ACTIONS:
        old = getattr(env_cfg.actions, name)
        setattr(env_cfg.actions, name, OpenArmSlewedGripperActionCfg(
            asset_name=old.asset_name, joint_names=old.joint_names,
            open_command_expr=old.open_command_expr, close_command_expr=old.close_command_expr,
        ))
