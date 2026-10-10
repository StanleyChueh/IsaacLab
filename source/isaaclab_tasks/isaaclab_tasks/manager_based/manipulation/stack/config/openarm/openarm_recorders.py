# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Recorders that make a generated OpenArm dataset convertible to the official lerobot format.

lerobot_openarm's real datasets (lerobot-record, --robot.type=openarm_umeow) store, per frame, the measured
joints as `observation.state` and the JOINT TARGET commanded that frame -- the Quest teleop's IK solution --
as `action`. A Mimic-generated episode's own `actions` are end-effector deltas, and its `states` hold only
measured joints, so the closest thing to the real action used to be "the next measured pose": a different
quantity (it lags the command, and a closed gripper's next pose is wherever the object stopped it rather
than the closed command).

The joint targets do exist in the sim: the IK action term writes them to the articulation every step.
LeRobotJointsRecorder stores, keyed by joint name so no column order has to be assumed downstream:

  lerobot/joint_pos/<joint>         PRE-step: the measured joints the step starts from, i.e. frame i's
                                    observation.state (aligned with obs/<camera>, also recorded pre-step)
  lerobot/joint_pos_target/<joint>  POST-step: the targets the action terms applied during step i, i.e.
                                    frame i's action

lerobot_openarm's mimic_to_lerobot.py turns them into the real datasets' keys and motor units.
"""

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


class LeRobotJointsRecorder(RecorderTerm):
    """Measured joints before each step and the joint targets applied during it, by joint name."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._robot = env.scene["robot"]
        self._names = list(self._robot.data.joint_names)

    def _by_name(self, values):
        return {name: values[:, i].clone() for i, name in enumerate(self._names)}

    def record_pre_step(self):
        return "lerobot/joint_pos", self._by_name(self._robot.data.joint_pos)

    def record_post_step(self):
        return "lerobot/joint_pos_target", self._by_name(self._robot.data.joint_pos_target)


@configclass
class LeRobotJointsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = LeRobotJointsRecorder


@configclass
class OpenArmLeRobotRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Isaac Lab's actions/states/observations recorders, plus the named joints lerobot needs."""

    record_lerobot_joints = LeRobotJointsRecorderCfg()
