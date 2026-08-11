# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_abs_retargeter import Se3AbsRetargeterCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from . import stack_joint_pos_env_cfg

@configclass
class OpenarmCubeStackEnvCfg(stack_joint_pos_env_cfg.OpenarmCubeStackEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 強制使用 IK 動作配置
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_joint[1-7]"],
            body_name="openarm_left_ee_tcp",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            # Both finger joints are commanded. joint2 is a PhysX mimic of joint1 AND is
            # actuated (see openarm.py's "openarm_gripper" comment): the mimic enforces
            # pos2 == pos1 and these two targets are equal, so they agree rather than fight.
            # Naming joint1 alone would leave the actuated joint2 pinned at its default and
            # genuinely fight the mimic.
            joint_names=["openarm_left_finger_joint1", "openarm_left_finger_joint2"],
            open_command_expr={"openarm_left_finger_joint1": 0.044, "openarm_left_finger_joint2": 0.044},
            close_command_expr={"openarm_left_finger_joint1": 0.0, "openarm_left_finger_joint2": 0.0},
        )

        # Teleop 設備配置 (用於鍵盤或手勢錄製)
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3AbsRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, 
                            sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )