
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm dual-arm cube-stack environment with wrist + body cameras and absolute IK control for visuomotor recording."""

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from . import stack_joint_pos_env_cfg

@configclass
class ObservationsCfg:
    """3 台相機視覺觀測配置"""
    @configclass
    class PolicyCfg(ObsGroup):
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        right_wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        body_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("body_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()

@configclass
class OpenarmDualArmCubeStackVisuomotorAbsIKEnvCfg(stack_joint_pos_env_cfg.OpenarmCubeStackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # self.scene.robot.spawn.usd_path = f"path/to/your_custom_openarm.usd"

        self.observations.policy = ObservationsCfg.PolicyCfg()
        
        self.actions.left_arm = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_joint[1-7]"],
            body_name="openarm_left_ee_tcp",
            controller=DifferentialIKControllerCfg(
                command_type="pose", 
                use_relative_mode=False, 
                ik_method="dls"
            ),
        )
        
        self.actions.right_arm = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_joint[1-7]"],
            body_name="openarm_right_ee_tcp",
            controller=DifferentialIKControllerCfg(
                command_type="pose", 
                use_relative_mode=False, 
                ik_method="dls"
            ),
        )

        self.actions.left_gripper = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_finger_joint.*"],
            open_command_expr={"openarm_left_finger_joint.*": 0.044},
            close_command_expr={"openarm_left_finger_joint.*": 0.0},
        )
        self.actions.right_gripper = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],
            open_command_expr={"openarm_right_finger_joint.*": 0.044},
            close_command_expr={"openarm_right_finger_joint.*": 0.0},
        )

        # ── Left wrist camera (attached to openarm_left_ee_tcp) ──────────
        # Position tuned manually in Isaac Sim:
        #   Isaac Sim stage Translate = (0.06156, 0.00072, -0.20867)
        #   Isaac Sim stage Orient    = (-180, 6.001, -90) degrees
        # The translation maps directly to pos; the orientation is unchanged
        # from the original (±180° on X is the same rotation).
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_left_ee_tcp/wrist_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.06156, 0.00072, -0.20867),
                rot=(-0.70614, 0.03701, 0.03701, -0.70614),
                convention="ros",
            ),
        )

        # ── Right wrist camera (attached to openarm_right_ee_tcp) ─────────
        # Same offset as the left wrist camera.
        self.scene.right_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_right_ee_tcp/right_wrist_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.06156, 0.00072, -0.20867),
                rot=(-0.70614, 0.03701, 0.03701, -0.70614),
                convention="ros",
            ),
        )

        # ── Body camera (attached to openarm_body_link, front of torso at shoulder height) ─
        # Positioned at the front face of the central body pillar, ~65 cm above the base,
        # matching the real OpenArm robot's chest-mounted camera location.
        # pos=(0.12, 0, 0.65) relative to openarm_body_link; looks toward workspace center.
        # Rotation: camera looks forward (+X world) and 60° downward toward the workspace.
        # To retune: adjust pos/rot manually in Isaac Sim, then copy values here.
        self.scene.body_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link/body_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 3.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.01, 0.0, 0.65),
                rot=(0.17853, -0.68420, 0.68420, -0.17853),
                convention="ros",
            ),
        )
        
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"
        self.image_obs_list = ["wrist_cam", "right_wrist_cam", "body_cam"]