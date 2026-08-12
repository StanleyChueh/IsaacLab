# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm cube-stack environment with front + wrist cameras for visuomotor recording."""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp_core
from . import stack_ik_abs_env_cfg


@configclass
class ObservationsCfg:
    """Camera-only policy observations (concatenate_terms=False so each cam is separate)."""

    @configclass
    class PolicyCfg(ObsGroup):
        front_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("front_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        right_wrist_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        body_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("body_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class OpenarmCubeStackVisuomotorEnvCfg(stack_ik_abs_env_cfg.OpenarmCubeStackEnvCfg):
    """OpenArm stack env with two cameras for visuomotor data collection.

    Cameras:
      front_cam — fixed table-side view (640×480)
      wrist_cam — mounted on openarm_left_ee_tcp (640×480)

    Record with:
      ./isaaclab.sh -p scripts/tools/record_demos.py \\
          --task Isaac-Stack-Cube-OpenArm-IK-Abs-Visuomotor-v0 \\
          --dataset_file logs/demos/openarm_visuomotor.hdf5
    """

    def __post_init__(self):
        super().__post_init__()  # runs parent (sets empty policy, IK action, etc.)

        # Override policy observations with camera terms
        self.observations.policy = ObservationsCfg.PolicyCfg()

        # ── Front / table-side camera ─────────────────────────────────────
        # Positioned ~1 m in front of the robot, looking down at the workspace.
        # Adjust pos/rot to match your real camera mount if needed.
        self.scene.front_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 4.0),
            ),
            # ROS convention (w, x, y, z): points toward robot from front-right, slightly down
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.5),
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
                convention="ros",
            ),
        )

        # ── Left wrist camera (attached to openarm_left_ee_tcp) ──────────
        # Position matches the real camera mount defined in the v1_camera URDF
        # (source: isaaclab_assets/data/v1_camera_isaac/urdf/v1_camera.urdf):
        #   openarm_left_camera_joint: parent=openarm_left_link7, xyz=(0,0,0.06), rpy=(0,0,0)
        #   openarm_left_ee_tcp_joint: parent=openarm_left_link7, xyz=(0,0,0), rpy=(0,0,0)
        # ee_tcp is coincident with link7 (identity offset), so relative to ee_tcp the camera
        # mount is a plain +6cm translation along local Z, no rotation -- pos below is that
        # translation. The URDF has no optical-axis convention for the mount (fixed joints only
        # define where the camera sits, not which way its lens points), so `rot` below is kept
        # from the previous manually-tuned value (Isaac Sim stage Orient = (-180, 6.001, -90)
        # degrees) since ee_tcp/link7's orientation didn't change -- only the origin moved from
        # a hand-eyeballed 20cm-behind-the-gripper position to the URDF's true 6cm mount point.
        # Re-check the rendered image after this change: field of view differs substantially
        # this close to the gripper, so `rot` may need re-tuning in the Isaac Sim GUI.
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
                pos=(0.06, 0.001, 0.11125),
                rot=(0.68057, -0.19188, -0.19188, 0.68057),
                convention="ros",
            ),
        )
        # rot:(x,-w,-z,y)=>Please follow this order,don't asy why

        # ── Right wrist camera (attached to openarm_right_ee_tcp) ─────────
        # Same URDF-derived offset as the left wrist camera (openarm_right_camera_joint is
        # identical: parent=openarm_right_link7, xyz=(0,0,0.06), rpy=(0,0,0); ee_tcp is likewise
        # coincident with link7). See the wrist_cam comment above for the full derivation.
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
                pos=(0.06, 0.001, 0.11125),
                rot=(0.68057, -0.19188, -0.19188, 0.68057),
                convention="ros",
            ),
        )

        # ── Body camera (attached to openarm_body_link0, front of torso at shoulder height) ─
        # Positioned at the front face of the central body pillar, ~65 cm above the base,
        # matching the real OpenArm robot's chest-mounted camera location.
        # pos=(0.12, 0, 0.65) relative to openarm_body_link0; looks toward workspace center.
        # Rotation: camera looks forward (+X world) and 60° downward toward the workspace.
        # To retune: adjust pos/rot manually in Isaac Sim, then copy values here.
        self.scene.body_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link0/body_cam",
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
                pos=(0.04563, 0.0, 0.97524),
                rot=(0.1203, -0.6968, 0.6968, -0.1203),
                convention="ros",
            ),
        )
        # rot:(x,-w,-z,y)=>Please follow this order,don't asy why

        # Re-render after reset so the first frames are clean
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"

        # Tells the converter which obs keys are cameras
        self.image_obs_list = ["front_cam", "wrist_cam", "right_wrist_cam", "body_cam"]
