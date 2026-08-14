# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of OpenArm robots.

The following configurations are available:

* :obj:`OPENARM_BI_CFG`: OpenArm robot with two arms.
* :obj:`OPENARM_BI_HIGH_PD_CFG`: OpenArm robot with two arms and stiffer PD control.
* :obj:`OPENARM_UNI_CFG`: OpenArm robot with one arm.
* :obj:`OPENARM_UNI_HIGH_PD_CFG`: OpenArm robot with one arm and stiffer PD control.

References:
OpenArm repositories:
* https://github.com/enactic/openarm
* https://github.com/enactic/openarm_isaac_lab

Motor spec sheets:
* Joint 1–2 (DM-J8009P-2EC):
    https://cdn.shopify.com/s/files/1/0673/6848/5000/files/DM-J8009P-2EC_User_Manual.pdf?v=1755481750
* Joint 3–4 (DM-J4340P-2EC / DM-J4340-2EC):
    https://cdn.shopify.com/s/files/1/0673/6848/5000/files/DM-J4340-2EC_User_Manual.pdf?v=1756883905
* Joint 5–8 (DM-J4310-2EC V1.1):
    https://files.seeedstudio.com/products/Damiao/DM-J4310-en.pdf
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

OPENARM_BI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/csl/Stanley_ws/IsaacLab/source/isaaclab_assets/data/v1_camera_isaac/v1_camera_isaac.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "openarm_left_joint1": 0.0,
            "openarm_left_joint2": 0.0,
            "openarm_left_joint3": 0.0,
            "openarm_left_joint4": 1.570796,
            "openarm_left_joint5": 0.0,
            "openarm_left_joint6": 0.0,
            "openarm_left_joint7": 0.0,
            
            "openarm_right_joint1": 0.0,
            "openarm_right_joint2": 0.0,
            "openarm_right_joint3": 0.0,
            "openarm_right_joint4": 1.570796, 
            "openarm_right_joint5": 0.0,
            "openarm_right_joint6": 0.0,
            "openarm_right_joint7": 0.0,
            "openarm_left_finger_joint.*": 0.044,
            "openarm_right_finger_joint.*": 0.044,
        },
    ),
    # spec sheet for reference
    # DM-J8009P-2EC (Joint 1, 2):
    # https://cdn.shopify.com/s/files/1/0673/6848/5000/files/DM-J8009P-2EC_User_Manual.pdf?v=1755481750
    # DM-J4340P-2EC, DM-J4340-2EC (Joint 3, 4):
    # https://cdn.shopify.com/s/files/1/0673/6848/5000/files/DM-J4340-2EC_User_Manual.pdf?v=1756883905
    # DM-J4310-2EC V1.1 (Joint 5, 6, 7, 8):
    # https://files.seeedstudio.com/products/Damiao/DM-J4310-en.pdf
    actuators={
        "openarm_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "openarm_left_joint[1-7]",
                "openarm_right_joint[1-7]",
            ],
            velocity_limit_sim={
                "openarm_left_joint[1-2]": 2.175,
                "openarm_right_joint[1-2]": 2.175,
                "openarm_left_joint[3-4]": 2.175,
                "openarm_right_joint[3-4]": 2.175,
                "openarm_left_joint[5-7]": 2.61,
                "openarm_right_joint[5-7]": 2.61,
            },
            effort_limit_sim={
                "openarm_left_joint[1-2]": 40.0,
                "openarm_right_joint[1-2]": 40.0,
                "openarm_left_joint[3-4]": 27.0,
                "openarm_right_joint[3-4]": 27.0,
                "openarm_left_joint[5-7]": 7.0,
                "openarm_right_joint[5-7]": 7.0,
            },
            stiffness=1000.0,
            damping=10.0,
        ),
        "openarm_gripper": ImplicitActuatorCfg(
            # BOTH finger joints must be listed. *_finger_joint2 (which drives the
            # *_left_finger link) is additionally a native PhysX mimic joint of joint1
            # (gearing=-1) baked into v1_camera_isaac.usd by the URDF importer, and it is
            # tempting to therefore leave it out and let the mimic carry it. That is wrong:
            # a joint absent from every actuator group keeps the USD's drive, and the
            # importer authors NO PhysicsDriveAPI on a mimic joint, so joint2 came up with
            # stiffness=0, damping=0 and effort_limit=0 -- measured directly off
            # root_physx_view. Its only tie to joint1 was then the mimic constraint, which
            # the importer makes *compliant* (physxMimicJoint:rotY:naturalFrequency=25,
            # dampingRatio=0.005). Per PhysX's own compliance formulation that lands ~23x
            # softer than a rigid constraint, so the left finger simply yielded whenever it
            # pushed on anything -- the gripper looked like it closed but could not hold.
            # Listing joint2 here does NOT fight the mimic: the mimic enforces pos2 == pos1,
            # and the action terms command both joints the same value, so drive and
            # constraint agree. Keep the two in sync -- every BinaryJointPositionActionCfg
            # for this robot must name both finger joints (see the openarm stack/pickup
            # env cfgs); driving joint1 alone while joint2 is actuated would peg joint2 at
            # its default and genuinely fight the mimic.
            joint_names_expr=[
                "openarm_left_finger_joint.*",
                "openarm_right_finger_joint.*",
            ],
            velocity_limit_sim=2.0,
            effort_limit_sim=333.33,
            stiffness=3000,
            damping=30,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of OpenArm Bimanual robot."""

OPENARM_UNI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/OpenArm/openarm_unimanual/openarm_unimanual.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "openarm_joint1": 1.57,
            "openarm_joint2": 0.0,
            "openarm_joint3": -1.57,
            "openarm_joint4": 1.57,
            "openarm_joint5": 0.0,
            "openarm_joint6": 0.0,
            "openarm_joint7": 0.0,
            "openarm_finger_joint.*": 0.044,
        },
    ),
    actuators={
        "openarm_arm": ImplicitActuatorCfg(
            joint_names_expr=["openarm_joint[1-7]"],
            velocity_limit_sim={
                "openarm_joint[1-2]": 2.175,
                "openarm_joint[3-4]": 2.175,
                "openarm_joint[5-7]": 2.61,
            },
            effort_limit_sim={
                "openarm_joint[1-2]": 40.0,
                "openarm_joint[3-4]": 27.0,
                "openarm_joint[5-7]": 7.0,
            },
            stiffness=2000.0,
            damping=10.0,
        ),
        "openarm_gripper": ImplicitActuatorCfg(
            joint_names_expr=["openarm_finger_joint.*"],
            velocity_limit_sim=0.2,
            effort_limit_sim=333.33,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of OpenArm Unimanual robot."""

OPENARM_BI_HIGH_PD_CFG = OPENARM_BI_CFG.copy()
OPENARM_BI_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
OPENARM_BI_HIGH_PD_CFG.actuators["openarm_arm"].stiffness = 2000
OPENARM_BI_HIGH_PD_CFG.actuators["openarm_arm"].damping = 10.0
OPENARM_BI_HIGH_PD_CFG.actuators["openarm_gripper"].stiffness = 500.0
OPENARM_BI_HIGH_PD_CFG.actuators["openarm_gripper"].damping = 10.0
"""Configuration of OpenArm Bimanual robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""

OPENARM_UNI_HIGH_PD_CFG = OPENARM_UNI_CFG.copy()
OPENARM_UNI_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
OPENARM_UNI_HIGH_PD_CFG.actuators["openarm_arm"].stiffness = 400.0
OPENARM_UNI_HIGH_PD_CFG.actuators["openarm_arm"].damping = 80.0
"""Configuration of OpenArm Unimanual robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
