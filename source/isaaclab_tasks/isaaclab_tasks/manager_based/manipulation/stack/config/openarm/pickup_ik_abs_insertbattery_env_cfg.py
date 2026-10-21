# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm battery insertion, rendered to look like the real rig (``Isaac-InsertBattery-OpenArm-IK-Abs-v0``).

The real task (ethanCSL/openarm_insertbattery_lerobot_real_v01): the left arm picks up a charging station and the right arm
puts an AA battery into it. The scene is the OpenArm pick-up scene with the can replaced by the two props of
:mod:`openarm_battery_assets` and the real look of ``battery_v01`` (:mod:`openarm_real_look`) applied: fitted robot colours,
light and camera softness, and a photo backdrop built from this dataset's own frames (it has a white panel the pick-up
scene does not).

The charger has one open bay (the empty one in the photo) the battery can be set into; ``battery_in_charger`` is the success
term. There are no subtask signals; episodes end when the recorder says so. Objects start where the 9 real episodes had them
(``reset_root_state_uniform`` around their median, over their observed spread).

Not a can task: ``lerobot_sim_server.py`` skips ``apply_task_mode`` for it (``task_mode_applicable``).
"""

import os

import isaaclab.envs.mdp as mdp_core
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from . import openarm_battery_assets, openarm_real_look, pickup_ik_abs_env_cfg

BATTERY_LOOK_DIR = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "real2sim", "battery_v01")
PAD_TOP = 0.28
CHARGER_START = (0.325, 0.070)
BATTERY_START = (0.320, -0.092)


@configclass
class InsertBatteryEventCfg:
    """Reset the scene, then scatter the two props over the spread the 9 real episodes showed."""

    init_robot_pose = EventTerm(func=mdp_core.reset_scene_to_default, mode="reset")
    sync_camera_mounts = pickup_ik_abs_env_cfg.PickUpEventCfg().sync_camera_mounts
    randomize_charger = EventTerm(
        func=mdp_core.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.035, 0.025), "y": (-0.01, 0.02), "yaw": (-0.30, 0.08)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("charger"),
        },
    )
    randomize_battery = EventTerm(
        func=mdp_core.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.07, 0.05), "y": (-0.02, 0.012), "yaw": (-0.15, 0.12)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("battery"),
        },
    )


@configclass
class OpenarmInsertBatteryEnvCfg(pickup_ik_abs_env_cfg.OpenarmPickUpRedCubeEnvCfg):
    """Isaac-InsertBattery-OpenArm-IK-Abs-v0."""

    real_look_dir: str = BATTERY_LOOK_DIR
    task_mode_applicable: bool = False
    report_objects: tuple = ("charger", "battery")
    arm_gravity_comp: bool = False
    robot_self_collisions: bool = (os.environ.get("OPENARM_SELF_COLLISIONS", "0") == "1")
    """The arms' links feel no gravity (a gravity-compensated arm). The PD gains are the real arm's (kp 230-290 on the big joints), and
    at a table-top grasp the sim arm's own weight sags the shoulder by 0.1-0.5 rad and saturates its torque, so the hand never
    gets within centimetres of an object it was sent to. The props (charger, battery) keep their weight."""

    def __post_init__(self):
        super().__post_init__()  # scene, cameras, actions, pad -- and the cube, which goes below
        spawn = self.scene.robot.spawn
        self.scene.robot.spawn = spawn.replace(articulation_props=spawn.articulation_props.replace(enabled_self_collisions=self.robot_self_collisions))
        if self.arm_gravity_comp:
            spawn = self.scene.robot.spawn
            self.scene.robot.spawn = spawn.replace(rigid_props=spawn.rigid_props.replace(disable_gravity=True))
        self.scene.cube_2 = None
        self.scene.contact_right_left_finger = None  # it was filtered against the cube
        self.observations.subtask_terms = None  # grasp / lift signals read the cube
        self.terminations.success = DoneTerm(func=openarm_battery_assets.battery_in_charger)
        self.terminations.time_out = None

        self.scene.charger = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Charger",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(*CHARGER_START, PAD_TOP + openarm_battery_assets.CHARGER_SIZE[2] / 2 + 0.0005), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=openarm_battery_assets.charger_cfg(self.real_look_dir),
        )
        self.scene.battery = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Battery",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(*BATTERY_START, PAD_TOP + openarm_battery_assets.BATTERY_RADIUS + 0.0005), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=openarm_battery_assets.battery_cfg(),
        )
        self.events = InsertBatteryEventCfg()
        applied = openarm_real_look.apply_real_look(self, self.real_look_dir)
        print(f"[OpenarmInsertBatteryEnvCfg] real look: {'; '.join(applied) or 'nothing found in ' + self.real_look_dir}")
