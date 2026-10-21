# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm pick-up task registrations.

Both tasks below are the SAME task on two robot variants, and both manipulate a chips can
(pringles), not a cube -- the "RedCube" in the ids is historical and is kept only because the
recorded HDF5 datasets, the Mimic env id and every command in the README already reference it.
The can is swapped in for cube_2 by ``apply_task_mode`` (see :mod:`openarm_task_modes`), which
is why running either task WITHOUT ``--task_mode`` silently gets you the original cube --
``record_demos_openarm.py`` and ``eval_smolvla_jointspace.py`` both refuse that up front via
:data:`openarm_task_modes.CAN_TARGET_TASKS`.

The cube stacking / cube reach variants that used to be registered here are gone. The
``stack_*_env_cfg`` modules that remain are NOT tasks any more -- they are the base-class chain
(``stack_joint_pos`` -> ``stack_ik_abs`` -> ``stack_ik_abs_visuomotor``) that carries the scene,
robot, pad and camera definitions the pick-up cfg inherits, so they stay importable but
unregistered.
"""

import gymnasium as gym

from . import (
    pickup_ik_abs_cammount_env_cfg,
    pickup_ik_abs_env_cfg,
    pickup_ik_abs_insertbattery_env_cfg,
    pickup_ik_abs_reallook_env_cfg,
    plate_wipe_ik_abs_env_cfg,
)

gym.register(
    id="Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pickup_ik_abs_env_cfg.OpenarmPickUpRedCubeEnvCfg,
    },
)

# The same task rendered to look like the real rig (fitted robot colours, light, photo table and room, camera sensor model --
# openarm_real_look.py) with the contact physics fitted to the real episodes. Actions and success are identical to v0;
# domain randomization works on top (openarm_real_look_randomization.py). Mimic: Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v1.
gym.register(
    id="Isaac-PickUp-RedCube-OpenArm-IK-Abs-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pickup_ik_abs_reallook_env_cfg.OpenarmPickUpRedCubeRealLookEnvCfg,
    },
)

# Battery insertion (left arm lifts the charging station, right arm brings the battery), real look built in. The charger's
# slots are drawn, not modelled: grasp-level physics only. See pickup_ik_abs_insertbattery_env_cfg.py.
gym.register(
    id="Isaac-InsertBattery-OpenArm-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pickup_ik_abs_insertbattery_env_cfg.OpenarmInsertBatteryEnvCfg,
    },
)

# Camera-mount robot variant -- does not replace or alias the task above, both remain available.
gym.register(
    id="Isaac-PickUp-RedCube-OpenArm-CamMount-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pickup_ik_abs_cammount_env_cfg.OpenarmPickUpRedCubeCamMountEnvCfg,
    },
)

# Bimanual plate-wiping task: right arm picks the plate off a dish rack, left arm wipes it with
# a rag, left arm drops the rag, right arm returns the plate. Raw teleop recording only for now
# -- not in openarm_task_modes.CAN_TARGET_TASKS, so --task_mode stays optional/unused for it.
gym.register(
    id="Isaac-PlateWipe-OpenArm-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": plate_wipe_ik_abs_env_cfg.OpenarmPlateWipeEnvCfg,
    },
)
