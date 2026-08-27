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
)

gym.register(
    id="Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pickup_ik_abs_env_cfg.OpenarmPickUpRedCubeEnvCfg,
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
