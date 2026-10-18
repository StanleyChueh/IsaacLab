# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The OpenArm pick-up task, rendered to look like the real rig (``Isaac-PickUp-RedCube-OpenArm-IK-Abs-v1``).

Exactly :class:`pickup_ik_abs_env_cfg.OpenarmPickUpRedCubeEnvCfg` -- same scene, physics, actions, observations,
success and task modes -- with the look fitted from the real dataset applied (see :mod:`openarm_real_look`): the
real robot's colours and lighting, the photo-textured table and room behind it, and the real cameras' blur and
noise, which ``lerobot_sim_server.py`` applies on its own when a task carries ``real_look_dir``. The physics is
untouched; the pad and table keep their colliders and only stop being drawn.

Domain randomization: use ``--domain_randomization none``. The randomizers move the pad and change the light, both of
which the photo backdrop has baked in.
"""

from isaaclab.utils import configclass

from . import openarm_real_look, pickup_ik_abs_env_cfg


@configclass
class OpenarmPickUpRedCubeRealLookEnvCfg(pickup_ik_abs_env_cfg.OpenarmPickUpRedCubeEnvCfg):
    """Isaac-PickUp-RedCube-OpenArm-IK-Abs-v1."""

    real_look_dir: str = openarm_real_look.LOOK_DIR
    """Folder of the fitted look. The server reads it to apply the camera sensor model too."""

    def __post_init__(self):
        super().__post_init__()
        applied = openarm_real_look.apply_real_look(self, self.real_look_dir)
        print(f"[OpenarmPickUpRedCubeRealLookEnvCfg] real look: {'; '.join(applied) or 'nothing found in ' + self.real_look_dir}")
