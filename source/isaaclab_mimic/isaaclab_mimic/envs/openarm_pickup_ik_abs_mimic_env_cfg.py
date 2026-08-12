# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab Mimic env config for the OpenArm pick-up task (IK relative)."""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_task_modes import (
    TASK_MODE_LEFT,
    apply_task_mode,
)
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.pickup_ik_abs_env_cfg import (
    OpenarmPickUpRedCubeEnvCfg,
)


@configclass
class OpenArmPickUpIKAbsMimicEnvCfg(OpenarmPickUpRedCubeEnvCfg, MimicEnvCfg):
    """Isaac Lab Mimic environment config for Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0.

    The subtask structure, the subtask term signals, the success condition and cube_2's spawn
    side all come from a task mode (see
    isaaclab_tasks/.../config/openarm/openarm_task_modes.py). This cfg installs the ``left``
    mode -- single-arm pick-up with the left arm, right arm idle -- so the env is coherent on
    its own; ``annotate_demos.py --task_mode`` / ``generate_dataset.py --task_mode`` re-apply a
    different one on the parsed cfg before gym.make() when the demos were recorded under it.

    Whichever mode is in force, it must be the SAME one the demos were recorded with
    (``record_demos_openarm.py --task_mode``): the mode decides which arm the subtask signals
    describe, and a signal that never fires makes annotate_demos.py reject every episode.
    """

    def __post_init__(self):
        super().__post_init__()

        # Dataset generation settings (relative mode = delta pose controller)
        self.datagen_config.name = "demo_src_pickup_openarm_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        # KEY: False = both subtasks come from ONE source demo → no seam discontinuity.
        # True would pick a different source demo for each subtask, creating a hard
        # EEF-pose jump at the boundary that is the main cause of jerkiness.
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_transform_first_robot_pose = True
        # True = interpolation at subtask seams starts from the LAST COMMANDED pose
        # (smoother when IK tracking is good). Set False if the robot often lags
        # behind its command, which would cause a jump from actual→commanded position.
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # Subtask sequences, per-arm subtask term signals, success condition and cube spawn
        # side. The eef keys this installs ("left_eef"/"right_eef") are the ones
        # OpenArmPickUpIKAbsMimicEnv serves; re-applying another mode later replaces all of it
        # consistently, which is why none of it is spelled out here.
        apply_task_mode(self, TASK_MODE_LEFT)
