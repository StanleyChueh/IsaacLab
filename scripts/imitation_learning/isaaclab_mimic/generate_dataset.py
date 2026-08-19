# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Main data generation script.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--generation_num_trials", type=int, help="Number of demos to be generated.", default=None)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to instantiate for generating datasets."
)
parser.add_argument("--input_file", type=str, default=None, required=True, help="File path to the source dataset file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded and generated episodes.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="pause after every subtask during generation for debugging - only useful with render flag",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--task_mode",
    type=str,
    default=None,
    choices=["left", "right", "handover"],
    help=(
        "OpenArm pick-up tasks only: which task variant to generate. MUST match the --task_mode"
        " the source demos were annotated with (annotate_demos.py), because the mode decides the"
        " subtask structure and the per-arm subtask term signals generation matches against the"
        " source dataset -- generating a handover dataset against the left-arm mode's signals"
        " finds none of them. Omit to use whatever the task cfg already declares."
    ),
)
parser.add_argument(
    "--use_skillgen",
    action="store_true",
    default=False,
    help="use skillgen to generate motion trajectories",
)
parser.add_argument(
    "--enable_domain_randomization",
    action="store_true",
    default=False,
    help=(
        "Enable optional domain randomization while generating the dataset. Off by default. What"
        " gets randomized is chosen by --domain_randomization_profile. Only takes effect for tasks"
        " whose event cfg defines a matching randomized variant (currently: the OpenArm"
        " pick-up-red-cube task family)."
    ),
)
parser.add_argument(
    "--domain_randomization_profile",
    type=str,
    default="full",
    choices=["full", "visual"],
    help=(
        "How strong the randomization from --enable_domain_randomization is (ignored without it)."
        " 'full' (default, the original behavior) randomizes cube/pad color across the whole RGB"
        " cube, pad position, lighting intensity over a 24x range INCLUDING a background skybox"
        " swap, and all four camera poses. 'visual' is the mild appearance-only profile: lighting"
        " intensity/brightness around nominal with NO background change, surface roughness, colour"
        " SATURATION and brightness around each asset's authored colour (hue untouched, so the red"
        " cube stays red), and camera angle on all four cameras -- pad position and the skybox are"
        " left alone. Use 'visual' when 'full' produces images too far from the real setup to"
        " train against."
    ),
)
parser.add_argument(
    "--randomize_object_size",
    action="store_true",
    default=False,
    help=(
        "OpenArm pick-up tasks only (needs --task_mode): give each environment its own"
        " Pringles-can size, drawn from --object_length_range / --object_radius_range. Off by"
        " default, in which case every env gets the nominal 200 mm x 60 mm can."
        " The sizes actually drawn are printed as a [OBJ SIZE] table, and also written to the"
        " isaaclab log under /tmp/isaaclab/logs/ -- with --enable_cameras the rendering app"
        " captures stdout, so the log file is where to read them back.\n"
        " NOTE: one size per ENVIRONMENT, fixed for the whole run -- USD scale is baked into the"
        " physics at startup and cannot be redrawn per episode, so a run with --num_envs 4 covers"
        " only 4 sizes however many trials it generates. Raise --num_envs (each env is an"
        " independent draw) and/or repeat the run for real coverage. Also forces"
        " replicate_physics off, which makes the scene slower to build."
    ),
)
parser.add_argument(
    "--object_length_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=[-0.025, 0.025],
    help=(
        "Metres to ADD to the can's 200 mm length (ignored without --randomize_object_size)."
        " Default -0.025 0.025 spans the requested 5 cm of variation, symmetric about the real"
        " can, i.e. lengths of 175-225 mm. Pass '0 0.05' instead to only ever grow it."
    ),
)
parser.add_argument(
    "--object_radius_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=[-0.01, 0.01],
    help=(
        "Metres to ADD to the can's 30 mm radius (ignored without --randomize_object_size)."
        " Default -0.01 0.01 spans the requested 2 cm of variation, symmetric about the real can,"
        " i.e. diameters of 40-80 mm. Note the ceiling: the hand opens to 88 mm, so anything past"
        " +0.014 here draws cans that cannot be grasped at all -- '0 0.02' (up to 100 mm across)"
        " is over it, and the script warns rather than clamping."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version
    # installed by IsaacLab and not the one installed by Isaac Sim.
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import inspect
import logging
import random

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv

import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401

# import logger
logger = logging.getLogger(__name__)


def main():
    num_envs = args_cli.num_envs

    # Setup output paths and get env name
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    task_name = args_cli.task
    if task_name:
        task_name = args_cli.task.split(":")[-1]
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)

    # Configure environment
    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
        task_mode=args_cli.task_mode,
    )

    # Optional domain randomization -- opt-in via --enable_domain_randomization, off by default,
    # with --domain_randomization_profile choosing how strong it is.
    if args_cli.enable_domain_randomization:
        from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.pickup_ik_abs_env_cfg import (
            PickUpEventCfg,
            attach_domain_randomization,
        )

        if isinstance(env_cfg.events, PickUpEventCfg):
            # Attaches to the existing events cfg rather than replacing it -- the task mode has
            # already patched that object, and a fresh instance would throw those patches away
            # (see attach_domain_randomization's docstring for what exactly breaks).
            profile = args_cli.domain_randomization_profile
            attached = attach_domain_randomization(env_cfg, profile)
            print(f"[DR] Domain randomization enabled (profile '{profile}'): {', '.join(attached)}")
            if profile == "visual":
                print(
                    "[DR] Lighting varies WITHOUT a background/skybox swap; object and pad vary in"
                    " saturation/brightness/roughness with hue unchanged; camera angle varies on"
                    " all four cameras; pad position is not jittered."
                )
        else:
            print(
                f"[DR] --enable_domain_randomization was set, but task '{env_name}' has no matching"
                " domain-randomization event cfg; ignoring."
            )

    # Optional per-env object size randomization -- opt-in via --randomize_object_size.
    # Attached AFTER the task mode (setup_env_config applies it) because the can it resizes is
    # what the task mode puts in the scene, and BEFORE gym.make() because the scale is a USD
    # property that has to be written before the simulation starts.
    if args_cli.randomize_object_size:
        from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_task_modes import (
            attach_object_size_randomization,
        )

        summary = attach_object_size_randomization(
            env_cfg,
            length_delta_range=tuple(args_cli.object_length_range),
            radius_delta_range=tuple(args_cli.object_radius_range),
        )
        print(f"[OBJ SIZE] {summary}")

    # Create environment
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    # Check if the mimic API from this environment contains decprecated signatures
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        logger.warning(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    # Set seed for generation
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    # Reset before starting
    env.reset()

    motion_planners = None
    if args_cli.use_skillgen:
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

        # Create one motion planner per environment
        motion_planners = {}
        for env_id in range(num_envs):
            print(f"Initializing motion planner for environment {env_id}")
            # Create a config instance from the task name
            planner_config = CuroboPlannerCfg.from_task_name(env_name)

            # Ensure visualization is only enabled for the first environment
            # If not, sphere and plan visualization will be too slow in isaac lab
            # It is efficient to visualize the spheres and plan for the first environment in rerun
            if env_id != 0:
                planner_config.visualize_spheres = False
                planner_config.visualize_plan = False

            motion_planners[env_id] = CuroboPlanner(
                env=env,
                robot=env.scene["robot"],
                config=planner_config,  # Pass the config object
                env_id=env_id,  # Pass environment ID
            )

        env.cfg.datagen_config.use_skillgen = True

    # Setup and run async data generation
    async_components = setup_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
        motion_planners=motion_planners,  # Pass the motion planners dictionary
    )

    try:
        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        print("Tasks were cancelled.")
    finally:
        # Cancel all async tasks when env_loop finishes
        data_gen_tasks.cancel()
        try:
            # Wait for tasks to be cancelled
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            print("Remaining async tasks cancelled and cleaned up.")
        except Exception as e:
            print(f"Error cancelling remaining async tasks: {e}")
        # Cleanup of motion planners and their visualizers
        if motion_planners is not None:
            for env_id, planner in motion_planners.items():
                if getattr(planner, "plan_visualizer", None) is not None:
                    print(f"Closing plan visualizer for environment {env_id}")
                    planner.plan_visualizer.close()
                    planner.plan_visualizer = None
            motion_planners.clear()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # Close sim app
    simulation_app.close()
