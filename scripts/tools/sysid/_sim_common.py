# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Helpers shared by the sim<->real system-identification tools. Import AFTER the Kit app is up."""

TASK = "Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0"
"""The task whose robot cfg (actuator gains, limits, mimic joints) every OpenArm data set is generated with."""


def strip_sim_cameras(env_cfg) -> list[str]:
    """Drop every camera sensor from ``env_cfg.scene`` and every term that refers to one.

    Same logic as record_demos_openarm.strip_sim_cameras (that module runs argparse at import, so it
    cannot be imported). System identification only needs joint states, and a camera left in the scene
    makes the env refuse to build without --enable_cameras and slows every step by a render.
    """
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import CameraCfg

    cameras = [name for name, value in vars(env_cfg.scene).items() if isinstance(value, CameraCfg)]
    for name in cameras:
        setattr(env_cfg.scene, name, None)

    def refers_to_camera(term) -> bool:
        params = getattr(term, "params", None) or {}
        return any(isinstance(v, SceneEntityCfg) and v.name in cameras for v in params.values())

    def strip_terms(group) -> None:
        for term_name, term in list(vars(group).items()):
            if refers_to_camera(term):
                setattr(group, term_name, None)

    for group in vars(env_cfg.observations).values():
        if group is not None and hasattr(group, "__dict__"):
            strip_terms(group)
    if getattr(env_cfg, "events", None) is not None:
        strip_terms(env_cfg.events)
    if hasattr(env_cfg, "image_obs_list"):
        env_cfg.image_obs_list = []
    return cameras


def build_env(num_envs: int, device: str, task: str = TASK, disable_robot_gravity: bool | None = None):
    """Camera-free env for the OpenArm pick-up task, unwrapped, not yet reset.

    ``disable_robot_gravity`` sets gravity on the ROBOT links only; None keeps what the task cfg does.
    That is DISABLED: openarm.py sets ``OPENARM_BI_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True``,
    which is the sim counterpart of the real follower's model-based gravity feed-forward torque
    (OpenArmFollower.send_action) -- both hold position without sagging by tau_gravity / kp.
    """
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401  (registers the task)
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    strip_sim_cameras(env_cfg)
    if disable_robot_gravity is not None:
        env_cfg.scene.robot.spawn.rigid_props.disable_gravity = disable_robot_gravity
    env_cfg.terminations.success = None  # nothing should end a replay early
    env_cfg.recorders = type(env_cfg.recorders)() if hasattr(env_cfg, "recorders") else None
    return gym.make(task, cfg=env_cfg).unwrapped, env_cfg
