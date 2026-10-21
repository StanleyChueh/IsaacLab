# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Keep link-mounted cameras on their authored mount under Fabric.

`XformPrimView` initializes Fabric lazily, on a camera's first world-pose read
(`_initialize_fabric` + `_sync_fabric_from_usd_once`), by copying the camera's USD WORLD transform
into Fabric. PhysX never writes link poses back to USD, so that world transform is
`rest_parent * offset` -- the robot's authored rest pose -- while the parent link's Fabric matrix is
already the live physics pose. The Fabric hierarchy then keeps the camera's LOCAL matrix as
`live_parent^-1 * rest_parent * offset`, and that error rides along with the link for the rest of
the session, whatever the USD xformOps say.

On the OpenArm the first read happens after the arms have sagged for a few physics steps from the
rest pose, so the error is a small rotation about the shoulder: measured 1.3 deg / 1.5 cm for
wrist_cam and 4.9 deg / 4.2 cm for right_wrist_cam, which was enough to push the right gripper's
fingertips out of right_wrist_cam's view. body_cam hangs off the base link (rest == live), so it
was never affected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.sim.utils.stage import get_current_stage, get_current_stage_id

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def sync_mounted_camera_fabric_pose(env: ManagerBasedEnv, env_ids: torch.Tensor | None, camera_names: tuple[str, ...]):
    """Rewrite each camera's Fabric local matrix from its USD local transform (the cfg mount offset).

    Run as a "reset" event; cheap (one matrix per camera per reset env). Cameras the cfg dropped are skipped.
    """
    import usdrt
    from pxr import UsdGeom

    stage = get_current_stage()
    rt_stage = usdrt.Usd.Stage.Attach(get_current_stage_id())
    ids = range(env.num_envs) if env_ids is None else env_ids.tolist()
    for name in camera_names:
        if name not in env.scene.sensors:
            continue
        view = env.scene[name]._view
        view.get_world_poses()  # force the one-time USD->Fabric sync now, so it cannot overwrite this later
        for env_id in ids:
            path = view.prim_paths[env_id]
            local = UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation()
            rt_stage.GetPrimAtPath(path).GetAttribute("omni:fabric:localMatrix").Set(
                usdrt.Gf.Matrix4d(*[local[i][j] for i in range(4) for j in range(4)])
            )
