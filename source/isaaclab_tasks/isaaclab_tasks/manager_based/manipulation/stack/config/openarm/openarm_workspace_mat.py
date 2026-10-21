# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""The two thin black mats the real rig's can stands on, drawn on top of the sim's workspace pad.

Measured off the real pringles recording (ethanCSL/openarm_pringles_lerobot_real_v00): the median of its 50 episode
start frames from body_cam, back-projected through the calibrated body_cam onto the pad-top plane (z 0.28). Two mats
lie side by side on the white table with a ~4 mm gap at y +0.035..+0.039 (left of the robot's centre line, not on it),
and their far edges end at x 0.458, leaving ~5 cm of white table before its far edge. The outer and near edges are under
the arms in every frame, so the mats run to the pad's own side and near edges there. Corners are rounded (~1.5 cm).

The mats are children of ``workspace_pad`` so they follow everything that moves the pad: the DR profile's position
jitter and its prestartup height rescale. They are VISUAL ONLY and their top is only ``MAT_LIFT`` above the pad top:
the can's resting height (pad top = 2 x the pad's z, used across openarm_task_modes) was calibrated against the real
can standing on these very mats, so the surface the sim can rests on stays where it was; the rest of each mat's
thickness lies inside the pad box. The pad-colour randomizer binds on the pad's own geometry prim, so it does not touch
them. The real-look profile hides the whole pad (and with it the mats) behind its photographed backdrop.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.shapes.shapes import spawn_cuboid
from isaaclab.sim.utils import bind_visual_material, clone
from isaaclab.utils import configclass

MAT_THICKNESS = 0.003
"""m: a thin rubber desk mat."""
MAT_LIFT = 0.0005
"""m the mats' top sits above the pad top: enough to never z-fight the pad's top face, too little to see the can sink."""
MAT_CORNER_RADIUS = 0.015
MAT_COLOR = (0.008, 0.008, 0.009)
MAT_RECTS = (
    # (x_near, x_far, y_right, y_left) in the env frame, m
    (0.03, 0.458, 0.039, 0.285),  # robot's left (+y) mat
    (0.03, 0.458, -0.285, 0.035),  # robot's right (-y) mat
)


def _rounded_rect(x0: float, x1: float, y0: float, y1: float, r: float, n: int = 6) -> list[tuple[float, float]]:
    """Counter-clockwise outline of a rounded rectangle (convex, so a fan from its centre triangulates it)."""
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = []
    for cx, cy, a0 in ((x1 - r, y1 - r, 0.0), (x0 + r, y1 - r, 90.0), (x0 + r, y0 + r, 180.0), (x1 - r, y0 + r, 270.0)):
        for k in range(n + 1):
            a = math.radians(a0 + 90.0 * k / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _define_mat(stage, path: str, outline: list[tuple[float, float]], z0: float, z1: float):
    """A closed slab mesh: ``outline`` extruded from z0 to z1."""
    from pxr import Gf, UsdGeom, Vt

    m = len(outline)
    cx, cy = sum(p[0] for p in outline) / m, sum(p[1] for p in outline) / m
    points = [Gf.Vec3f(x, y, z0) for x, y in outline] + [Gf.Vec3f(x, y, z1) for x, y in outline]
    points += [Gf.Vec3f(cx, cy, z0), Gf.Vec3f(cx, cy, z1)]
    bottom_c, top_c = 2 * m, 2 * m + 1
    counts, idx = [], []
    for i in range(m):
        j = (i + 1) % m
        idx += [top_c, m + i, m + j]  # top, facing +z
        idx += [bottom_c, j, i]  # bottom, facing -z
        idx += [i, j, m + j, m + i]  # side, facing out
        counts += [3, 3, 4]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*(min(p[k] for p in points) for k in range(3))),
                                         Gf.Vec3f(*(max(p[k] for p in points) for k in range(3)))]))
    return mesh


@clone
def spawn_pad_with_mats(prim_path: str, cfg: PadWithMatsCfg, translation=None, orientation=None, **kwargs):
    """``spawn_cuboid``, then the mats under the pad prim, in its frame (the pad is unrotated and centred on ``translation``)."""
    from isaaclab.sim.utils import get_current_stage

    prim = spawn_cuboid(prim_path, cfg, translation, orientation, **kwargs)
    stage = get_current_stage()
    tx, ty, tz = translation if translation is not None else (0.0, 0.0, cfg.size[2] / 2)
    top = cfg.size[2] / 2 + cfg.mat_lift  # pad-local z of the mats' top face
    for k, (x0, x1, y0, y1) in enumerate(cfg.mat_rects):
        outline = _rounded_rect(x0 - tx, x1 - tx, y0 - ty, y1 - ty, cfg.mat_corner_radius)
        _define_mat(stage, f"{prim_path}/mats/mat_{k}", outline, top - cfg.mat_thickness, top)
    material_path = f"{prim_path}/mats/material"
    cfg.mat_material.func(material_path, cfg.mat_material)
    bind_visual_material(f"{prim_path}/mats", material_path)
    return prim


@configclass
class PadWithMatsCfg(sim_utils.CuboidCfg):
    """The workspace pad's cuboid plus the real rig's black mats on its top face."""

    func: Callable = spawn_pad_with_mats
    mat_rects: tuple = MAT_RECTS
    mat_thickness: float = MAT_THICKNESS
    mat_lift: float = MAT_LIFT
    mat_corner_radius: float = MAT_CORNER_RADIUS
    mat_material: sim_utils.PreviewSurfaceCfg = sim_utils.PreviewSurfaceCfg(diffuse_color=MAT_COLOR, roughness=0.9)


def add_workspace_mats(env_cfg) -> None:
    """Swap ``env_cfg.scene.workspace_pad``'s spawner for one that also lays the mats (same cuboid, same collider)."""
    spawn = env_cfg.scene.workspace_pad.spawn
    fields = {f.name: getattr(spawn, f.name) for f in dataclasses.fields(spawn) if f.name != "func"}
    env_cfg.scene.workspace_pad.spawn = PadWithMatsCfg(**fields)
