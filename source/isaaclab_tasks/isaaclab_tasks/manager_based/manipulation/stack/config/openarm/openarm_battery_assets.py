# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The battery-insertion task's two props, built from the real recordings (ethanCSL/openarm_insertbattery_lerobot_real_v01).

* **Charger** -- the white "enerpad" charging station the left arm picks up: a rigid box of the footprint measured from the
  real body-camera frames (11.5 x 6.5 cm; ``prepare_battery_replay.py``) and the thickness at which the real left gripper's
  jaws stop when they hold it (~3.75 cm), with its top face textured by the rectified real photo
  (``build_charger_texture.py``): lid, three batteries in their slots, the empty slot, the LED.
* **Battery** -- one AA cell lying flat, 5.05 cm x 1.45 cm, 27 g.

The leftmost, empty slot has an approximate compound box collider, so a battery can physically enter it.
Its dimensions are estimated from the rectified photo and must be replaced with measured geometry before claiming
sub-millimetre insertion fidelity (see the sysid README, "Battery insertion").
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import MISSING

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

CHARGER_SIZE = (0.115, 0.0652, 0.0375)
"""(length, width, height) in m: the measured footprint (11.50 x 6.52 cm, std 0.15 cm over 9 episodes) and the jaw gap while held."""
BATTERY_RADIUS, BATTERY_LENGTH, BATTERY_MASS = 0.00725, 0.0505, 0.027
CHARGER_MASS = 0.25
"""kg. An estimate (body plus three AA cells at 27 g); the contact fit can move it."""
EMPTY_SLOT_CENTER = (-0.011, 0.0215)
EMPTY_SLOT_SIZE = (0.056, 0.0175)
EMPTY_SLOT_DEPTH = 0.009
"""Metres in charger local x/y/z, estimated from charger_top.png. The leftmost bay is empty."""


def _slot_collider_parts():
    """Five non-overlapping boxes that form a charger body with one open rectangular bay."""
    L, W, T = CHARGER_SIZE
    sx, sy = EMPTY_SLOT_CENTER
    sl, sw = EMPTY_SLOT_SIZE
    d = EMPTY_SLOT_DEPTH
    x0, x1 = sx - sl / 2, sx + sl / 2
    y0, y1 = sy - sw / 2, sy + sw / 2
    return (
        ("base", (L, W, T - d), (0.0, 0.0, -d / 2)),
        ("end_pos", (L / 2 - x1, W, d), ((x1 + L / 2) / 2, 0.0, (T - d) / 2)),
        ("end_neg", (x0 + L / 2, W, d), ((x0 - L / 2) / 2, 0.0, (T - d) / 2)),
        ("side_pos", (sl, W / 2 - y1, d), (sx, (y1 + W / 2) / 2, (T - d) / 2)),
        ("side_neg", (sl, y0 + W / 2, d), (sx, (y0 - W / 2) / 2, (T - d) / 2)),
    )


def spawn_charger(prim_path: str, cfg: "ChargerCfg", translation=None, orientation=None, **kwargs):
    """A rigid cuboid with the real top-face photo on it, as a fine vertex-coloured, emissive, non-colliding grid.

    (A UsdUVTexture decal rendered black here, while vertex colours through an emissive UsdPreviewSurface are what the
    backdrop already uses, so the photo is baked into vertices: 0.5 mm apart, finer than the photo's own softness.)
    """
    import json

    from PIL import Image
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

    from isaaclab.sim.spawners.shapes.shapes import spawn_cuboid
    from isaaclab.sim import schemas

    from .openarm_real_look import emissive_from_srgb

    # The cuboid spawner binds its material to the root prim "stronger than descendants", which would override the
    # decal's own: spawn it unbound, then bind the body material to the box mesh alone.
    body_material, cfg.visual_material = cfg.visual_material, None
    try:
        prim = spawn_cuboid.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)
    finally:
        cfg.visual_material = body_material
    if body_material is not None:
        material_path = f"{prim_path}/geometry/body_material"
        body_material.func(material_path, body_material)
        sim_utils.bind_visual_material(f"{prim_path}/geometry/mesh", material_path, stronger_than_descendants=False)
    # Keep the root rigid body and its mass, but replace the cuboid collider and visual with
    # a compound five-box body. This leaves the empty bay physically open. All boxes belong
    # to the same rigid body; the photo decal below is visual only.
    stage = sim_utils.get_current_stage()
    original = stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")
    UsdGeom.Imageable(original).MakeInvisible()
    UsdPhysics.CollisionAPI(original).CreateCollisionEnabledAttr(False)
    physics_path = cfg.physics_material_path
    physics_path = physics_path if physics_path.startswith("/") else f"{prim_path}/geometry/{physics_path}"
    for name, size, centre in _slot_collider_parts():
        path = f"{prim_path}/geometry/{name}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        translation_op = cube.AddTranslateOp()
        translation_op.Set(Gf.Vec3d(*centre))
        scale_op = cube.AddScaleOp()
        scale_op.Set(Gf.Vec3d(*size))
        cube.SetXformOpOrder([translation_op, scale_op])
        schemas.define_collision_properties(path, cfg.collision_props, stage=stage)
        if body_material is not None:
            sim_utils.bind_visual_material(path, material_path, stronger_than_descendants=False)
        if cfg.physics_material is not None:
            sim_utils.bind_physics_material(path, physics_path, stage=stage)
    if not (cfg.top_texture and os.path.isfile(cfg.top_texture)):
        return prim
    L, W, T = cfg.size
    nx, ny = int(round(L / 0.0005)), int(round(W / 0.0005))  # rows along local x, columns along local y
    # the photo: row 0 = the +x end, column 0 = the +y side (build_charger_texture.py)
    img = np.array(Image.open(cfg.top_texture).convert("RGB").resize((ny + 1, nx + 1), Image.BILINEAR))
    tone = json.load(open(cfg.tone_path)) if cfg.tone_path and os.path.isfile(cfg.tone_path) else None
    colors = np.clip(emissive_from_srgb(img, tone), 0.0, 1.0).reshape(-1, 3).astype(np.float32)
    rows, cols = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
    x = L / 2 - rows * (L / nx)
    y = W / 2 - cols * (W / ny)
    pts = np.stack([x, y, np.full_like(x, T / 2 + 0.0004)], -1).reshape(-1, 3).astype(np.float32)
    idx = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    faces = np.concatenate([np.stack([a, c, b], 1), np.stack([a, d, c], 1)]).astype(np.int32)  # counter-clockwise from above
    # Remove photo triangles over the open bay, so the inserted battery is visible below the lid.
    triangle_centres = pts[faces].mean(axis=1)
    slot_x, slot_y = EMPTY_SLOT_CENTER
    slot_l, slot_w = EMPTY_SLOT_SIZE
    in_slot = ((np.abs(triangle_centres[:, 0] - slot_x) < slot_l / 2)
               & (np.abs(triangle_centres[:, 1] - slot_y) < slot_w / 2))
    faces = faces[~in_slot]
    decal = UsdGeom.Mesh.Define(stage, f"{prim_path}/top_decal")
    decal.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    decal.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    decal.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
    decal.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    decal.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(decal).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec3fArray.FromNumpy(colors))
    mat_path = f"{prim_path}/top_decal_mat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, f"{mat_path}/vertex_color")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.CreateOutput("result", Sdf.ValueTypeNames.Float3))
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(decal.GetPrim()).Bind(mat)
    return prim


@configclass
class ChargerCfg(sim_utils.CuboidCfg):
    """The charging station: a cuboid plus its top-face photo."""

    func: Callable = sim_utils.clone(spawn_charger)
    top_texture: str | None = None
    tone_path: str | None = None


def charger_cfg(look_dir: str, *, mass: float = CHARGER_MASS, friction: float = 1.0) -> ChargerCfg:
    return ChargerCfg(
        size=CHARGER_SIZE,
        top_texture=os.path.join(look_dir, "charger_top.png"),
        tone_path=os.path.join(look_dir, "tone.json"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, max_depenetration_velocity=1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=0.9 * friction),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.66, 0.64, 0.60), roughness=0.5),
        semantic_tags=[("class", "charger")],
    )


def battery_cfg(*, mass: float = BATTERY_MASS, friction: float = 1.0) -> sim_utils.CylinderCfg:
    return sim_utils.CylinderCfg(
        radius=BATTERY_RADIUS,
        height=BATTERY_LENGTH,
        axis="X",  # lies along local x; the task's yaw event turns it on the mat
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=32, max_depenetration_velocity=1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=0.9 * friction),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.85, 0.90), roughness=0.35),
        semantic_tags=[("class", "battery")],
    )
