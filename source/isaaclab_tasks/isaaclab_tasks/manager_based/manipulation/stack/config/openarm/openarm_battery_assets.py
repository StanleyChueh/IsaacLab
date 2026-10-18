# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The battery-insertion task's two props, built from the real recordings (ethanCSL/openarm_insertbattery_lerobot_real_v01).

* **Charger** -- the white "enerpad" charging station the left arm picks up: a rigid box of the footprint measured from the
  real body-camera frames (11.5 x 6.5 cm; ``prepare_battery_replay.py``) and the thickness at which the real left gripper's
  jaws stop when they hold it (~3.75 cm), with its top face textured by the rectified real photo
  (``build_charger_texture.py``): lid, three batteries in their slots, the empty slot, the LED.
* **Battery** -- one AA cell lying flat, 5.05 cm x 1.45 cm, 27 g.

Both are plain rigid bodies with box / cylinder colliders. The charger's slots are drawn, not modelled: nothing here lets
a battery drop into one. That needs the slot geometry to sub-millimetre accuracy, which a look cannot give (see the sysid
README, "Battery insertion").
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


def spawn_charger(prim_path: str, cfg: "ChargerCfg", translation=None, orientation=None, **kwargs):
    """A rigid cuboid with the real top-face photo on it, as a fine vertex-coloured, emissive, non-colliding grid.

    (A UsdUVTexture decal rendered black here, while vertex colours through an emissive UsdPreviewSurface are what the
    backdrop already uses, so the photo is baked into vertices: 0.5 mm apart, finer than the photo's own softness.)
    """
    import json

    from PIL import Image
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    from isaaclab.sim.spawners.shapes.shapes import spawn_cuboid

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
    if not (cfg.top_texture and os.path.isfile(cfg.top_texture)):
        return prim
    stage = sim_utils.get_current_stage()
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
