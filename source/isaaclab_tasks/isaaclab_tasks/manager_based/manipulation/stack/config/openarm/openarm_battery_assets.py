# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The battery-insertion task's two props, built from the real recordings (ethanCSL/openarm_insertbattery_lerobot_real_v01).

* **Charger** -- the white "enerpad" charging station the left arm picks up: a rigid box of the footprint measured from the
  real body-camera frames (11.5 x 6.5 cm; ``prepare_battery_replay.py``) and the thickness at which the real left gripper's
  jaws stop when they hold it (~3.75 cm), with its top face textured by the rectified real photo
  (``build_charger_texture.py``): lid, three batteries in their slots, the empty slot, the LED.
* **Battery** -- one AA cell lying flat, 5.05 cm x 1.45 cm, 27 g.

The charger is a rigid body of box colliders with ONE open slot (the empty bay in the photo): a trough a battery can be set
into, 1.8 x 5.5 cm and 1.2 cm deep against the 1.45 x 5.05 cm cell. Its size and place are read off the photo (about a
millimetre) and then relaxed by a millimetre or so on every side: the real bay is a spring-loaded cradle, which a rigid
box cannot imitate. The other three bays are only painted. ``battery_in_charger`` is the task's success test.
"""

from __future__ import annotations

import math
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
SLOT_X = (-0.0415, 0.0135)
SLOT_Y = (0.0122, 0.0302)
SLOT_DEPTH = 0.012
"""The empty bay, in the charger's frame (origin at the box centre, x along its length, +y the side of the empty bay): its
x and y extents and its depth below the top face. Photo: bay y 1.5 .. 3.0 cm, x -3.8 .. +0.9 cm (80 px/cm)."""
SLOT_FLOOR_COLOR = (0.30, 0.24, 0.18)


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
    # decal's own, and puts a solid collider on the box: spawn it with neither, hide its mesh, and build the body out of
    # box parts (colliders and visuals both) that leave the bay open.
    body_material, cfg.visual_material = cfg.visual_material, None
    collision, cfg.collision_props = cfg.collision_props, None
    try:
        prim = spawn_cuboid.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)
    finally:
        cfg.visual_material, cfg.collision_props = body_material, collision
    stage = sim_utils.get_current_stage()
    UsdGeom.Imageable(stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")).MakeInvisible()
    material_path = f"{prim_path}/geometry/body_material"
    if body_material is not None:
        body_material.func(material_path, body_material)
    physics_path = f"{prim_path}/geometry/{cfg.physics_material_path}" if not cfg.physics_material_path.startswith("/") else cfg.physics_material_path
    L, W, T = cfg.size
    sx, sy = cfg.slot_x, cfg.slot_y
    parts = {  # name: (x0, x1, y0, y1)
        "base": (-L / 2, L / 2, -W / 2, W / 2),
        "wall_ypos": (-L / 2, L / 2, sy[1], W / 2),
        "wall_yneg": (-L / 2, L / 2, -W / 2, sy[0]),
        "wall_xpos": (sx[1], L / 2, sy[0], sy[1]),
        "wall_xneg": (-L / 2, sx[0], sy[0], sy[1]),
    }
    for name, (x0, x1, y0, y1) in parts.items():
        z0, z1 = (-T / 2, T / 2 - cfg.slot_depth) if name == "base" else (-T / 2, T / 2)
        part = UsdGeom.Cube.Define(stage, f"{prim_path}/geometry/{name}")
        part.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(part)
        xf.AddTranslateOp().Set(Gf.Vec3d((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
        xf.AddScaleOp().Set(Gf.Vec3d(x1 - x0, y1 - y0, z1 - z0))
        if collision is not None:
            sim_utils.schemas.define_collision_properties(str(part.GetPath()), collision)
            sim_utils.bind_physics_material(str(part.GetPath()), physics_path)
        if body_material is not None:
            sim_utils.bind_visual_material(str(part.GetPath()), material_path, stronger_than_descendants=False)
    # a lead-in collar round the bay's mouth: two 45 degree slabs leaning outwards, so a cell set down a few millimetres off (or a few degrees
    # round) slides in and turns into line instead of landing on the rim. The real cradle is spring loaded; a rigid box needs this.
    c, t = cfg.collar, 0.003
    for name, edge, sign in (("collar_ypos", sy[1], 1.0), ("collar_yneg", sy[0], -1.0)):
        slab = UsdGeom.Cube.Define(stage, f"{prim_path}/geometry/{name}")
        slab.CreateSizeAttr(1.0)
        n = np.array([sign, -1.0]) / math.sqrt(2)  # thickness direction in (y, z): away from the bay, below the ramp
        mid = np.array([edge + sign * c / 2, T / 2 + c / 2]) + n * t / 2
        xf = UsdGeom.Xformable(slab)
        xf.AddTranslateOp().Set(Gf.Vec3d((sx[0] + sx[1]) / 2, mid[0], mid[1]))
        xf.AddRotateXOp().Set(-45.0 * sign)
        xf.AddScaleOp().Set(Gf.Vec3d(sx[1] - sx[0], t, c * math.sqrt(2)))
        if collision is not None:
            sim_utils.schemas.define_collision_properties(str(slab.GetPath()), collision)
            sim_utils.bind_physics_material(str(slab.GetPath()), physics_path)
        if body_material is not None:
            sim_utils.bind_visual_material(str(slab.GetPath()), material_path, stronger_than_descendants=False)
    floor = UsdGeom.Cube.Define(stage, f"{prim_path}/geometry/slot_floor")  # the bay's dark lining, seen where the photo is cut away
    floor.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(floor)
    xf.AddTranslateOp().Set(Gf.Vec3d((sx[0] + sx[1]) / 2, (sy[0] + sy[1]) / 2, T / 2 - cfg.slot_depth + 0.0003))
    xf.AddScaleOp().Set(Gf.Vec3d(sx[1] - sx[0], sy[1] - sy[0], 0.0006))
    floor.CreateDisplayColorAttr([Gf.Vec3f(*SLOT_FLOOR_COLOR)])
    if not (cfg.top_texture and os.path.isfile(cfg.top_texture)):
        return prim
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
    cx, cy = (x[:-1, :-1] + x[1:, 1:]) / 2, (y[:-1, :-1] + y[1:, 1:]) / 2  # quad centres
    keep = ~((cx > sx[0]) & (cx < sx[1]) & (cy > sy[0]) & (cy < sy[1])).ravel()  # the bay is open: no photo over it
    a, b, c, d = (v.ravel()[keep] for v in (idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]))
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
    slot_x: tuple = SLOT_X
    slot_y: tuple = SLOT_Y
    slot_depth: float = SLOT_DEPTH
    collar: float = 0.005
    """m, how far the lead-in collar leans out of the bay's mouth (and how high it stands over the top face)."""


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
        # a cylinder on a plane has no rolling resistance in PhysX: left alone this one creeps across the mat at about a centimetre a
        # second, so angular and linear damping stand in for the little a real cell's contact would give
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=32, max_depenetration_velocity=1.0,
                                                     linear_damping=2.0, angular_damping=20.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=0.9 * friction),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.85, 0.90), roughness=0.35),
        semantic_tags=[("class", "battery")],
    )


def battery_in_charger(env, tol_xy: float = 0.007, tol_axis_deg: float = 10.0):
    """True per env while the battery sits in the charger's open bay: its centre over the bay (in the charger's frame), BELOW
    the top face (a cell resting on the rim is not in), and its axis along the bay's."""
    import torch

    from isaaclab.utils.math import quat_apply_inverse

    charger, battery = env.scene["charger"], env.scene["battery"]
    rel = quat_apply_inverse(charger.data.root_quat_w, battery.data.root_pos_w - charger.data.root_pos_w)
    axis_c = quat_apply_inverse(charger.data.root_quat_w, _rotate(battery.data.root_quat_w, torch.tensor([1.0, 0.0, 0.0], device=env.device)))
    cx, cy = (SLOT_X[0] + SLOT_X[1]) / 2, (SLOT_Y[0] + SLOT_Y[1]) / 2
    seated_z = CHARGER_SIZE[2] / 2 - SLOT_DEPTH + BATTERY_RADIUS
    in_xy = ((rel[:, 0] - cx).abs() < 0.012) & ((rel[:, 1] - cy).abs() < tol_xy)
    low = rel[:, 2] < seated_z + 0.004
    aligned = axis_c[:, 0].abs() > math.cos(math.radians(tol_axis_deg))
    return in_xy & low & aligned


def _rotate(q, v):
    from isaaclab.utils.math import quat_apply

    return quat_apply(q, v.expand(q.shape[0], 3))
