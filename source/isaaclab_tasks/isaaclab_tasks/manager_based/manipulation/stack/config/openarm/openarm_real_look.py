# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Apply a dataset-fitted visual profile to an OpenArm task scene.

Each profile folder identifies its source dataset and compatible tasks in ``profile.json``. The
Pringles and battery folders both use this renderer. Task geometry, reset events and success
conditions remain in their task cfgs. See scripts/tools/sysid/README.md for the calibration
pipeline that produced the optional files in each folder:

* ``albedo.json``     -- colour / roughness / metallic of every robot part (one material on one link), fitted
  by comparing the sim robot posed at real joint states against the real robot in the body camera.
* ``backdrop_*.npz``  -- the static real scene (table, mats, room) as vertex-coloured meshes in the env
  frame, built from the cameras' clean plates and monocular depth. They are emissive: their colours are
  the real pixels, already lit by the real room, so sim lighting must not shade them again.
* ``lighting.json``   -- intensity / colour of the dome light that lights dynamic scene objects (the backdrop
  is emissive and ignores it), fitted so the robot's shaded parts land on the real pixel values.
* ``contact_physics.json`` -- optional task-object friction / mass and gripper values fitted by replay;
  applied at run time by :func:`apply_contact_physics` after the env is built.
* ``sensor.json``     -- per-camera blur / noise (the real D435i frames, after its lens, ISP and the dataset's video
  encoding, are softer than the renderer's), applied by :class:`SensorModel` to each frame the server sends.
* ``tone.json``       -- per-channel curve from a wanted output pixel value to the emissive value that
  produces it through the RTX tone mapper (measured, not assumed).

:func:`apply_real_look` patches an env cfg in place, the same way ``apply_task_mode`` does, so every entry
point (the LeRobot sim server, the render/calibration tools) gets the identical scene. The physics is not
touched: the task's objects and hidden background surfaces still collide as configured; only what the cameras see changes.
"""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Callable
from dataclasses import MISSING

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from .real_look_profile import load_profile

LOOK_DIR = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "real2sim", "pringles_v00")
"""Legacy default for Pringles calibration scripts; the server selects a task profile explicitly."""


def _set_input(shader_prim, name: str, value, type_name) -> None:
    attr = shader_prim.GetAttribute(f"inputs:{name}")
    if not attr:
        attr = shader_prim.CreateAttribute(f"inputs:{name}", type_name)
    attr.Set(value)


# ── Robot materials ──────────────────────────────────────────────────────────


def recolor_robot(prim_path: str, albedo: dict[str, dict]) -> int:
    """Rewrite the robot's OmniPBR inputs from ``albedo`` ({"<link>/<material>": {diffuse, roughness, metallic}}).

    The asset's visuals are instanceable, i.e. their materials live in read-only prototypes, so the visual
    prims are de-instanced first (visual meshes only -- physics never reads them). Returns the number of
    materials changed.
    """
    from pxr import Gf, Sdf, Usd, UsdShade

    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    for prim in Usd.PrimRange(root):
        if prim.IsInstanceable() and "visuals" in prim.GetName():
            prim.SetInstanceable(False)
    changed = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdShade.Material):
            continue
        link = prim.GetPath().pathString[len(prim_path):].split("/")[1]
        look = albedo.get(f"{link}/{prim.GetName()}")
        if look is None:
            continue
        for shader in prim.GetChildren():
            if not UsdShade.Shader(shader):
                continue
            _set_input(shader, "diffuse_color_constant", Gf.Vec3f(*look["diffuse"]), Sdf.ValueTypeNames.Color3f)
            _set_input(shader, "reflection_roughness_constant", float(look.get("roughness", 0.5)), Sdf.ValueTypeNames.Float)
            _set_input(shader, "metallic_constant", float(look.get("metallic", 0.0)), Sdf.ValueTypeNames.Float)
            changed += 1
    return changed


def _robot_spawner(albedo: dict[str, dict]):
    """spawn_from_usd that recolours the robot before it is cloned to the other envs."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd

    base = spawn_from_usd.__wrapped__  # undecorated: spawns the one source prim

    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        prim = base(prim_path, cfg, translation, orientation, **kwargs)
        n = recolor_robot(str(prim.GetPath()), albedo)
        print(f"[openarm_real_look] robot: {n} materials set from the real-look albedo table")
        return prim

    return sim_utils.clone(spawn)


# ── Backdrop meshes ──────────────────────────────────────────────────────────


def emissive_from_srgb(srgb: np.ndarray, tone: dict | None) -> np.ndarray:
    """Emissive (linear) values that render as the given sRGB uint8 colours, via the measured tone curve."""
    x = np.asarray(srgb, np.float64) / 255.0
    if tone is None:  # plain sRGB decode; ~right in shape, wrong in level until tone.json is measured
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    out = np.empty_like(x)
    for c in range(3):  # tone["output"] is increasing in tone["emissive"]; invert by interpolation
        out[..., c] = np.interp(x[..., c] * 255.0, tone["output"][c], tone["emissive"])
    return out


def spawn_backdrop_mesh(prim_path: str, cfg: "BackdropMeshCfg", translation=None, orientation=None, **kwargs):
    """A vertex-coloured, emissive, non-shadow-casting mesh from an npz (points, faces, colors sRGB uint8)."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    stage = sim_utils.get_current_stage()
    data = np.load(cfg.npz_path)
    tone = json.load(open(cfg.tone_path)) if cfg.tone_path and os.path.isfile(cfg.tone_path) else None
    xform = UsdGeom.Xform.Define(stage, prim_path)
    # The scene setup writes the asset's init_state through the canonical translate/orient/scale ops.
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
    xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(data["points"].astype(np.float32)))
    faces = data["faces"].astype(np.int32)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    emissive = (emissive_from_srgb(data["colors"], tone) * cfg.gain).astype(np.float32)
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec3fArray.FromNumpy(emissive))
    # The real pixels already carry the real room's light and shadow: render them as light, not as paint.
    mat_path = f"{prim_path}/emissive"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, f"{mat_path}/vertex_color")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3))
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    mesh.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    return xform.GetPrim()


@configclass
class BackdropMeshCfg(sim_utils.SpawnerCfg):
    """A static, photo-coloured mesh of the real scene (see :func:`spawn_backdrop_mesh`)."""

    func: Callable = sim_utils.clone(spawn_backdrop_mesh)
    npz_path: str = MISSING
    tone_path: str | None = None
    gain: float = 1.0


class SensorModel:
    """Soften and add noise to rendered frames the way the real camera's pipeline does (per camera, from sensor.json)."""

    def __init__(self, look_dir: str):
        path = os.path.join(look_dir, "sensor.json")
        self.params = json.load(open(path)) if os.path.isfile(path) else {}
        self._kernels = {}

    def _kernel(self, sigma: float, device):
        key = (sigma, str(device))
        if key not in self._kernels:
            import torch

            r = max(1, int(np.ceil(3 * sigma)))
            x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
            k = torch.exp(-0.5 * (x / sigma) ** 2)
            self._kernels[key] = (k / k.sum(), r)
        return self._kernels[key]

    def __call__(self, camera: str, image):
        """image: (H, W, 3) uint8 torch tensor -> same, blurred and noised."""
        import torch
        import torch.nn.functional as F

        p = self.params.get(camera)
        if not p:
            return image
        x = image[..., :3].permute(2, 0, 1)[None].float()
        if p.get("blur_sigma", 0) > 0.05:
            k, r = self._kernel(p["blur_sigma"], x.device)
            x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
            x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)
        if p.get("noise_sigma", 0) > 0:
            x = x + torch.randn_like(x) * p["noise_sigma"]
        return x.clamp(0, 255).round().to(torch.uint8)[0].permute(1, 2, 0)


def apply_real_look(env_cfg, look_dir: str) -> list[str]:
    """Patch a task cfg with its dataset-fitted visual profile.

    Returns a list of what was applied, for the caller to print. Missing files are skipped, so a partially
    fitted look still works.
    """
    profile = load_profile(look_dir, getattr(env_cfg, "env_name", None))
    applied = []
    albedo_path = os.path.join(look_dir, "albedo.json")
    if os.path.isfile(albedo_path):
        albedo = json.load(open(albedo_path))
        env_cfg.scene.robot.spawn.func = _robot_spawner(albedo)
        applied.append(f"robot colours ({len(albedo)} parts)")

    lighting_path = os.path.join(look_dir, "lighting.json")
    if os.path.isfile(lighting_path):
        lighting = json.load(open(lighting_path))
        env_cfg.scene.light.spawn.intensity = float(lighting["intensity"])
        env_cfg.scene.light.spawn.color = tuple(float(c) for c in lighting["color"])
        applied.append(f"dome light {lighting['intensity']:.0f} {tuple(lighting['color'])}")

    backdrops = sorted(glob.glob(os.path.join(look_dir, "backdrop_*.npz")))
    if backdrops:
        tone = os.path.join(look_dir, "tone.json")
        for path in backdrops:
            name = os.path.splitext(os.path.basename(path))[0]
            setattr(env_cfg.scene, f"real_{name}", AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Real_{name}",
                spawn=BackdropMeshCfg(npz_path=path, tone_path=tone),
            ))
        # Hide only the static surfaces named by this profile. Their colliders remain active.
        for name in profile.get("hide_visuals", []):
            entity = getattr(env_cfg.scene, name, None)
            if entity is not None:
                entity.spawn.visible = False
        applied.append(f"real backdrop ({', '.join(os.path.basename(p) for p in backdrops)});"
                       f" hidden visuals {profile.get('hide_visuals', [])}")
    return applied


# ── Contact physics ─────────────────────────────────────────────────────────


CONTACT_DEFAULTS = {"can_friction": None, "finger_friction": None, "can_mass": None,
                    "grip_stiffness": None, "grip_damping": None, "grip_effort": None}
"""Parameter names :func:`apply_contact_physics` understands; None / missing leaves the cfg's value alone."""


def apply_contact_physics(env, params: dict, object_asset: str = "can") -> dict:
    """Set contact physics on a built env for the profile's manipulated scene object.

    ``object_friction`` / ``object_mass`` work for any object. The older ``can_*`` names remain
    accepted for the Pringles fitter and its saved profile. Multi-object tasks can call this once
    per fitted object; shared gripper parameters need only be supplied in one call.

    ``object_friction`` / ``finger_friction`` set static and dynamic friction (dynamic = 0.9 x static)
    on every shape of the object / robot; PhysX combines the touching surfaces' values.
    """
    import torch

    scene = env.scene
    obj, robot = scene[object_asset], scene["robot"]
    out = {}

    def set_friction(asset, mu):
        view = asset.root_physx_view
        mats = view.get_material_properties().clone()
        mats[..., 0], mats[..., 1] = mu, 0.9 * mu
        view.set_material_properties(mats, torch.arange(mats.shape[0], dtype=torch.int32))

    object_friction = params.get("object_friction", params.get("can_friction"))
    object_mass = params.get("object_mass", params.get("can_mass"))
    if object_friction is not None:
        set_friction(obj, float(object_friction))
    if params.get("finger_friction") is not None:
        set_friction(robot, float(params["finger_friction"]))
    if object_mass is not None:
        view = obj.root_physx_view
        masses = view.get_masses().clone()
        masses[:] = float(object_mass)
        view.set_masses(masses, torch.arange(masses.shape[0], dtype=torch.int32))
    ids = robot.find_joints(["openarm_.*_finger_joint.*"])[0]
    gripper = robot.actuators["openarm_gripper"]
    if params.get("grip_stiffness") is not None:
        robot.write_joint_stiffness_to_sim(float(params["grip_stiffness"]), joint_ids=ids)
        gripper.stiffness[:] = float(params["grip_stiffness"])
    if params.get("grip_damping") is not None:
        robot.write_joint_damping_to_sim(float(params["grip_damping"]), joint_ids=ids)
        gripper.damping[:] = float(params["grip_damping"])
    if params.get("grip_effort") is not None:
        robot.write_joint_effort_limit_to_sim(float(params["grip_effort"]), joint_ids=ids)
    prefix = "can" if object_asset == "can" else "object"
    out[f"{prefix}_friction"] = float(obj.root_physx_view.get_material_properties()[..., 0].mean())
    out["finger_friction"] = float(robot.root_physx_view.get_material_properties()[..., 0].mean())
    out[f"{prefix}_mass"] = float(obj.root_physx_view.get_masses().mean())
    out["grip_stiffness"] = float(robot.data.joint_stiffness[0, ids].mean())
    out["grip_damping"] = float(robot.data.joint_damping[0, ids].mean())
    return out
