# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Make the OpenArm scene look like the real rig the pringles dataset was recorded on.

Everything here is fitted from the real recordings (ethanCSL/openarm_pringles_lerobot_real_v00), not drawn
by hand -- see scripts/tools/sysid/README.md, "Real look", for the pipeline that produced each file in
``LOOK_DIR``:

* ``albedo.json``     -- colour / roughness / metallic of every robot part (one material on one link), fitted
  by comparing the sim robot posed at real joint states against the real robot in the body camera.
* ``backdrop_*.npz``  -- the static real scene as vertex-coloured meshes in the env frame: ``backdrop_body`` (the table
  and mats, from body_cam's clean plate and monocular depth) and ``backdrop_room`` (the walls around the rig, fused
  from thousands of real frames of all three cameras by build_room_backdrop.py). They are emissive: their colours are
  the real pixels, already lit by the real room, so sim lighting must not shade them again.
* ``lighting.json``   -- intensity / colour of the dome light that lights the robot and the can (the backdrop
  is emissive and ignores it), fitted so the robot's shaded parts land on the real pixel values.
* ``contact_physics.json`` -- friction / mass / gripper-squeeze values fitted by replaying the real episodes in sim until the
  can stays in the gripper as it does in the real recordings (scripts/tools/sysid/fit_contact_physics.py);
  applied by :func:`apply_contact_physics` as a startup event once the env is built.
* ``sensor.json``     -- per-camera blur / noise (the real D435i frames, after its lens, ISP and the dataset's video
  encoding, are softer than the renderer's), applied by :class:`SensorModel` inside the camera observations, so
  the server, Mimic generation and every other consumer of the observations get the same frames.
* ``tone.json``       -- per-channel curve from a wanted output pixel value to the emissive value that
  produces it through the RTX tone mapper (measured, not assumed).

:func:`apply_real_look` patches an env cfg in place, the same way ``apply_task_mode`` does, so every entry
point (the LeRobot sim server, Mimic, the render/calibration tools) gets the identical scene. Collision geometry is not
touched: the pad, table and can collide exactly as before. Domain randomization on top of the look:
:mod:`openarm_real_look_randomization`.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections.abc import Callable
from dataclasses import MISSING

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

LOOK_DIR = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "real2sim", "pringles_v00")
"""Fitted look of the real rig. Swap the folder (``look_dir``) to target another recording setup."""


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


FINGER_TIP_Z = 0.080
"""m, where the sim finger meshes end in their own link frame (measured from the mesh; the frame sits 0.1025 m along the tool axis)."""
FINGER_INNER_Y = {"left_finger": -0.0078, "right_finger": 0.0078}
"""m, the stock finger collision meshes' inner faces in their link frames (finger.stl spans y -0.0078 .. +0.0271 for the left
finger, its mirror for the right one). With the official finger joint offsets (add_openarm_hand.py, 5 mm a side since
2026-10-21) the jaw gap is 2 q - 0.56 cm; before that, with both joints at the hand's centre line, it was 2 q - 1.56 cm."""
PAD_WIDTH, PAD_THICKNESS, PAD_OUTSET = 0.022, 0.016, 0.007
"""m: the foam fingertip pad's width (along the finger's x), its thickness (along its y, outwards from its inner face), and how
far that inner face sits outside the stock finger's. The stock jaws open to 7.2 cm, 3.6 mm a side more than the 6.5 cm charger
they must straddle; the pads open to 7.8 cm, and still close on the 1.45 cm cell (the closed gap is 2 q - 1.56 + 0.6 cm). Those are
the figures the battery expert was tuned with; the official finger offsets (2026-10-21) add 1 cm to every gap."""
PAD_COLOR = (0.035, 0.035, 0.035)


def extend_fingers(prim_path: str, length: float) -> int:
    """Lengthen the four gripper fingers by ``length`` metres along the tool axis with a foam-padded fingertip: the battery
    recording's real pads reach about 7 cm further than the stock finger (the right-wrist camera sees them as big dark
    padded jaws closing on the cell).

    One box per finger, collider and visual both: ``PAD_WIDTH`` wide, ``PAD_THICKNESS`` thick, its inner face ``PAD_OUTSET`` outside
    the stock finger's, continuing from the stock tip. The stock finger meshes stay (they are what the hand looks like) but stop
    colliding: they are 6 cm wide plates that leave 3.6 mm either side of the 6.5 cm charger, so a hand yawed a few degrees off
    the charger lands on it, whereas the real jaws are the 2 cm pads.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = sim_utils.get_current_stage()
    added = 0
    for side in ("left", "right"):
        for finger, inner in FINGER_INNER_Y.items():
            link = stage.GetPrimAtPath(f"{prim_path}/openarm_{side}_{finger}")
            if not link.IsValid():
                continue
            for prim in Usd.PrimRange(link):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
            pad = UsdGeom.Cube.Define(stage, f"{link.GetPath()}/tip_extension_pad")
            pad.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(pad)
            face = inner - math.copysign(PAD_OUTSET, inner)  # the pad's inner face, further from the centre than the finger's
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, face - math.copysign(PAD_THICKNESS / 2, inner), FINGER_TIP_Z + length / 2))
            xf.AddScaleOp().Set(Gf.Vec3d(PAD_WIDTH, PAD_THICKNESS, length))
            UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
            UsdPhysics.MassAPI.Apply(pad.GetPrim()).CreateMassAttr(0.004)
            pad.CreateDisplayColorAttr([Gf.Vec3f(*PAD_COLOR)])
            added += 1
    return added


def finger_extension(look_dir: str) -> float:
    """Fingertip extension in metres: OPENARM_FINGER_EXTENSION if set, else ``gripper.json`` in the look folder, else 0."""
    if os.environ.get("OPENARM_FINGER_EXTENSION"):
        return float(os.environ["OPENARM_FINGER_EXTENSION"])
    path = os.path.join(look_dir, "gripper.json")
    return float(json.load(open(path)).get("finger_extension", 0.0)) if os.path.isfile(path) else 0.0


def _robot_spawner(albedo: dict[str, dict] | None, extension: float = 0.0):
    """spawn_from_usd that recolours the robot before it is cloned to the other envs."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd

    base = spawn_from_usd.__wrapped__  # undecorated: spawns the one source prim

    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        prim = base(prim_path, cfg, translation, orientation, **kwargs)
        if albedo:
            n = recolor_robot(str(prim.GetPath()), albedo)
            print(f"[openarm_real_look] robot: {n} materials set from the real-look albedo table")
        if extension > 0:
            print(f"[openarm_real_look] robot: {extend_fingers(str(prim.GetPath()), extension)} fingers lengthened by {extension * 100:.1f} cm")
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
    role: str = "scene"
    """"table" (moves with the pad's height, tinted with the pad colour), "room" (hidden by the skybox swap), or "scene"."""

    @property
    def follows_pad(self) -> bool:
        return self.role == "table"


def _with_sensor(image_func: Callable, sensor: "SensorModel", camera: str) -> Callable:
    """An image observation term that returns the frames the real camera would: ``image_func``'s, through ``sensor``."""
    import functools

    import torch

    @functools.wraps(image_func)  # the observation manager checks the term's parameters against its signature
    def term(env, **kwargs):
        images = image_func(env, **kwargs)
        if images.dtype != torch.uint8:  # normalized float images are left as they are
            return images
        out = images.clone()
        for i in range(images.shape[0]):
            out[i, ..., :3] = sensor(camera, images[i])
        return out

    return term


def _contact_physics_startup(env, env_ids, params: dict) -> None:
    print(f"[openarm_real_look] contact physics fitted to the real episodes: {apply_contact_physics(env, params)}")


class SensorModel:
    """Soften and add noise to rendered frames the way the real camera's pipeline does (per camera, from sensor.json)."""

    def __init__(self, look_dir: str = LOOK_DIR):
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


def apply_real_look(env_cfg, look_dir: str = LOOK_DIR) -> list[str]:
    """Patch ``env_cfg`` (an OpenArm pick-up cfg, after apply_task_mode) to the fitted real look.

    Returns a list of what was applied, for the caller to print. Missing files are skipped, so a partially
    fitted look still works.
    """
    from isaaclab.managers import EventTermCfg

    applied = []
    env_cfg.real_look_dir = look_dir  # consumers (DR, the server) read it to know the look is on
    albedo_path = os.path.join(look_dir, "albedo.json")
    albedo = json.load(open(albedo_path)) if os.path.isfile(albedo_path) else None
    extension = finger_extension(look_dir)
    if albedo or extension > 0:
        env_cfg.scene.robot.spawn.func = _robot_spawner(albedo, extension)
        if albedo:
            applied.append(f"robot colours ({len(albedo)} parts)")
        if extension > 0:
            applied.append(f"fingers {extension * 100:.1f} cm longer")

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
            role = {"backdrop_body": "table", "backdrop_room": "room"}.get(name, "scene")
            setattr(env_cfg.scene, f"real_{name}", AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Real_{name}",
                spawn=BackdropMeshCfg(npz_path=path, tone_path=tone, role=role),
            ))
        # The real scene replaces the sim's own table, pad and floor -- visually only: the pad keeps its
        # collider (an invisible prim still collides), so the can rests and is grasped exactly as before.
        for name in ("table", "workspace_pad", "plane"):
            entity = getattr(env_cfg.scene, name, None)
            if entity is not None:
                entity.spawn.visible = False
        applied.append(f"real backdrop ({', '.join(os.path.basename(p) for p in backdrops)}); sim table/pad/floor hidden")

    contact_path = os.path.join(look_dir, "contact_physics.json")
    if os.path.isfile(contact_path) and getattr(env_cfg, "events", None) is not None:
        params = json.load(open(contact_path))["params"]
        env_cfg.events.real_look_contact_physics = EventTermCfg(
            func=_contact_physics_startup, mode="startup", params={"params": params})
        applied.append(f"contact physics ({', '.join(f'{k} {v:g}' for k, v in params.items())})")

    sensor = SensorModel(look_dir)
    policy = getattr(getattr(env_cfg, "observations", None), "policy", None)
    wrapped = []
    for camera in sensor.params:
        term = getattr(policy, camera, None)
        if term is not None and not getattr(term.func, "_real_look_sensor", False):
            term.func = _with_sensor(term.func, sensor, camera)
            term.func._real_look_sensor = True
            wrapped.append(camera)
    if wrapped:
        env_cfg.real_look_sensor_in_obs = True
        applied.append(f"camera blur/noise in the observations ({', '.join(wrapped)})")
    return applied


# ── Contact physics ─────────────────────────────────────────────────────────


CONTACT_DEFAULTS = {"can_friction": None, "finger_friction": None, "can_mass": None,
                    "grip_stiffness": None, "grip_damping": None, "grip_effort": None}
"""Parameter names :func:`apply_contact_physics` understands; None / missing leaves the cfg's value alone."""


def apply_contact_physics(env, params: dict) -> dict:
    """Set contact-related physics on a built env (all envs): friction of the can and the robot's shapes, the can's mass,
    and the gripper's PD gains and effort limit. Returns the values read back, for the caller to log.

    ``can_friction`` / ``finger_friction`` set static and dynamic friction (dynamic = 0.9 x static) on every shape of the
    can / robot -- only the fingers ever touch the can, and PhysX averages the two surfaces' values.
    """
    import torch

    scene = env.scene
    robot = scene["robot"]
    props = {name: scene[name] for name in ("can", "charger", "battery") if name in scene.rigid_objects}
    out = {}

    def set_friction(asset, mu):
        view = asset.root_physx_view
        mats = view.get_material_properties().clone()
        mats[..., 0], mats[..., 1] = mu, 0.9 * mu
        view.set_material_properties(mats, torch.arange(mats.shape[0], dtype=torch.int32))

    if params.get("can_friction") is not None and "can" in props:
        set_friction(props["can"], float(params["can_friction"]))
    if params.get("object_friction") is not None:  # every prop the task has (charger and battery, say)
        for asset in props.values():
            set_friction(asset, float(params["object_friction"]))
    if params.get("finger_friction") is not None:
        set_friction(robot, float(params["finger_friction"]))
    if params.get("can_mass") is not None and "can" in props:
        view = props["can"].root_physx_view
        masses = view.get_masses().clone()
        masses[:] = float(params["can_mass"])
        view.set_masses(masses, torch.arange(masses.shape[0], dtype=torch.int32))
    if params.get("arm_effort_scale") is not None:  # the shoulder / elbow joints saturate holding the arm out at a table grasp
        arm_ids = robot.find_joints(["openarm_.*_joint[1-4]"])[0]
        scaled = robot.data.joint_effort_limits[:, arm_ids] * float(params["arm_effort_scale"])
        robot.write_joint_effort_limit_to_sim(scaled, joint_ids=arm_ids)
        out["arm_effort_scale"] = float(params["arm_effort_scale"])
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
    for name, asset in props.items():
        out[f"{name}_friction"] = float(asset.root_physx_view.get_material_properties()[..., 0].mean())
        out[f"{name}_mass"] = float(asset.root_physx_view.get_masses().mean())
    out["finger_friction"] = float(robot.root_physx_view.get_material_properties()[..., 0].mean())
    out["grip_stiffness"] = float(robot.data.joint_stiffness[0, ids].mean())
    out["grip_damping"] = float(robot.data.joint_damping[0, ids].mean())
    return out
