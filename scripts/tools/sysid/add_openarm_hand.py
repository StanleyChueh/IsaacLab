# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Add the OpenArm parallel gripper's hand (the linear-rail housing the fingers slide on) to v1_camera_isaac.usd.

The asset was imported from urdf/v1_camera.urdf, which has no hand link: the fingers hang straight off link7, so
the gripper renders hollow between the wrist and the fingers and nothing there collides. The official model
(mesh/end_effector/parallel_link: urdf/parallel_link_standalone.urdf.xacro + config/kinematics*.yaml) fixes the
hand to link7 at z = 0.1025 and draws hand.dae / hand.stl at z = -0.661001 in that frame. The OpenArm meshes share
one global frame, so that is the same -0.5585 offset link7's own meshes use: the plate sits on link7's top face.

This writes
  configuration/v1_camera_isaac_hand_mesh.usd   hand.dae converted (geometry + its two materials)
  configuration/v1_camera_isaac_hand.usda       on both link7s: the hand visual, and hand.stl as a collider
                                                (convex decomposition: the rail plate is far from convex)
and adds the .usda as a sublayer of v1_camera_isaac.usd (the original is kept as v1_camera_isaac.pre-hand.usd.bak).
The finger joints are put where the official model has them too: the import left both at y = 0, where kinematics.yaml
puts them at -+0.005 (finger_joint y), i.e. each finger 5 mm too far in -- the closed fingers passed 11.6 mm through each
other and the open jaws spanned 72.4 mm instead of the real 82.4 mm, 0.8 mm a side around the real 7 cm can's rim. The
override moves each joint's parent frame (localPos0) and the finger body's rest pose by the same 5 mm outwards, so the
joint frames stay consistent. With it the open finger carriages meet the rail's end stops as on the real gripper.

The real fingers also carry a black foam pad on each fingertip's inner face, and the pads are what the jaw width is
between: measured in the real recording's left wrist camera (open frames, pad inner edges vs the sim fingers' at the same
joint states), the real pads open to ~77.8 mm -- 8.6 mm less than the bare official fingers' 86.4. So each finger gets a
FOAM_PAD-thick pad over its tip's inner contact face (visual and collider: it is what touches the can), and the joints'
lower limit is set where the two pads meet, so a gripper closed on nothing stops pad to pad instead of the fingers
passing through each other (the stock range ran to 0, where the bare fingers overlap).
The hand goes on link7, the fingers' parent, so PhysX filters hand-finger contacts like any jointed pair. Link7's
authored mass is unchanged (the official hand weighs 0.127 kg; the fitted actuators were identified without it).

  ./isaaclab.sh -p scripts/tools/sysid/add_openarm_hand.py
"""

import argparse
import asyncio
import os
import shutil

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app = AppLauncher(args_cli).app

import numpy as np
import trimesh
import yaml
from scipy.spatial import ConvexHull
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

ASSET = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "v1_camera_isaac")
EE = os.path.join(ASSET, "mesh/end_effector/parallel_link/meshes")
MESH_USD = os.path.join(ASSET, "configuration/v1_camera_isaac_hand_mesh.usd")
LAYER = os.path.join(ASSET, "configuration/v1_camera_isaac_hand.usda")
ROOT = os.path.join(ASSET, "v1_camera_isaac.usd")
HAND_Z = -0.5585  # link7 frame -> mesh frame: official hand joint z 0.1025 + hand mesh z -0.661001
MM = 0.001
FINGER_JOINT_Y = abs(float(yaml.safe_load(open(os.path.join(EE, "../config/kinematics.yaml")))["finger_joint"]["kinematic"]["y"]))
"""m: the official finger joints' offset from the hand's centre line (kinematics.yaml: finger_joint y -0.005, mirrored)."""
FOAM_PAD = 0.0043
"""m: foam pad thickness on each fingertip's inner face, so the open jaws span the real ~77.8 mm (see the docstring)."""
FOAM_COLOR = (0.035, 0.035, 0.035)
FINGERS = {"finger_joint1": ("right_finger", -1.0), "finger_joint2": ("left_finger", 1.0)}
"""joint -> (finger link, side its finger slides to in link7's y): joint1 drives the -y finger, joint2 the +y one (urdf)."""


async def convert_visual() -> None:
    enable_extension("omni.kit.asset_converter")
    import omni.kit.asset_converter as converter

    ctx = converter.AssetConverterContext()
    ctx.ignore_materials = False
    ctx.ignore_camera = True
    ctx.ignore_light = True
    task = converter.get_instance().create_converter_task(os.path.join(EE, "visual/hand.dae"), MESH_USD, None, ctx)
    if not await task.wait_until_finished():
        raise SystemExit(f"hand.dae conversion failed: {task.get_status()} {task.get_error_message()}")


def tip_face(stage, finger_path: str) -> tuple[float, np.ndarray]:
    """The fingertip's inner contact face in the finger's frame: (its y, the (x, z) outline of the flat patch -- convex hull,
    counter-clockwise). The face narrows to a rounded tip, so a rectangle over it would stick out past the finger."""
    to_f = np.linalg.inv(np.array(UsdGeom.Xformable(stage.GetPrimAtPath(finger_path)).ComputeLocalToWorldTransform(0)))
    pts = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(f"{finger_path}/visuals"), Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh):
            v = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)) @ to_f
            pts.append((np.c_[v, np.ones(len(v))] @ m)[:, :3])
    pts = np.concatenate(pts)
    inner = pts[:, 1].min() if pts[:, 1].mean() > 0 else pts[:, 1].max()  # the face toward the other finger
    face = pts[np.abs(pts[:, 1] - inner) < 0.001][:, [0, 2]]
    hull = face[ConvexHull(face).vertices]
    return float(inner), hull


def finger_overrides() -> dict:
    """{prim path: {attribute: value}} moving each finger joint and finger body FINGER_JOINT_Y outwards, read off the
    asset as it is without this layer (so re-runs do not add the offset twice)."""
    stage = Usd.Stage.Open(ROOT)
    if os.path.exists(LAYER):
        stage.MuteLayer(Sdf.Layer.FindOrOpen(LAYER).identifier)
    out = {}
    for side in ("left", "right"):
        link7 = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(f"/openarm/openarm_{side}_link7")).ComputeLocalToWorldTransform(0))
        for joint_name, (finger, sign) in FINGERS.items():
            joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(f"/openarm/joints/openarm_{side}_{joint_name}"))
            pos0 = joint.GetLocalPos0Attr().Get()
            out[str(joint.GetPath())] = {"physics:localPos0": Gf.Vec3f(pos0[0], pos0[1] + sign * FINGER_JOINT_Y, pos0[2])}
            body = stage.GetPrimAtPath(f"/openarm/openarm_{side}_{finger}")
            translate = body.GetAttribute("xformOp:translate")
            shift = np.array([0.0, sign * FINGER_JOINT_Y, 0.0]) @ link7[:3, :3]  # link7's y in the fingers' parent frame
            out[str(body.GetPath())] = {"xformOp:translate": type(translate.Get())(*(np.array(translate.Get()) + shift))}
            # the pad over the tip's inner face, and the joint stops where the two pads meet
            inner, outline = tip_face(stage, str(body.GetPath()))
            out[f"{body.GetPath()}/foam_pad"] = {"pad": (inner, outline)}
            out[str(joint.GetPath())]["physics:lowerLimit"] = float(abs(inner) + FOAM_PAD - FINGER_JOINT_Y)
    return out


def write_layer() -> None:
    hand = trimesh.load(os.path.join(EE, "collision/hand.stl"), force="mesh")
    fingers = finger_overrides()
    # The converter reads the .dae's millimetre values as metres and writes them in centimetres (metersPerUnit
    # 0.01); a reference does not rescale units, so the visual's scale takes that factor too.
    visual_scale = MM * UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(MESH_USD))
    if os.path.exists(LAYER):
        os.remove(LAYER)
    layer = Sdf.Layer.CreateNew(LAYER)
    stage = Usd.Stage.Open(layer)
    stage.SetDefaultPrim(stage.OverridePrim("/openarm"))
    for side in ("left", "right"):
        link = f"/openarm/openarm_{side}_link7"
        stage.OverridePrim(link)
        for name, kind in (("hand_visual", "visual"), ("hand_collision", "collision")):
            xf = UsdGeom.Xform.Define(stage, f"{link}/{name}")
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, HAND_Z))
            s = visual_scale if kind == "visual" else MM
            xf.AddScaleOp().Set(Gf.Vec3f(s, s, s))
            if kind == "visual":
                xf.GetPrim().GetReferences().AddReference(os.path.relpath(MESH_USD, os.path.dirname(LAYER)))
                continue
            mesh = UsdGeom.Mesh.Define(stage, f"{link}/{name}/mesh")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(hand.vertices, dtype=np.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(hand.faces)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(hand.faces.reshape(-1).tolist()))
            mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)  # collision only, never rendered
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexDecomposition")
    foam = UsdShade.Material.Define(stage, "/openarm/Looks/foam_pad")
    shader = UsdShade.Shader.Define(stage, "/openarm/Looks/foam_pad/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*FOAM_COLOR))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.95)
    foam.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for path, attrs in fingers.items():
        if "pad" in attrs:
            inner, outline = attrs["pad"]
            toward = np.sign(inner)  # from the face toward the other finger (the face is on that side of the finger)
            # the face outline extruded FOAM_PAD toward the other finger: a convex prism
            n = len(outline)
            back = [Gf.Vec3f(x, inner, z) for x, z in outline]
            front = [Gf.Vec3f(x, inner + toward * FOAM_PAD, z) for x, z in outline]
            faces, counts = list(range(n)), [n, n]
            faces += list(range(2 * n - 1, n - 1, -1))
            for i in range(n):
                j = (i + 1) % n
                faces += [i, n + i, n + j, j]
                counts.append(4)
            pad = UsdGeom.Mesh.Define(stage, path)
            pad.CreatePointsAttr(Vt.Vec3fArray(back + front))
            pad.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
            pad.CreateFaceVertexIndicesAttr(Vt.IntArray(faces))
            pad.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            pad.CreateDoubleSidedAttr(True)
            pad.CreateDisplayColorAttr([Gf.Vec3f(*FOAM_COLOR)])
            UsdShade.MaterialBindingAPI.Apply(pad.GetPrim()).Bind(foam)
            UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(pad.GetPrim()).CreateApproximationAttr("convexHull")
            continue
        prim = stage.OverridePrim(path)
        for name, value in attrs.items():
            if name == "physics:lowerLimit":
                prim.CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=False).Set(value)
                continue
            type_name = Sdf.ValueTypeNames.Double3 if isinstance(value, Gf.Vec3d) else Sdf.ValueTypeNames.Float3
            prim.CreateAttribute(name, type_name, custom=False).Set(value)
    layer.Save()


def add_sublayer() -> None:
    backup = ROOT.replace(".usd", ".pre-hand.usd.bak")
    if not os.path.exists(backup):
        shutil.copy2(ROOT, backup)
    root = Sdf.Layer.FindOrOpen(ROOT)
    rel = os.path.relpath(LAYER, ASSET)
    if rel not in root.subLayerPaths:
        root.subLayerPaths.append(rel)
        root.Save()


asyncio.get_event_loop().run_until_complete(convert_visual())
write_layer()
add_sublayer()
stage = Usd.Stage.Open(ROOT)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide])
for side in ("left", "right"):
    for name in ("openarm_{}_link7", "openarm_{}_link7/hand_visual", "openarm_{}_link7/hand_collision"):
        path = "/openarm/" + name.format(side)
        box = cache.ComputeRelativeBound(stage.GetPrimAtPath(path), stage.GetPrimAtPath(f"/openarm/openarm_{side}_link7")).ComputeAlignedRange()
        print(f"[add_openarm_hand] {path}: link7-frame bounds {np.round(box.GetMin(), 4)} .. {np.round(box.GetMax(), 4)}")
app.close()
