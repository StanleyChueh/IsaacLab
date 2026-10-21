# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Mount the wrist D435i on the real robot's camera bar (mesh/arm/visual/d435i_mount.stl) in v1_camera_isaac.usd.

The imported camera assembly (cam.dae, one mesh per openarm_*_camera_link) is a wrist clamp ending in a fork, plus a
D435i housing CENTRED on the fork by its own short stub. The real camera is screwed to a 90 mm bar whose 12.5 mm
pivot tab sits 13-25.5 mm from one end; the tab goes into the fork, so the housing sits 25.75 mm to one side of it
(its RGB end over the fork) and 13.5 mm out from the pivot rather than 4.5. Real body_cam frames at the same joint
states show the housing to the robot's right of its post on BOTH arms, i.e. +y of camera_link.

The bar is then turned about the fork pivot (camera_link y) until the housing faces the way the wrist camera SENSOR looks
(SENSOR_ROT, the env cfg's CameraCfg rot): the real bar pivots there, and the imported assembly pointed the housing
11.8 deg further forward than the calibrated view, i.e. away from the gripper.

The wrist clamp and fork are the real ones too (mesh/arm/visual/d435i_ring_right.stl / d435i_ring_left.stl, the two halves of
the clamp ring; the right half carries the fork): the imported assembly's own clamp and plate only locate the pivot and are
replaced. The ring's pivot hole goes on that pivot with its axis along camera_link y, the ring around the wrist axis (z) --
the hole sits 72.1 mm out from the ring centre and 24.6 mm above its mid-plane, so the ring lands centred on the wrist
(within 0.5 mm) with its mid-plane at the camera_link origin. The left half bolts on opposite, flange to flange.

The USB cable goes in the housing's far (+y) end as on the real robot (body_cam frames of the real recording): a black
USB-C plug, then a 4.5 mm cable that arcs down past that end and back over the top of the wrist toward the forearm.
On the real arm it hangs loose and is tied further back; here it is one rigid shape on camera_link (CABLE_PATH), visual
only -- a rigid cable with a collider would snag the can and the other arm in ways the real one never does.

This writes
  configuration/v1_camera_isaac_cammount_mesh.usd   the clamp ring halves, the housing moved onto the bar (its stub
                                                    removed), the bar, the plug and the cable
  configuration/v1_camera_isaac_cammount.usda       on both camera links: the old visuals/collisions deactivated, the
                                                    new parts as visual (material "材質" -- the name the real-look albedo
                                                    tables key on -- and "d435_aluminium" for the housing) and as
                                                    convex-decomposed colliders
and adds the .usda as a sublayer of v1_camera_isaac.usd. The wrist camera SENSOR pose (the env cfg's CameraCfg offset) is
not touched; the script prints where the sensor's view axis leaves the housing's front face (put the lens there).

  ./isaaclab.sh -p scripts/tools/sysid/add_openarm_camera_mount.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app = AppLauncher(args_cli).app

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

ASSET = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "v1_camera_isaac")
BAR_STL = os.path.join(ASSET, "mesh/arm/visual/d435i_mount.stl")
RING_RIGHT_STL = os.path.join(ASSET, "mesh/arm/visual/d435i_ring_right.stl")
RING_LEFT_STL = os.path.join(ASSET, "mesh/arm/visual/d435i_ring_left.stl")
MESH_USD = os.path.join(ASSET, "configuration/v1_camera_isaac_cammount_mesh.usd")
LAYER = os.path.join(ASSET, "configuration/v1_camera_isaac_cammount.usda")
ROOT = os.path.join(ASSET, "v1_camera_isaac.usd")
MAT = "材質"
HOUSING_MAT = "d435_aluminium"  # the D435's housing: silver aluminium, unlike the black printed ring and bar

# The bar, in its STL's millimetres: length along x (0..90), tab x 64.5..77 with its pivot hole at y 7, z 4.5; the
# camera's back face lies on z = 18.
BAR_HOLE = np.array([70.75, 7.0, 4.5])
BAR_FACE_Z = 18.0
# The clamp ring halves, in their STLs' millimetres: the ring lies in x-z around the axis y (band y 0..10), bulging to +x;
# the right half's fork tabs (z 35.2..40.2 and 52.8..57.8) carry the pivot hole along z.
RING_RIGHT_CENTRE = np.array([0.0, 5.0, 46.5])  # ring centre on the band's mid-plane
RING_RIGHT_HOLE = np.array([72.11, 29.57, 46.5])
RING_LEFT_CENTRE = np.array([0.0, 5.0, 47.0])
STUB_U = 0.0038  # m from the pivot along the housing's depth axis: behind this is the sim housing's own fork stub
CABLE_RADIUS = 0.00225
PLUG_SIZE = (0.0065, 0.022, 0.012)  # m: USB-C overmold, thickness x length (out of the port) x width
# m, camera_link: where the cable runs after leaving the plug (its first point is computed: the plug's end)
CABLE_PATH = np.array([
    [0.000, 0.010, -0.004], [-0.004, 0.016, -0.020], [-0.008, 0.012, -0.040],  # relative to the plug end: out, then down
]), np.array([
    [0.040, 0.050, -0.060], [0.046, 0.024, -0.092], [0.046, -0.008, -0.128], [0.040, -0.035, -0.180],
    # absolute: over the top of the wrist motor, to the forearm's -y side
])
CABLE_COLOR = (0.02, 0.02, 0.02)
VISUAL_ONLY = ("plug", "cable", "housing_front", "housing")
COLLISION_ONLY = ("housing_hull",)
SENSOR_ROT = (0.68057, -0.19188, -0.19188, 0.68057)  # w,x,y,z, ROS optical (z = view), in ee_tcp = camera_link axes
SENSOR_POS = np.array([0.06, 0.001, 0.05125])  # the cfg's original wrist-camera mount point, in camera_link


def camera_link_mesh(stage, side: str) -> trimesh.Trimesh:
    """The imported camera assembly in its camera_link frame (instance proxies included)."""
    link = stage.GetPrimAtPath(f"/openarm/openarm_{side}_camera_link")
    to_link = UsdGeom.Xformable(link).ComputeLocalToWorldTransform(0).GetInverse()
    for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh) and "/visuals/" in str(prim.GetPath()):
            mesh = UsdGeom.Mesh(prim)
            pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
            m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0) * to_link)
            pts = (np.c_[pts, np.ones(len(pts))] @ m)[:, :3]
            counts, idx, faces, k = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get(), [], 0
            for c in counts:
                faces += [[idx[k], idx[k + t], idx[k + t + 1]] for t in range(1, c - 1)]
                k += c
            return trimesh.Trimesh(pts, np.array(faces), process=True)
    raise SystemExit(f"no camera visual mesh under {link.GetPath()}")


def material_inputs(stage, side: str) -> dict:
    shader = stage.GetPrimAtPath(
        f"/openarm/openarm_{side}_camera_link/visuals/openarm_{side}_camera_visual/Scene/Looks/{MAT}/{MAT}"
    )
    return {a.GetName(): (a.GetTypeName(), a.Get()) for a in shader.GetAttributes() if a.HasAuthoredValue()}


def build_parts(assembly: trimesh.Trimesh) -> dict[str, trimesh.Trimesh]:
    housing, bracket, plate = sorted(assembly.split(only_watertight=False), key=lambda b: -len(b.faces))
    # the fork's pivot: centre of the bracket vertices on its hole (x-z circle near (72, 24) mm)
    v = bracket.vertices
    near = v[np.hypot(v[:, 0] - 0.072, v[:, 2] - 0.024) < 0.0035]
    a = np.c_[2 * near[:, 0], 2 * near[:, 2], np.ones(len(near))]
    cx, cz, _ = np.linalg.lstsq(a, near[:, 0] ** 2 + near[:, 2] ** 2, rcond=None)[0]
    pivot = np.array([cx, 0.0, cz])
    # the housing's depth axis: from the pivot to its centre, in the fork's x-z plane (the stub runs along it). Its
    # cross-section is ~25 x 25 mm, so the shortest extent cannot tell depth from height.
    u = housing.sample(40000).mean(0) - pivot
    u[1] = 0.0
    u /= np.linalg.norm(u)
    # drop the stub, then move the housing onto the bar: 25.75 mm along +y, and out to the bar's face along u
    keep = ((housing.triangles_center - pivot) @ u) > STUB_U
    body = housing.submesh([np.flatnonzero(keep)], append=True)
    back = float(np.percentile((body.sample(40000) - pivot) @ u, 2))  # the flat back face, not the stub's cut edges
    shift = np.array([0.0, (BAR_HOLE[0] - 45.0) / 1000.0, 0.0]) + u * ((BAR_FACE_Z - BAR_HOLE[2]) / 1000.0 - back)
    body.apply_translation(shift)
    # the bar: STL x -> -y (its long end toward +y), z -> u, hole on the pivot
    bar = trimesh.load(BAR_STL, force="mesh")
    ex, ez = np.array([0.0, -1.0, 0.0]), u
    ey = np.cross(ez, ex)
    rot = np.c_[ex, ey, ez]
    bar_v = (bar.vertices - BAR_HOLE) / 1000.0 @ rot.T + pivot
    bar = trimesh.Trimesh(bar_v, bar.faces, process=True)
    # turn bar + housing about the pivot (camera_link y) so the housing's depth axis is the sensor's view axis
    w, x, y, z = SENSOR_ROT
    view = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    tilt = np.arctan2(view[0], view[2]) - np.arctan2(u[0], u[2])
    c, s_ = np.cos(tilt), np.sin(tilt)
    rot_y = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
    for part in (body, bar):
        part.vertices = (part.vertices - pivot) @ rot_y.T + pivot
    # where the sensor's view axis leaves the housing's front face
    hits, _, _ = body.ray.intersects_location([SENSOR_POS], [view])
    face = float(((hits - SENSOR_POS) @ view).max()) if len(hits) else float("nan")
    print(f"[add_openarm_camera_mount] pivot {np.round(pivot * 1000, 2)} mm, depth axis {np.round(u, 3)}, housing back"
          f" {back * 1000:.1f} mm from the pivot, housing moved {np.round(shift * 1000, 1)} mm, bar turned"
          f" {np.degrees(tilt):.1f} deg about the pivot; the view axis from {SENSOR_POS * 1000} mm leaves the housing"
          f" {face * 1000:.1f} mm along {np.round(view, 4)}", flush=True)
    plug, cable = build_cable(body)
    ring_right, ring_left = build_rings(pivot)
    # the D435's front is a black glass panel, the rest of the housing bare aluminium: drawn as two parts with their own
    # materials (the collider keeps the whole closed housing)
    front = body.face_normals @ view > 0.9
    shell = body.submesh([np.flatnonzero(~front)], append=True)
    glass = body.submesh([np.flatnonzero(front)], append=True)
    return {"ring_right": ring_right, "ring_left": ring_left, "housing": shell, "housing_front": glass,
            "housing_hull": body, "bar": bar, "plug": plug, "cable": cable}


def build_rings(pivot: np.ndarray) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """The clamp ring halves in camera_link: the right half's pivot hole on ``pivot``, the left half opposite it."""
    # STL x (the fork arm) -> camera_link x, STL y (ring axis) -> z, STL z (pivot axis) -> -y
    to_link = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    right = trimesh.load(RING_RIGHT_STL, force="mesh")
    right.vertices = (right.vertices - RING_RIGHT_HOLE) / 1000.0 @ to_link.T + pivot
    centre = (RING_RIGHT_CENTRE - RING_RIGHT_HOLE) / 1000.0 @ to_link.T + pivot
    # the left half bulges the other way: turned 180 deg about the ring axis, centre on the right half's
    left = trimesh.load(RING_LEFT_STL, force="mesh")
    left.vertices = (left.vertices - RING_LEFT_CENTRE) / 1000.0 @ (np.diag([-1.0, -1.0, 1.0]) @ to_link).T + centre
    print(f"[add_openarm_camera_mount] clamp ring centre {np.round(centre * 1000, 2)} mm (the wrist axis is x = y = 0)", flush=True)
    return right, left


def tube(path: np.ndarray, radius: float, sides: int = 12) -> trimesh.Trimesh:
    """A closed tube of ``radius`` along a polyline (rotation-minimising frames, so it never twists)."""
    t = np.gradient(path, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.cross(t[0], [1.0, 0.0, 0.0] if abs(t[0][0]) < 0.9 else [0.0, 1.0, 0.0])
    n /= np.linalg.norm(n)
    rings = []
    for i, ti in enumerate(t):
        if i:
            n = n - (n @ ti) * ti
            n /= np.linalg.norm(n)
        b = np.cross(ti, n)
        a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        rings.append(path[i] + radius * (np.outer(np.cos(a), n) + np.outer(np.sin(a), b)))
    v = np.concatenate(rings + [path[:1], path[-1:]])
    m, f = len(path), []
    for i in range(m - 1):
        for k in range(sides):
            a0, a1, b0, b1 = i * sides + k, i * sides + (k + 1) % sides, (i + 1) * sides + k, (i + 1) * sides + (k + 1) % sides
            f += [[a0, a1, b1], [a0, b1, b0]]
    c0, c1 = m * sides, m * sides + 1
    f += [[c0, (k + 1) % sides, k] for k in range(sides)]
    f += [[c1, (m - 1) * sides + k, (m - 1) * sides + (k + 1) % sides] for k in range(sides)]
    return trimesh.Trimesh(v, np.array(f), process=True)


def build_cable(housing: trimesh.Trimesh) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """USB-C plug in the housing's +y end face, and the cable from it along CABLE_PATH (Catmull-Rom through it)."""
    v = housing.vertices
    end = v[v[:, 1] > v[:, 1].max() - 0.001]
    port = np.array([(end[:, 0].min() + end[:, 0].max()) / 2, v[:, 1].max(), (end[:, 2].min() + end[:, 2].max()) / 2])
    # the plug's thin side along the housing's depth axis (its front face's normal, x-z plane), its width across it
    depth = housing.vertices[:, [0, 2]] - housing.vertices[:, [0, 2]].mean(0)
    _, _, vt = np.linalg.svd(depth, full_matrices=False)
    plug = trimesh.creation.box(PLUG_SIZE)
    across = np.array([vt[0][0], 0.0, vt[0][1]])
    thin = np.cross(across, [0.0, 1.0, 0.0])
    plug.vertices = plug.vertices @ np.c_[thin, [0.0, 1.0, 0.0], across].T + port + [0.0, PLUG_SIZE[1] / 2, 0.0]
    start = port + [0.0, PLUG_SIZE[1], 0.0]
    rel, absolute = CABLE_PATH
    ctrl = np.concatenate([[start - [0.0, 0.01, 0.0], start], start + rel, absolute, [2 * absolute[-1] - absolute[-2]]])
    pts = []
    for i in range(1, len(ctrl) - 2):  # Catmull-Rom, centripetal enough at these spacings
        p0, p1, p2, p3 = ctrl[i - 1], ctrl[i], ctrl[i + 1], ctrl[i + 2]
        for t in np.linspace(0, 1, 12, endpoint=False):
            pts.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3))
    pts.append(ctrl[-2])
    return plug, tube(np.array(pts), CABLE_RADIUS)


def write_mesh_usd(parts: dict[str, trimesh.Trimesh]) -> None:
    if os.path.exists(MESH_USD):
        os.remove(MESH_USD)
    stage = Usd.Stage.CreateNew(MESH_USD)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/mount")
    stage.SetDefaultPrim(root.GetPrim())
    for name, part in parts.items():
        mesh = UsdGeom.Mesh.Define(stage, f"/mount/{name}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(part.vertices.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(part.faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(part.faces.reshape(-1).tolist()))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*part.bounds[0]), Gf.Vec3f(*part.bounds[1])]))
    stage.GetRootLayer().Save()


def write_layer(parts: dict, shader_inputs: dict) -> None:
    if os.path.exists(LAYER):
        os.remove(LAYER)
    layer = Sdf.Layer.CreateNew(LAYER)
    stage = Usd.Stage.Open(layer)
    stage.SetDefaultPrim(stage.OverridePrim("/openarm"))
    mesh_ref = os.path.relpath(MESH_USD, os.path.dirname(LAYER))
    for side in ("left", "right"):
        link = f"/openarm/openarm_{side}_camera_link"
        for old in ("visuals", "collisions"):
            stage.OverridePrim(f"{link}/{old}").SetActive(False)
        vis = UsdGeom.Xform.Define(stage, f"{link}/mount_visual")
        vis.GetPrim().GetReferences().AddReference(mesh_ref)
        materials = {}
        for mat_name in (MAT, HOUSING_MAT):  # same OmniPBR shader; the real-look albedo table colours each on its own
            material = UsdShade.Material.Define(stage, f"{link}/mount_visual/Looks/{mat_name}")
            shader = UsdShade.Shader.Define(stage, f"{link}/mount_visual/Looks/{mat_name}/{mat_name}")
            for name, (type_name, value) in shader_inputs[side].items():
                attr = shader.GetPrim().CreateAttribute(name, type_name, custom=False)
                attr.Set(value)
            material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
            materials[mat_name] = material
        cable_mat = UsdShade.Material.Define(stage, f"{link}/mount_visual/Looks/cable")
        cable_shader = UsdShade.Shader.Define(stage, f"{link}/mount_visual/Looks/cable/shader")
        cable_shader.CreateIdAttr("UsdPreviewSurface")
        cable_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*CABLE_COLOR))
        cable_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
        cable_mat.CreateSurfaceOutput().ConnectToSource(cable_shader.ConnectableAPI(), "surface")
        col = UsdGeom.Xform.Define(stage, f"{link}/mount_collision")
        col.GetPrim().GetReferences().AddReference(mesh_ref)
        col.CreatePurposeAttr(UsdGeom.Tokens.guide)  # collision only, never rendered
        for name in parts:
            visual = stage.OverridePrim(f"{link}/mount_visual/{name}")
            if name in COLLISION_ONLY:
                visual.SetActive(False)
            else:
                UsdShade.MaterialBindingAPI.Apply(visual).Bind(
                    cable_mat if name in ("plug", "cable") else materials[HOUSING_MAT if name == "housing" else MAT])
            prim = stage.OverridePrim(f"{link}/mount_collision/{name}")
            if name in VISUAL_ONLY:
                prim.SetActive(False)
                continue
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexDecomposition")
    layer.Save()


def add_sublayer() -> None:
    root = Sdf.Layer.FindOrOpen(ROOT)
    rel = os.path.relpath(LAYER, ASSET)
    if rel not in root.subLayerPaths:
        root.subLayerPaths.append(rel)
        root.Save()


stage = Usd.Stage.Open(ROOT)
stage.MuteLayer(Sdf.Layer.FindOrOpen(LAYER).identifier if os.path.exists(LAYER) else LAYER)  # re-runs read the original
# The right arm's assembly is the un-mirrored one; the real cameras and bars are the same part on both arms.
parts = build_parts(camera_link_mesh(stage, "right"))
inputs = {side: material_inputs(stage, side) for side in ("left", "right")}
stage = None
write_mesh_usd(parts)
write_layer(parts, inputs)
add_sublayer()
for name, part in parts.items():
    print(f"[add_openarm_camera_mount] {name}: camera_link-frame bounds {np.round(part.bounds * 1000, 1).tolist()} mm", flush=True)
app.close()
