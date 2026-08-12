# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the water-bottle prop used by the OpenArm hand-over task.

Run this only to regenerate the asset; the produced .usd is committed, so normal use just
loads it.

    ./isaaclab.sh -p scripts/tools/make_bottle_asset.py
    # or, without booting the simulator, any interpreter that can import pxr:
    python scripts/tools/make_bottle_asset.py --out <path>.usd

Why a generated prop rather than a stock Nucleus one
----------------------------------------------------
The hand-over needs an object BOTH grippers can take hold of, one after the other, without the
grasp depending on how the object happens to be turned. The pick-up task's 55 mm cube fails
that on two counts: it is only as tall as it is wide, so there is no second place for the
receiving hand to hold, and its faces mean the approach direction matters. The obvious stock
alternatives are no better -- Isaac's YCB bottles (006_mustard_bottle, 021_bleach_cleanser)
have flattened cross-sections around 58 x 85 mm, and the OpenArm gripper only opens 88 mm
(2 x the 44 mm per-jaw travel), so they are graspable from one direction and jam from the other.

A round bottle solves both: constant 55 mm diameter -- the same width the gripper is already
known to close on for the cube -- over a 130 mm body, so it can be grasped from any horizontal
direction and at any height. The left hand takes it low, the right hand takes it high, and the
two grips never have to compete for the same spot.

Geometry is a stack of cylinders (body / shoulder / neck / cap) under a single rigid body.
Cylinders rather than a tapered mesh so every collision shape stays analytic -- no convex
decomposition, no cooked mesh, nothing that can approximate the graspable diameter away.
"""

import argparse
import os

parser = argparse.ArgumentParser(description="Generate the OpenArm hand-over task's bottle prop.")
parser.add_argument(
    "--out",
    type=str,
    default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "source/isaaclab_assets/data/props/water_bottle.usd",
    ),
    help="Path of the .usd file to write.",
)
args = parser.parse_args()

# UsdPhysics only -- deliberately no PhysxSchema. Its schema plugins are only registered inside
# a running Kit app, so importing it here breaks the "regenerate without booting the simulator"
# path, and nothing it offers is needed: PhysX-specific collision/rigid-body properties are
# applied by Isaac Lab at spawn time from the task cfg's collision_props/rigid_props.
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

# ── Dimensions (metres) ───────────────────────────────────────────────────────
# BODY_RADIUS is the load-bearing number: 2 x 0.0275 = 55 mm, matching the cube width the
# OpenArm gripper is already known to close on (recorded grasps settle around a 65 mm opening
# on the 55 mm cube). Widening it eats into the 88 mm maximum opening; narrowing it makes the
# jaws travel further before contact, which costs grip force under the gripper's soft PD gains.
BODY_RADIUS = 0.0275
BODY_HEIGHT = 0.130
SHOULDER_RADIUS = 0.021
SHOULDER_HEIGHT = 0.030
NECK_RADIUS = 0.014
NECK_HEIGHT = 0.025
CAP_RADIUS = 0.017
CAP_HEIGHT = 0.018
TOTAL_HEIGHT = BODY_HEIGHT + SHOULDER_HEIGHT + NECK_HEIGHT + CAP_HEIGHT  # 0.203

ORIGIN_Z = -BODY_HEIGHT / 2.0
"""Where the bottle's own origin sits in the stack: the middle of the graspable body, not the
base. The base would be the obvious choice for placing it on a pad, but every "is this arm
holding it" test measures |object_origin - TCP| against a threshold of a few centimetres (see
openarm_task_modes.cube_grasped_by), and in a hand-over the two hands deliberately hold the
bottle at different heights. With the origin at the base, the receiving hand's grip ~10 cm up
lands outside any threshold that still means "holding it" for the other hand. Centring the
origin on the body halves the worst-case distance and makes one threshold cover both grips."""

MASS = 0.15
"""kg -- a part-filled 500 ml bottle. Kept light on purpose: the OpenArm gripper runs at
stiffness 80 N/m in OPENARM_BI_HIGH_PD_CFG, so grip force is small and a heavy prop slides out
of a grasp that looks perfectly closed."""

FRICTION = 1.0
"""Static and dynamic friction, for the same reason: friction is what actually holds the bottle
once the jaws are on it."""

# (name, radius, height, colour) stacked bottom-up along +Z from the object origin.
_SECTIONS = [
    ("body", BODY_RADIUS, BODY_HEIGHT, (0.24, 0.52, 0.78)),
    ("shoulder", SHOULDER_RADIUS, SHOULDER_HEIGHT, (0.24, 0.52, 0.78)),
    ("neck", NECK_RADIUS, NECK_HEIGHT, (0.85, 0.90, 0.94)),
    ("cap", CAP_RADIUS, CAP_HEIGHT, (0.92, 0.92, 0.35)),
]


def main() -> None:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Root: one rigid body for the whole stack, so the sections behave as a single object.
    root = UsdGeom.Xform.Define(stage, "/Bottle")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(MASS)
    # Leave inertia and centre of mass unset: PhysX derives them from the collision shapes and
    # rescales to MASS, which puts the centre of mass low (the body dominates the volume) --
    # exactly what keeps a bottle upright on the pad instead of toppling when nudged.

    # Physics material, bound to the whole object.
    material = UsdShade.Material.Define(stage, "/Bottle/PhysicsMaterial")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(FRICTION)
    material_api.CreateDynamicFrictionAttr(FRICTION)
    material_api.CreateRestitutionAttr(0.0)

    z_cursor = 0.0
    for name, radius, height, color in _SECTIONS:
        section = UsdGeom.Cylinder.Define(stage, f"/Bottle/{name}")
        section.CreateAxisAttr(UsdGeom.Tokens.z)
        section.CreateRadiusAttr(radius)
        section.CreateHeightAttr(height)
        # A USD cylinder is centred on its own origin, so each section is translated to sit on
        # top of the previous one, then the whole stack is shifted by ORIGIN_Z.
        section.CreateExtentAttr([
            Gf.Vec3f(-radius, -radius, -height / 2.0),
            Gf.Vec3f(radius, radius, height / 2.0),
        ])
        UsdGeom.Xformable(section).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, ORIGIN_Z + z_cursor + height / 2.0)
        )
        section.CreateDisplayColorAttr([Gf.Vec3f(*color)])

        UsdPhysics.CollisionAPI.Apply(section.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(section.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
        z_cursor += height

    stage.GetRootLayer().Save()
    print(f"Wrote {args.out}")
    print(f"  total height {TOTAL_HEIGHT * 1000:.0f} mm, graspable body {BODY_RADIUS * 2000:.0f} mm dia"
          f" x {BODY_HEIGHT * 1000:.0f} mm, mass {MASS} kg")
    print(f"  origin at body centre: base is {-ORIGIN_Z * 1000:.0f} mm below it, cap"
          f" {(TOTAL_HEIGHT + ORIGIN_Z) * 1000:.0f} mm above -- so resting on a pad means"
          f" init_state.pos[2] = pad_top + {-ORIGIN_Z:.4f}")


if __name__ == "__main__":
    main()
