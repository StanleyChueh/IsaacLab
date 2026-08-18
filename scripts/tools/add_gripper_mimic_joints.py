# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Couple each OpenArm gripper's two jaws with a PhysX mimic joint, in the robot asset.

WHAT WAS WRONG
--------------
On the real gripper one motor drives both jaws through a mechanical coupling, so they are
always mirror images: closing on an off-centre object slides that object to the gripper's
centreline. In v1_camera_isaac.usd the two jaws are independent PhysicsPrismaticJoints,
each with its own drive, and the ONLY thing that kept them together was that every action
term commands both to the same number (see openarm.py's "openarm_gripper" actuator and the
tasks' BinaryJointPositionActionCfg).

Equal commands are not a constraint. The moment a jaw touches something it stalls where the
contact is, while the other keeps going -- so the pair ends up split, the aperture straddles
whatever the object's actual offset was, and the jaws visibly wander instead of opening and
closing about their centre. Measured on the last frame of the ten teleop demos in
logs/demos/pickup_pringles_V8.hdf5, the grasping hand's jaws differ by up to 17 mm
(0.0336 / 0.0163 m in demo_5) while their sum stays at roughly the can's width. Open jaws,
which touch nothing, are identical to four decimals -- the split is entirely contact.

WHAT THIS DOES
--------------
Applies PhysxMimicJointAPI to each *_finger_joint2, referencing its *_finger_joint1, with
gearing = -1 and offset = 0. The schema's relation is

    jointPosition + gearing * referenceJointPosition + offset = 0

so gearing -1 means pos(joint2) == pos(joint1) -- exactly the coupling the comments across
openarm.py and the task cfgs already assumed was there. It is an articulation-level
constraint solved with the joints, not a drive, so contact cannot break it: a jaw stopped by
the can now holds the other jaw back too, and the can centres itself.

The constraint is rigid. PhysxMimicJointAPI in this PhysX schema version (107.3) has no
naturalFrequency/dampingRatio attributes, so there is no compliance to tune and none of the
softness a URDF-importer-authored mimic used to bring with it.

Both jaws keep their drives and stay in the "openarm_gripper" actuator group. Drive and
constraint agree (both say pos2 == pos1), so they do not fight -- and the drives are still
what produces grip force. Do NOT "simplify" by dropping joint2 from the actuator group: an
undriven jaw would be dragged along by the constraint alone and contribute no squeeze.

USAGE
-----
    ./isaaclab.sh -p scripts/tools/add_gripper_mimic_joints.py            # apply
    ./isaaclab.sh -p scripts/tools/add_gripper_mimic_joints.py --check    # report only
    ./isaaclab.sh -p scripts/tools/add_gripper_mimic_joints.py --revert   # remove again

Idempotent: re-running after a successful apply reports "already coupled" and writes
nothing. The physics layer is backed up once, next to itself, before the first edit.

Isaac Sim has to be up before `pxr` is importable, which is why this runs under a headless
AppLauncher rather than as a plain Python script.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Couple each OpenArm gripper's jaws with a PhysX mimic joint.")
parser.add_argument(
    "--usd",
    type=str,
    default=None,
    help="Robot USD to edit. Defaults to the usd_path OPENARM_BI_CFG spawns.",
)
parser.add_argument("--check", action="store_true", help="Report the current state and exit without editing.")
parser.add_argument("--revert", action="store_true", help="Remove the mimic joints instead of adding them.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import shutil
import sys

from pxr import PhysxSchema, Sdf, Usd

from isaaclab_assets.robots.openarm import OPENARM_BI_CFG


def report(message: str) -> None:
    """Print to stderr, unbuffered -- Kit swallows this process's stdout once the app is up."""
    print(message, file=sys.stderr, flush=True)


# The jaw that mimics, and the jaw it mimics -- leader first everywhere else in the stack too
# (record_demos_openarm.py's LEFT_GRIPPER_JOINT_NAME / LEFT_GRIPPER_FOLLOWER_JOINT_NAME).
JAW_PAIRS = [
    ("openarm_left_finger_joint2", "openarm_left_finger_joint1"),
    ("openarm_right_finger_joint2", "openarm_right_finger_joint1"),
]

# Instance name of the multiple-apply schema. Ignored for a single-DOF joint (the axis is
# implicit for PhysicsPrismaticJoint, see the schema doc), but one has to be picked; "rotY"
# is what the URDF importer uses.
MIMIC_INSTANCE = "rotY"

# pos(joint2) + gearing * pos(joint1) + offset == 0, i.e. the two jaws stay equal.
GEARING = -1.0
OFFSET = 0.0


def joints_scope(stage: Usd.Stage) -> Sdf.Path:
    """Path of the prim the four finger joints live under, e.g. /openarm/joints."""
    for prim in stage.Traverse():
        if prim.GetName() == JAW_PAIRS[0][1]:
            return prim.GetPath().GetParentPath()
    raise RuntimeError(f"No prim named {JAW_PAIRS[0][1]} anywhere in {stage.GetRootLayer().identifier}.")


def describe(prim: Usd.Prim) -> str:
    """One jaw's current coupling, for the --check report."""
    api = PhysxSchema.PhysxMimicJointAPI(prim, MIMIC_INSTANCE)
    if not prim.HasAPI(PhysxSchema.PhysxMimicJointAPI, MIMIC_INSTANCE):
        return "independent (no mimic joint)"
    targets = api.GetReferenceJointRel().GetTargets()
    ref = targets[0].name if targets else "<no reference joint>"
    return f"mimics {ref} with gearing={api.GetGearingAttr().Get()}, offset={api.GetOffsetAttr().Get()}"


def main() -> None:
    usd_path = args_cli.usd or OPENARM_BI_CFG.spawn.usd_path
    stage = Usd.Stage.Open(usd_path)
    scope = joints_scope(stage)

    # Author into the layer that DEFINES the joints (the physics sublayer), not into whichever
    # layer happens to be the stage's root -- v1_camera_isaac.usd is a variant/payload stub, and
    # an override stranded there would be silently dropped by a different variant selection.
    leader_prim = stage.GetPrimAtPath(scope.AppendChild(JAW_PAIRS[0][1]))
    physics_layer = leader_prim.GetPrimStack()[0].layer
    report(f"[mimic] robot usd     : {usd_path}")
    report(f"[mimic] physics layer : {physics_layer.identifier}")
    report(f"[mimic] joints scope  : {scope}")

    # Reopen that layer on its own so the finger joints are local prims: applying an API through
    # the composed stage would land the opinion in the root layer across the payload arc.
    layer_stage = Usd.Stage.Open(physics_layer.identifier)

    report("[mimic] before:")
    for follower, _ in JAW_PAIRS:
        report(f"          {follower}: {describe(layer_stage.GetPrimAtPath(scope.AppendChild(follower)))}")
    if args_cli.check:
        return

    if not args_cli.revert:
        backup = f"{os.path.splitext(physics_layer.realPath)[0]}.pre-mimic.usd.bak"
        if not os.path.exists(backup):
            shutil.copy2(physics_layer.realPath, backup)
            report(f"[mimic] backed up physics layer -> {backup}")

    changed = False
    for follower, leader in JAW_PAIRS:
        follower_prim = layer_stage.GetPrimAtPath(scope.AppendChild(follower))
        leader_path = scope.AppendChild(leader)
        if not follower_prim.IsValid():
            raise RuntimeError(f"{scope.AppendChild(follower)} is not defined in {physics_layer.identifier}.")
        if not layer_stage.GetPrimAtPath(leader_path).IsValid():
            raise RuntimeError(f"{leader_path} is not defined in {physics_layer.identifier}.")

        has_api = follower_prim.HasAPI(PhysxSchema.PhysxMimicJointAPI, MIMIC_INSTANCE)
        if args_cli.revert:
            if not has_api:
                continue
            follower_prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, MIMIC_INSTANCE)
            # RemoveAPI drops the schema from apiSchemas but leaves its attributes behind, and a
            # stale physxMimicJoint:* property block is exactly the sort of thing that looks
            # applied in a USD diff. Clear them too.
            for prop in list(follower_prim.GetPropertyNames()):
                if prop.startswith(f"physxMimicJoint:{MIMIC_INSTANCE}:"):
                    follower_prim.RemoveProperty(prop)
            changed = True
            continue

        api = PhysxSchema.PhysxMimicJointAPI.Apply(follower_prim, MIMIC_INSTANCE)
        api.CreateReferenceJointRel().SetTargets([leader_path])
        api.CreateGearingAttr().Set(GEARING)
        api.CreateOffsetAttr().Set(OFFSET)
        changed = True

    if changed:
        physics_layer.Save()
        report(f"[mimic] saved {physics_layer.identifier}")
    else:
        report("[mimic] nothing to do")

    # Read back through a freshly composed stage rather than trusting the in-memory edit: what
    # matters is what the simulation will resolve when it spawns the robot.
    verify = Usd.Stage.Open(usd_path)
    report("[mimic] after:")
    for follower, _ in JAW_PAIRS:
        report(f"          {follower}: {describe(verify.GetPrimAtPath(scope.AppendChild(follower)))}")


if __name__ == "__main__":
    main()
    simulation_app.close()
