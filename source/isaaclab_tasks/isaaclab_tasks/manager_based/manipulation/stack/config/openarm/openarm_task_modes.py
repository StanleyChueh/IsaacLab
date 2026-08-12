# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Task-mode variants of the OpenArm pick-up task, shared by recording, annotation and Mimic.

The base task (:mod:`pickup_ik_abs_env_cfg`) spawns cube_2 anywhere across the full pad width
and calls the episode a success purely on cube height. That is fine for raw teleop, but not
once the demos are meant to be Mimic source data:

* Mimic needs a FIXED end-effector -> subtask mapping. With the manipulated object landing on
  either side, some demos are left-arm grasps and some are right-arm grasps, so any per-arm
  subtask signal is dead in part of the dataset and ``annotate_demos.py`` rejects those episodes.
* A height-only success fires even when nothing was grasped -- an object nudged off the pad, or
  (observed in logs/demos/pickup.hdf5) a demo whose "grasping" hand never closed at all.

A task mode fixes both ends of that: which arm(s) the demo is about, and what actually counts
as done. Pick one at record time (``record_demos_openarm.py --task_mode``) and use the SAME one
when annotating, otherwise the signals the annotator looks for are not the ones the demo was
recorded against.

Every mode replaces the base task's cube_2 with a single shared prop -- a chips can (see
:data:`CAN_USD_PATH`) -- instead of a cube. A cube worked for a one-handed grasp but not for
hand-over: it is no taller than it is wide, so there is no second place for the receiving hand
to hold that the giving hand is not already on, and its flat faces make the grasp depend on
which way it happens to be turned. The can's shape (round, ~60 mm diameter, 200 mm tall, standing
upright -- see :data:`CAN_SCALE` / :data:`CAN_ROT`) fixes
both problems with ONE asset: round means the grasp never depends on yaw, and tall means the two
hands in a hand-over can hold it at genuinely different heights without colliding -- so the same
object now serves the single-arm modes too, rather than needing a second, hand-over-specific
prop.

Modes
-----
The can spawns anywhere on the reachable pad in EVERY mode -- the mode does not confine it to one
half (see :func:`reset_object_free`). What the mode fixes is which arm may move and what counts as
done, which is all Mimic needs; a narrower object distribution would only give it less to
generalise from.

``left`` / ``right``
    Single-arm pick-up (Mimic plan A: one arm leads, the other idles). ONLY that arm is
    teleoperable (see :data:`CONTROLLED_ARMS` -- the other arm is held at its rest pose, though
    both arms' joints are still recorded), and success requires that arm to be *holding* the can
    while it is lifted.

``handover``
    Bimanual hand-over (Mimic plan C). Both arms are teleoperable, and the episode is the whole
    sequence: the right arm picks the can up and presents it, the left arm takes it while the
    right is still holding on, the right lets go, and finally the left arm releases it and the can
    drops back down. Success fires on that last landing -- see :func:`handover_success`, which
    walks a stage counter through the sequence rather than testing any single instant, because no
    instant distinguishes a hand-over from a plain left-arm pick now that the can can spawn under
    either hand.

Subtask term signals published per mode -- these are what ``annotate_demos.py`` annotates and
what a Mimic env cfg's ``subtask_term_signal`` entries must reference:

============  ===================================================================
``left``      ``grasp``, ``lift``
``right``     ``grasp``, ``lift``          (same names, right arm -- so one single-arm
                                            Mimic cfg covers either side unchanged)
``handover``  ``grasp_right``, ``handover``, ``grasp_left``   (giving arm first)
============  ===================================================================

Every signal function here takes its arm-specific inputs (TCP frame, jaw joints, open value,
threshold) as explicit params rather than reading a single global ``env.cfg.gripper_*`` set --
that global set is exactly what makes :func:`stack.mdp.observations.object_grasped` unable to
describe two arms at once. Signatures are spelled out in full for the same reason there are no
``**kwargs`` anywhere below: the manager statically matches a term function's arguments against
its params (see ``ManagerBase._resolve_common_term_cfg``) and a ``**kwargs`` catch-all reads to
it as an unsatisfied mandatory argument.
"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.math import quat_apply_inverse


# ── Modes ─────────────────────────────────────────────────────────────────────
TASK_MODE_LEFT = "left"
TASK_MODE_RIGHT = "right"
TASK_MODE_HANDOVER = "handover"
TASK_MODES = (TASK_MODE_LEFT, TASK_MODE_RIGHT, TASK_MODE_HANDOVER)

# ── Per-arm gripper description ───────────────────────────────────────────────
# Both jaws are checked. joint2 is independently actuated (see openarm.py's "openarm_gripper"
# actuator comment), so a jaw left behind by a bad command would still read as "closed" if only
# joint1 were consulted.
LEFT_FINGER_JOINTS = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
RIGHT_FINGER_JOINTS = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]
GRIPPER_OPEN_VAL = 0.044
"""Fully-open finger position (m) -- matches the tasks' BinaryJointPositionActionCfg."""
GRIPPER_THRESHOLD = 0.010
"""How far (m) a jaw must have travelled from open before it counts as closed on something.

Sized against the can, and it has to be: a jaw closing on a 60 mm-wide can stops at the can's
surface, so it travels much LESS than one closing on the small cube this number was originally
picked for. Measured at equilibrium with the gripper commanded shut on the can, the two jaws
settle at 0.0285 and 0.0308 -- travels of 0.0155 and 0.0132 from the 0.044 open position. The
old 0.018 sat above both, so a genuinely closed, load-bearing grip read as "not closed" and no
grasp signal could ever fire.

0.010 clears the smaller of those two travels with margin while still being far from the ~0 travel
of a jaw that is simply open. It does not have to carry the whole "is this a grasp" decision on its
own -- :func:`object_grasped_by` also requires the hand to be on the can."""

# ── The manipulated object -- one prop, every mode ────────────────────────────
CAN_NAME = "can"

CAN_USD_PATH = (
    "/home/csl/Stanley_ws/IsaacLab/Isaac_task_asset/"
    "Lightwheel_JJP7OOnKjX_chips_can_berkeley_meshes/chips_can_berkeley_meshes.usd"
)
"""Third-party (Lightwheel/Berkeley) prop. Unlike make_bottle_asset.py's hand-built bottle, this
one ships its own RigidBodyAPI/MassAPI (0.205 kg), three convex-hull collision meshes and a PBR
material -- nothing here re-authors physics or appearance, only where/how it's placed."""

CAN_SCALE = 0.8
"""Uniform spawn scale, matching the Isaac Sim stage transform verified by hand (Scale
0.8/0.8/0.8). Post-scale the can is ~60 mm across and ~200 mm tall."""

CAN_ROT = (1.0, 0.0, 0.0, 0.0)
"""(w, x, y, z) spawn orientation -- identity, i.e. the can stands upright on the pad.

Deliberately NOT the -90 deg X rotation that the Isaac Sim "Add Reference" panel shows when the
raw asset is dropped into a stage by hand: that panel's value is relative to the referencing
prim, whereas this one is the rigid body's own world orientation. Measuring the mesh directly
(``Sites/reg_bbox`` and the render/collision meshes all agree) shows the can is authored with its
tall axis along local +Z, and the stage's up-axis is Z, so identity already stands it up -- adding
-90 about X lays it on its side, which is what the first attempt at this actually produced."""

CAN_HALF_HEIGHT = 0.1
"""m -- half the can's vertical extent once :data:`CAN_SCALE` is applied (0.8 * 250.2 mm / 2), and
how far its own origin sits above its base: the asset's origin is at the geometric centre, not the
base, so resting on the pad means ``pos[2] = pad_top + CAN_HALF_HEIGHT``."""

CAN_REST_OFFSET = -0.002
"""m -- PhysX rest offset for the can's colliders. Negative, i.e. the surface PhysX holds other
shapes off of sits INSIDE the can's visible surface, so a closing gripper keeps going until its
fingers slightly overlap the render mesh instead of stopping short of it.

This exists because the asset's colliders are not one shape but three, and the widest is not the
one you grasp. Measured in-sim at :data:`CAN_SCALE` (world diameters):

    body collider   0.0580   vs body visual  0.0586   -- collider already 0.6 mm inside
    bottom collider 0.0588
    lid-ring collider 0.0603  vs lid visual   0.0600   -- and 1.7 mm PROUD of the body visual

So a gripper closing on the can's body is stopped ~0.9 mm early per side by a lid rim it never
looks like it is touching -- the visible gap. -2 mm pulls every collider comfortably inside the
render mesh (lid ring becomes 0.0563 effective, under the 0.0586 body) and leaves the grip looking
closed. Scaling the collision meshes themselves would be the other fix, but they live inside a
third-party USD, and per-collider scale is not something UsdFileCfg can express.

Costs ~1.3 mm of the can sinking into the pad at rest (rest offsets sum across a contact pair),
which is well under what is visible on a 200 mm can. Do not make this much larger for that reason,
and do not make it positive -- a positive rest offset is an actual air gap at every contact."""

EEF_TO_CAN_THRESHOLD = 0.07
"""Max distance (m) from a TCP to the can's origin for a single-arm grasp to count as holding it.
A hand steadying a free-standing 200 mm can (60 mm diameter, 30 mm radius post-scale) grips near
its centre of mass, not at an extreme, so budget only a modest vertical offset from the origin
(~5 cm) on top of the radius: sqrt(0.03^2 + 0.05^2) ~= 0.058 m, plus a margin for
fingertip-vs-TCP offset."""

EEF_TO_CAN_THRESHOLD_HANDOVER = 0.11
"""Max distance (m) from a TCP to the can's origin for a hand-over grasp to count as holding it.
Larger than :data:`EEF_TO_CAN_THRESHOLD` because a hand-over holds the can at two DIFFERENT
heights at once (giving hand low, receiving hand high) -- budgeting a ~9 cm vertical offset from
the origin on each side: sqrt(0.03^2 + 0.09^2) ~= 0.095 m, plus margin."""

GRASP_AXIAL_TOLERANCE = 0.12
"""m -- how far along the can's own axis, measured from its centre, a hand may be and still count
as holding it. The can is 200 mm long, so a hand on it is anywhere within ~100 mm of the centre
along that axis; this is that half-length plus margin for gripping right at the rim.

This exists because "distance from the hand to the object's origin" is the wrong test for a long
object: gripping the can perfectly at its top rim puts the hand ~100 mm from the origin, which no
sane origin-distance threshold accepts, while a threshold loose enough to accept it would also
accept a hand floating 100 mm off to the side. :func:`object_grasped_by` therefore splits the
hand-to-can vector into its along-the-can and across-the-can parts and bounds them separately --
a cylinder test rather than a sphere test."""

EEF_TCP_TO_GRASP_OFFSET = (0.0, 0.0, 0.1025)
"""m -- offset from the ``*_ee_tcp`` link's origin to the point midway between its jaws, in that
link's LOCAL frame. Applied to both arms' FrameTransformers (see :func:`apply_task_mode`).

Despite the name, ``openarm_*_ee_tcp`` is NOT at the tool centre point: measured in-sim it sits on
the wrist, coincident with ``link7`` and 102.5 mm short of the fingers, and the offset is identical
on both arms and invariant to arm pose (re-measured after moving the arm: 0.10250). Reporting that
link as the end-effector makes every hand-to-object distance overshoot by 102.5 mm, which is enough
on its own to stop any grasp signal from ever firing -- a real closed grasp measured 0.1236 m
against a 0.07 m threshold. It also feeds Mimic, which transforms source segments relative to the
eef pose it is given, so a frame 10 cm behind the hand mis-places every generated trajectory."""


# ── Signal primitives ─────────────────────────────────────────────────────────

def _gripper_is_closed(
    robot: Articulation, joint_names: list[str], open_val: float, threshold: float
) -> torch.Tensor:
    """(N,) bool: every named jaw has travelled more than *threshold* away from *open_val*."""
    joint_ids, _ = robot.find_joints(joint_names)
    closed = torch.ones(robot.num_instances, dtype=torch.bool, device=robot.device)
    for joint_id in joint_ids:
        closed &= torch.abs(robot.data.joint_pos[:, joint_id] - open_val) > threshold
    return closed


def object_grasped_by(
    env,
    ee_frame_cfg: SceneEntityCfg,
    gripper_joint_names: list[str],
    object_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_open_val: float = GRIPPER_OPEN_VAL,
    gripper_threshold: float = GRIPPER_THRESHOLD,
    diff_threshold: float = EEF_TO_CAN_THRESHOLD,
    axial_threshold: float = GRASP_AXIAL_TOLERANCE,
) -> torch.Tensor:
    """(N,) bool: the arm behind *ee_frame_cfg* has its gripper closed ON the object.

    "On the object" is a cylinder test, not a sphere one: the hand-to-can vector is split into the
    part along the can's own axis and the part across it, and each is bounded separately
    (*axial_threshold* and *diff_threshold*). See :data:`GRASP_AXIAL_TOLERANCE` for why -- a 200 mm
    can gripped at its rim is 100 mm from its own origin, so no single radius both accepts that and
    rejects a hand hovering 100 mm to the side.

    The can's axis is taken from its current orientation rather than assumed vertical, so this
    keeps working while the can is tilted mid-hand-over or lying down.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # Hand position relative to the can, expressed in the CAN's frame: component z is along the
    # can's length, (x, y) is across it.
    to_hand_w = ee_frame.data.target_pos_w[:, 0, :] - obj.data.root_pos_w
    to_hand_local = quat_apply_inverse(obj.data.root_quat_w, to_hand_w)

    radial = torch.linalg.vector_norm(to_hand_local[:, :2], dim=1)
    axial = torch.abs(to_hand_local[:, 2])

    on_object = torch.logical_and(radial < diff_threshold, axial < axial_threshold)
    return torch.logical_and(
        on_object,
        _gripper_is_closed(robot, gripper_joint_names, gripper_open_val, gripper_threshold),
    )


def object_grasped_by_obs(
    env,
    ee_frame_cfg: SceneEntityCfg,
    gripper_joint_names: list[str],
    object_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_open_val: float = GRIPPER_OPEN_VAL,
    gripper_threshold: float = GRIPPER_THRESHOLD,
    diff_threshold: float = EEF_TO_CAN_THRESHOLD,
) -> torch.Tensor:
    """(N, 1) float version of :func:`object_grasped_by`, for a subtask-terms observation group."""
    return object_grasped_by(
        env,
        ee_frame_cfg=ee_frame_cfg,
        gripper_joint_names=gripper_joint_names,
        object_cfg=object_cfg,
        robot_cfg=robot_cfg,
        gripper_open_val=gripper_open_val,
        gripper_threshold=gripper_threshold,
        diff_threshold=diff_threshold,
    ).unsqueeze(-1).float()


def object_lifted_and_grasped(
    env,
    ee_frame_cfg: SceneEntityCfg,
    gripper_joint_names: list[str],
    min_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_open_val: float = GRIPPER_OPEN_VAL,
    gripper_threshold: float = GRIPPER_THRESHOLD,
    diff_threshold: float = EEF_TO_CAN_THRESHOLD,
) -> torch.Tensor:
    """(N,) bool: the object is above *min_height* AND that arm is holding it.

    This is the single-arm modes' success/auto-end condition, replacing the base task's
    height-only one. Height alone accepts an object that was merely knocked upwards and -- the
    case that actually bit -- accepts a demo in which the grasping hand never closed at all,
    producing "successful" episodes with no grasp anywhere in them.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.logical_and(
        obj.data.root_pos_w[:, 2] > min_height,
        object_grasped_by(
            env,
            ee_frame_cfg=ee_frame_cfg,
            gripper_joint_names=gripper_joint_names,
            object_cfg=object_cfg,
            robot_cfg=robot_cfg,
            gripper_open_val=gripper_open_val,
            gripper_threshold=gripper_threshold,
            diff_threshold=diff_threshold,
        ),
    )


def object_lifted_and_grasped_obs(
    env,
    ee_frame_cfg: SceneEntityCfg,
    gripper_joint_names: list[str],
    min_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_open_val: float = GRIPPER_OPEN_VAL,
    gripper_threshold: float = GRIPPER_THRESHOLD,
    diff_threshold: float = EEF_TO_CAN_THRESHOLD,
) -> torch.Tensor:
    """(N, 1) float version of :func:`object_lifted_and_grasped`."""
    return object_lifted_and_grasped(
        env,
        ee_frame_cfg=ee_frame_cfg,
        gripper_joint_names=gripper_joint_names,
        min_height=min_height,
        object_cfg=object_cfg,
        robot_cfg=robot_cfg,
        gripper_open_val=gripper_open_val,
        gripper_threshold=gripper_threshold,
        diff_threshold=diff_threshold,
    ).unsqueeze(-1).float()


def object_held_by_both_obs(
    env,
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_open_val: float = GRIPPER_OPEN_VAL,
    gripper_threshold: float = GRIPPER_THRESHOLD,
    diff_threshold: float = EEF_TO_CAN_THRESHOLD,
) -> torch.Tensor:
    """(N, 1) float: BOTH grippers are closed on the object at the same time.

    This is the hand-over instant itself -- the only moment in the episode when the object sits
    inside two closed grippers -- so it marks the end of the right arm's "present it" subtask and
    the point from which the left arm carries it.
    """
    both = torch.logical_and(
        object_grasped_by(
            env,
            ee_frame_cfg=left_ee_frame_cfg,
            gripper_joint_names=LEFT_FINGER_JOINTS,
            object_cfg=object_cfg,
            robot_cfg=robot_cfg,
            gripper_open_val=gripper_open_val,
            gripper_threshold=gripper_threshold,
            diff_threshold=diff_threshold,
        ),
        object_grasped_by(
            env,
            ee_frame_cfg=right_ee_frame_cfg,
            gripper_joint_names=RIGHT_FINGER_JOINTS,
            object_cfg=object_cfg,
            robot_cfg=robot_cfg,
            gripper_open_val=gripper_open_val,
            gripper_threshold=gripper_threshold,
            diff_threshold=diff_threshold,
        ),
    )
    return both.unsqueeze(-1).float()


HANDOVER_STAGE_WAITING = 0
HANDOVER_STAGE_RIGHT_HELD = 1
HANDOVER_STAGE_PASSED_TO_LEFT = 2
HANDOVER_STAGE_DONE = 3
"""How far through the hand-over each env has got. See :func:`handover_success`."""

_HANDOVER_STAGE_ATTR = "_openarm_handover_stage"
"""Attribute :func:`handover_success` stores its per-env stage counter under."""


def handover_stage(env) -> torch.Tensor:
    """(N,) long: how far each env has got through the hand-over. Created on first use.

    Exposed so a caller can report progress -- record_demos_openarm.py prints each transition, so
    a hand-over that never completes shows WHICH step it stalled on rather than just failing to
    end the episode.
    """
    stage = getattr(env, _HANDOVER_STAGE_ATTR, None)
    if stage is None or stage.shape[0] != env.num_envs:
        stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        setattr(env, _HANDOVER_STAGE_ATTR, stage)
    return stage


def reset_handover_stage(env, env_ids: torch.Tensor) -> None:
    """Clear the hand-over progress of the envs being reset. Installed as a "reset" event term.

    An event term rather than something :func:`handover_success` infers for itself, because it is
    the only mechanism that is guaranteed to run on exactly the envs being reset, exactly when they
    reset. Deriving the boundary inside the success function instead means guessing from the step
    counter, and under record_demos_openarm.py's button-X flow that guess is wrong: the success
    check is not evaluated at all until the episode is armed, so a short episode followed by a
    late arming looks like time moving forwards and the previous hand-over's completed stage
    survives into the new episode -- which would end it instantly, on the operator's first press.
    """
    handover_stage(env)[env_ids] = HANDOVER_STAGE_WAITING


def handover_success(
    env,
    min_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_open_val: float = GRIPPER_OPEN_VAL,
    gripper_threshold: float = GRIPPER_THRESHOLD,
    diff_threshold: float = EEF_TO_CAN_THRESHOLD,
    drop_tolerance: float = 0.02,
) -> torch.Tensor:
    """(N,) bool: the whole hand-over has played out, ending with the can back down on a surface.

    The episode is the entire sequence, not any single instant, so this walks a per-env stage
    counter forward and only reports success at the end of it:

    ========================================  ==========================================
    :data:`HANDOVER_STAGE_WAITING`            nothing yet
    :data:`HANDOVER_STAGE_RIGHT_HELD`         the RIGHT (giving) arm has grasped the can
    :data:`HANDOVER_STAGE_PASSED_TO_LEFT`     the LEFT (receiving) arm has it, and had it
                                              while the right arm was still holding on --
                                              i.e. an actual pass, not a re-pick off the pad
    :data:`HANDOVER_STAGE_DONE`               the left arm let go and the can came back down
    ========================================  ==========================================

    Stages only ever advance, so a momentary loss of contact mid-pass cannot walk the episode
    backwards, and the counter is reset per episode.

    Why a sequence and not a snapshot: the can now spawns anywhere on the pad (see
    :func:`reset_object_free`), including directly under the left hand, so no instantaneous
    condition can distinguish a hand-over from a plain left-arm pick. Requiring the right arm to
    have held it first, and to still be holding it at the moment the left arm takes it, is what
    makes the demo a hand-over rather than "the left arm picked something up".

    Args:
        min_height: how high the can must get to count as carried rather than dragged. Applied at
            the pass, which is the point of the task -- not at the end, where the can is by
            definition back down.
        drop_tolerance: how close (m) to its resting height the can must return before it counts
            as having landed. Absolute height, so a can that rolls off the pad onto the floor
            satisfies it too.
    """
    obj: RigidObject = env.scene[object_cfg.name]

    stage = handover_stage(env)

    def _held_by(frame_name, fingers):
        return object_grasped_by(
            env,
            ee_frame_cfg=SceneEntityCfg(frame_name),
            gripper_joint_names=fingers,
            object_cfg=object_cfg,
            robot_cfg=robot_cfg,
            gripper_open_val=gripper_open_val,
            gripper_threshold=gripper_threshold,
            diff_threshold=diff_threshold,
        )

    right_holds = _held_by("right_ee_frame", RIGHT_FINGER_JOINTS)
    left_holds = _held_by(left_ee_frame_cfg.name, LEFT_FINGER_JOINTS)

    # 0 -> 1: the giving hand has the can.
    stage[(stage == HANDOVER_STAGE_WAITING) & right_holds] = HANDOVER_STAGE_RIGHT_HELD

    # 1 -> 2: both hands on it at once, up in the air. Demanding the overlap (rather than just
    # "left holds it now") is what rules out the right arm putting the can down and the left arm
    # picking it back up, which is not a hand-over.
    lifted = obj.data.root_pos_w[:, 2] > min_height
    passed = (stage == HANDOVER_STAGE_RIGHT_HELD) & right_holds & left_holds & lifted
    stage[passed] = HANDOVER_STAGE_PASSED_TO_LEFT

    # 2 -> 3: nobody is holding it and it has come back down. Checking both grippers, not just the
    # left, keeps a "pass it back" ending from counting.
    landed = obj.data.root_pos_w[:, 2] <= (_object_rest_z(env) + drop_tolerance)
    released = ~left_holds & ~right_holds
    dropped = (stage == HANDOVER_STAGE_PASSED_TO_LEFT) & released & landed
    stage[dropped] = HANDOVER_STAGE_DONE

    return stage == HANDOVER_STAGE_DONE


# ── Object spawn region -- the SAME for every mode ────────────────────────────
# The can goes anywhere on the reachable pad, whichever mode is being recorded. Positive y is
# the robot's LEFT-arm side. Coordinates here are absolute (env-local metres), not the offsets
# off a default pose that PickUpEventCfg.randomize_cube_2 uses -- see :func:`reset_object_free`.
#
# Modes deliberately do NOT restrict the can to "their" half. Which arm the demo is about is
# already pinned by two other things -- only that arm can be driven (:data:`CONTROLLED_ARMS`) and
# only that arm satisfies the success condition -- so a spawn-side restriction adds nothing except
# a narrower object distribution for Mimic to generalise from.
_OBJECT_X_RANGE = (0.20, 0.46)
"""m -- forward band. The near bound is the base task's (>=8 cm clearance from the robot origin,
so the gripper cannot clip the base while reaching); the far bound keeps the can's far side ~2 cm
inside the pad's x=0.51 edge."""

_OBJECT_Y_RANGE = (-0.27, 0.27)
"""m -- full pad width, both arms' halves AND the middle, keeping the base task's ~1.5 cm margin
from the pad's y=+-0.285 half-width."""

# ── Where the can must NOT spawn: on top of a parked arm ──────────────────────
# Both arms park at y=+-0.153, z=0.478 -- only 0.198 m above the pad, i.e. level with the rim of a
# standing can (top z=0.480) -- and they sweep through that spot as they settle out of the pose
# reset_scene_to_default teleports them back to on every reset. A can spawned there is not nudged
# but launched: measured peak speeds of 5-7 m/s, with some cans clearing the table entirely.
#
# Measured over the real reset path on a grid of spawn points, the failures fill |y|=[0.10,0.20]
# for every x up to ~0.40, and vanish completely by x>=0.425 (the parked fingertips reach x=0.318)
# and outside |y|=[0.10,0.20] at any x. These boxes are that measured region plus ~1 cm of margin.
# Everything else on the pad is fair game, which is what keeps the distribution wide.
#
# If the arms' rest pose is ever raised to clear a standing can (they would need ~7 cm), these
# boxes can shrink or go away entirely -- they describe the rest pose, not the object.
_ARM_KEEP_OUT_BOXES = (
    (0.19, 0.42, 0.085, 0.215),    # left arm's parked footprint  (x_min, x_max, y_min, y_max)
    (0.19, 0.42, -0.215, -0.085),  # right arm's, mirrored
)


def reset_object_free(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
    x_range: tuple[float, float] = _OBJECT_X_RANGE,
    y_range: tuple[float, float] = _OBJECT_Y_RANGE,
    keep_out_boxes: tuple = _ARM_KEEP_OUT_BOXES,
    rot: tuple[float, float, float, float] = CAN_ROT,
) -> None:
    """Stand the object upright at a uniformly random spot on the pad, outside *keep_out_boxes*.

    Used instead of :func:`~isaaclab.envs.mdp.events.reset_root_state_uniform` because that one
    samples a plain rectangle, and the rectangle we want has two bites taken out of it (see
    :data:`_ARM_KEEP_OUT_BOXES` -- spawning inside a parked arm launches the can rather than
    placing it). Rejection sampling keeps the accepted distribution exactly uniform over the
    remaining area, which a "clamp it to the nearest legal spot" fix would not: clamping piles
    probability onto the keep-out borders, which is the opposite of the wide spread this exists
    for.

    Coordinates are absolute env-local metres rather than the default-pose offsets the base task's
    ranges use -- one less indirection between this code and the measured geometry it encodes.
    Orientation is forced to *rot* (and velocity to zero) rather than jittered: a can lying down
    is a failed episode, not a harder one, so the only randomised thing here is where it stands.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    device = asset.device
    n = len(env_ids)

    x = torch.empty(n, device=device).uniform_(*x_range)
    y = torch.empty(n, device=device).uniform_(*y_range)

    # Resample only the rejected ones, so every env keeps an independent uniform draw. The loop is
    # bounded because an unbounded one would hang forever on a mis-specified keep-out that covers
    # the whole region; the legal area here is ~70% of the rectangle, so exhausting 100 rounds has
    # probability ~1e-52 per env. Falling back to the last (possibly illegal) draw rather than
    # raising keeps a pathological config from killing a recording session mid-run.
    for _ in range(100):
        rejected = torch.zeros(n, dtype=torch.bool, device=device)
        for x_min, x_max, y_min, y_max in keep_out_boxes:
            rejected |= (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        if not bool(rejected.any()):
            break
        k = int(rejected.sum())
        x[rejected] = torch.empty(k, device=device).uniform_(*x_range)
        y[rejected] = torch.empty(k, device=device).uniform_(*y_range)

    states = asset.data.default_root_state[env_ids].clone()
    states[:, 0] = env.scene.env_origins[env_ids, 0] + x
    states[:, 1] = env.scene.env_origins[env_ids, 1] + y
    states[:, 2] = env.scene.env_origins[env_ids, 2] + _object_rest_z(env)
    states[:, 3:7] = torch.tensor(rot, device=device, dtype=states.dtype)
    states[:, 7:] = 0.0  # a spawn should not inherit the velocity it had when the episode ended
    asset.write_root_state_to_sim(states, env_ids=env_ids)


def _object_rest_z(env) -> float:
    """Height (env-local m) at which the can's origin sits when standing on the pad."""
    return 2.0 * float(env.cfg.scene.workspace_pad.init_state.pos[2]) + CAN_HALF_HEIGHT

CONTROLLED_ARMS = {
    TASK_MODE_LEFT: ("left",),
    TASK_MODE_RIGHT: ("right",),
    TASK_MODE_HANDOVER: ("left", "right"),
}
"""Which arms the operator may actually drive while recording each mode.

A single-arm mode is not just a spawn range and a success condition: it is a promise that the
demo contains exactly one arm's pick. Leaving the idle arm live lets an operator nudge the can
with it, producing a "left" demo whose real grasp came from the right hand -- which no per-arm
subtask signal can describe. The idle arm is therefore frozen at its rest pose during recording
(see record_demos_openarm.py's arm gating).

This gates CONTROL only. Both arms' joint positions/actions stay in the recorded dataset either
way, because Mimic needs full-width action rows for both end-effectors regardless of which arm
moved -- see the ``_idle_subtasks`` an idle arm still gets."""

_SIGNAL_NAMES = ("grasp", "lift", "grasp_left", "grasp_right", "handover")
"""Every signal any mode can publish -- cleared before a mode installs its own, so switching
modes on one cfg can't leave a previous mode's signal behind for the annotator to wait on."""

LEFT_EEF = "left_eef"
RIGHT_EEF = "right_eef"
"""Mimic end-effector keys. Every mode files subtasks under BOTH, even when one arm only
idles: an eef absent from ``subtask_configs`` has no source segment at all, so Mimic would
have nothing to say about that arm's action columns."""


# ── Mimic subtask structures ──────────────────────────────────────────────────

def _pick_subtasks(term_signal_grasp: str) -> list:
    """The two-subtask pick-up sequence used by both single-arm modes: grasp, then lift."""
    from isaaclab.envs.mimic_env_cfg import SubTaskConfig

    return [
        SubTaskConfig(
            object_ref=CAN_NAME,
            subtask_term_signal=term_signal_grasp,
            # Short tail: 2-5 extra steps keeps the seam tight and limits the positional
            # spread at the transition.
            subtask_term_offset_range=(2, 5),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 10},
            action_noise=0.001,
            num_interpolation_steps=50,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            description="Reach and grasp the can",
            next_subtask_description="Lift the can",
        ),
        SubTaskConfig(
            object_ref=CAN_NAME,
            subtask_term_signal=None,  # last subtask -- ends with the episode
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 10},
            action_noise=0.001,
            num_interpolation_steps=10,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            description="Lift the can above the table",
        ),
    ]


def _idle_subtasks() -> list:
    """One do-nothing subtask spanning the whole episode, for the arm that just holds still.

    ``object_ref=None`` on purpose: there is no object this arm is manipulating, so its segment
    is transformed by a plain delta pose rather than relative to the can. That also rules out
    the 'nearest_neighbor_object' strategy, which needs an object to measure against.
    """
    from isaaclab.envs.mimic_env_cfg import SubTaskConfig

    return [
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="random",
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            description="Stay clear of the working arm",
        ),
    ]


def _handover_subtask_configs() -> tuple[dict, list]:
    """The bimanual hand-over structure: (subtask_configs, task_constraint_configs).

    right : grasp the can low -> present it -> let go and retreat   (the GIVING hand)
    left  : approach and take it higher up -> carry it away         (the RECEIVING hand)

    The two arms are tied together by one SEQUENTIAL constraint: the left arm's "take it"
    subtask may not *finish* before the right arm's "present it" subtask has. It is allowed to
    run right up to its last :data:`_HANDOVER_MIN_TIME_DIFF` steps first, so the approach still
    overlaps the right arm's lift instead of starting only after it -- which is both faster and
    what the human demo actually looks like.
    """
    from isaaclab.envs.mimic_env_cfg import SubTaskConfig, SubTaskConstraintConfig, SubTaskConstraintType

    def _subtask(term_signal, description, interp=20, offset=(0, 0)):
        return SubTaskConfig(
            object_ref=CAN_NAME,
            subtask_term_signal=term_signal,
            subtask_term_offset_range=offset,
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 10},
            action_noise=0.001,
            num_interpolation_steps=interp,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            description=description,
        )

    # Insertion order stays left-then-right to match the single-arm modes, so the eef ordering a
    # Mimic env sees never depends on the mode -- only the CONTENT differs, the left arm now
    # being the receiving hand.
    subtask_configs = {
        LEFT_EEF: [
            _subtask("grasp_left", "Approach the presented can and take it"),
            # Last subtask, so it runs to the end of the episode -- which now means carrying the
            # can clear and then letting it drop, since that landing is what ends the demo.
            _subtask(None, "Carry the can away, then release it and let it drop"),
        ],
        RIGHT_EEF: [
            _subtask("grasp_right", "Reach and grasp the can", interp=50, offset=(2, 5)),
            _subtask("handover", "Lift the can and present it to the left hand"),
            _subtask(None, "Release the can and retreat"),
        ],
    }
    constraints = [
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[(RIGHT_EEF, 1), (LEFT_EEF, 0)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
            sequential_min_time_diff=_HANDOVER_MIN_TIME_DIFF,
        )
    ]
    return subtask_configs, constraints


_HANDOVER_MIN_TIME_DIFF = 20
"""Steps of the left arm's take-it subtask that are held back until the right arm has finished
presenting. Everything before that runs concurrently. -1 would serialise the two arms
completely (left waits, motionless, for the whole right-arm pick)."""


# ── The one entry point ───────────────────────────────────────────────────────

def apply_task_mode(env_cfg, mode: str, lift_height_offset: float = 0.025) -> None:
    """Specialise a parsed OpenArm pick-up cfg for *mode*. Call BEFORE ``gym.make()``.

    Patches: the right-arm TCP frame (added unconditionally, since without it no right-arm
    signal can be computed at all), the right EEF pose observations a second Mimic end-effector
    needs, the manipulated object (cube_2 is always replaced by the can -- see the module
    docstring for why one prop now covers every mode), the subtask term signals, the
    success/auto-end condition, the object's spawn range, and -- on a Mimic cfg -- the subtask
    structure.

    Args:
        env_cfg: the parsed OpenarmPickUpRedCubeEnvCfg (or a subclass of it).
        mode: one of :data:`TASK_MODES`.
        lift_height_offset: how far (m) above the object's resting height counts as lifted.
            Defaults to the base task's offset, which was calibrated against the peak heights
            actually reached in recorded teleop demos -- read
            OpenarmPickUpRedCubeEnvCfg.__post_init__ before changing it.
    """
    if mode not in TASK_MODES:
        raise ValueError(f"Unknown task mode '{mode}'. Expected one of {TASK_MODES}.")
    if not hasattr(env_cfg.scene, "ee_frame"):
        raise ValueError(
            "The task cfg has no 'ee_frame' -- apply_task_mode() expects the OpenArm pick-up"
            " task (or a subclass), whose left TCP frame it mirrors for the right arm."
        )

    # ── Per-arm grasp-point frames ────────────────────────────────────────────
    # BOTH are (re)built here, including the left one the base task already defines, because both
    # need the same correction: the *_ee_tcp link they target is on the wrist, 102.5 mm short of
    # the jaws, so the frames as shipped report a point no grasp ever happens at. See
    # EEF_TCP_TO_GRASP_OFFSET. Overriding the left frame here rather than in the base task keeps
    # the fix scoped to the modes whose signals depend on it.
    def _grasp_frame(side: str) -> FrameTransformerCfg:
        return FrameTransformerCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/openarm_{side}_link1",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/openarm_{side}_ee_tcp",
                    name="end_effector",
                    offset=OffsetCfg(pos=EEF_TCP_TO_GRASP_OFFSET),
                )
            ],
        )

    env_cfg.scene.ee_frame = _grasp_frame("left")
    env_cfg.scene.right_ee_frame = _grasp_frame("right")

    # ── Right EEF pose observations ──────────────────────────────────────────
    # The task publishes only the left TCP as eef_pos/eef_quat. A dual-arm Mimic env needs one
    # pose per end-effector key, and even in single-arm 'right' mode the right pose is the one
    # the demo is actually about.
    from isaaclab_tasks.manager_based.manipulation.stack import mdp

    env_cfg.observations.policy.right_eef_pos = ObsTerm(
        func=mdp.ee_frame_pos, params={"ee_frame_cfg": SceneEntityCfg("right_ee_frame")}
    )
    env_cfg.observations.policy.right_eef_quat = ObsTerm(
        func=mdp.ee_frame_quat, params={"ee_frame_cfg": SceneEntityCfg("right_ee_frame")}
    )

    # ── The manipulated object -- the can, every mode ────────────────────────
    # A fresh SceneEntityCfg per term rather than one shared instance: the managers call
    # resolve() on each param at play time, and a config object shared across terms is a
    # standing invitation for one term's resolution to be read as another's.
    object_cfg = lambda: SceneEntityCfg(CAN_NAME)  # noqa: E731

    # The pad's top surface is what the can stands on. Derived from the pad cfg rather than
    # hardcoded, for the same reason the base task derives cube_2's resting height: a pad-height
    # change must not silently leave a literal behind (workspace_pad is a full-height box centred
    # at PAD_HEIGHT/2, so its top is twice its centre height).
    pad_top_z = 2.0 * float(env_cfg.scene.workspace_pad.init_state.pos[2])
    object_rest_z = pad_top_z + CAN_HALF_HEIGHT
    env_cfg.scene.can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Can",
        # Only the pose the can is spawned at before the first reset -- reset_object_free
        # overwrites x/y on every reset, and derives the same z from the pad itself.
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, object_rest_z], rot=CAN_ROT),
        spawn=sim_utils.UsdFileCfg(
            usd_path=CAN_USD_PATH,
            scale=(CAN_SCALE, CAN_SCALE, CAN_SCALE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(rest_offset=CAN_REST_OFFSET),
            semantic_tags=[("class", CAN_NAME)],
        ),
    )
    # A cube left in the scene would be a second rigid object in every datagen_info's
    # object_poses and an extra thing for the arms to knock over.
    env_cfg.scene.cube_2 = None
    # The debug contact sensor (pickup_ik_abs_env_cfg.py's __post_init__) is wired to filter
    # against Cube_2's prim path by default; repoint it at whatever object actually exists now,
    # or it silently reports zero contact forever.
    if hasattr(env_cfg.scene, "contact_right_left_finger"):
        env_cfg.scene.contact_right_left_finger.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Can"]

    diff_threshold = EEF_TO_CAN_THRESHOLD_HANDOVER if mode == TASK_MODE_HANDOVER else EEF_TO_CAN_THRESHOLD
    lift_height = object_rest_z + lift_height_offset

    left_frame = SceneEntityCfg("ee_frame")
    right_frame = SceneEntityCfg("right_ee_frame")

    def common():
        return {"object_cfg": object_cfg(), "diff_threshold": diff_threshold}

    # ── Subtask term signals + success condition ─────────────────────────────
    subtask_terms = env_cfg.observations.subtask_terms
    for stale_signal in _SIGNAL_NAMES:
        if hasattr(subtask_terms, stale_signal):
            setattr(subtask_terms, stale_signal, None)

    if mode in (TASK_MODE_LEFT, TASK_MODE_RIGHT):
        frame = left_frame if mode == TASK_MODE_LEFT else right_frame
        fingers = LEFT_FINGER_JOINTS if mode == TASK_MODE_LEFT else RIGHT_FINGER_JOINTS
        # Signal names stay 'grasp'/'lift' for both sides so one single-arm Mimic cfg covers
        # either arm -- only the eef key they are filed under changes.
        subtask_terms.grasp = ObsTerm(
            func=object_grasped_by_obs,
            params={"ee_frame_cfg": frame, "gripper_joint_names": fingers, **common()},
        )
        subtask_terms.lift = ObsTerm(
            func=object_lifted_and_grasped_obs,
            params={
                "ee_frame_cfg": frame,
                "gripper_joint_names": fingers,
                "min_height": lift_height,
                **common(),
            },
        )
        env_cfg.terminations.success = TerminationTermCfg(
            func=object_lifted_and_grasped,
            params={
                "ee_frame_cfg": frame,
                "gripper_joint_names": fingers,
                "min_height": lift_height,
                **common(),
            },
            time_out=False,
        )
    else:
        subtask_terms.grasp_left = ObsTerm(
            func=object_grasped_by_obs,
            params={"ee_frame_cfg": left_frame, "gripper_joint_names": LEFT_FINGER_JOINTS, **common()},
        )
        subtask_terms.handover = ObsTerm(func=object_held_by_both_obs, params=dict(common()))
        subtask_terms.grasp_right = ObsTerm(
            func=object_grasped_by_obs,
            params={"ee_frame_cfg": right_frame, "gripper_joint_names": RIGHT_FINGER_JOINTS, **common()},
        )
        env_cfg.terminations.success = TerminationTermCfg(
            func=handover_success, params={"min_height": lift_height, **common()}, time_out=False
        )
        # handover_success is stateful, so its state has to be cleared per episode by something
        # that actually runs on reset -- see reset_handover_stage.
        env_cfg.events.reset_handover_stage = EventTerm(func=reset_handover_stage, mode="reset")

    # ── Object spawn ─────────────────────────────────────────────────────────
    # The base task's own randomisation term is dropped in every mode: the object it targeted
    # (cube_2) no longer exists once the can has replaced it above.
    #
    # Identical in every mode -- the can goes anywhere on the pad regardless of which arm the demo
    # is about. See _OBJECT_X_RANGE / _ARM_KEEP_OUT_BOXES.
    env_cfg.events.randomize_cube_2 = None
    env_cfg.events.randomize_object = EventTerm(
        func=reset_object_free,
        mode="reset",
        params={"asset_cfg": object_cfg()},
    )

    # ── Mimic subtask structure ──────────────────────────────────────────────
    # Only present when the cfg is a Mimic one (annotate_demos.py / generate_dataset.py); the
    # plain recording task has no subtask_configs field and skips this entirely.
    if not hasattr(env_cfg, "subtask_configs"):
        return
    if mode == TASK_MODE_HANDOVER:
        env_cfg.subtask_configs, env_cfg.task_constraint_configs = _handover_subtask_configs()
    else:
        working_eef = LEFT_EEF if mode == TASK_MODE_LEFT else RIGHT_EEF
        idle_eef = RIGHT_EEF if mode == TASK_MODE_LEFT else LEFT_EEF
        # Insertion order is left-then-right regardless of which one works, so the eef ordering
        # a Mimic env sees never depends on the mode.
        subtask_configs = {LEFT_EEF: None, RIGHT_EEF: None}
        subtask_configs[working_eef] = _pick_subtasks(term_signal_grasp="grasp")
        subtask_configs[idle_eef] = _idle_subtasks()
        env_cfg.subtask_configs = subtask_configs
        env_cfg.task_constraint_configs = []
