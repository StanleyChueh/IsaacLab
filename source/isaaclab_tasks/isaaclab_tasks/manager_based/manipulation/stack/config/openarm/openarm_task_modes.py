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
What a mode fixes is which arm may move, what counts as done, and where the can may spawn. The
spawn region is NOT "one half of the pad for one arm": it is the region the demos for that mode
actually cover, which is a different thing and is why it is per-mode (see
:data:`OBJECT_SPAWN_RANGES`).

``left`` / ``right``
    Single-arm pick-up (Mimic plan A: one arm leads, the other idles). ONLY that arm is
    teleoperable (see :data:`CONTROLLED_ARMS` -- the other arm is held at its rest pose, though
    both arms' joints are still recorded), and success requires that arm to be *holding* the can
    while it is lifted.

``handover``
    Bimanual hand-over (Mimic plan C). Both arms are teleoperable, and the episode is the whole
    sequence: the right arm picks the can up and presents it, the left arm takes it, and the right
    lets go. The demo is DONE there -- the left arm simply carries the can back to a neutral pose
    holding it. It is not put down again, and nothing about where it ends up gates success. See
    :func:`handover_success`, which walks a stage counter through the sequence rather than testing
    any single instant, because no instant distinguishes a hand-over from a plain left-arm pick when
    the can spawns midway between the two hands, in reach of either.

Subtask term signals published per mode -- these are what ``annotate_demos.py`` annotates and
what a Mimic env cfg's ``subtask_term_signal`` entries must reference:

============  ===================================================================
``left``      ``grasp``, ``lift``
``right``     ``grasp``, ``lift``          (same names, right arm -- so one single-arm
                                            Mimic cfg covers either side unchanged)
``handover``  ``reach_right``, ``grasp_right``, ``presented``, ``reach_left``,
              ``handover`` ~ ``grasp_left``, ``departed_right``
                                           (that is also the order they fire in;
                                            handover and grasp_left land 0-3 steps apart)
============  ===================================================================

Only ``grasp_right`` and ``handover`` are milestones of the hand-over itself. The rest exist to cut
each arm's episode into segments Mimic can transform correctly and constrain against:

``reach_right``     ends the giving arm's approach, starts its close.
``presented``       ends the receiving arm's wait AND the giving arm's initial lift. Without it the
                    receiving arm's approach is one segment starting at step 0, which Mimic anchors
                    on the can's pose ON THE PAD -- but that hand reaches for the can up in the
                    giving hand, 3-26 cm away (median ~13 cm).
``reach_left``      ends the receiving arm's approach, starts its close-and-take. Exists so the
                    giving arm's release can be constrained against the take alone.
``grasp_left``      ends the take.
``departed_right``  ends the giving arm's release -- the pass has happened and that hand has pulled
                    physically clear of the can. Exists so the receiving arm's carry-away can be
                    constrained on the release being over, not on the whole retreat.

See :func:`_handover_subtask_configs` for the full structure and its three sequential constraints.

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
GRIPPER_THRESHOLD = 0.005
"""How far (m) a jaw must have travelled from open before it counts as closed on something.

Sized against the can, and it has to be: a jaw closing on a 60 mm-wide can stops at the can's
surface, so it travels much LESS than one closing on the small cube this number was originally
picked for. Two rounds of measurement, each of which moved this number down:

* commanding the gripper fully shut on the can in isolation settles the jaws at 0.0285/0.0308,
  i.e. travels of 0.0155/0.0132. That ruled out the original 0.018, which sat above both.
* but a real recorded teleop grasp closes LESS far than a full command does. Across the ten
  hand-over demos in logs/demos/pickup_pringle.hdf5 the gripping jaw reaches only ~0.0343-0.0358,
  i.e. travels of 0.0082-0.0097 -- every one of them under 0.010. That is why annotation rejected
  all ten episodes: right_holds was False on all 116 frames of every demo, so the hand-over
  sequence never left stage 0.

0.005 clears the smallest measured real grasp (0.0082) with room to spare while staying far from
the ~0 travel of a jaw that is simply open. It does not have to carry the whole "is this a grasp"
decision on its own -- :func:`object_grasped_by` also requires the hand to be on the can.

Beware of tuning this against a commanded close rather than a recorded one: the two differ by
roughly a factor of two, and only the recorded value is what any annotation actually sees."""

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

DEPART_RADIAL_THRESHOLD = 0.25
DEPART_AXIAL_THRESHOLD = 0.25
"""m -- the cylinder the giving hand must be OUTSIDE of for ``departed_right`` to fire.

Hysteresis: deliberately much larger than the grasp cylinder the hand entered
(:data:`EEF_TO_CAN_THRESHOLD_HANDOVER` / :data:`GRASP_AXIAL_TOLERANCE`), because "no longer inside
the grasp cylinder" is already true at the moment the pass is DETECTED and so cuts a zero-length
subtask.

The reason is the stage machine's detection lag, not the robot: stage PASSED depends on
``right_holding_since_pickup``, a latch that survives the giving hand's grasp detection dropping
out, so by the time the pass is declared the hand has often already drifted out of the grasp
cylinder. Measured over the ten demos in logs/demos/pickup_pringle.hdf5, the gap from ``handover``
to this signal by exit radius:

    0.11 (grasp)  ->  min gap 0 steps   (demo_6 and demo_8 both exactly 0 -- episode rejected)
    0.14 / 0.16   ->  min gap 0 steps   (demo_6 still 0: its hand is already >16 cm out)
    0.18          ->  min gap 2 steps
    0.25          ->  min gap 3 steps, and the release subtask still leaves >=12 steps of episode
                      after it on every demo

0.25 is the first value with real margin on every episode while still firing well before the end.
It reads as "the hand has retreated a good way from the can", which is what the release subtask is
supposed to contain anyway. Raising it further eats into the withdraw subtask for no benefit."""

REACH_RADIAL_THRESHOLD = 0.09
"""m -- the ``reach_right`` signal's radius. Paired with the normal :data:`GRASP_AXIAL_TOLERANCE`;
it is the RADIAL bound alone that has to do the work here, and 0.09 is the only value that works.

"The giving hand has arrived at the can" has to fire on approach and stay quiet while the arm sits
at rest, and the can it reaches for stands on the pad, possibly close under the resting hand. The
two grasp radii both fail, measured rather than assumed:

* the single-arm grasp radius (0.07) NEVER fires: the grasp point of a hand closed on the can sits
  0.037-0.079 radial from the can's axis across the ten demos, so the signal would not exist at all
  on the episodes at the top of that range.
* the hand-over grasp radius (0.11) fires on a transient in demo_9 -- a 6-step clip of the cylinder
  corner, 23 steps before the real approach. 0.10 does the same in demo_7 (a SINGLE frame).

Radial separates the two states cleanly where axial does not. Measured over the ten demos: radial
is 0.132-0.187 at rest and 0.037-0.079 at the grasp -- disjoint with room. Axial is 0.082-0.117 at
rest and 0.023-0.071 at the grasp -- OVERLAPPING, so no axial bound can discriminate; an earlier
version used axial < 0.06 and it made ``reach_right`` a 2-frame transient in demo_6, whose hand
approaches on a diagonal that crosses that bound while the radial is still shrinking.

0.09 is bounded on both sides by geometry, not by these ten episodes:

* it cannot fire at rest ANYWHERE in the spawn region. Sweeping the whole of
  :data:`_HANDOVER_X_RANGE` x :data:`_HANDOVER_Y_RANGE` against the measured resting grasp point
  (~0.328, -0.183) gives a worst-case radial of 0.171 -- and that worst case is a corner of the
  box, not a typical spawn.
* it always fires before the grasp: the largest grasp radial is 0.079.

At 0.09 all ten demos give one contiguous run starting 8-24 steps before ``grasp_right``.

The rest-side bound used to be the tight one, back when the sampled y range reached -0.0754: this
box-to-point sweep put the worst case at 0.108 then (the original entry here recorded 0.097, from a
sweep against the per-demo rest points rather than the single nominal one -- either way it was a
margin of one to two centimetres). Re-centring the range on y=0 (see :data:`_HANDOVER_Y_RANGE`) moved
the box edge nearest the giving hand 6.3 cm away from it, and since the sampled x range still spans
the hand's own x=0.328 the worst case is now a pure y gap: 0.183 - 0.0120 = 0.171. The new sampled
box is a strict subset of the old one, so this direction is safe by construction -- narrowing the
spawn region can only push a resting hand further from the can, never closer.

So the binding constraint is now the GRASP side alone: 0.09 has 0.011 of margin over the widest
measured grasp radial (0.079) and ~0.08 to the worst-case rest. Re-measure this number if the arm's
rest pose changes, or if the spawn region is ever WIDENED again -- narrowing needs no re-measurement,
but a wider y immediately puts the rest-side bound back in play."""

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
    return torch.logical_and(
        hand_on_object(env, ee_frame_cfg, object_cfg, diff_threshold, axial_threshold),
        _gripper_is_closed(robot, gripper_joint_names, gripper_open_val, gripper_threshold),
    )


def hand_to_object_offsets(
    env,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
) -> tuple[torch.Tensor, torch.Tensor]:
    """((N,), (N,)) the radial and axial distance from a hand to the object, in the OBJECT's frame.

    The two quantities :func:`hand_on_object` thresholds, returned raw. Exposed so a caller can
    report HOW FAR a grasp signal is from firing rather than only that it has not -- see
    record_demos_openarm.py's hand-over progress diagnostic.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # Hand position relative to the can, expressed in the CAN's frame: component z is along the
    # can's length, (x, y) is across it.
    to_hand_w = ee_frame.data.target_pos_w[:, 0, :] - obj.data.root_pos_w
    to_hand_local = quat_apply_inverse(obj.data.root_quat_w, to_hand_w)
    return torch.linalg.vector_norm(to_hand_local[:, :2], dim=1), torch.abs(to_hand_local[:, 2])


def gripper_jaw_positions(
    env,
    gripper_joint_names: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(N, J) current position of each named jaw. What :func:`_gripper_is_closed` thresholds.

    Exposed for the same reason as :func:`hand_to_object_offsets`: a hand-over that stalls because
    one jaw of two never closed should say so.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    joint_ids, _ = robot.find_joints(gripper_joint_names)
    return robot.data.joint_pos[:, joint_ids]


def hand_on_object(
    env,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
    diff_threshold: float = EEF_TO_CAN_THRESHOLD,
    axial_threshold: float = GRASP_AXIAL_TOLERANCE,
) -> torch.Tensor:
    """(N,) bool: the hand behind *ee_frame_cfg* is ON the object, whatever its gripper is doing.

    The positional half of :func:`object_grasped_by`, factored out because the hand-over needs the
    two halves separately: "the receiving hand has arrived at the can" is what ends its approach
    subtask (see :func:`object_reached_by_obs`), and that has to fire BEFORE the jaws close, not
    with them.
    """
    radial, axial = hand_to_object_offsets(env, ee_frame_cfg, object_cfg)
    return torch.logical_and(radial < diff_threshold, axial < axial_threshold)


def object_reached_by_obs(
    env,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
    diff_threshold: float = EEF_TO_CAN_THRESHOLD,
    axial_threshold: float = GRASP_AXIAL_TOLERANCE,
) -> torch.Tensor:
    """(N, 1) float: the hand has arrived at the can -- the ``reach_left``/``reach_right`` signals.

    Splits an arm's approach from its close-on-the-object, so the two can be separate subtasks
    (see :func:`_handover_subtask_configs` for what each side's split buys).

    Deliberately NOT latched, unlike ``presented``/``handover``: only the first 0->1 transition is
    read (DataGenInfoPool cuts the boundary there), so latching would change nothing about the
    segmentation, and leaving it raw keeps it the same kind of signal as ``grasp_left`` /
    ``grasp_right``, which are also instantaneous.

    Used directly only for ``reach_right``, with its own tighter radius
    (:data:`REACH_RADIAL_THRESHOLD`): the can the giving hand reaches for stands on the pad,
    possibly right under that hand's rest pose, and neither grasp radius both fires on arrival and
    stays quiet at rest. See that constant for the measurements.

    ``reach_left`` goes through :func:`receiver_reached_obs` instead -- the same cylinder test, but
    gated on ``presented``. The receiving arm has the same rest-pose-proximity problem and cannot be
    fixed by a radius: the can it reaches for has MOVED by then, so there is no fixed geometry to
    tune against, only "has the pick-up happened yet".
    """
    return hand_on_object(
        env,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=object_cfg,
        diff_threshold=diff_threshold,
        axial_threshold=axial_threshold,
    ).unsqueeze(-1).float()


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


"""Note: there is deliberately no time limit on how long the giving hand's grasp stays "recent".

An earlier version expired it after a fixed number of steps, on the theory that a long gap meant
the can had been put down rather than passed. Measured on real demos that theory is simply wrong:
in demo_8 of logs/demos/pickup_pringle.hdf5 the right arm holds the can continuously from step 52
to step 134, but DETECTION of that grasp drops out for 19 steps in the middle while the can sits at
0.404-0.425 -- nowhere near the pad. Any fixed window is a guess about detection dropout length,
and this one rejected a hand-over that plainly happens.

What actually separates a pass from "set it down, pick it up again" is whether the can was ever put
down, which :data:`HANDOVER_PUT_DOWN_MARGIN` tests directly and physically. So that is the only
thing that clears the latch."""

HANDOVER_PUT_DOWN_MARGIN = 0.005
"""m above resting height below which an unheld can counts as having been SET DOWN, cancelling the
giving hand's "has held it since picking it up" latch, which is what lets the pass be
recognised across a detection dropout.

Deliberately tight, because it has to separate two things that are only centimetres apart: a can resting on the pad (rest height exactly)
from a can being handed over low. Measured on real demos the lowest a can gets DURING a pass is
0.3878, i.e. 7.8 mm above its 0.380 resting height -- an operator bringing it right down to meet
the other hand. 5 mm sits between the two with a little room on each side. Widen it and genuine
low hand-overs start being read as put-downs."""

_HANDOVER_MEMORY_ATTR = "_openarm_handover_memory"


def handover_latches(env) -> dict[str, torch.Tensor]:
    """The two latches the hand-over's 1->2 transition needs besides the receiving hand's grasp.

    ``was_lifted``                 the giving arm has held the can above the lift height.
    ``right_holding_since_pickup`` it has not put the can back down since picking it up.

    Read-only view, exposed for the same reason :func:`handover_stage` is: a hand-over that stalls
    at stage 1 should say WHICH precondition is missing. Advancing them is :func:`_handover_tick`'s
    job -- calling this does not tick the machine.
    """
    memory = _handover_memory(env)
    return {
        "was_lifted": memory["was_lifted"],
        "right_holding_since_pickup": memory["right_holding_since_pickup"],
    }


def _handover_memory(env) -> dict:
    """Per-env buffers the hand-over stage machine carries between steps, created on first use."""
    mem = getattr(env, _HANDOVER_MEMORY_ATTR, None)
    if mem is None or mem["was_lifted"].shape[0] != env.num_envs:
        mem = {
            "was_lifted": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "right_holding_since_pickup": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "last_step": -1,
        }
        setattr(env, _HANDOVER_MEMORY_ATTR, mem)
    return mem


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
    memory = _handover_memory(env)
    memory["was_lifted"][env_ids] = False
    memory["right_holding_since_pickup"][env_ids] = False


def handover_success(
    env,
    min_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_open_val: float = GRIPPER_OPEN_VAL,
    gripper_threshold: float = GRIPPER_THRESHOLD,
    diff_threshold: float = EEF_TO_CAN_THRESHOLD,
) -> torch.Tensor:
    """(N,) bool: the hand-over has played out, ending with the can in the LEFT hand.

    The episode is the entire sequence, not any single instant, so this walks a per-env stage
    counter forward and only reports success at the end of it:

    ========================================  ==========================================
    :data:`HANDOVER_STAGE_WAITING`            nothing yet
    :data:`HANDOVER_STAGE_RIGHT_HELD`         the RIGHT (giving) arm has grasped the can
                                              and lifted it off the pad
    :data:`HANDOVER_STAGE_PASSED_TO_LEFT`     the LEFT (receiving) arm has taken it, while
                                              the right arm still had it (or had, just
                                              before) -- an actual pass, not a re-pick
    :data:`HANDOVER_STAGE_DONE`               the right arm has let go: the can is the
                                              left arm's, and the task is complete
    ========================================  ==========================================

    The demo ends there. The left arm carries the can back to a neutral pose still holding it; it
    is NOT put back down, and nothing about where the can finishes gates success. An earlier
    version required a final release-and-land, which rejected real demos outright -- with manual
    saving the operator stops the episode while still holding the can.

    Stages only ever advance, so a momentary loss of contact mid-pass cannot walk the episode
    backwards, and the counter is reset per episode.

    Why a sequence and not a snapshot: the can spawns on the midline between the two hands (see
    :data:`_HANDOVER_Y_RANGE`), i.e. equally within either arm's reach, so no instantaneous
    condition can distinguish a hand-over from a plain left-arm pick. Requiring the right arm to
    have held and lifted it first, and to still have it (or have just had it) when the left arm
    takes over, is what makes the demo a hand-over rather than "the left arm picked something up".

    Args:
        min_height: how high the can must get, WHILE THE RIGHT ARM CARRIES IT, to count as picked
            up rather than dragged. Not applied at the moment of the pass -- a hand-over is
            naturally made by raising the can and bringing it back down to the other hand.
    """
    return _handover_tick(
        env,
        min_height=min_height,
        object_cfg=object_cfg,
        robot_cfg=robot_cfg,
        left_ee_frame_cfg=left_ee_frame_cfg,
        gripper_open_val=gripper_open_val,
        gripper_threshold=gripper_threshold,
        diff_threshold=diff_threshold,
    ) >= HANDOVER_STAGE_DONE


def handover_passed_obs(
    env,
    min_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_open_val: float = GRIPPER_OPEN_VAL,
    gripper_threshold: float = GRIPPER_THRESHOLD,
    diff_threshold: float = EEF_TO_CAN_THRESHOLD,
) -> torch.Tensor:
    """(N, 1) float: the pass itself has happened -- the ``handover`` subtask term signal.

    Reads the SAME stage machine :func:`handover_success` does, deliberately. The signal and the
    success condition have to agree about what a hand-over is, and when they disagree the failure
    is quiet and confusing: an earlier version marked this signal with a plain "both grippers
    closed on the can in this very step" test while success used the staged one, so on real demos
    success fired and the signal never did, and annotate_demos.py rejected the episodes with
    "Did not detect completion for the subtask handover" -- for demos whose hand-over is plainly
    visible. The two grasps are detected together for only 0-4 steps, and in 3 of 10 measured
    episodes for zero steps, because an operator opens the giving hand as the receiving hand
    closes.

    Latching (it stays 1 once the pass has happened) matches the other subtask signals, which stay
    true for as long as the condition holds.
    """
    stage = _handover_tick(
        env,
        min_height=min_height,
        object_cfg=object_cfg,
        robot_cfg=robot_cfg,
        left_ee_frame_cfg=left_ee_frame_cfg,
        gripper_open_val=gripper_open_val,
        gripper_threshold=gripper_threshold,
        diff_threshold=diff_threshold,
    )
    return (stage >= HANDOVER_STAGE_PASSED_TO_LEFT).unsqueeze(-1).float()


def handover_presented_obs(
    env,
    min_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_open_val: float = GRIPPER_OPEN_VAL,
    gripper_threshold: float = GRIPPER_THRESHOLD,
    diff_threshold: float = EEF_TO_CAN_THRESHOLD,
) -> torch.Tensor:
    """(N, 1) float: the giving hand has the can UP -- the ``presented`` subtask term signal.

    Boundary for BOTH arms (see :func:`_handover_subtask_configs`): it splits the receiving arm's
    wait from its approach, so the approach segment Mimic transforms is anchored on the can where
    it is actually reached for -- up in the right hand -- instead of on the pad where it started;
    and it ends the giving arm's initial-lift subtask, giving the "left waits until the can is up"
    constraint a precondition that does not depend on the left arm itself.

    Reads the same stage machine as :func:`handover_success` and :func:`handover_passed_obs`, for
    the same reason they share it: three descriptions of one hand-over that can disagree is three
    ways for annotation to reject a demo that plainly happened. Concretely it is the machine's
    "the right arm has held the can and raised it past *min_height*" latch, i.e. it fires strictly
    after ``grasp_right`` and strictly before ``handover``/``grasp_left``.

    Measured on the ten demos in logs/demos/pickup_pringle.hdf5 it lands 6-10 steps after
    ``grasp_right`` and 21-55 steps before ``grasp_left``, so both segments it creates are
    comfortably non-empty on every episode -- which matters because a subtask boundary that lands on
    or past the next one makes DataGenInfoPool reject the episode outright.

    Latches, like the other signals: it describes something that has happened, not something that is
    currently true, so a momentary loss of grasp detection while the arm carries the can must not
    un-fire it and re-split the segment.
    """
    _handover_tick(
        env,
        min_height=min_height,
        object_cfg=object_cfg,
        robot_cfg=robot_cfg,
        left_ee_frame_cfg=left_ee_frame_cfg,
        gripper_open_val=gripper_open_val,
        gripper_threshold=gripper_threshold,
        diff_threshold=diff_threshold,
    )
    # Read AFTER ticking: the tick is what sets this latch, and it is idempotent within a step, so
    # calling it here costs nothing when handover_passed_obs/handover_success already ticked.
    return _handover_memory(env)["was_lifted"].unsqueeze(-1).float()


def receiver_reached_obs(
    env,
    min_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_open_val: float = GRIPPER_OPEN_VAL,
    gripper_threshold: float = GRIPPER_THRESHOLD,
    diff_threshold: float = EEF_TO_CAN_THRESHOLD,
) -> torch.Tensor:
    """(N, 1) float: the can is UP and the receiving hand has arrived at it -- ``reach_left``.

    :func:`object_reached_by_obs` gated on ``presented``, and the gate is not cosmetic: without it
    this signal is a bare proximity test, and the receiving arm's REST pose can sit within the
    cylinder's tolerance of a can standing on the pad. Measured on demo_3 of
    logs/demos/pickup_pringle.hdf5 the resting hand is 0.117-0.123 radial from the can -- straddling
    the 0.11 grasp radius -- so a single frame of physics jitter dipped under it at step 45 while the
    arm had not moved at all. That lone frame became the subtask boundary, landing 8 steps BEFORE
    ``presented``, and DataGenInfoPool rejected the episode ("signal is not increasing"). The same
    episode annotated an hour earlier had that frame land just above the threshold and passed, which
    is what a coin-flip boundary looks like.

    Tightening the radius, which is what fixes the equivalent problem on the giving arm (see
    :data:`REACH_RADIAL_THRESHOLD`), cannot work here. That fix relies on a fixed geometry -- a
    resting arm and a can standing on the pad -- so a radius can be placed between the two. The
    receiving hand instead reaches for a can that has been picked up and carried to wherever the
    giving arm presents it, so there is no rest-vs-approach distance to separate. (Axial is no help
    on this side either: 0.073-0.085 at rest against 0.100 during the real approach -- the wrong way
    round.)

    Gating on ``presented`` removes the entire failure class instead of tuning around it. Rest-pose
    proximity only happens while the can is still on the pad, and ``presented`` is exactly "the can
    is off the pad in the giving hand" -- after which any cylinder hit by this arm is a real
    approach. Measured across the ten demos it fires 7-28 steps after ``presented`` and 8-28 steps
    before ``grasp_left``.
    """
    _handover_tick(
        env,
        min_height=min_height,
        object_cfg=object_cfg,
        robot_cfg=robot_cfg,
        left_ee_frame_cfg=left_ee_frame_cfg,
        gripper_open_val=gripper_open_val,
        gripper_threshold=gripper_threshold,
        diff_threshold=diff_threshold,
    )
    presented = _handover_memory(env)["was_lifted"]
    arrived = hand_on_object(
        env, ee_frame_cfg=left_ee_frame_cfg, object_cfg=object_cfg, diff_threshold=diff_threshold
    )
    return (presented & arrived).unsqueeze(-1).float()


def departed_after_pass_obs(
    env,
    min_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
    gripper_open_val: float = GRIPPER_OPEN_VAL,
    gripper_threshold: float = GRIPPER_THRESHOLD,
    diff_threshold: float = EEF_TO_CAN_THRESHOLD,
    depart_radial_threshold: float = DEPART_RADIAL_THRESHOLD,
    depart_axial_threshold: float = DEPART_AXIAL_THRESHOLD,
) -> torch.Tensor:
    """(N, 1) float: the pass has happened AND the giving hand has pulled clear of the can -- the
    ``departed_right`` subtask term signal.

    Ends the giving arm's release subtask, which is what lets the receiving arm's carry-away be
    constrained on the release being over rather than on the giving arm's entire retreat (see
    :func:`_handover_subtask_configs`).

    Built from position, not from the jaws, on purpose. A jaw-open test cannot cut this boundary
    robustly: in half of the ten recorded demos grasp DETECTION of the giving hand lapses at or
    before the pass itself (demo_8's jaws are fully open 2 steps before ``handover`` fires), so a
    jaw-based boundary lands on or before the previous one and DataGenInfoPool rejects the episode
    with "signal is not increasing".

    "Clear" is a deliberately BIGGER cylinder than the grasp one -- see
    :data:`DEPART_RADIAL_THRESHOLD` for why the grasp cylinder is not usable here.

    The ``stage >= PASSED`` mask is what stops the trivial early fire: "hand outside the cylinder"
    is true from step 0 (the arm starts at rest, nowhere near the can), and without the mask the
    signal would start at 1 and its first transition would be garbage. With it, the signal is 0
    until the pass and rises exactly once, when the giving hand pulls clear.
    """
    stage = _handover_tick(
        env,
        min_height=min_height,
        object_cfg=object_cfg,
        robot_cfg=robot_cfg,
        left_ee_frame_cfg=left_ee_frame_cfg,
        gripper_open_val=gripper_open_val,
        gripper_threshold=gripper_threshold,
        diff_threshold=diff_threshold,
    )
    away = ~hand_on_object(
        env,
        ee_frame_cfg=right_ee_frame_cfg,
        object_cfg=object_cfg,
        diff_threshold=depart_radial_threshold,
        axial_threshold=depart_axial_threshold,
    )
    return ((stage >= HANDOVER_STAGE_PASSED_TO_LEFT) & away).unsqueeze(-1).float()


def _handover_tick(
    env,
    min_height: float,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    left_ee_frame_cfg: SceneEntityCfg,
    gripper_open_val: float,
    gripper_threshold: float,
    diff_threshold: float,
) -> torch.Tensor:
    """Advance and return the hand-over stage counter. At most one advance per environment step.

    Both the success condition and the ``handover`` subtask signal call this, and both are
    evaluated every step -- so without the guard the per-step bookkeeping (notably the put-down latch) would be applied twice per step.
    """
    memory = _handover_memory(env)
    stage = handover_stage(env)
    step_id = int(getattr(env, "common_step_counter", -1))
    if step_id != -1 and step_id == memory["last_step"]:
        return stage  # already advanced this step by the other caller
    memory["last_step"] = step_id

    obj: RigidObject = env.scene[object_cfg.name]

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

    memory = _handover_memory(env)

    # 0 -> 1: the giving hand has the can, and it gets off the pad while holding it. The lift is
    # checked HERE, over the whole time the right arm holds the can, rather than at the moment of
    # the pass: a human hands an object over by raising it and then bringing it back down to meet
    # the other hand, so demanding height at the instant of the pass rejects the natural motion.
    # Measured on real demos, the can peaks at 0.416-0.427 while the right arm carries it and is
    # back down at 0.388-0.398 by the time both hands are on it.
    lifted_now = obj.data.root_pos_w[:, 2] > min_height
    memory["was_lifted"] |= right_holds & lifted_now
    stage[(stage == HANDOVER_STAGE_WAITING) & right_holds] = HANDOVER_STAGE_RIGHT_HELD

    # 1 -> 2: the receiving hand takes it while the giving hand still has it -- or has had it,
    # continuously, since picking it up. That "or" is what makes this survive real demos: the two
    # grasps are DETECTED together for only 0-4 steps, in 3 of 10 measured episodes for zero steps,
    # and in demo_8 detection of the right grasp drops out for 19 consecutive steps while the arm
    # is demonstrably still holding the can 4 cm above the pad. Requiring a detected overlap, or
    # any fixed-length grace window, rejects hand-overs that plainly happen on video.
    #
    # The latch is cleared only by the can being PUT DOWN -- back near resting height with the
    # right hand not on it. That is the real difference between a pass and the case this guards
    # against ("right sets it down, left picks it up later"): a pass keeps the can in the air the
    # whole way across. Height separates them; elapsed time does not.
    put_down = ~right_holds & (obj.data.root_pos_w[:, 2] <= _object_rest_z(env) + HANDOVER_PUT_DOWN_MARGIN)
    memory["right_holding_since_pickup"] |= right_holds
    memory["right_holding_since_pickup"] &= ~put_down
    passed = (
        (stage == HANDOVER_STAGE_RIGHT_HELD)
        & left_holds
        & memory["right_holding_since_pickup"]
        & memory["was_lifted"]
    )
    stage[passed] = HANDOVER_STAGE_PASSED_TO_LEFT

    # 2 -> 3: the giving hand has let go, leaving the can with the receiving hand. THIS is the
    # hand-over being complete, and it is where success fires.
    #
    # What the can does afterwards is deliberately not required. An earlier version demanded it be
    # released by both hands and back down near its resting height, to match "and then it is put
    # down" -- but that tail is often not in the recording at all: with manual saving the operator
    # stops the episode while still holding the can, and 3 of 10 measured demos end with it at
    # 0.408-0.413, mid-air. Gating success on it discards good hand-overs for want of a coda.
    handed_over = (stage == HANDOVER_STAGE_PASSED_TO_LEFT) & ~right_holds
    stage[handed_over] = HANDOVER_STAGE_DONE

    return stage


# ── Object spawn region ───────────────────────────────────────────────────────
# Positive y is the robot's LEFT-arm side. Coordinates here are absolute (env-local metres), not
# the offsets off a default pose that PickUpEventCfg.randomize_cube_2 uses -- see
# :func:`reset_object_free`.
#
# Modes still do NOT restrict the can to "their" half: which arm the demo is about is already
# pinned by only that arm being drivable (:data:`CONTROLLED_ARMS`) and only that arm satisfying the
# success condition. What the per-mode ranges below encode is something else -- how far the demos
# for that mode reach.
_OBJECT_X_RANGE = (0.20, 0.46)
"""m -- widest forward band the pad geometry allows. The near bound is the base task's (>=8 cm
clearance from the robot origin, so the gripper cannot clip the base while reaching); the far bound
keeps the can's far side ~2 cm inside the pad's x=0.51 edge.

This is a bound on the PAD, not on any arm's reach or on any recorded demo -- see
:data:`OBJECT_SPAWN_RANGES` before using it for a mode whose demos feed Mimic."""

_OBJECT_Y_RANGE = (-0.27, 0.27)
"""m -- full pad width, both arms' halves AND the middle, keeping the base task's ~1.5 cm margin
from the pad's y=+-0.285 half-width. Same caveat as :data:`_OBJECT_X_RANGE`."""

_HANDOVER_SOURCE_X_RANGE = (0.2374, 0.3751)
_HANDOVER_SOURCE_Y_RANGE = (-0.0592, 0.0120)
"""m -- the region the recorded hand-over demos actually put the can in, measured from their reset
poses rather than chosen. NOT what generation samples (that is :data:`_HANDOVER_X_RANGE` /
:data:`_HANDOVER_Y_RANGE`, a subset) -- this is the record of what the source data covers, kept
separately so the subset relationship below is checkable and so the measured fact survives any
retuning of the sampled range.

This is the INTERSECTION over every annotated demo set that is still in use, not any one of them.
Read straight off ``initial_state/rigid_object/can/root_pose`` in each file:

    logs/demos/pickup_pringle_annotated_v4.hdf5   x [0.224656, 0.418972]  y [-0.075389, 0.014304]
    logs/demos/pickup_pringle_annotated_v5.hdf5   x [0.237397, 0.375137]  y [-0.059279, 0.012062]
    intersection, ROUNDED INWARD (this constant)  x [0.2374,   0.3751  ]  y [-0.0592,   0.0120  ]

Two ten-demo sets recorded by hand do NOT cover the same box, and v5's is the tighter one on all
four sides. Taking the intersection is what makes the sampled range below safe whichever file is
passed to ``generate_dataset.py --input_file``; taking either set alone silently extrapolates when
the other is used. Re-measure and re-intersect when a demo set is added or retired -- an entry left
here for a file no longer used only costs range, but a missing entry costs correctness.

Rounded INWARD (ceil the lower bound, floor the upper) rather than to nearest, so the constant is a
subset of the real data at full precision. Rounding to nearest puts the bound a fraction of a
millimetre OUTSIDE the measured extreme, which is harmless physically but makes the subset assertion
below a lie -- and that assertion is the only thing standing between a future edit and silent
extrapolation.

Why the source demos cluster this tightly rather than filling the pad: it is what the GIVING (right)
arm can reach with a hand-over's posture. The right arm has to grasp the can low, raise it and
present it across the body, which costs reach that a plain pick-up does not -- so the operator's
demos land in a ~14 x 7 cm patch slightly right of centre.

These are the measured bounds verbatim, with no margin added, so they can be re-derived from the
datasets and so it stays obvious that they describe the source data, not a preference.

To cover MORE of the pad the order is: widen the sampled range below, record a new demo set against
the wider range, then set both this constant and the sampled one to what that set actually turned
out to cover. Widening the sampled range WITHOUT recording is the failure this pair exists to
prevent -- see :data:`_HANDOVER_X_RANGE` for the invariant that catches it."""

_HANDOVER_X_RANGE = (0.2374, 0.3751)
_HANDOVER_Y_RANGE = (-0.0120, 0.0120)
"""m -- what hand-over generation actually samples: the source region above, with y re-centred on
the midline between the two hands and shrunk to fit.

Mimic can only re-anchor a source segment onto a new object pose, so the distance from a generated
spawn to its source demo's spawn is the size of the rigid translation applied to every waypoint in
that segment. Two consequences drive this range:

* **It must stay INSIDE the source region.** A spawn outside it is extrapolation, not
  interpolation. (For scale, the full-pad y range is +-0.27 against a source spread of
  [-0.075, 0.014]: it asks Mimic to translate the whole grasp segment up to ~26 cm sideways from
  anything it has seen, which lands the transformed waypoints outside the arm's reach.)
* **Smaller is easier.** Every centimetre of spread is a centimetre of translation error the
  bimanual constraints and the receiving arm's approach have to absorb. NVIDIA's own bimanual
  hand-over (``Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-Mimic-v0``) randomises its object by only
  +-0.01 m in x and y, which is what lets that task get away with 3 subtask segments, one manual
  annotation and no constraints at all.

y is therefore centred on 0 -- the midline between the arms, which park mirror-symmetrically at
y=+-0.153 (see :data:`_ARM_KEEP_OUT_BOXES`) -- so the can sits equidistant from both hands rather
than 2.4 cm into the giving arm's side as the raw source centre (-0.0236) does. Centring costs
width: the source's own upper bound is +0.0120, so the widest y interval that is both centred on 0
and a subset of the source is exactly +-0.0120. That is where this number comes from -- it is a
geometric consequence of "centred AND no extrapolation", not a tuned value, and it happens to land
in the same 2-3 cm band NVIDIA uses.

x is left at the source bounds unchanged. The same centring argument does not bite there: the source
x interval is already centred on its own data, so re-centring would not move it, and there is no
second requirement (nothing analogous to "between the hands") pulling it in. Forward/back spread also
costs the hand-over less than sideways spread -- both arms reach forward alike, whereas a sideways
offset pushes the can toward one arm and out of the other's comfortable reach.

INVARIANT: this range must remain a subset of :data:`_HANDOVER_SOURCE_X_RANGE` /
:data:`_HANDOVER_SOURCE_Y_RANGE` unless a new demo set is recorded first.

Note that narrowing does NOT invalidate the existing annotated dataset: the source demos still
bracket every pose that can now be sampled. Recording, though, goes through :func:`apply_task_mode`
too (record_demos_openarm.py), so a new demo set recorded now would come out inside this narrow band
-- which is fine as long as the two constants above are updated to match it, and wrong if they are
left describing the old, wider set."""

OBJECT_SPAWN_RANGES = {
    # Single-arm modes keep the full-pad range: their own demos have not been re-measured the way
    # the hand-over ones above were, so there is no source-derived range to put here yet. If Mimic
    # generation for these modes also fails at spawns far from the source demos, measure their
    # recordings' reset poses and narrow these the same way -- the reasoning in
    # _HANDOVER_X_RANGE's docstring is not specific to hand-over.
    TASK_MODE_LEFT: (_OBJECT_X_RANGE, _OBJECT_Y_RANGE),
    TASK_MODE_RIGHT: (_OBJECT_X_RANGE, _OBJECT_Y_RANGE),
    TASK_MODE_HANDOVER: (_HANDOVER_X_RANGE, _HANDOVER_Y_RANGE),
}
"""Where each mode may spawn the can, as (x_range, y_range) in env-local metres."""

# The invariant from :data:`_HANDOVER_X_RANGE`, enforced rather than described: a sampled range that
# escapes the source demos' coverage is extrapolation, and its symptom (Mimic translating whole
# segments past the arm's reach) shows up as a low generation success rate rather than as an error,
# so it is worth failing at import instead. Narrowing is always safe; widening needs a new demo set
# and both constants updated together.
assert (
    _HANDOVER_SOURCE_X_RANGE[0] <= _HANDOVER_X_RANGE[0]
    and _HANDOVER_X_RANGE[1] <= _HANDOVER_SOURCE_X_RANGE[1]
    and _HANDOVER_SOURCE_Y_RANGE[0] <= _HANDOVER_Y_RANGE[0]
    and _HANDOVER_Y_RANGE[1] <= _HANDOVER_SOURCE_Y_RANGE[1]
), (
    f"hand-over spawn range x={_HANDOVER_X_RANGE} y={_HANDOVER_Y_RANGE} is not a subset of what the "
    f"source demos cover (x={_HANDOVER_SOURCE_X_RANGE} y={_HANDOVER_SOURCE_Y_RANGE}); record a new "
    "demo set against the wider range and update both pairs of constants."
)


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
#
# Kept for every mode even though the hand-over range (:data:`_HANDOVER_Y_RANGE`, |y| <= 0.0120)
# comes nowhere near them: they guard the rest pose, so they must stay correct for whatever range a
# mode is given, including a wider one restored later. The single-arm modes' full-pad range does
# still reach them, which is what keeps the rejection sampling in :func:`reset_object_free` load
# bearing.
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
    # the whole region; for the full-pad range the legal area is ~70% of the rectangle, so
    # exhausting 100 rounds has probability ~1e-52 per env, and a range that misses the keep-out
    # entirely (the hand-over one does) leaves on the first pass. Falling back to the last
    # (possibly illegal) draw rather than raising keeps a pathological config from killing a
    # recording session mid-run.
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

_SIGNAL_NAMES = (
    "grasp",
    "lift",
    "grasp_left",
    "grasp_right",
    "reach_left",
    "reach_right",
    "presented",
    "handover",
    "departed_right",
)
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

    left  (RECEIVING): 0 stay clear | 1 approach | 2 close and take | 3 carry away
    right (GIVING)   : 0 approach | 1 grasp | 2 lift | 3 raise to the pass point and hold
                       | 4 release | 5 withdraw to rest

    Three SEQUENTIAL constraints (all ``min_time_diff=-1``: the constrained subtask does not start
    at all until its precondition has finished):

    #. ``(right, 2) -> (left, 1)``: the left arm does not set off until the can is UP.
    #. ``(left, 2) -> (right, 4)``: the right hand does not begin releasing until the left hand has
       closed on the can. Its precondition is the CLOSE subtask alone -- which is why the left
       arm's approach and close are separate subtasks.
    #. ``(right, 4) -> (left, 3)``: the left arm does not carry the can away until the right hand
       is off it. Its precondition is the RELEASE subtask alone (ends at ``departed_right``, the
       hand physically clear of the can) -- which is why release and withdraw are separate
       subtasks: keying this on the withdraw instead would make the left arm idle through the
       right arm's entire trip back to rest.

    Constraint #2 makes generation deliberately more conservative than the demos: the operator
    opens the giving hand 0-4 steps after closing the receiving one (demo_8: 1 step BEFORE), so
    the enforced hold-until-taken is a task invariant imposed on generation, not a property of the
    source data. The runtime gate (OpenArmPickUpIKAbsMimicEnv._hold_giving_hand_until_taken) is
    load-bearing here: while (right, 4) waits at its first waypoint, the replayed gripper column
    may already say "open" (in half the demos the jaws open at or before the pass), and the gate
    is what keeps them physically closed until the take has actually happened.

    Segment boundaries vs the constraints they serve
    ------------------------------------------------
    Two of the right arm's boundaries exist purely so a constraint can anchor on them, which is
    why "lift" (2) and "raise-and-hold" (3) are separate subtasks even though they read as one
    motion:

    * ``presented`` (end of 2) fires when the can leaves the pad -- the earliest moment that does
      NOT depend on the left arm. Constraint #1 must anchor there: anchoring it on any later
      right-arm boundary (``handover`` fires when the LEFT hand takes the can) would have the left
      arm waiting for an event that needs the left arm to move first -- a deadlock.
    * ``handover`` (end of 3) is where the release content begins in the source. Constraint #2's
      constrained subtask must START there so that, once released, the jaws open within a few
      steps. Folding segment 3's content into the release subtask instead would have the right arm
      freeze at the 2.5 cm-lift point (where ``presented`` fires) waiting for a left hand that is
      reaching for a can the source says should be up at the pass point -- and, once taken, replay
      20-40 steps of hold before opening, with both position-controlled grippers clamping the can
      the whole time.

    Segment 3 itself carries the raise-and-hold content between those two boundaries. It has no
    constraint of its own and needs none: it plays out while the left arm runs its approach, which
    is exactly the source demos' phasing.

    Why the receiving arm's wait is its OWN subtask, and object-free
    ----------------------------------------------------------------
    An earlier version gave the left arm just two subtasks, the first running from step 0 all the
    way to ``grasp_left`` with ``object_ref=CAN_NAME``. That single segment is ~90 steps on these
    demos of which the first ~60 are the arm sitting still, and Mimic re-anchors a WHOLE segment on
    the object pose at the segment's start, so it was wrong in two ways at once:

    * the idle part was re-anchored too. The left arm's rest pose got rigidly translated by however
      far the can had spawned from the source demo's can, which walked the idle arm into the
      workspace from step 0 -- visible in the failed generations as the left hand arriving at the
      can around step 25 and knocking it over.
    * the approach was anchored to the can's pose ON THE PAD, because that is where the segment
      starts. But the receiving hand does not reach for the can on the pad, it reaches for the can
      held up in the giving hand, 3-26 cm away (median ~13 cm, measured across the ten demos).

    Splitting at ``presented`` fixes both: the wait becomes ``object_ref=None`` so nothing
    transforms it, and the approach segment now starts when the can is already up in the right
    hand, so the pose it is anchored on is the presentation pose.

    Constraint bookkeeping
    ----------------------
    (right, 4) is the constrained half of #2 AND the precondition of #3. DataGenerator routes
    constraint records into one dict per role (latter / former / coordination), so one subtask can
    hold both -- an earlier single shared dict silently dropped one of the two entries, which is
    exactly the failure its per-role asserts now catch loudly. No deadlock in the chain: right-2
    finishes on its own and releases left-1; left-2 finishes and releases right-4; right-4
    finishes and releases left-3.
    """
    from isaaclab.envs.mimic_env_cfg import SubTaskConfig, SubTaskConstraintConfig, SubTaskConstraintType

    # term_signal deliberately untyped: SubTaskConfig annotates it as plain `str` while defaulting
    # it to None, so an honest `str | None` here just moves the type error into the
    # SubTaskConfig(...) call. Same loose style as _idle_subtasks.
    def _subtask(term_signal, description, interp: int = 20, object_ref: str | None = CAN_NAME):
        # 'nearest_neighbor_object' needs an object to measure against, so a segment with no
        # object_ref falls back to 'random' -- same pairing as _idle_subtasks.
        # subtask_term_offset_range is NOT a parameter here: _stagger owns it, see below.
        return SubTaskConfig(
            object_ref=object_ref,  # type: ignore[arg-type]  (annotated `str` upstream, None is valid)
            subtask_term_signal=term_signal,
            selection_strategy="nearest_neighbor_object" if object_ref is not None else "random",
            selection_strategy_kwargs={"nn_k": 10} if object_ref is not None else {},
            action_noise=0.001,
            num_interpolation_steps=interp,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            description=description,
        )

    def _stagger(subtasks: list) -> list:
        """Give one arm's subtasks end offsets 0, 1, 2, ... so tied signals cannot collide.

        DataGenInfoPool cuts each boundary at its signal's first 0->1 transition and rejects the
        whole episode unless the boundaries strictly increase. Two of these signals firing on the
        SAME simulation step is not a bug to be tuned away -- at 20 Hz these events are genuinely
        within one step of each other, and which pairs tie varies per recording session. Measured
        over two independent ten-demo sets, every pair below tied in at least one episode:

            handover / departed_right   3 of 10 (the giving hand is already clear of the can by
                                        the time the machine declares the pass -- no distance
                                        threshold can separate them, it has physically gone)
            presented / reach_left      1 of 10
            grasp_right / presented     1 of 10

        Since ``randomize_subtask_boundaries`` adds these offsets to the END index and takes each
        subtask's START from the previous END, a staircase of 0, 1, 2, ... turns any tie into a
        one-step segment instead of a zero-step one: length_i becomes
        ``raw_end_i - raw_end_i-1 + 1``, which is >= 1 for any NON-DECREASING signal sequence.
        Inversions are still fatal, and are what the signal definitions themselves guard against
        (see receiver_reached_obs); this only has to survive ties.

        The last subtask keeps (0, 0): its end is the episode length, and an offset would index
        past the recorded actions.

        The cost is that no subtask boundary is randomised any more -- an earlier version gave
        ``grasp_right`` a (2, 4) random end offset for generation diversity, which is not
        compatible with a fixed staircase and is not worth re-deriving a per-subtask safe range for
        while ties are still the binding constraint.
        """
        for index, subtask in enumerate(subtasks[:-1]):
            subtask.subtask_term_offset_range = (index, index)
        subtasks[-1].subtask_term_offset_range = (0, 0)
        return subtasks

    # Insertion order stays left-then-right to match the single-arm modes, so the eef ordering a
    # Mimic env sees never depends on the mode -- only the CONTENT differs, the left arm now
    # being the receiving hand.
    subtask_configs = {
        LEFT_EEF: _stagger([
            # Nothing to interpolate towards: an untransformed segment starts at the same rest pose
            # the arm is already sitting in, so this keeps the short idle interpolation.
            _subtask(
                "presented",
                "Stay clear while the right arm picks the can up",
                interp=5,
                object_ref=None,
            ),
            _subtask("reach_left", "Approach the presented can"),
            # Continues straight out of the approach, so there is barely anything to interpolate.
            _subtask("grasp_left", "Close on the can and take it", interp=5),
            # Last subtask, so it runs to the end of the episode: the receiving arm carries the
            # can back to a neutral pose, still holding it. It is not put down -- the demo is over
            # once the can is safely in this hand.
            _subtask(None, "Carry the can back to a neutral pose, still holding it", interp=15),
        ]),
        RIGHT_EEF: _stagger([
            _subtask("reach_right", "Approach the can", interp=50),
            _subtask("grasp_right", "Close on the can", interp=5),
            _subtask("presented", "Lift the can off the pad", interp=10),
            _subtask("handover", "Raise the can to the pass point and hold it there", interp=5),
            _subtask("departed_right", "Open the hand and pull clear of the can", interp=5),
            _subtask(None, "Withdraw to a neutral pose", interp=10, object_ref=None),
        ]),
    }
    constraints = [
        # 1: the left arm does not set off until the right arm has the can off the pad.
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[(RIGHT_EEF, 2), (LEFT_EEF, 1)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
            sequential_min_time_diff=-1,
        ),
        # 2: the right hand does not begin releasing until the left hand has closed on the can.
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[(LEFT_EEF, 2), (RIGHT_EEF, 4)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
            sequential_min_time_diff=-1,
        ),
        # 3: the left arm does not carry the can away until the right hand is clear of it.
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[(RIGHT_EEF, 4), (LEFT_EEF, 3)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
            sequential_min_time_diff=-1,
        ),
    ]
    return subtask_configs, constraints


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
        # Splits the receiving arm's wait from its approach, so the approach is anchored on the can
        # in the giving hand instead of on the pad -- see handover_presented_obs.
        subtask_terms.presented = ObsTerm(
            func=handover_presented_obs, params={"min_height": lift_height, **common()}
        )
        # Splits that approach from the close-and-take, which is what the giving arm's release is
        # constrained against -- see _handover_subtask_configs. Gated on 'presented' rather than
        # being a bare proximity test: the receiving arm's REST pose straddles the grasp radius of
        # a can standing on the pad -- see receiver_reached_obs.
        subtask_terms.reach_left = ObsTerm(
            func=receiver_reached_obs, params={"min_height": lift_height, **common()}
        )
        # The giving arm's own approach/close split. Deliberately NOT the handover radius from
        # common(): that one fires on a transient while the arm is still at rest near a can
        # standing on the pad -- see REACH_RADIAL_THRESHOLD for the measurements. The axial bound
        # stays the normal one; it cannot discriminate here either way.
        subtask_terms.reach_right = ObsTerm(
            func=object_reached_by_obs,
            params={
                "ee_frame_cfg": right_frame,
                "object_cfg": object_cfg(),
                "diff_threshold": REACH_RADIAL_THRESHOLD,
            },
        )
        # Shares handover_success's stage machine rather than testing "both grippers closed right
        # now" -- see handover_passed_obs for why that instantaneous test rejects real hand-overs.
        subtask_terms.handover = ObsTerm(
            func=handover_passed_obs, params={"min_height": lift_height, **common()}
        )
        # Ends the giving arm's release: the pass has happened and the right hand has physically
        # pulled clear of the can -- see departed_after_pass_obs for why this is positional.
        subtask_terms.departed_right = ObsTerm(
            func=departed_after_pass_obs,
            params={"min_height": lift_height, "right_ee_frame_cfg": right_frame, **common()},
        )
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
    # The range is per-mode and passed explicitly rather than left to reset_object_free's defaults,
    # because it has to be the region that mode's SOURCE DEMOS cover -- Mimic re-anchors a source
    # segment onto the new object pose, so a spawn outside that region is extrapolation. See
    # OBJECT_SPAWN_RANGES / _HANDOVER_X_RANGE.
    object_x_range, object_y_range = OBJECT_SPAWN_RANGES[mode]
    env_cfg.events.randomize_cube_2 = None
    env_cfg.events.randomize_object = EventTerm(
        func=reset_object_free,
        mode="reset",
        params={"asset_cfg": object_cfg(), "x_range": object_x_range, "y_range": object_y_range},
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
