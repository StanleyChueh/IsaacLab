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
    lets go while the left keeps hold of it. The demo is DONE there -- the left arm simply carries
    the can back to a neutral pose holding it. It is not put down again, and nothing about where it
    ends up gates success. See :func:`handover_success`, which walks a stage counter through the
    sequence rather than testing any single instant, because no instant distinguishes a hand-over
    from a plain left-arm pick when the can spawns midway between the two hands, in reach of
    either. "Keeps hold of it" is checked and is not a formality: the receiving hand must hold the
    can for a sustained stretch, with an aperture band that a hand shut on empty air fails
    (:func:`receiver_grip_confirmed`), because dropping the can just after the pass is the single
    most common way a GENERATED hand-over goes wrong.

Subtask term signals published per mode -- these are what ``annotate_demos.py`` annotates and
what a Mimic env cfg's ``subtask_term_signal`` entries must reference:

============  ===================================================================
``left``      ``grasp``, ``lift``
``right``     ``grasp``, ``lift``          (same names, right arm -- so one single-arm
                                            Mimic cfg covers either side unchanged)
``handover``  ``grasp_right``               ONE boundary in the whole mode; see
                                            :func:`_handover_subtask_configs`
============  ===================================================================

``handover`` mode publishes two further observations, ``presented`` and ``handover``, which are
NOT subtask boundaries and must not be referenced by any ``subtask_term_signal``. They exist for
OpenArmPickUpIKAbsMimicEnv._hold_giving_hand_until_taken, which reads them at generation time to
keep the giving hand shut until the receiving hand has really taken the can. They are still
recorded by ``annotate_demos.py`` (it records the whole ``subtask_terms`` group), so its auto mode
still requires them to fire -- which is wanted: a demo where neither fires is not a hand-over.

This mode used to publish four more (``reach_right``, ``reach_left``, ``grasp_left``,
``departed_right``) and to use ``presented``/``handover`` as boundaries too, cutting the episode
into ten segments held together by three constraints. Cutting it that finely is what broke
generation rather than what made it work -- see :func:`_handover_subtask_configs` for the
measurements and for the one-transform invariant that replaced the lot.

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
roughly a factor of two, and only the recorded value is what any annotation actually sees.

Note what this test canNOT tell you: a hand that closed on NOTHING passes it, because jaws shut on
air travel further than jaws shut on a can. That is why the hand-over's completion uses the
two-sided :data:`HANDOVER_RECEIVER_APERTURE_RANGE` instead of this one -- see
:func:`receiver_grip_confirmed`."""

HANDOVER_RECEIVER_APERTURE_RANGE = (0.015, 0.055)
"""(min, max) total jaw aperture (m, the two finger joints summed) at which the RECEIVING hand
counts as having actually got hold of the can. Fully open is 0.088; fully shut is ~0.

Two-sided on purpose, and that is the whole point of it. :func:`_gripper_is_closed` only asks
whether the jaws left the open position, which a hand that closed on empty air passes just as
well as one holding the can -- and empty air is the dominant failure of generated hand-overs, not
a rare one. An aperture that settles near HALF of open means something roughly can-width is
wedged between the jaws; an aperture near zero means they met, i.e. the can is not there.

Measured, over the frames after the pass:

* real teleop hand-overs (10 demos, logs/demos/pickup_pringle_annotated_V6.hdf5): 0.0441-0.0538,
  i.e. the jaws stop just past half-closed, held there for the rest of the episode.
* Mimic-generated episodes that visibly end with the can in the left hand: 0.0447-0.0537.
* generated episodes whose left hand shut on nothing: 0.0000-0.0054.
* generated episodes whose left hand never really closed: 0.0651-0.0789.

So the two failure modes sit on OPPOSITE sides of the real grasps, and both bounds are needed.
0.055 clears the widest real grasp (0.0538) and still rejects the 0.0651 near-miss; 0.015 is well
clear of the 0.0267 low outlier above and of the ~0.005 shut-on-air band below.

The can's diameter is what pins this: post-scale it is ~60 mm (:data:`CAN_SCALE`), so the jaws
physically cannot go much past half-closed while it is between them. Do not "tighten" the upper
bound to exactly half (0.044) on the theory that a firm grip closes further -- measured, that
rejects 10 real hand-overs out of 10."""

HANDOVER_RECEIVER_HOLD_STEPS = 20
"""How many CONSECUTIVE steps the receiving hand must satisfy :func:`receiver_grip_confirmed`,
with the giving hand already off the can, before the hand-over counts as complete.

A single-frame check is not enough, because Mimic's per-episode success is an OR over every step
(``data_generator.generate``): one frame that happens to look like a grasp permanently marks the
episode successful, and the can dropping immediately afterwards cannot undo it. Measured on the 20
episodes of logs/demos/pickup_generated_V6.hdf5 -- all 20 marked successful by the old rule -- the
longest run of confirmed grip after the pass separates cleanly:

    real teleop demos          27-41 steps (and running when the recording stops)
    generated, can kept        30, 52, 79, 79 steps
    generated, can dropped     0-14 steps

20 sits in that gap. Raising it past ~25 starts eating into the real demos, whose recordings end
while the arm is still carrying the can."""

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


def gripper_aperture(
    env,
    gripper_joint_names: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(N,) how far open the gripper is: its jaw positions summed.

    Both jaws contribute because the object is generally NOT centred between them -- it pushes one
    jaw wide and lets the other come in close, so either jaw read alone says more about where the
    can sits in the hand than about how hard the hand is shut. Measured on a real left-hand grasp
    the two jaws sit at ~0.030 and ~0.014 while their sum stays at 0.044-0.049 across every demo.

    What :data:`HANDOVER_RECEIVER_APERTURE_RANGE` thresholds; see :func:`receiver_grip_confirmed`.
    """
    return gripper_jaw_positions(env, gripper_joint_names, robot_cfg).sum(dim=1)


def receiver_grip_confirmed(
    env,
    ee_frame_cfg: SceneEntityCfg,
    gripper_joint_names: list[str],
    object_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    diff_threshold: float = EEF_TO_CAN_THRESHOLD,
    axial_threshold: float = GRASP_AXIAL_TOLERANCE,
    aperture_range: tuple[float, float] = HANDOVER_RECEIVER_APERTURE_RANGE,
) -> torch.Tensor:
    """(N,) bool: the hand has the object BETWEEN its jaws, not merely near it and not shut on air.

    Stricter than :func:`object_grasped_by`, which it is otherwise shaped like: the jaw test is the
    two-sided aperture band rather than "has travelled away from open". The upper bound is the
    grasp ("the jaws did close on something"); the lower bound is the part
    :func:`object_grasped_by` cannot express ("...and that something exists"). See
    :data:`HANDOVER_RECEIVER_APERTURE_RANGE` for the measurements behind both.

    Used for the receiving hand at the end of a hand-over, where the distinction decides whether
    the episode is a demonstration of a completed pass or of the can hitting the floor.
    """
    aperture = gripper_aperture(env, gripper_joint_names, robot_cfg)
    min_aperture, max_aperture = aperture_range
    return (
        hand_on_object(env, ee_frame_cfg, object_cfg, diff_threshold, axial_threshold)
        & (aperture <= max_aperture)
        & (aperture >= min_aperture)
    )


def hand_on_object(
    env,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg(CAN_NAME),
    diff_threshold: float = EEF_TO_CAN_THRESHOLD,
    axial_threshold: float = GRASP_AXIAL_TOLERANCE,
) -> torch.Tensor:
    """(N,) bool: the hand behind *ee_frame_cfg* is ON the object, whatever its gripper is doing.

    The positional half of :func:`object_grasped_by`, factored out because the hand-over needs the
    two halves separately: :func:`receiver_grip_confirmed` pairs this same "the hand is on the can"
    test with an aperture BAND rather than with "the jaws left the open position", which is what
    lets it tell a hand holding the can from a hand shut on empty air.
    """
    radial, axial = hand_to_object_offsets(env, ee_frame_cfg, object_cfg)
    return torch.logical_and(radial < diff_threshold, axial < axial_threshold)


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
    """The state the hand-over's transitions need besides the two hands' current grasps.

    ``was_lifted``                 the giving arm has held the can above the lift height.
    ``right_holding_since_pickup`` it has not put the can back down since picking it up.
    ``receiver_grip_steps``        consecutive steps the receiving hand has been confirmed holding
                                   the can with the giving hand off it -- the 2->3 counter, see
                                   :data:`HANDOVER_RECEIVER_HOLD_STEPS`.

    Read-only view, exposed for the same reason :func:`handover_stage` is: a hand-over that stalls
    at stage 1 or 2 should say WHICH precondition is missing. Advancing them is
    :func:`_handover_tick`'s job -- calling this does not tick the machine.
    """
    memory = _handover_memory(env)
    return {
        "was_lifted": memory["was_lifted"],
        "right_holding_since_pickup": memory["right_holding_since_pickup"],
        "receiver_grip_steps": memory["receiver_grip_steps"],
    }


def _handover_memory(env) -> dict:
    """Per-env buffers the hand-over stage machine carries between steps, created on first use."""
    mem = getattr(env, _HANDOVER_MEMORY_ATTR, None)
    if mem is None or mem["was_lifted"].shape[0] != env.num_envs:
        mem = {
            "was_lifted": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "right_holding_since_pickup": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "receiver_grip_steps": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
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
    memory["receiver_grip_steps"][env_ids] = 0


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
    :data:`HANDOVER_STAGE_DONE`               the right arm has let go AND the left arm is
                                              confirmed still holding the can, for
                                              :data:`HANDOVER_RECEIVER_HOLD_STEPS` steps
                                              running: the can is the left arm's, and the
                                              task is complete
    ========================================  ==========================================

    "Confirmed holding" is :func:`receiver_grip_confirmed`, not the ordinary grasp test: the jaw
    aperture has to sit in a BAND around half-closed, so a hand that shut on empty air fails it.
    That is the difference between an episode that hands the can over and one that drops it, and
    on generated episodes it is the common case rather than the corner case -- see the 2->3
    comment in :func:`_handover_tick` for the measurements.

    This function is TRUE ONLY WHILE THAT GRIP HOLDS: reaching stage 3 is necessary but not
    sufficient, and letting go afterwards makes it False again. Whether an episode that lets go at
    the very end is nevertheless exported as successful is then a question about the CALLER, not
    about this function -- Mimic ORs its per-step success over the whole episode, so see
    ``DataGenConfig.generation_success_at_final_state``, which the OpenArm Mimic cfg turns on
    precisely so this "...and still has it" survives to the exported label.

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
    # NOT simply `stage >= DONE`. The stage counter latches -- it has to, so that a detection
    # dropout mid-pass cannot walk the episode backwards -- but that makes it a record of what has
    # happened, and "the hand-over happened at some point" is not the same claim as "the robot has
    # the can". A latched-only success is true forever after the pass, including in the frames
    # where the receiving hand has opened and the can is on its way to the floor.
    #
    # So the live half is re-asserted here: the receiving hand must be holding the can AT THE
    # MOMENT THIS IS ASKED. The counter carries that for free -- _handover_tick zeroes it on any
    # step the grip is not confirmed -- so this reads "the pass completed, and the grip that
    # completed it has not been let go since".
    return (stage >= HANDOVER_STAGE_DONE) & (
        _handover_memory(env)["receiver_grip_steps"] >= HANDOVER_RECEIVER_HOLD_STEPS
    )


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
    """(N, 1) float: the pass itself has happened -- the ``handover`` observation.

    NOT a subtask boundary (see :func:`_handover_subtask_configs`). It is the releasing half of
    OpenArmPickUpIKAbsMimicEnv._hold_giving_hand_until_taken: that gate clamps the giving hand shut
    until this reads 1, so with no task constraints left it is the only thing ordering the two
    hands' jaws -- and "the receiving hand shut on empty air and the can fell" is the single most
    common way a generated hand-over fails (40 of the 69 failures in
    logs/demos/pickup_pringle_V7_generated_failed.hdf5).

    Reads the SAME stage machine :func:`handover_success` does, deliberately. The signal and the
    success condition have to agree about what a hand-over is, and when they disagree the failure
    is quiet and confusing: an earlier version marked this signal with a plain "both grippers
    closed on the can in this very step" test while success used the staged one, so on real demos
    success fired and the signal never did, and annotate_demos.py rejected the episodes with
    "Did not detect completion for the subtask handover" -- for demos whose hand-over is plainly
    visible. The two grasps are detected together for only 0-4 steps, and in 3 of 10 measured
    episodes for zero steps, because an operator opens the giving hand as the receiving hand
    closes.

    Latching (it stays 1 once the pass has happened) is load-bearing for the gate: the raw per-step
    grasp detection flickers badly right after contact, and an unlatched gate would let the giving
    hand re-close mid-release.
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
    """(N, 1) float: the giving hand has the can UP -- the ``presented`` observation.

    NOT a subtask boundary. It used to end the receiving arm's wait, which is exactly the split
    that broke generation: it fires when the can clears the pad by *min_height*, a median 0.142 m
    from where the can actually is at the take, and the receiving arm's whole approach was rigidly
    anchored on it. See :func:`_handover_subtask_configs`.

    What it is for now is the arming half of
    OpenArmPickUpIKAbsMimicEnv._hold_giving_hand_until_taken: until the giving hand has the can up
    there is nothing to drop, so that gate stays out of the way entirely and a trial whose right
    arm never picked the can up is not left with a hand welded shut.

    Reads the same stage machine as :func:`handover_success` and :func:`handover_passed_obs`, for
    the same reason they share it: three descriptions of one hand-over that can disagree is three
    ways to be wrong about the same episode. Concretely it is the machine's "the right arm has held
    the can and raised it past *min_height*" latch, i.e. it fires strictly after ``grasp_right``
    and strictly before ``handover``. Measured over the 20 demos of
    logs/demos/pickup_pringles_VR_V7.hdf5 it lands 4-12 steps after ``grasp_right`` and 60-90 steps
    before the take.

    Latches: it describes something that has happened, not something that is currently true, so a
    momentary loss of grasp detection while the arm carries the can must not un-fire it and let the
    gate disarm mid-hand-over.
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

    # 2 -> 3: the giving hand has let go AND the receiving hand is demonstrably still holding the
    # can, for long enough that it is not a passing frame. THIS is the hand-over being complete,
    # and it is where success fires.
    #
    # The receiving-hand check is not redundant with stage 2. Stage 2 fires at the INSTANT the two
    # grasps overlap and then latches, so on its own "the right hand let go" completes the
    # hand-over whether the can went to the left hand or to the floor -- and going to the floor is
    # what generated episodes mostly do. Measured on logs/demos/pickup_generated_V6.hdf5, all 20
    # episodes were marked successful under that rule while only 4 of them end with the can in the
    # left hand; 11 end with the left jaws shut on nothing at all. Hence both halves here:
    #   * receiver_grip_confirmed -- an object of about the can's width is between the jaws, not
    #     merely near them and not absent (the two-sided aperture band, which is the part a plain
    #     "jaws are closed" test cannot express),
    #   * for HANDOVER_RECEIVER_HOLD_STEPS consecutive steps -- because Mimic ORs success over
    #     every step of the episode, so one lucky frame is permanent and a later drop cannot
    #     retract it.
    #
    # What the can does after that is still deliberately not required. An earlier version demanded
    # it be released by both hands and back down near its resting height, to match "and then it is
    # put down" -- but that tail is often not in the recording at all: with manual saving the
    # operator stops the episode while still holding the can, and 3 of 10 measured demos end with
    # it at 0.408-0.413, mid-air. Gating success on it discards good hand-overs for want of a coda.
    receiver_firm = receiver_grip_confirmed(
        env,
        ee_frame_cfg=left_ee_frame_cfg,
        gripper_joint_names=LEFT_FINGER_JOINTS,
        object_cfg=object_cfg,
        robot_cfg=robot_cfg,
        diff_threshold=diff_threshold,
    )
    carrying = (stage >= HANDOVER_STAGE_PASSED_TO_LEFT) & receiver_firm & ~right_holds
    grip_steps = memory["receiver_grip_steps"]
    grip_steps += 1
    grip_steps[~carrying] = 0
    handed_over = (stage == HANDOVER_STAGE_PASSED_TO_LEFT) & (grip_steps >= HANDOVER_RECEIVER_HOLD_STEPS)
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
modes on one cfg can't leave a previous mode's signal behind for the annotator to wait on.

``grasp_left`` / ``reach_left`` / ``reach_right`` / ``departed_right`` are no mode's signal any
more (see :func:`_handover_subtask_configs`) and are kept here ON PURPOSE: this tuple's job is to
clear stale attributes off a cfg, and a name dropped from it stops being cleared. A task cfg that
declares one of them in its own ``subtask_terms`` group would otherwise survive apply_task_mode()
and be recorded as a signal that never fires, which annotate_demos.py's auto mode rejects every
episode over."""

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

    left  (RECEIVING): 0 the whole episode -- wait, take the can, carry it away
    right (GIVING)   : 0 approach and grasp | 1 present, release and withdraw

    One subtask term signal in the whole mode (``grasp_right``), and no constraints. That is
    deliberately the shape of NVIDIA's own bimanual hand-over
    (``Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-Mimic-v0``: 2 right subtasks, 1 left, 0 constraints),
    arrived at here from the other direction -- by measuring what the previous ten-segment
    structure actually did to generated episodes.

    The one-transform invariant
    ---------------------------
    Mimic transforms each segment by a rigid ``T = object_pose_generated @ object_pose_source^-1``
    taken at that segment's START (see ``DataGenerator.generate_trajectory``), so:

        the number of subtasks IS the number of independent coordinate transforms, and any
        difference between the two arms' transforms is the error at the hand-over.

    Two hands can only meet if they are under the SAME transform. That is what this structure
    guarantees and what the previous one could not:

    * both arms already draw from ONE source demo (``generation_select_src_per_arm`` defaults to
      False and the selection is made at the first subtask), so their relative geometry is
      whatever the operator actually did in that episode;
    * the left arm is a single ``object_ref=None`` segment, i.e. transform = identity, replayed
      verbatim in world coordinates. Its first waypoint is its own rest pose, so there is nothing
      to interpolate towards and no seam anywhere in the episode;
    * the right arm's segment 1 is ``object_ref=None`` too, which is the load-bearing part. After
      the grasp the can is IN that hand, so its pose is no longer an independent variable -- it is
      the right arm's own execution. Re-anchoring on it feeds that arm's error back in as a
      transform AND moves the pass point away from where the untransformed left hand is waiting.
      Returning to identity instead puts the pass point exactly where the source demo's was, which
      is demonstrably where that same demo's left hand goes.

    So exactly ONE segment in the episode is transformed: the right arm's approach-and-grasp, the
    only part that genuinely has to adapt to a randomly spawned can. The seam at ``grasp_right``
    costs a ``|T|``-sized drag of the just-grasped can back onto the source trajectory, which is
    why ``nn_k`` is 3 rather than 10 -- measured over the 20 spawns of
    logs/demos/pickup_pringles_VR_V7.hdf5 against the sampled spawn range:

        nn_k=10   |T| median 2.0 cm, p90 4.7 cm, max 7.3 cm
        nn_k=3    |T| median 1.0 cm, p90 2.0 cm, max 3.5 cm
        nn_k=1    |T| median 0.7 cm, p90 1.4 cm, max 2.2 cm

    At nn_k=3 the drag is ~1 cm spread over the seam's interpolation, i.e. a millimetre per step.
    Note this is free: the spawn range is unchanged, the 19 source demos simply cover it densely
    enough that the nearest few are close.

    What the ten-segment structure measured
    ---------------------------------------
    The previous version cut the left arm at ``presented`` (wait | approach | take | carry) and the
    right arm five ways, with three SEQUENTIAL constraints, on the theory that each boundary let
    Mimic re-anchor closer to the truth. Measured on logs/demos/pickup_pringle_V7_generated*.hdf5
    (30 successes out of 99 trials) it did the opposite:

    * at the instant the left jaws start to close, the hand is 0.031-0.046 m radial from the can in
      all 20 SOURCE demos, but a median 0.073 m (p75 0.171) in the failed generations -- only 43%
      of them are inside the grasp cylinder at all;
    * 40 of the 69 failures end with the left aperture at ~0.0000, i.e. the receiving hand closed
      on empty air and the can fell;
    * the receiving arm's approach was anchored at ``presented``, which fires when the can clears
      the pad by 2.5 cm -- a median 0.142 m (0.083-0.193) from where the can actually is at the
      take. Every one of those centimetres was open-loop error that the left hand never corrected,
      because its whole segment is one rigid transform of the source.

    Splitting more finely cannot fix that: each new boundary is another independent transform, i.e.
    another chance for the two arms to disagree.

    Synchronisation without constraints
    -----------------------------------
    With no constraints the arms stay in step only because they replay one source demo whose
    phasing already worked. The one thing that can break it is the extra interpolation the right
    arm spends at its single seam and the left arm does not -- hence ``interp=5`` everywhere, so
    the drift is ~5 steps.

    That margin is thin, and measured rather than assumed: across the 20 demos of
    logs/demos/pickup_pringles_VR_V7.hdf5 the can settles at the pass point a median of 1 step
    AFTER the receiving hand has already arrived (range -26 to +24). The operator does a moving
    hand-off and never dwells. Recording with the giving hand deliberately holding still at the
    pass point for ~20 steps before the other hand closes would turn that into real slack, and is
    the cheapest thing that can be done for this task at record time.

    If a future recording still proves too tight, the minimal repair is NOT to go back to ten
    segments: split the left arm into wait (``object_ref=None``) and take-and-carry
    (``object_ref=None``) and add ONE SEQUENTIAL constraint ``(right, 0) -> (left, 1)``, so the
    receiving arm cannot set off before the can has been grasped. Both halves stay untransformed,
    so the invariant above survives while the two arms get ordered explicitly.

    The giving hand's JAWS are ordered by
    OpenArmPickUpIKAbsMimicEnv._hold_giving_hand_until_taken, not by anything here: it reads the
    ``presented`` / ``handover`` observations and clamps the right gripper shut until the receiving
    hand has really taken the can. It used to be the runtime half of the old constraint #2; with
    the constraints gone it is the ONLY thing between a replayed "open" command and the can hitting
    the floor, which is why those two observations stay published even though nothing references
    them as a subtask boundary any more.
    """
    from isaaclab.envs.mimic_env_cfg import SubTaskConfig

    # term_signal deliberately untyped: SubTaskConfig annotates it as plain `str` while defaulting
    # it to None, so an honest `str | None` here just moves the type error into the
    # SubTaskConfig(...) call. Same loose style as _idle_subtasks.
    def _subtask(term_signal, description, interp: int = 5, object_ref: str | None = None):
        # 'nearest_neighbor_object' needs an object to measure against, so a segment with no
        # object_ref falls back to 'random' -- same pairing as _idle_subtasks. 'random' does not
        # mean this segment picks its own demo: with generation_select_src_per_arm False the demo
        # is already fixed by whichever subtask selected first, and only the first subtask of each
        # arm selects at all (generation_select_src_per_subtask is False too).
        return SubTaskConfig(
            object_ref=object_ref,  # type: ignore[arg-type]  (annotated `str` upstream, None is valid)
            subtask_term_signal=term_signal,
            # No staircase of end offsets any more. _stagger() existed because ten boundaries meant
            # several pairs of signals could land on the same simulation step, and DataGenInfoPool
            # rejects an episode whose boundaries are not strictly increasing. With one boundary in
            # the whole mode there is nothing left to tie with.
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object" if object_ref is not None else "random",
            # 3, not 10: |T| is now the episode's ONLY transform, so halving it is worth more than
            # the source diversity it costs. See the table in this function's docstring.
            selection_strategy_kwargs={"nn_k": 3} if object_ref is not None else {},
            action_noise=0.001,
            num_interpolation_steps=interp,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            description=description,
        )

    # Insertion order stays left-then-right to match the single-arm modes, so the eef ordering a
    # Mimic env sees never depends on the mode -- only the CONTENT differs, the left arm here
    # being the receiving hand.
    subtask_configs = {
        # One segment, untransformed, from step 0 to the end of the episode: the arm waits, takes
        # the can where the source demo's giving hand presents it, and carries it away. There is no
        # boundary to place because there is nothing this arm has to adapt to -- see the
        # one-transform invariant above.
        LEFT_EEF: [
            _subtask(None, "Wait, take the presented can and carry it away"),
        ],
        RIGHT_EEF: [
            # The only transformed segment in the mode: the can is wherever it spawned.
            _subtask("grasp_right", "Reach and grasp the can", object_ref=CAN_NAME),
            # Untransformed, so it brings the can back onto the source trajectory and hands it over
            # at the source's pass point. object_ref=CAN_NAME here would re-anchor on a pose this
            # very arm is producing and walk the pass point away from the waiting left hand.
            _subtask(None, "Present the can, release it and withdraw"),
        ],
    }
    return subtask_configs, []


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
        # The mode's ONLY subtask boundary: it ends the giving arm's approach-and-grasp, the one
        # segment that has to adapt to where the can spawned. Everything after it is replayed
        # untransformed by both arms -- see _handover_subtask_configs.
        subtask_terms.grasp_right = ObsTerm(
            func=object_grasped_by_obs,
            params={"ee_frame_cfg": right_frame, "gripper_joint_names": RIGHT_FINGER_JOINTS, **common()},
        )
        # NOT subtask boundaries. These two feed
        # OpenArmPickUpIKAbsMimicEnv._hold_giving_hand_until_taken at generation time, which clamps
        # the right gripper shut until the receiving hand has really taken the can -- with no
        # constraints left, that gate is the only thing ordering the two hands' jaws. Both read the
        # same stage machine as the success condition, so there is one definition of "the pass has
        # happened" in play; see handover_passed_obs for why an instantaneous "both grippers
        # closed" test does not work.
        #
        # They are still recorded (annotate_demos.py records the whole subtask_terms group) and its
        # auto mode still requires them to fire, which is wanted: an episode where neither fires is
        # not a hand-over. They must not be named by any subtask_term_signal.
        subtask_terms.presented = ObsTerm(
            func=handover_presented_obs, params={"min_height": lift_height, **common()}
        )
        subtask_terms.handover = ObsTerm(
            func=handover_passed_obs, params={"min_height": lift_height, **common()}
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
