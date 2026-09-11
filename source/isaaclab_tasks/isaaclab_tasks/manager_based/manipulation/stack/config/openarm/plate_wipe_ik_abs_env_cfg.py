# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm plate-wiping task: right arm picks the plate off a dish rack, left arm picks up a
rag and wipes it, left arm drops the rag, right arm returns the plate to the rack.

Action space (14D flat) -- identical layout to the pick-up task, inherited unchanged:
  [0:6]   left arm IK delta pose  (dx dy dz drx dry drz)
  [6]     left gripper command    (±1.0)
  [7:13]  right arm IK delta pose (dx dy dz drx dry drz)
  [13]    right gripper command   (±1.0)

Scene: cube_1/cube_2/cube_3 are all removed. Three new props replace them:
  * ``dish_rack``  -- dynamic RigidObject, kinematic (PhysxRigidBodyAPI with kinematicEnabled=True
    on dish_rack_kinematic.usdc's ``/root/dish_rack`` -- see this module's docstring below on why
    it needed a RigidBodyAPI at all). Sits toward the pad's FAR edge, centered in y between the
    two arms (see RACK_POS above) -- NOT tucked against the right arm's base the way it originally
    was, and NOT in the more central position tried after that either: both put it close enough
    for the arm's own links to clip it during normal motion (see the docstring for
    ``randomize_dish_rack_and_plate`` below).
    dish_rack_kinematic.usdc's 7 collision sub-meshes were ALSO reauthored from
    PhysxTriangleMeshCollisionAPI to PhysxSDFMeshCollisionAPI (see the plate_wiping asset-fix
    conversation) -- raw triangle-mesh collision only gives reliable two-way solid contact for
    STATIC colliders; once RigidBodyAPI made dish_rack kinematic (needed so it can be repositioned
    at all -- see below), the plate started visibly penetrating the rack's pegs instead of resting
    against them. SDF mesh collision is PhysX's documented fix for exactly this (dynamic/kinematic
    non-convex mesh colliders).
  * ``plate``      -- dynamic RigidObject, spawned resting/leaning in the rack. Its lean is no
    longer a single hand-measured, always-identical pose -- see ``randomize_dish_rack_and_plate``
    below for how it's now dropped into the rack fresh each reset. PLATE_REST_POS/PLATE_REST_ROT
    above are the nominal (unrotated, ungeneralized) pose that reset drops FROM; the position was
    originally measured by actually dropping the plate onto the rack under gravity in a standalone
    headless sim and reading back where it settled (see the plate_wiping asset-fix conversation --
    same "drop it and read the rest pose" methodology this file's sibling configs use for
    CUBE_Z/CAN_HALF_HEIGHT/etc), then re-expressed as RACK_TO_PLATE_OFFSET from RACK_POS so it
    transfers correctly now that RACK_POS itself moved and randomizes.
  * ``rag``        -- dynamic DeformableObject (PhysX FEM soft body, not a rigid box -- a rigid
    collider can't bend, drape over the plate's edge, or conform to a surface while wiping),
    spawned flat on the pad's left side (positive y, left-arm reach). blue_rag_deformable.usdc
    was built by taking the original blue_rag.usdc render mesh (a 0.3x0.3m, ~2mm-thick
    displaced plane) and applying PhysX's PhysxDeformableBodyAPI + a soft, terrycloth-like
    material to it directly (see the plate_wiping asset-fix conversation); the old rigid
    version's separate 3mm collision-proxy cube is gone since the deformable mesh provides its
    own collision. Its resting height below was measured the same "drop it and read the rest
    pose" way, against this new asset. Self-collision is ALSO enabled on it (selfCollision=True,
    selfCollisionFilterDistance=3mm) -- needed for ``randomize_rag_drop`` below to be able to
    produce a lastingly creased/crumpled landing instead of one that relaxes back flat.

Per-episode randomization (see ``events`` below, added on top of ``PickUpEventCfg``):
  * ``dish_rack`` and ``plate`` move as ONE rigid group every reset (``randomize_dish_rack_and_plate``,
    a task-local event function defined in this file): a shared xy offset AND a shared yaw
    rotation about the rack's own pivot -- the plate rests leaning IN the rack, so randomizing
    either independently would either float the plate away from the rack or clip it through a
    wall, and a shared yaw needs the plate's TARGET position (not just its orientation) rotated
    along with the rack's, not translated in a fixed direction regardless of the rack's new
    heading. dish_rack.usdc originally had no RigidBodyAPI at all (a purely static, non-physics
    prop); IsaacLab's episode-reset event API only knows how to reposition
    RigidObject/Articulation/DeformableObject entities (``write_root_pose_to_sim`` /
    ``write_nodal_state_to_sim``), not a bare AssetBaseCfg, so the rack was promoted to a
    *kinematic* RigidObject (dish_rack_kinematic.usdc) purely so it can be told where to sit each
    reset -- kinematic means PhysX still won't push it around via gravity/contacts, only this
    event term moves it. On top of the group placement, the plate ALSO gets its own small
    roll/pitch lean jitter and an actual physics drop (not a teleport) into its settled resting
    contact against the rack, with a retry if it doesn't land back at roughly the expected height
    (i.e. it tipped out of the rack) -- see that function's docstring for the full reasoning,
    including why the rack's collision meshes needed reauthoring (see the ``dish_rack`` docstring
    entry above) before a physics-based plate drop could even work correctly.
  * ``rag`` gets an actual TOSS, not just an xy nudge (``randomize_rag_drop``, a task-local event
    function defined in this file): each reset it's displaced up + tilted AND given a genuine
    tumble (random-axis angular velocity) from its flat rest pose, then physically settled for
    ~1.5s of physics steps BEFORE the episode's first observation -- see that function's docstring
    for why stepping physics inside a "reset" event is safe here, and why the tumble specifically
    (not just height/tilt) is what's needed for a landing to actually stay creased. This exists
    because the naive fix (``reset_nodal_state_uniform``, which only offsets xy and teleports
    straight into the settled shape) always produced the exact same taut, flush-with-the-pad
    rectangle -- unlike a real rag someone tossed down, and also nothing for a parallel gripper to
    get a fingertip under. A first attempt at fixing that (height+tilt only, no spin, no
    self-collision on the asset) was verified to still relax back flat within ~0.15s regardless --
    see the plate_wiping asset-fix conversation's drop-test GIFs for both the flat-relaxing and the
    actually-crumpled version side by side. Even the tumbling version doesn't fold on every single
    toss (randomness -- some draws just don't self-overlap enough), so a flat landing is no longer
    silently accepted: the function checks the settled shape's flatness and automatically re-tosses
    (fresh random draw, up to ``max_attempts`` times) any env that's still too flat -- see that
    function's docstring for the exact criterion and why this is a retry loop, not an absolute
    guarantee. Every toss-and-settle pass also ends by explicitly zeroing the rag's nodal
    velocity, rather than trusting PhysX's own damping/sleep heuristics to have fully killed it by
    the time ``settle_steps`` runs out -- reported symptom this fixes: the rag (and, verified the
    same way, sometimes the plate) visibly still shaking/moving once the episode had already
    started, instead of sitting still.
  * ``freeze_dynamic_props`` runs LAST (after both events above): an unconditional final
    zero-velocity pass on the plate and rag. Both of the events above already end with their own
    freeze, but ``randomize_dish_rack_and_plate`` runs BEFORE ``randomize_rag_drop``, and the
    latter's settle loop steps the WHOLE scene -- so a freeze done during the rack/plate event
    isn't actually the last word on the plate's velocity by the time the full reset finishes (the
    plate keeps physically settling further during however many extra steps the rag's own toss
    needed). See that function's docstring for the full reasoning.
  * Every range below (rack/plate xy+yaw, plate lean, rag's drop height/tilt/spin) is a
    conservative starting guess, not verified against either arm's actual reach the way
    PickUpEventCfg's cube_2 band was -- see that class' docstring for the same caveat, and see
    RACK_POS's own comment for why the rack moved to mid-pad in the first place (its old position
    let the right arm's own links clip it). Widen them if teleop feels too repetitive; narrow them
    if a corner becomes unreachable, the rack starts overhanging the pad edge, the rack/plate
    group's yaw makes the plate hang off an edge, or the rag's toss drifts/settles it out of the
    left arm's reach.

This is DELIBERATELY the raw-teleop-only version: no task_mode, no per-arm subtask/success
signals, no Mimic wrapper. The pick-up task's own history (see openarm_task_modes.py's module
docstring) is that EVERY grasp/aperture/success threshold it now uses was tuned by measuring
real recorded demos -- there were none to measure yet for a four-stage bimanual wipe when this
file was written. Get raw demos recorded first (this file is enough for that:
``record_demos_openarm.py`` does not require ``--task_mode`` for any task outside
``openarm_task_modes.CAN_TARGET_TASKS``, and this task is not in that tuple), then build the
stage-machine / subtask-signal layer against real measurements the same way the pick-up task's
was, instead of guessing thresholds blind.
"""

import math
import os

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, DeformableObject, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import pickup_ik_abs_env_cfg
from .openarm_task_modes import TASK_ASSET_DIR

PLATE_WIPING_ASSET_DIR = os.path.join(TASK_ASSET_DIR, "plate_wiping")
RACK_USD_PATH = os.path.join(PLATE_WIPING_ASSET_DIR, "dish_rack_kinematic.usdc")
PLATE_USD_PATH = os.path.join(PLATE_WIPING_ASSET_DIR, "plate.usd")
RAG_USD_PATH = os.path.join(PLATE_WIPING_ASSET_DIR, "blue_rag", "blue_rag_deformable.usdc")

# Toward the pad's far side (pad spans x:[0.03,0.51], y:[-0.285,0.285] -- see
# stack_joint_pos_env_cfg.py's PAD_NEAR_EDGE_X/PAD_SIZE_X/PAD_SIZE_Y), not off to the right side,
# and not near-edge either. History: RACK_POS = (0.30, -0.15, 0.28) sat close enough to the right
# arm's own base that its own links could clip the rack during normal motion (reported after
# actually teleoperating) -> (0.24, 0.0) still let the rack's pose_range wander close enough to
# x=0 (the robot's own base, at the pad's NEAR edge) and to either arm's shoulder in y to cause the
# same problem again (reported again, this time as the rack visibly under the arms' own workspace)
# -> (0.40, 0.0) pushed far enough out that with pose_range's x reaching 0.48, only ~3cm short of
# the ACTUAL far edge (0.51), the plate's own lean+physics-drop (see
# randomize_dish_rack_and_plate) could tip it (and, separately, the rag -- see
# randomize_rag_drop's rack-avoidance push) clean off the pad onto the floor below (verified:
# both landed at z~0, plate's x around 0.70 -- nowhere near the pad any more). 0.36 keeps real
# margin from BOTH the robot-side near edge and the far edge (pose_range keeps x within
# [0.30,0.42], i.e. ~9cm clear of the far edge at 0.51, not ~3cm) while still being clearly out
# past where the robot's own base sat. Same PAD_HEIGHT (0.28m) top the plate-drop measurement sim
# used.
RACK_POS = (0.36, 0.0, 0.28)

# The plate comes in two sizes -- see PLATE_SIZE_CONFIGS below for the full explanation of why
# each size needs its OWN measured rest pose, not just a scaled copy of the other's.
#
# "large" is plate.usd at its native scale: measured outline (see the plate_wiping asset-fix
# conversation) is a 26cm outer diameter, ~19cm inner (well) diameter.
PLATE_SCALE_LARGE = 1.0
#
# REVISED (user feedback, screenshot + reference photo of a real dish rack): the original
# far-off-center recipe here (offset magnitude ~7.1cm, rotation a steep ~55-70deg lean) made the
# plate hang mostly OUTSIDE the peg cluster, leaning against a single corner peg -- real contact,
# no penetration, but clearly NOT "seated in the middle, between the pegs" the way every plate in
# a real rack sits.
#
# REVISED AGAIN: the "much smaller off-center offset, same 55deg-roll/-25deg-yaw lean" recipe that
# used to be here (and the shallow-lean small-plate recipe it was copied from) was never actually
# verified by watching the real task settle it -- checked now the same way RACK_TO_PLATE_OFFSET_SMALL
# was re-verified (gym.make + real env.reset() through THIS task's own default
# pose_range/yaw_range/tolerances, root pose logged AND body_cam rendered every reset, not a solo
# hand-derived drop test): it passes its own height/xy tolerance while visibly floating disconnected
# above/beside the rack in every rendered frame, for the same reason the small-plate one did -- the
# shallow off-center start pose barely falls under gravity at all, so "within tolerance of the
# nominal target" just means "still near where it was dropped," not "resting against anything."
#
# What actually produces a genuine, repeatable rest against the peg cluster (confirmed the same way,
# 15+ resets, mixed solo-controlled and full randomize_dish_rack_and_plate calls through real
# env.reset()): a CENTERED xy offset (0, 0) with a NEARLY-upright lean, 85deg roll about the rack's
# local x-axis -- not a full 90deg. Exactly 90deg (verified separately) is a knife-edge unstable
# balance that settles flat about as often as it settles leaning, in either yaw direction, since
# nothing breaks the left/right symmetry; 85deg is enough asymmetry to consistently fall (and stay)
# leaning the same way every time, while still reading as "upright, resting in the middle of the
# slot" rather than the old shallow far-off-center lean. The settled root height above the rack ends
# up close to the plate's own outer radius (0.13m here) regardless of the rack's own randomized
# xy/yaw and the plate's own lean jitter -- consistent with the plate's bottom rim resting on the
# rack's tray floor while the rest of the disc leans against a peg for lateral support, the same
# relationship RACK_TO_PLATE_OFFSET_SMALL's docstring found for the small plate at its own radius.
RACK_TO_PLATE_OFFSET_LARGE = (0.0, 0.0, 0.13)
PLATE_REST_ROT_LARGE = (0.7372773289680481, 0.6755901575088501, 0.0, 0.0)

# "small": requested as an exact 19cm outer / 14cm inner diameter plate. No separate small-plate
# asset exists, but plate.usd's own inner:outer ratio (19cm:26cm, measured above) already lands
# almost exactly on the requested numbers under a uniform scale: 0.19 * (19/26) = 13.88cm inner,
# ~1mm off the requested 14cm -- close enough that authoring a whole second mesh wasn't worth it.
# So "small" is the SAME plate.usd, spawned with UsdFileCfg.scale = (PLATE_SCALE_SMALL,)*3, not a
# different file.
PLATE_SCALE_SMALL = 19.0 / 26.0
# Measured the SAME "drop it and read where it settles" way as the large plate, dropped onto
# dish_rack_kinematic.usdc at its own RACK_POS -- NOT just RACK_TO_PLATE_OFFSET_LARGE /
# PLATE_REST_ROT_LARGE rescaled by PLATE_SCALE_SMALL. This matters because the rack's peg spacing
# is fixed geometry sized around the LARGE plate's 26cm rim; a scaled offset/rotation assumes the
# smaller plate leans against those same pegs the same way, and it does not automatically.
#
# An earlier measurement dropped the small plate CENTERED and FLAT above the rack -- its own
# inner hole (7cm radius) is bigger than the whole peg cluster's half-diagonal (3.6cm), so a
# centered flat drop just lowers the hole over all 4 pegs and rests flat on their tops, never
# actually caught by the rack at all. A second attempt (steep lean, offset ~7.9cm off-center)
# fixed the "not caught at all" problem but over-corrected into "hangs off to one side against a
# single peg" -- exactly the same not-centered failure mode as the large plate's original recipe,
# per the user's follow-up correction (screenshot + a real dish rack reference photo showing every
# plate centered in its slot, not leaning to one side).
#
# REVISED (second correction -- the "half offset, same 55deg-roll/-25deg-yaw lean" values that used
# to be here, verified against ``record_demos_openarm.py``'s ACTUAL reset path in-process (gym.make
# + repeated env.reset(), not a solo hand-derived drop test): logged the plate's settled root pose
# every reset and rendered body_cam to actually LOOK at it (see the plate_wiping asset-fix
# conversation's follow-up -- this file's own uncommitted history had the "half offset"
# numbers passing their OWN height/xy tolerance check yet visibly floating disconnected above/beside
# the rack in every rendered frame: the recipe's start pose barely moved under gravity at all, i.e.
# it was resting on nothing, just frozen near its spawn height by the reset event's own hard
# zero-velocity freeze -- tolerance passing is not proof of actual contact). A perfectly centered,
# NEARLY-UPRIGHT drop (no xy offset at all, ~85deg roll about the rack's local x-axis rather than a
# shallow ~55deg lean) is what actually produces a genuine, repeatable rest against the peg cluster
# for THIS plate/rack pair at this scale: confirmed over 20+ resets (mixed: solo controlled calls
# with the rack held fixed, and full ``randomize_dish_rack_and_plate`` calls through real
# ``env.reset()`` with the task's own default pose_range/yaw_range/tolerances/settle_steps/
# plate_drop_height, i.e. no test-only loosening) that it settles to the same root height/offset
# from the rack (root pos ~9.1cm above rack root in z, ~3.0cm off in xy) regardless of the rack's
# own randomized xy/yaw and the plate's own +-8deg lean jitter -- and body_cam renders across many
# of those confirm it visually as a plate genuinely leaning IN the peg cluster (one edge caught,
# other side clear of the rack), not floating apart from it or perched flat on the peg tips (the
# centered-and-shallow failure mode above).
RACK_TO_PLATE_OFFSET_SMALL = (0.0, 0.0, 0.095)
PLATE_REST_ROT_SMALL = (0.7372773289680481, 0.6755901575088501, 0.0, 0.0)

# Outer radius of the plate disc itself (native mesh measured at 26cm outer diameter -> 13cm
# radius, scaled the same way as the spawn scale). Needed so randomize_dish_rack_and_plate can
# keep the whole disc -- not just its root point -- clear of the arms: a root-position-only check
# missed exactly this class of bug (see this module's `apply_plate_size`/arm-avoidance comments).
PLATE_RADIUS_LARGE = 0.13
PLATE_RADIUS_SMALL = 0.095

PLATE_SIZE_CONFIGS = {
    "small": {
        "scale": PLATE_SCALE_SMALL,
        "offset": RACK_TO_PLATE_OFFSET_SMALL,
        "rot": PLATE_REST_ROT_SMALL,
        "radius": PLATE_RADIUS_SMALL,
    },
    "large": {
        "scale": PLATE_SCALE_LARGE,
        "offset": RACK_TO_PLATE_OFFSET_LARGE,
        "rot": PLATE_REST_ROT_LARGE,
        "radius": PLATE_RADIUS_LARGE,
    },
}
# Default per direct instruction ("small plate is in default setting").
DEFAULT_PLATE_SIZE = "small"

# Left/right arm gripper+camera cluster, projected to XY, measured directly (robot.data.body_pos_w
# at the default reset pose -- see the plate_wiping asset-fix conversation's diag_arm_links.py) as
# the centroid of link6/link7/ee_tcp/camera_link/both fingers. Symmetric about y=0. Used to keep
# the plate's full disc (not just its root point) clear of the arms -- verified necessary: a plate
# whose ROOT stayed within the rack's own randomization range still let the disc's rim swing to
# within ~4mm of the left camera-link housing under some yaw draws (rigorous mesh-vs-body-point
# clearance check, not just centroid distance).
# NOTE: with either plate's own radius already close to half the ~30.6cm arm-to-arm spacing, an
# overly generous cluster radius here makes required_dist (see randomize_dish_rack_and_plate)
# exceed half that spacing and the two keepout circles overlap, leaving no valid position near the
# pad's actual center -- keep this at the housing/finger's own rough half-width, not a padded
# whole-cluster radius, and let plate_radius (already the dominant term) carry the real margin.
ARM_KEEPOUT_XY_LEFT = (0.26, 0.153)
ARM_KEEPOUT_XY_RIGHT = (0.26, -0.153)
ARM_KEEPOUT_RADIUS = 0.02
# Extra padding vs. the bare housing/finger half-width: the keepout check below is XY-only (rack
# xy + yaw), but plate_lean_jitter_deg can stand the plate up steeply enough that its TOP edge
# reaches noticeably higher in z than a flat disc would, closing some of the gap to the camera-link
# housing that a pure top-down xy distance doesn't see -- verified directly (a rare steep-lean
# trial still showed ~1cm of real plate/camera-link overlap at margin=0.01). 0.03 covers that
# without rejecting so much of the pad that resampling stops converging.
ARM_KEEPOUT_MARGIN = 0.03


def _plate_rest_pos(offset: tuple[float, float, float]) -> tuple[float, float, float]:
    return (RACK_POS[0] + offset[0], RACK_POS[1] + offset[1], RACK_POS[2] + offset[2])


# Kept as the names the rest of this module (and its docstring) already refers to -- resolve to
# whichever size is DEFAULT_PLATE_SIZE. apply_plate_size() below overrides all of this on an
# already-constructed env_cfg for the non-default choice; these are just what the scene/event
# params are initially built with in __post_init__.
RACK_TO_PLATE_OFFSET = PLATE_SIZE_CONFIGS[DEFAULT_PLATE_SIZE]["offset"]
PLATE_REST_ROT = PLATE_SIZE_CONFIGS[DEFAULT_PLATE_SIZE]["rot"]
PLATE_REST_POS = _plate_rest_pos(RACK_TO_PLATE_OFFSET)

# Measured the same way: dropped flat onto the pad (as a PhysX deformable body, not a rigid
# collider -- see this module's docstring) and read back its settled nodal centroid.
RAG_REST_POS = (0.30, 0.15, 0.2811)
RAG_REST_ROT = (1.0, 0.0, 0.0, 0.0)


def _settle_plate(
    env: ManagerBasedEnv,
    plate,
    ids: torch.Tensor,
    start_pos: torch.Tensor,
    start_rot: torch.Tensor,
    settle_steps: int,
):
    """Write the plate to a starting pose (typically hovering a little above where it should end
    up resting) with zero velocity, then physically step it forward so gravity/contact settles it
    -- rather than teleporting straight into a hand-picked "resting" pose the way this task did
    before. Mirrors ``_toss_and_settle_rag``'s reasoning for why stepping physics inside a "reset"
    event is safe (see ``randomize_rag_drop``'s docstring): every "reset"-mode event runs before
    the episode's first observation is computed, so the settled result IS the starting state.

    Ends by explicitly zeroing the plate's velocity, the same "hard freeze" ``_toss_and_settle_rag``
    uses and for the same reason: verified directly (root_lin_vel_w read right after a reset, before
    this fix existed) that some resets left the plate still moving at up to ~0.5 m/s -- a lean angle
    that only finishes settling into full contact gradually, still sliding/rocking slightly when
    ``settle_steps`` ran out, rather than a hard equilibrium PhysX itself would report near-zero
    velocity for (contrast with the rag, where the residual motion was self-collision jitter around
    an already-valid rest state; here it's just "hadn't finished settling yet"). Either way, a
    reset event's job is to hand back a stationary starting state, not whatever the physics was
    mid-way through -- so force it.
    """
    plate.write_root_pose_to_sim(torch.cat([start_pos, start_rot], dim=-1), env_ids=ids)
    plate.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=plate.device), env_ids=ids)
    env.scene.write_data_to_sim()
    for _ in range(settle_steps):
        env.sim.step(render=False)
    env.scene.update(env.sim.get_physics_dt())

    plate.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=plate.device), env_ids=ids)


def randomize_dish_rack_and_plate(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    yaw_range_deg: tuple[float, float],
    plate_lean_jitter_deg: dict[str, tuple[float, float]],
    plate_drop_height: float,
    plate_height_tolerance: float,
    plate_xy_tolerance: float,
    settle_steps: int,
    max_attempts: int,
    rack_cfg: SceneEntityCfg,
    plate_cfg: SceneEntityCfg,
    rack_to_plate_offset: tuple[float, float, float] = RACK_TO_PLATE_OFFSET,
    plate_rest_rot: tuple[float, float, float, float] = PLATE_REST_ROT,
    plate_radius: float = PLATE_SIZE_CONFIGS[DEFAULT_PLATE_SIZE]["radius"],
):
    """Place ``dish_rack`` and ``plate`` as one rigid group (shared xy offset AND shared yaw
    rotation about the rack's own pivot), then separately re-settle the plate's lean within the
    rack via a small physics drop -- instead of teleporting both into a single hand-measured,
    always-identical relative pose the way this task did before.

    Why the rack and plate move together: the plate rests leaning IN the rack (see
    ``rack_to_plate_offset``); randomizing either one independently would drift the plate out of
    the rack -- floating next to it or clipped through a wall, depending on which way two
    independent draws happened to point. Both effects (xy translation, yaw rotation) are applied to
    dish_rack's actual root pose, and to the plate's TARGET pose via the same rigid transform
    (``rack_to_plate_offset`` rotated by the same yaw, ``plate_rest_rot`` composed with the same
    yaw) -- so the plate's target always sits correctly relative to wherever the rack ends up, not
    just the original RACK_POS/PLATE_REST_POS. dish_rack is written directly (kinematic, no
    settling needed -- see the parent docstring on why it's a RigidObject at all); the plate's
    target is a STARTING point for a physics drop, not a final write, so its lean can vary.

    ``rack_to_plate_offset``/``plate_rest_rot`` default to the module-level constants (which
    resolve to whichever size DEFAULT_PLATE_SIZE names), but are real parameters, not just
    module-global reads, specifically so ``apply_plate_size`` can override them per env_cfg
    instance for the OTHER plate size -- see PLATE_SIZE_CONFIGS' docstring for why the small and
    large plates need genuinely different measured values here, not a scaled copy of one offset.

    Plate leaning direction: rather than trust an arbitrary rotation jitter to still be a physically
    valid "resting in the rack" configuration (a plate is thin and rigid -- a rotation that isn't
    actually the shape the rack can support just clips through a wall or tips over), this drops the
    plate from ``plate_drop_height`` above the group's nominal target with a small extra
    ``plate_lean_jitter_deg`` roll/pitch tilt (applied in the rack's OWN local frame, i.e. relative
    to the originally-measured lean, so it means the same thing regardless of the group's yaw), and
    lets ``settle_steps`` of physics resolve it into whatever valid contact configuration gravity
    and the (now SDF-collision, see dish_rack_kinematic.usdc) rack actually supports. Checks the
    settled height against the nominal height (``plate_height_tolerance``) and re-tosses (fresh
    jitter draw) any env where it doesn't land close to where a plate resting in the rack should be
    -- e.g. tipped over onto the pad instead of staying in the rack -- up to ``max_attempts`` times,
    same retry-loop pattern (and same caveat: not an absolute guarantee) as ``randomize_rag_drop``.
    """
    rack = env.scene[rack_cfg.name]
    plate = env.scene[plate_cfg.name]
    n = len(env_ids)
    device = rack.device

    # ── position the rack: shared xy offset + shared yaw, about its own pivot ──────────
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ("x", "y")]
    ranges = torch.tensor(range_list, device=device)
    offset_xy = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (n, 2), device=device)

    yaw_lo, yaw_hi = math.radians(yaw_range_deg[0]), math.radians(yaw_range_deg[1])
    yaw = yaw_lo + (yaw_hi - yaw_lo) * torch.rand(n, device=device)
    z_axis = torch.zeros((n, 3), device=device)
    z_axis[:, 2] = 1.0
    yaw_quat = math_utils.quat_from_angle_axis(yaw, z_axis)

    rack_default = rack.data.default_root_state[env_ids].clone()
    rack_pos = rack_default[:, 0:3] + env.scene.env_origins[env_ids]
    rack_pos[:, 0:2] += offset_xy
    # RACK_POS's own spawn rotation is identity (see its RigidObjectCfg.InitialStateCfg below), so
    # composing with yaw_quat here is equivalent to just using yaw_quat directly -- written this
    # way (rather than assuming that) so it stays correct if that default ever changes.
    rack_rot = math_utils.quat_mul(yaw_quat, rack_default[:, 3:7])

    # ── keep the plate's full disc clear of the arm gripper/camera clusters ────────────
    # The plate's world center is rack_pos + (rack_rot-rotated) rack_to_plate_offset. Rather than
    # PUSH a violating draw away from the nearest keepout circle (tried first, reverted: with the
    # plate's own radius already more than a third of the arm-to-arm spacing, the two keepout
    # circles overlap, so a push satisfying one side can drive the plate into the other -- observed
    # directly as a runaway multi-meter "correction" that then made the plate's physics drop start
    # deep inside the rack and explode away on contact resolution), just REDRAW a fresh (offset_xy,
    # yaw) for whichever envs violate and recheck -- independent fresh samples can't compound into
    # an out-of-range position the way an iterative push can, since every candidate is still drawn
    # from the exact same validated pose_range/yaw_range as everyone else.
    offset_local_const = torch.tensor(rack_to_plate_offset, device=device)
    keepout_left = torch.tensor(ARM_KEEPOUT_XY_LEFT, device=device)
    keepout_right = torch.tensor(ARM_KEEPOUT_XY_RIGHT, device=device)
    required_dist = ARM_KEEPOUT_RADIUS + plate_radius + ARM_KEEPOUT_MARGIN

    def _violates_arm_keepout(pos_xy: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
        plate_xy = pos_xy + math_utils.quat_apply(rot, offset_local_const.expand(pos_xy.shape[0], 3))[:, 0:2]
        d_left = (plate_xy - keepout_left).norm(dim=-1)
        d_right = (plate_xy - keepout_right).norm(dim=-1)
        return (d_left < required_dist) | (d_right < required_dist)

    rack_base_xy = rack_default[:, 0:2] + env.scene.env_origins[env_ids][:, 0:2]
    bad = _violates_arm_keepout(rack_pos[:, 0:2], rack_rot)
    for _ in range(6):
        if not bool(bad.any()):
            break
        m = int(bad.sum().item())
        redraw_xy = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (m, 2), device=device)
        redraw_yaw = yaw_lo + (yaw_hi - yaw_lo) * torch.rand(m, device=device)
        redraw_yaw_quat = math_utils.quat_from_angle_axis(redraw_yaw, z_axis[bad])
        rack_pos[bad, 0:2] = rack_base_xy[bad] + redraw_xy
        rack_rot[bad] = math_utils.quat_mul(redraw_yaw_quat, rack_default[bad, 3:7])
        bad = _violates_arm_keepout(rack_pos[:, 0:2], rack_rot)

    rack.write_root_pose_to_sim(torch.cat([rack_pos, rack_rot], dim=-1), env_ids=env_ids)
    rack.write_root_velocity_to_sim(torch.zeros_like(rack_default[:, 7:13]), env_ids=env_ids)
    # refresh rack.data so root_pos_w/root_quat_w below reflect the write just made, not last
    # episode's values.
    env.scene.write_data_to_sim()
    env.scene.update(env.sim.get_physics_dt())

    # ── drop the plate into the now-placed rack, with a random lean, and retry on a bad
    # landing ──────────────────────────────────────────────────────────────────────────
    offset_local = torch.tensor(rack_to_plate_offset, device=device)
    plate_local_rot_const = torch.tensor(plate_rest_rot, device=device)
    lean_range_list = [plate_lean_jitter_deg.get(key, (0.0, 0.0)) for key in ("roll", "pitch")]
    lean_ranges = torch.deg2rad(torch.tensor(lean_range_list, device=device))
    drop_offset = torch.tensor([0.0, 0.0, plate_drop_height], device=device)

    pending_ids = env_ids
    for attempt in range(max_attempts):
        m = len(pending_ids)
        # read the RACK's actual current pose for exactly these envs (indexed by absolute env id,
        # so this stays correct across retries even as pending_ids shrinks -- no need to carry a
        # separately-indexed copy of offset_xy/yaw_quat in sync with a shrinking subset).
        rack_pos_now = rack.data.root_pos_w[pending_ids]
        rack_rot_now = rack.data.root_quat_w[pending_ids]

        lean = math_utils.sample_uniform(lean_ranges[:, 0], lean_ranges[:, 1], (m, 2), device=device)
        lean_quat = math_utils.quat_from_euler_xyz(lean[:, 0], lean[:, 1], torch.zeros(m, device=device))
        # jitter applied in the LOCAL (pre-yaw, as-measured) frame, THEN the group's yaw on top --
        # see docstring on why this order keeps "leaning direction" meaning the same regardless of
        # how the rack itself has been rotated this episode.
        jittered_local_rot = math_utils.quat_mul(lean_quat, plate_local_rot_const.expand(m, 4))
        start_rot = math_utils.quat_mul(rack_rot_now, jittered_local_rot)
        start_pos = rack_pos_now + math_utils.quat_apply(rack_rot_now, offset_local.expand(m, 3)) + drop_offset

        _settle_plate(env, plate, pending_ids, start_pos, start_rot, settle_steps)

        # z doesn't change under a pure yaw-about-z rotation of offset_local, so the expected
        # height is just the rack's height plus the constant offset -- no need to re-derive it via
        # quat_apply. xy DOES change under that rotation, so it's derived the same way start_pos
        # was above.
        expected_pos = rack_pos_now + math_utils.quat_apply(rack_rot_now, offset_local.expand(m, 3))
        actual_pos = plate.data.root_pos_w[pending_ids]
        height_bad = (actual_pos[:, 2] - expected_pos[:, 2]).abs() > plate_height_tolerance
        # z-only was NOT enough: verified (screenshots from the plate_wiping asset-fix
        # conversation) that a plate can tip sideways out of the rack and land elsewhere on the
        # pad while still settling at a height that happened to fall inside plate_height_tolerance
        # -- e.g. leaning against the rack's OUTSIDE instead of sitting IN it. xy distance from the
        # expected in-rack position catches that the height check alone missed.
        xy_bad = (actual_pos[:, 0:2] - expected_pos[:, 0:2]).norm(dim=-1) > plate_xy_tolerance
        needs_retry = height_bad | xy_bad
        if not needs_retry.any() or attempt == max_attempts - 1:
            break
        pending_ids = pending_ids[needs_retry]


def _toss_and_settle_rag(
    env: ManagerBasedEnv,
    rag: DeformableObject,
    ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    tilt_range_deg: dict[str, tuple[float, float]],
    spin_rate_range: tuple[float, float],
    settle_steps: int,
    rack=None,
    min_rack_separation: float = 0.0,
    rack_avoid_margin: float = 0.05,
):
    """One toss-and-settle pass for the given (sub)set of env ids. Factored out of
    ``randomize_rag_drop`` so that function can call this again, on just the envs that are still
    too flat, instead of re-tossing envs that already landed wrinkled.

    Ends by explicitly zeroing nodal velocity (not just letting it settle "naturally" via PhysX's
    own sleep/damping heuristics). Reported symptom this fixes: the rag visibly shaking/jittering
    once the episode had already started, instead of sitting still -- self-collision contacts on a
    freshly-crumpled fold can keep chattering at low amplitude (repeatedly resolving a tiny
    penetration, which reintroduces a tiny velocity, which causes a tiny new penetration next step)
    well past the point where the shape has visually finished settling, and PhysX's own sleep
    threshold isn't guaranteed to trip before ``settle_steps`` runs out. A hard zero-velocity write
    is a stronger guarantee than tuning damping/sleep parameters further: whatever state the mesh
    is in when this returns, it starts the episode with zero momentum, full stop.

    If ``rack`` is given, this ACTIVELY pushes the landing target's xy away from the rack's current
    position -- by exactly enough to clear ``min_rack_separation`` (plus ``rack_avoid_margin``),
    not just resampled and hoped for. This exists because a purely-random retry (draw a fresh
    ``position_range`` offset, check the result, retry if still too close) turned out not to be
    enough on its own: randomize_dish_rack_and_plate can now place the rack almost anywhere on the
    pad, so its placement range can fully swallow the rag's own (much narrower) landing region --
    verified across 15 resets, ~47% still ended up under min_rack_separation even with
    position_range widened and max_attempts raised, because for those draws NO offset within
    position_range could have cleared the requirement; it wasn't a matter of bad luck to retry
    past. Deterministically pushing the target away from wherever the rack actually is fixes that
    at the source, and randomization is preserved: the underlying jitter is still random, this just
    guarantees the result clears the rack regardless of which way that jitter happened to point.
    """
    nodal_state = rag.data.default_nodal_state_w[ids].clone()
    n = len(ids)

    pos_range_list = [position_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")]
    pos_ranges = torch.tensor(pos_range_list, device=rag.device)
    pos_offset = math_utils.sample_uniform(pos_ranges[:, 0], pos_ranges[:, 1], (n, 3), device=rag.device)

    if rack is not None and min_rack_separation > 0.0:
        # the rest shape's own centroid (nodal_state hasn't been transformed yet at this point),
        # since transform_nodal_pos below applies pos_offset as an ADDITIVE offset on top of
        # exactly this centroid -- see its docstring in deformable_object.py.
        nominal_xy = nodal_state[..., 0:2].mean(dim=1)
        rack_xy = rack.data.root_pos_w[ids, 0:2]
        target_xy = nominal_xy + pos_offset[:, 0:2]
        required = min_rack_separation + rack_avoid_margin
        # Push-then-clamp, iterated: verified (10-trial visual check, plate_wiping asset-fix
        # conversation) that a SINGLE push-then-clamp pass could still leave the rag visibly
        # touching the rack/gripper in ~30-40% of resets. Root cause -- the push moves straight
        # away from the rack along whatever direction the (random) pre-push target happened to be
        # in; if that direction points at a nearby pad-boundary wall, clamping cuts the push short
        # RIGHT THERE, even when a different direction (e.g. straight down in y) would have had
        # plenty of room inside the same box to satisfy `required`. Re-deriving the push direction
        # from the just-clamped position and repeating converges toward a corner/edge of the box
        # that actually clears the rack, in the many cases where the box does contain one -- one
        # push along a fixed, possibly-unlucky direction does not.
        for _ in range(4):
            delta = target_xy - rack_xy
            dist = delta.norm(dim=-1).clamp_min(1e-6)
            push = (required - dist).clamp_min(0.0)
            if not bool((push > 0.0).any()):
                break
            direction = delta / dist.unsqueeze(-1)
            target_xy = target_xy + direction * push.unsqueeze(-1)
            # Clamp to a safe pad-interior box every iteration, not just at the end -- verified
            # separately that an unclamped push could land the rag off the pad entirely (one case
            # pushed to y=0.42, past the pad's actual y limit of 0.285), landing on the floor
            # below (z~0). Bounds leave margin for the rag's own ~0.3x0.3m extent plus tumble
            # drift, not just its centroid.
            target_xy[:, 0] = target_xy[:, 0].clamp(0.13, 0.45)
            target_xy[:, 1] = target_xy[:, 1].clamp(-0.22, 0.22)
        pos_offset[:, 0:2] = target_xy - nominal_xy

    tilt_range_list = [tilt_range_deg.get(key, (0.0, 0.0)) for key in ("roll", "pitch", "yaw")]
    tilt_ranges = torch.deg2rad(torch.tensor(tilt_range_list, device=rag.device))
    tilt = math_utils.sample_uniform(tilt_ranges[:, 0], tilt_ranges[:, 1], (n, 3), device=rag.device)
    tilt_quat = math_utils.quat_from_euler_xyz(tilt[:, 0], tilt[:, 1], tilt[:, 2])

    # transform_nodal_pos rotates/translates about the shape's OWN centroid (see its docstring in
    # deformable_object.py), so pos_offset/tilt_quat are relative to the already-flat rest pose,
    # not an absolute world pose.
    nodal_state[..., :3] = rag.transform_nodal_pos(nodal_state[..., :3], pos_offset, tilt_quat)

    # give it a genuine tumble -- random axis, magnitude sampled from spin_rate_range -- via
    # angular velocity about its own centroid (v = omega x r for every node). Rotation about the
    # centroid carries no net linear momentum, so this doesn't add its own xy drift on top of
    # pos_offset above.
    spin_axis = torch.randn((n, 3), device=rag.device)
    spin_axis = spin_axis / spin_axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    spin_rate = spin_rate_range[0] + (spin_rate_range[1] - spin_rate_range[0]) * torch.rand(
        (n, 1), device=rag.device
    )
    omega = (spin_axis * spin_rate).unsqueeze(1)
    centroid = nodal_state[..., :3].mean(dim=1, keepdim=True)
    nodal_state[..., 3:] = torch.cross(omega.expand_as(nodal_state[..., :3]), nodal_state[..., :3] - centroid, dim=-1)

    rag.write_nodal_state_to_sim(nodal_state, env_ids=ids)

    # let it fall, tumble, and crumple/settle -- see randomize_rag_drop's docstring for why extra
    # sim.step() calls inside a "reset" event are safe.
    env.scene.write_data_to_sim()
    for _ in range(settle_steps):
        env.sim.step(render=False)
    env.scene.update(env.sim.get_physics_dt())

    # hard-freeze: whatever residual velocity PhysX's own damping/sleep didn't fully kill, force it
    # to zero -- see docstring.
    final_state = rag.data.nodal_pos_w[ids]
    zero_vel = torch.zeros_like(final_state)
    rag.write_nodal_state_to_sim(torch.cat([final_state, zero_vel], dim=-1), env_ids=ids)


def randomize_rag_drop(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    tilt_range_deg: dict[str, tuple[float, float]],
    spin_rate_range: tuple[float, float],
    settle_steps: int,
    min_wrinkle_spread: float,
    max_safe_z: float,
    min_safe_z: float,
    min_rack_separation: float,
    max_attempts: int,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rack_cfg: SceneEntityCfg = SceneEntityCfg("dish_rack"),
):
    """Toss the rag -- height, tilt, AND a real tumble -- above its rest pose and physically settle
    it for a few steps before the episode starts, instead of teleporting it into the exact same
    flat pose every reset (what the plain xy-only randomizer this replaces did). Produces
    organically different creased/CRUMPLED shapes episode to episode -- a real rag someone tossed
    onto a table isn't a taut rectangle, and a taut rectangle flush with the pad also gives a
    parallel gripper nowhere to get a fingertip under it.

    An earlier version of this function only applied a static height+tilt offset (no spin) --
    verified (see the plate_wiping asset-fix conversation's drop-test GIFs) to relax back to a
    flat, uncreased rectangle within ~0.15s even at that version's max height/tilt, regardless of
    ``blue_rag_deformable.usdc``'s deformable-body material being soft. The missing piece was
    self-collision (now enabled on that asset) NEEDING an actual fold-over to have anything to
    hold in the first place -- a static tilt just settles flat, nothing self-intersects. Giving the
    nodal state a genuine angular velocity (``spin_rate_range``) about its own centroid -- i.e. an
    actual tumble in the air, not just a tilted static drop -- is what lets a corner fold UNDER
    another part of the cloth as it falls, which self-collision can then hold once it lands.

    Even with the tumble, not every toss folds -- verified across 6 resets, roughly a third landed
    close enough to flat (z-spread under ~10mm, vs. 14-46mm for the ones that visibly folded) that
    it wasn't a meaningfully different starting state from before. Since a flat spawn is exactly
    what this function exists to avoid, it doesn't just toss once and accept whatever happens: it
    checks each env's nodal z-spread against ``min_wrinkle_spread`` after settling, and RE-TOSSES
    (a fresh random draw, not the same one repeated) any env that's still too flat, up to
    ``max_attempts`` times. This is a retry loop, not a guarantee by construction -- it's still
    physically possible (just very unlikely across independent draws) for every attempt to land
    flat, in which case the last attempt's result is kept as-is rather than looping forever.

    The retry check ALSO rejects (and re-tosses) any outcome where a nodal point ends up above
    ``max_safe_z`` -- this isn't about flatness. The left gripper's fingers sit, at rest, only
    ~20cm above the rag's rest height (measured: gripper fingers z~0.477, rag rest z~0.281) --
    right above the rag, since that's exactly where the left arm needs to reach to pick it up
    later in the episode. A vigorous enough toss/tumble can fling part of the mesh up into contact
    with those fingers; once that happens the rag gets physically snagged on them and just hangs
    there -- diagnosed by stepping one such case for 1000 extra physics steps: node speed decayed
    to near-zero (a real, stable equilibrium, not still-falling) while a chunk of the mesh stayed
    pinned around z~0.53-0.59, well above the gripper. ``max_safe_z`` should sit comfortably below
    the gripper height and comfortably above any legitimate crumpled-pile height (a genuine
    scrunched rag sitting on the pad has no real reason to be taller than a few cm).

    Symmetrically, the retry check ALSO rejects any outcome where the rag's LOWEST nodal point
    ends up below ``min_safe_z``. Pushing the toss energy up (to make folds more consistently
    obvious, not just barely non-flat) surfaced this the same way max_safe_z's failure mode was
    found: an occasional toss now had enough energy to tumble the rag clean off the pad's edge,
    landing on the world ground plane below (z near 0) instead of settling on the pad (z~0.28) --
    a spurious "big z-spread" that would otherwise satisfy ``min_wrinkle_spread`` for entirely the
    wrong reason. ``min_safe_z`` should sit a few cm below the pad's actual top (RAG_REST_POS'
    z, i.e. some settling/penetration tolerance) but well above the ground plane.

    A third rejection reason, orthogonal to both of the above: the rag's landing centroid ending
    up within ``min_rack_separation`` of the rack's CURRENT position (``rack_cfg`` -- read live,
    not RACK_POS, since randomize_dish_rack_and_plate -- which runs first -- can now place the
    rack almost anywhere on the pad). Without this, a rack that happened to land near the rag's
    own spawn area had nothing stopping the two from visibly overlapping/interpenetrating.

    This event function does something PickUpEventCfg's/this file's other event terms don't: it
    steps physics itself (``env.sim.step()``, via ``_toss_and_settle_rag``), rather than just
    writing a state once. That's safe here because ``_reset_idx`` (manager_based_env.py) applies
    every "reset"-mode event BEFORE the observation manager computes the episode's first
    observation -- so by the time this function returns, the settled state IS the state the
    episode starts from, with nothing downstream the wiser that extra steps (now possibly several
    toss-settle-check cycles) happened.

    One thing that ordering does NOT give for free: ``_reset_idx`` also runs event_manager.apply
    BEFORE action_manager.reset(), so the robot's joint position TARGET is still whatever the
    previous episode last commanded, even though init_robot_pose (an earlier reset event, inherited
    from PickUpEventCfg) already teleported its joint STATE to the rest pose. Left alone, that stale
    target would pull the arms out of rest pose while the extra settle steps below run -- so this
    re-pins the target to the just-reset state first.
    """
    rag: DeformableObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    rack = env.scene[rack_cfg.name]

    # hold the robot still through the settle steps below -- see docstring
    robot.set_joint_position_target(robot.data.default_joint_pos[env_ids], env_ids=env_ids)

    pending_ids = env_ids
    for attempt in range(max_attempts):
        _toss_and_settle_rag(
            env,
            rag,
            pending_ids,
            position_range,
            tilt_range_deg,
            spin_rate_range,
            settle_steps,
            rack=rack,
            min_rack_separation=min_rack_separation,
        )

        nodal = rag.data.nodal_pos_w[pending_ids]
        zmax = nodal[..., 2].amax(dim=1)
        zmin = nodal[..., 2].amin(dim=1)
        spread = zmax - zmin
        out_of_bounds = (spread < min_wrinkle_spread) | (zmax > max_safe_z) | (zmin < min_safe_z)

        # this task's env cfg now lets randomize_dish_rack_and_plate place the rack ANYWHERE
        # across most of the pad (and at any rotation) -- it's no longer safe to assume the rag's
        # own position_range keeps it clear of wherever the rack+plate happened to land this
        # episode. Reject (and re-toss) any landing whose centroid ends up too close to the
        # rack's CURRENT position, rather than relying on the two objects' nominal positions
        # having been far enough apart by construction.
        rag_centroid_xy = nodal[..., 0:2].mean(dim=1)
        rack_xy = rack.data.root_pos_w[pending_ids, 0:2]
        too_close_to_rack = (rag_centroid_xy - rack_xy).norm(dim=-1) < min_rack_separation

        needs_retry = out_of_bounds | too_close_to_rack
        if not needs_retry.any() or attempt == max_attempts - 1:
            break
        pending_ids = pending_ids[needs_retry]


def freeze_dynamic_props(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
):
    """Zero velocity (root, or nodal for a DeformableObject) for every listed asset. Meant to be
    the LAST reset event this task registers, after both randomize_dish_rack_and_plate and
    randomize_rag_drop.

    Both of those already end with their own "hard freeze" (see their docstrings) -- this exists
    because event execution order made those individual freezes not actually the last word:
    randomize_dish_rack_and_plate runs BEFORE randomize_rag_drop, and randomize_rag_drop's own
    settle loop calls ``env.sim.step()``, which advances the WHOLE scene -- plate included, not
    just the rag. Verified directly (root_lin_vel_w read right after a reset): even with
    _settle_plate's own freeze in place, the plate could still end up with non-trivial residual
    velocity (observed up to ~0.19 m/s) by the time the full reset finished, because it kept
    settling further during however many hundreds/thousands of extra physics steps the rag's own
    (possibly multi-attempt) toss-and-settle contributed afterward. Rather than tie this task's
    event ORDER to which object's settle logic needs to run last (fragile -- breaks again the next
    time an event is added or reordered), this runs unconditionally after everything else and
    zeroes velocity one final time, on every object that should be stationary when the episode
    starts.
    """
    for asset_cfg in asset_cfgs:
        asset = env.scene[asset_cfg.name]
        if isinstance(asset, DeformableObject):
            pos = asset.data.nodal_pos_w[env_ids]
            asset.write_nodal_state_to_sim(torch.cat([pos, torch.zeros_like(pos)], dim=-1), env_ids=env_ids)
        else:
            asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=asset.device), env_ids=env_ids)


def apply_plate_size(env_cfg, size: str) -> str:
    """Switch an already-constructed ``OpenarmPlateWipeEnvCfg`` between the "small" (default) and
    "large" plate -- same post-hoc-mutation pattern ``record_demos_openarm.py`` already uses for
    the pick-up task's ``--randomize_object_size`` (``openarm_task_modes.attach_object_size_randomization``):
    called on the ``env_cfg`` returned by ``parse_env_cfg`` before ``gym.make``, not a constructor
    argument, since gym's task registry only takes an env cfg INSTANCE, not extra kwargs to build
    one with.

    Mutates three things, all needed together -- setting only the spawn scale would leave the
    plate visually resized but still targeting the OTHER size's measured rest pose, landing it
    wrong every reset (or failing ``plate_height_tolerance``/``plate_xy_tolerance`` every attempt):
      1. ``env_cfg.scene.plate.spawn.scale`` -- the actual mesh scale.
      2. ``env_cfg.scene.plate.init_state.pos``/``rot`` -- the pose the plate first spawns at,
         before any reset event has run once.
      3. ``env_cfg.events.randomize_dish_rack_and_plate.params["rack_to_plate_offset"``/
         ``"plate_rest_rot"]`` -- what every SUBSEQUENT reset targets. See ``PLATE_SIZE_CONFIGS``'
         docstring for why these three must be looked up together per size rather than the small
         plate's rest pose being derived by rescaling the large plate's.

    Args:
        env_cfg: An ``OpenarmPlateWipeEnvCfg`` instance (or any cfg with a ``scene.plate`` and an
            ``events.randomize_dish_rack_and_plate`` event term shaped like this task's).
        size: ``"small"`` or ``"large"``.

    Returns:
        A one-line summary suitable for printing, matching the
        ``attach_object_size_randomization`` convention.

    Raises:
        ValueError: ``size`` isn't a known key of ``PLATE_SIZE_CONFIGS``, or ``env_cfg`` doesn't
            look like this task's config (no ``scene.plate``) -- e.g. called against a task other
            than the plate-wiping one.
    """
    if size not in PLATE_SIZE_CONFIGS:
        raise ValueError(f"Unknown plate size '{size}'. Valid options: {list(PLATE_SIZE_CONFIGS)}.")
    if getattr(env_cfg.scene, "plate", None) is None:
        raise ValueError(
            "apply_plate_size: env_cfg.scene has no 'plate' entity -- is this the plate-wiping task?"
        )

    cfg = PLATE_SIZE_CONFIGS[size]
    scale = cfg["scale"]
    offset = cfg["offset"]
    rot = cfg["rot"]
    pos = _plate_rest_pos(offset)

    env_cfg.scene.plate.spawn.scale = (scale, scale, scale)
    env_cfg.scene.plate.init_state.pos = pos
    env_cfg.scene.plate.init_state.rot = rot

    rack_plate_event = env_cfg.events.randomize_dish_rack_and_plate
    rack_plate_event.params["rack_to_plate_offset"] = offset
    rack_plate_event.params["plate_rest_rot"] = rot
    rack_plate_event.params["plate_radius"] = cfg["radius"]

    outer_cm = 26.0 * scale
    return f"plate size = '{size}' (outer diameter ~{outer_cm:.1f}cm, spawn scale {scale:.4f})"


@configclass
class OpenarmPlateWipeEnvCfg(pickup_ik_abs_env_cfg.OpenarmPickUpRedCubeEnvCfg):
    """Raw-teleop plate-wiping scene: dish rack + plate + rag, no auto subtask/success signals.

    Subclasses the pick-up task purely to inherit its dual-arm IK-Abs action wiring, camera set,
    and env spacing -- none of that is specific to the red cube. Every cube-specific piece
    (cube_2 itself, its subtask observations, its height-based success termination) is removed
    below.
    """

    def __post_init__(self):
        super().__post_init__()  # cameras, dual-arm IK-Abs actions, pad, cube_2 (removed next)

        # ── Remove every cube -- none of them are part of this task ───────────
        self.scene.cube_2 = None

        # ── Cube-based Mimic/subtask scaffolding from the parent doesn't apply here ──
        # (grasp/lift observed cube_2's height; there is no cube_2 any more.) Left as None
        # rather than rebuilt against the new objects until there are real recorded demos to
        # measure per-object grasp/success thresholds against -- see this module's docstring.
        self.observations.subtask_terms = None
        self.terminations.success = None

        # ── Drop every event/sensor term that names cube_2 -- it no longer exists ────
        # PickUpEventCfg.randomize_cube_2 (inherited via self.events = PickUpEventCfg() in the
        # parent's __post_init__) resets a scene entity called "cube_2" every episode; with the
        # cube gone this is a hard KeyError at env.reset(), not a silent no-op. None is this
        # codebase's established idiom for "skip this term" (see PickUpDomainRandomizationEventCfg
        # for the same pattern).
        self.events.randomize_cube_2 = None

        # The parent's debug contact sensor filters specifically on "Cube_2" (see
        # OpenarmPickUpRedCubeEnvCfg.__post_init__) -- same reasoning, drop it.
        self.scene.contact_right_left_finger = None

        # ── Dish rack: kinematic RigidObject, right side of the pad (right-arm reach). Kinematic
        # (not dynamic) so PhysX never pushes it around via gravity/contacts -- only the
        # randomize_dish_rack_and_plate event term below moves it, once per reset. See this
        # module's docstring for why it needed a RigidBodyAPI at all (it didn't have one before).
        self.scene.dish_rack = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/DishRack",
            init_state=RigidObjectCfg.InitialStateCfg(pos=RACK_POS, rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.UsdFileCfg(usd_path=RACK_USD_PATH),
        )

        # ── Plate: dynamic, spawned already resting in the rack. Defaults to the SMALL size
        # (DEFAULT_PLATE_SIZE) -- call apply_plate_size(env_cfg, "large") after construction to
        # switch, same post-hoc-mutation pattern record_demos_openarm.py already uses for the
        # pick-up task's --randomize_object_size. See PLATE_SIZE_CONFIGS' docstring for why the
        # small/large variants need their own measured offset/rotation, not just a rescaled copy.
        self.scene.plate = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            init_state=RigidObjectCfg.InitialStateCfg(pos=PLATE_REST_POS, rot=PLATE_REST_ROT),
            spawn=sim_utils.UsdFileCfg(
                usd_path=PLATE_USD_PATH,
                scale=(PLATE_SIZE_CONFIGS[DEFAULT_PLATE_SIZE]["scale"],) * 3,
            ),
        )

        # ── Rag: dynamic PhysX deformable (soft) body, spawned flat on the pad's left side
        # (left-arm reach). A rigid collider can't bend/drape/conform the way a real rag does --
        # see this module's docstring -- so this is DeformableObjectCfg, not RigidObjectCfg.
        # reset_scene_to_default (used by PickUpEventCfg.init_robot_pose, inherited unchanged)
        # already handles deformable objects generically -- see isaaclab.envs.mdp.events -- so no
        # extra reset wiring was needed here.
        self.scene.rag = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Rag",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=RAG_REST_POS, rot=RAG_REST_ROT),
            spawn=sim_utils.UsdFileCfg(usd_path=RAG_USD_PATH),
        )

        # ── Per-episode pose randomization -- see this module's docstring for why dish_rack and
        # plate share one event term while rag gets its own. Every range is a conservative
        # starting guess (see docstring); widen/narrow them once teleop shows how they feel.
        self.events.randomize_dish_rack_and_plate = EventTerm(
            func=randomize_dish_rack_and_plate,
            mode="reset",
            params={
                # x=0.06 (was 0.08, alongside RACK_POS moving from 0.40 to 0.36): a wider range
                # let the rack land within ~3cm of the pad's actual far edge, which combined with
                # its now-unrestricted yaw was enough for the plate's lean+physics-drop to
                # occasionally tip it (and, separately, the rag) clean off the pad onto the floor
                # -- verified both landing at z~0. See RACK_POS's own comment for the full history.
                # x now stays within [0.30,0.42] (~9cm clear of the far edge at 0.51), y stays
                # within [-0.08,0.08] (tight around center, well clear of either arm's own
                # shoulder, which sits out around +-0.15 -- see the left arm's rest ee_tcp
                # position measured during the gripper-height check for the rag's max_safe_z).
                "pose_range": {"x": (-0.06, 0.06), "y": (-0.08, 0.08)},
                # No angle limit, per direct instruction -- full turn.
                "yaw_range_deg": (-180.0, 180.0),
                # roll/pitch jitter on the plate's lean, applied in the rack's own local frame
                # (see randomize_dish_rack_and_plate's docstring) -- how much it tips and which
                # way, while a physics drop (not a teleport) finds the actual valid resting
                # contact for that jitter.
                "plate_lean_jitter_deg": {"roll": (-8.0, 8.0), "pitch": (-8.0, 8.0)},
                "plate_drop_height": 0.015,
                # generous enough to accept genuine lean variation, tight enough to reject
                # "tipped out of the rack onto the pad" (pad top is ~9cm below the nominal plate
                # height -- see RACK_TO_PLATE_OFFSET's z component).
                "plate_height_tolerance": 0.05,
                # z-only wasn't enough to confirm the plate actually stayed IN the rack -- see
                # randomize_dish_rack_and_plate's docstring for the failure this catches (tipped
                # sideways out of the rack, landed elsewhere on the pad, but at a height that
                # coincidentally still passed plate_height_tolerance).
                "plate_xy_tolerance": 0.05,
                # Measured reset() wall time end-to-end (all three events together) at 3-20s with
                # settle_steps=150/max_attempts=10 here and 150/20 on randomize_rag below -- a real
                # problem for interactive teleop (pressing the reset key), not a GPU-memory limit.
                # freeze_dynamic_props (runs last, after randomize_rag too) is what actually makes
                # settle_steps safe to trim back down here: the plate keeps getting "free" extra
                # settling from randomize_rag's own steps regardless, and gets a final hard freeze
                # either way, so this doesn't need to fully converge entirely on its own anymore.
                "settle_steps": 150,
                "max_attempts": 6,
                "rack_cfg": SceneEntityCfg("dish_rack"),
                "plate_cfg": SceneEntityCfg("plate"),
                # Explicit (not just relying on the function's own defaults) so apply_plate_size
                # has a guaranteed key to overwrite in this dict for the non-default plate size.
                "rack_to_plate_offset": PLATE_SIZE_CONFIGS[DEFAULT_PLATE_SIZE]["offset"],
                "plate_rest_rot": PLATE_SIZE_CONFIGS[DEFAULT_PLATE_SIZE]["rot"],
                "plate_radius": PLATE_SIZE_CONFIGS[DEFAULT_PLATE_SIZE]["radius"],
            },
        )
        self.events.randomize_rag = EventTerm(
            func=randomize_rag_drop,
            mode="reset",
            params={
                # Height/spin are now free to be pushed for a high per-attempt fold rate, NOT
                # tuned down for safety -- max_safe_z below is what makes a too-vigorous toss
                # (one that reaches the gripper) safe: it gets detected and re-tossed like any
                # other rejected attempt, same as a too-flat one, rather than needing the toss
                # itself to be weak enough to never reach that high.
                # x/y widened from +-0.04 -- now that the rack can land almost anywhere on the pad
                # (see randomize_dish_rack_and_plate's pose_range), the rag needs more room to
                # actually find a landing spot that clears min_rack_separation below when the rack
                # happens to come down near its nominal spawn area, not just a narrow band to
                # sample within.
                "position_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "z": (0.08, 0.16)},
                "tilt_range_deg": {"roll": (-20.0, 20.0), "pitch": (-20.0, 20.0), "yaw": (-180.0, 180.0)},
                "spin_rate_range": (12.0, 24.0),
                # See randomize_dish_rack_and_plate's settle_steps comment -- measured reset()
                # taking 3-20s with this at 150/max_attempts=20; trimmed down for interactive
                # teleop use. The final freeze_dynamic_props pass (this task's actual LAST reset
                # event) means the hard velocity zero at the end of every toss-and-settle attempt
                # doesn't need to be the last word on its own either.
                "settle_steps": 100,
                # Raised from an earlier 10mm (which only rejected outright-flat landings, ~3-7mm)
                # to 30mm -- verified (plate_wiping asset-fix conversation, the "flat to me"
                # follow-up) that a merely-non-flat crease around 10-20mm still reads as close to
                # flat from body_cam's top-down angle. 30mm cleanly separates from that, closer to
                # the visibly obvious folds this function's docstring describes (14-46mm) rather
                # than borderline ones.
                "min_wrinkle_spread": 0.030,
                # left gripper fingers rest at z~0.477, rag rests at z~0.281 -- see
                # randomize_rag_drop's docstring for the snagging failure this guards against.
                "max_safe_z": 0.40,
                # rag rests at z~0.281 (RAG_REST_POS); 0.25 gives a few cm of settling/penetration
                # tolerance while still catching the "flew off the pad onto the ground plane"
                # failure (z near 0) the higher spin_rate_range surfaced -- see docstring.
                "min_safe_z": 0.25,
                # Now that randomize_dish_rack_and_plate can place the rack almost anywhere on the
                # pad (see that event's pose_range), the rack and rag can no longer be assumed far
                # apart just because their nominal positions are -- this is the check that
                # actually enforces it, against wherever the rack ACTUALLY ended up this episode
                # (rack_cfg below), not a fixed position. Deliberately modest (not a big personal-
                # space buffer): the rag's own position_range is much narrower than the rack's, so
                # it has limited room to "dodge" a rack that landed nearby -- set too large, this
                # would mostly just burn through max_attempts without ever finding a satisfying
                # offset. Tuned to stop literal overlap, not to guarantee generous clearance.
                "min_rack_separation": 0.22,
                # History: 20 -> 8 (reset() wall time was 3-20s, a real interactive-teleop
                # problem, not GPU memory) -> 14 (8 wasn't enough headroom when clearing
                # min_rack_separation was still partly down to retry-luck) -> 10 (the deterministic
                # rack-avoidance push meant separation was mostly solved on the first attempt, so
                # 10 seemed enough) -> 12: verified with 15 resets that 10 still left ~27% of
                # resets with a z-bounds violation (mostly from the SAME rack-proximity cause,
                # since the pad-boundary clamp on that push -- see _toss_and_settle_rag's docstring
                # -- can trade away some separation margin when the rack sits close to the rag's
                # own zone). Not fully eliminated by construction the way the basic case is, so
                # retries still matter here more than the comment this replaces assumed.
                "max_attempts": 12,
                "asset_cfg": SceneEntityCfg("rag"),
                "rack_cfg": SceneEntityCfg("dish_rack"),
            },
        )
        # Must run AFTER both events above -- see freeze_dynamic_props' docstring for why an
        # unconditional final freeze is needed even though both of those already do their own.
        self.events.freeze_dynamic_props = EventTerm(
            func=freeze_dynamic_props,
            mode="reset",
            params={"asset_cfgs": [SceneEntityCfg("plate"), SceneEntityCfg("rag")]},
        )
