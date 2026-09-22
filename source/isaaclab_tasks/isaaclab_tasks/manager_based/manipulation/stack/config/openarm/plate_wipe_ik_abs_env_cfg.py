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

import glob
import math
import os

import numpy as np
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

# See stack_joint_pos_env_cfg.py's PAD_NEAR_EDGE_X/PAD_SIZE_X/PAD_SIZE_Y for where these come from.
# Named here (not just inlined as literals) because _place_rag_crumpled derives its PER-ENV,
# radius-aware placement box from them -- see that function's docstring for why a fixed box that
# doesn't account for the object's own radius let the rag hang off the real pad edge.
PAD_X_RANGE = (0.03, 0.51)
PAD_Y_RANGE = (-0.285, 0.285)

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
#
# REVISED (user feedback: the rack/plate group's full 360deg yaw + wide xy range squeezed the rag's
# own placement channel between the robot and the rack, and occasionally left the plate visibly
# NOT seated in the rack -- both traced to the same cause, the rack's own randomization being wider
# than it needs to be for this task's actual requirement. Per direct instruction: the rack now
# spawns in only ONE fixed orientation -- the same one RACK_TO_PLATE_OFFSET/PLATE_REST_ROT were
# measured against in the first place (see those constants' docstrings) -- rather than a random yaw
# every reset; a wrong-looking plate-on-rack result is far more likely when the composed
# offset/rotation math has to hold up under an arbitrary yaw than when it never has to. Moved out to
# 0.40 (from 0.36) to hand the freed-up near-robot space to the rag's own channel -- still ~9cm
# clear of the far edge at 0.51 even with pose_range trimmed down alongside this (see the event
# params below).
#
# REVISED AGAIN (user feedback: wanted still more room freed up near the robot for the rag, now
# that the rack spawns rotated 90deg -- see RACK_REST_ROT below -- and no longer needs as much Y
# clearance from the arms). 0.44 keeps pose_range's reach (x in [0.42,0.46] at the current +-0.02
# range) ~5cm clear of the far edge at 0.51 -- less margin than 0.40's ~9cm, but still comfortably
# more than the ~3cm gap that was directly observed to tip the rack/plate off the pad in the 0.36->
# 0.40 investigation above, and the narrower post-rotation Y footprint means there's no longer a
# competing need to keep x low for arm clearance.
#
# REVISED (user request: push the rack -- and the rag -- further toward the pad's far edge, away
# from the robot body). NOTE the "~5cm clear" reasoning above measured margin from the rack's own
# CENTER to the edge, not from the rack's actual footprint -- direct mesh measurement (dish_rack_
# kinematic.usdc's rail prims) shows the post-90deg-yaw rack is 10cm wide along world X, i.e. its own
# outer edge already reaches RACK_POS[0]+0.05, not just RACK_POS[0]. At the old 0.44 (pose_range max
# 0.46), the rack's own far edge already touched 0.51 (the pad's literal edge) with zero margin,
# despite the comment above claiming 5cm -- so there was no slack left to move further out at all by
# that measure. Pushed anyway, since the rack is a KINEMATIC prop (see the scene.dish_rack comment
# above) -- it is always written directly to this pose every reset regardless of what's under it, so
# it can't topple/fall the way a dynamic object stepping past the pad edge could; a few cm of visual
# overhang past the nominal pad box is not the same failure mode the RACK_POS history above is about
# (a dynamic PLATE actually falling off, verified at the time by watching it land at world z~0). The
# plate itself stays physically supported by the rack's own tray/pegs regardless (see
# randomize_dish_rack_and_plate), not by the pad surface underneath, so moving the rack doesn't
# reopen that failure mode.
RACK_POS = (0.47, 0.0, 0.28)

# REVISED (user feedback + direct verification via rack_yaw_check.py, a script that renders the
# bare rack at yaw in {0, 90, -90, 180}): at the identity yaw used above, the rack's own bracket
# is WIDE along its local Y axis -- top-down, its two support arms run left-right, spanning close
# to both ARM_KEEPOUT_XY_LEFT/_RIGHT and clearly the "horizontal" orientation the user flagged as
# eating the arms' workspace. A 90deg yaw about Z swaps that: the wide bracket axis now runs along
# world X (toward/away from the robot -- "vertical" in the top-down camera the user was looking
# at), leaving a narrow footprint in Y right where the arms need clearance. RACK_TO_PLATE_OFFSET is
# purely vertical (0,0,z) and PLATE_REST_ROT/the randomize event's own jitter are both expressed
# RELATIVE to the rack's frame (composed as rack_rot * local_rot, see
# randomize_dish_rack_and_plate), so rotating this base orientation carries the plate's lean along
# for free -- no separate retuning needed.
RACK_REST_ROT = (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)

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
# NOTE: NOT yet re-verified against RACK_REST_ROT's 90deg rack rotation -- see
# RACK_TO_PLATE_OFFSET_SMALL's docstring for why the old identity-rack values can't just be reused
# as-is (a real drop test with these values landed unstable/nearly-flat, not caught in the slot).
# DEFAULT_PLATE_SIZE is "small", so this hasn't blocked anything yet, but re-measure the same way
# (plate_finalize.py) before actually using apply_plate_size(env_cfg, "large").
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
# REVISED (re-measured after RACK_REST_ROT's 90deg rack rotation): the composed-rotation approach
# -- keep this offset/rot as-is and let randomize_dish_rack_and_plate's own
# ``target_rot = rack_rot * local_rot`` carry it along -- is mathematically a rigid re-yaw of the
# whole rack+plate configuration and SHOULD preserve the same relative contact, but direct
# verification (plate_settle_diag.py: wrote the composed target_rot with the rotated rack, then let
# the event's own settle_steps run) instead showed the plate visibly rocking/tumbling away from
# that target every time (85deg tilt at write-time -> 98deg at 5 steps -> 66deg at 15 -> 134deg at
# 50, never converging) -- NOT a small settle, a genuinely different and unpredictable landing each
# reset, which lines up exactly with the user's screenshot showing the plate floating flat above
# the rack instead of leaning in it. Root cause (plate_finalize.py's bbox/drop investigation): the
# small plate's inner hole is wider than the peg cluster's own footprint (see the "flat drop" note
# below), so which local axis the lean happens about and where the drop starts BOTH matter for
# which contact it actually catches -- rolling about the rack's local X axis (correct for the old
# identity-yaw rack) no longer catches the same way once local X points along a different world
# direction. Re-measured with a REAL drop (not a direct write) against the rotated rack: leaning
# about local Y (pitch, not roll) 85deg, started ~2.5cm off-origin along the rack's local X and a
# small height above, converges to a genuinely stationary rest (speed/angspeed ~1e-4, not just
# "under a loose tolerance") that visibly reads as caught in the peg slot, not floating -- see
# plate_finalize.py. RACK_TO_PLATE_OFFSET_SMALL/PLATE_REST_ROT_SMALL below are that measured local
# offset/rotation (in the rack's own frame, so they still compose correctly if yaw_range_deg is
# ever widened again).
#
# REVISED (user report + reference photo: the ~2.5cm-off-center recipe above landed the plate
# leaning against a single peg near one EDGE of the rack, touching only that peg/the pad beside the
# rack, instead of standing centered inside the peg cluster the way the reference photo shows --
# same complaint as the pre-rotation "hangs off to one side" failure mode further up this docstring,
# recurring here after RACK_REST_ROT's rotation and the offset's own re-measurement). Re-tested
# directly against the real registered task env (gym.make + a genuine env.reset(), not a solo
# harness) with a battery of centered/off-center/roll-vs-pitch candidates, each direct-written (no
# free-fall -- a free-fall drop onto this symmetric 4-peg cluster was separately confirmed chaotic,
# see randomize_dish_rack_and_plate's docstring) and run 400 physics steps (not just settle_steps)
# to rule out a "looks fine at first, keeps sliding" result the same way the +-2.5cm recipe above
# was already caught doing under lean jitter. A CENTERED offset (0,0,z) with a PURE 85deg pitch (no
# small residual roll/yaw noise in the quaternion -- the old recipe's tiny x/z quaternion components
# turned out to matter: the same offset with the noisy quaternion measured earlier was still visibly
# tumbling at 400 steps, |w|~5.8, while the clean pure-pitch version settles to |v|<0.001,|w|<0.01)
# is the only candidate of six tested that both converges to a genuine stationary rest AND stays
# reasonably centered (settles ~1.5cm off-center in the rack's local y, not the old ~2.5-3cm hard
# lean to one edge) -- every off-center starting offset tested (0.0125m, and the front/back y=+-3cm
# pairs) either kept tumbling past 400 steps or, in one case, launched the plate clean off the pad.
#
# REVISED (re-verified through the actual randomize_dish_rack_and_plate reset event, not a solo
# direct-write bypass: 1 of 5 resets with the exact-centered (0,0,z) write above landed badly --
# 9cm off in xy, still visibly moving at the check's own settle_steps horizon). Root cause: writing
# EXACTLY centered starts the plate balanced squarely on the peg cluster's own symmetry line, which
# is the least stable place to start (any solver noise can send it left or right unpredictably) --
# the four resets that DID land well independently converged to the same slightly-off-center rest,
# not to the symmetric write target itself. Re-expressed as that already-converged local offset
# (measured the same way, direct-write + 400-step check through the real reset event) instead of the
# exact symmetric one, so the write starts AT the natural rest instead of forcing a knife-edge
# balance every time. plate_rest_rot unchanged (same clean pure-85deg pitch).
RACK_TO_PLATE_OFFSET_SMALL = (-0.0153, -0.0001, 0.0953)
PLATE_REST_ROT_SMALL = (0.737277336810124, 0.0, 0.6755902076156602, 0.0)

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

# REVISED (user feedback: the rag "penetrating" the arm, visibly draped across its FOREARM, not
# just close to the fingertip cluster -- verified directly via robot.data.body_pos_w at the default
# reset pose, same methodology ARM_KEEPOUT_XY_LEFT/_RIGHT themselves were measured with): those two
# constants are a single averaged POINT for the gripper/camera cluster, and that's fine for the
# plate above (its own radius already dominates required clearance, see ARM_KEEPOUT_RADIUS's
# comment) -- but measuring the actual link chain shows link5 through the fingers forms a nearly
# straight ~20cm SEGMENT at close to constant y (x runs ~0.08 to ~0.28 while y stays ~0.153-0.154
# the whole way), not a point. Pushing the rag away from just the far end of that segment (what the
# point-based keepout was doing) can push it laterally ALONG the segment's own length instead of
# away from it -- exactly the failure the screenshots showed. _place_rag_crumpled's arm-avoidance
# push uses point-to-SEGMENT distance against this instead of point-to-point, so "away from the arm"
# means away from the nearest part of the whole forearm, not just its end.
ARM_FOREARM_SEGMENT_LEFT = ((0.08, 0.154), (0.28, 0.154))
ARM_FOREARM_SEGMENT_RIGHT = ((0.08, -0.154), (0.28, -0.154))


def _closest_point_on_segment(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Closest point on segment ``a``-``b`` to each xy point in ``p`` -- all ``(..., 2)`` tensors,
    ``a``/``b`` broadcastable against ``p``. Standard clamped-projection formula."""
    ab = b - a
    ab_len_sq = (ab * ab).sum(dim=-1, keepdim=True).clamp_min(1e-9)
    t = ((p - a) * ab).sum(dim=-1, keepdim=True) / ab_len_sq
    t = t.clamp(0.0, 1.0)
    return a + t * ab


def _quat_mul(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]):
    """wxyz quaternion product q1*q2 -- plain Python (no torch), for module-load-time constants."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_apply(q: tuple[float, float, float, float], v: tuple[float, float, float]):
    """Rotate vector ``v`` by wxyz quaternion ``q`` -- plain Python, same math as
    ``isaaclab.utils.math.quat_apply`` but usable on plain tuples at module-load time."""
    w, x, y, z = q
    vx, vy, vz = v
    uv = (y * vz - z * vy, z * vx - x * vz, x * vy - y * vx)
    uuv = (y * uv[2] - z * uv[1], z * uv[0] - x * uv[2], x * uv[1] - y * uv[0])
    return (
        vx + 2 * (w * uv[0] + uuv[0]),
        vy + 2 * (w * uv[1] + uuv[1]),
        vz + 2 * (w * uv[2] + uuv[2]),
    )


def _plate_rest_pos(offset: tuple[float, float, float]) -> tuple[float, float, float]:
    # REVISED (bug found while diagnosing the "plate touches the table" report): this used to be a
    # plain unrotated add of ``offset`` onto RACK_POS, which was correct back when RACK_REST_ROT was
    # identity but silently went stale once the rack got its 90deg RACK_REST_ROT yaw -- the actual
    # per-reset placement (randomize_dish_rack_and_plate) rotates the offset by the rack's real
    # rotation before adding it, so this static ``scene.plate.init_state`` (only ever visible for
    # one frame before the first reset event runs) was pointing the plate at a different spot than
    # every reset actually uses. Harmless for normal play (the reset event always overwrites this
    # before the first observation), but fixed to match so the raw spawned stage isn't misleading.
    rx, ry, rz = _quat_apply(RACK_REST_ROT, offset)
    return (RACK_POS[0] + rx, RACK_POS[1] + ry, RACK_POS[2] + rz)


# Kept as the names the rest of this module (and its docstring) already refers to -- resolve to
# whichever size is DEFAULT_PLATE_SIZE. apply_plate_size() below overrides all of this on an
# already-constructed env_cfg for the non-default choice; these are just what the scene/event
# params are initially built with in __post_init__.
#
# RACK_TO_PLATE_OFFSET/PLATE_REST_ROT stay the RAW local (pre-rack-rotation) values -- they're also
# used as randomize_dish_rack_and_plate's own default parameters, which are meant to be local (the
# function composes them with the rack's actual rotation itself). PLATE_REST_ROT_WORLD is the
# separate, already-composed value scene.plate's static init_state needs (see _plate_rest_pos above
# for why the plain PLATE_REST_ROT alone isn't correct there any more either).
RACK_TO_PLATE_OFFSET = PLATE_SIZE_CONFIGS[DEFAULT_PLATE_SIZE]["offset"]
PLATE_REST_ROT = PLATE_SIZE_CONFIGS[DEFAULT_PLATE_SIZE]["rot"]
PLATE_REST_POS = _plate_rest_pos(RACK_TO_PLATE_OFFSET)
PLATE_REST_ROT_WORLD = _quat_mul(RACK_REST_ROT, PLATE_REST_ROT)

# Measured the same way: dropped flat onto the pad (as a PhysX deformable body, not a rigid
# collider -- see this module's docstring) and read back its settled nodal centroid.
#
# REVISED (user feedback: rag spawns too close to the robot body): the original (0.30, 0.15) put
# the rag's nominal centroid only ~4cm from ARM_KEEPOUT_XY_LEFT (0.26, 0.153) -- the left arm's OWN
# resting gripper/camera cluster -- so body_cam renders consistently showed the rag landing right
# up against or draped over the arm hardware, not just "on the pad near the arm." Verified via the
# same gym.make + real env.reset() methodology used for the plate fixes above (root/nodal state
# logged and body_cam rendered every reset).
#
# REVISED AGAIN (user feedback after trying (0.18, 0.20) live: still too close to the robot body):
# that first fix only optimized distance from the arm's resting GRIPPER cluster -- it actually
# moved x from 0.30 down to 0.18, i.e. noticeably CLOSER to the robot's own base/mount at x=0 than
# the original was, even though it was farther from the gripper cluster specifically. "Robot body"
# is the whole robot -- base included, not just where the idle gripper happens to hang -- so pulling
# x in toward the base traded one kind of closeness for another instead of fixing it.
# (0.35, 0.18) instead pushes OUT in x (~39cm from the robot origin, farther than the ORIGINAL
# (0.30, 0.15)'s ~33cm, not just farther than the first attempt's ~27cm) while keeping y far enough
# from ARM_KEEPOUT_XY_LEFT's y=0.153 for ~9cm clearance from the gripper cluster too -- both
# distances improved together this time, not traded off against each other. Still within the
# pick-up task's reach-validated x band (see cube_2's pose_range in pickup_ik_abs_env_cfg.py) and
# with enough margin from the pad's y=0.285 edge for the toss's own tumble drift not to carry it
# off the pad (an earlier y=0.24 attempt did exactly that -- fell off the pad edge in 3 of 8 test
# resets; see the randomize_rag event's params below for the toss energy this pairs with).
#
# REVISED AGAIN (this task moved from a toss to direct template placement -- see
# _place_rag_crumpled -- which now derives its own per-env, radius-aware safe box from PAD_X_RANGE/
# PAD_Y_RANGE and actively pushes clear of both arms every reset; see that function and RAG_SCALE's
# docstrings). This nominal spot barely matters any more beyond being A reasonable starting point
# for that push -- (0.35, 0.18) sits almost exactly ON the arm's own keepout circle at RAG_SCALE, so
# the push was doing all the real work anyway. (0.30, 0.12) is closer to the actual feasible
# placement region worked out for RAG_SCALE (see that constant's docstring), so a typical reset
# pushes less far from its starting point -- purely a minor efficiency/predictability tweak, not a
# correctness one, since the push+clamp guarantees a valid final position regardless of where this
# starts.
#
# REVISED AGAIN (user feedback + suggestion: occasional arm contact persisted, at y=0.12 the rag
# sits notably closer to the LEFT arm than the right one, so the push only ever has to fight ONE
# keepout circle at a time and only has the narrow strip between that circle and the pad edge to
# work with -- user's own suggestion was to use the gap BETWEEN the two arms instead). y=0 is
# equidistant from ARM_KEEPOUT_XY_LEFT/_RIGHT by construction (they're mirrored about y=0), so a
# nominal spot there starts already roughly clearing both keepouts before the push even runs, rather
# than starting deep inside one of them -- checked directly: (0.18, 0.0) is ~17.7cm from EITHER
# arm's keepout point already, versus (0.30, 0.12)'s ~5cm from the left one alone. The push-then-
# clamp loop still runs every reset (this is a better STARTING point, not a substitute for it).
#
# REVISED AGAIN (user feedback: narrowed the rack's own randomization and moved RACK_POS out to
# 0.40 specifically to hand the freed-up near-robot space to the rag -- see RACK_POS's docstring).
# Nudged x from 0.18 to 0.22 to actually use that freed room (still ~15.8cm from either arm's
# keepout point, comfortably clearing arm_required at this rag's typical planar_radius) instead of
# leaving it clustered near the robot-side edge of its own placement channel.
#
# REVISED (user request, alongside RACK_POS's own push toward the far edge -- see that constant's
# docstring): moved from 0.22 to 0.30 to move with it. This alone wouldn't have been enough --
# _place_rag_crumpled's min_rack_separation push (see its own EventTerm params) actively shoves the
# rag AWAY from wherever the rack actually ends up by at least min_rack_separation+rack_avoid_margin;
# at the old 0.22 separation setting, a rag this close to the now-further-out rack would just get
# pushed straight back toward the robot every reset, silently undoing this move. Reduced together
# with min_rack_separation (0.22 -> 0.12) and _place_rag_crumpled's own hardcoded x_hi safety cap
# (0.34 -> 0.38, see that function) so the rag can actually settle out here instead of being pushed
# back by a stale separation requirement sized for the rack's old position.
RAG_REST_POS = (0.30, 0.0, 0.2811)
RAG_REST_ROT = (1.0, 0.0, 0.0, 0.0)

# REVISED (user feedback: the rag "penetrates the robot arm oftenly"): at native scale, this rag's
# own crumpled footprint (~20cm planar radius -- see _load_rag_crumple_template_radii's docstring)
# is simply too big to place with real clearance from the left arm's resting gripper cluster
# anywhere reachable on this pad -- the two arms are only ~31cm apart center-to-center, so a
# keepout circle at that radius from EACH arm nearly spans the whole gap between them, same problem
# the plate already solves with a small/large SCALE option (see PLATE_SIZE_CONFIGS), not by pushing
# harder.
#
# REVISED AGAIN, then REVERTED (user feedback: still sometimes hangs off / falls off the pad edge,
# and still reads as flat where it lands): tried 0.5 next, reasoning that 0.65's ~10-12.5cm crumpled
# radius was right at the edge of geometric feasibility (worked out analytically -- the max radius
# that can clear BOTH the arm-keepout requirement and keep the full footprint, not just the
# centroid, inset from the pad's real edges comes out to ~0.115m) rather than comfortably inside it.
# 0.5 (~8.8cm radius) IS comfortably inside that geometric region -- but broke something else: this
# rag's PhysX deformable material (self-collision filter distance, stiffness) is authored as
# ABSOLUTE values in the USD, not something UsdFileCfg.scale rescales along with the mesh geometry.
# Shrunk far enough, those fixed absolute values become relatively too stiff for the now-smaller
# mesh, and a placed crumple springs back toward flat almost immediately instead of holding its
# fold -- verified directly: at 0.5, 13 of 15 resets collapsed to near-zero z-spread within the same
# settle_steps that held a real crumple fine at 0.65 (0.65's own instability was mild by comparison
# -- mostly held shape, one low outlier). The scene-wide disturbance from that instability even
# knocked the plate over in a couple of those 0.5-scale resets. So back to 0.65 -- the actual fix
# for pad-edge overhang is the per-env radius-aware placement box in _place_rag_crumpled (a real
# bug, independent of scale: it clamped the CENTROID without ever subtracting the object's own
# radius from the bounds), and the arm clearance is now a partial (not full) keepout factor -- see
# that function's docstring for why full clearance genuinely isn't achievable at 0.65 either, and
# what "partial" means concretely.
RAG_SCALE = 0.65


def _settle_plate(
    env: ManagerBasedEnv,
    plate,
    ids: torch.Tensor,
    start_pos: torch.Tensor,
    start_rot: torch.Tensor,
    settle_steps: int,
):
    """Write the plate directly to its known-good target pose (see
    ``randomize_dish_rack_and_plate``'s docstring for why this is a direct write now, not a drop
    from height) with zero velocity, then physically step it forward a short, fixed number of
    steps so contact can relax away the last mm or two of interpenetration an analytic pose leaves
    -- not to discover whether the pose itself is valid, which prior measurement already settled.
    Stepping physics inside a "reset" event is safe here for the same reason it's safe in
    ``randomize_rag_drop``: every "reset"-mode event runs before the episode's first observation is
    computed, so the settled result IS the starting state.

    Ends by explicitly zeroing the plate's velocity, the same "hard freeze" ``_place_rag_crumpled``
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
    settle_steps: int,
    rack_cfg: SceneEntityCfg,
    plate_cfg: SceneEntityCfg,
    rack_to_plate_offset: tuple[float, float, float] = RACK_TO_PLATE_OFFSET,
    plate_rest_rot: tuple[float, float, float, float] = PLATE_REST_ROT,
    plate_radius: float = PLATE_SIZE_CONFIGS[DEFAULT_PLATE_SIZE]["radius"],
):
    """Place ``dish_rack`` and ``plate`` as one rigid group (shared xy offset AND shared yaw
    rotation about the rack's own pivot), then place the plate DIRECTLY at its known-good leaning
    pose relative to the rack -- instead of a physics drop-and-retry (what this function used to do)
    or a single hand-measured, always-identical relative pose (what this task did before that).

    Why the rack and plate move together: the plate rests leaning IN the rack (see
    ``rack_to_plate_offset``); randomizing either one independently would drift the plate out of
    the rack -- floating next to it or clipped through a wall, depending on which way two
    independent draws happened to point. Both effects (xy translation, yaw rotation) are applied to
    dish_rack's actual root pose, and to the plate's TARGET pose via the same rigid transform
    (``rack_to_plate_offset`` rotated by the same yaw, ``plate_rest_rot`` composed with the same
    yaw) -- so the plate's target always sits correctly relative to wherever the rack ends up, not
    just the original RACK_POS/PLATE_REST_POS. dish_rack is written directly (kinematic, no
    settling needed -- see the parent docstring on why it's a RigidObject at all).

    ``rack_to_plate_offset``/``plate_rest_rot`` default to the module-level constants (which
    resolve to whichever size DEFAULT_PLATE_SIZE names), but are real parameters, not just
    module-global reads, specifically so ``apply_plate_size`` can override them per env_cfg
    instance for the OTHER plate size -- see PLATE_SIZE_CONFIGS' docstring for why the small and
    large plates need genuinely different measured values here, not a scaled copy of one offset.

    REVISED (user feedback: resetting -- pressing 'r' in the interactive teleop tool -- took too
    long; confirmed as real physics-stepping wall time, not a GPU/CPU headroom problem). This used
    to drop the plate from a small height above the target with a random ``plate_lean_jitter_deg``
    tilt, physics-settle it for ``settle_steps``, check the settled pose against the target within a
    tolerance, and retry (fresh jitter draw) up to ``max_attempts`` times on a bad landing -- the
    same pattern ``randomize_rag_drop`` used for the rag. That was worth it while
    RACK_TO_PLATE_OFFSET_SMALL/_LARGE and PLATE_REST_ROT_SMALL/_LARGE were still being re-measured
    (see those constants' own docstrings for that whole history), since physics was the only way to
    tell a genuinely-caught-by-a-peg lean from one that just clips through the rack. Once measured,
    though, that same physics settle turned out to be extremely repeatable across yaw/jitter draws
    ON AVERAGE (verified over 20+ resets per size, both here and while re-measuring those constants:
    settled root height above the rack varied by around 1cm, xy by a few mm). This writes the
    jittered target pose directly (no drop offset) and runs a short ``settle_steps`` afterward
    purely so contact can relax away the last mm or two of interpenetration an analytic pose leaves
    -- much cheaper than a real drop, since the pose is already known-good, not discovered from
    scratch.

    A first version of this direct-write removed the tolerance check and retry loop entirely, on
    the theory that "extremely repeatable on average" meant "always good enough" -- verified WRONG
    over the next 10-reset check: a small tail of (lean jitter, rack yaw) combinations still produced
    a genuinely bad landing (one plate ended up ~28cm below the rack, i.e. fell off the pad onto the
    world ground plane), not just a slower-to-settle one -- a short, fixed settle has no way to catch
    that a bare write doesn't retry. So a coarse sanity check and a SMALL capped retry (3 attempts,
    each a fresh jitter draw, each still only the same short settle_steps) are back -- see the
    function body -- just loose enough to essentially never fire on a normal landing, tight enough to
    catch a plate that's fallen away from the rack rather than merely leaning a little differently.

    REVISED (found while re-verifying RACK_TO_PLATE_OFFSET_SMALL/PLATE_REST_ROT_SMALL against the
    real reset event: with plate_lean_jitter_deg now zeroed, every retry attempt above used to
    redraw the exact same (zero) lean and reproduce the exact same bad landing -- the retry loop had
    quietly become a no-op against exactly the failure mode it exists to catch). Each attempt now
    also draws a small (+-4mm) position jitter independent of the lean angle, so a failed attempt has
    something to try differently, and the cap was raised 3 -> 5 to give that extra draw more chances.
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

    # ── place the plate directly at its known-good leaning pose relative to the now-placed rack,
    # with a cheap, capped retry on a bad landing -- see docstring for why this isn't a bare,
    # unchecked write ──────────────────────────────────────────────────────────────────────────
    offset_local = torch.tensor(rack_to_plate_offset, device=device)
    plate_local_rot_const = torch.tensor(plate_rest_rot, device=device)
    lean_range_list = [plate_lean_jitter_deg.get(key, (0.0, 0.0)) for key in ("roll", "pitch")]
    lean_ranges = torch.deg2rad(torch.tensor(lean_range_list, device=device))

    # read the RACK's actual just-written pose for these envs.
    rack_pos_now = rack.data.root_pos_w[env_ids]
    rack_rot_now = rack.data.root_quat_w[env_ids]

    # REVISED (found while re-verifying RACK_TO_PLATE_OFFSET_SMALL above through the real reset
    # event: a bad landing's retry used to redraw ``lean`` from plate_lean_jitter_deg, but that's
    # now (0,0) -- so every retry attempt wrote the EXACT same target and reproduced the EXACT same
    # bad landing, making the retry loop below pure dead weight against the "starts on a knife-edge"
    # failure mode that motivated re-centering the offset in the first place. A small POSITION
    # jitter, independent of the lean angle, gives a failed attempt something different to try --
    # just enough to break exact symmetry without reopening the +-8deg lean instability that caused
    # the original "touches the table" report.
    retry_pos_jitter_range = torch.tensor([[-0.004, 0.004], [-0.004, 0.004]], device=device)

    pending_ids = env_ids
    pending_rack_pos = rack_pos_now
    pending_rack_rot = rack_rot_now
    for attempt in range(5):
        m = len(pending_ids)
        lean = math_utils.sample_uniform(lean_ranges[:, 0], lean_ranges[:, 1], (m, 2), device=device)
        lean_quat = math_utils.quat_from_euler_xyz(lean[:, 0], lean[:, 1], torch.zeros(m, device=device))
        # jitter applied in the LOCAL (pre-yaw, as-measured) frame, THEN the group's yaw on top --
        # see docstring on why this order keeps "leaning direction" meaning the same regardless of
        # how the rack itself has been rotated this episode.
        jittered_local_rot = math_utils.quat_mul(lean_quat, plate_local_rot_const.expand(m, 4))
        target_rot = math_utils.quat_mul(pending_rack_rot, jittered_local_rot)
        pos_jitter_xy = math_utils.sample_uniform(
            retry_pos_jitter_range[:, 0], retry_pos_jitter_range[:, 1], (m, 2), device=device
        )
        jittered_offset_local = offset_local.expand(m, 3).clone()
        jittered_offset_local[:, 0:2] += pos_jitter_xy
        target_pos = pending_rack_pos + math_utils.quat_apply(pending_rack_rot, jittered_offset_local)

        _settle_plate(env, plate, pending_ids, target_pos, target_rot, settle_steps)

        # REVISED (user report: plate ends up touching the pad/table instead of the rack, every
        # reset -- traced to this check, not just plate_lean_jitter_deg above): 0.08m was loose
        # enough to silently PASS a plate that had already slid most of the way down off the peg
        # contact onto the tray/pad -- verified directly (see plate_lean_jitter_deg's docstring):
        # a jittered landing measured only ~4-8cm below target at this check's own settle_steps
        # horizon, comfortably under the old 0.08 threshold, yet kept sliding to within 0.3-1.6cm of
        # the rack root (i.e. onto the pad) once physics kept running into the actual episode. Even
        # with jitter now zeroed, this stays tight (not just reverted) as a real regression guard:
        # the verified-stable zero-jitter pose itself only moves <1mm from target once settled, so
        # 0.02 has ample margin above normal settle noise while still catching a genuine bad landing
        # (off the rack, through the pad) instead of only a catastrophic one.
        actual_pos = plate.data.root_pos_w[pending_ids]
        bad = (actual_pos[:, 2] - target_pos[:, 2]).abs() > 0.02
        bad |= (actual_pos[:, 0:2] - target_pos[:, 0:2]).norm(dim=-1) > 0.02
        if not bool(bad.any()) or attempt == 4:
            break
        pending_ids = pending_ids[bad]
        pending_rack_pos = pending_rack_pos[bad]
        pending_rack_rot = pending_rack_rot[bad]


RAG_TEMPLATE_DIR = os.path.join(PLATE_WIPING_ASSET_DIR, "rag_crumple_templates")
_RAG_TEMPLATES_CACHE: dict[str, torch.Tensor] = {}


_RAG_TEMPLATE_RADII_CACHE: dict[str, torch.Tensor] = {}


def _load_rag_crumple_templates(device: str) -> torch.Tensor:
    """Load (and cache, per device) the pre-captured crumpled-rag shapes from
    ``RAG_TEMPLATE_DIR`` as one ``(num_templates, num_nodes, 3)`` tensor, centroid-relative (mean
    zero on all 3 axes) so a caller can drop any one of them at any target center/yaw with a plain
    rotate + translate. See ``randomize_rag_drop``'s docstring for why templates replace the
    physics toss this task used to do at reset time.

    Captured once, offline, by a throwaway script running this task's OLD toss-and-tumble physics
    (the same ``spin``/``tilt``/height-drop recipe this function used to run every reset) with a
    generous settle and no time budget, keeping only the landings that came out genuinely crumpled
    (real z-spread, ranked by that over a large batch -- see the capture script's own history for
    why footprint, not spread, turned out to be the metric this rag mesh just can't be tuned much
    below ~30cm on: even the LOWEST-energy tosses tried never produced a footprint under that,
    across 50+ attempts -- so template selection optimizes for genuinely 3D/creased over flat,
    accepts that footprint stays roughly rag-sized, and leans on ``_load_rag_crumple_template_radii``
    /the arm-keepout push in ``_place_rag_crumpled`` to keep that real footprint clear of the arms
    rather than trying to shrink it away) and saving their settled nodal positions, recentered on
    their own centroid. Multiple templates (not just one) so resets still look different from each
    other -- picking a random template plus a random yaw per reset is what stands in for the old
    toss's own randomness now.
    """
    if device not in _RAG_TEMPLATES_CACHE:
        paths = sorted(glob.glob(os.path.join(RAG_TEMPLATE_DIR, "template_*.npy")))
        if not paths:
            raise FileNotFoundError(
                f"No rag crumple templates found under {RAG_TEMPLATE_DIR} -- randomize_rag_drop "
                "needs at least one template_*.npy (num_nodes, 3) file, centroid-relative."
            )
        arrays = [np.load(p) for p in paths]
        stacked = np.stack(arrays, axis=0)
        _RAG_TEMPLATES_CACHE[device] = torch.tensor(stacked, dtype=torch.float32, device=device)
    return _RAG_TEMPLATES_CACHE[device]


def _load_rag_crumple_template_radii(device: str) -> torch.Tensor:
    """Per-template planar (xy) bounding radius -- max centroid-to-node distance projected onto
    xy -- as one ``(num_templates,)`` tensor. A CIRCULAR bound, not the tighter rectangular
    footprint the capture script filtered on, specifically because ``_place_rag_crumpled`` applies
    a random YAW to each template before placing it: a rectangular footprint's own extent changes
    with yaw, a circular one doesn't, so this is what the arm-keepout push there can actually rely
    on being correct regardless of which way a given placement happens to be rotated.
    """
    if device not in _RAG_TEMPLATE_RADII_CACHE:
        templates = _load_rag_crumple_templates(device)
        planar = templates[..., 0:2]
        radii = planar.norm(dim=-1).amax(dim=-1)
        _RAG_TEMPLATE_RADII_CACHE[device] = radii
    return _RAG_TEMPLATE_RADII_CACHE[device]


def _place_rag_crumpled(
    env: ManagerBasedEnv,
    rag: DeformableObject,
    ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    settle_steps: int,
    rack=None,
    min_rack_separation: float = 0.0,
    rack_avoid_margin: float = 0.05,
):
    """Drop a random pre-crumpled template (see ``_load_rag_crumple_templates``) at a random
    yaw and a randomized xy target near the rag's own nominal spawn position, write it straight to
    sim, and run a short, fixed settle purely to relax the couple mm of self-penetration a raw
    rigid transform of a real, non-convex crumpled shape can leave at its folds -- NOT to discover
    whether the shape is crumpled, which the template already guarantees.

    Ends by explicitly zeroing nodal velocity, same reasoning ``randomize_rag_drop``'s old
    toss-based version used: self-collision contacts on a fold can keep chattering at low amplitude
    past the point the shape has visually finished settling, so a hard zero-velocity write is a
    stronger guarantee than trusting PhysX's own sleep/damping heuristics to have fully caught up
    by the time ``settle_steps`` runs out.

    If ``rack`` is given, this ACTIVELY pushes the landing target's xy away from the rack's current
    position -- by exactly enough to clear ``min_rack_separation`` (plus ``rack_avoid_margin``),
    not just resampled and hoped for. Unchanged from the old toss-based version -- see git history /
    the plate_wiping asset-fix conversation for why a single push-then-clamp pass, or a purely
    random retry, both turned out not to be enough on their own; the iterated push-then-clamp below
    is what actually converges to a valid, on-pad, clear-of-the-rack target.

    REVISED (user feedback: the rag "penetrates the robot arm oftenly" in the real interactive
    tool): this rag mesh's real crumpled footprint is roughly rag-sized (~20-25cm across -- see
    ``_load_rag_crumple_templates``' docstring), while RAG_REST_POS sits close to the arm's own
    reach by construction (it has to, to stay reachable) -- nowhere near enough clearance for an
    object this size without an explicit avoidance check. The OLD version only checked/avoided the
    RACK's position, on the (wrong) assumption that RAG_REST_POS being "far enough" from the arms by
    construction made an explicit arm check unnecessary -- exactly the same mistake
    ``ARM_KEEPOUT_XY_LEFT``/``_RIGHT`` already exist to avoid for the plate.

    REVISED AGAIN (same complaint persisted after adding a point-based arm keepout: screenshots
    showed the rag draped across the arm's FOREARM, not just close to the fingertip cluster).
    Measured directly (robot.data.body_pos_w at the default reset pose, same way
    ARM_KEEPOUT_XY_LEFT/_RIGHT were themselves measured) that link5 through the fingers form a
    nearly straight ~20cm SEGMENT at close to constant y (see ARM_FOREARM_SEGMENT_LEFT/_RIGHT), not
    a point. Two different fixes built on that measurement -- a batched random-candidate-search, and
    point-to-SEGMENT distance dropped into the existing push-then-clamp loop -- BOTH caused a reset
    to hang for many minutes with no GPU activity (i.e. stuck computing, not frozen): whatever
    position either approach converged to, on some draws, appears to leave PhysX's contact solver
    stuck resolving something on the very next settle step. Root cause not pinned down given the
    time already sunk chasing it (two independent, structurally different implementations both
    triggering it points at something about how large a correction is being asked for near this
    segment, not a bug specific to either implementation). Reverted BOTH back to the simpler,
    verified-non-hanging point-based keepout (``ARM_KEEPOUT_XY_LEFT``/``_RIGHT``, same as the
    plate's own) -- it doesn't fully solve the forearm-overlap complaint (a point still can't
    represent a whole segment), but it is known stable, which matters more right now. Revisiting the
    segment model is future work, to be done WITHOUT wiring it straight into a live reset event
    again until the hang is understood in isolation.
    """
    n = len(ids)
    templates = _load_rag_crumple_templates(str(rag.device))
    template_radii = _load_rag_crumple_template_radii(str(rag.device))
    template_idx = torch.randint(0, templates.shape[0], (n,), device=rag.device)
    local = templates[template_idx]  # (n, num_nodes, 3), centroid-relative
    planar_radius = template_radii[template_idx]  # (n,)

    nominal_xy = rag.data.default_nodal_state_w[ids][..., 0:2].mean(dim=1)

    pos_range_list = [position_range.get(key, (0.0, 0.0)) for key in ("x", "y")]
    pos_ranges = torch.tensor(pos_range_list, device=rag.device)
    xy_offset = math_utils.sample_uniform(pos_ranges[:, 0], pos_ranges[:, 1], (n, 2), device=rag.device)
    target_xy = nominal_xy + xy_offset

    # Point-based keepout (same convention as the plate's ARM_KEEPOUT_RADIUS/_MARGIN) -- see
    # docstring for why the more accurate segment-based version isn't in use right now.
    keepout_centers = []
    keepout_required = []
    if rack is not None and min_rack_separation > 0.0:
        keepout_centers.append(rack.data.root_pos_w[ids, 0:2])
        keepout_required.append(torch.full((n,), min_rack_separation + rack_avoid_margin, device=rag.device))
    arm_required = 0.85 * planar_radius + ARM_KEEPOUT_RADIUS + ARM_KEEPOUT_MARGIN
    keepout_centers.append(torch.tensor(ARM_KEEPOUT_XY_LEFT, device=rag.device).expand(n, 2))
    keepout_required.append(arm_required)
    keepout_centers.append(torch.tensor(ARM_KEEPOUT_XY_RIGHT, device=rag.device).expand(n, 2))
    keepout_required.append(arm_required)

    # Pad-interior box derived PER-ENV from each env's own planar_radius -- NOT a fixed constant.
    # A first version clamped the CENTROID to a fixed box picked to roughly line up with the pad
    # edges, without subtracting the rag's own radius from it -- so the box let the centroid sit
    # close enough to a real edge that the template's outer extent (which the box never accounted
    # for) still hung past it, letting the rag slide/tumble off the pad edge in practice (verified:
    # exactly the failure the user reported). Insetting these bounds by planar_radius + a small
    # margin on every side is what actually keeps the FULL footprint on the pad, not just its center.
    #
    # REVISED (RAG_REST_POS moved to the GAP BETWEEN the two arms, y=0, instead of tucked to the
    # left arm's side -- see that constant's docstring): the box now stays CENTERED around that gap
    # (a symmetric +-0.15m y band) rather than spanning from just off the pad edge all the way to
    # y=0.02, and capped in x to the channel between the robot base and the rack's own roaming zone
    # (x~0.30-0.42) -- both bounded with min/max against the true pad-margin inset so a large
    # planar_radius still can't push the box outside the real pad. Old y range (0.02 up to the pad
    # edge at 0.285) meant a violating draw only ever had ONE direction to escape toward (further
    # from the pad edge, i.e. further into the left arm) and comparatively little room before hitting
    # that edge; centering the box gives the push room on both sides of a central position that
    # already starts roughly equidistant from both arms.
    # x_hi raised 0.30 -> 0.34 alongside RACK_POS moving out to 0.40 with a tighter own pose_range
    # (now x:[0.38,0.42] -- see that constant's docstring): still comfortably clear of the rack's
    # new, narrower roaming zone even before the dynamic min_rack_separation push below runs, while
    # handing the rag the room RACK_POS's own move freed up instead of leaving it unused.
    #
    # REVISED (user request: push both the rack and the rag further toward the pad's far edge --
    # see RACK_POS's and RAG_REST_POS's own docstrings). Raised 0.34 -> 0.38 alongside RACK_POS's
    # move out to 0.47: the rack's own footprint (10cm wide post-rotation, see RACK_POS's docstring)
    # now starts around x~0.40 at its nearest, so 0.38 keeps a couple cm clear of that before the
    # dynamic min_rack_separation push (reduced alongside this, see the EventTerm params) does the
    # rest. Still under the PAD_X_RANGE-derived limit below (~0.39 at this rag's typical
    # planar_radius), so this hardcoded cap remains the binding one, same as before.
    pad_margin = 0.02
    x_lo = torch.maximum(PAD_X_RANGE[0] + planar_radius + pad_margin, torch.full_like(planar_radius, 0.14))
    x_hi = torch.minimum(PAD_X_RANGE[1] - planar_radius - pad_margin, torch.full_like(planar_radius, 0.38))
    y_lo = torch.maximum(PAD_Y_RANGE[0] + planar_radius + pad_margin, torch.full_like(planar_radius, -0.15))
    y_hi = torch.minimum(PAD_Y_RANGE[1] - planar_radius - pad_margin, torch.full_like(planar_radius, 0.15))

    for _ in range(6):
        moved = False
        for center, required in zip(keepout_centers, keepout_required):
            delta = target_xy - center
            dist = delta.norm(dim=-1).clamp_min(1e-6)
            push = (required - dist).clamp_min(0.0)
            if bool((push > 0.0).any()):
                moved = True
                direction = delta / dist.unsqueeze(-1)
                target_xy = target_xy + direction * push.unsqueeze(-1)
        target_xy[:, 0] = torch.maximum(torch.minimum(target_xy[:, 0], x_hi), x_lo)
        target_xy[:, 1] = torch.maximum(torch.minimum(target_xy[:, 1], y_hi), y_lo)
        if not moved:
            break

    # random yaw per env, applied to the template about its OWN centroid (already at local origin)
    yaw = torch.rand(n, device=rag.device) * (2.0 * math.pi)
    z_axis = torch.zeros((n, 3), device=rag.device)
    z_axis[:, 2] = 1.0
    yaw_quat = math_utils.quat_from_angle_axis(yaw, z_axis)
    num_nodes = local.shape[1]
    quat_per_node = yaw_quat.unsqueeze(1).expand(-1, num_nodes, -1).reshape(-1, 4)
    rotated = math_utils.quat_apply(quat_per_node, local.reshape(-1, 3)).reshape(n, num_nodes, 3)

    # place so the template's own lowest node just touches the pad top (RAG_REST_POS'  z, same
    # pad-top reference the plate/rack recipes use) -- the couple-mm settle below then finds real
    # contact instead of leaving it floating or embedded.
    local_zmin = rotated[..., 2].amin(dim=1)
    target_z = RAG_REST_POS[2] - local_zmin + 0.002
    target_center = torch.stack([target_xy[:, 0], target_xy[:, 1], target_z], dim=-1)

    world_pos = rotated + target_center.unsqueeze(1)
    zero_vel = torch.zeros_like(world_pos)
    rag.write_nodal_state_to_sim(torch.cat([world_pos, zero_vel], dim=-1), env_ids=ids)

    env.scene.write_data_to_sim()
    for _ in range(settle_steps):
        env.sim.step(render=False)
    env.scene.update(env.sim.get_physics_dt())

    # hard-freeze -- see docstring.
    final_state = rag.data.nodal_pos_w[ids]
    rag.write_nodal_state_to_sim(torch.cat([final_state, torch.zeros_like(final_state)], dim=-1), env_ids=ids)

    # TRIED AND REVERTED: kinematically pinning every node at this sculpted pose, then releasing via
    # an "interval" event a short fixed time into the episode, to buy time before the material's
    # elastic relaxation (rest shape = the flat authored mesh; confirmed the SAME fast collapse at
    # every youngsModulus tried, 150 through 15000 -- this is not a stiffness tuning problem)
    # flattens it. Directly verified (plate_settle_diag-style diagnostic, contact_wake_test.py,
    # partial_kinematic_test.py) that once ANY node of this deformable has ever had a kinematic
    # target written via write_nodal_kinematic_target_to_sim, the WHOLE body -- including nodes that
    # were never marked kinematic -- stops responding to gravity, direct nodal writes, AND genuine
    # rigid-body contact (a real dropped plate landed on it with zero effect) even long after every
    # flag is set back to "free". No wake_up equivalent is exposed for soft bodies on this IsaacLab/
    # PhysX version's SoftBodyView (only rigid bodies have one). That would make the rag permanently
    # ungraspable after release -- worse than the flat-rag problem this was meant to fix -- so this
    # is NOT used. See RAG_KINEMATIC_HOLD_SECONDS' docstring (kept, unused) if revisiting this with a
    # fixed/updated API.


def randomize_rag_drop(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    settle_steps: int,
    min_rack_separation: float,
    min_wrinkle_spread: float,
    max_attempts: int,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rack_cfg: SceneEntityCfg = SceneEntityCfg("dish_rack"),
):
    """Place a pre-crumpled rag template (see ``_place_rag_crumpled``) near the rag's nominal spot
    at a random yaw, instead of tossing/tumbling it from height and physically settling the result
    every reset (what this function used to do -- name kept for the EventTerm wiring below, but the
    body is a full rewrite).

    REVISED (user feedback: the rag needs to actually be crumpled -- "a circle or another
    easy-to-grasp shape" -- and resetting was taking too long; confirmed as real physics-stepping
    wall time, not GPU/CPU headroom). The old approach (real toss: height, tilt, and a genuine
    angular-velocity tumble, then settle_steps of physics, then check the settled nodal z-spread and
    RE-TOSS up to max_attempts times if it landed too flat) produced organically different crumples
    episode to episode, but was fundamentally a gamble every reset: even at tuned toss energy, a
    real fraction of individual tosses still landed flat or half-folded (verified over many resets
    while tuning this), so max_attempts had to be raised repeatedly to keep the failure rate down --
    directly trading reset speed for landing quality, since every attempt costs a real settle_steps
    worth of physics regardless of whether it succeeds.

    Using a handful of already-verified-crumpled templates instead of a real toss removes MOST of
    that gamble: the shape being placed is guaranteed good at capture time (a real settled crumple).
    A first version of this concluded that meant no check/retry was needed at all -- verified WRONG
    over a 10-reset check: even a pure rigid yaw + short settle can still relax a template flatter
    than intended in a real minority of resets (the short ``settle_steps`` this needs for placement
    speed apparently isn't always enough for self-collision at the template's own folds to fully
    re-stabilize after being rewritten via ``write_nodal_state_to_sim``, which resets the deformable
    solver's state more abruptly than the gradual physics settle the templates were originally
    captured with). So the z-spread check and retry loop ARE still here -- just much cheaper than
    the old toss-based one: each retry redraws a fresh (template, yaw) pair and re-runs the same
    short ``settle_steps``, capped at ``max_attempts`` (small, since the per-attempt success rate is
    already high with real templates -- this is mopping up a minority tail, not fighting head-on
    unreliability the way the toss's retry loop had to).

    What's NOT changed from the old version: the rack-avoidance push (``min_rack_separation``,
    inside ``_place_rag_crumpled``) -- randomize_dish_rack_and_plate can still place the rack almost
    anywhere on the pad, so the rag's landing still needs to actively dodge wherever it ended up
    this episode, not just assume its own position_range keeps it clear by construction. Also
    unchanged: this still re-pins the robot's joint position TARGET to its just-reset state before
    the settle steps below run, since ``_reset_idx`` (manager_based_env.py) applies event_manager
    BEFORE action_manager.reset() -- see the old docstring's reasoning, still accurate here.
    """
    rag: DeformableObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    rack = env.scene[rack_cfg.name]

    robot.set_joint_position_target(robot.data.default_joint_pos[env_ids], env_ids=env_ids)

    pending_ids = env_ids
    for attempt in range(max_attempts):
        _place_rag_crumpled(
            env,
            rag,
            pending_ids,
            position_range,
            settle_steps,
            rack=rack,
            min_rack_separation=min_rack_separation,
        )
        nodal = rag.data.nodal_pos_w[pending_ids]
        spread = nodal[..., 2].amax(dim=1) - nodal[..., 2].amin(dim=1)
        needs_retry = spread < min_wrinkle_spread
        if not bool(needs_retry.any()) or attempt == max_attempts - 1:
            break
        pending_ids = pending_ids[needs_retry]


# How long into the episode the rag stays kinematically pinned at its sculpted crumpled pose (see
# _place_rag_crumpled's docstring) before release_rag_kinematic_hold switches it back to a normal,
# physics-driven soft body. Long enough that the crumple is still genuinely there once a
# teleoperator's VR view has loaded and they start reaching for it, short enough that it's fully
# free well before that reach completes -- 30s-long episodes give plenty of room either way. Not
# yet re-tuned against a live teleop session; adjust if the rag still visibly "unfreezes" too late
# (feels rigid/unresponsive when grasped) or too early (visibly collapses before it's reachable).
RAG_KINEMATIC_HOLD_SECONDS = 1.5


def release_rag_kinematic_hold(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rag"),
):
    """Switch every node of the rag back to free/physics-driven (kinematic flag -> 1.0), letting it
    behave like a normal soft body again. Registered as an "interval" event at
    RAG_KINEMATIC_HOLD_SECONDS -- see that constant's docstring -- so it fires once per env, a
    fixed short time after THAT env's own reset (interval-mode timers reset at episode start; see
    EventManager). Writing the CURRENT live nodal positions back (not the original sculpted ones)
    means release is seamless -- no snap/jump -- regardless of where physics may have nudged the
    still-kinematic nodes via contact with the robot in the meantime.
    """
    rag = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=rag.device)
    current = rag.data.nodal_pos_w[env_ids]
    free_target = torch.cat([current, torch.ones_like(current[..., :1])], dim=-1)
    rag.write_nodal_kinematic_target_to_sim(free_target, env_ids=env_ids)


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
    wrong every reset:
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=RACK_POS, rot=RACK_REST_ROT),
            spawn=sim_utils.UsdFileCfg(usd_path=RACK_USD_PATH),
        )

        # ── Plate: dynamic, spawned already resting in the rack. Defaults to the SMALL size
        # (DEFAULT_PLATE_SIZE) -- call apply_plate_size(env_cfg, "large") after construction to
        # switch, same post-hoc-mutation pattern record_demos_openarm.py already uses for the
        # pick-up task's --randomize_object_size. See PLATE_SIZE_CONFIGS' docstring for why the
        # small/large variants need their own measured offset/rotation, not just a rescaled copy.
        self.scene.plate = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            init_state=RigidObjectCfg.InitialStateCfg(pos=PLATE_REST_POS, rot=PLATE_REST_ROT_WORLD),
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
            spawn=sim_utils.UsdFileCfg(usd_path=RAG_USD_PATH, scale=(RAG_SCALE,) * 3),
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
                #
                # REVISED (user feedback, alongside RACK_POS's own move -- see that constant's
                # docstring): trimmed way down from +-0.06/+-0.08 to +-0.02/+-0.03 -- "randomize a
                # little bit," not wander far enough to eat into the rag's own placement channel or
                # increase the odds of a yaw/offset combination that leaves the plate looking
                # unseated. The rack/plate group still moves a little every reset, just not far.
                "pose_range": {"x": (-0.02, 0.02), "y": (-0.03, 0.03)},
                # REVISED (user feedback: "only spawn one direction... the direction that can make
                # the plate is in the same direction as arm does" -- was full 360deg per an earlier
                # direct instruction, superseded by this one). Fixed at 0deg -- the SAME orientation
                # RACK_TO_PLATE_OFFSET_SMALL/_LARGE and PLATE_REST_ROT_SMALL/_LARGE were actually
                # measured against (see those constants' docstrings) -- rather than composed with an
                # arbitrary yaw every reset. The plate-on-rack math is verified solid at this one
                # orientation; an arbitrary yaw was never separately verified to hold up as well,
                # and "plate not visibly seated" reports line up with that gap.
                "yaw_range_deg": (0.0, 0.0),
                # REVISED (user report: plate lands touching the pad/table instead of resting on
                # the rack, reproducible across repeated resets): the +-8deg range here was NEVER
                # actually a safe perturbation of PLATE_REST_ROT_SMALL/RACK_TO_PLATE_OFFSET_SMALL --
                # verified directly with a standalone rack+plate physics harness (real gravity, no
                # tolerance shortcuts): the committed target pose IS a genuine stable equilibrium
                # at zero jitter (settles at the documented ~9.27cm above the rack and stays there,
                # velocity -> 0, for a full 5s of physics), but every jittered pitch draw tested
                # (e.g. +8deg pitch, or the -8/+8 combo) slides CONTINUOUSLY off the peg over the
                # next several seconds -- from ~9.2cm down to 0.3-1.6cm above the rack root, i.e.
                # down onto the tray/pad -- well past what the short settle_steps below or the
                # sanity check's tolerance actually observes at reset time. Since this event only
                # steps physics for settle_steps before the episode starts, a jittered reset that
                # "looked fine" at the 50-step mark keeps sliding during the ACTUAL teleop session
                # afterward, unobserved -- exactly the reported symptom. Zeroed out (not just
                # narrowed) until a genuinely wider stable basin is found and re-verified the same
                # way; this makes every reset land at the one pose that's actually been confirmed
                # to hold, at the cost of per-episode lean variety.
                "plate_lean_jitter_deg": {"roll": (0.0, 0.0), "pitch": (0.0, 0.0)},
                # REVISED (user feedback: resetting takes too long, AND "just spawn it directly in
                # the ideal state/position if you can" -- see randomize_dish_rack_and_plate's
                # docstring for the full reasoning): no more plate_drop_height -- the plate is
                # written straight to its measured target pose now, not dropped from height. The
                # height/xy tolerance check and retry loop are still here (just no longer configurable
                # per-event-term -- hardcoded in the function body, loose and capped at 3 attempts)
                # since a fully unchecked direct write was verified to occasionally produce a real
                # failure (see the function's docstring), not just a slower-to-settle landing.
                # settle_steps trimmed way down to match: each attempt now only needs to relax a
                # couple mm of analytic-pose interpenetration, not converge a real drop from scratch.
                #
                # First attempt cut this to 20 -- too aggressive: verified over 10 resets that the
                # plate could still be caught mid-depenetration when the hard freeze (see
                # _settle_plate's docstring) landed, some frozen still floating clear of the rack,
                # one frozen BELOW the rack's own root (a still-resolving SDF-collision push that
                # hadn't finished). Raised to 50 -- re-verify with the same 10-reset check before
                # trusting this number.
                #
                # REVISED (raised 50 -> 120, alongside zeroing plate_lean_jitter_deg above): the
                # standalone physics harness used to diagnose the "touches the table" report showed
                # a bad (jittered) landing can still be sliding measurably at 50 steps and only
                # reveals how far it will actually fall after several hundred more -- so 50 steps
                # was too short to trust the very sanity check below that's supposed to catch a bad
                # landing. 120 is still cheap (this direct-write settle only needs to relax mm-scale
                # interpenetration for the now-deterministic, pre-verified target pose, not discover
                # a resting pose from scratch) but gives a real bad landing enough time to show up
                # as still moving, instead of looking momentarily fine.
                "settle_steps": 120,
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
                # REVISED (user feedback: "in any way if the rag can be spawned with [a crumpled]
                # state, do it, you can skip the time-consuming dropping physics if you can spawn it
                # directly"): this now places a pre-crumpled template (see
                # ``_load_rag_crumple_templates``) instead of tossing/tumbling the rag from height
                # every reset -- see ``randomize_rag_drop``'s docstring for the full history of why
                # (repeated tuning rounds trading reset speed for landing-quality retries) that made
                # this the better fix. position_range only needs xy now (no more toss height/tilt/
                # spin params) -- kept modest so the template stays near RAG_REST_POS, not so wide it
                # wanders toward the rack's own roaming zone before the avoidance push below even
                # runs.
                "position_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                # REVISED (root cause finally isolated with a direct diagnostic, not another guess):
                # every earlier settle_steps value here (20, 40, 60) was chosen assuming MORE physics
                # steps were needed to let self-collision "finish resolving" a rigid-transform seam --
                # exactly backwards. Isolated by placing the SAME captured template repeatedly at
                # settle_steps in {0,2,5,10,20,40,60} and reading the settled z-spread each time: it
                # degrades MONOTONICALLY with more steps -- 0-2 steps preserves ~95-100% of the
                # template's own captured spread (e.g. 0.115 captured -> 0.105-0.114 settled), 10
                # steps is already down to ~0.05-0.06, 20+ collapses to near-zero (0.002-0.03) almost
                # every time. This mesh's self-collision does NOT need real settle time to "lock in" a
                # rigid-transformed crumple the way it needed real fall-and-tumble time to CREATE one
                # during capture -- given extra physics steps with nothing driving it (no gravity
                # doing new work, no tumble), it just relaxes toward the material's flat rest state,
                # the opposite of what every previous settle_steps bump here was trying to fix. 2
                # keeps just enough of a step for write_data_to_sim's effects to register and a
                # velocity read to be meaningful, without giving the relaxation room to run.
                "settle_steps": 2,
                # Unchanged from the old toss-based recipe: randomize_dish_rack_and_plate can still
                # place the rack almost anywhere on the pad, so the rag's landing still needs to
                # actively dodge wherever it ended up this episode -- see _place_rag_crumpled's
                # docstring for the push-then-clamp mechanics this drives.
                #
                # REVISED (user request: push the rag further toward the pad's far edge, alongside
                # RACK_POS's own move out to 0.47 -- see RAG_REST_POS's docstring): lowered
                # 0.22 -> 0.12. RAG_REST_POS's new nominal (0.30) sits only ~0.17m from RACK_POS
                # (0.47) -- the OLD 0.22 separation requirement (0.22+0.05 margin = 0.27m) would have
                # been violated by that nominal spot alone, so the push-away-from-rack logic would
                # have shoved the rag straight back toward the robot every single reset, silently
                # undoing the move. 0.12 (+0.05 margin = 0.17m) matches the new nominal gap instead of
                # fighting it, while still giving the push something real to do whenever the rack's
                # own per-reset xy jitter happens to land closer than that.
                "min_rack_separation": 0.12,
                # REINSTATED (a first version of this direct-spawn recipe dropped this entirely,
                # believing the template guarantee made it unnecessary -- verified wrong, see
                # randomize_rag_drop's docstring: ~40% of individual placements still settled flatter
                # than intended over one 10-reset check). Lower than the old toss-based recipe's 45mm
                # -- these templates' worst realistic outcome is "didn't fully hold its shape", not
                # "landed basically flat" the way a bad toss could, so 30mm is enough to catch a
                # genuinely-degraded placement without demanding every retry match the single best
                # template's own spread.
                "min_wrinkle_spread": 0.030,
                # Raised 4 -> 6 alongside settle_steps' bump above -- 4 attempts at settle_steps=40
                # still left 2 of 10 resets flat (see that comment); see randomize_rag_drop's
                # docstring for why this mops up a minority tail rather than fighting head-on
                # unreliability the way the old toss's retry loop had to.
                "max_attempts": 6,
                "asset_cfg": SceneEntityCfg("rag"),
                "rack_cfg": SceneEntityCfg("dish_rack"),
            },
        )
        # NOT registered: release_rag_kinematic_hold / RAG_KINEMATIC_HOLD_SECONDS (see
        # _place_rag_crumpled's docstring on the kinematic-pin approach this went with, and why it
        # was reverted -- the functions are kept, unused, in case a future PhysX/IsaacLab version
        # exposes a working soft-body wake and this is worth revisiting).
        # Must run AFTER both events above -- see freeze_dynamic_props' docstring for why an
        # unconditional final freeze is needed even though both of those already do their own.
        self.events.freeze_dynamic_props = EventTerm(
            func=freeze_dynamic_props,
            mode="reset",
            params={"asset_cfgs": [SceneEntityCfg("plate"), SceneEntityCfg("rag")]},
        )
