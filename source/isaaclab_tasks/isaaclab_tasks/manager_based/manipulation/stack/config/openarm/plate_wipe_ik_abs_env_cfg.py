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
  * ``dish_rack``  -- static holder (no RigidBodyAPI in its own USD), sits on the pad's right
    side (negative y, within the right arm's reach -- see the pick-up task's "Looking at the
    robot from the front camera" y-sign convention).
  * ``plate``      -- dynamic RigidObject, spawned already resting/leaning in the rack. The
    spawn pose below is NOT hand-guessed: it was measured by actually dropping the plate onto
    the rack under gravity in a standalone headless sim and reading back where it settled
    (see the plate_wiping asset-fix conversation -- same "drop it and read the rest pose"
    methodology this file's sibling configs use for CUBE_Z/CAN_HALF_HEIGHT/etc).
  * ``rag``        -- dynamic DeformableObject (PhysX FEM soft body, not a rigid box -- a rigid
    collider can't bend, drape over the plate's edge, or conform to a surface while wiping),
    spawned flat on the pad's left side (positive y, left-arm reach). blue_rag_deformable.usdc
    was built by taking the original blue_rag.usdc render mesh (a 0.3x0.3m, ~2mm-thick
    displaced plane) and applying PhysX's PhysxDeformableBodyAPI + a soft, terrycloth-like
    material to it directly (see the plate_wiping asset-fix conversation); the old rigid
    version's separate 3mm collision-proxy cube is gone since the deformable mesh provides its
    own collision. Its resting height below was measured the same "drop it and read the rest
    pose" way, against this new asset.

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

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.utils import configclass

from . import pickup_ik_abs_env_cfg
from .openarm_task_modes import TASK_ASSET_DIR

PLATE_WIPING_ASSET_DIR = os.path.join(TASK_ASSET_DIR, "plate_wiping")
RACK_USD_PATH = os.path.join(PLATE_WIPING_ASSET_DIR, "dish_rack.usdc")
PLATE_USD_PATH = os.path.join(PLATE_WIPING_ASSET_DIR, "plate.usd")
RAG_USD_PATH = os.path.join(PLATE_WIPING_ASSET_DIR, "blue_rag", "blue_rag_deformable.usdc")

# Right side of the pad (negative y = right-arm side), same PAD_HEIGHT (0.28m) top the
# measurement sim used -- see stack_joint_pos_env_cfg.py's PAD_HEIGHT.
RACK_POS = (0.30, -0.15, 0.28)

# Measured by dropping the plate onto the rack under gravity and reading its settled pose --
# see this module's docstring. World-frame; independent of RACK_POS above (both used the exact
# same pad height/size in the measurement sim, so this transfers as-is).
PLATE_REST_POS = (0.2549925744533539, -0.09456774592399597, 0.37341415882110596)
PLATE_REST_ROT = (0.864859938621521, 0.48673415184020996, -0.07166199386119843, 0.09985882043838501)

# Measured the same way: dropped flat onto the pad (as a PhysX deformable body, not a rigid
# collider -- see this module's docstring) and read back its settled nodal centroid.
RAG_REST_POS = (0.30, 0.15, 0.2811)
RAG_REST_ROT = (1.0, 0.0, 0.0, 0.0)


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

        # ── Dish rack: static holder, right side of the pad (right-arm reach) ──
        self.scene.dish_rack = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/DishRack",
            init_state=AssetBaseCfg.InitialStateCfg(pos=RACK_POS, rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.UsdFileCfg(usd_path=RACK_USD_PATH),
        )

        # ── Plate: dynamic, spawned already resting in the rack ─────────────
        self.scene.plate = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            init_state=RigidObjectCfg.InitialStateCfg(pos=PLATE_REST_POS, rot=PLATE_REST_ROT),
            spawn=sim_utils.UsdFileCfg(usd_path=PLATE_USD_PATH),
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
