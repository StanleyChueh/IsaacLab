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
    it needed a RigidBodyAPI at all). Sits on the pad's right side (negative y, within the right
    arm's reach -- see the pick-up task's "Looking at the robot from the front camera" y-sign
    convention).
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

Per-episode randomization (see ``events`` below, added on top of ``PickUpEventCfg``):
  * ``dish_rack`` and ``plate`` are moved by the SAME sampled xy offset every reset
    (``randomize_dish_rack_and_plate``, a task-local event function defined in this file) --
    the plate rests leaning IN the rack, so randomizing either one independently would either
    float the plate away from the rack or clip it through a rack wall. dish_rack.usdc originally
    had no RigidBodyAPI at all (a purely static, non-physics prop -- see the parent class'
    ``dish_rack`` docstring entry above); IsaacLab's episode-reset event API only knows how to
    reposition RigidObject/Articulation/DeformableObject entities (``write_root_pose_to_sim`` /
    ``write_nodal_state_to_sim``), not a bare AssetBaseCfg, so the rack was promoted to a
    *kinematic* RigidObject (dish_rack_kinematic.usdc) purely so it can be told where to sit each
    reset -- kinematic means PhysX still won't push it around via gravity/contacts, only this
    event term moves it.
  * ``rag`` gets its own independent xy offset each reset (``randomize_rag``, using IsaacLab's
    built-in ``reset_nodal_state_uniform`` for deformable objects) -- it isn't touching the plate
    or rack at rest, so it doesn't need the coupled treatment above.
  * Both ranges below are a conservative starting guess, not verified against either arm's actual
    reach the way PickUpEventCfg's cube_2 band was -- see that class' docstring for the same
    caveat. Widen them if teleop feels too repetitive; narrow them if a corner becomes unreachable
    or the rack starts overhanging the pad edge.

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

import torch

import isaaclab.envs.mdp as mdp_core
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObjectCfg, RigidObjectCfg
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


def randomize_dish_rack_and_plate(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    rack_cfg: SceneEntityCfg,
    plate_cfg: SceneEntityCfg,
):
    """Move ``dish_rack`` and ``plate`` together by ONE shared per-episode xy offset.

    The plate rests leaning IN the rack (see PLATE_REST_POS/RACK_POS above); randomizing either
    one independently (e.g. with two separate ``reset_root_state_uniform`` terms, each drawing
    its own sample) would drift the plate out of the rack -- floating next to it or clipped
    through a wall, depending on which way the two independent draws happened to point. Both
    assets keep their measured z height and orientation; only x/y shifts, and by the same amount.
    """
    rack = env.scene[rack_cfg.name]
    plate = env.scene[plate_cfg.name]

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ("x", "y")]
    ranges = torch.tensor(range_list, device=rack.device)
    offset_xy = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 2), device=rack.device)

    for asset in (rack, plate):
        root_states = asset.data.default_root_state[env_ids].clone()
        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
        positions[:, 0:2] += offset_xy
        asset.write_root_pose_to_sim(torch.cat([positions, root_states[:, 3:7]], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros_like(root_states[:, 7:13]), env_ids=env_ids)


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

        # ── Per-episode pose randomization -- see this module's docstring for why dish_rack and
        # plate share one event term while rag gets its own. Both ranges are a conservative
        # starting guess (see docstring); widen/narrow them once teleop shows how they feel.
        self.events.randomize_dish_rack_and_plate = EventTerm(
            func=randomize_dish_rack_and_plate,
            mode="reset",
            params={
                "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                "rack_cfg": SceneEntityCfg("dish_rack"),
                "plate_cfg": SceneEntityCfg("plate"),
            },
        )
        self.events.randomize_rag = EventTerm(
            func=mdp_core.reset_nodal_state_uniform,
            mode="reset",
            params={
                "position_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("rag"),
            },
        )
