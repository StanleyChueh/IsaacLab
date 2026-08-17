# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm pick-up task: grasp and lift the red cube (cube_2), reachable by either arm.

Action space (14D flat):
  [0:6]   left arm IK delta pose  (dx dy dz drx dry drz)
  [6]     left gripper command    (±1.0)
  [7:13]  right arm IK delta pose (TAB-switch to control the right arm if cube_2 lands on
          that side -- see Cube randomisation below)
  [13]    right gripper command   (±1.0)

Scene: only cube_2 (red) is present — cube_1 and cube_3 are removed.

Cube randomisation: full pad width -- left arm, right arm, and the middle.
  Looking at the robot from the front camera (1,0,0.5):
    x = forward from robot base (depth in front of robot)
    y = left  (positive y = robot's left arm side, negative y = right arm side)
  cube_2 is randomized within x:[0.20, 0.27]  y:[-0.27, 0.27]
  — keeps >=8cm clearance from the robot base, stays within 15cm of the pad's near edge, and
    reaches near both of the pad's side edges in y (middle falls out for free as the continuum
    between the two halves); see PickUpEventCfg's docstring for the full reasoning.

Success condition: cube_2 centre rises 2.5 cm above its resting height (see `cube_rest_z` in
OpenarmPickUpRedCubeEnvCfg.__post_init__ -- resting height itself depends on the pad/cube
geometry in stack_joint_pos_env_cfg.py, so this is computed at cfg-build time, not a literal;
the +2.5 cm offset is calibrated against real recorded teleop demos' peak lift height, not
picked by analogy to an older design -- see the comment above `cube_rest_z` for why).
"""

import torch

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

import isaaclab.envs.mdp as mdp_core
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import openarm_domain_randomization

from . import stack_ik_abs_visuomotor_env_cfg


# ── Subtask / termination helpers ─────────────────────────────────────────────

def cube_is_lifted(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    min_height: float = 0.30,
) -> torch.Tensor:
    """Return 1.0 when cube centre is above *min_height* world-Z."""
    obj = env.scene[asset_cfg.name]
    return (obj.data.root_pos_w[:, 2] > min_height).unsqueeze(-1).float()


def cube_pickup_success(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    min_height: float = 0.20,
) -> torch.Tensor:
    """Episode terminates with success when cube centre rises above *min_height* world-Z."""
    obj = env.scene[asset_cfg.name]
    return obj.data.root_pos_w[:, 2] > min_height


# ── Events ────────────────────────────────────────────────────────────────────

@configclass
class PickUpEventCfg:
    """Reset robot to default pose; randomize cube_2 across the full pad width (left arm,
    right arm, and the middle), keeping the same forward/depth reach as before.

    cube_2 default world position: [0.55, 0.05, CUBE_Z]  (CUBE_Z ≈ 0.15 m)
    Target randomisation range:    x=[0.20, 0.27]  y=[-0.27, 0.27]
    Required offsets:              x=[-0.35, -0.28]  y=[-0.32, 0.22]

    x range keeps the cube within 15cm of the pad's near edge (the side closest to the robot
    base, x=0.12 -- see PAD_NEAR_EDGE_X in stack_joint_pos_env_cfg.py): x=0.27 is exactly
    0.12+0.15. The near bound (x=0.20) still keeps >=8cm clearance from the robot origin -- an
    earlier, more compact x=[0.17,0.27] band had a near corner only ~17cm from the origin, close
    enough for the gripper to clip the base while reaching; widening the near edge out to 0.20
    fixed that.

    y now spans both arms symmetrically: [-0.27, 0.27] (positive y = robot's left-arm side,
    negative y = right-arm side, near-zero = middle -- see the module docstring's "Looking at
    the robot from the front camera" convention). This mirrors the previous left-only bound
    (0.08 to 0.27) onto the negative side too, keeping the same ~1.5cm margin from the pad's
    real y=+-0.285 half-width on both edges, and folds the middle in "for free" since it's just
    the continuum between the two halves. During teleop, TAB-switch to whichever arm the cube
    lands in front of -- gripper_joint_names/gripper_open_val/gripper_threshold below are still
    left-arm-only, so if you Mimic-annotate these demos afterward, right-arm/middle grasps won't
    be recognized as "gripper closed" by that logic; this only affects raw teleop recording.
    Not verified against either arm's actual comfortable reach limit at the y=+-0.27 corners --
    if teleop feels strained or Mimic's generation success rate drops, narrow it back down.
    """

    init_robot_pose = EventTerm(
        func=mdp_core.reset_scene_to_default,
        mode="reset",
    )

    randomize_cube_2 = EventTerm(
        func=mdp_core.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.35, -0.28),   # cube_2 default x=0.55 → actual x:[0.20, 0.27]
                "y": (-0.32, 0.22),    # cube_2 default y=0.05 → actual y:[-0.27, 0.27]
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_2"),
        },
    )


@configclass
class PickUpDomainRandomizationEventCfg(PickUpEventCfg):
    """Extends `PickUpEventCfg` with optional appearance/pose domain-randomization terms.

    Not used by default -- only attached to the env cfg when domain randomization is
    explicitly requested (see generate_dataset.py's --enable_domain_randomization flag).
    Visual randomization requires scene.replicate_physics=False, which is already the
    default for this task family (see stack_env_cfg.py's ObjectTableSceneCfg).
    """

    # NOTE: no texture (image-swap) randomization for cube_2/workspace_pad. An earlier version
    # routed both color AND texture through the Replicator API (`rep.get.prims` +
    # `rep.randomizer.color`/`.texture`), which creates a *new* OmniGraph node + OmniPBR
    # material every single call. Invoked once per "reset" across many episodes, this produced
    # a growing pile of orphaned nodes and surfaced as repeated `UsdExpiredPrimAccessError: Used
    # null prim` errors from `OgnSampleOmniPBR` during an actual generate_dataset.py run (visible
    # once enough resets had happened for an older node's cached prim to be replaced/invalidated
    # by a newer call -- non-fatal to that particular run, but not something to keep doing across
    # a full 50-trial generation). Color below now writes directly to the material's Shader prim
    # (`randomize_visual_color`'s docstring), which has none of that risk -- but true image
    # texture swapping needs an actual UV-mapped texture shader graph, which isn't safe to
    # hand-build without a live Isaac Sim session to verify against. Color + roughness jitter is
    # the appearance-randomization budget for these two assets until that's built properly.
    randomize_cube_2_color = EventTerm(
        func=openarm_domain_randomization.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_2"),
            "colors": {"r": (0.05, 0.95), "g": (0.05, 0.95), "b": (0.05, 0.95)},
            "roughness_range": (0.0, 1.0),
        },
    )

    randomize_pad_color = EventTerm(
        func=openarm_domain_randomization.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("workspace_pad"),
            "colors": {"r": (0.05, 0.95), "g": (0.05, 0.95), "b": (0.05, 0.95)},
            "roughness_range": (0.0, 1.0),
        },
    )

    randomize_pad_position = EventTerm(
        func=openarm_domain_randomization.randomize_static_asset_pose,
        mode="reset",
        params={
            # Small jitter only -- height (z) and orientation stay fixed so the deliberately
            # tuned pad height / robot clearance (see stack_joint_pos_env_cfg.py) still holds.
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
            # Must match workspace_pad's spawn position in stack_joint_pos_env_cfg.py
            # (PAD_CENTER_X, 0.0, PAD_HEIGHT / 2) -- update this if that changes.
            "base_pos": (0.3925, 0.0, 0.095),
            "asset_cfg": SceneEntityCfg("workspace_pad"),
        },
    )

    randomize_light = EventTerm(
        func=openarm_domain_randomization.randomize_scene_lighting,
        mode="reset",
        params={
            "intensity_range": (500.0, 12000.0),
            "color_range": (0.3, 1.0),
            # Every path below is one already referenced by other shipped Isaac Lab task
            # configs/tests (grepped for "Skies/"+".hdr" across the repo) -- kept to
            # confirmed-existing Nucleus assets rather than guessed filenames, since a wrong
            # path fails silently (no texture applied) rather than erroring.
            "skybox_textures": [
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ],
            "asset_cfg": SceneEntityCfg("light"),
        },
    )

    # ── Cameras ────────────────────────────────────────────────────────────
    # front_cam's world-pose jitter (randomize_fixed_camera_pose) is switched off along with the
    # camera itself -- OpenarmPickUpRedCubeEnvCfg.__post_init__ sets `self.scene.front_cam = None`,
    # so SceneEntityCfg("front_cam") would have nothing to resolve against at reset. `None` is the
    # idiom EventManager._prepare_terms understands for "skip this term entirely".
    randomize_camera_pose = None

    # wrist/right_wrist/body cams are mounted on moving robot links, so they use
    # randomize_mounted_camera_pose (mount-offset jitter relative to their parent link) rather
    # than a world-frame jitter -- see that function's docstring for why.
    #
    # Kept moderate (not maxed out like color/lighting above): these mounts sit close to the
    # gripper and are the views the grasp itself depends on, so over-jittering risks losing the
    # cube/gripper out of frame.
    randomize_wrist_cam_pose = EventTerm(
        func=openarm_domain_randomization.randomize_mounted_camera_pose,
        mode="reset",
        params={
            "pos_range": {"x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (-0.015, 0.015)},
            "rot_range": {"roll": (-0.14, 0.14), "pitch": (-0.14, 0.14), "yaw": (-0.14, 0.14)},
            "asset_cfg": SceneEntityCfg("wrist_cam"),
        },
    )

    randomize_right_wrist_cam_pose = EventTerm(
        func=openarm_domain_randomization.randomize_mounted_camera_pose,
        mode="reset",
        params={
            "pos_range": {"x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (-0.015, 0.015)},
            "rot_range": {"roll": (-0.14, 0.14), "pitch": (-0.14, 0.14), "yaw": (-0.14, 0.14)},
            "asset_cfg": SceneEntityCfg("right_wrist_cam"),
        },
    )

    randomize_body_cam_pose = EventTerm(
        func=openarm_domain_randomization.randomize_mounted_camera_pose,
        mode="reset",
        params={
            "pos_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
            "rot_range": {"roll": (-0.14, 0.14), "pitch": (-0.14, 0.14), "yaw": (-0.14, 0.14)},
            "asset_cfg": SceneEntityCfg("body_cam"),
        },
    )

    # NOTE: robot/table texture randomization (via franka_stack_events.randomize_visual_texture_material)
    # was removed here -- it goes through the same repeated-Replicator-call pattern that produced
    # the "cube_2"/"workspace_pad" texture bug above (see the NOTE near randomize_cube_2_color).
    # That function is pre-existing, shared code used elsewhere in this task family, so it wasn't
    # rewritten; it's just not worth attaching two more copies of a known-fragile call pattern to
    # this event cfg on top of the two (cube/pad) that already caused a real, observed failure.


@configclass
class PickUpVisualDomainRandomizationEventCfg(PickUpDomainRandomizationEventCfg):
    """A deliberately mild, appearance-only alternative to `PickUpDomainRandomizationEventCfg`.

    Selected by generate_dataset.py's `--domain_randomization_profile visual`. Covers exactly four
    axes -- lighting, surface finish, colour saturation, and camera angle on all three mounted
    cameras -- and nothing else. What it leaves alone is as much the point as what it varies:

    * **No skybox swap.** The full profile replaces the dome light's HDR environment map with one
      of thirteen unrelated scenes (a parking lot, a hospital room, a photo studio), which changes
      the entire background AND the ambient colour cast the whole scene is lit by. Here the light
      only brightens and dims; the background it sits in stays put.
    * **No hue randomization.** The full profile samples each asset's diffuse colour as an
      absolute RGB triple over almost the whole cube, so cube_2 comes out green or blue as often
      as red. This profile perturbs SATURATION and brightness around the authored colour and
      leaves hue fixed (see `randomize_visual_saturation`), so a red cube stays a red cube -- a
      policy trained on this data can still use colour to find the object, which in a task named
      "PickUp-RedCube" is a cue worth keeping rather than training away.
    * **No pad position jitter.** That moves the geometry the demo was generated against, not its
      appearance, and appearance is all this profile is for.

    Ranges are set to roughly +-50% of nominal for lighting (3000 intensity, 0.75 grey -- see
    `stack_env_cfg.py`) and +-30% for saturation, versus the full profile's 0.17x-4x intensity and
    effectively unbounded colour. Camera jitter is carried over from the full profile unchanged:
    those ranges were already dialled back once to keep the workspace in frame (see the comments
    on `randomize_camera_pose` above), and camera angle is a variation this profile explicitly
    wants.
    """

    randomize_light = EventTerm(
        func=openarm_domain_randomization.randomize_scene_lighting,
        mode="reset",
        params={
            # Nominal is 3000 / 0.75 grey. Wide enough that a policy cannot assume one exposure,
            # narrow enough that the workspace is never blown out or nearly black -- the two
            # extremes of the full profile's (500, 12000), where the cube is genuinely hard to see.
            "intensity_range": (1800.0, 4800.0),
            "color_range": (0.6, 0.9),
            # Omitted on purpose -- this is the background swap. Leaving it None makes
            # randomize_scene_lighting skip the texture write entirely.
            "skybox_textures": None,
            "asset_cfg": SceneEntityCfg("light"),
        },
    )

    randomize_cube_2_color = EventTerm(
        func=openarm_domain_randomization.randomize_visual_saturation,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_2"),
            "saturation_range": (0.7, 1.3),
            "value_range": (0.85, 1.15),
            # Roughness is the only "texture" knob a flat UsdPreviewSurface offers (there is no
            # image map on these assets to swap -- see the NOTE in the full profile above). Kept
            # off the extremes: a 0.0 mirror finish turns the pad into a specular highlight that
            # tracks the camera, which is a much larger appearance change than a matte/satin shift.
            "roughness_range": (0.25, 0.75),
        },
    )

    randomize_pad_color = EventTerm(
        func=openarm_domain_randomization.randomize_visual_saturation,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("workspace_pad"),
            "saturation_range": (0.7, 1.3),
            "value_range": (0.85, 1.15),
            "roughness_range": (0.25, 0.75),
        },
    )

    # Geometry, not appearance -- switched off. `None` is how EventManager._prepare_terms is told
    # to skip an inherited term entirely (it `continue`s on any term whose cfg is None), which is
    # the same idiom the task cfgs elsewhere in this tree use to drop a manager they don't want.
    randomize_pad_position = None

    # The three mounted-camera angle terms are inherited from the parent unchanged and
    # deliberately not restated here -- their mount-offset jitter is already the moderate range
    # this profile wants. (front_cam and its jitter term are off entirely; see the parent.)


def _manipulated_object_name(env_cfg) -> str:
    """The scene entity the task actually manipulates, which is NOT always `cube_2`.

    `apply_task_mode` (openarm_task_modes.py) replaces cube_2 with the can in every mode --
    `env_cfg.scene.cube_2 = None`, `env_cfg.scene.can = RigidObjectCfg(...)` -- and the Mimic cfg
    calls it unconditionally from its own `__post_init__`, so by the time domain randomization is
    attached there is usually no cube_2 left to randomize. Rather than guess, read the answer off
    the spawn term that same function installs (`randomize_object`, whose `asset_cfg` names the
    object it moves each reset); fall back to cube_2 only when no mode was ever applied.
    """
    spawn_term = getattr(env_cfg.events, "randomize_object", None)
    name = spawn_term.params["asset_cfg"].name if spawn_term is not None else "cube_2"
    if getattr(env_cfg.scene, name, None) is None:
        raise ValueError(
            f"Cannot attach domain randomization: the task's manipulated object '{name}' is not in"
            f" the scene cfg. Available: {sorted(k for k, v in vars(env_cfg.scene).items() if v is not None)}"
        )
    return name


DOMAIN_RANDOMIZATION_PROFILES = {
    "full": PickUpDomainRandomizationEventCfg,
    "visual": PickUpVisualDomainRandomizationEventCfg,
}
"""Selectable randomization strengths -- see generate_dataset.py's --domain_randomization_profile."""


def attach_domain_randomization(env_cfg, profile: str = "full") -> list[str]:
    """Add *profile*'s randomization terms to `env_cfg.events` IN PLACE, and return their names.

    Deliberately NOT `env_cfg.events = PickUpDomainRandomizationEventCfg()`. That was the original
    approach and it silently discarded every event term `apply_task_mode` had already installed on
    the cfg, because a fresh cfg instance carries the class defaults rather than the patched ones.
    Three things were lost, in increasing order of subtlety:

    1. `randomize_cube_2 = None` reverted to the inherited term, which calls
       `reset_root_state_uniform` on a cube_2 the task mode had deleted from the scene -- an
       immediate `KeyError: "Scene entity with key 'cube_2' not found"` on the very first
       `env.reset()`.
    2. `randomize_object` (the can's per-reset spawn pose) disappeared, so had the crash not
       happened first, every generated episode would have started with the can at one fixed pose.
    3. In handover mode, `reset_handover_stage` disappeared -- the stage machine is stateful, and
       without its reset term it would have carried one episode's progress into the next.

    Attaching only the terms the DR cfg ADDS (its fields minus `PickUpEventCfg`'s) leaves all of
    that untouched. Terms aimed at cube_2 are retargeted onto whatever object the task really has
    (see :func:`_manipulated_object_name`) for the same reason: the appearance terms were written
    against the pre-task-mode scene.
    """
    if profile not in DOMAIN_RANDOMIZATION_PROFILES:
        raise ValueError(
            f"Unknown domain randomization profile '{profile}'."
            f" Expected one of {sorted(DOMAIN_RANDOMIZATION_PROFILES)}."
        )

    base_term_names = set(vars(PickUpEventCfg()))
    dr_cfg = DOMAIN_RANDOMIZATION_PROFILES[profile]()
    object_name = _manipulated_object_name(env_cfg)

    attached = []
    for term_name, term in vars(dr_cfg).items():
        if term_name in base_term_names:
            continue  # a base term -- whatever apply_task_mode left there is the correct version
        if term is not None:
            asset_cfg = term.params.get("asset_cfg")
            if asset_cfg is not None and asset_cfg.name == "cube_2":
                term.params["asset_cfg"] = SceneEntityCfg(object_name)
            attached.append(term_name)
        # None is attached too, not skipped: that is how a profile switches OFF a term it inherits
        # from another profile (EventManager._prepare_terms skips any term whose cfg is None).
        setattr(env_cfg.events, term_name, term)
    return attached


# ── Observations ──────────────────────────────────────────────────────────────

@configclass
class PickUpObservationsCfg:
    """Camera observations + EEF pose (needed by Mimic) + subtask completion signals."""

    @configclass
    class PolicyCfg(ObsGroup):
        # front_cam is switched off for this task -- see OpenarmPickUpRedCubeEnvCfg.__post_init__,
        # where `self.scene.front_cam = None` removes the sensor itself. The three mounted cams
        # below are the ones the grasp actually depends on.
        wrist_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        right_wrist_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        body_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("body_cam"), "data_type": "rgb", "normalize": False},
        )
        # EEF pose — required by OpenArmPickUpIKAbsMimicEnv.get_robot_eef_pose()
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Subtask termination signals for Mimic auto-annotation.

        Both signals use cube height so they are robust and require no EEF tracking:
          grasp (min_height=cube_rest_z + 5 mm):  cube_2 has just left the pad.
                                                   Marks end of the reach-and-grasp segment.
          lift  (min_height=cube_rest_z + 2.5 cm): cube_2 is clearly picked up.
                                                    Stored for debugging; not used as a Mimic
                                                    term signal (it is the last subtask). Must
                                                    still fire at least once during a demo or
                                                    annotate_demos.py rejects the episode --
                                                    calibrated against real recorded peak
                                                    heights, see __post_init__.

        The literal `min_height` values below are placeholders overwritten in
        OpenarmPickUpRedCubeEnvCfg.__post_init__ once cube_2's actual resting height
        (`cube_rest_z`) is known -- see that method for why a hardcoded literal here is
        wrong whenever the pad height / cube size changes.
        """

        grasp = ObsTerm(
            func=cube_is_lifted,
            params={"asset_cfg": SceneEntityCfg("cube_2"), "min_height": 0.155},
        )
        lift = ObsTerm(
            func=cube_is_lifted,
            params={"asset_cfg": SceneEntityCfg("cube_2"), "min_height": 0.30},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


# ── Environment config ────────────────────────────────────────────────────────

@configclass
class OpenarmPickUpRedCubeEnvCfg(stack_ik_abs_visuomotor_env_cfg.OpenarmCubeStackVisuomotorEnvCfg):
    """OpenArm pick-up task: grasp and lift the red cube (cube_2), reachable by either arm.

    Key differences from the visuomotor stack env:
      - Only cube_2 (red) is present in the scene (cube_1 and cube_3 removed).
      - cube_2 is randomized across the full pad width -- left arm, right arm, and the middle.
      - ee_frame targets openarm_left_ee_tcp (required by eef_pos/eef_quat obs) -- still
        left-only, so Mimic auto-annotation (gripper_joint_names/eef tracking below) only
        understands left-arm grasps even though the cube can now land on the right side too.
      - subtask_terms observation group added for Mimic auto-annotation.
      - Success terminates when cube_2 rises above 0.20 m.
      - Right arm IK action kept so TAB-switching works during teleoperation.

    Mimic task: Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0
    """

    gripper_joint_names: list = ["openarm_left_finger_joint.*"]
    gripper_open_val: float = 0.044
    gripper_threshold: float = 0.018

    def __post_init__(self):
        super().__post_init__()  # cameras, IK-Abs action, pad, cubes

        # ── Remove blue and green cubes — only red cube participates ──────────
        # The scene builder skips attributes set to None.
        self.scene.cube_1 = None
        self.scene.cube_3 = None

        # ── Drop front_cam ───────────────────────────────────────────────────
        # The parent visuomotor cfg builds four non-tiled CameraCfg sensors at 640x480, and a
        # non-tiled Camera is one independent RTX render product PER ENV -- so the camera count
        # multiplies with --num_envs and is what runs a 16 GB card out of VRAM during dataset
        # generation. front_cam is the one that pays for itself least here: it is a fixed
        # table-side view, while the grasp/hand-over this task is about is decided by the two
        # wrist cams and body_cam. Dropping it removes one render product per env.
        #
        # Removing the sensor is not enough on its own -- `image_obs_list` below and the
        # PolicyCfg obs term and the DR jitter term all name it, and each would fail to resolve
        # against a scene that no longer has it. All three are handled (see PickUpObservationsCfg
        # and PickUpDomainRandomizationEventCfg.randomize_camera_pose).
        self.scene.front_cam = None

        # ── Observations ─────────────────────────────────────────────────────
        self.observations.policy = PickUpObservationsCfg.PolicyCfg()
        self.observations.subtask_terms = PickUpObservationsCfg.SubtaskCfg()

        # cube_2's resting height (its spawn Z, set in stack_joint_pos_env_cfg.py from
        # PAD_HEIGHT/CUBE_SIZE) drives every height threshold below. Reading it back here
        # instead of hardcoding a literal keeps "grasp"/"lift"/"success" correct if pad height
        # or cube size ever change again -- a stale literal here previously broke the "grasp"
        # subtask signal outright (it sat *below* the resting height, so cube_is_lifted was
        # always True and the 0->1 transition annotate_demos.py/generate_dataset.py look for
        # never happened).
        #
        # The +0.025/+0.025 offsets below (lift/success) are calibrated against the actual
        # peak cube_2 height reached across 10 recorded teleop demos under this same geometry
        # (rest=0.1975, observed max_z ranged 0.2006-0.2444, 9/10 demos cleared ~0.232+) -- a
        # first attempt used a much larger +0.15/+0.05 offset by analogy to an older design
        # tuned for a lower pad, and it made annotate_demos.py's success_term/"lift" signal
        # unreachable by *every* recorded demo (all fell short of 0.245+). Keep any future
        # offset choice grounded in the demos' actual achieved lift height, not a guessed value.
        cube_rest_z = float(self.scene.cube_2.init_state.pos[2])
        self.observations.subtask_terms.grasp.params["min_height"] = cube_rest_z + 0.005
        self.observations.subtask_terms.lift.params["min_height"] = cube_rest_z + 0.025

        # ── ee_frame: track left EEF (required by eef_pos / eef_quat obs) ────
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_left_link1",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_left_ee_tcp",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                )
            ],
        )

        # ── Debug: contact sensor on the right arm's left finger ───────────────
        # Diagnoses the "right arm left finger doesn't seem to grip" symptom -- reports
        # whether that link is generating contact force against cube_2 at all (vs.
        # generating contact but not holding, which would look identical from the outside).
        # Not part of the task's observation/action space; only read for debug printing in
        # record_demos_openarm.py's main loop.
        # debug_vis stays OFF: it draws a marker at every contact point, which lands a red ball on
        # the object exactly when the fingers close on it -- i.e. in every frame of the recorded
        # wrist/front camera images, which are training data.
        self.scene.contact_right_left_finger = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_right_left_finger",
            update_period=0.0,
            history_length=6,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube_2"],
        )

        # ── Terminations ─────────────────────────────────────────────────────
        self.terminations.time_out = None
        self.terminations.success = TerminationTermCfg(
            func=cube_pickup_success,
            params={"asset_cfg": SceneEntityCfg("cube_2"), "min_height": cube_rest_z + 0.025},
            time_out=False,
        )

        # ── Events: robot reset + cube_2 randomised across the full pad width ────
        self.events = PickUpEventCfg()

        # ── Right arm action (for TAB-switching during keyboard recording) ────
        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_joint[1-7]"],
            body_name="openarm_right_ee_tcp",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
        )
        self.actions.right_gripper_action = mdp_core.BinaryJointPositionActionCfg(
            asset_name="robot",
            # Both finger joints are commanded. joint2 is a PhysX mimic of joint1 AND is
            # actuated (see openarm.py's "openarm_gripper" comment): the mimic enforces
            # pos2 == pos1 and these two targets are equal, so they agree rather than fight.
            # Naming joint1 alone would leave the actuated joint2 pinned at its default and
            # genuinely fight the mimic.
            joint_names=["openarm_right_finger_joint1", "openarm_right_finger_joint2"],
            open_command_expr={"openarm_right_finger_joint1": 0.044, "openarm_right_finger_joint2": 0.044},
            close_command_expr={"openarm_right_finger_joint1": 0.0, "openarm_right_finger_joint2": 0.0},
        )
        
        self.image_obs_list = ["wrist_cam", "right_wrist_cam", "body_cam"]
