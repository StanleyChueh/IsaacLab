# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Optional domain-randomization event terms for the OpenArm pick-up/stack tasks.

These terms are not attached to any env cfg's `EventCfg` by default. They are opt-in,
attached only when `--enable_domain_randomization` is passed to generate_dataset.py
(see scripts/imitation_learning/isaaclab_mimic/generate_dataset.py), so normal recording
and generation runs are unaffected unless a caller explicitly asks for randomization.
"""

from __future__ import annotations

import colorsys
import random
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils.stage import get_current_stage

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# NOTE: every term below is a plain function, deliberately not a `ManagerTermBase` class.
# Class-based "reset"-mode terms (e.g. the core `isaaclab.envs.mdp.events.randomize_visual_color`)
# rely on the manager swapping `term_cfg.func` from the class to an instance the first time the
# simulation starts playing; in a freshly `gym.make(...)`-created env, that swap is driven by a
# deferred scene-entity-resolution callback that can lose the race against the very first
# `env.reset()` call, e.g. in scripts/imitation_learning/isaaclab_mimic/generate_dataset.py. The
# observed failure is `TypeError: randomize_visual_color.__init__() got an unexpected keyword
# argument 'asset_cfg'` -- the manager calling the raw class instead of an instance. Plain
# functions (matching how `franka_stack_events.randomize_visual_texture_material` is already
# used successfully in "reset" mode elsewhere in this task family) never go through that swap.


def randomize_static_asset_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    base_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg,
    keep_current_z: bool = False,
):
    """Jitter a static (non-rigid-body) `AssetBaseCfg` asset's world pose around a known default
    local position, e.g. the workspace pad.

    `AssetBaseCfg`-backed assets are exposed to the scene as an `XformPrimView`, not a
    `RigidObject`/`Articulation` -- they have no `write_root_pose_to_sim`/`default_root_state`,
    so pose changes go through `XformPrimView.set_world_poses` instead, and the pose to jitter
    around must be passed in explicitly as `base_pos` (the asset's env-local spawn position, see
    e.g. `PAD_CENTER_X`/`PAD_HEIGHT` in stack_joint_pos_env_cfg.py) rather than read back from a
    cached default -- reading it back would need a class instance to cache it in, which is
    exactly the pattern this module avoids (see module docstring above).

    Set ``keep_current_z`` when something ELSE owns the asset's height. The pad is exactly that
    case: `openarm_task_modes.randomize_pad_height` re-scales and re-centres it at prestartup, so
    a per-env pad is not sitting at the cfg's authored ``base_pos[2]`` any more. Writing the
    authored z back here would drop the box to nominal-centre height while leaving it scaled --
    a pad whose top is at neither the nominal nor the randomized height, and whose bottom is
    through the floor. Reading the live z back instead makes this term do only what its name says
    (jitter the pad's position on the ground) and leaves height to the term that owns it.
    """
    asset = env.scene[asset_cfg.name]
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    base_pos_t = torch.tensor(base_pos, device=env.device, dtype=torch.float32)
    positions = env.scene.env_origins[env_ids] + base_pos_t.unsqueeze(0) + rand_samples[:, 0:3]
    if keep_current_z:
        current_pos, _ = asset.get_world_poses(indices=env_ids.tolist())
        positions[:, 2] = current_pos[:, 2].to(positions.dtype)
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])

    asset.set_world_poses(positions, orientations, indices=env_ids.tolist())


def _resolve_preview_surface_shaders(asset, env_ids: torch.Tensor) -> list:
    """Find the `UsdPreviewSurface` Shader prim(s) created by a `PreviewSurfaceCfg` override,
    one per requested env id.

    `spawn_preview_surface` (`isaaclab.sim.spawners.materials.visual_materials`) creates the
    material at a known, deterministic path and always names the shader child `"Shader"`:
    `UsdShade.Material.Define(stage, prim_path)` then `CreateShaderPrimFromSdrCommand(...,
    parent_path=prim_path, name="Shader")` -> shader lives at `f"{prim_path}/Shader"`.

    The material's own path depends on which spawn function bound it (`visual_material_path`
    defaults to `"material"` in both, but relative to a different base):
    - `spawn_from_usd` (`cube_2`, a Nucleus block prop) binds directly on the asset root:
      `f"{asset.cfg.prim_path}/material"` (`from_files.py:_spawn_from_usd_file`).
    - `spawn_cuboid` (`workspace_pad`, a primitive shape) binds on the nested geometry prim,
      never the root: `f"{asset.prim_paths[env_id]}/geometry/material"`
      (`shapes.py:_spawn_geom_from_prim_type`).

    `asset.cfg.prim_path` (used for the first case) is NOT a concrete per-env path -- it's the
    env-regex TEMPLATE (`InteractiveScene` resolves `{ENV_REGEX_NS}` into the literal string
    `/World/envs/env_.*` once at spawn time via `.format(...)`, and never substitutes it down to
    a concrete `env_0`/`env_1`/... after that). Feeding that `"env_.*"` wildcard straight into
    `stage.GetPrimAtPath(...)` produces an "Ill-formed SdfPath" USD warning (`GetPrimAtPath`
    needs a literal path, not a regex) and always resolves to an invalid prim -- silently a
    no-op every time, confirmed in practice: `cube_2`'s color never actually changed. Substitute
    the real env index for each requested `env_id` instead of using the raw template.

    Returns one Shader `Usd.Prim` per `env_id` (or `None` in that slot if it doesn't exist --
    e.g. the asset was never given a `PreviewSurfaceCfg` visual_material override).
    """
    stage = get_current_stage()
    shaders = []
    for env_id in env_ids.tolist():
        if hasattr(asset, "cfg"):
            concrete_prim_path = asset.cfg.prim_path.replace("env_.*", f"env_{env_id}")
            material_path = f"{concrete_prim_path}/material"
        else:
            material_path = f"{asset.prim_paths[env_id]}/geometry/material"
        shader_prim = stage.GetPrimAtPath(f"{material_path}/Shader")
        shaders.append(shader_prim if shader_prim.IsValid() else None)
    return shaders


def randomize_visual_color(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]],
    roughness_range: tuple[float, float] | None = None,
):
    """Randomize an asset's `PreviewSurfaceCfg` material color (and optionally roughness) by
    writing directly to its Shader prim's USD attributes -- independently per env id.

    Earlier versions of this function (and the equivalent core
    `isaaclab.envs.mdp.events.randomize_visual_color`) went through the Replicator API
    (`rep.get.prims` + `rep.randomizer.color`), which creates a *new* OmniGraph SDGPipeline node
    and a *new* OmniPBR material every call. Called once per "reset" across many episodes, that
    leaves behind a growing pile of orphaned nodes -- observed in practice as repeated
    `UsdExpiredPrimAccessError: Used null prim` errors from `OgnSampleOmniPBR` once an older
    node's cached material/shader prim gets replaced by a newer call. Setting the existing
    Shader prim's `inputs:diffuseColor`/`inputs:roughness` attributes directly (the same
    approach `randomize_scene_lighting` already uses for the dome light, which never hit this
    issue) has none of that: no new nodes, no growing pile, nothing to expire.

    `colors` is either a list of `(r, g, b)` tuples to sample from, or a dict
    `{"r": (low, high), "g": (low, high), "b": (low, high)}` to sample a uniform range from.
    """
    asset = env.scene[asset_cfg.name]
    for shader_prim in _resolve_preview_surface_shaders(asset, env_ids):
        if shader_prim is None:
            continue

        if isinstance(colors, dict):
            r = random.uniform(*colors["r"])
            g = random.uniform(*colors["g"])
            b = random.uniform(*colors["b"])
        else:
            r, g, b = random.choice(list(colors))
        shader_prim.GetAttribute("inputs:diffuseColor").Set((r, g, b))

        if roughness_range is not None:
            shader_prim.GetAttribute("inputs:roughness").Set(random.uniform(*roughness_range))


_DIFFUSE_COLOR_ATTRS = ("inputs:diffuseColor", "inputs:diffuse_color_constant", "inputs:diffuse_tint")
"""Diffuse-colour attribute names, most specific first.

`UsdPreviewSurface` (what a `PreviewSurfaceCfg` override creates) calls it `inputs:diffuseColor`;
the OmniPBR/MDL shaders inside a prop's own USD file call it `inputs:diffuse_color_constant`, with
`inputs:diffuse_tint` as a multiplier on top. Tint is last because it defaults to white, and white
has zero saturation -- scaling the saturation of white leaves it white, so it is only worth writing
when nothing better exists."""

_ROUGHNESS_ATTRS = ("inputs:roughness", "inputs:reflection_roughness_constant")
"""Same idea for surface finish: UsdPreviewSurface name first, OmniPBR/MDL name second."""

_NO_MATERIAL_WARNED_ATTR = "_openarm_dr_no_material_warned"


def _asset_prim_path(asset, env_id: int) -> str:
    """The concrete (non-regex) prim path of *asset* in env *env_id* -- see
    :func:`_resolve_preview_surface_shaders` for why `asset.cfg.prim_path` needs substituting."""
    if hasattr(asset, "cfg"):
        return asset.cfg.prim_path.replace("env_.*", f"env_{env_id}")
    return asset.prim_paths[env_id]


def _resolve_material_shaders(asset, env_ids: torch.Tensor) -> list[list]:
    """Every Shader prim worth randomizing for each env id -- a superset of
    :func:`_resolve_preview_surface_shaders`, which only ever finds a `PreviewSurfaceCfg` override.

    That narrower resolver silently returns nothing for any asset spawned straight from a USD file
    without a `visual_material` override, because `spawn_from_usd` only creates the `material` prim
    when `cfg.visual_material is not None` (`from_files.py`). The can that `apply_task_mode`
    substitutes for cube_2 is exactly such an asset -- so an appearance term pointed at it did
    nothing at all, every reset, without a word. That is the worst possible failure for a
    randomization term: the dataset looks randomized and isn't.

    So: use the override's shader when there is one (it is the material actually bound, and the one
    whose authored colour the task cfg chose), and otherwise fall back to whatever shaders the
    prop's own USD brought with it.
    """
    from pxr import Usd

    stage = get_current_stage()
    resolved = []
    for env_id in env_ids.tolist():
        root = _asset_prim_path(asset, env_id)
        # Both override locations spawn_preview_surface can bind at -- see
        # _resolve_preview_surface_shaders' docstring for why they differ per spawn function.
        override = next(
            (
                prim
                for prim in (
                    stage.GetPrimAtPath(f"{root}/material/Shader"),
                    stage.GetPrimAtPath(f"{root}/geometry/material/Shader"),
                )
                if prim.IsValid()
            ),
            None,
        )
        if override is not None:
            resolved.append([override])
            continue
        root_prim = stage.GetPrimAtPath(root)
        resolved.append(
            [p for p in Usd.PrimRange(root_prim) if p.GetTypeName() == "Shader"]
            if root_prim.IsValid()
            else []
        )
    return resolved


def _first_present_attr(shader_prim, names: tuple[str, ...]):
    """The first of *names* this shader actually has, or None -- shaders differ by type, so the
    caller cannot assume any single attribute name exists."""
    for name in names:
        if shader_prim.HasAttribute(name):
            return shader_prim.GetAttribute(name)
    return None


_BASE_COLOR_CACHE_ATTR = "_openarm_dr_base_diffuse_colors"


def _base_diffuse_color(env, color_attr) -> tuple[float, float, float]:
    """The shader's ORIGINAL diffuse color, latched the first time it is asked for.

    :func:`randomize_visual_saturation` perturbs relative to the nominal color, so it has to read
    that nominal value from somewhere. Reading the shader live every reset would instead perturb
    the *previous* episode's already-perturbed color, and repeated multiplication by a mean-1.0
    factor is a random walk -- saturation would drift toward 0 or 1 over a long generation run and
    the "+-20%" the cfg asks for would stop meaning anything by trial 50.

    Cached on the env object (keyed by the shader's prim path, which is unique per asset AND per
    env) rather than in a class instance, for the reason in this module's header: these terms must
    stay plain functions.
    """
    cache = getattr(env, _BASE_COLOR_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(env, _BASE_COLOR_CACHE_ATTR, cache)
    key = str(color_attr.GetPath())
    if key not in cache:
        value = color_attr.Get()
        cache[key] = (1.0, 1.0, 1.0) if value is None else (float(value[0]), float(value[1]), float(value[2]))
    return cache[key]


def randomize_visual_saturation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    saturation_range: tuple[float, float] = (0.7, 1.3),
    value_range: tuple[float, float] = (0.85, 1.15),
    hue_range: tuple[float, float] = (0.0, 0.0),
    roughness_range: tuple[float, float] | None = None,
):
    """Perturb an asset's appearance AROUND its authored color, in HSV, independently per env id.

    The difference from :func:`randomize_visual_color` is what stays fixed. That one samples an
    absolute RGB triple, so the red cube comes out green, blue or brown -- a hue randomization,
    which is the strongest visual perturbation available and destroys any color cue a policy could
    legitimately use to find the object it was told to pick up. This one multiplies saturation and
    value by factors around 1.0 and leaves hue alone by default, so the cube stays recognisably
    red while its vividness and brightness vary: the lighting/camera-response variation a real
    camera actually produces, rather than a different object.

    ``saturation_range``/``value_range`` are MULTIPLIERS on the base color's S and V (so
    ``(1.0, 1.0)`` disables that axis); ``hue_range`` is an ADDITIVE shift in fractions of the
    colour wheel, wrapping at 1.0, and defaults to no shift at all. ``roughness_range`` is the
    absolute surface-finish range, same as :func:`randomize_visual_color` -- the closest thing to
    "texture" this material has, since these assets carry a flat ``UsdPreviewSurface`` with no
    image map to swap (see the NOTE in PickUpDomainRandomizationEventCfg for why true texture
    swapping is not wired up here).
    """
    asset = env.scene[asset_cfg.name]
    written = 0
    for shader_prims in _resolve_material_shaders(asset, env_ids):
        for shader_prim in shader_prims:
            color_attr = _first_present_attr(shader_prim, _DIFFUSE_COLOR_ATTRS)
            if color_attr is None:
                continue

            h, s, v = colorsys.rgb_to_hsv(*_base_diffuse_color(env, color_attr))
            h = (h + random.uniform(*hue_range)) % 1.0
            s = min(1.0, max(0.0, s * random.uniform(*saturation_range)))
            v = min(1.0, max(0.0, v * random.uniform(*value_range)))
            color_attr.Set(colorsys.hsv_to_rgb(h, s, v))
            written += 1

            if roughness_range is not None:
                roughness_attr = _first_present_attr(shader_prim, _ROUGHNESS_ATTRS)
                if roughness_attr is not None:
                    roughness_attr.Set(random.uniform(*roughness_range))

    # A randomization term that quietly does nothing produces a dataset that looks randomized and
    # is not -- which is only discoverable by training on it and wondering why. Say so instead,
    # once per asset, so a missing material shows up in the generation log rather than in a model.
    if written == 0:
        warned = getattr(env, _NO_MATERIAL_WARNED_ATTR, None)
        if warned is None:
            warned = set()
            setattr(env, _NO_MATERIAL_WARNED_ATTR, warned)
        if asset_cfg.name not in warned:
            warned.add(asset_cfg.name)
            print(
                f"[DR][WARNING] '{asset_cfg.name}' has no shader with any of {_DIFFUSE_COLOR_ATTRS};"
                " its colour/saturation is NOT being randomized. Lighting and camera randomization"
                " are unaffected. Give the asset a PreviewSurfaceCfg visual_material if you need"
                " its appearance varied."
            )


def randomize_fixed_camera_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pos_range: dict[str, tuple[float, float]],
    look_at_offset: tuple[float, float, float] = (0.45, 0.0, 0.15),
    look_at_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("front_cam"),
):
    """Jitter a fixed, env-anchored camera's world position AND, via `look_at_range`, the point
    it's aimed at -- so the viewing angle actually varies, not just the eye position with a
    constant gaze direction.

    Only appropriate for cameras whose prim is anchored directly under the env root (like
    `front_cam`) -- cameras mounted on a moving robot link use `randomize_mounted_camera_pose`
    instead, since their nominal pose is relative to a moving parent link, not the env origin.
    """
    camera = env.scene[asset_cfg.name]
    range_list = [pos_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=env.device)

    base_offset = torch.tensor(camera.cfg.offset.pos, device=env.device, dtype=torch.float32)
    env_origins = env.scene.env_origins[env_ids]
    eyes = env_origins + base_offset.unsqueeze(0) + rand_samples

    look_at_range = look_at_range or {}
    target_range_list = [look_at_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    target_ranges = torch.tensor(target_range_list, device=env.device)
    target_jitter = math_utils.sample_uniform(
        target_ranges[:, 0], target_ranges[:, 1], (len(env_ids), 3), device=env.device
    )
    targets = env_origins + torch.tensor(look_at_offset, device=env.device, dtype=torch.float32).unsqueeze(
        0
    ) + target_jitter

    camera.set_world_poses_from_view(eyes, targets, env_ids=env_ids)


def randomize_mounted_camera_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pos_range: dict[str, tuple[float, float]],
    rot_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Jitter a robot-mounted camera's mount offset (position + small rotation) relative to its
    parent link, e.g. `wrist_cam`/`right_wrist_cam`/`body_cam`.

    Unlike `front_cam`, these cameras' nominal pose is a LOCAL offset relative to a moving robot
    link (`CameraCfg.offset`), not a world-frame constant. The jitter is therefore written as a
    LOCAL pose on the camera prim -- the very same `xformOp:translate` / `xformOp:orient` pair
    that `Camera._initialize_impl` authors from `cfg.offset` when it spawns the prim, so this is
    literally "re-author the mount offset, with noise". Being parent-relative, it rides along
    with the link for the rest of the episode and needs no knowledge of where that link currently
    is.

    DO NOT reimplement this via `Camera.set_world_poses`. That was the original implementation and
    it silently destroyed the wrist views: it composed a target world pose from the parent link's
    pose in `robot.data.body_pos_w`/`body_quat_w` (live, physics/Fabric-side) and handed it to
    `Camera.set_world_poses` -> `XformPrimView.set_world_poses`, which converts world -> local by
    dividing out the parent's world transform *as read from USD*
    (`_set_world_poses_usd`'s `XformCache.GetLocalToWorldTransform(parent_prim)`). With Fabric
    enabled -- the default -- PhysX publishes link transforms to Fabric and never writes them back
    to USD, so that USD-side parent transform is the robot's authored REST pose, not where the
    link actually is. The camera's local offset came out as
    `rest_parent^-1 * live_parent * nominal_offset`: an arbitrary transform that threw the camera
    off into empty space, producing a uniformly blank (~244/255) image for every frame of every
    episode.

    That failure is invisible on `body_cam` and fatal on the wrist cams for the same reason: the
    error is exactly the gap between a link's rest pose and its live pose. `body_cam` hangs off
    `openarm_body_link0`, the base link, which never moves -- rest == live, the bogus factor is
    identity, and its images look perfectly fine (including correctly varying camera angle). Both
    wrist cams hang off `openarm_*_ee_tcp`, which by grasp time is most of a metre and a large
    rotation away from where the asset authored it. So "body cam works, wrist cams are blank" is
    the signature of this bug, not evidence that the wrist ranges below are too wide.

    Writing the local pose sidesteps all of it: no parent world transform is consulted, from
    either side, so there is nothing for USD and Fabric to disagree about.
    """
    camera = env.scene[asset_cfg.name]

    # `cfg.offset.rot` is authored in the cfg's own convention ("ros" for every camera in this
    # task family); the prim's `xformOp:orient` holds it in USD/OpenGL convention. Jitter is
    # composed in the "world" convention (+X forward, +Z up) so that a `rot_range` of roll/pitch/
    # yaw means what it reads like -- about the camera's forward/right/up axes -- and only then
    # converted to OpenGL for the write. Convention conversion is a right-multiplication by a
    # constant quaternion, so converting the LOCAL orientation here yields exactly the same final
    # world orientation the old world-space composition was aiming for.
    offset_pos = torch.tensor(camera.cfg.offset.pos, device=env.device, dtype=torch.float32)
    offset_quat_native = torch.tensor(camera.cfg.offset.rot, device=env.device, dtype=torch.float32).unsqueeze(0)
    offset_quat_world = math_utils.convert_camera_frame_orientation_convention(
        offset_quat_native, origin=camera.cfg.offset.convention, target="world"
    )

    num_envs = len(env_ids)
    pos_ranges = torch.tensor([pos_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z")], device=env.device)
    pos_delta = math_utils.sample_uniform(pos_ranges[:, 0], pos_ranges[:, 1], (num_envs, 3), device=env.device)
    rot_ranges = torch.tensor([rot_range.get(k, (0.0, 0.0)) for k in ("roll", "pitch", "yaw")], device=env.device)
    rot_delta = math_utils.sample_uniform(rot_ranges[:, 0], rot_ranges[:, 1], (num_envs, 3), device=env.device)
    delta_quat = math_utils.quat_from_euler_xyz(rot_delta[:, 0], rot_delta[:, 1], rot_delta[:, 2])

    local_pos = offset_pos.unsqueeze(0) + pos_delta
    local_quat_world = math_utils.quat_mul(offset_quat_world.expand(num_envs, -1), delta_quat)
    local_quat_opengl = math_utils.convert_camera_frame_orientation_convention(
        local_quat_world, origin="world", target="opengl"
    )

    # `Camera` exposes no local-pose setter, so this goes through its `XformPrimView`. That view
    # is the same object `Camera.set_world_poses` writes through, and `set_local_poses` takes the
    # USD path in both Fabric and non-Fabric mode (see `XformPrimView._set_local_poses_fabric`),
    # writing the two xformOps the camera prim already carries.
    camera._view.set_local_poses(local_pos, local_quat_opengl, env_ids)


_SKYBOX_SWAP_STATE_ATTR = "_openarm_dr_skybox_swap_state"


def randomize_scene_lighting(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    color_range: tuple[float, float] = (0.6, 1.0),
    skybox_textures: list[str] | None = None,
    skybox_swap_every: int = 8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    """Randomize the scene dome light's intensity, grayscale brightness, and (if
    `skybox_textures` is given) background HDR skybox.

    The dome light prim (`/World/light`) is spawned once outside the per-env namespace and
    shared by every cloned env, so there is exactly one prim to randomize regardless of
    `num_envs` -- this samples a single draw per call, not one per env id.

    `skybox_swap_every` throttles ONLY the HDR skybox swap (intensity/color still re-roll on
    every single call). This term runs in `mode="reset"`, which fires once per env per episode
    -- across a `generate_dataset.py` run with several dozen trials, `num_envs 4`, and Mimic's
    per-subtask retries, that is easily hundreds of calls. Writing `inputs:texture:file` is not
    a cheap attribute set like intensity/color: it is a direct Sdf value authoring that bypasses
    whatever reference-counted caching Kit's asset/texture pipeline normally relies on (unlike,
    say, Replicator's dedicated texture randomizer -- itself avoided here for a different,
    already-documented leak, see `randomize_visual_color`'s docstring), so each call forces the
    RTX render delegate to load and GPU-upload a fresh copy of the chosen HDR, even when the
    path is unchanged from what is already resident. Observed in practice: GPU memory climbed
    continuously across a generation run rather than plateauing near "13 textures' worth" of
    VRAM, and eventually OOM'd while allocating one of these exact configured files. Re-rolling
    the skybox only every `skybox_swap_every`-th call (and skipping the write entirely when the
    new draw happens to match what's already applied) cuts total texture uploads roughly
    `skybox_swap_every`-fold for the same 13-way diversity across the generated dataset.
    """
    light_asset = env.scene[asset_cfg.name]
    light_prim = light_asset.prims[0]

    intensity = random.uniform(*intensity_range)
    light_prim.GetAttribute("inputs:intensity").Set(intensity)

    gray = random.uniform(*color_range)
    light_prim.GetAttribute("inputs:color").Set((gray, gray, gray))

    if skybox_textures:
        state = getattr(env, _SKYBOX_SWAP_STATE_ATTR, None)
        if state is None:
            state = {"calls": 0, "current": None}
            setattr(env, _SKYBOX_SWAP_STATE_ATTR, state)
        state["calls"] += 1

        if state["calls"] % max(skybox_swap_every, 1) == 1:
            choice = random.choice(skybox_textures)
            if choice != state["current"]:
                light_prim.GetAttribute("inputs:texture:file").Set(choice)
                state["current"] = choice
