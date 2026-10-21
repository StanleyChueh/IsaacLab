# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Domain randomization on top of the real look (:mod:`openarm_real_look`).

The DR profiles (pickup_ik_abs_env_cfg.DOMAIN_RANDOMIZATION_PROFILES) were written for the plain sim scene, where
the dome light lights everything and its HDR skybox IS the background. Under the real look three things differ, and
:func:`adapt_to_real_look` re-targets the affected terms after attach_domain_randomization has added them:

* **Lighting.** The look's dome light is 500 (lighting.json), the profiles' ranges are written around 3000 / 0.75
  grey (stack_env_cfg.py); applied as written they would light the robot up to 24x too bright. And the photo backdrop
  is emissive, so a light change would leave the table and room untouched while the robot brightens. Here the same
  relative draw scales the look's light, and brightens / tints the backdrop with it.
* **Background swap** (the full profile's skybox). The photo room hides the dome, so a swapped skybox would never be
  seen. Here, each time the skybox swaps, the room walls are hidden with probability ``room_hide_prob`` -- the wrist
  cameras then see the random skybox, as in the plain scene -- and shown otherwise.
* **Table colour.** The sim pad the colour terms recolour is invisible under the look; the visible table is the
  backdrop. Its colour is varied instead (:func:`randomize_backdrop_tint`, per env).

Pad height (prestartup) is handled where it is drawn -- randomize_pad_height moves every backdrop marked
``follows_pad`` with the pad, so the photo table stays under the can. Pad position jitter moves only the invisible
collider, a flat plane under a flat photo, so it stays as it is. Camera pose jitter and shake need nothing.
"""

from __future__ import annotations

import random

import numpy as np

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

REFERENCE_LIGHT = (3000.0, 0.75)
"""Intensity and grey of the plain scene's dome light (stack_env_cfg.py): what the DR profiles' ranges are written around."""
TINT = {
    "visual": {"value_range": (0.85, 1.15), "channel_range": (0.95, 1.05)},
    "full": {"value_range": (0.6, 1.4), "channel_range": (0.7, 1.3)},
}
"""Per-env table tint per profile: overall brightness x per-channel gain (visual keeps the hue, as its pad term does)."""
_STATE_ATTR = "_openarm_real_look_backdrop_state"


def _state(env) -> dict:
    state = getattr(env, _STATE_ATTR, None)
    if state is None:
        from .openarm_real_look import BackdropMeshCfg

        backdrops = [(name, cfg.spawn) for name, cfg in vars(env.cfg.scene).items()
                     if isinstance(getattr(cfg, "spawn", None), BackdropMeshCfg)]
        state = {"backdrops": backdrops, "gain": np.ones(3), "tint": {}, "base": {}, "room_visible": True}
        setattr(env, _STATE_ATTR, state)
    return state


def _write(env, env_ids) -> None:
    """Set the backdrop meshes' emissive vertex colours of ``env_ids`` to base x global light gain (x tint on tables)."""
    from pxr import UsdGeom, Vt

    from isaaclab.sim.utils import get_current_stage

    state = _state(env)
    stage = get_current_stage()
    for name, spawn in state["backdrops"]:
        asset_root = getattr(env.cfg.scene, name).prim_path.split("/")[-1]
        for i in env_ids:
            path = f"{env.scene.env_prim_paths[int(i)]}/{asset_root}/mesh"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
            if path not in state["base"]:
                state["base"][path] = np.array(pv.Get(), dtype=np.float32)
            scale = state["gain"] * (state["tint"].get(int(i), np.ones(3)) if spawn.follows_pad else 1.0)
            pv.Set(Vt.Vec3fArray.FromNumpy((state["base"][path] * scale[None]).astype(np.float32)))


def _set_room_visible(env, visible: bool) -> None:
    from pxr import UsdGeom

    from isaaclab.sim.utils import get_current_stage

    state = _state(env)
    if state["room_visible"] == visible:
        return
    stage = get_current_stage()
    for name, spawn in state["backdrops"]:
        if spawn.role != "room":
            continue
        asset_root = getattr(env.cfg.scene, name).prim_path.split("/")[-1]
        for env_path in env.scene.env_prim_paths:
            prim = stage.GetPrimAtPath(f"{env_path}/{asset_root}")
            if prim.IsValid():
                (UsdGeom.Imageable(prim).MakeVisible if visible else UsdGeom.Imageable(prim).MakeInvisible)()
    state["room_visible"] = visible


def randomize_real_look_lighting(
    env,
    env_ids,
    intensity_range: tuple[float, float],
    color_range: tuple[float, float] = (0.6, 1.0),
    skybox_textures: list[str] | None = None,
    skybox_swap_every: int = 8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
    look_intensity: float = 500.0,
    look_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    backdrop_gain_exponent: float = 0.5,
    room_hide_prob: float = 0.5,
):
    """openarm_domain_randomization.randomize_scene_lighting, relative to the look's light, and carried onto the backdrop.

    One draw per call for the one shared dome light (as the plain term). The draw is read as a ratio to the plain scene's
    nominal light (REFERENCE_LIGHT) and applied to the look's: intensity x ratio, colour x grey ratio. The emissive backdrop
    gets the grey ratio and ratio ** ``backdrop_gain_exponent`` (exposure: a 4x brighter lamp does not make a lit room
    look 4x brighter in an auto-exposed camera, and the robot, which the dome does light, keeps the larger change).
    The skybox swap and its throttle are the plain term's; each swap also re-draws whether the photo room is shown.
    """
    from ...mdp.openarm_domain_randomization import _SKYBOX_SWAP_STATE_ATTR

    light_prim = env.scene[asset_cfg.name].prims[0]
    ratio = random.uniform(*intensity_range) / REFERENCE_LIGHT[0]
    grey = random.uniform(*color_range) / REFERENCE_LIGHT[1]
    light_prim.GetAttribute("inputs:intensity").Set(float(look_intensity * ratio))
    light_prim.GetAttribute("inputs:color").Set(tuple(float(min(1.0, c * grey)) for c in look_color))

    if skybox_textures:
        swap = getattr(env, _SKYBOX_SWAP_STATE_ATTR, None)
        if swap is None:
            swap = {"calls": 0, "current": None}
            setattr(env, _SKYBOX_SWAP_STATE_ATTR, swap)
        swap["calls"] += 1
        if swap["calls"] % max(skybox_swap_every, 1) == 1:
            choice = random.choice(skybox_textures)
            if choice != swap["current"]:
                light_prim.GetAttribute("inputs:texture:file").Set(choice)
                swap["current"] = choice
            _set_room_visible(env, random.random() >= room_hide_prob)

    state = _state(env)
    state["gain"] = np.full(3, grey * ratio**backdrop_gain_exponent)
    _write(env, range(env.scene.num_envs))


def randomize_backdrop_tint(env, env_ids, value_range=(0.85, 1.15), channel_range=(0.95, 1.05)):
    """Per env: the photo table's colour, brightness x per-channel gain (the plain scene's pad colour term, under the look)."""
    state = _state(env)
    ids = range(env.scene.num_envs) if env_ids is None else [int(i) for i in env_ids]
    for i in ids:
        state["tint"][i] = random.uniform(*value_range) * np.array([random.uniform(*channel_range) for _ in range(3)])
    _write(env, ids)


def adapt_to_real_look(env_cfg, profile: str) -> list[str]:
    """Re-target the DR terms attach_domain_randomization put on ``env_cfg.events`` to a scene with the real look."""
    notes = []
    light = getattr(env_cfg.events, "randomize_light", None)
    if light is not None:
        params = dict(light.params)
        params.update(
            look_intensity=float(env_cfg.scene.light.spawn.intensity),
            look_color=tuple(float(c) for c in env_cfg.scene.light.spawn.color),
            room_hide_prob=0.5 if params.get("skybox_textures") else 0.0,
        )
        env_cfg.events.randomize_light = EventTerm(func=randomize_real_look_lighting, mode=light.mode, params=params)
        notes.append("lighting relative to the look's light, carried onto the photo backdrop"
                     + ("; skybox swap hides the photo room half the time" if params["room_hide_prob"] else ""))
    if getattr(env_cfg.events, "randomize_pad_color", None) is not None:
        env_cfg.events.randomize_pad_color = EventTerm(func=randomize_backdrop_tint, mode="reset", params=dict(TINT[profile]))
        notes.append("pad colour -> photo table tint")
    return notes
