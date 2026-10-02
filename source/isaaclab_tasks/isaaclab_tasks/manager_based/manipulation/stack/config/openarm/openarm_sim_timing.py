# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Single source of truth for the OpenArm pick-up task family's control rate.

WHY THIS EXISTS. The task family was tuned at 20 Hz (``sim.dt = 0.01`` x ``decimation = 5`` in
stack_env_cfg.py), but the LeRobot datasets were converted with ``--fps 30`` and every real-robot
script (mirror_bridge, deploy_*) runs at 30 Hz. ``convert_hdf5_to_lerobot.py`` only RELABELS the
frame rate, it does not resample -- so 20 Hz sim motion was being played back on the real arm 1.5x
too fast. Running the sim at the rate everything else uses removes that mismatch at its source.

The rate is chosen once, here, and everything that counts steps derives from it:

  * ``CONTROL_HZ``            env step rate. Default 30; ``OPENARM_CONTROL_HZ=20`` restores the
                              legacy behaviour exactly (dt 0.01, decimation 5, unscaled counts).
  * ``scale_steps(n)``        a step count that was tuned at 20 Hz, rescaled so it spans the same
                              wall-clock duration at CONTROL_HZ.
  * ``apply_control_rate()``  writes decimation / sim.dt / render_interval onto an env cfg.
  * ``source_rate_hz()`` /    read the rate a recorded HDF5 was made at (it is stored in
    ``check_source_rate()``   ``data.attrs['env_args']['sim_args']``) and refuse to replay it in an
                              env running at a different rate.

Only the two rates below are supported because the physics step is fixed per rate, not derived:
a rate that is not an integer number of physics steps would silently change the sim time per step.

WHAT CHANGES AT 30 HZ. Physics runs at 120 Hz (was 100 Hz) with 4 substeps per step (was 5), so the
step costs slightly LESS physics than before while advancing 33 ms of sim time instead of 50 ms.
Contacts are resolved at a finer substep than before, not a coarser one.

WHAT DOES NOT CHANGE. Anything expressed in seconds or rad/s (episode_length_s, joint velocity
limits, ``--max-joint-speed``) is rate-independent. Anything expressed in STEPS is not, which is
the whole reason ``scale_steps`` exists -- and why a source demo recorded at one rate cannot be
replayed by Mimic at another: the generator replays source waypoints one per env step.
"""

import json
import os

REFERENCE_HZ = 20.0
"""Rate at which every step-count constant in this task family was originally tuned."""

# control rate (Hz) -> (physics dt in s, decimation). dt * decimation == 1 / rate exactly.
_SUPPORTED = {
    20: (0.01, 5),
    30: (1.0 / 120.0, 4),
}


def _read_control_hz() -> int:
    raw = os.environ.get("OPENARM_CONTROL_HZ", "30")
    try:
        hz = int(raw)
    except ValueError as e:
        raise ValueError(f"OPENARM_CONTROL_HZ must be an integer, got {raw!r}") from e
    if hz not in _SUPPORTED:
        raise ValueError(f"OPENARM_CONTROL_HZ={hz} is unsupported; choose one of {sorted(_SUPPORTED)}")
    return hz


CONTROL_HZ: int = _read_control_hz()
STEP_SCALE: float = CONTROL_HZ / REFERENCE_HZ


def scale_steps(n_at_reference: float, minimum: int = 1) -> int:
    """Rescale a step count tuned at ``REFERENCE_HZ`` to span the same duration at ``CONTROL_HZ``."""
    return max(minimum, int(round(n_at_reference * STEP_SCALE)))


def scale_odd(n_at_reference: float) -> int:
    """Like :func:`scale_steps` but rounded DOWN to an odd count -- for symmetric filter windows.

    A centred moving average is only phase-free with an odd width (``smooth_eef_pose_segment``
    rounds an even one UP, which would silently widen the filter), so pick the odd width ourselves.
    5 -> 5 at 20 Hz, 7 at 30 Hz.
    """
    return max(1, int(n_at_reference * STEP_SCALE) | 1)


def physics_params(hz: int | None = None) -> tuple[float, int]:
    """(physics dt, decimation) for a control rate."""
    hz = CONTROL_HZ if hz is None else hz
    if hz not in _SUPPORTED:
        raise ValueError(f"unsupported control rate {hz} Hz; choose one of {sorted(_SUPPORTED)}")
    return _SUPPORTED[hz]


def apply_control_rate(env_cfg, hz: int | None = None) -> None:
    """Set decimation, physics dt and render interval on *env_cfg* IN PLACE.

    ``render_interval`` is set to the decimation so the cameras render once per env step, on the
    last substep, and are never stale. (The legacy 2-with-decimation-5 rendered twice per step and
    left the second render one substep before the observation was read; record_demos_openarm.py's
    ``--sim_decimation`` override already follows the render_interval == decimation convention.)
    At 20 Hz the legacy ``render_interval = 2`` is kept so that mode reproduces old behaviour.
    """
    hz = CONTROL_HZ if hz is None else hz
    dt, decimation = physics_params(hz)
    env_cfg.sim.dt = dt
    env_cfg.decimation = decimation
    env_cfg.sim.render_interval = 2 if hz == 20 else decimation


def env_control_hz(env_cfg) -> float:
    """The control rate an env cfg actually resolves to."""
    return 1.0 / (env_cfg.sim.dt * env_cfg.decimation)


def source_rate_hz(hdf5_path: str) -> float | None:
    """Control rate a recorded Isaac Lab HDF5 was made at, or None if it does not say.

    Isaac Lab's recorder stores ``sim_args`` (dt, decimation, ...) in ``data.attrs['env_args']``.
    """
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        raw = f["data"].attrs.get("env_args")
    if raw is None:
        return None
    try:
        sim = json.loads(raw)["sim_args"]
        return 1.0 / (float(sim["dt"]) * int(sim["decimation"]))
    except (KeyError, TypeError, ValueError):
        return None


def check_source_rate(hdf5_path: str, env_cfg, allow_mismatch: bool = False) -> None:
    """Refuse to replay a source HDF5 in an env running at a different control rate.

    Mimic annotates and regenerates by replaying the source actions one per env step, so a 20 Hz
    recording replayed in a 30 Hz env is not smoother, it is the same path run 1.5x too fast -- and
    it would silently produce a dataset whose timing no longer matches what was recorded.
    """
    src = source_rate_hz(hdf5_path)
    env = env_control_hz(env_cfg)
    if src is None:
        print(f"[RATE] {hdf5_path} does not record its control rate; assuming it matches the env ({env:.1f} Hz).")
        return
    if abs(src - env) < 0.5:
        print(f"[RATE] source demos and env both run at {env:.1f} Hz.")
        return
    msg = (
        f"[RATE] {hdf5_path} was recorded at {src:.1f} Hz but this env steps at {env:.1f} Hz. Replaying it"
        f" here plays the demos {env / src:.2f}x too fast (one source waypoint per env step). Either"
        f" re-record at {env:.0f} Hz, or run with OPENARM_CONTROL_HZ={src:.0f} to match the source."
    )
    if allow_mismatch:
        print(msg + "  (continuing: --allow_rate_mismatch)")
        return
    raise SystemExit(msg)
