# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Apply a fitted real-arm actuator profile to the OpenArm env cfg (opt-in).

The profile is the JSON written by ``scripts/tools/sysid/fit_sim_actuators.py`` -- per joint group, the
stiffness / damping / armature that made the sim arm track the real arm's recorded response. This module
only WRITES those numbers onto the robot's actuator cfgs before the env is built; the sim stays an
ordinary Isaac Lab implicit-PD arm.

Opt-in on purpose. The default actuator values (stack_joint_pos_env_cfg.py, taken from the OpenArm
MuJoCo XML) were checked against the real arm's recorded motion and already agree with it as well as the
data can tell, so nothing is applied unless ``OPENARM_ACTUATOR_PROFILE=<path to json>`` is set. Groups the
fit did not improve carry the baseline values, so applying such a profile is a no-op by construction.

Rebuilding the whole per-joint dict, rather than adding one key, is deliberate: the existing cfgs key
stiffness / damping by regex over joint pairs ("openarm_left_joint[1-2]"), and Isaac Lab refuses a joint
that two patterns match. Every joint in the actuator group is therefore spelled out by its exact name.
"""

import json
import os
import re

ENV_VAR = "OPENARM_ACTUATOR_PROFILE"
_FIELDS = (("stiffness", "stiffness"), ("damping", "damping"), ("armature", "armature"))


def _resolve(value, joint_name: str):
    """Value a (float | dict-of-regex) actuator parameter takes for one joint, or None if unset."""
    if value is None or isinstance(value, (int, float)):
        return value
    for pattern, v in value.items():
        if re.fullmatch(pattern, joint_name):
            return v
    return None


def _joints_of(actuator_cfg) -> list[str]:
    """Concrete joint names an actuator cfg's ``joint_names_expr`` stands for (left/right x index)."""
    names = []
    for expr in actuator_cfg.joint_names_expr:
        for side in ("left", "right"):
            # the robot's real joints: arm 1..7 and exactly two (mimicked) finger joints per hand
            candidates = [f"openarm_{side}_joint{n}" for n in range(1, 8)] + [f"openarm_{side}_finger_joint{n}" for n in (1, 2)]
            for cand in candidates:
                if re.fullmatch(expr, cand) and cand not in names:
                    names.append(cand)
    return names


def apply_actuator_profile(env_cfg, path: str | None = None) -> str | None:
    """Write a fitted profile onto ``env_cfg.scene.robot``'s actuators. Returns a summary line, or None."""
    path = path or os.environ.get(ENV_VAR)
    if not path:
        return None
    profile = json.load(open(os.path.expanduser(path)))
    per_joint = {}
    for group in profile["groups"].values():
        for name in group["joint_names"]:
            per_joint[name] = {f: group[f] for f, _ in _FIELDS}

    changed = []
    for act_name, act in env_cfg.scene.robot.actuators.items():
        joints = _joints_of(act)
        for field, attr in _FIELDS:
            current = getattr(act, attr, None)
            new = {}
            for j in joints:
                fitted = per_joint.get(j, {}).get(field)
                base = _resolve(current, j)
                if fitted is None and base is None:
                    continue
                new[j] = float(fitted if fitted is not None else base)
                if fitted is not None and base is not None and abs(float(fitted) - float(base)) > 1e-9:
                    changed.append(f"{j}.{field}")
            if new:
                setattr(act, attr, new)
    return (f"[ACTUATORS] profile {path}: {len(changed)} value(s) differ from the cfg defaults"
            + (f" ({', '.join(changed[:6])}{' ...' if len(changed) > 6 else ''})" if changed else " (profile equals baseline)"))
