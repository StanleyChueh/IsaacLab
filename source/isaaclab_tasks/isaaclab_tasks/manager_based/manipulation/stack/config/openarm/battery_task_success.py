"""Physical success check for the OpenArm battery/charger task.

Both objects must be lifted during the episode and the loose cell must remain seated in the
open bay for 15 control steps. The slot geometry is approximate until measured CAD is supplied.
"""

import torch

from isaaclab.utils import math as math_utils

from .openarm_battery_assets import CHARGER_SIZE, EMPTY_SLOT_CENTER, EMPTY_SLOT_DEPTH, BATTERY_RADIUS

_MEMORY_ATTR = "_openarm_battery_insertion_progress"


def _memory(env):
    memory = getattr(env, _MEMORY_ATTR, None)
    if memory is None or len(memory["seated_steps"]) != env.num_envs:
        memory = {
            "charger_lifted": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "battery_lifted": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "seated_steps": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        }
        setattr(env, _MEMORY_ATTR, memory)
    return memory


def reset_battery_progress(env, env_ids: torch.Tensor):
    memory = _memory(env)
    for value in memory.values():
        value[env_ids] = 0


def battery_insertion_success(env) -> torch.Tensor:
    charger, battery = env.scene["charger"], env.scene["battery"]
    memory = _memory(env)
    origins = env.scene.env_origins
    memory["charger_lifted"] |= charger.data.root_pos_w[:, 2] - origins[:, 2] > (
        float(env.cfg.scene.charger.init_state.pos[2]) + 0.035)
    memory["battery_lifted"] |= battery.data.root_pos_w[:, 2] - origins[:, 2] > (
        float(env.cfg.scene.battery.init_state.pos[2]) + 0.020)

    relative = math_utils.quat_apply_inverse(
        charger.data.root_quat_w, battery.data.root_pos_w - charger.data.root_pos_w)
    x, y = EMPTY_SLOT_CENTER
    seated_z = CHARGER_SIZE[2] / 2 - EMPTY_SLOT_DEPTH + BATTERY_RADIUS
    axis = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
    charger_axis = math_utils.quat_apply(charger.data.root_quat_w, axis)
    battery_axis = math_utils.quat_apply(battery.data.root_quat_w, axis)
    aligned = torch.abs((charger_axis * battery_axis).sum(dim=-1)) > 0.90
    seated = (
        (relative[:, 0] - x).abs() < 0.010
    ) & (
        (relative[:, 1] - y).abs() < 0.004
    ) & (
        (relative[:, 2] - seated_z).abs() < 0.005
    ) & aligned
    memory["seated_steps"] = torch.where(seated, memory["seated_steps"] + 1, 0)
    charger_held_now = charger.data.root_pos_w[:, 2] - origins[:, 2] > (
        float(env.cfg.scene.charger.init_state.pos[2]) + 0.025)
    return (memory["charger_lifted"] & memory["battery_lifted"] & charger_held_now
            & (memory["seated_steps"] >= 15))
