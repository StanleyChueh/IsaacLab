"""Small, simulator-independent contract for a dataset-fitted OpenArm scene profile.

The profile identifies where a look came from and which task scenes it may be applied to.
Task geometry and success logic live in the task cfg; this module only describes reusable
rendering assets and an optional physics calibration target.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_profile(directory: str, task: str | None = None) -> dict:
    root = Path(directory).expanduser().resolve()
    path = root / "profile.json"
    if not path.is_file():
        raise ValueError(f"No real-to-sim profile at {path}; provide a folder with profile.json.")
    profile = json.loads(path.read_text())
    if profile.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version 1")
    if not isinstance(profile.get("dataset_repo_id"), str) or not profile["dataset_repo_id"]:
        raise ValueError(f"{path}: dataset_repo_id is required")
    tasks = profile.get("compatible_tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(t, str) for t in tasks):
        raise ValueError(f"{path}: compatible_tasks must be a nonempty list of task IDs")
    if task is not None and task not in tasks:
        raise ValueError(f"{path}: profile is for {tasks}, not {task!r}")
    cameras = profile.get("cameras")
    if not isinstance(cameras, list) or not cameras or not all(isinstance(c, str) for c in cameras):
        raise ValueError(f"{path}: cameras must be a nonempty list")
    if len(cameras) != len(set(cameras)):
        raise ValueError(f"{path}: duplicate cameras")
    randomization = profile.get("compatible_randomization", ["none"])
    if not isinstance(randomization, list) or not randomization or not all(isinstance(r, str) for r in randomization):
        raise ValueError(f"{path}: compatible_randomization must be a nonempty list")
    hidden = profile.get("hide_visuals", [])
    if not isinstance(hidden, list) or not all(isinstance(name, str) for name in hidden):
        raise ValueError(f"{path}: hide_visuals must be a list of scene entity names")
    contact_asset = profile.get("contact_asset")
    if contact_asset is not None and not isinstance(contact_asset, str):
        raise ValueError(f"{path}: contact_asset must be a scene entity name")
    required = profile.get("required_assets", [])
    if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
        raise ValueError(f"{path}: required_assets must be a list of relative file paths")
    for name in required:
        asset = Path(name)
        if asset.is_absolute() or ".." in asset.parts or not (root / asset).is_file():
            raise ValueError(f"{path}: required asset missing or outside profile: {name!r}")
    return profile


def contact_targets(profile: dict, calibration: dict) -> list[tuple[str, dict]]:
    """Normalize legacy single-object and new multi-object contact fits.

    New files use ``{"targets": [{"asset": "battery", "params": {...}}, ...]}``.
    The older Pringles file uses ``{"params": {...}}`` and the profile's contact_asset.
    """
    if "targets" in calibration:
        targets = calibration["targets"]
        if not isinstance(targets, list) or not targets:
            raise ValueError("contact_physics.json: targets must be a nonempty list")
        result = []
        for item in targets:
            if not isinstance(item, dict) or not isinstance(item.get("asset"), str) or not isinstance(item.get("params"), dict):
                raise ValueError("contact_physics.json: each target needs asset and params")
            result.append((item["asset"], item["params"]))
        if len({asset for asset, _ in result}) != len(result):
            raise ValueError("contact_physics.json: duplicate target asset")
        return result
    asset = profile.get("contact_asset")
    if not asset or not isinstance(calibration.get("params"), dict):
        raise ValueError("contact_physics.json: expected targets, or params with profile contact_asset")
    return [(asset, calibration["params"])]
