"""Run with: python -m unittest scripts.tools.test.test_real_look_profile"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE = (
    Path(__file__).resolve().parents[3]
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/openarm/real_look_profile.py"
)
spec = importlib.util.spec_from_file_location("real_look_profile", MODULE)
profile_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile_module)


class TestRealLookProfile(unittest.TestCase):
    def test_shipped_profiles_are_task_scoped(self):
        assets = Path(__file__).resolve().parents[3] / "source/isaaclab_assets/data/real2sim"
        pringles = profile_module.load_profile(
            str(assets / "pringles_v00"), "Isaac-PickUp-RedCube-OpenArm-IK-Abs-v1"
        )
        battery = profile_module.load_profile(
            str(assets / "battery_v01"), "Isaac-InsertBattery-OpenArm-IK-Abs-v0"
        )
        self.assertEqual(pringles["contact_asset"], "can")
        self.assertNotIn("contact_asset", battery)
        self.assertEqual(pringles["compatible_randomization"], ["none"])
        self.assertEqual(battery["compatible_randomization"], ["none"])
        fitted = json.loads((assets / "pringles_v00/contact_physics.json").read_text())
        self.assertEqual(profile_module.contact_targets(pringles, fitted)[0][0], "can")
        with self.assertRaisesRegex(ValueError, "profile is for"):
            profile_module.load_profile(str(assets / "pringles_v00"), "Isaac-InsertBattery-OpenArm-IK-Abs-v0")

    def test_rejects_invalid_camera_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "dataset_repo_id": "example/task",
                "compatible_tasks": ["Example-v0"],
                "cameras": ["body_cam", "body_cam"],
            }))
            with self.assertRaisesRegex(ValueError, "duplicate cameras"):
                profile_module.load_profile(directory, "Example-v0")

    def test_rejects_missing_required_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "profile.json").write_text(json.dumps({
                "schema_version": 1,
                "dataset_repo_id": "example/task",
                "compatible_tasks": ["Example-v0"],
                "cameras": ["body_cam"],
                "required_assets": ["missing.png"],
            }))
            with self.assertRaisesRegex(ValueError, "required asset missing"):
                profile_module.load_profile(directory, "Example-v0")

    def test_contact_targets_support_multiple_objects_and_legacy_fit(self):
        profile = {"contact_asset": "can"}
        self.assertEqual(profile_module.contact_targets(profile, {"params": {"can_mass": 0.1}}),
                         [("can", {"can_mass": 0.1})])
        fit = {"targets": [
            {"asset": "charger", "params": {"object_friction": 0.5}},
            {"asset": "battery", "params": {"object_mass": 0.027}},
        ]}
        self.assertEqual([name for name, _ in profile_module.contact_targets({}, fit)], ["charger", "battery"])
        with self.assertRaisesRegex(ValueError, "duplicate target asset"):
            profile_module.contact_targets({}, {"targets": [fit["targets"][0], fit["targets"][0]]})


if __name__ == "__main__":
    unittest.main()
