# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Minimal URDF forward kinematics (numpy), for tools that need link poses at thousands of real states.

Rendering in Isaac takes ~0.5 s a frame; this takes microseconds and agrees with the articulation's own
link poses (checked against render_real_states.py's poses.npz -- see ``python urdf_fk.py --check``).
"""

import argparse
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_URDF = "source/isaaclab_assets/data/v1_camera_isaac/urdf/v1_camera.urdf"


def _origin(el) -> np.ndarray:
    T = np.eye(4)
    if el is not None:
        T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in el.get("rpy", "0 0 0").split()]).as_matrix()
        T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    return T


class URDFChain:
    def __init__(self, path: str = DEFAULT_URDF):
        root = ET.parse(path).getroot()
        self.joints = {}  # child link -> (parent link, joint name, type, origin, axis)
        for j in root.findall("joint"):
            axis = j.find("axis")
            self.joints[j.find("child").get("link")] = (
                j.find("parent").get("link"), j.get("name"), j.get("type"), _origin(j.find("origin")),
                np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else np.zeros(3))

    def pose(self, link: str, q: dict[str, float], base: str = "openarm_body_link0") -> np.ndarray:
        """4x4 pose of ``link`` in ``base`` for joint values ``q`` (sim joint names, rad / m)."""
        T = np.eye(4)
        while link != base:
            parent, name, kind, origin, axis = self.joints[link]
            M = origin.copy()
            if kind in ("revolute", "continuous"):
                M[:3, :3] = M[:3, :3] @ Rotation.from_rotvec(axis * q.get(name, 0.0)).as_matrix()
            elif kind == "prismatic":
                M[:3, 3] += M[:3, :3] @ (axis * q.get(name, 0.0))
            T = M @ T
            link = parent
        return T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", required=True, help="poses.npz + frames.npz dir pair: <poses.npz>,<frames.npz>")
    ap.add_argument("--calibration", default="/home/csl/Stanley_ws/lerobot_openarm/calibration.json")
    args = ap.parse_args()
    import json
    import sys

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from real_states import real_state_to_sim

    poses_path, frames_path = args.check.split(",")
    poses, frames = np.load(poses_path), np.load(frames_path)
    calib = json.load(open(args.calibration))
    names = [str(n) for n in frames["names"]]
    bodies = [str(b) for b in poses["body_names"]]
    chain = URDFChain()
    b0 = bodies.index("openarm_body_link0")
    worst = 0.0
    for k in range(0, len(frames["state"]), max(1, len(frames["state"]) // 20)):
        q = real_state_to_sim(frames["state"][k], names, calib)
        R0 = Rotation.from_quat(poses["body_quat_w"][k][b0][[1, 2, 3, 0]]).as_matrix()
        for link in ("openarm_left_ee_tcp", "openarm_right_ee_tcp", "openarm_left_link4"):
            i = bodies.index(link)
            R = Rotation.from_quat(poses["body_quat_w"][k][i][[1, 2, 3, 0]]).as_matrix()
            t = R0.T @ (poses["body_pos_w"][k][i] - poses["body_pos_w"][k][b0])
            T = chain.pose(link, q)
            dp = np.linalg.norm(T[:3, 3] - t) * 1000
            dr = np.degrees(Rotation.from_matrix(T[:3, :3].T @ (R0.T @ R)).magnitude())
            worst = max(worst, dp)
            print(f"frame {k:3d} {link:22s} {dp:6.2f} mm {dr:5.2f} deg")
    print(f"worst position error {worst:.2f} mm")


if __name__ == "__main__":
    main()
