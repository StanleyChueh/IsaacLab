# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""A scripted two-arm expert for the battery task: the left OpenArm lifts the charger, the right one sets a battery in its bay.

Plain numpy (no Isaac): inverse kinematics on the robot's own URDF (``urdf_fk.URDFChain``, the same chain the articulation
is built from) and a Cartesian waypoint planner. Everything is in the env frame; the robot's base pose
(``obs["robot_base"]``) turns it into the arm frame the kinematics live in.

The grasp point of a hand is a point on its tool axis (``GRASP_S`` from the wrist, link7), between the finger pads.
"""

import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from urdf_fk import URDFChain  # noqa: E402

ARM_JOINTS = {side: [f"openarm_{side}_joint{i}" for i in range(1, 8)] for side in ("left", "right")}
FINGER_TIP = 0.1829
"""m from the wrist (link7) along the tool axis to the stock finger's tip (finger frame 0.1025 + mesh end 0.0804)."""

_RHOME = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
"""Tool frame at the home pose (columns = tool x, y, z in the arm frame): tool z forward, tool x up, jaws close along y."""


def tool_rotation(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    """Rotation of the tool frame (arm frame): tool axis forward, turned ``yaw`` about z, pitched down by ``pitch`` (rad), and
    rolled by ``roll`` about the tool axis."""
    R = Rotation.from_euler("z", yaw).as_matrix() @ Rotation.from_euler("y", pitch).as_matrix() @ _RHOME
    return R @ Rotation.from_euler("z", roll).as_matrix()


REAL_RANGE = {  # per joint (low, high) in rad over the 9 real battery episodes (sim joint names), what a real operator used
    "openarm_left_joint1": (-0.16, 0.45), "openarm_left_joint2": (-0.58, 0.11), "openarm_left_joint3": (-0.02, 0.72),
    "openarm_left_joint4": (0.93, 1.93), "openarm_left_joint5": (-0.39, 0.23), "openarm_left_joint6": (-0.73, 0.08),
    "openarm_left_joint7": (-0.41, 0.36),
    "openarm_right_joint1": (-0.74, 0.31), "openarm_right_joint2": (-0.19, 0.85), "openarm_right_joint3": (-0.91, 0.01),
    "openarm_right_joint4": (0.85, 2.3), "openarm_right_joint5": (-0.36, 0.45), "openarm_right_joint6": (-0.32, 0.78),
    "openarm_right_joint7": (-0.99, 0.57),
}
GRASP_POSTURE = {
    "left": [0.093, 0.004, 0.127, 1.051, 0.079, -0.074, -0.199],
    "right": [-0.335, -0.176, -0.117, 1.188, 0.028, -0.097, 0.13],
}
"""Median joint posture (rad) of each real arm when its gripper closed on the charger / the battery, over the 9 real episodes."""
RANGE_MARGIN = float(os.environ.get("RANGE_MARGIN", 0.6))
"""The IK stays within the real operator's joint ranges, widened by this (rad): beyond them the sim arm presses on its own
body (left joint 2 above ~0.1, say), the joint stops tracking, and the hand ends up somewhere else."""


class ArmIK:
    """Damped-least-squares IK for one arm, to a pose of the wrist (link7) in the arm frame."""

    def __init__(self, side: str, chain: URDFChain | None = None):
        self.side = side
        self.chain = chain or URDFChain()
        self.names = ARM_JOINTS[side]
        self.link = f"openarm_{side}_link7"
        lo = np.array([self.chain.limits[n][0] for n in self.names])
        hi = np.array([self.chain.limits[n][1] for n in self.names])
        margin = np.array([0.04 if n.endswith("joint2") else RANGE_MARGIN for n in self.names])  # joint 2 swings the arm into its body
        rlo = np.array([REAL_RANGE[n][0] for n in self.names]) - margin
        rhi = np.array([REAL_RANGE[n][1] for n in self.names]) + margin
        self.lo, self.hi = np.maximum(lo, rlo), np.minimum(hi, rhi)
        j2 = self.names.index(f"openarm_{side}_joint2")  # beyond the real operator's range the sim arm presses on its own body
        cap = float(os.environ.get("J2_CAP", 0.0))  # the sim arms stall at j2 = +/-0.0 .. 0.13 depending on the other joints (swept)
        if side == "left":
            self.hi[j2] = min(self.hi[j2], cap)
        else:
            self.lo[j2] = max(self.lo[j2], -cap)
        self.lo_real, self.hi_real = self.lo.copy(), self.hi.copy()
        self.lo_wide, self.hi_wide = lo.copy(), hi.copy()  # the URDF's own limits (with the same joint-2 cap), for poses outside the operator's habits
        self.lo_wide[j2], self.hi_wide[j2] = self.lo[j2], self.hi[j2]
        self.q_pref = np.array(GRASP_POSTURE[side])  # the real operator's own posture when the jaws closed: sits in the collision-free region
        self.q_pref = np.clip(self.q_pref, self.lo, self.hi)

    def fk(self, q: np.ndarray) -> np.ndarray:
        return self.chain.pose(self.link, dict(zip(self.names, q)))

    def grasp_point(self, q: np.ndarray, s: float) -> np.ndarray:
        T = self.fk(q)
        return T[:3, 3] + T[:3, 2] * s

    def ik(self, T: np.ndarray, q0: np.ndarray, iters: int = 200, lam: float = 0.02, rot_weight: float = 0.5, tol=(1e-4, 2e-3)):
        """Solve for the wrist pose ``T`` (4x4) from ``q0``. Returns (q, position error m, rotation error rad)."""
        q = q0.astype(float).copy()
        for _ in range(iters):
            Tc = self.fk(q)
            ep = T[:3, 3] - Tc[:3, 3]
            er = Rotation.from_matrix(T[:3, :3] @ Tc[:3, :3].T).as_rotvec()
            if np.linalg.norm(ep) < tol[0] and np.linalg.norm(er) < tol[1]:
                break
            J = np.zeros((6, 7))
            for j in range(7):
                dq = q.copy()
                dq[j] += 1e-5
                Td = self.fk(dq)
                J[:3, j] = (Td[:3, 3] - Tc[:3, 3]) / 1e-5
                J[3:, j] = Rotation.from_matrix(Td[:3, :3] @ Tc[:3, :3].T).as_rotvec() / 1e-5
            e = np.concatenate([ep, rot_weight * er])
            Jw = J.copy()
            Jw[3:] *= rot_weight
            Jp = Jw.T @ np.linalg.inv(Jw @ Jw.T + lam**2 * np.eye(6))
            dq = Jp @ e + (np.eye(7) - Jp @ Jw) @ (0.5 * (self.q_pref - q))  # and drift to the middle of the real range
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        Tc = self.fk(q)
        er = Rotation.from_matrix(T[:3, :3] @ Tc[:3, :3].T).as_rotvec()
        return q, float(np.linalg.norm(T[:3, 3] - Tc[:3, 3])), float(np.linalg.norm(er))

    def ik_ls(self, T: np.ndarray, q0: np.ndarray, rot_weight: float = 0.35, posture: float = 4.0):
        """Bounded least-squares IK (scipy) to the wrist pose ``T``: millimetres of position error, rot_weight x 1000 per radian of
        rotation error, and a faint pull toward the real operator's posture. Returns (q, position error m, rotation error rad)."""
        from scipy.optimize import least_squares

        def residual(q):
            Tc = self.fk(q)
            er = Rotation.from_matrix(T[:3, :3] @ Tc[:3, :3].T).as_rotvec()
            return np.concatenate([1000.0 * (T[:3, 3] - Tc[:3, 3]), 1000.0 * rot_weight * er, posture * (q - self.q_pref)])

        res = least_squares(residual, np.clip(q0, self.lo + 1e-4, self.hi - 1e-4), bounds=(self.lo, self.hi), max_nfev=80)
        Tc = self.fk(res.x)
        er = Rotation.from_matrix(T[:3, :3] @ Tc[:3, :3].T).as_rotvec()
        return res.x, float(np.linalg.norm(T[:3, 3] - Tc[:3, 3])), float(np.linalg.norm(er))

    def ik_axis(self, p_wrist: np.ndarray, z_dir: np.ndarray, q0: np.ndarray, posture: float = 4.0):
        """Bounded least-squares IK for a wrist POSITION and TOOL DIRECTION only: the roll about the tool axis is free (a cell held
        end-on in the pads just turns about its own axis). Returns (q, position error m, axis error rad)."""
        from scipy.optimize import least_squares

        z_dir = z_dir / np.linalg.norm(z_dir)

        def residual(q):
            Tc = self.fk(q)
            return np.concatenate([1000.0 * (p_wrist - Tc[:3, 3]), 350.0 * np.cross(Tc[:3, 2], z_dir), posture * (q - self.q_pref)])

        res = least_squares(residual, np.clip(q0, self.lo + 1e-4, self.hi - 1e-4), bounds=(self.lo, self.hi), max_nfev=100)
        Tc = self.fk(res.x)
        return res.x, float(np.linalg.norm(p_wrist - Tc[:3, 3])), float(np.linalg.norm(np.cross(Tc[:3, 2], z_dir)))

    def ik_cell(self, target: np.ndarray, a_h: np.ndarray, c_h: np.ndarray, axis_w: np.ndarray, up_w: np.ndarray, q0: np.ndarray, posture: float = 4.0):
        """Bounded least-squares IK putting a held cell where it goes: its centre (``c_h``, in the wrist frame) at ``target``, its axis
        (``a_h`` in the wrist frame) along ``axis_w`` (either way), and the jaw axis (the wrist's y) parallel to the surface whose
        normal is ``up_w`` so the pads hang either side of the cell clear of the charger. The roll of hand and cell together about
        the cell's own axis is what the last constraint pins. Returns (q, position error m, axis error rad)."""
        from scipy.optimize import least_squares

        def residual(q):
            T = self.fk(q)
            R = T[:3, :3]
            return np.concatenate([1000.0 * (T[:3, 3] + R @ c_h - target), 350.0 * np.cross(R @ a_h, axis_w), [350.0 * float(R[:, 1] @ up_w)], posture * (q - self.q_pref)])

        res = least_squares(residual, np.clip(q0, self.lo + 1e-4, self.hi - 1e-4), bounds=(self.lo, self.hi), max_nfev=120)
        T = self.fk(res.x)
        R = T[:3, :3]
        return res.x, float(np.linalg.norm(T[:3, 3] + R @ c_h - target)), float(max(np.linalg.norm(np.cross(R @ a_h, axis_w)), abs(R[:, 1] @ up_w)))

    def ik_cell_multi(self, target, a_h, c_h, axis_w, up_w, seeds, restarts: int = 10):
        """``ik_cell`` from each of ``seeds``, then from random starts, until one lands (position < 2 mm, axes < 0.03 rad)."""
        best = None
        rng = np.random.default_rng(1)
        starts = list(seeds) + [self.q_pref + 0.4 * rng.standard_normal(7) for _ in range(restarts)]
        for q0 in starts:
            cand = self.ik_cell(target, a_h, c_h, axis_w, up_w, q0)
            if best is None or cand[1] + 0.05 * cand[2] < best[1] + 0.05 * best[2]:
                best = cand
            if best[1] < 2e-3 and best[2] < 0.03:
                break
        if best[1] >= 4e-3 or best[2] >= 0.05:  # not within the operator's joint habits: try the whole of the URDF's range
            self.lo, self.hi = self.lo_wide, self.hi_wide
            try:
                for q0 in list(seeds) + [self.q_pref]:
                    cand = self.ik_cell(target, a_h, c_h, axis_w, up_w, q0)
                    if cand[1] + 0.05 * cand[2] < best[1] + 0.05 * best[2]:
                        best = cand
            finally:
                self.lo, self.hi = self.lo_real, self.hi_real
        return best

    def ik_multi(self, T: np.ndarray, q0: np.ndarray, restarts: int = 12, **kw):
        """``ik_ls`` from ``q0`` and, when that stalls, from random starts inside the limits (the 7-DoF arm has several
        branches; the nearest one is wanted, but any that reaches beats one that does not)."""
        best = self.ik_ls(T, q0)
        if best[1] < 2e-3 and best[2] < 0.03:
            return best
        rng = np.random.default_rng(0)
        for k in range(restarts):
            start = self.q_pref + 0.4 * rng.standard_normal(7) if k % 2 == 0 else self.lo + rng.random(7) * (self.hi - self.lo)
            cand = self.ik_ls(T, start)
            if (cand[1] + 0.05 * cand[2]) < (best[1] + 0.05 * best[2]) - 1e-4:
                best = cand
            if best[1] < 2e-3 and best[2] < 0.03:
                break
        if best[1] >= 4e-3 or best[2] >= 0.05:  # not within the operator's joint habits: try the whole of the URDF's range
            self.lo, self.hi = self.lo_wide, self.hi_wide
            try:
                for start in (q0, self.q_pref):
                    cand = self.ik_ls(T, start)
                    if cand[1] + 0.05 * cand[2] < best[1] + 0.05 * best[2]:
                        best = cand
            finally:
                self.lo, self.hi = self.lo_real, self.hi_real
        return best

    def wrist_pose_for_grasp(self, point: np.ndarray, R: np.ndarray, s: float) -> np.ndarray:
        """Wrist pose that puts the grasp point (``s`` along the tool axis) at ``point`` with tool orientation ``R``."""
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = point - R[:, 2] * s
        return T


def base_transform(base_pose) -> np.ndarray:
    """4x4 pose of the robot's root link in the env frame, from ``obs["robot_base"]`` = [x, y, z, qw, qx, qy, qz]."""
    T = np.eye(4)
    T[:3, 3] = base_pose[:3]
    T[:3, :3] = Rotation.from_quat([base_pose[4], base_pose[5], base_pose[6], base_pose[3]]).as_matrix()
    return T


# ── The expert ───────────────────────────────────────────────────────────────

FINGER_OPEN, FINGER_CLOSED = 0.044, 0.0
"""Finger joint commands (m). Closed asks for q = 0, which the stock fingers cannot reach (their inner faces cross at the
centre): the commanded squeeze is whatever the gripper's effort limit gives."""
CONTROL_HZ = 30.0
PAD_TIP = FINGER_TIP + 0.07
"""m from the wrist to the fingertip of the padded fingers (the battery recording's, 7 cm longer than the stock ones)."""


def min_jerk(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u**3 * (10 - 15 * u + 6 * u**2)


def yaw_of(quat) -> float:
    return float(Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("zyx")[0])


def pose_matrix(pose) -> np.ndarray:
    """4x4 from [x, y, z, qw, qx, qy, qz]."""
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]]).as_matrix()
    return T


class Failed(RuntimeError):
    pass


class BatteryExpert:
    """Lift the charger with the left hand, set the battery into its bay with the right, put the charger back down.

    ``step(obs)`` takes the sim server's observation (``joints``, ``objects``, ``robot_base``) and returns the joint
    targets for the next control step as a dict of sim joint names, and whether the episode is over. Each stage plans from
    what the objects and hands are at when it starts, so a charger that slipped or a battery that rolled is planned around
    rather than ignored. Raises ``Failed`` when a pose is out of the arm's reach.
    """

    def __init__(self, **overrides):
        self.p = dict(
            # left hand (charger): tool pitch (rad), grasp point's distance in from the charger's -x end, its height over the mat
            l_pitch=np.radians(20.0), l_yaw=(-0.5, 0.3), r_yaw=(-0.2, 0.3), l_depth=0.0575, l_height=0.0235, l_s=PAD_TIP - 0.020,
            lift=0.07, hold_xy=(0.335, -0.015),
            # right hand (battery)
            r_pitch=np.radians(30.0), r_height=0.011, r_s=PAD_TIP - 0.004, r_lift=0.10, release=0.012, over=0.03,
            above=0.07, speed=1.5, ki=0.04,
        )
        self.p.update(overrides)
        meeting = self.p.get("meeting") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../source/isaaclab_assets/data/real2sim/battery_v01/meeting.json")
        entry = json.load(open(meeting))[0]  # battery_meeting.py's best: both arms' joints with the battery over the bay and the charger level
        self.meet_qL, self.meet_qR = np.array(entry["x"][:7]), np.array(entry["x"][7:])
        self.chain = URDFChain()
        self.arms = {side: ArmIK(side, self.chain) for side in ("left", "right")}
        self.stage = -1
        self.plan = []
        self.cmd = {}
        self.log = []
        self.retries = {}

    # -- helpers -----------------------------------------------------------------------------------------
    def _base(self, obs):
        return base_transform(obs["robot_base"])

    def _wrist_target(self, obs, side, point, yaw, pitch, s, roll=0.0):
        """Wrist pose (arm frame) that puts the hand's grasp point at ``point`` (env frame)."""
        Tb = self._base(obs)
        R_env = tool_rotation(yaw, pitch, roll)
        Rb = Tb[:3, :3]
        T = np.eye(4)
        T[:3, :3] = Rb.T @ R_env
        T[:3, 3] = Rb.T @ (np.asarray(point) - Tb[:3, 3]) - T[:3, :3][:, 2] * s
        return T

    def _q_now(self, side):
        return np.array([self.cmd[n] for n in self.arms[side].names])

    def _solve(self, obs, side, T_arm, label):
        q, ep, er = self.arms[side].ik_multi(T_arm, self._q_now(side))
        if ep > 0.010 or er > 0.08:
            raise Failed(f"{label}: {side} arm cannot reach the pose (error {ep * 1000:.1f} mm, {np.degrees(er):.1f} deg)")
        return q

    def _move(self, side, q_end, seconds, label, gripper=None, hold=0.0):
        """Queue a joint-space move of one arm (the other holds), optionally with the gripper going to ``gripper``."""
        end = dict(self.plan[-1]["end"] if self.plan else self.cmd)  # legs queued in one go build on each other (the gripper too)
        for n, v in zip(self.arms[side].names, q_end):
            end[n] = float(v)
        if gripper is not None:
            end[f"openarm_{side}_finger_joint1"] = float(gripper)
        self.plan.append(dict(end=end, steps=max(1, int(seconds / self.p["speed"] * CONTROL_HZ)), hold=int(hold * CONTROL_HZ), label=label))

    # -- the stages: each plans its moves from the observation at its start --------------------------------
    STAGES = ("settle", "r_grasp", "r_wait", "l_grasp", "l_lift", "l_carry", "r_over", "r_down", "r_open", "r_back", "l_return", "home")

    def _plan_stage(self, obs):
        P = self.p
        name = self.STAGES[self.stage] if 0 <= self.stage < len(self.STAGES) else None
        if name is None:
            self.stage = -2
            return
        ch, ba = pose_matrix(obs["objects"]["charger"]), pose_matrix(obs["objects"]["battery"])
        L, W, T = 0.115, 0.0652, 0.0375
        # did the last grasp take? a hand that closed on nothing (fingers nearly shut) or whose prop stayed on the mat goes round again
        retry = None
        j = obs["joints"]
        if name == "r_wait" and j["openarm_right_finger_joint1"] < 0.004:
            retry = "r_grasp"
        elif name == "l_grasp" and ba[2, 3] < 0.28 + 0.06:  # the right hand is meant to be holding the cell up by now
            retry = "r_grasp"
        elif name == "l_lift" and j["openarm_left_finger_joint1"] < 0.025:
            retry = "l_grasp"
        elif name == "l_carry" and ch[2, 3] < 0.28 + 0.0375 / 2 + 0.03:
            retry = "l_grasp"
        if retry is not None:
            self.retries[retry] = self.retries.get(retry, 0) + 1
            if self.retries[retry] > 2:
                raise Failed(f"{retry}: the grasp did not take after {self.retries[retry] - 1} retries")
            self.stage = self.STAGES.index(retry)
            return self._plan_stage(obs)
        if name == "settle":  # nothing to do for a moment
            self.plan.append(dict(end=dict(self.cmd), steps=15, hold=0, label="settle"))
        elif name == "r_grasp":  # right hand to the battery, pads either side of its middle: first, while the left hand is out of the way
            ax = ba[:3, :3] @ np.array([1.0, 0.0, 0.0])
            yaw = float(np.arctan2(ax[1], ax[0]))
            if abs(yaw) > np.pi / 2:
                yaw -= np.sign(yaw) * np.pi
            yaw = float(np.clip(yaw, *P["r_yaw"]))  # as far round as the arm reaches; the pads square it up
            self.grasp_r = np.array([ba[0, 3], ba[1, 3], 0.28 + P["r_height"]])
            self.yaw_r = yaw
            self._approach(obs, "right", self.grasp_r, yaw, P["r_pitch"], P["r_s"], "R")
            self._move("right", self.plan_q("right", -1), 0.9, "R_close", gripper=FINGER_CLOSED, hold=0.5)
        elif name == "r_wait":  # lift it clear of the mat and wait to the right, out of the way of the charger
            point = self.grasp_r + np.array([0.0, 0.0, P["r_lift"]])
            T_up = self._wrist_target(obs, "right", point, self.yaw_r, P["r_pitch"], P["r_s"])
            self._move("right", self._solve(obs, "right", T_up, "R_lift"), 1.6, "R_lift", hold=0.4)
        elif name == "l_grasp":  # left hand to the charger's middle, pads along its long sides
            yaw = float(np.clip(yaw_of(obs["objects"]["charger"][3:]), *P["l_yaw"]))
            end_pt = ch[:3, 3] + ch[:3, :3] @ np.array([-L / 2 + P["l_depth"], 0.0, 0.0])
            self.grasp_l = np.array([end_pt[0], end_pt[1], 0.28 + P["l_height"]])
            self.yaw_l = yaw
            self._approach(obs, "left", self.grasp_l, yaw, P["l_pitch"], P["l_s"], "L")
            self._move("left", self.plan_q("left", -1), 0.9, "L_close", gripper=FINGER_CLOSED, hold=0.5)
        elif name == "l_lift":  # straight up off the mat
            self.l_grasp_q = self._q_now("left")
            T_up = self._wrist_target(obs, "left", self.grasp_l + np.array([0.0, 0.0, P["lift"]]), self.yaw_l, P["l_pitch"], P["l_s"])
            self._move("left", self._solve(obs, "left", T_up, "L_lift"), 1.6, "L_lift", hold=0.5)
        elif name == "l_carry":  # to where the other hand will meet it
            # Not the nominal joints (the charger sits in the pads a little differently every time: turned, shifted, tilted) but the
            # hand pose that puts THE CHARGER at the nominal pose.
            Tb = self._base(obs)
            T_wh = Tb @ self.arms["left"].fk(self._q_now("left"))
            T_hc = np.linalg.inv(T_wh) @ ch  # the charger in the wrist frame, as gripped
            Rl = tool_rotation(0.0, P["l_pitch"])
            T_nom_h = np.eye(4)  # the nominal grip: the charger squared up to the pads, centred on the grasp point
            T_nom_h[:3, :3] = Rl.T
            T_nom_h[:3, 3] = Rl.T @ (Rl[:, 2] * P["l_s"] - np.array([0.0, 0.0, P["l_height"] + 0.28 - 0.2988]))
            T_c_nom = self.arms["left"].fk(self.meet_qL) @ T_nom_h  # the charger pose at the meeting posture (arm frame)
            T_goal = T_c_nom @ np.linalg.inv(T_hc)  # wrist pose that puts the charger there (arm frame)
            q_goal, ep, er = self.arms["left"].ik_multi(T_goal, self.meet_qL)
            if os.environ.get("EXPERT_DEBUG"):
                from scipy.spatial.transform import Rotation as _R
                print(f"    [expert] l_carry: charger {np.round(ch[:3, 3], 3)}; in hand {np.round(T_hc[:3, 3], 3)} vs nominal {np.round(T_nom_h[:3, 3], 3)}; "
                      f"rot diff {np.degrees(_R.from_matrix(T_hc[:3, :3].T @ T_nom_h[:3, :3]).magnitude()):.1f} deg; ik err {ep * 1000:.1f} mm {np.degrees(er):.1f} deg", flush=True)
            if ep > 0.006 or er > 0.06:
                raise Failed(f"L_carry: left arm cannot put the charger at the meeting pose (error {ep * 1000:.1f} mm, {np.degrees(er):.1f} deg)")
            self._move("left", q_goal, 3.2, "L_carry", hold=1.5)
        elif name in ("r_over", "r_down"):  # put the cell in the bay: over it first, then down into it, planned from where the charger
            # and the cell are NOW (the grips may have turned either)
            Tb = self._base(obs)
            q_r = np.array([obs["joints"][n] for n in self.arms["right"].names])
            T_wh = Tb @ self.arms["right"].fk(q_r)  # wrist in the env frame
            cx, cy = (-0.0415 + 0.0135) / 2, (0.0122 + 0.0302) / 2
            seat = ch[:3, 3] + ch[:3, :3] @ np.array([cx, cy, T / 2 - 0.012 + 0.00725])
            up, axis_c = ch[:3, :3][:, 2], ch[:3, :3][:, 0]
            height = P["over"] + P["release"] if name == "r_over" else P["release"]
            target = seat + up * height  # where the cell's centre goes
            a_h = T_wh[:3, :3].T @ ba[:3, :3][:, 0]  # the cell's axis and centre in the wrist frame: how it is held
            c_h = T_wh[:3, :3].T @ (ba[:3, 3] - T_wh[:3, 3])
            Rb_inv = np.linalg.inv(Tb[:3, :3])
            seeds = [self.meet_qR, self._q_now("right")] if name == "r_over" else [self._q_now("right"), self.meet_qR]
            q, ep, er = self.arms["right"].ik_cell_multi(Rb_inv @ (target - Tb[:3, 3]), a_h, c_h, Rb_inv @ axis_c, Rb_inv @ up, seeds)
            label = "R_over" if name == "r_over" else "R_down"
            if os.environ.get("EXPERT_DEBUG"):
                print(f"    [expert] {name}: cell {np.round(ba[:3, 3], 3)} -> {np.round(target, 3)}; charger {np.round(ch[:3, 3], 3)} axis {np.round(axis_c, 2)} up {np.round(up, 2)}; ik err {ep * 1000:.1f} mm {np.degrees(er):.1f} deg", flush=True)
            if ep > 0.004 or er > 0.05:
                raise Failed(f"{label}: right arm cannot reach the pose (error {ep * 1000:.1f} mm, {np.degrees(er):.1f} deg)")
            self._move("right", q, 3.4 if name == "r_over" else 1.6, label, hold=0.3)
        elif name == "r_open":  # let go
            self._move("right", self._q_now("right"), 0.7, "R_open", gripper=FINGER_OPEN, hold=0.6)
        elif name == "r_back":  # back off up and away
            q_r = np.array([obs["joints"][n] for n in self.arms["right"].names])
            Tb = self._base(obs)
            T_up = self.arms["right"].fk(q_r)
            T_up[:3, 3] += np.linalg.inv(Tb)[:3, :3] @ np.array([-0.06, -0.03, 0.08])
            self._move("right", self._solve(obs, "right", T_up, "R_back"), 1.6, "R_back", hold=0.2)
        elif name == "l_return":  # put the charger back on the mat, the way it came
            T_up = self._wrist_target(obs, "left", self.grasp_l + np.array([0.0, 0.0, P["lift"]]), self.yaw_l, P["l_pitch"], P["l_s"])
            q_up = self._solve(obs, "left", T_up, "L_return")
            self._move("left", q_up, 3.0, "L_return")
            self._move("left", self.l_grasp_q, 1.6, "L_down", hold=0.4)
            self._move("left", self.l_grasp_q, 0.8, "L_open", gripper=FINGER_OPEN, hold=0.4)
        elif name == "home":  # both hands home
            for side in ("left", "right"):
                self._move(side, self.q_home[side], 2.6, f"{side}_home", gripper=FINGER_OPEN)

    def _approach(self, obs, side, point, yaw, pitch, s, tag):
        """Open the hand and bring it to ``point``: up and clear of the props first (the home pose is in front of them), then
        back and above the grasp, then in along the tool axis. Each leg is solved from the previous one."""
        P = self.p
        R = tool_rotation(yaw, pitch)
        legs = []
        T_now = self.arms[side].fk(self._q_now(side))
        T_up = T_now.copy()
        T_up[:3, 3] = T_now[:3, 3] + np.array([-0.03, 0.0, 0.09])
        legs.append((T_up, 1.6, f"{tag}_up"))
        legs.append((self._wrist_target(obs, side, point + np.array([0.0, 0.0, P["above"]]), yaw, pitch, s), 2.2, f"{tag}_above"))
        legs.append((self._wrist_target(obs, side, point, yaw, pitch, s), 1.6, f"{tag}_in"))
        seed = dict(self.cmd)
        for k, (T, secs, label) in enumerate(legs):
            q = self._solve_from(obs, side, T, label, seed)
            self._move(side, q, secs, label, gripper=FINGER_OPEN if k == 0 else None)
            seed = self.plan[-1]["end"]
        return seed

    def plan_q(self, side, idx):
        end = self.plan[idx]["end"]
        return np.array([end[n] for n in self.arms[side].names])

    def _solve_from(self, obs, side, T_arm, label, seed):
        q0 = np.array([seed[n] for n in self.arms[side].names])
        q, ep, er = self.arms[side].ik_multi(T_arm, q0)
        if ep > 0.010 or er > 0.08:
            raise Failed(f"{label}: {side} arm cannot reach the pose (error {ep * 1000:.1f} mm, {np.degrees(er):.1f} deg)")
        return q

    # -- run ------------------------------------------------------------------------------------------------
    def reset(self, obs):
        self.cmd = dict(obs["joints"])  # the planned reference (before the integral correction)
        self.integral = {n: 0.0 for side in ARM_JOINTS for n in ARM_JOINTS[side]}
        self.ref_last = dict(self.cmd)
        self.retries = {}
        self.q_home = {side: np.array([obs["joints"][n] for n in self.arms[side].names]) for side in ("left", "right")}
        self.stage, self.plan, self.cur, self.k, self.done = -1, [], None, 0, False
        self.start = None

    def step(self, obs):
        """Next joint targets: the plan's reference plus an integral of how far each arm joint lags it (the PD joints sag under
        the arm's own weight by several centimetres at the hand, which a plan in Cartesian space cannot afford)."""
        for n in self.integral:
            self.integral[n] = float(np.clip(self.integral[n] + self.p["ki"] * (self.ref_last[n] - obs["joints"][n]), -0.4, 0.4))
        ref, done = self._step_ref(obs)
        self.ref_last = dict(ref)
        out = dict(ref)
        for n in self.integral:
            out[n] = ref[n] + self.integral[n]
        return out, done

    def _step_ref(self, obs):
        while True:
            if self.cur is None:
                if not self.plan:
                    self.stage += 1
                    if self.stage >= len(self.STAGES):
                        self.done = True
                        return dict(self.cmd), True
                    self._plan_stage(obs)
                    continue
                self.cur = self.plan.pop(0)
                self.start, self.k = dict(self.cmd), 0
            seg = self.cur
            self.k += 1
            if self.k <= seg["steps"]:
                u = min_jerk(self.k / seg["steps"])
                for n, v in seg["end"].items():
                    self.cmd[n] = self.start[n] + u * (v - self.start[n])
                return dict(self.cmd), False
            if self.k <= seg["steps"] + seg["hold"]:
                return dict(self.cmd), False
            self.cur = None
