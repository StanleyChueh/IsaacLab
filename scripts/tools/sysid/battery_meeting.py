# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Find where the two OpenArm hands can meet to put the battery into the charger, over BOTH arms' joints at once.

The left hand holds the charger up level, the right hand holds the battery over its open bay with the battery's axis along
the bay's. How the objects sit in the two hands is fixed by how they were grasped off the mat (pads at the charger's
centre, pads on the battery's centre, both tools pitched down by ``pitch``), so what is free is where in the air this
happens and which way the hands face. The hand-picked meeting poses do not exist for these arms' joint ranges (the arms cannot
both cross the middle at those heights); letting every joint of both arms vary, with the charger kept level and above
``min_hold``, finds the ones that do. Result: ``meeting.json`` next to this file's look folder (joint postures of both arms
at the bay, the charger's pose), used by ``battery_expert.BatteryExpert`` as the pose to carry the charger to.

  python scripts/tools/sysid/battery_meeting.py --out source/isaaclab_assets/data/real2sim/battery_v01/meeting.json
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery_expert import PAD_TIP, ArmIK, tool_rotation  # noqa: E402
from urdf_fk import URDFChain  # noqa: E402

SL, SR = PAD_TIP - 0.020, PAD_TIP - 0.004  # grasp points along each tool axis (battery_expert's l_s / r_s)
GRASP_UP_L, GRASP_UP_R = 0.0047, 0.0037  # grasp point above the charger's / the battery's centre
BAY = np.array([(-0.0415 + 0.0135) / 2, (0.0122 + 0.0302) / 2, 0.0375 / 2 - 0.012 + 0.00725])  # the bay's seat, charger frame


def relations(pitch_l: float, pitch_r: float):
    """The charger in the left hand and the battery in the right hand, as (rotation, centre) in the wrist frames.

    Both are gripped as they are off the mat: the charger across its width at its centre, the battery across its diameter
    at its centre (pads straddle it, the cell's axis in the pads' plane, tools pitched down by the given angles)."""
    Rl, Rr = tool_rotation(0.0, pitch_l), tool_rotation(0.0, pitch_r)
    t_hc = Rl.T @ (Rl[:, 2] * SL - np.array([0, 0, GRASP_UP_L]))
    t_hb = Rr.T @ (Rr[:, 2] * SR - np.array([0, 0, GRASP_UP_R]))
    return (Rl.T, t_hc), (Rr.T, t_hb)


class Problem:
    def __init__(self, pitch_l, pitch_r, min_hold, drop, chain, max_tilt=35.0, max_roll=35.0):
        self.max_tilt, self.max_roll = max_tilt, max_roll
        self.want = (0.40, -0.02, min_hold + 0.01)
        self.pull = 150.0
        self.L, self.R = ArmIK("left", chain), ArmIK("right", chain)
        (self.Rhc, self.thc), (self.Rhb, self.thb) = relations(pitch_l, pitch_r)
        self.min_hold, self.drop = min_hold, drop

    def poses(self, x):
        TL, TR = self.L.fk(x[:7]), self.R.fk(x[7:])
        self.R_hand = TR[:3, :3]
        Rc, pc = TL[:3, :3] @ self.Rhc, TL[:3, 3] + TL[:3, :3] @ self.thc
        Rb, pb = TR[:3, :3] @ self.Rhb, TR[:3, 3] + TR[:3, :3] @ self.thb
        return Rc, pc, Rb, pb

    def residuals(self, x):
        Rc, pc, Rb, pb = self.poses(x)
        target = pc + Rc @ (BAY + np.array([0, 0, self.drop]))
        r = list(1000.0 * (pb - target))  # mm
        r += list(300.0 * np.cross(Rb[:, 0], Rc[:, 0]))  # battery axis along the bay's (either way)
        r.append(300.0 * max(0.0, abs(Rc[2, 1]) - np.sin(np.radians(self.max_roll))))  # the charger may roll about its long axis by up to max_roll (a tilted trough still holds the cell) ...
        r.append(300.0 * float(self.R_hand[:, 1] @ Rc[:, 2]))  # the right hand's jaw axis stays parallel to the charger's top: the pads hang either side of the cell, clear of the charger
        r.append(300.0 * max(0.0, np.cos(np.radians(self.max_tilt)) - Rc[2, 2]))  # ... and it may pitch end-down by up to max_tilt
        r.append(1000.0 * max(0.0, self.min_hold - pc[2]))
        r.append(1000.0 * max(0.0, pc[2] - self.min_hold - 0.04))  # and not much higher
        r += list(self.pull * (pc - np.array(self.want)))  # and toward where there is room: in front of the robot, clear of where the cell is picked up
        r += list(4.0 * (x[:7] - self.L.q_pref)) + list(4.0 * (x[7:] - self.R.q_pref))
        lo, hi = np.concatenate([self.L.lo, self.R.lo]), np.concatenate([self.L.hi, self.R.hi])
        r += list(12.0 * np.maximum(0.0, 0.12 - (x - lo))) + list(12.0 * np.maximum(0.0, 0.12 - (hi - x)))  # keep off the stops
        return np.array(r)


def solve(job):
    seed, pitch_l, pitch_r, min_hold, drop = job
    rng = np.random.default_rng(seed)
    chain = URDFChain()
    prob = Problem(np.radians(pitch_l), np.radians(pitch_r), min_hold, drop, chain, max_tilt=float(os.environ.get('MAX_TILT', 35)), max_roll=float(os.environ.get('MAX_ROLL', 35)))
    lo, hi = np.concatenate([prob.L.lo, prob.R.lo]), np.concatenate([prob.L.hi, prob.R.hi])
    x0 = lo + rng.random(14) * (hi - lo)
    res = least_squares(prob.residuals, x0, bounds=(lo, hi), max_nfev=150)
    Rc, pc, Rb, pb = prob.poses(res.x)
    target = pc + Rc @ (BAY + np.array([0, 0, drop]))
    return dict(seed=seed, pitch_l=pitch_l, pitch_r=pitch_r, x=res.x.tolist(), pos_err_mm=float(1000 * np.linalg.norm(pb - target)),
                axis_err_deg=float(np.degrees(np.arcsin(min(1.0, np.linalg.norm(np.cross(Rb[:, 0], Rc[:, 0])))))),
                tilt_deg=float(np.degrees(np.arccos(np.clip(Rc[2, 2], -1, 1)))), roll_deg=float(np.degrees(np.arcsin(np.clip(Rc[2, 1], -1, 1)))), jaw_err_deg=float(np.degrees(np.arcsin(np.clip(abs(prob.R_hand[:, 1] @ Rc[:, 2]), 0, 1)))), charger=pc.tolist(), charger_yaw=float(np.arctan2(Rc[1, 0], Rc[0, 0])),
                cost=float(res.cost))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--starts", type=int, default=240)
    ap.add_argument("--min_hold", type=float, default=0.36, help="Charger centre height at the bay (m); the mat is at 0.28.")
    ap.add_argument("--drop", type=float, default=0.014, help="Battery height over its seat at release (m).")
    ap.add_argument("--pitches", default="20:20,20:30,25:25,30:30")
    args = ap.parse_args()
    jobs = []
    for s in range(args.starts):
        pl, pr = [float(v) for v in args.pitches.split(",")[s % len(args.pitches.split(","))].split(":")]
        jobs.append((s, pl, pr, args.min_hold, args.drop))
    with Pool(14) as pool:
        out = pool.map(solve, jobs, chunksize=4)
    good = [o for o in out if o["pos_err_mm"] < 1.0 and o["axis_err_deg"] < 2.0 and o["tilt_deg"] < 50.0 and abs(o["roll_deg"]) < 36.0 and o["jaw_err_deg"] < 3.0 and o["charger"][2] > args.min_hold - 0.005 and o["charger"][2] < args.min_hold + 0.045]
    print(f"{len(good)} of {len(out)} starts meet (position < 1 mm, axis < 2 deg, level, held at >= {args.min_hold} m)")
    good.sort(key=lambda o: o["cost"])
    for o in good[:10]:
        print(f"  cost {o['cost']:.0f} pitches {o['pitch_l']:.0f}/{o['pitch_r']:.0f} charger ({o['charger'][0]:.2f}, {o['charger'][1]:.2f}, {o['charger'][2]:.2f}) yaw {np.degrees(o['charger_yaw']):.0f} deg")
    json.dump(good[:20], open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
