"""Shared sim-side half of the OpenArm real-robot mirror bridge.

Nothing here ever touches hardware. It only broadcasts the sim robot's joint positions as UDP
JSON packets, and listens for whatever a separate, out-of-process bridge chooses to send back.
The hardware half lives in lerobot_openarm/mirror_bridge.py, which owns calibration, speed
clamping, the startup handshake, and every decision about stale or missing packets.

Extracted verbatim from record_demos_openarm.py (which imports it from here now) so that
scripts/imitation_learning/lerobot/eval_smolvla_jointspace.py can mirror an autonomous policy
rollout onto the real robot over the exact same wire format a teleop recording session uses --
one bridge process, one packet schema, one plot, whichever sim-side script is driving.

Wire format (sim -> bridge, UDP JSON): {"seq": int, "t": float, "joints": {joint_name: radians}}
Feedback   (bridge -> sim, UDP JSON): {"t": float, "joints": {joint_name: radians}}
"""

import json
import os
import re
import socket
import threading
import time

import matplotlib
matplotlib.use("Agg")  # headless -- this module only ever saves a PNG, never shows a window
import matplotlib.pyplot as plt


class JointMirrorBroadcaster:
    """Best-effort UDP broadcaster of the robot's current joint positions.

    Fire-and-forget by design: never blocks and never raises into the sim loop.
    A separate, out-of-process bridge (e.g. lerobot_openarm/mirror_bridge.py) is
    responsible for everything hardware-related, including deciding what to do
    about stale/missing packets. This class does not know a real robot exists.
    """

    JOINT_NAME_PATTERNS = [
        r"openarm_left_joint[1-7]",
        r"openarm_right_joint[1-7]",
        r"openarm_left_finger_joint.*",
        r"openarm_right_finger_joint.*",
    ]

    @classmethod
    def resolve_mirror_joint_indices(cls, robot) -> tuple[list[int], list[str]]:
        """Return (indices, names) of the OpenArm joints within robot.data.joint_names,
        in the SAME order they appear in robot.data.joint_pos -- i.e. the same column
        order recorded into the HDF5 dataset's states/articulation/robot/joint_position.
        """
        all_names = robot.data.joint_names
        pattern = re.compile("|".join(f"(?:{p})" for p in cls.JOINT_NAME_PATTERNS))
        indices = [i for i, name in enumerate(all_names) if pattern.fullmatch(name)]
        names = [all_names[i] for i in indices]
        return indices, names

    WIDTH_PRINT_PERIOD_S = 0.5  # throttle -- printing every step at 30Hz would flood the console

    # Matches BinaryJointPositionActionCfg's open_command_expr/close_command_expr for the
    # finger joints in stack_joint_pos_env_cfg.py (both arms use the same values).
    GRIPPER_OPEN_VAL = 0.044
    GRIPPER_CLOSED_VAL = 0.0

    def __init__(self, robot, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)
        self._indices, self._names = self.resolve_mirror_joint_indices(robot)
        self._robot = robot
        self._seq = 0
        self._history: list[tuple[float, dict]] = []
        self._last_width_print_t = 0.0
        print(f"[MIRROR] Broadcasting {len(self._names)} joints to {host}:{port} -> {self._names}")

    def broadcast(self, left_gripper_state: float | None = None, right_gripper_state: float | None = None):
        """Broadcast the robot's mirrored joint state.

        Arm joints use the actual/measured position (`robot.data.joint_pos`) -- that's the real
        kinematic pose and should be mirrored as-is. Finger joints, if a `*_gripper_state` is
        given, are overridden to the fixed commanded open/close target instead of the measured
        position.

        Why: sim's cube and gripper fingers are rigid bodies, so once they contact each other the
        measured finger joint position stalls almost exactly at the object's geometric surface --
        that's correct rigid-body physics, not a bug. The real gripper's fingertip pads are
        compliant and keep squeezing past that same contact point to reach a firm grip (observed:
        sim/real agreed closely at first contact, ~52.6/53.4mm, but the real gripper needed to
        reach ~43.4mm for a grip that actually holds). Mirroring the *measured* sim position was
        capping the real robot's target at sim's rigid stopping point, never letting it command
        the real gripper to keep squeezing further. Mirroring the fixed open/close *target*
        instead lets the real hardware's own compliant pads decide how far they actually close,
        independent of whatever sim's specific rigid object happened to stop the fingers at.

        The `*_gripper_state` argument carries a +-1 binary command in the IK/keyboard modes but
        a position in metres in vr_joint_ros2_native (build_dual_action returns the gripper
        column it logged). The `> 0` test below reads both correctly only because that column is
        quantised to exactly 0.0 / 0.044 -- see snap_gripper_command. Feeding it an unquantised
        trigger position would classify every value but an exact 0.0 as "open", i.e. the real
        gripper would never be told to close.
        """
        joint_pos = self._robot.data.joint_pos[0, self._indices].tolist()
        joints = dict(zip(self._names, joint_pos))

        if left_gripper_state is not None:
            left_val = self.GRIPPER_OPEN_VAL if left_gripper_state > 0 else self.GRIPPER_CLOSED_VAL
            for name in joints:
                if name.startswith("openarm_left_finger_joint"):
                    joints[name] = left_val
        if right_gripper_state is not None:
            right_val = self.GRIPPER_OPEN_VAL if right_gripper_state > 0 else self.GRIPPER_CLOSED_VAL
            for name in joints:
                if name.startswith("openarm_right_finger_joint"):
                    joints[name] = right_val

        packet = {
            "seq": self._seq,
            "t": time.time(),
            "joints": joints,
        }
        self._seq += 1
        self._history.append((packet["t"], joints))
        try:
            self._sock.sendto(json.dumps(packet).encode("utf-8"), self._addr)
        except OSError:
            pass  # best-effort only -- never let a networking hiccup break recording

        if packet["t"] - self._last_width_print_t >= self.WIDTH_PRINT_PERIOD_S:
            self._last_width_print_t = packet["t"]
            # Both finger joints are prismatic, 0.0 (closed) .. 0.044m (open), moving symmetrically
            # outward -- see openarm_description.urdf finger_joint1/2 limits and mimic tag. Total
            # gripper opening width is the sum of both fingers' travel from the closed position.
            left_mm = joints.get("openarm_left_finger_joint1", 0.0) * 2000.0
            right_mm = joints.get("openarm_right_finger_joint1", 0.0) * 2000.0
            print(f"[SIM GRIPPER]  left={left_mm:5.1f}mm  right={right_mm:5.1f}mm")

    def history(self) -> list[tuple[float, dict]]:
        return self._history


class JointFeedbackReceiver:
    """Best-effort UDP listener for ACTUAL joint feedback broadcast back by a
    real-robot bridge (e.g. lerobot_openarm/mirror_bridge.py's --feedback-port).
    Logs the full history (not just the latest packet) for a sim-vs-real comparison
    plot when this script exits. Runs in a background thread; never blocks the sim
    loop. This process still never talks to hardware -- it only listens for numbers
    a separate, out-of-process bridge chooses to send back.
    """

    def __init__(self, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.settimeout(0.5)
        self._lock = threading.Lock()
        self._log: list[tuple[float, dict]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[MIRROR] Listening for real-robot feedback on {host}:{port}")

    def _run(self):
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                packet = json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            with self._lock:
                self._log.append((packet["t"], packet["joints"]))

    def history(self) -> list[tuple[float, dict]]:
        with self._lock:
            return list(self._log)

    def stop(self):
        self._stop.set()
        self._sock.close()



def save_sim_vs_real_plot(
    mirror_broadcaster,
    feedback_receiver,
    *,
    t_start: float | None = None,
    t_end: float | None = None,
    title: str | None = None,
    out_path: str | None = None,
) -> None:
    """Compare the sim joint positions this process broadcast against the real
    robot's actual joint feedback received back from mirror_bridge.py, and save a
    per-joint time-series plot (PNG) to the current working directory.

    t_start/t_end restrict the plot to one wall-clock window of a longer session, so a script
    running several rollouts back to back can save one figure per rollout instead of a single
    figure in which every rollout, and every operator pause between them, is drawn on top of the
    others. Both processes stamp their packets with time.time() on the same machine, so the two
    histories share a clock and one window selects the same interval in both. Defaults keep the
    whole-session behaviour unchanged.
    """
    sim_history = mirror_broadcaster.history() if mirror_broadcaster is not None else []
    real_history = feedback_receiver.history()

    if t_start is not None or t_end is not None:
        lo = -float("inf") if t_start is None else t_start
        hi = float("inf") if t_end is None else t_end
        sim_history = [(t, j) for t, j in sim_history if lo <= t <= hi]
        real_history = [(t, j) for t, j in real_history if lo <= t <= hi]

    if not sim_history or not real_history:
        print("[MIRROR] No data collected for a sim-vs-real comparison plot (one or both histories are"
              " empty) -- skipping. Did mirror_bridge.py have --feedback-port set to match?")
        return

    # Minimum y-axis half-range per joint category -- prevents matplotlib's auto-scaling
    # from zooming into sensor noise or a small steady-state offset and making it look
    # like a large sim-vs-real gap. Chosen relative to each category's own meaningful
    # scale, not one-size-fits-all: 0.1 rad matches the arm joints' own handshake
    # tolerance elsewhere in this pipeline (the threshold already established as "worth
    # taking seriously"), while the gripper's entire physical range is only ~0.044 rad,
    # so it gets a proportionally smaller floor rather than being flattened by the arm's.
    ARM_AXIS_TOLERANCE = 0.1
    GRIPPER_AXIS_TOLERANCE = 0.005

    t0 = sim_history[0][0]
    joint_names = list(sim_history[0][1].keys())
    ncols = 4
    nrows = (len(joint_names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, name in enumerate(joint_names):
        ax = axes[idx // ncols][idx % ncols]
        sim_t = [t - t0 for t, j in sim_history if name in j]
        sim_v = [j[name] for t, j in sim_history if name in j]
        real_t = [t - t0 for t, j in real_history if name in j]
        real_v = [j[name] for t, j in real_history if name in j]
        ax.plot(sim_t, sim_v, label="sim", linewidth=1)
        ax.plot(real_t, real_v, label="real", linewidth=1, linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("s", fontsize=7)
        ax.set_ylabel("rad", fontsize=7)

        all_v = sim_v + real_v
        if all_v:
            tolerance = GRIPPER_AXIS_TOLERANCE if "finger" in name else ARM_AXIS_TOLERANCE
            data_min, data_max = min(all_v), max(all_v)
            center = (data_min + data_max) / 2
            half_range = max((data_max - data_min) / 2 * 1.1, tolerance)
            ax.set_ylim(center - half_range, center + half_range)

    for idx in range(len(joint_names), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(title or "Sim vs real joint positions (live mirroring session)")
    fig.tight_layout()
    if out_path is None:
        out_path = os.path.join(os.getcwd(), f"sim_vs_real_{time.strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(out_path, dpi=120)
    # Released explicitly: a caller saving one figure per rollout would otherwise accumulate them
    # for the whole session (pyplot keeps every figure alive until closed) and trip matplotlib's
    # own "More than 20 figures have been opened" warning partway through a long run.
    plt.close(fig)

    worst = _worst_gap(sim_history, real_history)
    if worst is not None:
        name, gap = worst
        print(f"[MIRROR] Saved sim-vs-real comparison plot to {out_path}"
              f"  (largest median gap: {name} at {gap:.4f} rad)")
    else:
        print(f"[MIRROR] Saved sim-vs-real comparison plot to {out_path}")


def _worst_gap(sim_history, real_history):
    """The joint whose real position sits furthest from sim's, as a median over the window.

    Median rather than max: the real arm is always a little behind sim during fast motion, and a
    peak-error figure mostly reports how fast the arm was asked to move at one instant. The median
    offset is the part that says the two are actually tracking differently. Returns (name, rad),
    or None if the two histories share no joint.
    """
    names = set(sim_history[0][1]) & set(real_history[0][1])
    worst = None
    for name in sorted(names):
        sim_pts = [(t, j[name]) for t, j in sim_history if name in j]
        real_pts = [(t, j[name]) for t, j in real_history if name in j]
        if not sim_pts or not real_pts:
            continue
        # Nearest-in-time sim sample for each real sample: the two streams are logged by different
        # processes at different rates, so they are not index-aligned.
        gaps = []
        i = 0
        for t, rv in real_pts:
            while i + 1 < len(sim_pts) and abs(sim_pts[i + 1][0] - t) <= abs(sim_pts[i][0] - t):
                i += 1
            gaps.append(abs(rv - sim_pts[i][1]))
        if not gaps:
            continue
        gaps.sort()
        median = gaps[len(gaps) // 2]
        if worst is None or median > worst[1]:
            worst = (name, median)
    return worst
