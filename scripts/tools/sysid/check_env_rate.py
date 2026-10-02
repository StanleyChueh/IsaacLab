# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Print the control rate the OpenArm pick-up env resolves to, the step counts derived from it, and
whether a source HDF5 matches it. Read-only; no robot, no recording.

  ./isaaclab.sh -p scripts/tools/sysid/check_env_rate.py --headless [--hdf5 logs/demos/foo.hdf5] [--steps 30]
  OPENARM_CONTROL_HZ=20 ./isaaclab.sh -p scripts/tools/sysid/check_env_rate.py --headless   # legacy
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--hdf5", type=str, default=None, help="Source HDF5 to compare against the env's rate.")
parser.add_argument("--steps", type=int, default=0, help="Also step the env this many times and time it.")
parser.add_argument("--report", type=str, default="check_env_rate.txt",
                    help="File the report is also written to (Kit can swallow stdout in headless runs).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
app = AppLauncher(args).app

import time  # noqa: E402
import traceback  # noqa: E402

_report = open(args.report, 'w')


def say(msg=''):
    print(msg, flush=True)
    _report.write(msg + '\n')
    _report.flush()


try:
    import torch  # noqa: E402

    from isaaclab_tasks.manager_based.manipulation.stack.config.openarm import openarm_sim_timing as T  # noqa: E402
    from isaaclab_tasks.manager_based.manipulation.stack.config.openarm import openarm_task_modes as TM  # noqa: E402

    import _sim_common  # noqa: E402

    env, cfg = _sim_common.build_env(num_envs=1, device=args.device)
    say(f"\nOPENARM_CONTROL_HZ        = {T.CONTROL_HZ}  (step scale x{T.STEP_SCALE:.2f} vs the {T.REFERENCE_HZ:.0f} Hz reference)")
    say(f"sim.dt                    = {cfg.sim.dt:.6f} s  ({1 / cfg.sim.dt:.1f} Hz physics)")
    say(f"decimation                = {cfg.decimation}")
    say(f"render_interval           = {cfg.sim.render_interval}")
    say(f"env.step_dt               = {env.step_dt:.6f} s  -> {1 / env.step_dt:.2f} Hz")
    say(f"HANDOVER_RECEIVER_HOLD    = {TM.HANDOVER_RECEIVER_HOLD_STEPS} steps ({TM.HANDOVER_RECEIVER_HOLD_STEPS * env.step_dt:.2f} s)")
    say(f"scale_steps(15)/(5)/odd(5)= {T.scale_steps(15)} / {T.scale_steps(5)} / {T.scale_odd(5)}")
    robot = env.scene["robot"]
    for jn in ("openarm_left_joint2", "openarm_right_joint5", "openarm_left_finger_joint1"):
        jid = robot.find_joints(jn)[0][0]
        say(f"actuator {jn:28s} kp {robot.data.joint_stiffness[0, jid].item():8.2f}  kd {robot.data.joint_damping[0, jid].item():7.3f}"
            f"  armature {robot.data.joint_armature[0, jid].item():.4f}")
    if args.hdf5:
        say()
        src = T.source_rate_hz(args.hdf5)
        say(f"source {args.hdf5}: {'unknown' if src is None else f'{src:.1f} Hz'}")
        try:
            T.check_source_rate(args.hdf5, cfg)
            say("rate guard: OK (source and env match)")
        except SystemExit as e:
            say(f"rate guard: REFUSES -> {e}")
    if args.steps:
        env.reset()
        robot = env.scene["robot"]
        zero = robot.data.joint_pos.clone()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            robot.set_joint_position_target(zero)
            env.sim.step(render=False)  # physics only, as the identification tools do
        say(f"\n{args.steps} physics steps in {time.perf_counter() - t0:.2f} s")
    env.close()

except Exception:
    say(traceback.format_exc())
finally:
    _report.close()
    app.close()
