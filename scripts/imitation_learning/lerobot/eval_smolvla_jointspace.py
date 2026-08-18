"""
Evaluate a JOINT-SPACE-trained SmolVLA policy inside Isaac Sim.

Sibling of eval_smolvla.py, for checkpoints trained on joint-space actions/observations
(LJ1.pos..LJ8.pos + RJ1.pos..RJ8.pos, see convert_hdf5_to_lerobot.py). Unlike eval_smolvla.py,
this does NOT go through the env's IK-based action term at all: the model's output IS the
target joint configuration directly, so this
bypasses the action manager entirely and drives the robot's joints directly every physics
substep, replicating everything else ManagerBasedRLEnv.step() normally does (scene.write_data_
to_sim(), sim.step(), sim.render() at the configured interval, scene.update() for fresh sensor/
camera data) -- see _step_direct() below, confirmed line-by-line against
source/isaaclab/isaaclab/envs/manager_based_rl_env.py's own step() (lines ~153-197), not guessed.
The two calls that ARE skipped are action_manager.process_action() and .apply_action() -- the IK
solve and its resulting set_joint_position_target() call, replaced by our own direct one.

Run smolvla_server.py FIRST (the same server eval_smolvla.py uses -- it's checkpoint-agnostic,
does plain state-normalize / policy.select_action / action-unnormalize with no assumption about
what the action represents). It stays up across client disconnects, so one server serves any
number of eval runs; only Ctrl-C stops it. Then this:

  Terminal 1 (lerobot env) -- --task must be the dataset's task string VERBATIM (read it from
  the dataset's own meta/tasks.parquet; SmolVLA is language-conditioned, an unseen prompt such
  as "." silently degrades the policy):
    cd ~/CSL/lerobot
    conda run -n lerobot python \\
        ~/Stanley_ws/IsaacLab/scripts/imitation_learning/lerobot/smolvla_server.py \\
        --checkpoint ethanCSL/openarm_visuomotor_VR_pringles_V8_generated_500 \\
        --task "Pick up the Pringles can with the right arm, hand it to the left arm"

  Terminal 2 (Isaac Sim) -- --task is the GYM ENV and --task_mode the variant applied on top
  of it; BOTH must match what the demos were recorded with, because the env id alone does not
  determine the scene (see below):
    cd ~/Stanley_ws/IsaacLab
    ./isaaclab.sh -p scripts/imitation_learning/lerobot/eval_smolvla_jointspace.py \\
        --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \\
        --task_mode handover \\
        --num_rollouts 5 \\
        --horizon 300 \\
        --enable_cameras \\
        --cameras right_wrist_cam,wrist_cam,body_cam

--task_mode: on the CAN_TARGET_TASKS the manipulated object in the cfg is the base task's red
cube_2, and apply_task_mode() is the only thing that replaces it with the can -- along with the
per-mode object spawn range and success condition. Recording goes through
record_demos_openarm.py --task_mode, so a dataset's env id does NOT by itself describe the scene
it was recorded in; evaluating the same env id without the mode silently puts the policy in
front of a cube. This script refuses that combination rather than running it, the same way the
recording script does. Which mode a dataset used is readable from its own annotated HDF5:
data/demo_0/obs/datagen_info/subtask_term_signals -- grasp_right/handover/presented is
``handover``, grasp/lift is ``left`` or ``right`` (see openarm_task_modes.py's table).

State (16D by default): LJ1..LJ8 then RJ1..RJ8, i.e. openarm_left_joint1..7 + left gripper,
then openarm_right_joint1..7 + right gripper, extracted via BOTH_IDX below. The physics
Articulation's joint order is INTERLEAVED dual-arm (even indices 0,2,...,12 = left arm 1-7, odd
1,3,...,13 = right arm 1-7, then 14 / 16 = the left / right gripper's actuated finger),
confirmed against a live Isaac Sim robot.data.joint_names printout and matching
convert_hdf5_to_lerobot.py's own LJ_IDX / RJ_IDX / BOTH_IDX exactly. This deliberately does NOT
reuse eval_smolvla.py's _get_state() (joint_pos[:7]) -- a plain leading slice mixes left- and
right-arm values under this interleaved ordering and would feed the model a wrong state.

Action (16D by default): applied DIRECTLY as target joint positions for those same joints -- no
IK, no delta interpretation, no gripper open/close threshold. Each gripper's finger_joint2 is a
native PhysX mimic of its finger_joint1 in this USD (confirmed: no PhysicsDriveAPI on joint2),
so only the one actuated joint per hand needs a target -- the mimic partner follows.

The layout MUST match the --arms convert_hdf5_to_lerobot.py exported this checkpoint's dataset
with. Pass --arms left for an older 8D left-arm-only checkpoint; then only LJ_IDX is driven and
the right arm holds wherever env.reset() put it. A mismatch raises in _step_direct on the first
step rather than being silently truncated.

--cameras: comma-separated env camera names in policy slot order, same contract as
eval_smolvla.py's own --cameras -- MUST match this exact checkpoint's --rename_map at train
time. Verify before trusting a rollout; a mismatch is silent (no error), it just degrades to
generic/wrong behavior.
"""

"""Launch Isaac Sim first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate a joint-space-action SmolVLA policy in Isaac Lab via smolvla_server.py"
)
parser.add_argument("--task",        type=str, required=True, help="Gym env id")
parser.add_argument("--num_rollouts",type=int, default=5,   help="Number of evaluation episodes")
parser.add_argument("--horizon",     type=int, default=300,  help="Max steps per episode")
parser.add_argument("--port",        type=int, default=5556, help="Policy server TCP port")
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument(
    "--task_mode", type=str, default=None, choices=["left", "right", "handover"],
    help=(
        "Task-mode variant to apply to the env cfg -- MUST be the same one the demos were"
        " recorded and annotated with (record_demos_openarm.py --task_mode). The can-target"
        " tasks' own cfgs still describe the base task's red CUBE; apply_task_mode() is what"
        " swaps in the can, and also sets the per-mode object spawn range and success"
        " condition. Omitting it on such a task evaluates the policy against the wrong object"
        " entirely, so it is refused rather than run (see CAN_TARGET_TASKS)."
    ),
)
parser.add_argument(
    "--arms", type=str, default="both", choices=["both", "left"],
    help=(
        "Which joint layout the checkpoint was trained on -- MUST match the --arms used for "
        "convert_hdf5_to_lerobot.py on this checkpoint's dataset. 'both' (default) = 16D "
        "LJ1..LJ8+RJ1..RJ8; 'left' = 8D LJ1..LJ8 only, for older left-arm-only checkpoints. "
        "A mismatch is caught at the first step (see _step_direct), not silently ignored."
    ),
)
parser.add_argument(
    "--cameras", type=str, default="right_wrist_cam,wrist_cam,body_cam",
    help=(
        "Comma-separated env camera names, IN POLICY SLOT ORDER (1st -> camera1, 2nd -> camera2, "
        "...). The default is the OpenArm pick-up task's three cameras in the order"
        " convert_hdf5_to_lerobot.py is documented to convert them (see README). front_cam used"
        " to lead this list but the task no longer builds that sensor at all. "
        "MUST match the --rename_map used for lerobot-train on this exact checkpoint. "
        "Sending the wrong env camera into a policy slot is silent (no error, no crash)."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything else after Isaac Sim is up."""

import pickle
import random
import socket
import struct

import carb
import carb.input
import omni.appwindow
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.stack  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.openarm_task_modes import (
    CAN_TARGET_TASKS,
    apply_task_mode,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# ── TCP client helpers (identical to eval_smolvla.py, must match server) ───────

def _recv_exactly(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Policy server closed connection")
        buf.extend(chunk)
    return bytes(buf)

def _send(sock, payload: bytes):
    sock.sendall(struct.pack(">I", len(payload)) + payload)

def _recv(sock) -> dict:
    raw_len = _recv_exactly(sock, 4)
    (n,) = struct.unpack(">I", raw_len)
    return pickle.loads(_recv_exactly(sock, n))

def policy_reset(sock):
    _send(sock, pickle.dumps({"cmd": "reset"}))
    _recv(sock)

def policy_step(sock, state: np.ndarray, cameras: dict) -> np.ndarray:
    _send(sock, pickle.dumps({"cmd": "step", "state": state, "cameras": cameras}))
    resp = _recv(sock)
    if "error" in resp:
        raise RuntimeError(f"Policy server error: {resp['error']}")
    return np.asarray(resp["action"], dtype=np.float32)

def policy_close(sock):
    _send(sock, pickle.dumps({"cmd": "close"}))
    _recv(sock)


# ── joint-space state/action ────────────────────────────────────────────────────

# Matches convert_hdf5_to_lerobot.py's LJ_IDX / RJ_IDX / BOTH_IDX exactly. The physics
# Articulation's joint order is INTERLEAVED dual-arm: even indices 0,2,...,12 = left arm 1-7,
# odd indices 1,3,...,13 = right arm 1-7, then 14 = left gripper's actuated finger and 16 = the
# right one (15 / 17 are their PhysX-mimic partners, which follow automatically).
LJ_IDX = list(range(0, 14, 2)) + [14]
RJ_IDX = list(range(1, 14, 2)) + [16]
# Concatenation order is LJ1..LJ8 then RJ1..RJ8 -- the exact column order of the dataset's
# observation.state / action, and of the real robot's action dict.
BOTH_IDX = LJ_IDX + RJ_IDX

# The joints this run actually drives, and the exact column order of the state it sends.
DRIVEN_IDX = BOTH_IDX if args_cli.arms == "both" else LJ_IDX


def _get_state(env) -> np.ndarray:
    """Return the joint-space state in DRIVEN_IDX order (16D LJ1..LJ8+RJ1..RJ8 by default,
    8D LJ-only under --arms left), matching the exact index order the training dataset used --
    see module docstring for why this is NOT a plain joint_pos[:n] slice."""
    joint_pos = env.scene["robot"].data.joint_pos[0]   # (num_joints,)
    return joint_pos[DRIVEN_IDX].cpu().numpy().astype(np.float32)


def _get_cameras(obs_dict: dict, camera_names: list[str]) -> dict:
    """Extract camera images as uint8 HWC numpy arrays (identical to eval_smolvla.py)."""
    policy_obs = obs_dict["policy"]
    cameras = {}
    for i, env_key in enumerate(camera_names):
        policy_key = f"camera{i + 1}"
        if env_key in policy_obs:
            img = policy_obs[env_key]        # (1, H, W, 3) uint8
            cameras[policy_key] = img.squeeze(0).cpu().numpy().astype(np.uint8)
        else:
            print(f"[warn] Camera '{env_key}' (-> {policy_key}) not found in policy observations")
    return cameras


def _step_direct(env, target_q: np.ndarray) -> dict:
    """Advance the sim by one env "step" worth of physics (env.cfg.decimation substeps), driving
    the DRIVEN_IDX joints directly instead of going through the env's IK action term -- see
    module docstring for the exact ManagerBasedRLEnv.step() sequence this replicates. Returns a
    fresh observation dict in the same {"policy": {...}} shape env.step()/env.reset() return.
    """
    if target_q.shape[0] != len(DRIVEN_IDX):
        raise ValueError(
            f"Policy returned a {target_q.shape[0]}D action but --arms {args_cli.arms} drives"
            f" {len(DRIVEN_IDX)} joints. The checkpoint's action dim must match the layout"
            " convert_hdf5_to_lerobot.py exported its dataset with: 16 = --arms both"
            " (LJ1..LJ8+RJ1..RJ8), 8 = --arms left. Re-run with the matching --arms."
        )
    robot = env.scene["robot"]
    target_t = torch.as_tensor(target_q, dtype=torch.float32, device=env.device).unsqueeze(0)  # (1, n)

    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        robot.set_joint_position_target(target_t, joint_ids=DRIVEN_IDX)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if (env._sim_step_counter % env.cfg.sim.render_interval == 0) and is_rendering:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)

    return env.observation_manager.compute()


# ── keyboard reset (identical to eval_smolvla.py) ───────────────────────────────

class KeyboardHandler:
    """Listens for R = manual reset, Q = quit."""

    def __init__(self):
        self.reset_requested = False
        self.quit_requested  = False
        app_window = omni.appwindow.get_default_app_window()
        self._keyboard = app_window.get_keyboard()
        self._input    = carb.input.acquire_input_interface()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_key
        )
        print("[keyboard] Press  R  to reset episode early,  Q  to quit.")

    def _on_key(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.R:
                self.reset_requested = True
                print("[keyboard] Manual reset requested.")
            elif event.input == carb.input.KeyboardInput.Q:
                self.quit_requested = True
                print("[keyboard] Quit requested.")
        return True

    def consume_reset(self) -> bool:
        if self.reset_requested:
            self.reset_requested = False
            return True
        return False


# ── rollout ────────────────────────────────────────────────────────────────────

def rollout(env, sock, success_term, horizon: int, kb: "KeyboardHandler", camera_names: list[str]) -> bool:
    obs_dict, _ = env.reset()
    policy_reset(sock)

    for step in range(horizon):
        if kb.consume_reset():
            print(f"  [step {step+1:3d}] manually reset")
            return False

        state   = _get_state(env)
        cameras = _get_cameras(obs_dict, camera_names)

        if not cameras:
            print("[warn] No camera images found in obs — check image_obs_list config")
            break

        policy_action = policy_step(sock, state, cameras)   # (n,) absolute joint targets

        obs_dict = _step_direct(env, policy_action)

        if bool(success_term.func(env, **success_term.params)[0]):
            print(f"  [step {step+1:3d}] SUCCESS")
            return True

    print(f"  [step {step+1:3d}] failed")
    return False


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )

    # Same refusal record_demos_openarm.py makes, for the same reason: on these tasks the object
    # is a red cube until apply_task_mode() replaces it with the can, and a rollout against the
    # wrong object looks completely normal apart from failing every episode.
    bare_task = args_cli.task.split(":")[-1]
    if args_cli.task_mode is None and bare_task in CAN_TARGET_TASKS:
        raise SystemExit(
            f"'{args_cli.task}' targets the can, but that swap only happens via --task_mode,"
            " which was not passed -- this run would have evaluated against the base task's"
            " CUBE. Re-run with the SAME --task_mode the demos were recorded with"
            " (left|right|handover)."
        )

    # Task mode first: it rewrites terminations.success, so it has to land before the
    # success_term is lifted off the cfg below.
    if args_cli.task_mode is not None:
        apply_task_mode(env_cfg, args_cli.task_mode)

    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None

    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    print(f"\nEnv obs keys   : {list(env.observation_space.spaces.keys())}")
    print("Action mode    : DIRECT joint-space (bypassing the env's own IK action term)")
    print(f"Joint layout   : --arms {args_cli.arms} -> {len(DRIVEN_IDX)}D state/action")
    print(f"Task mode      : {args_cli.task_mode or 'none (base task cfg as-is)'}")

    print(f"\nConnecting to policy server on port {args_cli.port} …")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", args_cli.port))
    print("Connected.\n")

    kb = KeyboardHandler()

    camera_names = [name.strip() for name in args_cli.cameras.split(",") if name.strip()]
    print(f"[INFO] Camera slots (in order -> camera1, camera2, ...): {camera_names}")

    results = []
    for trial in range(args_cli.num_rollouts):
        if kb.quit_requested:
            print("[keyboard] Quitting early.")
            break
        print(f"── Trial {trial + 1}/{args_cli.num_rollouts} ──────────────────")
        success = rollout(env, sock, success_term, args_cli.horizon, kb, camera_names)
        results.append(success)

    n_ok = results.count(True)
    print(f"\n{'='*50}")
    print(f"Success: {n_ok} / {len(results)}")
    print(f"Rate   : {n_ok / len(results):.1%}")
    print(f"Results: {results}")

    policy_close(sock)
    sock.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
