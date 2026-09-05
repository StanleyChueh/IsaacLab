#!/usr/bin/env python3
"""
GR00T (N1.7) policy server -- runs in the 'lerobot-latest' conda env.

Sibling of smolvla_server.py: loads a trained GR00T N1.7 LeRobot checkpoint and exposes it over
the SAME TCP protocol, so eval_groot_jointspace.py (running in a different Python env, isaaclab)
can query it exactly like it queries smolvla_server.py. eval_groot_jointspace.py itself has no
GR00T-specific code at all -- it only speaks this wire protocol -- so everything GR00T-specific
lives here.

Why this is NOT a copy of smolvla_server.py's PolicyServer.step() despite the identical protocol:
GR00T's own LeRobot wrapper (lerobot.policies.groot.modeling_groot.GrootPolicy) does its
normalization and (for this checkpoint) RELATIVE-action decoding entirely inside its own
preprocessor/postprocessor pipeline (lerobot.policies.groot.processor_groot), not via the plain
mean/std buffers SmolVLA uses -- see GrootConfig.normalization_mapping, which is IDENTITY for
every feature specifically because GR00T does not use LeRobot's generic Normalizer step. So this
server calls that pipeline (preprocessor -> policy.predict_action_chunk -> postprocessor) instead
of hand-rolling normalization the way smolvla_server.py does.

More importantly, THIS CHECKPOINT was trained with use_relative_actions=True (see its
config.json): each action chunk the model predicts is a set of joint-position DELTAS relative to
the observation state at the moment the chunk was requested (the gripper columns are the
exception -- excluded from the delta and always absolute, see relative_exclude_joints). Because of
that, GrootPolicy.select_action() (the call smolvla_server.py's equivalent code uses) deliberately
raises NotImplementedError for this checkpoint:

    "GrootPolicy.select_action does not support relative-action policies because cached relative
    chunk actions can be decoded against newer observation states. Use predict_action_chunk and
    postprocess the full chunk before queuing actions ..."

-- i.e. select_action's internal queue pops ONE normalized action at a time and only unnormalizes
it lazily on each pop, which is fine for an absolute-action policy (SmolVLA) but silently wrong for
a relative one: by the time a later element of the chunk is popped, the observation.state that its
delta must be added to is gone. This server follows GrootPolicy's own instruction instead: run
predict_action_chunk() once per fresh observation to get the WHOLE (1, n_action_steps, action_dim)
chunk still in normalized/relative model space, run the WHOLE chunk through the postprocessor in
one call (this is what actually adds each timestep's delta back onto the state that produced the
chunk -- see GrootN17ActionDecodeStep, which reads that state off a cache the preprocessor filled
one call earlier and raises loudly if asked to decode one timestep at a time), and only THEN queue
the now-absolute per-step actions for popping. A fresh chunk is requested once the queue empties
(every --checkpoint's config.n_action_steps steps -- 16 for this checkpoint), executed open-loop
in between, exactly like GrootPolicy.select_action's own cadence, just with decoding correctly
ordered around it.

Usage (in a separate terminal FIRST -- note the different conda env from smolvla_server.py; GR00T's
Qwen3-VL backbone needs transformers>=5, which conflicts with the transformers<5 pin the `lerobot`
env uses for SmolVLA/ACT/etc):
  conda activate lerobot-latest && cd ~/Stanley_ws/IsaacLab
  python scripts/imitation_learning/lerobot/gr00t_server.py \\
      --checkpoint ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz_gr00t \\
      --task "Pick up the Pringles can with the right arm, hand it to the left arm." \\
      --port 5556

Protocol (identical to smolvla_server.py -- TCP, framed by 4-byte big-endian length prefix +
pickle payload):
  Client -> Server:
    {"cmd": "reset"}                   -> clear the action chunk queue
    {"cmd": "step",
     "state": ndarray(state_dim,),
     "cameras": {"<env_cam_name>": uint8 HWC, ...}}   -- see --cameras: keys must be the ACTUAL
     camera names this checkpoint's dataset used (right_wrist_cam / wrist_cam / body_cam by
     default), NOT the camera1/camera2/... slot names smolvla_server.py's protocol uses -- this
     checkpoint's input_features were never renamed to slot names at train time (see
     config.json's input_features on the HF repo), unlike the SmolVLA checkpoints this repo's
     other scripts target.
    {"cmd": "close"}                   -> end this client session
  Server -> Client:
    {"ok": True}                       -> after reset / close
    {"action": ndarray(action_dim,)}   -> after step
    {"error": str}                     -> a step raised; the session and server both survive

The server keeps listening after a client disconnects (or sends "close"), so eval runs can be
repeated back-to-back without restarting it and re-loading the checkpoint. Stop it with Ctrl-C.
"""

import argparse
import os
import pickle
import socket
import struct
import sys

import numpy as np
import torch

# A lerobot checkout living OUTSIDE this repo, so there is nothing repo-relative to resolve
# against. Only needed when lerobot is not already importable; set LEROBOT_SRC to point at your
# own checkout's src/ if it is somewhere else. Unlike smolvla_server.py's default (a separate
# checkout pinned to transformers<5), this one must be a checkout new enough to carry the GR00T
# N1.7 policy (Qwen3-VL backbone, transformers>=5) -- the `lerobot-latest` conda env already has
# such a checkout installed editable, so this is a no-op there.
LEROBOT_SRC = os.environ.get("LEROBOT_SRC", "/home/csl/Stanley_ws/lerobot_experiment/lerobot/src")
if LEROBOT_SRC and LEROBOT_SRC not in sys.path and os.path.isdir(LEROBOT_SRC):
    sys.path.insert(0, LEROBOT_SRC)

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


# ── TCP framing helpers (identical to smolvla_server.py) ───────────────────────

def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return b""
        buf.extend(chunk)
    return bytes(buf)

def send_msg(sock: socket.socket, data: bytes):
    sock.sendall(struct.pack(">I", len(data)) + data)

def recv_msg(sock: socket.socket) -> bytes:
    raw_len = _recv_exactly(sock, 4)
    if not raw_len:
        return b""
    (n,) = struct.unpack(">I", raw_len)
    return _recv_exactly(sock, n)


# ── policy wrapper ─────────────────────────────────────────────────────────────

class PolicyServer:
    def __init__(self, checkpoint: str, task: str, device: str, dtype: str = "bfloat16"):
        print(f"[server] Loading  {checkpoint}")
        self.device = device
        self.task = task

        # Load with config.device forced to "cpu" -- NOT a redundant step before the cast below.
        # PreTrainedPolicy.from_pretrained() (the base class GrootPolicy.from_pretrained() defers
        # to for a fine-tuned checkpoint like this one) calls policy.to(config.device) itself
        # before returning, and this checkpoint's saved config.json says device="cuda". Left
        # alone, the FULL fp32 model lands on the GPU before our code ever runs, and casting it to
        # bf16 there needs the fp32 and bf16 copies resident AT THE SAME TIME -- confirmed by
        # measurement: the previous version of this method (dtype cast done after .from_pretrained()
        # with no device override) left gr00t_server.py holding 13.6GB of GPU memory even after
        # settling, near the full fp32 footprint, not the ~6GB bf16 should need, because CUDA's
        # caching allocator keeps the freed fp32 blocks reserved for reuse within the process
        # rather than returning them to the driver. Forcing "cpu" here keeps the fp32 load (and the
        # cast below) entirely in system RAM, which has room to spare, so the GPU only ever sees
        # the already-bf16-sized model in the .to(device) call further down.
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.device = "cpu"

        # strict=False: GR00T's Qwen backbone ties its (unused) LM head weight to the input
        # embedding (see groot_n1_7._tie_unused_qwen_lm_head, "the unused LM head stays frozen
        # and is omitted on save"). Loading strict finds embed_tokens.weight in the checkpoint's
        # safetensors file but not as its own entry in the freshly constructed (already-tied)
        # model's state_dict, and raises "Unexpected key(s)" -- the same tensor loads correctly
        # under lm_head.weight either way, so this is a harmless duplicate, not a missing weight.
        self.policy = GrootPolicy.from_pretrained(checkpoint, config=config, strict=False)

        # GrootConfig.model_params_fp32=True (this checkpoint's own training config) keeps
        # parameters in fp32 so the *training* recipe's bf16-autocast forward pass has an fp32
        # master copy for the optimizer step -- irrelevant here, there is no optimizer. Left at
        # fp32, the 3B-parameter backbone alone is ~12GB of static weight memory, which does not
        # leave room for activations on a 16GB GPU and OOMs on the very first predict_action_chunk
        # (confirmed: "Tried to allocate 18.00 MiB ... 14.74 GiB memory in use" on a 15.46 GiB
        # card). predict_action_chunk already runs its forward pass under
        # torch.autocast(dtype=torch.bfloat16) whenever config.use_bf16 (True for this checkpoint)
        # regardless of the stored parameter dtype -- autocast only affects the dtype ops compute
        # in, not what's resident in memory -- so storing the weights in bf16 to begin with does
        # not change what the forward pass numerically computes, it just removes the redundant
        # fp32 copy of every weight autocast was going to cast down anyway. Done here, still on
        # CPU, before the one and only move to the GPU below.
        torch_dtype = getattr(torch, dtype)
        if torch_dtype is not torch.float32:
            self.policy.to(dtype=torch_dtype)

        # Now pin config.device to where this process actually runs (used by the processor
        # pipeline built below) and make the single CPU -> GPU transfer, already at the small size.
        self.policy.config.device = device
        self.policy.to(device)
        self.policy.eval()

        # GR00T's preprocessor/postprocessor pipeline does everything SmolVLA's server does by hand
        # (state normalize, image prep, language conditioning, action unnormalize) PLUS the
        # relative-action decode described in the module docstring. Built the same way
        # lerobot_eval.py builds it for every policy type, not something GR00T-specific we invented
        # here -- see lerobot.policies.factory.make_pre_post_processors.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": device},
                # No rename: this checkpoint's own input_features already use the raw env camera
                # names (see module docstring), so nothing needs mapping to a slot name here.
                "rename_observations_processor": {"rename_map": {}},
            },
        )

        # Chunk queue: predict_action_chunk() is called once per fresh chunk, the postprocessor
        # decodes the WHOLE chunk in one call (see module docstring for why that ordering is
        # mandatory for this checkpoint's relative actions), and the resulting per-timestep
        # absolute actions are popped one at a time until the queue empties again.
        self._queue: list[np.ndarray] = []

        state_dim = self.policy.config.input_features[OBS_STATE].shape[0]
        action_dim = self.policy.config.output_features[ACTION].shape[0]
        print(
            f"[server] Ready — state_dim={state_dim}  action_dim={action_dim}"
            f"  chunk={self.policy.config.n_action_steps}"
            f"  relative_actions={self.policy.config.use_relative_actions}"
        )
        print(f"[server] Task: {task!r}")

    def reset(self):
        self.policy.reset()
        self._queue = []

    @torch.no_grad()
    def step(self, state: np.ndarray, cameras: dict) -> np.ndarray:
        if not self._queue:
            batch = {
                OBS_STATE: torch.from_numpy(state).float(),
                "task": self.task,
            }
            for cam_key, img_hwc in cameras.items():
                # HWC uint8 -> CHW uint8, no batch dim (the preprocessor's own
                # AddBatchDimensionProcessorStep adds it) -- GR00T's Qwen3-VL image processor does
                # its own resize/normalize, unlike SmolVLA's manual /255.0 in smolvla_server.py.
                key = f"{OBS_IMAGES}.{cam_key}"
                batch[key] = torch.from_numpy(img_hwc).permute(2, 0, 1).contiguous()

            batch = self.preprocessor(batch)
            # Still normalized / relative-to-current-state model output, shape
            # (1, n_action_steps, action_dim) -- NOT queued yet, see module docstring.
            action_chunk = self.policy.predict_action_chunk(batch)
            # Decodes the whole chunk in one call while the preprocessor's cached raw state (the
            # state each timestep's delta must be added back onto) still matches this observation.
            action_chunk = self.postprocessor(action_chunk)
            self._queue = list(action_chunk.squeeze(0).cpu().numpy().astype(np.float32))

        return self._queue.pop(0)


# ── server loop (identical to smolvla_server.py) ────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="HF repo id or local path to a fine-tuned GR00T N1.7 LeRobot checkpoint"
                         " (contains config.json + model.safetensors + policy_preprocessor.json +"
                         " policy_postprocessor.json)")
    ap.add_argument("--task", required=True,
                    help="Language instruction sent to GR00T at every chunk prediction")
    ap.add_argument("--port",   type=int, default=5556)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"],
        help=(
            "Parameter dtype to cast the loaded policy to before moving it onto --device. The"
            " checkpoint's own training config keeps fp32 master weights (~12GB for this 3B"
            " backbone), which does not leave headroom for inference activations on a 16GB GPU --"
            " see PolicyServer.__init__ for why bf16 (the default) does not change what the"
            " forward pass computes. Use float32 only if you have a GPU with enough spare memory"
            " and want to rule out a precision-related difference."
        ),
    )
    args = ap.parse_args()

    server = PolicyServer(args.checkpoint, args.task, args.device, args.dtype)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    print(f"[server] Listening on 127.0.0.1:{args.port} — waiting for Isaac Sim client …")
    print("[server] Stays up across client disconnects; Ctrl-C to shut down.")

    try:
        while True:
            conn, addr = srv.accept()
            print(f"[server] Client connected from {addr}")
            server.reset()
            try:
                with conn:
                    while True:
                        raw = recv_msg(conn)
                        if not raw:
                            print("[server] Client disconnected.")
                            break
                        msg = pickle.loads(raw)
                        cmd = msg.get("cmd", "step")

                        if cmd == "close":
                            send_msg(conn, pickle.dumps({"ok": True}))
                            print("[server] Client closed the session.")
                            break
                        elif cmd == "reset":
                            server.reset()
                            send_msg(conn, pickle.dumps({"ok": True}))
                        elif cmd == "step":
                            try:
                                action = server.step(msg["state"], msg["cameras"])
                            except Exception as exc:
                                print(f"[server] step failed: {exc!r}")
                                send_msg(conn, pickle.dumps({"error": f"{type(exc).__name__}: {exc}"}))
                                continue
                            send_msg(conn, pickle.dumps({"action": action.tolist()}))
                        else:
                            send_msg(conn, pickle.dumps({"error": f"unknown cmd {cmd!r}"}))
            except OSError as exc:
                print(f"[server] Connection lost: {exc!r}")
            print(f"[server] Listening again on 127.0.0.1:{args.port} — Ctrl-C to shut down.")
    except KeyboardInterrupt:
        print("\n[server] Interrupted.")
    finally:
        srv.close()
        print("[server] Shut down.")


if __name__ == "__main__":
    main()
