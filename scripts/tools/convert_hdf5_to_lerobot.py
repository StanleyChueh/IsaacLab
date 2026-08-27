#!/usr/bin/env python3
"""
Convert IsaacLab HDF5 demo recordings → LeRobot v3.0 dataset format.

Run with the lerobot conda env:
  conda run -n lerobot python -u scripts/tools/convert_hdf5_to_lerobot.py \\
      --hdf5 logs/demos/pickup.hdf5 \\
      --output ~/datasets/openarm \\
      --task "Pick up the pringles can." \\
      --fps 30 --cameras right_wrist_cam wrist_cam body_cam

--cameras names are matched against what the HDF5 actually contains and any that are missing
are dropped with a warning (see `active_cameras`), so an over-long list is harmless here --
unlike eval_smolvla*.py's --cameras, where the position in the list IS the policy slot. The
OpenArm pick-up task used to record a fourth view, front_cam; it no longer builds that sensor,
so datasets generated after that change have only the three above.

Scalable streaming pipeline — RAM usage is O(one episode), not O(all episodes).
Handles 1000+ episodes without OOM.
"""

import argparse
import json
import queue
import shutil
import sys
import threading
from pathlib import Path

import av
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


# ── output helpers ─────────────────────────────────────────────────────────────

def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)

def _tqdm(*args, **kwargs):
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(*args, **kwargs)


# ── PyAV video writer ──────────────────────────────────────────────────────────

def _make_av_stream(container, fps: int, H: int, W: int):
    """Try h264_nvenc (GPU) first, fall back to libx264 (CPU)."""
    for codec, opts, label in [
        ("h264_nvenc", {"preset": "p4", "rc": "constqp", "qp": "18"}, "h264_nvenc (GPU)"),
        ("libx264",    {"preset": "fast", "crf": "18"},                "libx264 (CPU)"),
    ]:
        try:
            stream = container.add_stream(codec, rate=fps)
            stream.width  = W
            stream.height = H
            stream.pix_fmt = "yuv420p"
            stream.options = opts
            return stream, label
        except Exception:
            continue
    raise RuntimeError("No H.264 encoder found (tried h264_nvenc and libx264)")


class _VideoWriter:
    """PyAV writer that stays open across episodes in the same chunk.

    Frames are written episode-by-episode — no need to buffer an entire
    chunk in RAM before encoding.
    """
    def __init__(self, path: Path, fps: int, H: int, W: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(path), mode="w")
        self._stream, self.label = _make_av_stream(self._container, fps, H, W)

    def write(self, frames_nhwc: np.ndarray):
        """Write (N, H, W, 3) uint8 RGB frames into the open container."""
        for frame_hwc in frames_nhwc:
            av_frame = av.VideoFrame.from_ndarray(frame_hwc, format="rgb24")
            for pkt in self._stream.encode(av_frame):  # type: ignore[union-attr]
                self._container.mux(pkt)

    def close(self):
        for pkt in self._stream.encode():  # type: ignore[union-attr]  flush encoder
            self._container.mux(pkt)
        self._container.close()


# ── online statistics (Welford batch algorithm) ────────────────────────────────

class _OnlineStats:
    """Compute min/max/mean/std/quantiles incrementally.

    Supports both 1-D feature vectors (actions, states) and image tensors.
    Memory: O(quant_samples), independent of dataset size.
    """

    def __init__(self, C: int, is_image: bool = False,
                 quant_target: int = 60_000, sample_frac: float = 1e-3):
        self.C = C
        self.is_image = is_image
        self.n         = 0
        self._mean     = np.zeros(C, np.float64)
        self._M2       = np.zeros(C, np.float64)
        self._min      = np.full(C,  np.inf,  np.float64)
        self._max      = np.full(C, -np.inf, np.float64)
        self._samples: list[np.ndarray] = []
        self._quant_target = quant_target
        self._sample_frac  = sample_frac

    def update(self, data: np.ndarray):
        """data: (T, C) for vectors, (T, H, W, C) uint8 for images."""
        if self.is_image:
            n_frames, H, W, C = data.shape
            # Sample at frame level — avoids allocating a 600 MB float32 array
            # for all frames AND avoids np.random.choice(50M, replace=False) which
            # is O(N_pixels) = ~2.3s per camera.  10 frames → 9 MB, negligible time.
            n_sf = max(10, min(n_frames, -(-self._quant_target // (H * W))))
            fi   = np.random.randint(0, n_frames, size=n_sf)   # O(n_sf), not O(N_pix)
            flat = data[fi].reshape(-1, C).astype(np.float32)
            flat /= 255.0   # in-place: skip the second 200 MB allocation
        else:
            flat = data.astype(np.float64)

        n_new = flat.shape[0]
        if n_new == 0:
            return

        new_mean = flat.mean(axis=0)
        new_var  = flat.var(axis=0, ddof=0)

        # Parallel / batch Welford combine
        n_tot = self.n + n_new
        delta = new_mean - self._mean
        self._mean = (self.n * self._mean + n_new * new_mean) / n_tot
        self._M2   = self._M2 + new_var * n_new + delta**2 * self.n * n_new / n_tot
        self.n = n_tot

        self._min = np.minimum(self._min, flat.min(axis=0))
        self._max = np.maximum(self._max, flat.max(axis=0))

        # Store only a fixed budget per update so _samples stays small at scale.
        # 1000 px/episode × 1000 episodes = 1M px total → trivial to concatenate.
        n_store  = min(flat.shape[0], 1000)
        store_idx = np.random.randint(0, flat.shape[0], size=n_store)
        self._samples.append(flat[store_idx])

    def finalize(self) -> dict:
        std = np.sqrt(np.maximum(0.0, self._M2 / max(1, self.n)))
        samples = np.concatenate(self._samples, axis=0)   # bounded: ≤1k px/ep
        if len(samples) > self._quant_target:
            # With-replacement: O(quant_target), not O(len(samples))
            idx = np.random.randint(0, len(samples), size=self._quant_target)
            samples = samples[idx]
        qs = np.quantile(samples, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)

        if self.is_image:
            def w(v): return [[[float(x)]] for x in v]   # (C,1,1) for image channels
        else:
            def w_vec(v): return [float(x) for x in v]
            w = w_vec  # type: ignore[assignment]

        return {
            "min":   w(self._min),
            "max":   w(self._max),
            "mean":  w(self._mean),
            "std":   w(std),
            "count": [int(self.n)],
            "q01":   w(qs[0]),
            "q10":   w(qs[1]),
            "q50":   w(qs[2]),
            "q90":   w(qs[3]),
            "q99":   w(qs[4]),
        }


# ── HDF5 episode loader ────────────────────────────────────────────────────────

def _scan_episodes(hdf5_files: list) -> list:
    """Return list of (hdf5_path_str, ep_name) without loading any frames."""
    infos = []
    for path in hdf5_files:
        _print(f"Scanning {path} ...")
        with h5py.File(str(path), "r") as f:
            for name in sorted(f["data"].keys()):
                infos.append((str(path), name))
    return infos


# Isaac Sim's physics Articulation for the bimanual OpenArm orders its 18 joint DOFs as:
#    0,2,4,6,8,10,12 = openarm_left_joint1..7      (arms INTERLEAVED, left on even indices)
#    1,3,5,7,9,11,13 = openarm_right_joint1..7
#   14 = openarm_left_finger_joint1    15 = its mimic follower joint2
#   16 = openarm_right_finger_joint1   17 = its mimic follower joint2
# Re-confirmed 2026-08-12 by printing robot.data.joint_names from a live Articulation for this
# task. An older revision of this comment described a 22-DOF layout with the right gripper at
# index 17 -- that is stale, and 17 is the right gripper's MIMIC FOLLOWER, not the actuated joint.
# Taking it would mostly work (the mimic tracks joint1) and quietly stop working the moment the
# follower lags, so RJ_IDX uses 16.
#
# Only the leader finger of each gripper is exported: its follower is a pure USD mimic with no
# drive (matching the URDF's <mimic joint="..."/> tag), so it carries no independent information.
# The resulting per-arm 8 values, in this order, are exactly what
# OpenArmFollower.get_observation()/send_action() call LJ1.pos..LJ8.pos / RJ1.pos..RJ8.pos on the
# real robot -- so a deploy or replay script can build a motor frame from these names directly.
LJ_IDX = list(range(0, 14, 2)) + [14]
LJ_NAMES = [f"LJ{i}.pos" for i in range(1, 8)] + ["LJ8.pos"]
RJ_IDX = list(range(1, 14, 2)) + [16]
RJ_NAMES = [f"RJ{i}.pos" for i in range(1, 8)] + ["RJ8.pos"]

# Dual-arm layout is left-arm-block THEN right-arm-block (not interleaved like the articulation),
# because that is the order the real robot's action dict is written and read in.
BOTH_IDX = LJ_IDX + RJ_IDX
BOTH_NAMES = LJ_NAMES + RJ_NAMES

# Which columns are grippers, per layout width -- see _smooth_actions, which must not blur them.
GRIPPER_DIMS = {8: (7,), 16: (7, 15)}


def _smooth_actions(actions: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian-smooth every arm-joint dim; leave each gripper dim untouched.

    Generated Mimic trajectories are ~5× more jerky than source demos due to
    nearest-neighbour stitching discontinuities in the EE-pose commands used to step the
    sim -- discontinuities that still show up as small jumps in the resulting joint-position
    trajectory. sigma=2.0 gives ~95% jerk reduction while preserving the overall trajectory
    shape. Grippers are kept sharp so grasp/release timing is not blurred.

    The gripper columns are looked up by width rather than assumed to be the last one: in the
    dual-arm layout the left gripper sits at 7, in the MIDDLE of the vector, with the right arm's
    joints after it. Smoothing "everything but the last column" would blur a grasp closed and
    round off the right arm's joints twice.
    """
    gripper_dims = GRIPPER_DIMS.get(actions.shape[1])
    if gripper_dims is None:
        raise ValueError(
            f"Don't know where the gripper columns are for a {actions.shape[1]}-wide action;"
            f" expected one of {sorted(GRIPPER_DIMS)}."
        )
    arm_dims = [d for d in range(actions.shape[1]) if d not in gripper_dims]
    smoothed = actions.copy()
    smoothed[:, arm_dims] = gaussian_filter1d(actions[:, arm_dims].astype(np.float64),
                                              sigma=sigma, axis=0).astype(np.float32)
    return smoothed


def _load_episode(hdf5_path: str, ep_name: str, cameras: list, joint_idx: list = None) -> dict | None:
    """Load a single episode. Each call opens its own h5py handle (thread-safe)."""
    with h5py.File(hdf5_path, "r") as f:
        ep = f["data"][ep_name]

        # Action = joint-space, NOT the recorded EEF-delta `actions` field. The Isaac Lab task
        # records actions in whatever space its action term uses (here: left-arm IK delta +
        # gripper), which is what Mimic's data generation needs to adapt recorded demos to new
        # randomized cube poses -- but it's a different representation than `observation.state`
        # below (joint angles), which is exactly the mismatch that required a differential-IK
        # bridge on the real robot at deployment time. Instead, derive the action directly from
        # the NEXT timestep's measured joint position: for a trajectory that was physically
        # tracked step by step in sim, action[t] ~= state[t+1] is a standard proxy for "the joint
        # target commanded at step t", and it puts action and observation in the identical LJ1..8
        # representation -- so a policy trained on this data can drive the real robot directly,
        # no IK bridge needed. Costs the episode's last frame (no "next" state exists for it).
        # Read every DOF, then reorder in numpy. h5py's fancy indexing requires strictly
        # increasing indices, and the dual-arm layout deliberately is not: it is the left block
        # followed by the right block, while the articulation interleaves them. The full array is
        # (T, 18) floats, so reading all of it costs nothing worth optimising.
        joint_pos = ep["states/articulation/robot/joint_position"][:][:, joint_idx or LJ_IDX]
        if len(joint_pos) < 2:
            return None
        states  = joint_pos[:-1]
        actions = joint_pos[1:]
        T = len(actions)

        ep_dict: dict = {
            "name":    ep_name,
            "actions": _smooth_actions(actions.astype(np.float32)),
            "states":  states.astype(np.float32),
            "success": bool(ep.attrs.get("success", False)),
        }
        for cam in cameras:
            key = f"obs/{cam}"
            if key in ep:
                raw  = ep[key][:]
                imgs = raw[:, 0] if raw.ndim == 5 else raw
                ep_dict[f"cam_{cam}"] = imgs[:T]  # drop the same last frame, stays aligned
    return ep_dict


# ── name helpers ───────────────────────────────────────────────────────────────
# action and observation.state are now both the same 8D joint-space representation (see
# LJ_NAMES / LJ_IDX above) -- one name helper for both, instead of two different EEF-vs-joint
# naming schemes. These names deliberately match OpenArmFollower's real observation/action dict
# keys (LJ1.pos..LJ8.pos) exactly, so a deploy script can build a state/action frame straight
# from those dict keys with no renaming step.

def _joint_names(dim: int) -> list:
    if dim == 8:
        return LJ_NAMES
    if dim == 16:
        return BOTH_NAMES
    return [f"joint_{i}" for i in range(dim)]


# ── parquet helpers ────────────────────────────────────────────────────────────

def _write_parquet(path: Path, table: pa.Table):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def _write_data_chunk(ci: int, out_dir: Path,
                      actions: np.ndarray, states: np.ndarray,
                      timestamps: np.ndarray, frame_idx: np.ndarray,
                      ep_idx: np.ndarray, global_idx: np.ndarray,
                      task_idx: np.ndarray):
    table = pa.table({
        "action":            pa.array(actions.tolist(),    pa.list_(pa.float32())),
        "observation.state": pa.array(states.tolist(),     pa.list_(pa.float32())),
        "timestamp":         pa.array(timestamps.tolist(), pa.float32()),
        "frame_index":       pa.array(frame_idx.tolist(),  pa.int64()),
        "episode_index":     pa.array(ep_idx.tolist(),     pa.int64()),
        "index":             pa.array(global_idx.tolist(), pa.int64()),
        "task_index":        pa.array(task_idx.tolist(),   pa.int64()),
    })
    _write_parquet(out_dir / f"data/chunk-{ci:03d}/file-000.parquet", table)


def _write_episodes_chunk(ci: int, out_dir: Path,
                          ep_rows: list, active_cameras: list):
    def col(key, dtype):  return pa.array([r[key] for r in ep_rows], type=dtype)
    def lcol(key, it=pa.float64()): return pa.array([r[key] for r in ep_rows], type=pa.list_(it))

    cols: dict = {
        "episode_index":             col("episode_index",             pa.int64()),
        "tasks":                     pa.array([r["tasks"] for r in ep_rows], pa.list_(pa.string())),
        "length":                    col("length",                    pa.int64()),
        "data/chunk_index":          col("data/chunk_index",          pa.int64()),
        "data/file_index":           col("data/file_index",           pa.int64()),
        "dataset_from_index":        col("dataset_from_index",        pa.int64()),
        "dataset_to_index":          col("dataset_to_index",          pa.int64()),
        "meta/episodes/chunk_index": col("meta/episodes/chunk_index", pa.int64()),
        "meta/episodes/file_index":  col("meta/episodes/file_index",  pa.int64()),
    }
    for sk in ("action", "observation.state"):
        for q in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
            cols[f"stats/{sk}/{q}"] = lcol(f"stats/{sk}/{q}")
        cols[f"stats/{sk}/count"] = lcol(f"stats/{sk}/count", pa.int64())
    for cam in active_cameras:
        base = f"videos/observation.images.{cam}"
        cols[f"{base}/chunk_index"]    = col(f"{base}/chunk_index",    pa.int64())
        cols[f"{base}/file_index"]     = col(f"{base}/file_index",     pa.int64())
        cols[f"{base}/from_timestamp"] = col(f"{base}/from_timestamp", pa.float64())
        cols[f"{base}/to_timestamp"]   = col(f"{base}/to_timestamp",   pa.float64())

    _write_parquet(out_dir / f"meta/episodes/chunk-{ci:03d}/file-000.parquet",
                   pa.table(cols))


# ── per-episode mini stats (needed for episodes parquet) ─────────────────────

def _ep_stats(arr: np.ndarray) -> dict:
    return {
        "min":   arr.min(axis=0).tolist(),
        "max":   arr.max(axis=0).tolist(),
        "mean":  arr.mean(axis=0).tolist(),
        "std":   arr.std(axis=0).tolist(),
        "count": [int(len(arr))],
        "q01":   np.quantile(arr, 0.01, axis=0).tolist(),
        "q10":   np.quantile(arr, 0.10, axis=0).tolist(),
        "q50":   np.quantile(arr, 0.50, axis=0).tolist(),
        "q90":   np.quantile(arr, 0.90, axis=0).tolist(),
        "q99":   np.quantile(arr, 0.99, axis=0).tolist(),
    }


# ── main streaming pipeline ────────────────────────────────────────────────────

RIGHT_ARM_MOVED_THRESHOLD = 0.05
"""rad -- how far any right-arm joint must travel across an episode for --arms auto to call the
recording bimanual. Comfortably above sensor/settling noise (a held-still arm drifts ~1e-3) and
far below any real motion."""


def _right_arm_moves(hdf5_path: str, ep_name: str) -> float:
    """Peak-to-peak travel (rad) of the most-moved right-arm joint in one episode."""
    with h5py.File(hdf5_path, "r") as f:
        jp = f["data"][ep_name]["states/articulation/robot/joint_position"][:, RJ_IDX]
    if len(jp) < 2:
        return 0.0
    return float(np.ptp(jp, axis=0).max())


def hdf5_to_lerobot(
    hdf5_files: list,
    out: str,
    task_description: str,
    fps: int,
    cameras: list,
    chunks_size: int,
    arms: str = "auto",
):
    out_dir = Path(out)
    if out_dir.exists():
        _print(f"Removing existing {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # ── Phase 1: scan episode names (no frame loading) ────────────────────────
    ep_infos = _scan_episodes(hdf5_files)
    if not ep_infos:
        raise ValueError("No valid episodes found.")

    total_episodes = len(ep_infos)

    # ── Phase 1b: decide which arms to export ─────────────────────────────────
    # Exporting only the left arm out of a bimanual recording is silent and irreversible-looking:
    # the dataset is well-formed, trains, and replays -- it just has no right arm in it, which
    # only shows up when a hand-over demo turns out to be impossible to reproduce. So 'auto'
    # measures whether the right arm actually did anything, and always says which way it went.
    if arms == "auto":
        travel = max(_right_arm_moves(path, name) for path, name in ep_infos)
        joint_idx = BOTH_IDX if travel > RIGHT_ARM_MOVED_THRESHOLD else LJ_IDX
        verdict = "BOTH arms" if joint_idx is BOTH_IDX else "LEFT arm only"
        _print(f"Arms: {verdict} (auto -- right arm moved at most {travel:.3f} rad across"
               f" {total_episodes} episode(s), threshold {RIGHT_ARM_MOVED_THRESHOLD})")
    elif arms == "both":
        joint_idx = BOTH_IDX
        _print("Arms: BOTH arms (forced by --arms both)")
    else:
        joint_idx = LJ_IDX
        _print("Arms: LEFT arm only (forced by --arms left)")

    # ── Phase 2: probe first episode for dimensions ───────────────────────────
    first = _load_episode(ep_infos[0][0], ep_infos[0][1], cameras, joint_idx)
    assert first is not None, "First episode is empty."

    active_cameras = [c for c in cameras if f"cam_{c}" in first]
    if cameras and not active_cameras:
        _print("Warning: no camera data found — state-only dataset")

    action_dim = first["actions"].shape[1]
    state_dim  = first["states"].shape[1]
    cam_hw     = {c: first[f"cam_{c}"].shape[1:3] for c in active_cameras}

    _print(f"\n{total_episodes} episodes | action={action_dim}D | state={state_dim}D"
           + (f" | cameras={list(cam_hw.keys())}" if cam_hw else ""))

    # ── Phase 3: streaming conversion ─────────────────────────────────────────
    # Encoder is detected once from the first writer we open.
    enc_label = "unknown"

    # Global stats accumulators (updated per episode, no full concat needed)
    action_ost = _OnlineStats(action_dim, is_image=False)
    state_ost  = _OnlineStats(state_dim,  is_image=False)
    cam_ost    = {c: _OnlineStats(3, is_image=True) for c in active_cameras}

    # Global frame counter
    global_frame_idx = 0

    # Chunk-level state  (reset at each new chunk)
    current_ci       = -1
    video_writers: dict  = {}   # cam -> _VideoWriter
    cam_chunk_fcnt   = {c: 0 for c in active_cameras}  # frames written to current vid
    chunk_actions:   list = []
    chunk_states:    list = []
    chunk_timestamps:list = []
    chunk_frame_idx: list = []
    chunk_ep_idx:    list = []
    chunk_global_idx:list = []
    ep_rows:         list = []   # episodes-parquet rows for this chunk

    def _flush_chunk(ci: int):
        nonlocal enc_label
        if not chunk_actions:
            return
        # Close video writers
        for vw in video_writers.values():
            vw.close()
        if enc_label == "unknown" and video_writers:
            enc_label = next(iter(video_writers.values())).label
        # Write data parquet
        _write_data_chunk(
            ci, out_dir,
            np.concatenate(chunk_actions,    axis=0),
            np.concatenate(chunk_states,     axis=0),
            np.concatenate(chunk_timestamps, axis=0),
            np.concatenate(chunk_frame_idx,  axis=0),
            np.concatenate(chunk_ep_idx,     axis=0),
            np.concatenate(chunk_global_idx, axis=0),
            np.zeros(sum(len(a) for a in chunk_actions), dtype=np.int64),
        )
        # Write episodes parquet
        _write_episodes_chunk(ci, out_dir, ep_rows, active_cameras)
        _print(f"  chunk {ci:03d}: {len(ep_rows)} ep  "
               f"{sum(len(a) for a in chunk_actions)} frames written")

    def _reset_chunk_buffers():
        chunk_actions.clear()
        chunk_states.clear()
        chunk_timestamps.clear()
        chunk_frame_idx.clear()
        chunk_ep_idx.clear()
        chunk_global_idx.clear()
        ep_rows.clear()
        video_writers.clear()
        for c in active_cameras:
            cam_chunk_fcnt[c] = 0

    # ── prefetch loader: background thread loads the next episode while the
    # main thread encodes + writes the current one (overlaps HDF5 I/O with GPU).
    PREFETCH = 4
    load_q: queue.Queue = queue.Queue(maxsize=PREFETCH)

    def _loader_worker():
        for ep_path, ep_name in ep_infos:
            load_q.put(_load_episode(ep_path, ep_name, cameras, joint_idx))
        load_q.put(None)   # sentinel

    loader_thread = threading.Thread(target=_loader_worker, daemon=True)
    loader_thread.start()

    _print(f"\nConverting {total_episodes} episodes  [prefetch={PREFETCH}] ...")

    with _tqdm(total=total_episodes, desc="Episodes", unit="ep") as pbar:
        for abs_ep_idx in range(total_episodes):

            # ── get pre-loaded episode ────────────────────────────────────────
            ep = load_q.get()
            if ep is None:
                break

            ci = abs_ep_idx // chunks_size

            # ── new chunk: flush previous, open new video writers ─────────────
            if ci != current_ci:
                if current_ci >= 0:
                    _flush_chunk(current_ci)
                _reset_chunk_buffers()
                current_ci = ci
                for cam in active_cameras:
                    H, W = cam_hw[cam]
                    vpath = (out_dir /
                             f"videos/observation.images.{cam}"
                             f"/chunk-{ci:03d}/file-000.mp4")
                    video_writers[cam] = _VideoWriter(vpath, fps, H, W)

            if ep is None:   # _load_episode returned None (empty episode)
                pbar.update(1)
                continue

            T = len(ep["actions"])

            # ── write camera frames to persistent PyAV writers ────────────────
            ep_cam_ts: dict[str, tuple[float, float]] = {}
            for cam in active_cameras:
                cam_key = f"cam_{cam}"
                if cam_key not in ep:
                    continue
                frames = ep[cam_key]   # (T, H, W, 3) uint8

                from_ts = cam_chunk_fcnt[cam] / fps
                to_ts   = (cam_chunk_fcnt[cam] + T) / fps
                cam_chunk_fcnt[cam] += T
                ep_cam_ts[cam] = (from_ts, to_ts)

                video_writers[cam].write(frames)
                cam_ost[cam].update(frames)

            # ── update global stats ───────────────────────────────────────────
            action_ost.update(ep["actions"])
            state_ost.update(ep["states"])

            # ── build parquet rows for this episode ───────────────────────────
            ep_from = global_frame_idx
            ep_to   = global_frame_idx + T - 1
            global_frame_idx += T

            ts_arr  = (np.arange(T, dtype=np.float32) / fps)
            fi_arr  = np.arange(T, dtype=np.int64)
            ei_arr  = np.full(T, abs_ep_idx, dtype=np.int64)
            gi_arr  = np.arange(ep_from, ep_from + T, dtype=np.int64)

            chunk_actions.append(ep["actions"])
            chunk_states.append(ep["states"])
            chunk_timestamps.append(ts_arr)
            chunk_frame_idx.append(fi_arr)
            chunk_ep_idx.append(ei_arr)
            chunk_global_idx.append(gi_arr)

            # Episode-level stats for episodes.parquet
            a_st = _ep_stats(ep["actions"])
            s_st = _ep_stats(ep["states"])
            ep_row: dict = {
                "episode_index": abs_ep_idx,
                "tasks":         [task_description],
                "length":        T,
                "data/chunk_index": ci,
                "data/file_index":  0,
                "dataset_from_index": ep_from,
                "dataset_to_index":   ep_to,
                "meta/episodes/chunk_index": ci,
                "meta/episodes/file_index":  0,
                **{f"stats/action/{k}":            v for k, v in a_st.items()},
                **{f"stats/observation.state/{k}": v for k, v in s_st.items()},
            }
            for cam, (fts, tts) in ep_cam_ts.items():
                base = f"videos/observation.images.{cam}"
                ep_row[f"{base}/chunk_index"]    = ci
                ep_row[f"{base}/file_index"]     = 0
                ep_row[f"{base}/from_timestamp"] = fts
                ep_row[f"{base}/to_timestamp"]   = tts
            ep_rows.append(ep_row)

            # ── free episode memory ───────────────────────────────────────────
            del ep
            pbar.update(1)

    # ── flush final chunk ─────────────────────────────────────────────────────
    if current_ci >= 0 and chunk_actions:
        _flush_chunk(current_ci)

    total_frames = global_frame_idx

    # ── tasks.parquet ─────────────────────────────────────────────────────────
    tasks_df = pd.DataFrame({"task_index": [0]}, index=[task_description])
    tasks_df.index.name = None
    tasks_path = out_dir / "meta/tasks.parquet"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_df.to_parquet(tasks_path)

    # ── info.json ─────────────────────────────────────────────────────────────
    features: dict = {
        "action":            {"dtype": "float32", "shape": [action_dim],
                              "names": _joint_names(action_dim)},
        "observation.state": {"dtype": "float32", "shape": [state_dim],
                              "names": _joint_names(state_dim)},
        "timestamp":   {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64",   "shape": [1], "names": None},
        "episode_index":{"dtype": "int64",  "shape": [1], "names": None},
        "index":       {"dtype": "int64",   "shape": [1], "names": None},
        "task_index":  {"dtype": "int64",   "shape": [1], "names": None},
    }
    for cam in active_cameras:
        h, w = cam_hw[cam]
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": h, "video.width": w,
                "video.codec": "h264", "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False, "video.fps": fps,
                "video.channels": 3, "has_audio": False,
            },
        }
    info = {
        "codebase_version": "v3.0", "robot_type": "openarm",
        "total_episodes": total_episodes, "total_frames": total_frames,
        "total_tasks": 1, "chunks_size": chunks_size, "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path":  "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta/info.json", "w") as fp:
        json.dump(info, fp, indent=2)

    # ── stats.json (Welford online stats, no GPU required) ───────────────────
    _print("\nFinalising statistics ...")
    stats: dict = {
        "action":            action_ost.finalize(),
        "observation.state": state_ost.finalize(),
    }
    for cam in active_cameras:
        stats[f"observation.images.{cam}"] = cam_ost[cam].finalize()
    with open(out_dir / "meta/stats.json", "w") as fp:
        json.dump(stats, fp, indent=2)

    # ── summary ───────────────────────────────────────────────────────────────
    _print(f"\n✓  {out_dir}")
    _print(f"   {total_episodes} episodes  |  {total_frames} frames  |  {fps} fps")
    _print(f"   Encoder: {enc_label}")
    _print(f"   action {action_dim}D  |  state {state_dim}D")
    for cam in active_cameras:
        h, w = cam_hw[cam]
        _print(f"   {cam}: {h}×{w}")
    _print(f"\nLoad with:")
    _print(f"  from lerobot.datasets.lerobot_dataset import LeRobotDataset")
    _print(f"  ds = LeRobotDataset(repo_id='local/openarm', root='{out_dir}', download_videos=False)")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HDF5 → LeRobot v3.0 (streaming)", epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5",        nargs="+", required=True)
    ap.add_argument("--output",      required=True)
    ap.add_argument("--task",        default="manipulation task")
    ap.add_argument("--fps",         type=int, default=20)
    ap.add_argument("--cameras",     nargs="*", default=[], metavar="CAM")
    ap.add_argument("--chunks-size", type=int, default=100,
                    help="Episodes per chunk (default 100)")
    ap.add_argument("--load-workers",type=int, default=8,
                    help="Parallel HDF5 reader threads (default 8)")
    ap.add_argument("--arms", choices=["auto", "left", "both"], default="auto",
                    help="Which arms to export. 'both' gives a 16D LJ1-8+RJ1-8 state/action;"
                         " 'left' gives the 8D LJ1-8 single-arm layout older datasets use;"
                         " 'auto' (default) picks 'both' when the right arm actually moves in the"
                         " recording, and prints which it chose.")
    args = ap.parse_args()

    hdf5_to_lerobot(
        hdf5_files=args.hdf5,
        out=args.output,
        task_description=args.task,
        fps=args.fps,
        cameras=args.cameras,
        chunks_size=args.chunks_size,
        arms=args.arms,
    )