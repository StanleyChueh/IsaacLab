# Sim <-> real system identification for the OpenArm

Tools to find out where the simulated arm and the real follower respond differently, and to make the sim
match where the data says it should. Lives on the `plate_wiping` branch.

```
real dataset ──analyze_real_dataset.py──> real_arm_response.json + traces.npz
deploy rollout ──deploy_smolvla_async.py --save-trace──> traces.npz           (large-signal data)
replay on real ──replay_hf_sim_episode_realgrip.py --log-csv──sysid_csv_to_trace.py──> traces.npz   (repeatable probe)
traces.npz ──fit_sim_actuators.py (Isaac)──> sim_actuator_profile.json ──OPENARM_ACTUATOR_PROFILE=──> env
```

| Tool | Runs in | Purpose |
|---|---|---|
| `analyze_real_dataset.py` | any python with numpy/scipy/pandas | Fit the real arm's response (speed cap, motor lag, delay, offset, gripper) from a mirror_bridge-recorded LeRobot dataset. Optional `--compare-root` against a sim dataset. |
| `fit_sim_actuators.py` | `./isaaclab.sh -p` | Replay real episodes open-loop in many parallel sim envs with different stiffness / damping / armature; keep what tracks best. Reports baseline-vs-fitted error on **held-out** episodes and only adopts a group that improves >= 10%. |
| `check_env_rate.py` | `./isaaclab.sh -p` | Print the control rate the env resolves to, derived step counts, resolved actuator gains, and whether a source HDF5 matches. |
| `_sim_common.py` | — | Camera-free env builder shared by the above. |

Kit can swallow stdout in headless runs, so the Isaac tools also write a report file (`fit_report.txt`,
`--report`).

## 30 Hz control rate

`openarm_sim_timing.py` is the single source of truth. Default is **30 Hz** (physics 120 Hz x 4 substeps);
`OPENARM_CONTROL_HZ=20` restores the legacy 20 Hz exactly. Every step count tuned at 20 Hz (`SEAM_INTERP`,
the left-arm phase compensation, smoothing windows, `HANDOVER_RECEIVER_HOLD_STEPS`, interpolation steps,
the release overlap) is rescaled to the same duration.

* Mimic replays the source demo **one waypoint per env step**, so a 20 Hz source demo run in a 30 Hz env
  plays 1.5x too fast. `generate_dataset.py` and `annotate_demos.py` now read the rate stored in the
  HDF5 (`data.attrs['env_args']['sim_args']`) and refuse a mismatch (`--allow_rate_mismatch` to override).
  **Re-record source demos at 30 Hz**, or run with `OPENARM_CONTROL_HZ=20` to keep using the old ones.
* `convert_hdf5_to_lerobot.py --fps 30` is now correct for 30 Hz data; use `--fps 20` for 20 Hz data.
* The plate-wipe env is pinned to 20 Hz (its deformable rag was not re-verified at the new physics step).
* `render_interval` equals the decimation at 30 Hz, so cameras render once per step on the last substep.

## What the identification found (ethanCSL/openarm_pringles_real_v00, 30 episodes)

* The real follower is rate-limited by **mirror_bridge's `--max-joint-speed`**, not by its motors: one cap of
  ~0.38 rad/s fits all 14 arm joints. Motor lag is 12–53 ms with no extra delay.
* The current sim actuators (kp 230/290/30, kd from the MuJoCo XML) already track the real arm's motion as
  well as the data can tell; the search found no >= 10% improvement. The sim robot already has gravity
  disabled, which matches the follower's gravity feed-forward (turning it on makes tracking clearly worse).
* This data cannot constrain large-signal behaviour (commands move <= 0.4 rad/s, sampled at 30 fps).
  Record a deploy rollout with `--save-trace` and refit to cover deploy speeds.

## Limits

* Sim and real traces must be at the same rate (`fit_sim_actuators.py` refuses otherwise).
* Per-joint identification treats joints as independent; a joint's tracking error also depends on its
  neighbours, which the search treats as noise.
* `resolvable: false` in `real_arm_response.json` means the real time constant is shorter than ~1 frame.
