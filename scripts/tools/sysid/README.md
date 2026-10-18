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
| `extract_real_frames.py`, `render_real_states.py`, `segment_robot_real.py`, `calibrate_camera.py` | see "Camera calibration" | Fit a sim camera's mount to the real one from robot silhouettes. |

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

## Camera calibration (real2sim vision)

Makes a sim camera see the robot where the real one does, from an ordinary real LeRobot dataset -- no
checkerboard. The real and sim images look nothing alike, so robot silhouettes are compared, not pixels.

```
real dataset ──extract_real_frames.py──> frames.npz (frames + joint states)
frames.npz ──render_real_states.py (Isaac)──> poses.npz (sim robot posed per frame: link poses, mesh samples, depth)
frames.npz ──segment_robot_real.py (Grounding DINO + SAM 2.1)──> robot_masks.npz
poses.npz + robot_masks.npz ──calibrate_camera.py──> camera_fit.json (CameraCfg.OffsetCfg)
```

```bash
PY=~/Stanley_ws/lerobot_openarm/.venv/bin/python3          # has av + transformers >= 5
$PY scripts/tools/sysid/extract_real_frames.py ~/.cache/huggingface/lerobot/ethanCSL/openarm_pringles_lerobot_real_v00 \
    logs/real2sim/body_cam_frames --episodes 25 --per_episode 5 --max_speed 0.3
./isaaclab.sh -p scripts/tools/sysid/render_real_states.py --enable_cameras --headless \
    --frames logs/real2sim/body_cam_frames/frames.npz --out logs/real2sim/body_cam_render --surface_points 1000000
$PY scripts/tools/sysid/segment_robot_real.py logs/real2sim/body_cam_frames/frames.npz
$PY scripts/tools/sysid/calibrate_camera.py logs/real2sim/body_cam_render/poses.npz \
    logs/real2sim/body_cam_frames/robot_masks.npz --intrinsics 607.22,607.22,320,240 \
    --frames logs/real2sim/body_cam_frames/frames.npz --out logs/real2sim/body_cam_fit
```

* **Intrinsics are measured, not fitted.** `rs-enumerate-devices -c` prints each D435i's colour intrinsics;
  the body cam (USB port 2.4.3, `/dev/rs_body`) is fx = fy = 607.2 at 640x480. Pass the **sim's** effective
  intrinsics to the fit (principal point at the centre): the renderer ignores aperture offsets, so the
  real (326.4, 238.7) cannot be reproduced and the fitted mount absorbs it as a ~0.6 deg tilt.
* `render_real_states.py --intrinsics/--cam_pos/--cam_rot` re-renders with a fit (sim_XX / overlay_XX.png);
  rendering from a fit and fitting again refines it.
* Low-IoU frames are usually segmentation misses (an arm not detected), not camera error: check mask_XX.png.

Result for body_cam (applied in stack_ik_abs_visuomotor_env_cfg.py): mount ~14 cm lower and ~6 deg steeper
than the hand-tuned one, robot-silhouette IoU 0.33 -> 0.71 median over 123 frames; restarts agree within
~2 cm / ~3 deg (mostly a sideways-shift vs yaw trade-off).

### Real look (2026-10-18): recording with `--real_look`

`Isaac-PickUp-RedCube-OpenArm-IK-Abs-v1` is the pick-up task with this look built in (same physics, actions, success and
task modes as `-v0`; `lerobot_sim_server.py --task ...-v1` needs no extra flag). On `-v0`, `--real_look` does the same.
It renders the scene with a look fitted from the real
dataset (`source/isaaclab_assets/data/real2sim/pringles_v00/`, applied by `openarm_real_look.py`):

| file | made by | what |
|---|---|---|
| `albedo.json` | `fit_robot_colors.py` (closed loop, brightness-only gains) | per-part robot colours |
| `lighting.json` | by hand, checked against part colours | dome light intensity / cool tint |
| `backdrop_body.npz` | `build_clean_plate.py` -> `body_cam_depth.py` -> `build_backdrop.py` | real table + room as a photo-coloured mesh, table re-projected onto the pad plane (z=0.28), room flattened onto a wall at x=0.9 |
| `backdrop_*wrist_cam.npz` | `build_wrist_backdrop.py` | rest-pose room view for each wrist cam, behind the wall |
| `tone.json` | `fit_tone.py` | emissive -> pixel curve of the RTX tone mapper |
| `sensor.json` | Laplacian-variance match | per-camera blur / noise applied to each frame the server sends |

Measured with `eval_realism.py` (DINOv2): paired real-vs-sim feature distance 0.63 -> 0.55 (body), 0.88 -> 0.72 (left wrist),
0.86 -> 0.75 (right wrist); a linear probe still separates real from sim perfectly (AUC 1.0).

Findings worth keeping: monocular depth put the table ~15 cm above the pad, contradicted by the real grasp heights (jaws at
z~0.33 m at grasp), so the pad was left at 0.28; `wrist_handeye.py`'s two identical cameras disagreed by 15 deg, so wrist
mounts were left at the URDF values and only the lenses (fx 605/608) were applied; COLMAP SfM of the room failed (blank walls,
soft frames). `--real_look` is for `--domain_randomization none`: the pad/light randomizers would move things the backdrop
has baked in.

### Transfer test (2026-10-18): does a sim look help on real frames?

`prepare_real_replay.py` (real trajectories + can start per episode) -> `replay_render_real.py --look legacy|plain|look`
(replays the real actions in sim, 50 episodes, same layout) -> `probe_transfer.py` / `probe_localization.py`
(frozen DINOv2 features of the 3 cameras, small head, trained on episodes 0-39 of a sim look, scored on the REAL
frames of episodes 40-49). Can start in sim = right jaw position at first closure from FK (+0.15 m along the tool axis,
picked because it lifted the can in 8/12 replays vs 0-3 for the others); camera-based can positions agree with it
(corr 0.99/0.97, 2 cm offset), which also confirms the table is at the pad height.

* Next-move prediction from features + joint state: real-trained R^2 0.862, state alone 0.837, sim-trained legacy 0.002,
  plain 0.039, look 0.113. The state explains almost everything, so this mostly measures feature-shift damage.
* Can position (x,y) from pre-grasp frames, no state (guessing the mean: 2.8 cm): real-trained 0.93 cm; legacy 4.43,
  plain 5.31, look 2.87 (all cameras). Body camera only: real 0.70, legacy 3.99, plain 4.96, look 2.30. Wrist only: real
  2.86 (no information), legacy 14.2, plain 8.1, look 4.2.

So the look helps in every test (it turns harmful sim features into neutral or mildly useful ones) but a head trained on
it still does far worse than one trained on real frames. Replaying the real actions carries the can in only 22/50 sim
episodes (finger geometry/friction differ), under every look.

### Contact physics fitted to the real episodes (2026-10-18)

`fit_contact_physics.py` replays all 50 real episodes at once in 50 parallel envs (no cameras, ~12 s per evaluation) and
searches friction / can mass / gripper stiffness, damping and squeeze force for the set under which the replayed can stays
in a jaw as long as it does in the real recordings (score: fraction of the carry phase with a jaw on the can). 90
evaluations, trained on episodes 0-34, scored on 35-49. cfg physics: follow 0.41 / 0.42 (train / held-out), can lifted and
still up at the end in 14/50 episodes. Fitted: 0.71 / 0.59, 29/50. Top candidates agree on: gripper squeeze force 26-30
(cfg 8), finger friction 0.22-0.29 (cfg 0.5), can mass at the 0.10 lower bound (cfg 0.205), can friction 1.0-1.3,
stiffness 455-636, damping 4-10. These are effective values (mass, friction and squeeze trade off), not measurements;
weigh the can to pin the mass. `contact_physics.json` in the look folder is applied by the server for `-v1`
(`openarm_real_look.apply_contact_physics`); through the live server 7/10 replayed real episodes now meet the hand-over
success condition (5/10 before).
