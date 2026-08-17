<img width="5152" height="2528" alt="image" src="https://github.com/user-attachments/assets/d65a0dad-00ac-4ab2-8849-91b8ba8e604a" />

# Record dataset

```
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/tools/record_demos_openarm.py     --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0     --dataset_file logs/demos/pickup.hdf5     --enable_cameras --num_demos 1 --teleop_device keyboard
```

```
=== OpenArm Dual-Arm Recording ===
Key	                  Action
W / S	                EE forward / backward (+X / -X)
A / D	                EE left / right (+Y / -Y)
PgUp / PgDn	            EE up / down (+Z / -Z)
↑ / ↓	                pitch ±
← / →	                yaw ±
[ / ]	                roll ±
K	                    gripper toggle
TAB	                    switch arm (left ↔ right)
N	                    save episode
R	                    reset/discard

```

# Record dataset w Meta Quest3 Pro

Subscribe joint topic from dora, and control robot in isaac sim

```
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/tools/record_demos_openarm.py \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \
    --dataset_file logs/demos/pickup_pringle.hdf5 \
    --enable_cameras \
    --num_demos 10 \
    --teleop_device vr_joint_ros2_native \
    --ros2_domain_id 1 \
    --task_mode handover \
    --manual_save 
```

Dora publish code

please refer to https://github.com/StanleyChueh/dora-openarm-data-collection.git 

```
cd dora-openarm-data-collection
dora run dataflow-vr-mujoco-ros2.yaml --uv
```

# Replay dataset

```
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/tools/replay_demos.py \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \
    --dataset_file logs/demos/pickup_pringle.hdf5 \
    --enable_cameras
```

# Environment Setup

If you want to change environment in IsaacSim, please refer to the following setting

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_joint_pos_env_cfg.py
```

### Table / Workspace (Pad)

PAD_HEIGHT = 0.13          # height of the gray platform (m)

CUBE_Z = PAD_HEIGHT + 0.02 # cube resting height = pad top + half-block

### Robot Base Height

```
self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)  # adjust Z to raise/lower
```

### Cube Size and Color

```
for i, (name, pos, color) in enumerate([
    ("cube_1", [0.2,  0.08,  CUBE_Z], "blue"),   # change "blue" → any color name
    ("cube_2", [0.55, 0.05,  CUBE_Z], "red"),     # the pickup cube
    ("cube_3", [0.60, -0.10, CUBE_Z], "green")
]):
    spawn=UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/{color}_block.usd",  # USD determines size too
    )
```

### Cube Randomization Range (each episode)

```
randomize_cube_2 = EventTerm(
    func=mdp_core.reset_root_state_uniform,
    params={
        "pose_range": {
            "x": (-0.38, -0.28),  # offset from default x=0.55 → actual x:[0.17, 0.27]
            "y": (-0.02,  0.12),  # offset from default y=0.05 → actual y:[0.03, 0.17]
        },
        ...
    },
)
```

# Isaac Lab Mimic

Record source demo (keyboard teleoperation)

```
/isaaclab.sh -p scripts/tools/record_demos_openarm.py   --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0   --dataset_file logs/demos/pickup_pringle.hdf5 --enable_cameras --num_demos 10   --teleop_device vr_joint_ros2_native --ros2_domain_id 1   --task_mode handover --manual_save
```

Annotate with subtask signals (auto-mode uses get_subtask_term_signals)

```
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0 \
  --task_mode handover --auto --from_states \
  --enable_cameras \
  --input_file logs/demos/pickup_pringle_annotated.hdf5 \
  --output_file logs/demos/pickup_pringle_annotated_v3.hdf5

```

Generate augmented dataset

```
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0 \
    --input_file logs/demos/pickup_pringle_annotated.hdf5 \
    --output_file logs/demos/pickup_generated.hdf5 \
    --task_mode handover \
    --generation_num_trials 50 --num_envs 4 --enable_cameras
```

Generate augemented dataset w domain randomization(No background changing)

```
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py     --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0     --input_file logs/demos/pickup_pringle_annotated_V6.hdf5     --output_file logs/demos/pickup_pringle_V6_generated.hdf5     --generation_num_trials 10 --num_envs 4 --enable_cameras     --enable_domain_randomization --domain_randomization_profile visual  --task_mode handove
```

Generate augemented dataset w domain randomization

```
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0 \
    --input_file logs/demos/pickup_annotated.hdf5 \
    --output_file logs/demos/pickup_generated.hdf5 \
    --generation_num_trials 50 --num_envs 4 --enable_cameras \
    --enable_domain_randomization
```

# Convert HDF5 to LeRobot format 

```
python -u scripts/tools/convert_hdf5_to_lerobot.py     --hdf5 logs/demos/pickup_pringle.hdf5     --output ~/Stanley_ws/IsaacLab/datasets/ethanCSL/openarm_visuomotor_VR_pringles     --task "Pick up the Pringles can with the right arm, hand it to the left arm"     --fps 30 --cameras right_wrist_cam wrist_cam body_cam
```

# Train in LeRobot format

```
cd ~/CSL/lerobot/ && conda activate lerobot
 lerobot-train   --policy.path=lerobot/smolvla_base   --dataset.repo_id=ethanCSL/openarm_visuomotor   --batch_size=16   --steps=40000   --output_dir=outputs/train/openarm_visuomotor   --job_name=my_smolvla_training   --policy.device=cuda   --policy.repo_id=ethanCSL/openarm_visuomotor  --wandb.enable=false   --rename_map='{
    "observation.images.front_cam": "observation.images.camera1",
    "observation.images.body_cam":   "observation.images.camera2",
    "observation.images.wrist_cam":  "observation.images.camera3"
  }'   --dataset.video_backend=pyav
```

# Deploy in Isaac Sim

Launch SmolVLA Policy Server

```
conda activate lerobot && cd ~/Stanley_ws/IsaacLab
python scripts/imitation_learning/lerobot/smolvla_server.py \
    --checkpoint ethanCSL/openarm_visuomotor_augmented_dataset_1000 \
    --task "Pick up the red cube." \
    --port 5556
```

Run Isaac Lab Eval

```
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/imitation_learning/lerobot/eval_smolvla_jointspace.py \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \
    --num_rollouts 5 --horizon 300 --enable_cameras \
    --cameras body_cam,wrist_cam
```

# Run in Real-world

```
cd ~/openarm_can/setup
```

```
sudo ./my_arm 
```

Deploy in joint states trained model

```
cd ~/Stanley_ws/lerobot_openarm
uv sync
source .venv/bin/activate
python deploy_smolvla_pickup_jointspace.py     --checkpoint ethanCSL/openarm_visuomotor_no_domain_randomization_1000_joints     --body-cam-index 4 --wrist-cam-index 10 --side-cam-index 12     --inference-hz 30 --max-joint-speed 1.0 --max-episode-seconds 30     --calibration calibration.json     --no-live-view --save-video rollout_v2.mp4
```

## OpenARM Motors Check

```
cd ~/Stanley_ws/lerobot_openarm
uv sync
source .venv/bin/activate
python safe_probe.py --side left --joint 1 --step 0.1 --max-kp 150 --skip-ctrl-mode
```

You can change joint number and arm to different testing 

## OpenARM Teleoperation Mirror Test

```
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/tools/record_demos_openarm.py     --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0     --dataset_file logs/demos/pickup.hdf5     --enable_cameras --num_demos 10 --teleop_device keyboard     --mirror_udp_port 9999 --mirror_feedback_port 9998
```

```
cd ~/Stanley_ws/lerobot_openarm
uv sync
source .venv/bin/activate
python mirror_bridge.py --calibration calibration.json --udp-port 9999     --right-port can0 --left-port can1     --model-path model/openarm_description_leader.urdf     --max-joint-speed 0.5     --feedback-port 9998
```

# Replay Trajectory(sim-to-real & real-to-sim)

sim-to-real

```
 env -u PYTHONPATH -u LD_LIBRARY_PATH ~/miniforge3/envs/lerobot-openarm-cf/bin/python   replay_hf_sim_episode_realgrip.py   --repo-id ethanCSL/openarm_visuomotor_VR_pringles_V6 --episode 1   --calibration calibration.json --model-path /home/csl/Stanley_ws/IsaacLab/source/isaaclab_assets/data/v1_camera_isaac/urdf/v1_camera.urdf   --grip-continuous --grip-input-closed 0.029 --grip-close-frac 1.0   --handshake-tolerance 1.0 --ramp-duration 3.0 --max-joint-speed 1.8   --max-steps 3000 --plot sim_vs_real_realgrip_continuous.png --playback-hz 7.5
```

real-to-sim

```
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/tools/replay_real_dataset_in_sim.py \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \
    --repo-id ethanCSL/0422_stanley_red_cube --episode 0 \
    --calibration ~/lerobot_openarm/calibration.json --enable_cameras
```

