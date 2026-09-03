# Workflow

<img width="5896" height="2260" alt="image" src="https://github.com/user-attachments/assets/4613ed23-8c02-4d02-97a2-5d3f5b0a8c2f" />

# Demo: 

<img width="1183" height="699" alt="image" src="https://github.com/user-attachments/assets/0ed8e20d-7193-43cb-8d66-a2308e4a1f14" />

video: https://youtu.be/4DKriauQ05g?si=INIHtNN7Frs6Tbzm

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
cd ~/Stanley_ws/dora-openarm-data-collection
source .venv/bin/activate
dora run dataflow-vr-mujoco-ros2.yaml --uv
```

note:
make sure the ip in meta quest3 pro setup is as same as your pc, if not, you can use the following command to do the mapping

```
sudo ip addr add 10.100.1.240/24 dev wlp7s0
```

Resume recording

Add --resume to resume recording 

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
    --resume
```

# Replay dataset

```
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/tools/replay_demos.py \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \
    --dataset_file logs/demos/pickup_pringle.hdf5 \
    --enable_cameras
```

If you want to replay simulation-recorded trajectory on real robot

```
 env -u PYTHONPATH -u LD_LIBRARY_PATH ~/miniforge3/envs/lerobot-openarm-cf/bin/python   replay_hf_sim_episode_realgrip.py   --repo-id ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz --episode 0   --calibration calibration.json --model-path /home/csl/Stanley_ws/IsaacLab/source/isaaclab_assets/data/v1_camera_isaac/urdf/v1_camera.urdf   --grip-continuous --grip-input-closed 0.029 --grip-close-frac 1.0   --handshake-tolerance 1.0 --ramp-duration 3.0 --max-joint-speed 1.8   --max-steps 3000 --plot sim_vs_real_realgrip_continuous.png --playback-hz 7.5
```

# Remove episode

```
./isaaclab.sh -p scripts/tools/remove_demos_hdf5.py \
    --dataset_file logs/demos/pickup_pringles_VR_V7.hdf5 --episodes 3 7 10~13 \
    --output logs/demos/pickup_pringles_VR_V7_fixed.hdf5
```

# Isaac Lab Mimic

## Record source demo (keyboard teleoperation)

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

## Annotate with subtask signals (auto-mode uses get_subtask_term_signals)

```
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0 \
  --task_mode handover --auto --from_states \
  --enable_cameras \
  --input_file logs/demos/pickup_pringle.hdf5 \
  --output_file logs/demos/pickup_pringles_annotated.hdf5
```

## Generate augmented dataset

```
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0 \
    --input_file logs/demos/pickup_pringles_annotated.hdf5 \
    --output_file logs/demos/pickup_pringles_generated.hdf5 \
    --task_mode handover \
    --generation_num_trials 50 --num_envs 4 --enable_cameras
```

### Generate augemented dataset w domain randomization(lighting,camera angle...)

```
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py     --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0     --input_file logs/demos/pickup_pringles_annotated.hdf5     --output_file logs/demos/pickup_pringles_dr_lighting_generated.hdf5     --generation_num_trials 10 --num_envs 4 --enable_cameras     --enable_domain_randomization --domain_randomization_profile visual  --task_mode handover
```

### Generate augemented dataset w domain randomization(pringles's size)

Add "--randomize_object_size" to randomize pringles'size

```
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0 \
    --input_file logs/demos/pickup_pringles_annotated.hdf5 \
    --output_file logs/demos/pickup_pringles_dr_size_generated.hdf5 \
    --generation_num_trials 50 --num_envs 4 --enable_cameras \
    --randomize_object_size \
    --task_mode handover 
```

### Generate augemented dataset w domain randomization(background changing)

```
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py     --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0     --input_file logs/demos/pickup_pringles_V9_annotated.hdf5     --output_file logs/demos/pickup_pringles_dr_strong_generated.hdf5     --generation_num_trials 50 --num_envs 4 --enable_cameras     --enable_domain_randomization --task_mode handover
```

# Convert HDF5 to LeRobot format 

```
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
python -u scripts/tools/convert_hdf5_to_lerobot.py     --hdf5 logs/demos/pickup_pringle.hdf5     --output ~/Stanley_ws/IsaacLab/datasets/ethanCSL/openarm_visuomotor_VR_pringles_test     --task "Pick up the Pringles can with the right arm, hand it to the left arm"     --fps 30 --cameras right_wrist_cam wrist_cam body_cam
```

# Push dataset to the Hub

```
conda activate lerobot
python scripts/push_to_hub.py
```

# Train in LeRobot format

```
cd ~/CSL/lerobot/ && conda activate lerobot
 lerobot-train   --policy.path=lerobot/smolvla_base   --dataset.repo_id=ethanCSL/openarm_visuomotor_VR_pringles_test   --batch_size=16   --steps=40000   --output_dir=outputs/train/openarm_visuomotor_VR_pringles_test   --job_name=my_smolvla_training   --policy.device=cuda   --policy.repo_id=ethanCSL/openarm_visuomotor_VR_pringles_test  --wandb.enable=false   --rename_map='{
    "observation.images.right_wrist_cam": "observation.images.camera1",
    "observation.images.wrist_cam":       "observation.images.camera2",
    "observation.images.body_cam":        "observation.images.camera3"
  }'   --dataset.video_backend=pyav
```

# Model Evaluation

## Deploy in Isaac Sim

Launch SmolVLA Policy Server

```
conda activate lerobot && cd ~/Stanley_ws/IsaacLab
python scripts/imitation_learning/lerobot/smolvla_server.py \
    --checkpoint ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz \
    --task "Pick up the Pringles can with the right arm, hand it to the left arm." \
    --port 5556
```

Run Isaac Lab Eval

```
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
 ./isaaclab.sh -p scripts/imitation_learning/lerobot/eval_smolvla_jointspace.py     --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0     --num_rollouts 5 --horizon 300 --enable_cameras     --cameras right_wrist_cam,wrist_cam,body_cam --task_mode handover
```

## Run Isaac Lab Eval, and send command to real robot

Checking possible sim2real gap with this experiment

<img width="1686" height="839" alt="image" src="https://github.com/user-attachments/assets/c4c7885c-d578-4fa7-9b7f-b1c1ff8908d3" />

Run smolvla server

```
cd ~/Stanley_ws/IsaacLab && conda activate lerobot
python scripts/imitation_learning/lerobot/smolvla_server.py     --checkpoint ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz     --task "Pick up the Pringles can with the right arm, hand it to the left arm."     --port 5556
```

Run Isaac Lab control

```
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/imitation_learning/lerobot/eval_smolvla_jointspace.py     --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0     --num_rollouts 5 --horizon 300 --enable_cameras     --cameras right_wrist_cam,wrist_cam,body_cam --task_mode handover     --mirror_udp_port 5557 --mirror_feedback_port 5558 --mirror_rate_hz 20
```

Run UDP to broadcast joint states

```
cd ~/Stanley_ws/lerobot_openarm && uv sync && source .venv/bin/activate
env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python mirror_bridge.py \
    --calibration calibration.json \
    --udp-port 5557 \
    --feedback-port 5558 \
    --model-path /home/csl/Stanley_ws/lerobot_openarm/model/openarm_description.urdf \
    --right-port can0 --left-port can1 \
    --max-joint-speed 0.3
```

### Press "Enter" in the "Run Isaac Lab control" terminal to start!!

# Run in Real-world

## Activate CAN-FD

```
cd ~/Stanley_ws/openarm_can/setup
```

```
sudo ./my_arm 
```

## Model evaluation on real robot

```
cd ~/Stanley_ws/lerobot_openarm
uv sync
source .venv/bin/activate
env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python deploy_smolvla_pickup_jointspace.py     --checkpoint ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz     --body-cam-index rs_body --wrist-cam-index rs_wrist_left --right-wrist-cam-index rs_wrist_right     --calibration calibration.json     --inference-hz 30 --max-joint-speed 1.5 --max-episode-seconds 600 
```

Deploy in async evaluation

```
cd ~/Stanley_ws/lerobot_openarm
uv sync
source .venv/bin/activate
env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python deploy_smolvla_async.py     --checkpoint ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz    --body-cam-index rs_body --wrist-cam-index rs_wrist_left --right-wrist-cam-index rs_wrist_right     --calibration calibration.json     --control-hz 30 --max-joint-speed 1.5     --actions-per-chunk 50 --chunk-size-threshold 0.8     --max-episode-seconds 25 --max-episodes 20
```




