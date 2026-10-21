# OpenArm × Isaac Lab (`openarm-dev`)

<img width="5896" height="2260" alt="workflow" src="https://github.com/user-attachments/assets/4613ed23-8c02-4d02-97a2-5d3f5b0a8c2f" />

<img width="1858" height="1053" alt="demo" src="https://github.com/user-attachments/assets/c82d8a0c-d545-45c4-acc3-1fc701660a3b" />

Video: https://youtu.be/4DKriauQ05g?si=INIHtNN7Frs6Tbzm

This repo is the **simulator half** of the [lerobot_openarm](https://github.com/StanleyChueh/lerobot_openarm/tree/official-lerobot-quest)
pipeline. Recording, training and evaluation — real and sim — are run from lerobot_openarm
(real: Steps 0-5, sim: Sim Steps 1-6). Here: the sim server, Isaac Lab Mimic, and the robot / scene assets.

## Setup

```bash
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
```

The robot's USD files are gitignored and live only on your machine (`source/isaaclab_assets/data/v1_camera_isaac/`);
back that folder up. After changing the gripper or camera-mount scripts, rebuild their layers:

```bash
./isaaclab.sh -p scripts/tools/sysid/add_openarm_hand.py
./isaaclab.sh -p scripts/tools/sysid/add_openarm_camera_mount.py
```

## Tasks

| Task | Use |
|---|---|
| `Isaac-PickUp-RedCube-OpenArm-IK-Abs-v1` | **Recommended.** Pringles hand-over with the real look (real colours, room, camera noise, fitted contact physics) |
| `Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0` | Same task, plain sim look |
| `Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v1` / `-Mimic-v0` | Mimic annotate / generate for the tasks above |
| `Isaac-PlateWipe-OpenArm-IK-Abs-v0` | Plate wiping |
| `Isaac-InsertBattery-OpenArm-IK-Abs-v0` | Battery insertion |

## 1. Sim server (what `--robot.type=openarm_isaac` connects to)

```bash
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/tools/lerobot_sim_server.py --enable_cameras --headless \
  --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v1 --task_mode handover \
  --domain_randomization none \
  --mimic_hdf5 logs/demos/pringles_sim_src_v00.hdf5
```

- Start `lerobot-record` / `lerobot-rollout` only after `[lerobot_sim_server] ready on 127.0.0.1:5710`.
- Drop `--mimic_hdf5 ...` for teleoperation and evaluation.
- Restart the server after pulling changes.

## 2. Mimic: synthetic data from the recorded sim demos

Stop the sim server first (Ctrl-C).

```bash
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab

# 2a. Annotate
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v1 \
  --task_mode handover --auto --from_states \
  --enable_cameras --headless \
  --input_file logs/demos/pringles_sim_src_v00.hdf5 \
  --output_file logs/demos/pringles_sim_src_v00_annotated.hdf5

# 2b. Generate (domain randomization: visual)
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v1 --task_mode handover \
  --input_file logs/demos/pringles_sim_src_v00_annotated.hdf5 \
  --output_file logs/demos/pringles_sim_mimic_v00.hdf5 \
  --generation_num_trials 500 --num_envs 4 --enable_cameras --headless \
  --enable_domain_randomization --domain_randomization_profile visual
```

Other generate options:

```bash
# no randomization
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v1 --task_mode handover \
  --input_file logs/demos/pringles_sim_src_v00_annotated.hdf5 \
  --output_file logs/demos/pringles_sim_mimic_v00.hdf5 \
  --generation_num_trials 500 --num_envs 4 --enable_cameras --headless

# strong randomization: background swap + any colour + table position + can size
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v1 --task_mode handover \
  --input_file logs/demos/pringles_sim_src_v00_annotated.hdf5 \
  --output_file logs/demos/pringles_sim_mimic_v00.hdf5 \
  --generation_num_trials 500 --num_envs 4 --enable_cameras --headless \
  --enable_domain_randomization --domain_randomization_profile full --randomize_object_size
```

Then convert to a LeRobot dataset with lerobot_openarm's `mimic_to_lerobot.py` (lerobot_openarm README, Sim Step 4).

## Check or clean recorded demos

```bash
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab

# replay one demo (episode 0)
./isaaclab.sh -p scripts/tools/replay_demos.py \
  --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v1 --task_mode handover \
  --dataset_file logs/demos/pringles_sim_src_v00.hdf5 --select_episodes 0 --enable_cameras

# remove bad demos (here 3, 7 and 11-13) into a new file
./isaaclab.sh -p scripts/tools/remove_demos_hdf5.py \
  --dataset_file logs/demos/pringles_sim_src_v00.hdf5 --episodes 3 7 11-13 \
  --output logs/demos/pringles_sim_src_v00_clean.hdf5
```

## Domain randomization profiles

| Profile | Changes |
|---|---|
| `none` | nothing (recording source demos) |
| `visual` | lighting, object/table colour (hue kept), camera mount jitter, table height |
| `full` | `visual` + background swap (random skybox), any object colour, table position |

On `-v1` / `-Mimic-v1` the lighting change also applies to the photographed room and table.

## Tips

- **SAFETY HOLD in sim** (`request is ... from the measured joint`): add `--robot.max_tracking_error=1.0` to `lerobot-record`.
- **Robot self-collisions** are off by default (the arms can pass through each other); turn them on with
  `OPENARM_SELF_COLLISIONS=1 ./isaaclab.sh -p ...`.
- **One sim server at a time** on a 16 GB GPU.
- **Disk:** Mimic output is ~250 MB per generated demo (cameras included).
- **New recording set:** bump `_v00` in the dataset / HDF5 names.

## Sim ↔ real fidelity

Gripper with the official finger offsets and foam pads (opens like the real one), real D435i bar + clamp ring + cable,
black mats, room backdrop fused from the real recordings, fitted robot colours and contact physics.
Tools and how each was measured: [`scripts/tools/sysid/README.md`](scripts/tools/sysid/README.md).

<details>
<summary><b>Legacy workflow</b> (before lerobot_openarm; kept for reference, not maintained)</summary>

Record in Isaac Sim with Quest 3 Pro (dora + MuJoCo IK):

```bash
cd ~/Stanley_ws/dora-openarm-data-collection && source .venv/bin/activate
dora run dataflow-vr-mujoco-ros2.yaml --uv
```

```bash
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/tools/record_demos_openarm.py \
  --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 --task_mode handover \
  --dataset_file logs/demos/pickup_pringle.hdf5 \
  --enable_cameras --num_demos 10 \
  --teleop_device vr_joint_ros2_native --ros2_domain_id 1 --manual_save
```

Add `--real_arm` to drive the real arms alongside (`--real_arm_dataset <dir>` also records the real cameras).

Convert an HDF5 to LeRobot (old converter):

```bash
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
python -u scripts/tools/convert_hdf5_to_lerobot.py \
  --hdf5 logs/demos/pickup_pringle.hdf5 \
  --output ~/Stanley_ws/IsaacLab/datasets/ethanCSL/openarm_visuomotor_VR_pringles_test \
  --task "Pick up the Pringles can with the right arm, hand it to the left arm" \
  --fps 30 --cameras right_wrist_cam wrist_cam body_cam
```

Evaluate a SmolVLA policy inside Isaac Lab (old eval loop):

```bash
conda activate lerobot && cd ~/Stanley_ws/IsaacLab
python scripts/imitation_learning/lerobot/smolvla_server.py \
  --checkpoint ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz \
  --task "Pick up the Pringles can with the right arm, hand it to the left arm." --port 5556
```

```bash
cd ~/Stanley_ws/IsaacLab && conda activate env_isaaclab
./isaaclab.sh -p scripts/imitation_learning/lerobot/eval_smolvla_jointspace.py \
  --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 --task_mode handover \
  --num_rollouts 5 --horizon 300 --enable_cameras --cameras right_wrist_cam,wrist_cam,body_cam
```

</details>
