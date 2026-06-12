Dining Cleanup Advanced-Level Submission README
=============================================

Task ID
-------
LeIsaac-HCIS-DiningCleanup-SingleArm-v0

Legacy compatibility alias:
HCIS-DiningCleanup-SingleArm-v0

Project Summary
---------------
This advanced-level task extends the entry-level tableware manipulation setting
into a multi-stage dining cleanup benchmark. A single Franka arm must:

1. Clear the tableware objects, bowl and spoon, into the tray.
2. Pick up the cloth.
3. Wipe the dirty region on the left side of the table.
4. Avoid disturbing protected objects, tissue and vase.

The task is implemented as a standalone Isaac Lab / LeIsaac manager-based
environment configuration and is registered through Gymnasium as
LeIsaac-HCIS-DiningCleanup-SingleArm-v0. The previous project-local task id
HCIS-DiningCleanup-SingleArm-v0 is also registered as a compatibility alias for
older rollout logs and scripts.

Required Configuration Files
----------------------------
The standalone advanced task configuration is:

packages/simulator/src/simulator/tasks/dining_cleanup/__init__.py
packages/simulator/src/simulator/tasks/dining_cleanup/dining_cleanup_env_cfg.py

The task is imported by:

packages/simulator/src/simulator/tasks/__init__.py

The environment inherits the shared Franka template from:

packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py

The object pose loader used by datagen, teleop, and rollout is:

packages/simulator/src/simulator/utils/object_poses_loader.py

The scripted Dining Cleanup FSM used for dataset generation is provided as a
supplementary supporting file, not as part of the standalone environment
configuration:

Supplementary/packages/simulator/src/simulator/datagen/state_machine/dining_cleanup.py

When reproducing FSM datagen in a clean checkout, merge Supplementary/packages/
into the repository root so the file lands at:

packages/simulator/src/simulator/datagen/state_machine/dining_cleanup.py

Evaluation / Dataset Config Files
---------------------------------
Three Dining Cleanup configs are provided:

configs/dining_cleanup/spoon_fixed_yaw.json
  Original bowl and spoon assets. Spoon yaw is fixed.

configs/dining_cleanup/spoon_random_yaw.json
  Original bowl and spoon assets. Spoon yaw is randomized.

configs/dining_cleanup/fork_bowl2_scaled.json
  Asset-shift evaluation. The scene keys remain bowl and spoon, but the bowl
  asset is replaced by bowl_2 and the spoon asset is replaced by wooden_fork.

The corresponding object pose splits are:

data/dining_clean/dining_cleanup_object_poses_500.json
data/dining_clean/dining_cleanup_spoon_random_yaw_200.json
data/dining_clean/dining_cleanup_fork_bowl2_scaled_200.json

Why these JSON files are inside Configurations:

Configurations/configs/dining_cleanup/ contains the three evaluation variant
configs claimed by this submission. These files select the task id, object USD
assets, object scales, pose-generation metadata, and default object-pose split
for each benchmark variant. They are therefore task/evaluation configuration
files, not supplementary documentation.

Configurations/data/dining_clean/ contains deterministic object-pose splits
referenced directly by the config JSON files through relative paths such as:

"object_poses": "data/dining_clean/dining_cleanup_spoon_random_yaw_200.json"

These pose splits define the initial bowl/spoon layouts used during datagen and
rollout evaluation. Keeping them under Configurations makes the configuration
folder self-contained and prevents the relative object_poses paths from
breaking when the evaluator copies or audits the configuration package. The
same files may be treated as supporting data, but they should not be moved only
to Supplementary unless every config and command path is also changed.

Supplementary is reserved for reproducibility helpers and documentation, such
as FSM source code, datagen/rollout scripts, layout figures, model links, and
extended implementation notes. Those files help reproduce or explain the
results, but they are not the minimal standalone environment/evaluation config
set.

Custom CAD / USD Assets
-----------------------
The Dining Cleanup task uses USD assets under:

packages/simulator/assets/scenes/dining_room/

For a self-contained Advanced-level submission, include at least:

packages/simulator/assets/scenes/dining_room/scene.usd
packages/simulator/assets/scenes/dining_room/objects/bowl/
packages/simulator/assets/scenes/dining_room/objects/spoon/
packages/simulator/assets/scenes/dining_room/objects/tray/
packages/simulator/assets/scenes/dining_room/objects/tissue/
packages/simulator/assets/scenes/dining_room/objects/vase/
packages/simulator/assets/scenes/dining_room/objects/cloth/
packages/simulator/assets/scenes/dining_room/objects/bowl_2/
packages/simulator/assets/scenes/dining_room/objects/wooden_fork/

The cloth used by the environment is represented as a stable CuboidCfg rigid
object in the task config. The cloth USD is included for documentation and
asset completeness, but the benchmark logic uses the cuboid cloth geometry.

Setup
-----
From the repository root:

make build-isaaclab

For an interactive container:

make launch-isaaclab

For GlowsAI, run host-side rollout scripts from the host shell, not from inside
the container prompt. The wrapper script starts its own Docker container.

Generate / Verify Object Pose Splits
------------------------------------
Fixed spoon-yaw split:

python3 scripts/generate_dining_cleanup_object_poses.py \
  --count 500 \
  --seed 2026053002 \
  --output data/dining_clean/dining_cleanup_object_poses_500.json

Visualize a split:

python3 scripts/visualize_dining_cleanup_layout.py \
  --input data/dining_clean/dining_cleanup_spoon_random_yaw_200.json \
  --output data/dining_clean/dining_cleanup_spoon_random_yaw_200_layout_xy.png

Datagen
-------
Run scripted FSM datagen with a specific config:

python scripts/datagen/generate.py \
  --task LeIsaac-HCIS-DiningCleanup-SingleArm-v0 \
  --dining_cleanup_config configs/dining_cleanup/spoon_fixed_yaw.json \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --record \
  --use_lerobot_recorder \
  --lerobot_dataset_repo_id ${HF_USER}/dining-cleanup-fixed-spoon \
  --lerobot_dataset_fps 30 \
  --dataset_file ./datasets/dining_cleanup_fixed_spoon.hdf5

The datagen episode count is determined by the object_poses file referenced by
the selected Dining Cleanup config.

Rollout / Evaluation
--------------------
Direct Python rollout:

python scripts/rollout.py \
  --headless \
  --task LeIsaac-HCIS-DiningCleanup-SingleArm-v0 \
  --dining_cleanup_config configs/dining_cleanup/spoon_fixed_yaw.json \
  --policy_type=lerobot-<policy_name> \
  --policy_checkpoint_path <local_pretrained_model_dir> \
  --policy_action_horizon <horizon> \
  --device=cuda \
  --enable_cameras \
  --show_wipe_mesh \
  --eval_rounds=30 \
  --episode_length_s=80 \
  --seed=2026061011 \
  --record_video \
  --video_dir outputs/rollout_videos \
  --video_fps=30 \
  --progress_interval_s=10

ACT model rollout from Hugging Face:

HF_REPO_ID=AI-Final/advanced-act-v3 \
MODEL_NAME=advanced-act-v3 \
CHECKPOINT=hf \
POLICY_TYPE=lerobot-act \
POLICY_ACTION_HORIZON=25 \
./rollout_advance_act.sh configs/dining_cleanup/spoon_fixed_yaw.json

SmolVLA checkpoint rollout from Hugging Face:

HF_REPO_ID=AI-Final/smolvla-advanced-v3 \
MODEL_NAME=smolvla-advanced-v3 \
CHECKPOINT=040000 \
POLICY_TYPE=lerobot-smolvla \
POLICY_ACTION_HORIZON=16 \
HF_MODEL_SUBDIR=training_runs/smolvla_advanced_v3_20260611_185615/checkpoints/040000 \
MODEL_DIR_REL=experiments/advance/smolvla-advanced-v3/training_runs/smolvla_advanced_v3_20260611_185615/checkpoints/040000/pretrained_model \
./rollout_advance_act.sh configs/dining_cleanup/spoon_fixed_yaw.json

Expected Outputs
----------------
Rollout logs and videos are written under:

experiments/advance/<model_name>/

Example output names:

rollout_040000_spoon_fixed_yaw_<timestamp>.log
rollout_040000_spoon_fixed_yaw_<timestamp>_videos/

Evaluation Success Criteria
---------------------------
An episode is successful when all main task checks pass:

1. Bowl and spoon are inside the tray success region.
2. Wiping coverage reaches at least 70% of the ideal planned dirty-region coverage.
3. Protected objects, tissue and vase, remain within the allowed XY displacement.

The rollout report prints tableware, wiping, protected-object status, and final
success rate. Videos are saved when --record_video is enabled.

Benchmarking Integrity Notes
----------------------------
All environment/task configuration files, object pose splits, and USD assets
needed for reproduction are listed above. No hidden runtime configuration is
required. If using Hugging Face checkpoints, the README commands identify the
exact repository and checkpoint directory used for rollout.

More Documentation
------------------
See docs/dining_cleanup/README.md for the full implementation guide and
docs/dining_cleanup/evaluation_configs.md for the three evaluation configs.
