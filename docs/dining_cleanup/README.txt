Dining Cleanup Advanced-Level Reproduction Guide

Task ID:
HCIS-DiningCleanup-SingleArm-v0

1. Generate the fixed evaluation object-pose split:

python3 scripts/generate_dining_cleanup_object_poses.py \
  --count 500 \
  --seed 2026053002 \
  --output data/dining_clean/dining_cleanup_object_poses_500.json

2. Visualize and inspect the tabletop layout:

python3 scripts/visualize_dining_cleanup_layout.py \
  --input data/dining_clean/dining_cleanup_object_poses_500.json \
  --output data/dining_clean/dining_cleanup_layout_xy.png

3. Generate scripted demonstrations in Isaac Lab:

python scripts/datagen/generate.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --object_poses data/dining_clean/dining_cleanup_object_poses_500.json \
  --record \
  --dataset_file ./datasets/dining_cleanup.hdf5

4. Generate a LeRobot-format dataset:

python scripts/datagen/generate.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --object_poses data/dining_clean/dining_cleanup_object_poses_500.json \
  --record \
  --use_lerobot_recorder \
  --lerobot_dataset_repo_id ${HF_USER}/dining-cleanup \
  --lerobot_dataset_fps 30 \
  --dataset_file ./datasets/dining_cleanup.hdf5

5. Evaluate a trained LeRobot policy:

python scripts/rollout.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --policy_type=lerobot-<policy_name> \
  --policy_checkpoint_path=<path/to/checkpoint> \
  --policy_action_horizon=1 \
  --device=cuda \
  --enable_cameras \
  --object_poses data/dining_clean/dining_cleanup_object_poses_500.json \
  --eval_rounds=50 \
  --episode_length_s=60 \
  --seed=2026053002

Evaluation success criteria:
- Bowl and spoon XY positions are both inside the relaxed tray success region.
- Bowl and spoon z positions are not checked for tray placement success.
- Bowl and spoon no longer need to be on fixed +y/-y sides of the tray.
- Actual cloth/table coverage must reach 70% of the ideal planned coverage.
- Tissue and vase XY displacement is at most 0.035 m.

Episode-end terminal reports:
- Datagen prints DiningCleanup FSM stage status for every replayed episode.
- Rollout prints DiningCleanup Eval stage status for every successful or timed-out episode.
- Each report includes tableware, wiping, protected-object success/fail states and the cloth coverage ratio.

Additional documentation:
docs/dining_cleanup/README.md
