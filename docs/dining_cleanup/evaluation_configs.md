# Dining Cleanup Evaluation Configs

本文件整理三組可在 datagen、rollout/evaluation 使用的 Dining Cleanup config。

## Config 檔案

| Config | 檔案 | Object pose split | 說明 |
|--------|------|-------------------|------|
| Fixed Spoon Yaw | `configs/dining_cleanup/spoon_fixed_yaw.json` | `data/dining_clean/dining_cleanup_object_poses_500.json` | 原始 bowl/spoon asset，spoon yaw 固定 45 deg |
| Random Spoon Yaw | `configs/dining_cleanup/spoon_random_yaw.json` | `data/dining_clean/dining_cleanup_spoon_random_yaw_200.json` | 原始 bowl/spoon asset，spoon yaw 隨機 |
| Fork + Bowl2 Scaled | `configs/dining_cleanup/fork_bowl2_scaled.json` | `data/dining_clean/dining_cleanup_fork_bowl2_scaled_200.json` | spoon scene key 改用 `wooden_fork` asset，bowl scene key 改用 `bowl_2` asset，套用 scale，fork yaw 固定 45 deg |

注意：第三組 config 中，scene object key 仍然是 `bowl` 和 `spoon`。
這樣現有 FSM、success check、`object_poses.json` loader 不需要改；只是實際載入的 USD asset 變成
`bowl_2` 和 `wooden_fork`。

## 1. Fixed Spoon Yaw

Config:

```text
configs/dining_cleanup/spoon_fixed_yaw.json
```

特性：

- bowl asset: `bowl/model_BalandaBowl_69323.usd`
- spoon asset: `spoon/model_Kitchen_Spoon_B008H2JLP8_LargeWooden_69323.usd`
- bowl scale: `[0.5, 0.5, 0.5]`
- spoon scale: `[0.6, 0.6, 0.6]`
- spoon yaw: fixed `45 deg`
- episodes: 500

## 2. Random Spoon Yaw

Config:

```text
configs/dining_cleanup/spoon_random_yaw.json
```

特性：

- bowl asset: `bowl/model_BalandaBowl_69323.usd`
- spoon asset: `spoon/model_Kitchen_Spoon_B008H2JLP8_LargeWooden_69323.usd`
- bowl scale: `[0.5, 0.5, 0.5]`
- spoon scale: `[0.6, 0.6, 0.6]`
- spoon yaw: random over approximately `[-180, 180] deg`
- episodes: 200

目前 pose split 統計：

```text
spoon yaw=[-178.7, 179.6] deg
unique rounded yaw=148
```

## 3. Fork + Bowl2 Scaled

Config:

```text
configs/dining_cleanup/fork_bowl2_scaled.json
```

特性：

- bowl scene key: `bowl`
- actual bowl asset: `bowl_2/model_B075HWDSDK_69323.usd`
- bowl scale: `[0.52, 0.52, 0.52]`
- spoon scene key: `spoon`
- actual spoon asset: `wooden_fork/model_WoodenFork_69323.usd`
- wooden fork scale: `[0.62, 0.62, 0.62]`
- yaw: fixed `45 deg`
- episodes: 200

目前 pose split 統計：

```text
spoon/fork yaw=[45.0, 45.0] deg
unique rounded yaw=1
```

## Rollout / Evaluation

三組 config 都可以用同一個 rollout command，只需要替換 `--dining_cleanup_config`。

Fixed Spoon Yaw:

```bash
python scripts/rollout.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --dining_cleanup_config configs/dining_cleanup/spoon_fixed_yaw.json \
  --policy_type=lerobot-<policy_name> \
  --policy_checkpoint_path=<path/to/checkpoint> \
  --policy_action_horizon=1 \
  --device=cuda \
  --enable_cameras \
  --eval_rounds=50 \
  --episode_length_s=60 \
  --seed=2026060901
```

Random Spoon Yaw:

```bash
python scripts/rollout.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --dining_cleanup_config configs/dining_cleanup/spoon_random_yaw.json \
  --policy_type=lerobot-<policy_name> \
  --policy_checkpoint_path=<path/to/checkpoint> \
  --policy_action_horizon=1 \
  --device=cuda \
  --enable_cameras \
  --eval_rounds=50 \
  --episode_length_s=60 \
  --seed=2026060902
```

Fork + Bowl2 Scaled:

```bash
python scripts/rollout.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --dining_cleanup_config configs/dining_cleanup/fork_bowl2_scaled.json \
  --policy_type=lerobot-<policy_name> \
  --policy_checkpoint_path=<path/to/checkpoint> \
  --policy_action_horizon=1 \
  --device=cuda \
  --enable_cameras \
  --eval_rounds=50 \
  --episode_length_s=60
```

如果你想臨時覆蓋 config 裡的 pose split，可以額外加：

```bash
--object_poses <path/to/another_object_poses.json>
```

`--object_poses` 的優先權高於 config 裡的 `object_poses`。

## Datagen

Fixed Spoon Yaw:

```bash
python scripts/datagen/generate.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --dining_cleanup_config configs/dining_cleanup/spoon_fixed_yaw.json \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --record \
  --dataset_file ./datasets/dining_cleanup_fixed_spoon_yaw.hdf5
```

Random Spoon Yaw:

```bash
python scripts/datagen/generate.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --dining_cleanup_config configs/dining_cleanup/spoon_random_yaw.json \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --record \
  --dataset_file ./datasets/dining_cleanup_random_spoon_yaw.hdf5
```

Fork + Bowl2 Scaled:

```bash
python scripts/datagen/generate.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --dining_cleanup_config configs/dining_cleanup/fork_bowl2_scaled.json \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --record \
  --dataset_file ./datasets/dining_cleanup_fork_bowl2_scaled.hdf5
```

## 確認是否有套用 config

啟動時 terminal 應該會看到：

```text
[rollout] using Dining Cleanup config: ...
[rollout] object_poses: ...
[DiningCleanupCfg] loaded ...
```

或 datagen：

```text
[datagen] using Dining Cleanup config: ...
[datagen] object_poses: ...
[DiningCleanupCfg] loaded ...
```

每個 reset 後也會印出目前套用的 object pose：

```text
[pose] bowl: pos=(...), yaw=...
[pose] spoon: pos=(...), yaw=...
```
