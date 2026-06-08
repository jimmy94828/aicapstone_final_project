# Dining Cleanup Dataset Generation

本文件整理 `Dining Cleanup` advanced task 的 dataset 生成方式，重點是產生不同的
`object_poses.json`，再用同一份 pose split 進行 FSM datagen、policy training 和 rollout
evaluation。

## 1. 相關檔案

| 功能 | 檔案 |
|------|------|
| pose generator | `scripts/generate_dining_cleanup_object_poses.py` |
| layout visualization | `scripts/visualize_dining_cleanup_layout.py` |
| FSM datagen | `scripts/datagen/generate.py` |
| policy rollout evaluation | `scripts/rollout.py` |
| task/env config | `packages/simulator/src/simulator/tasks/dining_cleanup/dining_cleanup_env_cfg.py` |
| FSM | `packages/simulator/src/simulator/datagen/state_machine/dining_cleanup.py` |

Task id:

```bash
HCIS-DiningCleanup-SingleArm-v0
```

## 2. Dataset 生成核心概念

目前 dining cleanup 的 dataset 生成分成兩層：

1. 先產生 `object_poses.json`
2. 再用 `object_poses.json` 跑 datagen 或 rollout

`object_poses.json` 只控制每個 episode 的 `bowl` 和 `spoon` 初始 pose。
`tray`、`tissue`、`vase`、`cloth` 目前固定在 env config 中，不會隨 dataset episode 改變。

目前 loader 會使用 env config 中的固定 `OBJECT_Z = 0.05`，因此 `object_poses.json` 裡的
`tvec[2]` 不會改變 bowl/spoon 的實際高度。也就是說：

- 支援 bowl/spoon 的 XY 位置變化。
- 支援 bowl/spoon 的 yaw 角度變化。
- 支援 spoon 在 top-down XY 平面上覆蓋 bowl。
- 尚未支援 spoon 物理上疊在 bowl 上方的不同 z 高度。

如果需要真正讓 spoon 物理疊在 bowl 上，需要另外修改 object pose loader 或 env config，讓
spoon 支援 per-object z，並可能需要調整 FSM 抓取順序。

## 3. Generator 參數

執行目錄固定使用 repo root：

```bash
cd /home/weichen/AI_capstone/aicapstone_final_project
```

主要參數：

| 參數 | 說明 |
|------|------|
| `--count` | 產生 episode 數量 |
| `--seed` | 固定 random seed，方便重現 |
| `--output` | 輸出的 `object_poses.json` |
| `--video-name` | 寫入 JSON 的 synthetic video name metadata |
| `--object-world-x-range` | bowl 的 world x 生成範圍；若沒有指定 spoon-only range，也會套用到 spoon |
| `--object-world-y-range` | bowl 的 world y 生成範圍；若沒有指定 spoon-only range，也會套用到 spoon |
| `--spoon-world-x-range` | spoon 專用 world x 生成範圍 |
| `--spoon-world-y-range` | spoon 專用 world y 生成範圍 |
| `--spoon-yaw-mode fixed` | 使用原本固定 spoon yaw |
| `--spoon-yaw-mode random` | 每個 episode 隨機 spoon yaw |
| `--spoon-placement-mode relative` | spoon 依 bowl 周圍距離採樣，原本預設模式 |
| `--spoon-placement-mode independent` | bowl 放好後，spoon 在指定範圍內獨立隨機採樣 |
| `--allow-bowl-spoon-overlap` | 允許 bowl/spoon 的 XY footprint overlap |
| `--min-clearance` | 物件 footprint 半徑以外的最小安全距離 |
| `--max-pair-distance` | relative mode 下 bowl/spoon 中心最大距離 |

## 4. Dataset A：不限制 spoon 朝向

目標：

- 產生 200 筆 episode。
- bowl/spoon 位置仍使用原本 relative placement。
- bowl/spoon 不允許 overlap。
- spoon yaw 每筆隨機。

生成指令：

```bash
python3 scripts/generate_dining_cleanup_object_poses.py \
  --count 200 \
  --seed 2026060801 \
  --spoon-yaw-mode random \
  --output data/dining_clean/dining_cleanup_spoon_random_yaw_200.json \
  --video-name synthetic_dining_cleanup_spoon_random_yaw_200.mp4
```

目前產出的統計：

```text
episodes=200
bowl world x=[0.100, 0.195], y=[-0.500, -0.221]
spoon world x=[0.100, 0.162], y=[-0.500, -0.220]
bowl-spoon world XY distance=[0.207, 0.278], mean=0.232
spoon world yaw deg=[-178.7, 179.6], mean=9.6
spoon placement mode=relative
allow bowl-spoon XY overlap=False
```

對應輸出：

```text
data/dining_clean/dining_cleanup_spoon_random_yaw_200.json
```

## 5. Dataset B：spoon 隨機朝向 + 獨立位置 + 可覆蓋 bowl

目標：

- 產生 200 筆 episode。
- spoon yaw 每筆隨機。
- bowl 先產生，spoon 再於範圍內獨立隨機產生。
- 允許 spoon 在 top-down XY 平面上覆蓋 bowl。

生成指令：

```bash
python3 scripts/generate_dining_cleanup_object_poses.py \
  --count 200 \
  --seed 2026060802 \
  --spoon-yaw-mode random \
  --spoon-placement-mode independent \
  --allow-bowl-spoon-overlap \
  --output data/dining_clean/dining_cleanup_spoon_random_yaw_independent_overlap_200.json \
  --video-name synthetic_dining_cleanup_spoon_random_yaw_independent_overlap_200.mp4
```

目前產出的統計：

```text
episodes=200
bowl world x=[0.100, 0.203], y=[-0.499, -0.220]
spoon world x=[0.100, 0.175], y=[-0.500, -0.222]
bowl-spoon world XY distance=[0.001, 0.258], mean=0.096
spoon world yaw deg=[-178.5, 178.5], mean=7.1
spoon placement mode=independent
allow bowl-spoon XY overlap=True
```

`bowl-spoon world XY distance` 最小值接近 `0.0 m`，代表此 dataset 確實允許 spoon
在 XY 平面上覆蓋 bowl。

對應輸出：

```text
data/dining_clean/dining_cleanup_spoon_random_yaw_independent_overlap_200.json
```

## 6. 產生 layout 圖

Dataset A:

```bash
python3 scripts/visualize_dining_cleanup_layout.py \
  --input data/dining_clean/dining_cleanup_spoon_random_yaw_200.json \
  --output data/dining_clean/dining_cleanup_spoon_random_yaw_200_layout_xy.png
```

Dataset B:

```bash
python3 scripts/visualize_dining_cleanup_layout.py \
  --input data/dining_clean/dining_cleanup_spoon_random_yaw_independent_overlap_200.json \
  --output data/dining_clean/dining_cleanup_spoon_random_yaw_independent_overlap_200_layout_xy.png
```

目前輸出圖檔：

```text
data/dining_clean/dining_cleanup_spoon_random_yaw_200_layout_xy.png
data/dining_clean/dining_cleanup_spoon_random_yaw_independent_overlap_200_layout_xy.png
```

圖中會顯示：

- table footprint
- dirty region
- tray/tissue/vase/cloth 固定佔據位置
- bowl/spoon sampled occupied envelope
- expected cloth path

## 7. 使用 dataset 跑 FSM datagen

`scripts/datagen/generate.py` 的 episode 數量由 `--object_poses` 中
`status == "full"` 的 entries 決定。因此 200 筆 pose JSON 會對應最多 200 個 FSM episode。

Dataset A:

```bash
python scripts/datagen/generate.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --object_poses data/dining_clean/dining_cleanup_spoon_random_yaw_200.json \
  --record \
  --dataset_file ./datasets/dining_cleanup_spoon_random_yaw_200.hdf5
```

Dataset B:

```bash
python scripts/datagen/generate.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --object_poses data/dining_clean/dining_cleanup_spoon_random_yaw_independent_overlap_200.json \
  --record \
  --dataset_file ./datasets/dining_cleanup_spoon_random_yaw_independent_overlap_200.hdf5
```

若要輸出 LeRobot dataset，可額外加入：

```bash
--use_lerobot_recorder \
--lerobot_dataset_repo_id ${HF_USER}/dining-cleanup-spoon-random-yaw \
--lerobot_dataset_fps 30
```

## 8. 使用 dataset 跑 rollout evaluation

rollout evaluation 只需要替換 `--object_poses`。

Dataset A:

```bash
python scripts/rollout.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --policy_type=lerobot-<policy_name> \
  --policy_checkpoint_path=<path/to/checkpoint> \
  --policy_action_horizon=1 \
  --device=cuda \
  --enable_cameras \
  --object_poses data/dining_clean/dining_cleanup_spoon_random_yaw_200.json \
  --eval_rounds=50 \
  --episode_length_s=60 \
  --seed=2026060801
```

Dataset B:

```bash
python scripts/rollout.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --policy_type=lerobot-<policy_name> \
  --policy_checkpoint_path=<path/to/checkpoint> \
  --policy_action_horizon=1 \
  --device=cuda \
  --enable_cameras \
  --object_poses data/dining_clean/dining_cleanup_spoon_random_yaw_independent_overlap_200.json \
  --eval_rounds=50 \
  --episode_length_s=60 \
  --seed=2026060802
```

## 9. 是否需要修改 rollout 或 FSM

目前這兩種 dataset 不需要修改 `rollout.py` 或 `scripts/datagen/generate.py`。
兩個程式都只是讀取 `--object_poses`，並在 reset 後把 episode pose 寫入 simulator。

FSM 是否需要修改取決於實驗目的：

| Dataset | 目前是否需要改 FSM | 原因 |
|---------|---------------------|------|
| Dataset A: random spoon yaw | 不需要 | FSM 會讀 spoon 當前 yaw，並據此設定 grasp orientation |
| Dataset B: random yaw + independent overlap | 可先不改 | 可作為較困難的 robustness dataset；但若 spoon 擋住 bowl，原本先 bowl 後 spoon 的順序可能降低成功率 |
| 真正 spoon 疊在 bowl 上方，不同 z 高度 | 需要 | 目前 loader 忽略 `tvec[2]`，需要支援 per-object z，FSM 也可能要改成先拿 spoon 再拿 bowl |

建議實驗順序：

1. 先用 Dataset A 訓練或評估，確認 random spoon yaw 的影響。
2. 再用 Dataset B 做 robustness evaluation。
3. 如果 Dataset B 失敗率高，先觀察失敗原因是 yaw grasp 還是 spoon/bowl overlap。
4. 若主要問題是 spoon 壓住 bowl，才考慮新增「先 spoon 後 bowl」的 FSM variant。

## 10. 建議 report 寫法

在 final report 可以把這兩組 dataset 當成 difficulty levels：

| Split | Episodes | Spoon yaw | Spoon position | Bowl-spoon overlap | 用途 |
|-------|----------|-----------|----------------|--------------------|------|
| RandomYaw-200 | 200 | random | relative to bowl | no | 測試對 spoon orientation 的泛化能力 |
| RandomYawOverlap-200 | 200 | random | independent after bowl | yes, XY overlap allowed | 測試 clutter/occlusion robustness |

評估時建議回報：

- overall success rate
- bowl in tray success rate
- spoon in tray success rate
- wiping coverage pass rate
- protected object stability pass rate
- average coverage ratio
- failure case examples

