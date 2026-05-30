# Dining Cleanup Advanced Project

本文件整理 advanced project 中 `Dining Cleanup` 任務的設計與執行方式，涵蓋三個主要部分：

1. FSM 軌跡生成
2. object pose 生成
3. object pose 與桌面佔據區域視覺化

任務目標是將原本的刀叉擺放任務延伸為「用餐完畢後的餐桌收拾與清潔」任務。機器手臂需要先把 bowl 與 spoon 收到 tray，接著拿起 cloth 並擦拭原本 bowl/spoon 所在的左半桌面區域。

## 相關檔案

| 功能 | 檔案 |
|------|------|
| Gym task 註冊 | `packages/simulator/src/simulator/tasks/dining_cleanup/__init__.py` |
| task/env/scene config | `packages/simulator/src/simulator/tasks/dining_cleanup/dining_cleanup_env_cfg.py` |
| FSM | `packages/simulator/src/simulator/datagen/state_machine/dining_cleanup.py` |
| datagen registry | `scripts/datagen/generate.py` |
| object pose generator | `scripts/generate_dining_cleanup_object_poses.py` |
| object pose visualization | `scripts/visualize_dining_cleanup_layout.py` |
| 預設 500 筆 pose | `data/dining_clean/dining_cleanup_object_poses_500.json` |
| 目前視覺化輸出 | `data/dining_clean/dining_cleanup_layout_xy.png` |

Gym task id：

```bash
HCIS-DiningCleanup-SingleArm-v0
```

## Scene 與物件配置設計

### 為什麼沒有直接修改 `scene.usd`

目前 dining room 的 `scene.usd` 是 binary USD crate，不適合直接用文字 patch 修改。專案既有的做法是保留背景 scene，再在 task env config 中用 `RigidObjectCfg` 掛上可互動物件。

因此本任務的做法是：

- `scene.usd` 保持作為 dining room/table 背景。
- bowl、spoon、tray、tissue、vase、cloth 在 `DiningCleanupSceneCfg` 中定義為 `RigidObjectCfg`。
- bowl/spoon 初始位置由 object pose JSON 控制。
- tray、tissue、vase、cloth 在 env cfg 中固定初始位置。

這樣可以避免破壞原本 cutlery task，也比較容易重複生成資料。

### 使用的 USD assets

所有物件 USD 都位於：

```text
packages/simulator/assets/scenes/dining_room/objects/
```

本任務使用：

```text
bowl/model_BalandaBowl_69323.usd
spoon/model_Kitchen_Spoon_B008H2JLP8_LargeWooden_69323.usd
tray/model_WhiteUtensilTray_69323.usd
tissue/model_tissue_001_69323.usd
vase/model_BlackVaseSmall_1_69323.usd
cloth/model_tablecloth.usd
```

### 桌面座標系

本任務沿用 dining room task 的世界座標設定。

桌面 world XY footprint 約為：

```text
x = [0.00, 0.70]
y = [-0.65, 0.00]
```

Franka 位於桌子前方。本 advanced task 採用目前需求指定的 convention：

- `+x` 方向是 Franka 視角的右手邊，也就是 tray 放置區。
- `-x` 方向是 Franka 視角的左手邊，也就是 dirty area。

### 固定物件位置

定義於：

```text
packages/simulator/src/simulator/tasks/dining_cleanup/dining_cleanup_env_cfg.py
```

目前固定位置：

| 物件 | world position |
|------|----------------|
| tray | `(0.57, -0.36, 0.05)` |
| tissue | `(0.35, -0.18, 0.074)` |
| vase | `(0.35, -0.32, 0.05)` |
| cloth | `(0.35, -0.49, 0.05)` |

中間區域的 y 順序符合需求：

```text
tissue: y = -0.18
vase:   y = -0.32
cloth:  y = -0.49
```

也就是從 y 大到小依序為 tissue、vase、cloth。

### bowl/spoon 初始區域

bowl/spoon 由 `object_poses` 隨 episode 隨機生成。generator 中的 world XY 範圍為：

```text
bowl x = [0.10, 0.16]
bowl y = [-0.30, -0.24]
spoon x = [0.11, 0.15]
spoon y = [-0.50, -0.46]
```

這些位置位於桌面左半側，採用較集中的兩個 cluster。bowl/spoon 仍會依 scaled footprint 半徑做 rejection sampling，避免彼此或固定物件重疊。

額外限制：

- bowl/spoon 與 tray/tissue/vase/cloth 依 scaled footprint 半徑避免重疊。
- bowl 與 spoon 之間的最小中心距離目前為 `0.195 m`，等於 bowl 半徑 `0.080` + spoon 半徑 `0.100` + clearance `0.015`。
- 只生成 `status == "full"` 的 episode

### 物件大小與 footprint 設定

更新後的 USD asset 已可用 USD API 讀到 mesh bounding box。原始尺寸中，cloth asset 是桌巾大小，raw footprint 約 `1.641 x 0.984 m`，比桌面還大；如果不縮放會直接超出桌面。因此 task config 會對部分物件套用 spawn scale，並以縮放後 footprint 做：

- object pose 生成時避免重疊
- visualization 中顯示桌面佔據大小
- tray success zone 估計

Raw USD bbox 與 task scale：

| 物件 | raw USD bbox size | task spawn scale | scaled XY footprint |
|------|-------------------|------------------|---------------------|
| bowl | `0.280 x 0.280 x 0.130 m` | `(0.57, 0.57, 0.57)` | `0.160 x 0.160 m` |
| spoon | `0.066 x 0.323 x 0.032 m` | `(0.62, 0.62, 0.62)` | `0.041 x 0.200 m` |
| tray | `0.304 x 0.147 x 0.054 m` | `(0.79, 1.77, 1.00)` | `0.240 x 0.260 m` |
| tissue | `0.073 x 0.103 x 0.050 m` | `(1.00, 1.00, 1.00)` | `0.073 x 0.103 m` |
| vase | `0.100 x 0.100 x 0.114 m` | `(1.00, 1.00, 1.00)` | `0.100 x 0.100 m` |
| cloth | `1.641 x 0.984 x 0.000 m` | `(0.10, 0.10, 1.00)` | `0.164 x 0.098 m` |

注意：cloth 目前 mesh bbox 的 z 厚度為 `0.000 m`，視覺化與 XY 避碰沒有問題；如果 PhysX collision 發生不穩定，應改成有薄厚度的 cloth/collider USD。

## Object Pose 生成

### 腳本

```text
scripts/generate_dining_cleanup_object_poses.py
```

### 輸出檔案

預設輸出：

```text
data/dining_clean/dining_cleanup_object_poses_500.json
```

### 生成指令

在 repo root 執行：

```bash
cd /home/weichen/AI_capstone/aicapstone

python3 scripts/generate_dining_cleanup_object_poses.py \
  --count 500 \
  --output data/dining_clean/dining_cleanup_object_poses_500.json
```

可指定 random seed：

```bash
python3 scripts/generate_dining_cleanup_object_poses.py \
  --count 500 \
  --seed 2026053001 \
  --output data/dining_clean/dining_cleanup_object_poses_500.json
```

### object_poses schema

產生的 JSON 採用專案既有 UMI-style per-episode schema：

```json
[
    {
        "video_name": "synthetic_dining_cleanup_poses.mp4",
        "episode_range": [0, 1309],
        "objects": [
            {
                "object_name": "bowl",
                "rvec": [0.0, 0.0, -1.242],
                "tvec": [0.120, -0.447, 0.048]
            },
            {
                "object_name": "spoon",
                "rvec": [0.0, 0.0, -1.179],
                "tvec": [0.235, -0.360, 0.054]
            }
        ],
        "status": "full"
    }
]
```

注意：

- `tvec` 是 raw anchor-frame pose。
- simulator loader 會用 `ANCHOR_WORLD_POSE = (0.40, 0.10, 0.0)` 轉成 world XY。
- 因此 `world_xy = raw_tvec_xy + (0.40, 0.10)`。
- 本 task 的 object pose 只控制 `bowl` 與 `spoon`。
- `tray`、`tissue`、`vase`、`cloth` 是固定 scene object，不在 object pose JSON 中隨 episode 變動。

### 目前 500 筆統計

目前已產生的 `data/dining_clean/dining_cleanup_object_poses_500.json` 統計如下：

```text
episodes = 500
bowl world x = [0.100, 0.160]
bowl world y = [-0.300, -0.240]
spoon world x = [0.110, 0.150]
spoon world y = [-0.500, -0.460]
bowl-spoon world XY distance = [0.195, 0.259]
scaled footprint clearance min = 0.195
```

### 驗證 object pose loader

可用以下方式確認 object pose 能被 loader 讀取：

```bash
PYTHONPATH=/home/weichen/AI_capstone/aicapstone/packages/simulator/src \
python3 - <<'PY'
from pathlib import Path
from simulator.utils.object_poses_loader import ObjectPoseConfig, load_episode_poses

cfg = ObjectPoseConfig(
    tag_to_object={1: "bowl", 2: "spoon"},
    anchor_tag_id=0,
    anchor_world_pose=(0.40, 0.10, 0.0),
    object_z=0.05,
    object_roll=0.0,
    object_pitch=0.0,
    per_object_yaw_offset={"bowl": 0.0, "spoon": 1.5707963267948966},
    use_fixed_yaw=False,
)

episodes = load_episode_poses(Path("data/dining_clean/dining_cleanup_object_poses_500.json"), cfg)
print(len(episodes))
print(sorted(episodes[0]))
PY
```

預期輸出：

```text
500
['bowl', 'spoon']
```

## Object Pose 視覺化

### 腳本

```text
scripts/visualize_dining_cleanup_layout.py
```

### 輸出檔案

預設輸出：

```text
data/dining_clean/dining_cleanup_layout_xy.png
```

### 產生圖檔

```bash
cd /home/weichen/AI_capstone/aicapstone

python3 scripts/visualize_dining_cleanup_layout.py \
  --input data/dining_clean/dining_cleanup_object_poses_500.json \
  --output data/dining_clean/dining_cleanup_layout_xy.png
```

### 圖中包含的資訊

視覺化圖使用 world XY 俯視座標，包含：

- table footprint
- 左右半桌分界線
- bowl 的 500 筆初始位置
- spoon 的 500 筆初始位置
- bowl/spoon 初始位置 bounding box
- bowl/spoon scaled footprint envelope
- tray 中心
- tray success zone
- bowl drop target，也就是 tray center 的 `+y`
- spoon drop target，也就是 tray center 的 `-y`
- tissue/vase/cloth 固定位置
- tray/tissue/vase/cloth scaled footprint
- tray/tissue/vase/cloth keep-out radius
- cloth wipe coverage region
- cloth wipe lane path

圖中的虛線圓圈是 generator 的 keep-out radius，加上 `0.015 m` clearance 後用來做避碰檢查；它不是物件的實際外形。實際桌面佔據大小以半透明矩形表示。

### 目前視覺化統計

執行 visualization 後會印出：

```text
bowl: n=500, x=[0.100, 0.160], y=[-0.300, -0.240]
spoon: n=500, x=[0.110, 0.150], y=[-0.500, -0.460]
table: x=[0.000, 0.700], y=[-0.650, 0.000]
wipe region: x=[0.040, 0.220], y=[-0.500, -0.150]
planned cloth/table coverage: 100.0%
tray success zone: x=[0.450, 0.690], y=[-0.490, -0.230]
tray scaled footprint: 0.240 x 0.260 m
tissue scaled footprint: 0.073 x 0.103 m
vase scaled footprint: 0.100 x 0.100 m
cloth scaled footprint: 0.164 x 0.098 m
bowl scaled footprint: 0.160 x 0.160 m
spoon scaled footprint: 0.041 x 0.200 m
```

## FSM 設計

### 腳本

```text
packages/simulator/src/simulator/datagen/state_machine/dining_cleanup.py
```

FSM class：

```python
DiningCleanupStateMachine
```

### 任務流程

FSM 依序執行三個大段落：

1. 將 bowl 放入 tray
2. 將 spoon 放入 tray
3. 拿起 cloth 並擦拭左半桌面

完整高階流程：

```text
bowl:
  move above bowl
  approach bowl edge
  close gripper
  lift bowl
  move above tray +y target
  lower
  release

spoon:
  move above spoon
  approach spoon
  close gripper
  lift spoon
  move above tray -y target
  lower
  release

cloth:
  move above cloth
  approach cloth
  close gripper
  lift cloth
  move above wipe start
  lower to wipe contact height
  sweep left table along y-axis lanes
  lift at final endpoint
```

### Bowl 夾取設計

需求中指定 bowl 要夾取邊緣，因此 FSM 不直接抓 bowl center。做法是將 grasp target 從 bowl center 往 robot base 方向 retreat：

```text
_GRASP_RETREAT_PER_OBJECT["bowl"] = 0.055
```

這會讓 gripper 目標點偏向 bowl 靠近 robot 的邊緣，而不是 bowl 幾何中心。

### Spoon 夾取設計

spoon 採用較小 retreat：

```text
_GRASP_RETREAT_PER_OBJECT["spoon"] = 0.020
```

spoon 的 grasp yaw 會根據 spoon object yaw，加上 `pi/2` 與小範圍 random yaw offset。

### Cloth 夾取與擦拭設計

cloth 夾取目標使用 cloth root / 幾何中心：

```text
_GRASP_RETREAT_PER_OBJECT["cloth"] = 0.000
_grasp_anchor_w("cloth") = cloth center
```

因此手臂在 approach 與 close gripper 時會對準抹布中心，而不是邊緣。

cloth 固定起始位置：

```text
cloth = (0.35, -0.49, 0.05)
```

擦拭區域：

```text
x = [0.04, 0.22]
y = [-0.50, -0.15]
```

擦拭採用 3 條 y-axis lanes，並以 cloth 的 scaled footprint `0.164 x 0.098 m` 規劃安全距離：

```text
x lanes = [0.08, 0.135, 0.19]
```

每條 lane 沿 y 方向掃過 dirty area。相鄰 lane 採用往返方式移動，避免每一條都需要回到同一端點：

```text
lane 0: y low  -> y high
lane 1: y high -> y low
lane 2: y low  -> y high
```

最右 lane 的 cloth 右緣約為 `0.272 m`，低於 vase/tissue 左側安全界線，因此擦拭時不會掃到 tissue 或 vase。以 cloth swept footprint 計算，目標擦拭區 `x=[0.04, 0.22]`, `y=[-0.50, -0.15]` 的 planned coverage 為 `100.0%`。FSM 與 env success 目前使用 `90%` coverage threshold。

### FSM phase 與 duration

bowl 與 spoon 各 7 個 phase：

| Phase | 說明 | steps |
|-------|------|-------|
| move_above_object | 移動到物件上方 | 180 |
| approach_object | 下降到抓取高度 | 130 |
| grasp_object | 關閉 gripper | 25 |
| lift_object | 抬起物件 | 100 |
| move_above_drop | 移到 tray drop target 上方 | 170 |
| lower_to_release | 下降到釋放高度 | 25 |
| retreat_from_drop | 開爪並上移 | 40 |

單一物件共：

```text
670 steps
```

bowl + spoon 共：

```text
1340 steps
```

cloth pick 與 wipe 前置 phase：

| Phase | 說明 | steps |
|-------|------|-------|
| move_above_object | 移動到 cloth 上方 | 160 |
| approach_object | 下降到 cloth | 110 |
| grasp_object | 關閉 gripper | 30 |
| lift_object | 抬起 cloth | 90 |
| move_above_wipe_start | 移動到擦拭起點上方 | 140 |
| lower_to_wipe | 下降到擦拭高度 | 80 |

擦拭 phase：

| Phase | 數量 | 每段 steps | 合計 |
|-------|------|------------|------|
| wipe_sweep | 3 | 150 | 450 |
| wipe_shift | 2 | 60 | 120 |
| wipe_lift_finish | 1 | 80 | 80 |

整個 FSM 預估長度：

```text
1340 + 610 + 650 = 2600 steps
MAX_STEPS = 2700
```

### Gripper command

FSM 的 action vector 是：

```text
[panda_joint1, ..., panda_joint7, gripper]
```

gripper command：

```text
open  =  1.0
close = -1.0
```

### IK 控制

FSM 以 world-space end-effector target pose 規劃，再轉成 Franka 7 軸 joint target。

控制流程：

1. 計算 target EE pose。
2. 將 target pose 轉到 robot root frame。
3. 限制每步最大位置變化：

```text
_MAX_CARTESIAN_DELTA = 0.018
```

4. 限制每步最大旋轉變化：

```text
_MAX_ROT_DELTA = 0.08
```

5. 使用 damped least-squares Jacobian IK：

```text
_IK_DLS_LAMBDA = 0.01
```

6. 輸出 joint target + gripper command。

### FSM success check

`DiningCleanupStateMachine.check_success(env)` 會檢查：

1. FSM 已完成完整 wipe timeline。
2. bowl 在 tray success zone 內。
3. spoon 在 tray success zone 內。
4. bowl 位於 tray center 的 `+y` 側。
5. spoon 位於 tray center 的 `-y` 側。
6. bowl/spoon 不重疊，距離至少 `0.04 m`。
7. planned cloth/table coverage ratio 至少 `90%`。
8. tissue/vase 的 XY 位移不超過 `0.035 m`，避免擦拭過程撞偏固定物件。

tray success zone：

```text
x half width = 0.12
y half width = 0.13
z range = [-0.05, 0.11] relative to tray z
```

## Datagen 執行方式

### 1. 產生 object poses

```bash
cd /home/weichen/AI_capstone/aicapstone

python3 scripts/generate_dining_cleanup_object_poses.py \
  --count 500 \
  --output data/dining_clean/dining_cleanup_object_poses_500.json
```

### 2. 視覺化確認物件範圍

```bash
python3 scripts/visualize_dining_cleanup_layout.py \
  --input data/dining_clean/dining_cleanup_object_poses_500.json \
  --output data/dining_clean/dining_cleanup_layout_xy.png
```

確認圖檔：

```text
data/dining_clean/dining_cleanup_layout_xy.png
```

### 3. 執行 datagen

需在 Isaac Lab 可用的 GPU 環境中執行。

基本 HDF5 輸出：

```bash
python scripts/datagen/generate.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --num_envs 1 \
  --device cuda \
  --object_poses data/dining_clean/dining_cleanup_object_poses_500.json \
  --record \
  --dataset_file ./datasets/dining_cleanup.hdf5
```

若需要 camera observation，加入：

```bash
--enable_cameras
```

完整範例：

```bash
python scripts/datagen/generate.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --object_poses data/dining_clean/dining_cleanup_object_poses_500.json \
  --record \
  --dataset_file ./datasets/dining_cleanup.hdf5
```

### 4. LeRobot recorder 輸出

若要直接輸出 LeRobot dataset：

```bash
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
```

## 快速檢查指令

### Python syntax check

```bash
python3 -m py_compile \
  packages/simulator/src/simulator/tasks/dining_cleanup/dining_cleanup_env_cfg.py \
  packages/simulator/src/simulator/tasks/dining_cleanup/__init__.py \
  packages/simulator/src/simulator/datagen/state_machine/dining_cleanup.py \
  scripts/generate_dining_cleanup_object_poses.py \
  scripts/visualize_dining_cleanup_layout.py \
  scripts/datagen/generate.py
```

### object pose 基本檢查

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("data/dining_clean/dining_cleanup_object_poses_500.json")
d = json.loads(p.read_text())

print("count", len(d))
print("full", sum(1 for e in d if e.get("status") == "full"))
print("contiguous", all(d[i]["episode_range"][1] == d[i + 1]["episode_range"][0] for i in range(len(d) - 1)))
print("object_sets", sorted({tuple(sorted(o["object_name"] for o in e["objects"])) for e in d}))
PY
```

預期：

```text
count 500
full 500
contiguous True
object_sets [('bowl', 'spoon')]
```

## 目前設計限制與後續可改進處

1. `scene.usd` 目前沒有直接修改，而是透過 env cfg 加入物件。若之後需要 permanently baked scene，可以再用 USD tooling 另存一個 dining cleanup 專用 USD。
2. env termination 會用 coarse XY grid 累積 cloth/table coverage；這是剛體 cloth footprint 的幾何覆蓋估計，不是 deformable cloth 真實接觸面積。
3. cloth 是以 rigid object 方式處理，不是 deformable cloth simulation。
4. tray/tissue/vase/cloth 目前固定位置。若之後要隨機化中間物件順序或 tray 位置，需要擴充 object pose loader 或新增 task-specific pose schema。
5. bowl edge grasp 的偏移量 `_GRASP_RETREAT_PER_OBJECT["bowl"] = 0.055` 可能需要依 bowl USD mesh 實際大小微調。

## 建議工作流程

每次改動 dining cleanup task 後，建議依序執行：

```bash
# 1. 語法檢查
python3 -m py_compile \
  packages/simulator/src/simulator/tasks/dining_cleanup/dining_cleanup_env_cfg.py \
  packages/simulator/src/simulator/datagen/state_machine/dining_cleanup.py \
  scripts/generate_dining_cleanup_object_poses.py \
  scripts/visualize_dining_cleanup_layout.py

# 2. 重新產生 poses
python3 scripts/generate_dining_cleanup_object_poses.py \
  --count 500 \
  --output data/dining_clean/dining_cleanup_object_poses_500.json

# 3. 視覺化檢查
python3 scripts/visualize_dining_cleanup_layout.py \
  --input data/dining_clean/dining_cleanup_object_poses_500.json \
  --output data/dining_clean/dining_cleanup_layout_xy.png

# 4. 在 Isaac Lab GPU 環境跑 datagen
python scripts/datagen/generate.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --object_poses data/dining_clean/dining_cleanup_object_poses_500.json \
  --record \
  --dataset_file ./datasets/dining_cleanup.hdf5
```
