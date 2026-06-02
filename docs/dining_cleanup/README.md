# Dining Cleanup Advanced Project

本文件整理 advanced project 中 `Dining Cleanup` 任務的設計與執行方式，涵蓋四個主要部分：

1. FSM 軌跡生成
2. object pose 生成
3. object pose 與桌面佔據區域視覺化
4. keyboard teleoperation 檢查與手動錄製

任務目標是將原本的刀叉擺放任務延伸為「用餐完畢後的餐桌收拾與清潔」任務。機器手臂需要先把 bowl 與 spoon 收到 tray，接著拿起 cloth 並擦拭原本 bowl/spoon 所在的左半桌面區域。

## 相關檔案

| 功能 | 檔案 |
|------|------|
| Gym task 註冊 | `packages/simulator/src/simulator/tasks/dining_cleanup/__init__.py` |
| task/env/scene config | `packages/simulator/src/simulator/tasks/dining_cleanup/dining_cleanup_env_cfg.py` |
| FSM | `packages/simulator/src/simulator/datagen/state_machine/dining_cleanup.py` |
| teleoperation | `scripts/teleop.py` |
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

餐具、托盤與固定障礙物 USD 位於：

```text
packages/simulator/assets/scenes/dining_room/objects/
```
Google Drive link: https://drive.google.com/drive/folders/1FHOizW83ahsts2zrd36hD3e29t2Z2hLI?usp=drive_link
本任務使用：

```text
bowl/model_BalandaBowl_69323.usd
spoon/model_Kitchen_Spoon_B008H2JLP8_LargeWooden_69323.usd
tray/model_WhiteUtensilTray_69323.usd
tissue/model_tissue_001_69323.usd
vase/model_B07JLBDT51_69323.usd
```

cloth 目前改用 `sim_utils.CuboidCfg` 建立穩定的薄片 rigid object，避免原本 `model_tablecloth.usd` 的 particle-cloth schema 在 simulator 啟動時造成初始位置漂移。
vase 目前使用 `model_B07JLBDT51_69323.usd` 的原生 mesh；其 local bbox 的 z min 為 `0.000 m`，因此固定位置 `z=0.05` 會讓 vase 底部貼齊桌面。task config 會將 vase scale 到約 `0.100 x 0.100 m` 的桌面 footprint。`dining_cleanup_env_cfg.py` 目前保留 vase USD 內建材質，避免額外 PreviewSurface override 讓整個 vase 被顯示成單一純色。

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
| tissue | `(0.35, -0.12, 0.074)` |
| vase | `(0.35, -0.26, 0.05)` |
| cloth | `(0.35, -0.43, 0.065)` |

中間區域的 y 順序符合需求：

```text
tissue: y = -0.12
vase:   y = -0.26
cloth:  y = -0.43
```

也就是從 y 大到小依序為 tissue、vase、cloth。

### bowl/spoon 初始區域

bowl/spoon 由 `object_poses` 隨 episode 隨機生成。generator 中的 world XY 範圍為：

```text
bowl/spoon shared x = [0.10, 0.24]
bowl/spoon shared y = [-0.50, -0.22]
```

這些位置位於桌面左半側 dirty area。generator 會先隨機放 bowl，再在 bowl 周圍採樣 spoon，使兩者不要過度分離，同時保留多樣性。

額外限制：

- bowl/spoon 與 tray/tissue/vase/cloth 依 scaled footprint 半徑避免重疊。
- bowl 與 spoon 之間的最小中心距離目前為 `0.207 m`，等於 bowl 半徑 `0.070` + spoon 半徑 `0.097` + clearance `0.040`。
- bowl 與 spoon 的最大中心距離目前為 `0.280 m`，避免兩者初始位置過度分離。
- 只生成 `status == "full"` 的 episode

### 物件大小與 footprint 設定

更新後的 USD asset 已可用 USD API 讀到 mesh bounding box。bowl、spoon、tray、tissue、vase 會從 USD 載入並套用 task spawn scale；cloth 則改用薄片 cuboid rigid object，避免原始 particle-cloth USD 在 simulator 啟動時漂移。因此 task config 會以實際 spawn 後的 footprint 做：

- object pose 生成時避免重疊
- visualization 中顯示桌面佔據大小
- tray success zone 估計

USD bbox / task geometry 與 task scale：

| 物件 | raw USD bbox size / task geometry | task spawn scale / rot | scaled world XY footprint |
|------|-------------------|------------------------|---------------------------|
| bowl | `0.280 x 0.280 x 0.130 m` | `(0.50, 0.50, 0.50)` | `0.140 x 0.140 m` |
| spoon | `0.066 x 0.323 x 0.032 m` | `(0.60, 0.60, 0.60)` | `0.040 x 0.194 m` |
| tray | `0.304 x 0.147 x 0.054 m` | `(0.79, 1.77, 1.00)` | `0.240 x 0.260 m` |
| tissue | `0.073 x 0.103 x 0.050 m` | `(1.00, 1.00, 1.00)` | `0.073 x 0.103 m` |
| vase | `0.169 x 0.169 x 0.240 m` | `(0.591, 0.591, 0.591)` | `0.100 x 0.100 m` |
| cloth | `CuboidCfg(size=(0.055, 0.115, 0.030))` | no USD scale / no z yaw | `0.055 x 0.115 m` |

Franka gripper 初始最大開口約為 `0.04 + 0.04 = 0.08 m`。目前 cloth cuboid 的窄邊為 `0.055 m`，可由夾爪夾取窄邊；z 厚度為 `0.030 m`，讓 PhysX 在啟動時有穩定的薄片碰撞幾何，並提高夾取時的接觸厚度。

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
python3 scripts/generate_dining_cleanup_object_poses.py \
  --count 500 \
  --output data/dining_clean/dining_cleanup_object_poses_500.json
```

可指定 random seed：

```bash
python3 scripts/generate_dining_cleanup_object_poses.py \
  --count 500 \
  --seed 2026053002 \
  --output data/dining_clean/dining_cleanup_object_poses_500.json
```

### object_poses schema

產生的 JSON 採用專案既有 UMI-style per-episode schema：

```json
[
    {
        "video_name": "synthetic_dining_cleanup_poses.mp4",
        "episode_range": [0, 1732],
        "objects": [
            {
                "object_name": "bowl",
                "rvec": [0.0, 0.0, 2.909],
                "tvec": [-0.225, -0.561, 0.057]
            },
            {
                "object_name": "spoon",
                "rvec": [0.0, 0.0, 2.356],
                "tvec": [-0.280, -0.350, 0.066]
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
- spoon 的 raw yaw 固定為 `3*pi/4`，加上 env config 的 spoon yaw offset `3*pi/2` 後，最終 world yaw 固定為 `pi/4`，也就是與 `+x` 軸夾 `45 deg`。
- 本 task 的 object pose 只控制 `bowl` 與 `spoon`。
- `tray`、`tissue`、`vase`、`cloth` 是固定 scene object，不在 object pose JSON 中隨 episode 變動。

### 目前 500 筆統計

目前已產生的 `data/dining_clean/dining_cleanup_object_poses_500.json` 統計如下：

```text
episodes = 500
bowl world x = [0.100, 0.197]
bowl world y = [-0.500, -0.220]
spoon world x = [0.100, 0.168]
spoon world y = [-0.500, -0.220]
bowl-spoon world XY distance = [0.207, 0.279]
scaled footprint clearance min = 0.207
configured bowl-spoon max distance = 0.280
```

### 驗證 object pose loader

可用以下方式確認 object pose 能被 loader 讀取：

```bash
PYTHONPATH=./packages/simulator/src \
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
    per_object_yaw_offset={"bowl": 0.0, "spoon": 4.71238898038469},
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
- bowl drop target
- spoon drop target
- tissue/vase/cloth 固定位置
- tray/tissue/vase/cloth scaled footprint
- tray/tissue/vase/cloth keep-out radius
- cloth wipe coverage region
- cloth wipe lane path

圖中的虛線圓圈是 generator 的 keep-out radius，加上 `0.040 m` clearance 後用來做避碰檢查；它不是物件的實際外形。實際桌面佔據大小以半透明矩形表示。

### 目前視覺化統計

執行 visualization 後會印出：

```text
bowl: n=500, x=[0.100, 0.197], y=[-0.500, -0.220]
spoon: n=500, x=[0.100, 0.168], y=[-0.500, -0.220]
table: x=[0.000, 0.700], y=[-0.650, 0.000]
wipe region: x=[0.080, 0.220], y=[-0.550, -0.100]
planned cloth/table coverage: 98.2%
coverage success threshold: 68.7%
tray success zone: x=[0.440, 0.700], y=[-0.500, -0.220]
tray scaled footprint: 0.240 x 0.260 m
tissue scaled footprint: 0.073 x 0.103 m
vase scaled footprint: 0.100 x 0.100 m
cloth scaled footprint: 0.055 x 0.115 m
bowl scaled footprint: 0.140 x 0.140 m
spoon scaled footprint: 0.040 x 0.194 m
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
  move above bowl drop target
  lower
  release

spoon:
  move above spoon
  approach spoon
  close gripper
  lift spoon
  move above spoon drop target
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

需求中指定 bowl 要夾取邊緣，因此 FSM 不直接抓 bowl center。做法是使用 signed retreat offset 將 grasp target 從 bowl center 移到 bowl 邊緣：

```text
_GRASP_RETREAT_PER_OBJECT["bowl"] = -0.075
```

目前負值會讓 gripper 目標點偏向 bowl 遠離 robot 的邊緣，而不是 bowl 幾何中心。FSM 的 `move_above_object`、`approach_object`、`grasp_object` 都使用同一個 grasp anchor XY，因此 bowl 夾取時會先移動到 grasp anchor 正上方，再沿 z 方向垂直下降。

### Spoon 夾取設計

spoon 採用較小 retreat：

```text
_GRASP_RETREAT_PER_OBJECT["spoon"] = 0.020
```

spoon 的 object pose yaw 會在 env config 中加上 `3*pi/2`，也就是原本 USD heading correction 再額外旋轉 180 度。FSM 的 grasp yaw 會根據這個 spoon object yaw，再加上 `_GRASP_YAW_OFFSET = pi/2` 與小範圍 random yaw offset。

### Cloth 夾取與擦拭設計

cloth 夾取目標使用 cloth root / 幾何中心：

```text
_GRASP_RETREAT_PER_OBJECT["cloth"] = 0.000
_grasp_anchor_w("cloth") = cloth center
```

因此手臂在 approach 與 close gripper 時會對準抹布中心，而不是邊緣。

cloth 固定起始位置：

```text
cloth = (0.35, -0.43, 0.065)
```

擦拭區域：

```text
x = [0.08, 0.22]
y = [-0.55, -0.10]
```

擦拭採用 3 條 y-axis lanes，並以 cloth 的 scaled footprint `0.055 x 0.115 m` 規劃安全距離：

```text
x lanes = [0.10, 0.15, 0.19]
```

每條 lane 沿 y 方向掃過 dirty area。相鄰 lane 採用往返方式移動，避免每一條都需要回到同一端點：

```text
lane 0: y low  -> y high
lane 1: y high -> y low
lane 2: y low  -> y high
```

最右 lane 的 cloth 右緣約為 `0.218 m`，低於 vase/tissue 左側安全界線，因此擦拭時不會掃到 tissue 或 vase。以 cloth swept footprint 計算，目標擦拭區 `x=[0.08, 0.22]`, `y=[-0.55, -0.10]` 的理想 planned coverage 為 `98.2%`。FSM 與 env success 目前都累積實際 cloth/table coverage，並使用理想覆蓋率的 `70%` 作為成功門檻，也就是 dirty region coverage ratio 至少約 `68.7%`。

### FSM phase 與 duration

bowl 與 spoon 各 7 個 phase：

| Phase | 說明 | steps |
|-------|------|-------|
| move_above_object | 移動到物件上方 | 180 |
| approach_object | 下降到抓取高度 | 160 |
| grasp_object | 關閉 gripper | 25 |
| lift_object | 抬起物件 | 130 |
| move_above_drop | 移到 tray drop target 上方 | 170 |
| lower_to_release | 下降到釋放高度 | 80 |
| retreat_from_drop | 開爪並上移 | 40 |

單一物件共：

```text
785 steps
```

bowl + spoon 共：

```text
1570 steps
```

cloth pick 與 wipe 前置 phase：

| Phase | 說明 | steps |
|-------|------|-------|
| move_above_object | 移動到 cloth 上方 | 160 |
| approach_object | 下降到 cloth | 130 |
| grasp_object | 關閉 gripper | 30 |
| lift_object | 抬起 cloth | 110 |
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
1570 + 650 + 650 = 2870 steps
MAX_STEPS = 2970
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
2. bowl 的 XY 位置在 tray success zone 內。
3. spoon 的 XY 位置在 tray success zone 內。
4. actual cloth/table coverage ratio 至少達到理想覆蓋率的 `70%`。
5. tray 的 XY 位移不超過 `0.05 m`，避免以推動 tray 的方式通過 placement check。
6. tissue/vase 的 XY 位移不超過 `0.035 m`，避免擦拭過程撞偏固定物件。
7. tray、cloth、tissue、vase 的 root z 不低於 `0.0 m`。

目前 bowl/spoon 的 tray placement 不再檢查 z range，也不再要求 bowl 在 tray `+y` 側、spoon 在 tray `-y` 側；只要兩者的 XY 位置都落在 tray success zone 即可。

tray success zone：

```text
x half width = 0.13
y half width = 0.14
z range = not checked for bowl/spoon placement
```

## Teleoperation 執行方式

teleoperation 適合用來檢查 dining cleanup 場景、手動測試 grasp/wipe 動作，或錄製少量人工示範。執行環境需使用 Isaac Lab 可用的 GPU/Docker 環境。

### 1. 基本啟動

只要切換 task id 即可使用 dining cleanup 任務：

```bash
python scripts/teleop.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --teleop_device keyboard \
  --num_envs 1 \
  --device cuda \
  --enable_cameras
```

這個指令會載入 `HCIS-DiningCleanup-SingleArm-v0`，也就是 `packages/simulator/src/simulator/tasks/dining_cleanup/dining_cleanup_env_cfg.py` 定義的場景。若沒有提供 `--object_poses`，bowl/spoon 會使用 env config 內的預設初始位置；tray、tissue、vase、cloth 則固定在 env config 中定義的位置。

`--enable_cameras` 會啟用 `wrist` 與 `front` camera observation。若只是快速檢查 robot motion，可以省略；若要錄製可訓練的 LeRobot dataset，建議保留。

### 2. 使用 dining cleanup object poses

若要使用目前預設的 500 筆 bowl/spoon 初始位置，加入 `--object_poses`：

```bash
python scripts/teleop.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --teleop_device keyboard \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --object_poses data/dining_clean/dining_cleanup_object_poses_500.json
```

`object_poses` 只會更新 bowl/spoon；tray、tissue、vase、cloth 維持固定場景配置。`scripts/teleop.py` 會載入 JSON 中 `status == "full"` 的 entries，第一次 reset 後套用第 1 筆 episode pose，之後每次按 `R` 或 `N` reset 時會前進到下一筆 episode pose。

### 3. Keyboard 控制

end-effector delta 以 `panda_hand` frame 表示，teleop script 會使用 Franka keyboard controller 將操作轉成 joint target 與 gripper command。

| Key | 功能 |
|-----|------|
| `W` / `S` | +x / -x 平移 |
| `A` / `D` | +y / -y 平移 |
| `J` / `K` | +z / -z 平移 |
| `H` / `L` | roll- / roll+ |
| `U` / `I` | pitch- / pitch+ |
| `Q` / `E` | yaw- / yaw+ |
| `C` | 打開 gripper |
| `M` | 關閉 gripper |
| `R` | reset environment；若有 `--object_poses`，切到下一筆 episode pose |
| `N` | 標記目前 episode 成功並 reset；錄製時會存成 successful demo |

可用 `--sensitivity` 調整平移與旋轉步長。例如 bowl edge grasp 或 cloth wipe 需要更細緻操作時，可降低 sensitivity：

```bash
python scripts/teleop.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --teleop_device keyboard \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --object_poses data/dining_clean/dining_cleanup_object_poses_500.json \
  --sensitivity 0.5
```

### 4. 錄製 HDF5 demonstrations

若要錄製 HDF5 demonstrations，加入 `--record` 與 `--dataset_file`：

```bash
python scripts/teleop.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --teleop_device keyboard \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --object_poses data/dining_clean/dining_cleanup_object_poses_500.json \
  --record \
  --dataset_file ./datasets/dining_cleanup_teleop.hdf5 \
  --num_demos 10
```

錄製流程：

1. 啟動後開始操作 robot；第一次送出 action 時會開始 recording。
2. 完成任務後按 `N`，將 episode 標記為 success 並 reset。
3. 若 episode 失敗或想重來，按 `R` reset；這次 episode 不會被標成 success。
4. 若有設定 `--object_poses`，每次 `R` 或 `N` reset 後會套用下一筆 pose。
5. `--num_demos 10` 代表錄到 10 筆 successful demos 後自動結束；若設為 `0`，可用 Ctrl+C 手動停止並 finalize dataset。

若輸出檔案已存在，請改新的 `--dataset_file`，或加入 `--resume` 續錄到既有檔案。

### 5. 錄製 LeRobot dataset

若要直接輸出 LeRobot dataset，加入 `--use_lerobot_recorder`、`--lerobot_dataset_repo_id` 與 FPS：

```bash
python scripts/teleop.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --teleop_device keyboard \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --object_poses data/dining_clean/dining_cleanup_object_poses_500.json \
  --record \
  --use_lerobot_recorder \
  --lerobot_dataset_repo_id ${HF_USER}/dining-cleanup-teleop \
  --lerobot_dataset_fps 30 \
  --dataset_file ./datasets/dining_cleanup_teleop.hdf5 \
  --num_demos 10
```

`--lerobot_dataset_repo_id` 會決定本地 LeRobot dataset 目錄名稱與後續上傳 Hugging Face Hub 時的 repo id。錄製完成後若要上傳，可再使用：

```bash
hf upload ${HF_USER}/dining-cleanup-teleop --repo-type dataset
```

### 6. Success 判定注意事項

`scripts/teleop.py` 在手動 teleoperation 模式會關閉 env 原本的 automatic `success` termination，避免操作過程中自動 reset。因此 teleop 錄製時的成功與否不是由 `dining_cleanup_success(...)` 自動判定，而是由操作者按鍵決定：

```text
N = successful episode
R = reset/discard current episode
```

這和 `scripts/datagen/generate.py` 不同；datagen 會在 FSM episode 結束後呼叫 `DiningCleanupStateMachine.check_success(env)`，再決定是否輸出 successful demo。

## Datagen 執行方式

### 1. 產生 object poses

```bash
cd /home/weichen/AI_capstone/aicapstone_final_project

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

## 固定 Evaluation Protocol

Advanced level evaluation 應使用固定 seed、固定 episode length、固定 object pose split，並在每個 rollout episode reset 後套用 `object_poses` 中的下一筆 bowl/spoon 初始位置。`scripts/rollout.py` 目前支援 `--object_poses`，因此 dining cleanup policy evaluation 可以直接使用同一份 500 筆 pose 檔。

建議本地 evaluation 指令：

```bash
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
```

目前 rollout success 由 env termination 判斷，包含：

- bowl/spoon 的 XY 位置位於 tray success zone。
- bowl/spoon placement 不檢查 z range，也不要求 bowl/spoon 在 tray 內分別位於固定 y 側。
- actual cloth/table coverage 達到理想覆蓋率的 `70%`，目前約等於 dirty region coverage ratio `68.7%`。
- tissue/vase XY 位移不超過 `0.035 m`。

每個 rollout episode 結束時，terminal 會額外印出 dining cleanup stage status，包含 tableware、wiping、protected 的 success/fail，以及 cloth coverage ratio、threshold、ideal coverage。

## 快速檢查指令

### Python syntax check

```bash
python3 -m py_compile \
  packages/simulator/src/simulator/tasks/dining_cleanup/dining_cleanup_env_cfg.py \
  packages/simulator/src/simulator/tasks/dining_cleanup/__init__.py \
  packages/simulator/src/simulator/datagen/state_machine/dining_cleanup.py \
  scripts/generate_dining_cleanup_object_poses.py \
  scripts/visualize_dining_cleanup_layout.py \
  scripts/teleop.py \
  scripts/datagen/generate.py \
  scripts/rollout.py
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
5. bowl edge grasp 的偏移量 `_GRASP_RETREAT_PER_OBJECT["bowl"] = -0.075` 可能需要依 bowl USD mesh 實際大小微調；目前 FSM 會先移動到同一個 grasp anchor 正上方，再垂直下降夾取。

## 建議工作流程

每次改動 dining cleanup task 後，建議依序執行：

```bash
# 1. 語法檢查
python3 -m py_compile \
  packages/simulator/src/simulator/tasks/dining_cleanup/dining_cleanup_env_cfg.py \
  packages/simulator/src/simulator/datagen/state_machine/dining_cleanup.py \
  scripts/generate_dining_cleanup_object_poses.py \
  scripts/visualize_dining_cleanup_layout.py \
  scripts/teleop.py

# 2. 重新產生 poses
python3 scripts/generate_dining_cleanup_object_poses.py \
  --count 500 \
  --output data/dining_clean/dining_cleanup_object_poses_500.json

# 3. 視覺化檢查
python3 scripts/visualize_dining_cleanup_layout.py \
  --input data/dining_clean/dining_cleanup_object_poses_500.json \
  --output data/dining_clean/dining_cleanup_layout_xy.png

# 4. 可選：在 Isaac Lab GPU 環境跑 keyboard teleoperation 檢查場景
python scripts/teleop.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --teleop_device keyboard \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --object_poses data/dining_clean/dining_cleanup_object_poses_500.json

# 5. 在 Isaac Lab GPU 環境跑 datagen
python scripts/datagen/generate.py \
  --task HCIS-DiningCleanup-SingleArm-v0 \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --object_poses data/dining_clean/dining_cleanup_object_poses_500.json \
  --record \
  --dataset_file ./datasets/dining_cleanup.hdf5

# 6. 在 Isaac Lab GPU 環境跑 policy rollout evaluation
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
```
