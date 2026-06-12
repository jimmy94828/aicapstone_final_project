# Dining Cleanup Advanced Project

This document is the implementation and reproduction guide for the Advanced-level
Dining Cleanup task.

The task extends the entry-level tableware manipulation setting into a
multi-stage dining cleanup benchmark. A single Franka arm must clear tableware
from a dirty region, place the tableware into a tray, pick up a cloth, wipe the
dirty table region, and avoid disturbing protected objects.

## Task Summary

Task ID:

```text
LeIsaac-HCIS-DiningCleanup-SingleArm-v0
```

Legacy compatibility alias:

```text
HCIS-DiningCleanup-SingleArm-v0
```

High-level objective:

```text
Clean up the dining table by moving the bowl and spoon into the tray, then
wiping the dirty left-side table region with the cloth while avoiding the
tissue and vase.
```

The task includes three major subgoals:

1. **Tableware clearing**: move `bowl` and `spoon` into the tray.
2. **Tool use**: pick up the `cloth` and sweep the dirty table region.
3. **Protected-object safety**: keep `tissue` and `vase` close to their initial positions.

## Important Files

| Purpose | Path |
|---|---|
| Gym task registration | `packages/simulator/src/simulator/tasks/dining_cleanup/__init__.py` |
| Standalone task/env config | `packages/simulator/src/simulator/tasks/dining_cleanup/dining_cleanup_env_cfg.py` |
| Shared Franka task template | `packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py` |
| Task auto-import registry | `packages/simulator/src/simulator/tasks/__init__.py` |
| Object pose loader | `packages/simulator/src/simulator/utils/object_poses_loader.py` |
| FSM datagen policy | `packages/simulator/src/simulator/datagen/state_machine/dining_cleanup.py` |
| Datagen entry point | `scripts/datagen/generate.py` |
| Rollout entry point | `scripts/rollout.py` |
| Host-side rollout wrapper | `rollout_advance_act.sh` |
| Object pose generator | `scripts/generate_dining_cleanup_object_poses.py` |
| Layout visualization | `scripts/visualize_dining_cleanup_layout.py` |
| Evaluation configs | `configs/dining_cleanup/*.json` |

## Standalone Configuration Export

Advanced-level submissions require a standalone environment configuration. The
Dining Cleanup task satisfies this through:

```text
packages/simulator/src/simulator/tasks/dining_cleanup/
  __init__.py
  dining_cleanup_env_cfg.py
```

The registration file defines the primary Gymnasium task id and a legacy
compatibility alias:

```python
for task_id in (
    "LeIsaac-HCIS-DiningCleanup-SingleArm-v0",
    "HCIS-DiningCleanup-SingleArm-v0",
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.dining_cleanup_env_cfg:DiningCleanupEnvCfg",
        },
    )
```

The config file subclasses the in-tree single-arm Franka template:

```text
DiningCleanupEnvCfg -> SingleArmFrankaTaskEnvCfg
DiningCleanupSceneCfg -> SingleArmFrankaTaskSceneCfg
```

This means the task can be reconstructed by importing `simulator.tasks` and
calling `parse_env_cfg("LeIsaac-HCIS-DiningCleanup-SingleArm-v0", ...)`.

Verification command:

```bash
PYTHONPATH=./packages/simulator/src python - <<'PY'
import simulator.tasks
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("LeIsaac-HCIS-DiningCleanup-SingleArm-v0", num_envs=1)
print(type(cfg).__module__, type(cfg).__qualname__)
print(cfg.task_description)
PY
```

Expected module:

```text
simulator.tasks.dining_cleanup.dining_cleanup_env_cfg DiningCleanupEnvCfg
```

## Scene Design

The dining room background is loaded from:

```text
packages/simulator/assets/scenes/dining_room/scene.usd
```

The interactive objects are spawned by `DiningCleanupSceneCfg` as task-level
objects. The background scene is not patched directly. This keeps the task
configuration reproducible and avoids mutating a shared USD scene file.

The environment contains:

| Object | Role |
|---|---|
| `bowl` | Movable tableware target |
| `spoon` | Movable tableware target; can also be replaced by wooden fork through config |
| `tray` | Placement target for tableware |
| `cloth` | Tool for wiping the dirty region |
| `tissue` | Protected object |
| `vase` | Protected object |

## Coordinate Convention

The dining table world XY footprint is approximately:

```text
x = [0.00, 0.70]
y = [-0.65, 0.00]
```

The task convention is:

| Region | Meaning |
|---|---|
| left / low x | dirty area and initial bowl/spoon region |
| right / high x | tray side |
| middle table | tissue, vase, and cloth positions |

The dirty region used for wiping is:

```text
x = [0.00, 0.22]
y = [-0.50, -0.10]
```

The planned wipe lanes are:

```text
x = 0.21, 0.18, 0.15, 0.11, 0.07
```

## Fixed Object Placement

Fixed object positions are defined in `dining_cleanup_env_cfg.py`.

| Object | World position |
|---|---|
| tray | `(0.57, -0.36, 0.05)` |
| tissue | `(0.35, -0.12, 0.074)` |
| vase | `(0.35, -0.26, 0.05)` |
| cloth | `(0.35, -0.43, 0.065)` |

The y-order on the middle table is:

```text
tissue -> vase -> cloth
```

## Object Footprints

The task uses scaled tabletop footprints for object overlap checks, layout
visualization, and success-region reasoning.

| Object | Source geometry | Spawn scale / size | Approx. XY footprint |
|---|---|---|---|
| bowl | USD | `(0.50, 0.50, 0.50)` | `0.140 x 0.140 m` |
| spoon | USD | `(0.60, 0.60, 0.60)` | `0.040 x 0.194 m` |
| tray | USD | `(0.79, 1.77, 1.00)` | `0.240 x 0.260 m` |
| tissue | USD | `(1.00, 1.00, 1.00)` | `0.073 x 0.103 m` |
| vase | USD | `(0.591, 0.591, 0.591)` | `0.100 x 0.100 m` |
| cloth | CuboidCfg | `(0.055, 0.115, 0.030)` | `0.055 x 0.115 m` |

The cloth is represented as a rigid cuboid for stable manipulation. The original
cloth USD asset is retained as part of the asset package, but the benchmark uses
the cuboid geometry defined in the config.

## USD Assets

Dining Cleanup uses assets under:

```text
packages/simulator/assets/scenes/dining_room/
```

For a self-contained Advanced-level submission, include these directories:

```text
packages/simulator/assets/scenes/dining_room/scene.usd
packages/simulator/assets/scenes/dining_room/objects/bowl/
packages/simulator/assets/scenes/dining_room/objects/spoon/
packages/simulator/assets/scenes/dining_room/objects/tray/
packages/simulator/assets/scenes/dining_room/objects/tissue/
packages/simulator/assets/scenes/dining_room/objects/vase/
packages/simulator/assets/scenes/dining_room/objects/cloth/
packages/simulator/assets/scenes/dining_room/objects/bowl_2/
packages/simulator/assets/scenes/dining_room/objects/wooden_fork/
```

The `bowl_2` and `wooden_fork` directories are required for the
`fork_bowl2_scaled` evaluation config.

USD files often reference nested `origin/` and `resource/` files. Include whole
asset directories rather than only the top-level `.usd` file.

## Evaluation Configs

Three configs are provided under `configs/dining_cleanup/`.

### 1. Fixed Spoon Yaw

```text
configs/dining_cleanup/spoon_fixed_yaw.json
```

This config uses the original bowl and spoon assets. The spoon yaw is fixed.

Object pose split:

```text
data/dining_clean/dining_cleanup_object_poses_500.json
```

### 2. Random Spoon Yaw

```text
configs/dining_cleanup/spoon_random_yaw.json
```

This config uses the original bowl and spoon assets, but the spoon yaw is
randomized across episodes.

Object pose split:

```text
data/dining_clean/dining_cleanup_spoon_random_yaw_200.json
```

### 3. Fork + Bowl2 Scaled

```text
configs/dining_cleanup/fork_bowl2_scaled.json
```

This config keeps the scene object keys as `bowl` and `spoon` so the FSM,
success checks, and object pose loader do not need to change. Internally:

| Scene key | Actual asset |
|---|---|
| `bowl` | `objects/bowl_2/model_B075HWDSDK_69323.usd` |
| `spoon` | `objects/wooden_fork/model_WoodenFork_69323.usd` |

Object pose split:

```text
data/dining_clean/dining_cleanup_fork_bowl2_scaled_200.json
```

## Object Pose Format

The task uses the existing UMI-style per-episode object pose schema:

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

Only entries with `status == "full"` are used by datagen and rollout.

The raw `tvec` values are anchor-frame positions. The loader converts them to
world coordinates using:

```text
ANCHOR_WORLD_POSE = (0.40, 0.10, 0.0)
world_xy = raw_tvec_xy + (0.40, 0.10)
```

The object pose file controls only the initial `bowl` and `spoon` poses. Tray,
tissue, vase, and cloth are fixed by the environment config.

## Generating Object Pose Splits

Generate the default fixed-yaw split:

```bash
python3 scripts/generate_dining_cleanup_object_poses.py \
  --count 500 \
  --seed 2026053002 \
  --output data/dining_clean/dining_cleanup_object_poses_500.json
```

Visualize a split:

```bash
python3 scripts/visualize_dining_cleanup_layout.py \
  --input data/dining_clean/dining_cleanup_spoon_random_yaw_200.json \
  --output data/dining_clean/dining_cleanup_spoon_random_yaw_200_layout_xy.png
```

The layout visualization shows:

- table footprint
- dirty region
- expected cloth path
- fixed object footprints
- bowl/spoon occupied envelopes
- per-episode bowl/spoon start points

## Datagen

FSM datagen is implemented in:

```text
packages/simulator/src/simulator/datagen/state_machine/dining_cleanup.py
```

Run datagen with a Dining Cleanup config:

```bash
python scripts/datagen/generate.py \
  --task LeIsaac-HCIS-DiningCleanup-SingleArm-v0 \
  --dining_cleanup_config configs/dining_cleanup/spoon_fixed_yaw.json \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --record \
  --dataset_file ./datasets/dining_cleanup_fixed_spoon.hdf5
```

Generate a LeRobot-format dataset:

```bash
HF_USER=AI-Final

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
```

The number of datagen episodes is determined by the selected config's
`object_poses` file.

## Rollout

The direct rollout entry point is:

```text
scripts/rollout.py
```

Direct Python command:

```bash
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
```

Important arguments:

| Argument | Meaning |
|---|---|
| `--dining_cleanup_config` | Selects asset overrides and object pose split |
| `--policy_type` | LeRobot policy type, e.g. `lerobot-act` or `lerobot-smolvla` |
| `--policy_checkpoint_path` | Local checkpoint directory containing `config.json` |
| `--policy_action_horizon` | Number of action steps consumed per policy call |
| `--eval_rounds` | Number of evaluation episodes |
| `--episode_length_s` | Episode timeout |
| `--record_video` | Saves rollout videos |

## GlowsAI / Docker Rollout Wrapper

The host-side wrapper is:

```text
rollout_advance_act.sh
```

Run it from the GlowsAI host shell, not from inside the container prompt. The
script starts its own Docker container.

ACT rollout from Hugging Face:

```bash
HF_REPO_ID=AI-Final/advanced-act-v3 \
MODEL_NAME=advanced-act-v3 \
CHECKPOINT=hf \
POLICY_TYPE=lerobot-act \
POLICY_ACTION_HORIZON=25 \
./rollout_advance_act.sh configs/dining_cleanup/spoon_fixed_yaw.json
```

SmolVLA rollout from Hugging Face:

```bash
HF_REPO_ID=AI-Final/smolvla-advanced-v3 \
MODEL_NAME=smolvla-advanced-v3 \
CHECKPOINT=040000 \
POLICY_TYPE=lerobot-smolvla \
POLICY_ACTION_HORIZON=16 \
HF_MODEL_SUBDIR=training_runs/smolvla_advanced_v3_20260611_185615/checkpoints/040000 \
MODEL_DIR_REL=experiments/advance/smolvla-advanced-v3/training_runs/smolvla_advanced_v3_20260611_185615/checkpoints/040000/pretrained_model \
./rollout_advance_act.sh configs/dining_cleanup/spoon_fixed_yaw.json
```

If the Hugging Face repository is private:

```bash
export HF_TOKEN=<your_token>
```

Rollout outputs are written under:

```text
experiments/advance/<model_name>/
```

Example:

```text
rollout_040000_spoon_fixed_yaw_<timestamp>.log
rollout_040000_spoon_fixed_yaw_<timestamp>_videos/
```

## Success Criteria

An episode succeeds when all major checks pass:

1. Bowl and spoon are inside the tray success region.
2. Wiping coverage reaches the threshold.
3. Tissue and vase remain within the protected-object displacement tolerance.

The relevant constants are defined in `dining_cleanup_env_cfg.py`:

| Constant | Meaning |
|---|---|
| `TRAY_SUCCESS_X_HALF_WIDTH` | Tray success region x half-width |
| `TRAY_SUCCESS_Y_HALF_WIDTH` | Tray success region y half-width |
| `WIPE_COVERAGE_THRESHOLD` | Required wipe coverage threshold |
| `WIPE_COVERAGE_RESOLUTION` | Wipe coverage grid resolution |
| `STATIC_OBJECT_XY_TOL` | Protected-object displacement tolerance |

The rollout script prints per-episode status and final success rate. Each
episode report includes tableware, wiping, and protected-object checks.

## Reproducibility Settings

Use fixed settings when reporting benchmark results:

```text
task: LeIsaac-HCIS-DiningCleanup-SingleArm-v0
seed: 2026061011
eval_rounds: 30
episode_length_s: 80
video_fps: 30
device: cuda
```

Recommended policy horizons:

| Policy | `--policy_type` | `--policy_action_horizon` |
|---|---|---|
| ACT | `lerobot-act` | `25` |
| SmolVLA | `lerobot-smolvla` | `16` |

## Submission Packaging Checklist

For Advanced-level submission, package at least:

```text
submission/
  README.txt
  Configurations/
    simulator/tasks/dining_cleanup/
      __init__.py
      dining_cleanup_env_cfg.py
    configs/dining_cleanup/
      spoon_fixed_yaw.json
      spoon_random_yaw.json
      fork_bowl2_scaled.json
    data/dining_clean/
      dining_cleanup_object_poses_500.json
      dining_cleanup_spoon_random_yaw_200.json
      dining_cleanup_fork_bowl2_scaled_200.json
  Custom_CAD_Models/
    packages/simulator/assets/scenes/dining_room/
      scene.usd
      objects/bowl/
      objects/spoon/
      objects/tray/
      objects/tissue/
      objects/vase/
      objects/cloth/
      objects/bowl_2/
      objects/wooden_fork/
  Supplementary/
    rollout_logs/
    selected_rollout_videos/
    layout_figures/
    packages/simulator/src/simulator/datagen/state_machine/
      dining_cleanup.py
    scripts/
      datagen/generate.py
      rollout.py
      generate_dining_cleanup_object_poses.py
      visualize_dining_cleanup_layout.py
    model_links.txt
```

`README.txt` should be copied to the submission root. This Markdown file can be
included as supplementary implementation documentation. The Dining Cleanup FSM
is included under `Supplementary/packages/...` because it is required for
scripted datagen reproducibility, while `Configurations/` is kept focused on the
standalone environment config and its import dependencies.

The `Configurations/configs/dining_cleanup/` directory should remain inside the
Configurations folder because those JSON files define the three claimed
evaluation variants: fixed spoon yaw, random spoon yaw, and the fork/bowl2 asset
shift. Each config selects the task id, object USD assets, scales, and default
pose split. They are part of the benchmark configuration, not just supporting
notes.

The `Configurations/data/dining_clean/` directory should also remain inside the
Configurations folder because the config JSON files reference those deterministic
pose splits with repo-root-relative paths, for example:

```json
"object_poses": "data/dining_clean/dining_cleanup_spoon_random_yaw_200.json"
```

These pose splits are required to reproduce the exact initial bowl/spoon layouts
used by datagen and rollout evaluation. Keeping them next to the configs makes
the configuration package self-contained and avoids broken relative paths during
evaluation. They can be documented in Supplementary, but they should not be moved
only to Supplementary unless all config and command paths are changed as well.

## Benchmarking Integrity Notes

- The task is reproducible from the standalone env config and Gym registration.
- The object pose splits used for evaluation are explicit JSON files.
- Asset-shift evaluation is controlled through config JSON files, not hidden code changes.
- No manual intervention is required during rollout.
- Videos and logs are generated locally and are not automatically uploaded to Hugging Face.
- If using Hugging Face checkpoints, the exact repository and checkpoint path should be stated in `README.txt`.

## Related Documentation

| Document | Purpose |
|---|---|
| `docs/standalone_env_config_export.md` | Advanced standalone config export requirement |
| `docs/dining_cleanup/evaluation_configs.md` | Three Dining Cleanup evaluation configs |
| `docs/dining_cleanup/dataset_generation.md` | Object pose generation and datagen details |
| `docs/lerobot_rollout.md` | General LeRobot rollout flow |
