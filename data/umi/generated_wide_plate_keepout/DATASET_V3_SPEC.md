# Cutlery Arrangement Dataset v3 — Spec

Synthetic LeRobot dataset for the HCIS Cutlery Arrangement task. Produced on
2026-05-25 with the v3 FSM and a new "wide + plate-keepout" object-pose set.

## Targets

- **Target HF dataset:** https://huggingface.co/datasets/AI-Final/AI-aiCapstoneData-lerobot-cutlery-v3
- **Source object poses:**
  https://huggingface.co/datasets/AI-Final/AI-aiCapstoneData-lerobot-cutlery/blob/main/umi/generated_wide_plate_keepout/generated_wide_plate_keepout.json
  - Local copy: `data/umi/generated_wide_plate_keepout/generated_wide_plate_keepout.json`
  - 200 scenes, JSON-list format (`video_name`, `episode_range`, `objects`,
    `status`)
- **Isaac Lab task ID:** `HCIS-CutleryArrangement-SingleArm-v0`
- **Runner script:** `run_datagen.sh` (Docker image `leisaac-isaaclab:latest`)
- **Log:** `datagen_v3.log` at repo root

## What's special about v3

1. **v3 FSM** — the
   `packages/simulator/src/simulator/datagen/state_machine/cutlery_arrangement.py`
   used here is the `exp_lift_v2` snapshot, which scored **6/10** on the
   shared 10-pose FSM test subset (see `fsm_experiments/exp_lift_v2/`). This
   FSM contains two corrections on top of v2 (the FSM used for the v2
   dataset):
   - **Phase-3 lift uses a fixed xy target** (captured at the first lift step)
     instead of chasing the held object's drifting xy. Eliminates the
     "extends outward then comes back" pre-place wobble.
   - **`_LIFT_Z_OFFSET` raised 0.15 → 0.20 m** so the lifted cutlery and arm
     stop clipping neighbouring objects mid-transport.
   - Carried over from `exp_lift_fix`: phase 3 absolute target z =
     `plate.z + LIFT_Z_OFFSET` (not relative to held object); phase 6 keeps
     gripper CLOSED until EE is within `_RELEASE_TOL_M = 0.015` of release
     altitude.

2. **v3 object poses (wide + plate keepout)** — the pose distribution is
   broad across the workspace (similar coverage goal to the v2 200-set) but
   explicitly **avoids positions too close to the plate**. This removes the
   tightly-spaced-near-plate scenes that the FSM was most likely to fail on
   in v2, so we expect a higher demo-success rate and cleaner demos.

## Key FSM constants used in this run

```python
_MAX_CARTESIAN_DELTA = 0.018
_HOVER_Z_OFFSET      = 0.15
_GRASP_Z_OFFSET      = 0.08
_LIFT_Z_OFFSET       = 0.20
_RELEASE_Z_OFFSET    = 0.06
_RELEASE_TOL_M       = 0.015
_PICK_ORDER          = (_KNIFE_NAME, _FORK_NAME)
_PLACE_X_SIGNS       = (-1.0, +1.0)   # knife → -x, fork → +x (matches leaderboard eval)
_PHASE_DURATIONS_PER_OBJECT = (180, 130, 20, 160, 170, 15, 30)
```

Most recent FSM commit at run time: `f84d85e` — "fix: fork & knife place
position, add: printing progress during data generation."

## How to reproduce

```bash
# 1. Make sure data/umi/generated_wide_plate_keepout/generated_wide_plate_keepout.json
#    exists (download from the source URL above with `hf download`).
# 2. Make sure FSM matches fsm_experiments/exp_lift_v2/cutlery_arrangement.py.
# 3. Launch in tmux:
tmux new-session -d -s datagen_v3 "bash run_datagen.sh"
# 4. Tail the log:
tail -f datagen_v3.log
```

The runner generates demos for every entry in the poses JSON, then uploads the
resulting LeRobot dataset at
`/root/.cache/huggingface/lerobot/AI-Final/AI-aiCapstoneData-lerobot-cutlery-v3`
to the target HF dataset repo.

## Cross-references

- v2 dataset spec / training plan: see memory
  `project_datagen_v2_and_training_plan.md`
- FSM iteration workflow: see memory `project_fsm_experiments_workflow.md`
- FSM v2 → v3 changes in detail: `fsm_experiments/exp_lift_v2/CHANGES.md`
- Pose distribution v2 (for comparison): `data/umi/generared_200/object_pose_200.json`
