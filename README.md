# AI Capstone

Sim-to-real imitation-learning pipeline for robot manipulation tasks. Record human demonstrations with UMI, process them through SLAM, generate synthetic data in Isaac Lab, train a diffusion policy with LeRobot, and evaluate it in simulation.

> **Platform:** Linux only.

For a complete step-by-step walkthrough, see [Getting Started](docs/getting_started.md).

# Human Demonstration Data Processing

1. **Installation**

   ```bash
   uv sync --package umi
   ```

2. **Activate the virtual environment**

   ```bash
   source .venv/bin/activate
   ```

   This makes `hf`, `lerobot-train`, and other installed commands available in your terminal.

3. **Hugging Face login**

   Create an access token at: <https://huggingface.co/docs/hub/en/security-tokens>

   Then log in:

   ```bash
   hf auth login --token <YOUR_HF_TOKEN>
   ```

4. **Set your Hugging Face username**

   Commands throughout this project use `${HF_USER}`. Set it once per terminal session:

   ```bash
   export HF_USER=<your-huggingface-username>
   ```

## After recording the demonstration videos, follow this practice

1. Under `data/`, create a directory for this demo. Suggested name: `YYYYMMDD-taskname`. Add a `raw_videos/` subdirectory under it.
2. Place the recorded videos in `data/YYYYMMDD-taskname/raw_videos/`.

## Verify the recorded demonstration videos

The SLAM mapping stage is fragile. To save time, run the verify pipeline first:

```bash
uv run umi run-slam-pipeline umi_pipeline_configs/verify_pipeline.yaml \
    --session-dir <demo_directory_name>
```

## If verification fails, re-record and copy into the demo directory

There are several failure modes:

### SLAM failures

Pipeline raises:

```
RuntimeError: SLAM mapping failed. Check logs at datasets/team_asia/demos/mapping/slam_stdout.txt for details.
```

Re-record the mapping video, replace the file, and re-run the verification pipeline.

## If verification succeeds, run the full pipeline

```bash
uv run umi run-slam-pipeline umi_pipeline_configs/build_dataset.yaml \
    --session-dir <demo_directory_name> \
    --task <kitchen|dining_room|living_room>
```

Upload the whole session directory to the Hugging Face Hub:

```bash
hf upload ${HF_USER}/<repo_id> data/<demo_directory_name>/demos/mapping/object_poses.json
```

# Data Creation in Simulator

## Prerequisites

1. **Linux machine with Nvidia GPU** — verify with `nvidia-smi`. Isaac Lab requires a Linux host with an Nvidia driver.
2. **Docker installed** — the simulator runs inside a container.
3. **Repository cloned** — if you haven't already:
   ```bash
   git clone https://github.com/HCIS-Lab/aicapstone.git
   cd aicapstone
   ```

## Launch Isaac Lab and run
Modify 
1. HF token
2. docker run arguments:     -v "/home/threedavatar/.cache/huggingface:/root/.cache/huggingface" \ # at line 39

```bash
bash run_datagen_adv-v1.sh
```
Upload the recorded dataset to Hugging Face Hub:
```bash
hf upload ${HF_USER}/<repo_id> ~/.cache/huggingface/lerobot/${HF_USER}/<repo_id>/
```

# LeRobot Training

Training runs on the **host machine** (not inside Docker) and produces a policy checkpoint from your generated dataset. Requires an Nvidia GPU.

See [LeRobot Training Procedure](docs/lerobot_training.md) for the full command reference, multi-GPU setup, and troubleshooting.

# LeRobot Rollout

Rollout loads your trained policy into the Isaac Lab simulator (inside the Docker container) to evaluate robot performance.

See [LeRobot Rollout (Policy Evaluation)](docs/lerobot_rollout.md) for the full procedure.

## Advanced Dining Cleanup

The Advanced-level Dining Cleanup task is documented separately:

- [Dining Cleanup implementation guide](docs/dining_cleanup/README.md)
- [Dining Cleanup evaluation configs](docs/dining_cleanup/evaluation_configs.md)
- [Dining Cleanup dataset generation](docs/dining_cleanup/dataset_generation.md)

The submission-ready Advanced package can be staged under:

```text
submission/dining_cleanup_advanced/
```

The staged submission folder should include a `README.txt` with the complete
execution and reproduction guide required by the course submission spec. The
source copy of that guide is maintained at `docs/dining_cleanup/README.txt`.

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting_started.md) | End-to-end pipeline walkthrough |
| [Developer Introduction](docs/dev/introduction.md) | Repo layout, environment setup, where to run what |
| [Isaac Lab + LeIsaac Configuration Tutorial](docs/isaaclab_leisaac_tutorial.md) | Configuring Isaac Lab with LeIsaac |
| [LeRobot Dataset Visualizer](docs/lerobot_dataset_visualizer.md) | Visualizing LeRobot datasets |
| [LeRobot Checkpoint Format](docs/lerobot_model_format.md) | Understanding LeRobot model checkpoint structure |
| [LeRobot Rollout (Policy Evaluation)](docs/lerobot_rollout.md) | Running trained policies in the simulator |
| [LeRobot Training Procedure](docs/lerobot_training.md) | Training imitation-learning policies |
| [Standalone Env Config Export](docs/standalone_env_config_export.md) | Exporting environment configs as standalone files |
| [Synthetic Data Generation Pipeline](docs/synthetic_data_generation.md) | Generating synthetic training data |
| [UMI Pipeline](docs/umi_pipeline.md) | Data collection and processing with UMI |
| [Dining Cleanup Implementation Guide](docs/dining_cleanup/README.md) | Advanced Dining Cleanup task design and reproduction guide |
| [Dining Cleanup Evaluation Configs](docs/dining_cleanup/evaluation_configs.md) | Fixed-yaw, random-yaw, and asset-shift evaluation configs |
| [Dining Cleanup Dataset Generation](docs/dining_cleanup/dataset_generation.md) | Object-pose split generation and layout visualization |

## License

MIT — see [LICENSE](LICENSE).
