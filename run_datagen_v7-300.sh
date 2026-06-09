#!/usr/bin/env bash
# Step 3 datagen runner — runs inside a Docker container after build.
set -euo pipefail

WORKSPACE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="leisaac-isaaclab:latest"
HF_USER="AI-Final"
DATASET_REPO_ID="${HF_USER}/AI-aiCapstoneData-lerobot-cutlery-v7-2-300"
TASK_ID="HCIS-CutleryArrangement-SingleArm-v0"
OBJECT_POSES="data/umi/augmented_300/augmented_300.json"
HF_TOKEN=""
LOG="${WORKSPACE}/datagen_v7-2-300.log"

# Resolve Vulkan ICD (same logic as Makefile)
VK_ICD=""
for icd in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
    if [ -f "$icd" ]; then VK_ICD="$icd"; break; fi
done
if [ -z "$VK_ICD" ]; then
    echo "WARNING: No NVIDIA Vulkan ICD found." >&2
fi

echo "=== Starting datagen in Docker ===" | tee "$LOG"
echo "Dataset repo: ${DATASET_REPO_ID}" | tee -a "$LOG"
echo "Task: ${TASK_ID}" | tee -a "$LOG"
echo "Object poses: ${OBJECT_POSES}" | tee -a "$LOG"
date | tee -a "$LOG"

docker run --rm \
    --name isaaclab-datagen-v7-2-300 \
    --gpus '"device=all"' \
    --net=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${WORKSPACE}:/workspace/aicapstone" \
    -v "/workspace/aicapstone/.venv" \
    -v "/home/threedavatar/.cache/huggingface:/root/.cache/huggingface" \
    ${VK_ICD:+-v "${VK_ICD%/*}:${VK_ICD%/*}:ro"} \
    -v "/usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro" \
    -e OMNI_KIT_ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,display,utility,compute \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HF_USER="${HF_USER}" \
    -e DATAGEN_VIDEO_DIR=/workspace/aicapstone/datagen_v7-2-300_preview/videos \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        echo '== GPU ==' && nvidia-smi || true

        # ffmpeg needed by the per-episode video recorder
        if ! command -v ffmpeg >/dev/null 2>&1; then
            apt-get update -qq && apt-get install -y -qq ffmpeg
        fi

        # Vulkan ICD
        for icd in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
            if [ -f \"\$icd\" ]; then export VK_ICD_FILENAMES=\"\$icd\"; break; fi
        done
        echo \"VK_ICD_FILENAMES=\${VK_ICD_FILENAMES:-NONE}\"

        cd /workspace/aicapstone

        echo '=== Running generate.py ==='
        python scripts/datagen/generate.py \\
            --task ${TASK_ID} \\
            --num_envs 1 \\
            --device cuda \\
            --enable_cameras \\
            --headless \\
            --record \\
            --use_lerobot_recorder \\
            --lerobot_dataset_repo_id ${DATASET_REPO_ID} \\
            --object_poses ${OBJECT_POSES} \\
            --lerobot_dataset_fps 30

        echo '=== generate.py complete ==='

        echo '=== Uploading dataset to HuggingFace ==='
        LEROBOT_DIR=\"/root/.cache/huggingface/lerobot/${DATASET_REPO_ID}\"
        if [ -d \"\$LEROBOT_DIR\" ]; then
            EPISODE_COUNT=\$(find \"\$LEROBOT_DIR\" -name '*.parquet' | wc -l || echo 'unknown')
            echo \"Dataset directory: \$LEROBOT_DIR\"
            echo \"Parquet files found: \$EPISODE_COUNT\"
            hf upload ${DATASET_REPO_ID} \"\$LEROBOT_DIR\" --repo-type dataset
            echo '=== Upload complete ==='
        else
            echo 'ERROR: LeRobot dataset directory not found at '\$LEROBOT_DIR >&2
            exit 1
        fi
    " 2>&1 | tee -a "$LOG"

echo "=== run_datagen.sh finished ===" | tee -a "$LOG"
date | tee -a "$LOG"
