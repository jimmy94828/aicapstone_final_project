#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="leisaac-isaaclab:latest"
HF_USER="AI-Final"
MODEL_NAME="act_cutlery_v7-300"
CHECKPOINT="200000"
TASK_ID="HCIS-DiningCleanup-SingleArm-v0"
DINING_CLEANUP_CONFIG="configs/dining_cleanup/spoon_fixed_yaw.json"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
MODEL_DIR_REL="experiments/${MODEL_NAME}/checkpoint/checkpoints/${CHECKPOINT}/pretrained_model"
VIDEO_DIR_REL="outputs/rollout/fixed_spoon_advanced_diffusion_v3_eval30_len80_fps30_progress_${RUN_STAMP}_videos"
LOG_REL="outputs/rollout/fixed_spoon_advanced_diffusion_v3_eval30_len80_fps30_progress_${RUN_STAMP}.log"
HOST_LOG_FILE="${WORKSPACE}/${LOG_REL}"

# Resolve Vulkan ICD (same logic as run_datagen_v7-300.sh)
VK_ICD=""
for icd in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
    if [ -f "$icd" ]; then VK_ICD="$icd"; break; fi
done
if [ -z "$VK_ICD" ]; then
    echo "WARNING: No NVIDIA Vulkan ICD found." >&2
fi

mkdir -p "$(dirname "$HOST_LOG_FILE")"

echo "=== Starting rollout in Docker ===" | tee "$HOST_LOG_FILE"
echo "Model dir: ${MODEL_DIR_REL}" | tee -a "$HOST_LOG_FILE"
echo "Task: ${TASK_ID}" | tee -a "$HOST_LOG_FILE"
echo "Dining cleanup config: ${DINING_CLEANUP_CONFIG}" | tee -a "$HOST_LOG_FILE"
echo "Video dir: ${VIDEO_DIR_REL}" | tee -a "$HOST_LOG_FILE"
date | tee -a "$HOST_LOG_FILE"

set +e
docker run --rm \
    --name isaaclab-rollout-entry \
    -i \
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
    -e HF_HOME=/workspace/aicapstone/hf_cache \
    -e HF_HUB_DISABLE_XET=1 \
    -e HF_USER="${HF_USER}" \
    -e MODEL_NAME="${MODEL_NAME}" \
    -e CHECKPOINT="${CHECKPOINT}" \
    -e TASK_ID="${TASK_ID}" \
    -e DINING_CLEANUP_CONFIG="${DINING_CLEANUP_CONFIG}" \
    -e MODEL_DIR_REL="${MODEL_DIR_REL}" \
    -e VIDEO_DIR_REL="${VIDEO_DIR_REL}" \
    -e LOG_REL="${LOG_REL}" \
    "${IMAGE}" \
    bash -s <<'EOF' |& tee -a "$HOST_LOG_FILE"
set -euo pipefail

cd /workspace/aicapstone

echo '===== stop old rollout ====='
jobs -l || true
pkill -9 -f 'python scripts/rollout.py' || true
pkill -9 -f ffmpeg || true

echo '===== verify no old rollout remains ====='
ps -eo pid,etime,pcpu,pmem,stat,args | grep -E '[p]ython scripts/rollout.py|[f]fmpeg' || true
echo '== GPU ==' && nvidia-smi || true

echo '===== env ====='
for icd in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
    if [ -f "$icd" ]; then export VK_ICD_FILENAMES="$icd"; break; fi
done
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-NONE}"

echo '===== ffmpeg ====='
command -v ffmpeg >/dev/null 2>&1 || {
    apt-get update -qq
    apt-get install -y -qq ffmpeg
}

echo '===== tqdm ====='
python - <<'PY' || python -m pip install tqdm
try:
  import tqdm
  print('tqdm ok:', tqdm.__version__)
except Exception as exc:
  print('tqdm missing:', exc)
  raise SystemExit(1)
PY

echo '===== model ====='
MODEL_DIR="/workspace/aicapstone/${MODEL_DIR_REL}"
test -d "$MODEL_DIR" || { echo "missing MODEL_DIR=$MODEL_DIR"; exit 1; }
grep -RniE '"type"|"policy_type"|diffusion|DiffusionConfig|act|ACTConfig' "$MODEL_DIR" 2>/dev/null | head -30 || true

echo '===== rollout ====='
VIDEO_DIR="/workspace/aicapstone/${VIDEO_DIR_REL}"
LOG="/workspace/aicapstone/${LOG_REL}"
mkdir -p "$(dirname "$LOG")" "$VIDEO_DIR"
echo "LOG=$LOG"
echo "VIDEO_DIR=$VIDEO_DIR"

set +e
timeout 12h python scripts/rollout.py \
    --headless \
    --task "$TASK_ID" \
    --dining_cleanup_config "$DINING_CLEANUP_CONFIG" \
    --policy_type=lerobot-diffusion \
    --policy_checkpoint_path "$MODEL_DIR" \
    --policy_action_horizon=25 \
    --device=cuda \
    --enable_cameras \
    --show_wipe_mesh \
    --eval_rounds=30 \
    --episode_length_s=80 \
    --seed=2026061011 \
    --record_video \
    --video_dir "$VIDEO_DIR" \
    --video_fps=30 \
    --progress_interval_s=10
STATUS=$?
set -e

echo '===== done ====='
echo "rollout exit status: $STATUS"
echo "LOG=$LOG"
echo "VIDEO_DIR=$VIDEO_DIR"

echo '===== key lines ====='
grep -nE 'Traceback|RuntimeError|Error|GPU solver pipeline failed|GPU Bp pipeline failed|Evaluating episode|Episode [0-9]+ timed out|Episode [0-9]+ is successful|Final success rate|video|progress' "$LOG" | tail -160 || true

echo '===== videos ====='
find "$VIDEO_DIR" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -40

exit "$STATUS"
EOF
STATUS=${PIPESTATUS[0]}
set -e

echo "=== rollout_entry finished ===" | tee -a "$HOST_LOG_FILE"
date | tee -a "$HOST_LOG_FILE"
exit "$STATUS"