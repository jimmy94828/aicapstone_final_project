#!/usr/bin/env bash
# ACT-v5 training — on v5 (DART + close_eval) dataset, WITH TEMPORAL ENSEMBLING.
# Key change vs v3/v4: chunk_size=100, n_action_steps=1, temporal_ensemble_coeff=0.01
# (the canonical ACT deployment we never used). Trains to 80k, uploads to HF.
set -euo pipefail

WORKSPACE="$(cd "$(dirname "$0")" && pwd)"
HF_USER="AI-Final"
VERSION="v7-300"
DATASET_REPO_ID="${HF_USER}/AI-aiCapstoneData-lerobot-cutlery-${VERSION}"
POLICY_REPO_ID="${HF_USER}/cutlery-act-${VERSION}"
EXPERIMENT_DIR="${WORKSPACE}/experiments/act_cutlery_${VERSION}"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoint"
LOG="${EXPERIMENT_DIR}/train.log"
HF_TOKEN=""

WANDB_ENTITY="fran3-14159265-national-yang-ming-chiao-tung-university"
WANDB_PROJECT="cutlery-act-${VERSION}"

mkdir -p "${EXPERIMENT_DIR}"

echo "=== ACT-v5 training (DART data + TEMPORAL ENSEMBLING) ===" | tee "${LOG}"
echo "Dataset:    ${DATASET_REPO_ID}"   | tee -a "${LOG}"
echo "Policy:     ${POLICY_REPO_ID}"    | tee -a "${LOG}"
echo "Wandb:      https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}" | tee -a "${LOG}"
date | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

source "${WORKSPACE}/.venv/bin/activate"
export HF_TOKEN="${HF_TOKEN}"
export HF_USER="${HF_USER}"
export WANDB_API_KEY="$(grep -A2 'machine api.wandb.ai' ~/.netrc | grep password | awk '{print $2}')"


export HOMEUSER="$(id -un)"
export REPO_ID=AI-aiCapstoneData-lerobot-cutlery-${VERSION}
mkdir -p /home/$HOMEUSER/.cache/huggingface/lerobot/AI-Final/$REPO_ID
huggingface-cli download AI-Final/$REPO_ID  --repo-type dataset  --include "meta/*" "data/*" "videos/*"  --local-dir /home/$HOMEUSER/.cache/huggingface/lerobot/AI-Final/$REPO_ID


# Temporal ensembling: chunk_size=100, n_action_steps=1, temporal_ensemble_coeff=0.01.
# Everything else matches v3/v4 (bs=16, lr=1.4e-5, cosine sched, image_transforms).
lerobot-train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.image_transforms.enable=true \
    --policy.type=act \
    --policy.chunk_size=100 \
    --policy.n_action_steps=1 \
    --policy.temporal_ensemble_coeff=0.01 \
    --policy.device=cuda \
    --policy.repo_id="${POLICY_REPO_ID}" \
    --use_policy_training_preset=false \
    --optimizer.type=adamw \
    --optimizer.lr=1.4e-5 \
    --optimizer.weight_decay=1e-4 \
    --optimizer.grad_clip_norm=10.0 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=1000 \
    --scheduler.num_decay_steps=119000 \
    --scheduler.peak_lr=1.4e-5 \
    --scheduler.decay_lr=1e-7 \
    --output_dir="${CHECKPOINT_DIR}" \
    --job_name=cutlery_act_v5-2 \
    --batch_size=16 \
    --steps=120000 \
    --save_freq=10000 \
    --log_freq=200 \
    --wandb.enable=true \
    --wandb.project="${WANDB_PROJECT}" \
    --wandb.entity="${WANDB_ENTITY}" \
    2>&1 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "=== v5 training complete ===" | tee -a "${LOG}"
date | tee -a "${LOG}"

# Upload final policy
echo "=== Uploading 120k policy to ${POLICY_REPO_ID} ===" | tee -a "${LOG}"
PRETRAINED_DIR="${CHECKPOINT_DIR}/pretrained_model"
LAST_CKPT="${CHECKPOINT_DIR}/checkpoints/last/pretrained_model"
UPLOAD_DIR=""
[ -d "${PRETRAINED_DIR}" ] && UPLOAD_DIR="${PRETRAINED_DIR}"
[ -z "${UPLOAD_DIR}" ] && [ -d "${LAST_CKPT}" ] && UPLOAD_DIR="${LAST_CKPT}"
if [ -n "${UPLOAD_DIR}" ]; then
    hf upload "${POLICY_REPO_ID}" "${UPLOAD_DIR}" --repo-type model 2>&1 | tail -3 | tee -a "${LOG}"
else
    echo "ERROR: no pretrained_model dir under ${CHECKPOINT_DIR}" | tee -a "${LOG}"; exit 1
fi

echo "=== v5 training + upload finished ===" | tee -a "${LOG}"
date | tee -a "${LOG}"

# ---- Auto-eval: 15 normal + 15 spaced on the 120k checkpoint ----
echo "" | tee -a "${LOG}"
echo "=== Eval 1/2: 15 attempts, NORMAL distribution ===" | tee -a "${LOG}"
bash "${WORKSPACE}/run_eval_v5_local.sh" 120000 15 2>&1 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "=== Eval 2/2: 15 attempts, SPACED distribution ===" | tee -a "${LOG}"
bash "${WORKSPACE}/run_eval_v5_local_spaced.sh" 120000 15 2>&1 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "=== v5 train + 15+15 eval finished ===" | tee -a "${LOG}"
date | tee -a "${LOG}"
