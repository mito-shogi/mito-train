#!/usr/bin/env bash
# board detector 学習: raw screenshot -> normalized bbox。
# 環境変数で上書き可: EPOCHS, BATCH_SIZE, NUM_WORKERS, LR, IMAGE_SIZE,
#                    OUT_DIR, WANDB_RUN_ID, RESUME_FROM, HF_REPO_ID, HF_CONFIG,
#                    CUDA_VISIBLE_DEVICES
# HF Hub から学習 (default):
#   HF_REPO_ID=ultemica/piyoshogi ./scripts/train_detector.sh
# 途中再開:
#   RESUME_FROM=runs/board-detector-v1/latest.pt EPOCHS=60 ./scripts/train_detector.sh
set -euo pipefail

EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
LR=${LR:-3e-4}
IMAGE_SIZE=${IMAGE_SIZE:-384}
OUT_DIR=${OUT_DIR:-runs/board-detector-v1}
HF_REPO_ID=${HF_REPO_ID:-ultemica/piyoshogi}
HF_CONFIG=${HF_CONFIG:-detector_paired}

extra_args=()
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
    extra_args+=(--wandb-run-id "${WANDB_RUN_ID}")
fi
if [[ -n "${WANDB_RESUME:-}" ]]; then
    extra_args+=(--wandb-resume "${WANDB_RESUME}")
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
    extra_args+=(--resume-from "${RESUME_FROM}")
fi
if [[ -n "${HF_REPO_ID}" ]]; then
    extra_args+=(--hf-repo-id "${HF_REPO_ID}" --hf-config "${HF_CONFIG}")
fi

uv run python -m mito_train.training.train_detector \
    --image-size "${IMAGE_SIZE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --lr "${LR}" \
    --out-dir "${OUT_DIR}" \
    "${extra_args[@]}"
