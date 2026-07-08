#!/usr/bin/env bash
# board OCR full 学習: mobilenet_v3_small + Apple Silicon MPS 前提。
# 環境変数で上書き可: EPOCHS, BATCH_SIZE, NUM_WORKERS, LR, BACKBONE, IMAGE_SIZE,
#                    CKPT_DIR, SAVE_EVERY, RESUME, WANDB_RUN_ID, HF_REPO_ID
# 途中再開したいときは: RESUME=latest EPOCHS=30 ./scripts/train.sh
# 過去の W&B run に強制的に紐づけたいときは:
#   WANDB_RUN_ID=xxxxxxxx RESUME=latest EPOCHS=30 ./scripts/train.sh
# HF Hub から学習したいとき (CUDA マシンで想定):
#   HF_REPO_ID=ultemica/piyoshogi EPOCHS=60 ./scripts/train.sh
set -euo pipefail

EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-8}
LR=${LR:-6e-4}
BACKBONE=${BACKBONE:-mobilenet_v3_small}
IMAGE_SIZE=${IMAGE_SIZE:-288}
CKPT_DIR=${CKPT_DIR:-./runs/board-ocr-v2}
SAVE_EVERY=${SAVE_EVERY:-5}

extra_args=()
if [[ -n "${RESUME:-}" ]]; then
    extra_args+=(--resume "${RESUME}")
fi
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
    extra_args+=(--wandb-run-id "${WANDB_RUN_ID}")
fi
if [[ -n "${HF_REPO_ID:-}" ]]; then
    extra_args+=(--hf-repo-id "${HF_REPO_ID}")
fi

uv run python -m mito_train.training.train_board_ocr \
    --mode full \
    --backbone "${BACKBONE}" \
    --image-size "${IMAGE_SIZE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --lr "${LR}" \
    --ckpt-dir "${CKPT_DIR}" \
    --save-every "${SAVE_EVERY}" \
    "${extra_args[@]}"
