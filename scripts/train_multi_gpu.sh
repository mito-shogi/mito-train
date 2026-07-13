#!/usr/bin/env bash
# Multi-GPU training via torchrun (single node).
#
# Same env-var overrides as train_backbones.sh, plus:
#   NPROC_PER_NODE    Number of GPUs to use (defaults to all visible CUDA devices)
#   MASTER_PORT       Rendezvous port (defaults to 29500)
#   PER_GPU_WORKERS   DataLoader workers per rank (defaults to 4; total across
#                     ranks is NPROC_PER_NODE * PER_GPU_WORKERS)
#   BACKBONE          Model backbone (defaults to mobilenet_v3_small)
#
# Examples:
#   ./scripts/train_multi_gpu.sh                           # all GPUs, defaults
#   NPROC_PER_NODE=8 BACKBONE=convnext_tiny ./scripts/train_multi_gpu.sh
#   NPROC_PER_NODE=4 EPOCHS=100 ./scripts/train_multi_gpu.sh

set -euo pipefail

# Default nproc = all CUDA devices the runtime can see.
detect_gpu_count() {
    uv run python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1
}

NPROC_PER_NODE=${NPROC_PER_NODE:-$(detect_gpu_count)}
MASTER_PORT=${MASTER_PORT:-29500}
PER_GPU_WORKERS=${PER_GPU_WORKERS:-4}

EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-8}
IMAGE_SIZE=${IMAGE_SIZE:-224}
LR=${LR:-3e-4}
BACKBONE=${BACKBONE:-mobilenet_v3_small}
HF_REPO_ID=${HF_REPO_ID:-ultemica/piyoshogi}
MODEL_NAME=${MODEL_NAME:-board-ocr-${BACKBONE}}
SAVE_EVERY=${SAVE_EVERY:-5}

extra_args=(--preload)
if [[ -n "${RESUME:-}" ]]; then
    extra_args+=(--resume "${RESUME}")
fi
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
    extra_args+=(--wandb-run-id "${WANDB_RUN_ID}")
fi
if [[ "${COMPILE:-0}" == "1" ]]; then
    extra_args+=(--compile)
fi

echo "[train_multi_gpu] nproc=${NPROC_PER_NODE} backbone=${BACKBONE} batch=${BATCH_SIZE} (per rank)"
echo "[train_multi_gpu] effective batch = ${BATCH_SIZE} x ${NPROC_PER_NODE} = $(( BATCH_SIZE * NPROC_PER_NODE ))"
echo "[train_multi_gpu] workers=${PER_GPU_WORKERS} per rank ($(( PER_GPU_WORKERS * NPROC_PER_NODE )) total)"
echo "[train_multi_gpu] model_name=${MODEL_NAME} (ckpt_dir=./runs/${MODEL_NAME})"

# --standalone: single-node rendezvous, avoids setting MASTER_ADDR/MASTER_PORT
#               manually for the common single-host case.
uv run torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    -m mito_train.training.train_board_ocr \
    --hf-repo-id "${HF_REPO_ID}" \
    --mode full \
    --backbone "${BACKBONE}" \
    --image-size "${IMAGE_SIZE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${PER_GPU_WORKERS}" \
    --prefetch-factor "${PREFETCH_FACTOR}" \
    --lr "${LR}" \
    --model-name "${MODEL_NAME}" \
    --save-every "${SAVE_EVERY}" \
    "${extra_args[@]}"
