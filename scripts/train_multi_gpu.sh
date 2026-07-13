#!/usr/bin/env bash
# Multi-GPU training via torchrun (single node).
#
# Same env-var overrides as train_backbones.sh, plus:
#   NPROC_PER_NODE    Number of GPUs to use (defaults to all visible CUDA devices)
#   MASTER_PORT       Rendezvous port (defaults to 29500)
#   PER_GPU_WORKERS   DataLoader workers per rank (defaults to 4; total across
#                     ranks is NPROC_PER_NODE * PER_GPU_WORKERS)
#   BACKBONE          Model backbone (defaults to mobilenet_v3_small)
#   AUTO_FREE_GPUS    When 1, restrict training to GPUs whose used memory is
#                     below FREE_GPU_MEM_MB (default 500). Sets
#                     CUDA_VISIBLE_DEVICES and NPROC_PER_NODE to the survivors
#                     so shared boxes don't step on running jobs.
#   FREE_GPU_MEM_MB   "Free" threshold in MiB (default 500). Only used when
#                     AUTO_FREE_GPUS=1.
#
# Examples:
#   ./scripts/train_multi_gpu.sh                           # all GPUs, defaults
#   NPROC_PER_NODE=8 BACKBONE=convnext_tiny ./scripts/train_multi_gpu.sh
#   NPROC_PER_NODE=4 EPOCHS=100 ./scripts/train_multi_gpu.sh
#   AUTO_FREE_GPUS=1 BACKBONE=mobilenet_v3_small ./scripts/train_multi_gpu.sh

set -euo pipefail

# Default nproc = all CUDA devices the runtime can see.
detect_gpu_count() {
    uv run python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1
}

# Select GPU indices whose used memory is under FREE_GPU_MEM_MB. Prints a
# comma-separated list (or an empty string if none qualify).
detect_free_gpus() {
    local threshold="${1:-500}"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F, -v t="${threshold}" '{
              gsub(/ /, "", $1); gsub(/ /, "", $2);
              if ($2+0 < t+0) picks[n++] = $1
          }
          END {
              for (i=0; i<n; i++) printf "%s%s", (i?",":""), picks[i]
          }'
}

if [[ "${AUTO_FREE_GPUS:-0}" == "1" ]]; then
    FREE_GPU_MEM_MB=${FREE_GPU_MEM_MB:-500}
    free_list=$(detect_free_gpus "${FREE_GPU_MEM_MB}")
    if [[ -z "${free_list}" ]]; then
        echo "[train_multi_gpu] AUTO_FREE_GPUS=1 but no GPU has memory < ${FREE_GPU_MEM_MB}MiB" >&2
        nvidia-smi --query-gpu=index,memory.used --format=csv >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="${free_list}"
    # torchrun sees the remapped devices, so nproc is just the count.
    NPROC_PER_NODE=$(awk -F, '{print NF}' <<<"${free_list}")
    echo "[train_multi_gpu] AUTO_FREE_GPUS: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (threshold=${FREE_GPU_MEM_MB}MiB)"
fi

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
