#!/usr/bin/env bash
# Backbone sweep: train each backbone for the same number of epochs and settings,
# so W&B side-by-side lets you spot the params-vs-accuracy Pareto front.
#
# Three launch modes; pick one via env vars:
#   (default)               1 run at a time, single GPU
#   NPROC_PER_NODE=N        1 run at a time, N-GPU DDP (torchrun)
#   PARALLEL_GPU=1          many runs concurrently, 1 GPU each (best for a
#                           multi-backbone sweep on a multi-GPU node)
#
# General env overrides:
#   EPOCHS BATCH_SIZE NUM_WORKERS PREFETCH_FACTOR IMAGE_SIZE LR
#   HF_REPO_ID  (defaults to ultemica/piyoshogi)
#   CKPT_ROOT   (dir under which each backbone gets its own ./<backbone>/)
#   BACKBONES   (space-separated subset; defaults to all 9, small->large)
#   SKIP_EXISTING=1      skip a backbone if ckpt_dir/latest.pt exists (blunt)
#   RESUME_INCOMPLETE=1  smart: skip if epoch-EPOCHS.pt exists (done), resume
#                        if only latest.pt exists (partial), else fresh
#   COMPILE=1            pass --compile to each run
#
# Parallel-mode extras:
#   NGPUS=<N>            override auto-detected GPU count
#   PARALLEL_LOG_DIR     directory for per-backbone stdout logs
#                        (defaults to CKPT_ROOT/logs)
#
# Examples:
#   ./scripts/train_backbones.sh
#   EPOCHS=30 BACKBONES="mobilenet_v3_small convnext_atto" ./scripts/train_backbones.sh
#   RESUME_INCOMPLETE=1 ./scripts/train_backbones.sh          # resume mid-sweep
#   NPROC_PER_NODE=8 ./scripts/train_backbones.sh             # DDP each backbone across 8 GPUs
#   PARALLEL_GPU=1 ./scripts/train_backbones.sh               # 8 backbones at once, 1 GPU each

set -euo pipefail

EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-16}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-8}
IMAGE_SIZE=${IMAGE_SIZE:-224}
LR=${LR:-3e-4}
HF_REPO_ID=${HF_REPO_ID:-ultemica/piyoshogi}
CKPT_ROOT=${CKPT_ROOT:-./runs}
SKIP_EXISTING=${SKIP_EXISTING:-0}
RESUME_INCOMPLETE=${RESUME_INCOMPLETE:-0}
COMPILE=${COMPILE:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29500}
PARALLEL_GPU=${PARALLEL_GPU:-0}
PARALLEL_LOG_DIR=${PARALLEL_LOG_DIR:-${CKPT_ROOT}/logs}

# Ordered small -> large so early results inform whether we need the heavies.
DEFAULT_BACKBONES=(
    mobilenet_v3_small
    mobilenet_v3_large
    convnext_atto
    convnext_femto
    efficientnet_b0
    convnext_pico
    efficientnet_b1
    convnext_nano
    convnext_tiny
)
if [[ -n "${BACKBONES:-}" ]]; then
    read -r -a BACKBONE_LIST <<< "${BACKBONES}"
else
    BACKBONE_LIST=("${DEFAULT_BACKBONES[@]}")
fi

extra_args=(--preload)
if [[ "${COMPILE}" == "1" ]]; then
    extra_args+=(--compile)
fi

# Mode-guard: parallel and DDP are mutually exclusive (parallel implies 1 GPU per run).
if [[ "${PARALLEL_GPU}" == "1" && "${NPROC_PER_NODE}" -gt 1 ]]; then
    echo "[sweep] error: PARALLEL_GPU=1 and NPROC_PER_NODE>1 are mutually exclusive." >&2
    exit 2
fi

detect_gpu_count() {
    uv run python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1
}
NGPUS=${NGPUS:-$(detect_gpu_count)}

echo "[sweep] epochs=${EPOCHS} batch=${BATCH_SIZE} workers=${NUM_WORKERS} image_size=${IMAGE_SIZE} lr=${LR}"
echo "[sweep] backbones: ${BACKBONE_LIST[*]}"
echo "[sweep] ckpt root: ${CKPT_ROOT}"
if [[ "${PARALLEL_GPU}" == "1" ]]; then
    echo "[sweep] parallel-GPU: ${NGPUS} GPUs, one backbone per GPU"
    echo "[sweep] per-run logs: ${PARALLEL_LOG_DIR}"
elif [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    echo "[sweep] multi-GPU DDP: nproc_per_node=${NPROC_PER_NODE} (torchrun)"
    echo "[sweep] effective batch = ${BATCH_SIZE} x ${NPROC_PER_NODE} = $(( BATCH_SIZE * NPROC_PER_NODE ))"
fi
echo

declare -a completed=()
declare -a skipped=()
declare -a failed=()

print_summary() {
    echo
    echo "[sweep] === summary ==="
    echo "[sweep] completed: ${completed[*]:-<none>}"
    echo "[sweep] skipped:   ${skipped[*]:-<none>}"
    echo "[sweep] failed:    ${failed[*]:-<none>}"
}

# Ctrl-C mid-loop should stop cleanly but still print a summary.
# In parallel mode, also kill any running child processes.
declare -a child_pids=()
cleanup_children() {
    for pid in "${child_pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap 'echo; echo "[sweep] interrupted"; cleanup_children; print_summary; exit 130' INT

final_ckpt_name=$(printf "epoch-%03d.pt" "${EPOCHS}")

# resolve_run_state — echoes one of:
#   SKIP   (final ckpt exists in RESUME_INCOMPLETE; or latest.pt in SKIP_EXISTING)
#   RESUME (partial ckpt found in RESUME_INCOMPLETE)
#   FRESH  (start from epoch 1)
resolve_run_state() {
    local latest="$1" final="$2"
    if [[ "${RESUME_INCOMPLETE}" == "1" ]]; then
        if [[ -f "${final}" ]]; then echo SKIP; return; fi
        if [[ -f "${latest}" ]]; then echo RESUME; return; fi
        echo FRESH; return
    fi
    if [[ "${SKIP_EXISTING}" == "1" && -f "${latest}" ]]; then
        echo SKIP; return
    fi
    echo FRESH
}

# run_single BACKBONE — launches a run in the FOREGROUND. Used by sequential
# and DDP modes. Returns the training exit code.
run_single() {
    local BB="$1"
    local model_name="board-ocr-${BB}"
    local ckpt_dir="${CKPT_ROOT}/${model_name}"
    local latest_ckpt="${ckpt_dir}/latest.pt"
    local final_ckpt="${ckpt_dir}/${final_ckpt_name}"

    local state
    state=$(resolve_run_state "${latest_ckpt}" "${final_ckpt}")

    local per_run_extra=()
    case "${state}" in
        SKIP)
            echo "[sweep] SKIP ${BB}"
            skipped+=("${BB}")
            return 0
            ;;
        RESUME)
            echo "[sweep] RESUME ${BB} from ${latest_ckpt}"
            per_run_extra+=(--resume latest)
            ;;
    esac

    echo
    echo "[sweep] ============================================================"
    echo "[sweep]  ${BB}  ->  ${ckpt_dir}"
    echo "[sweep] ============================================================"
    local start_ts
    start_ts=$(date +%s)

    local launcher=()
    if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
        launcher=(uv run torchrun
            --standalone
            --nproc_per_node="${NPROC_PER_NODE}"
            --master_port="${MASTER_PORT}"
            -m mito_train.training.train_board_ocr)
    else
        launcher=(uv run python -m mito_train.training.train_board_ocr)
    fi

    if "${launcher[@]}" \
        --hf-repo-id "${HF_REPO_ID}" \
        --mode full \
        --backbone "${BB}" \
        --image-size "${IMAGE_SIZE}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --prefetch-factor "${PREFETCH_FACTOR}" \
        --lr "${LR}" \
        --model-name "${model_name}" \
        "${extra_args[@]}" \
        "${per_run_extra[@]}"
    then
        local elapsed=$(( $(date +%s) - start_ts ))
        echo "[sweep] ${BB} DONE in ${elapsed}s"
        completed+=("${BB}")
    else
        local rc=$?
        echo "[sweep] ${BB} FAILED (exit ${rc})"
        failed+=("${BB}")
    fi
}

# launch_parallel BACKBONE GPU_IDX — starts a background run pinned to one GPU.
# Its stdout+stderr is redirected to per-backbone log file. Sets $!.
launch_parallel() {
    local BB="$1" gpu="$2"
    local model_name="board-ocr-${BB}"
    local ckpt_dir="${CKPT_ROOT}/${model_name}"
    local latest_ckpt="${ckpt_dir}/latest.pt"
    local final_ckpt="${ckpt_dir}/${final_ckpt_name}"
    local log_file="${PARALLEL_LOG_DIR}/${BB}.log"

    local state
    state=$(resolve_run_state "${latest_ckpt}" "${final_ckpt}")

    local per_run_extra=()
    case "${state}" in
        SKIP)
            echo "[sweep] SKIP ${BB}"
            skipped+=("${BB}")
            return 1  # signal caller: no process started
            ;;
        RESUME)
            echo "[sweep] RESUME ${BB} on GPU ${gpu} (log: ${log_file})"
            per_run_extra+=(--resume latest)
            ;;
        FRESH)
            echo "[sweep] START  ${BB} on GPU ${gpu} (log: ${log_file})"
            ;;
    esac

    # CUDA_VISIBLE_DEVICES pins this subprocess to a single GPU without touching
    # the parent env. torch inside the child sees exactly one device as cuda:0.
    CUDA_VISIBLE_DEVICES="${gpu}" \
    uv run python -m mito_train.training.train_board_ocr \
        --hf-repo-id "${HF_REPO_ID}" \
        --mode full \
        --backbone "${BB}" \
        --image-size "${IMAGE_SIZE}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --prefetch-factor "${PREFETCH_FACTOR}" \
        --lr "${LR}" \
        --model-name "${model_name}" \
        "${extra_args[@]}" \
        "${per_run_extra[@]}" \
        >"${log_file}" 2>&1 &
    return 0
}

if [[ "${PARALLEL_GPU}" == "1" ]]; then
    mkdir -p "${PARALLEL_LOG_DIR}"

    # Wave-based scheduler: launch up to NGPUS runs, wait for all to finish,
    # then start the next wave. Simple and predictable; a continuous
    # dispatcher would be slightly faster but hides failure modes.
    idx=0
    total=${#BACKBONE_LIST[@]}
    wave=0
    while (( idx < total )); do
        wave=$(( wave + 1 ))
        echo
        echo "[sweep] --- wave ${wave} ---"
        wave_pids=()
        wave_backbones=()
        for (( g=0; g<NGPUS && idx<total; g++ )); do
            BB="${BACKBONE_LIST[idx]}"
            idx=$(( idx + 1 ))
            if launch_parallel "${BB}" "${g}"; then
                wave_pids+=("$!")
                wave_backbones+=("${BB}")
                child_pids+=("$!")
            fi
        done

        # Wait per-pid so we can attribute exit codes back to backbone names.
        for i in "${!wave_pids[@]}"; do
            pid="${wave_pids[i]}"
            BB="${wave_backbones[i]}"
            if wait "${pid}"; then
                echo "[sweep] ${BB} DONE"
                completed+=("${BB}")
            else
                rc=$?
                echo "[sweep] ${BB} FAILED (exit ${rc}); see ${PARALLEL_LOG_DIR}/${BB}.log"
                failed+=("${BB}")
            fi
        done
        child_pids=()
    done
else
    for BB in "${BACKBONE_LIST[@]}"; do
        run_single "${BB}"
    done
fi

print_summary
