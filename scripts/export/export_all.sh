#!/usr/bin/env bash
# Export detector + BoardOCR backbones to ONNX in fp32 / fp16 / int8.
#
# Ships all files to ./models/ ready for R2/CDN upload:
#   board-detector-mnv3s-w384-{fp32,fp16,int8}.onnx
#   board-ocr-mobilenet_v3_large-w384-{fp32,fp16,int8}.onnx
#   board-ocr-efficientnet_b1-w384-{fp32,fp16,int8}.onnx
#   board-ocr-convnext_nano-w384-{fp32,fp16}.onnx   # int8 skipped (ConvNeXt LayerNorm)
#
# Usage:
#   ./scripts/export/export_all.sh                        # detector + 3 OCR backbones
#   BACKBONES="efficientnet_b1" ./scripts/export/export_all.sh  # subset
#   SKIP_INT8=1 ./scripts/export/export_all.sh            # fp32 + fp16 only
#   SKIP_FP16=1 ./scripts/export/export_all.sh            # fp32 + int8 only
#
# Env overrides:
#   BACKBONES        space-separated OCR backbones (default 3)
#   CKPT_ROOT        checkpoint search root (default ./runs)
#   OUT_DIR          output dir (default ./models)
#   DETECTOR_CKPT    detector checkpoint (default runs/board-detector-v1/latest.pt)
#   SKIP_DETECTOR    non-empty to skip detector export
#   SKIP_INT8        non-empty to skip int8 quantization
#   SKIP_FP16        non-empty to skip fp16 conversion

set -euo pipefail

CKPT_ROOT=${CKPT_ROOT:-./runs}
OUT_DIR=${OUT_DIR:-./models}
DETECTOR_CKPT=${DETECTOR_CKPT:-runs/board-detector-v1/latest.pt}
BACKBONES=${BACKBONES:-"mobilenet_v3_large efficientnet_b1 convnext_nano"}
SKIP_DETECTOR=${SKIP_DETECTOR:-}
SKIP_INT8=${SKIP_INT8:-}
SKIP_FP16=${SKIP_FP16:-}

mkdir -p "${OUT_DIR}"

export_one() {
    local prefix="$1" ckpt="$2" onnx_type="$3"
    local fp32_path="${OUT_DIR}/${prefix}-fp32.onnx"

    echo "[export_all] === ${prefix} ==="

    # fp32 baseline (always produced; other precisions derive from it).
    uv run python -m mito_train.export.to_onnx \
        --checkpoint "${ckpt}" --model "${onnx_type}" --out "${fp32_path}" \
        2>&1 | grep -E "wrote|epoch=" | tail -3
    if [[ ! -f "${fp32_path}" ]]; then
        echo "[export_all] ERROR: fp32 export failed for ${prefix}" >&2
        return 1
    fi

    if [[ -z "${SKIP_FP16}" ]]; then
        # keep-io-types leaves input/output as fp32 so the client doesn't need
        # to convert. Weights + intermediates become fp16, halving download.
        uv run python -m mito_train.export.to_fp16 \
            --input "${fp32_path}" --keep-io-types \
            --output "${OUT_DIR}/${prefix}-fp16.onnx" 2>&1 | tail -1
    fi

    if [[ -z "${SKIP_INT8}" ]]; then
        # ConvNeXt's LayerNorm-heavy graph trips onnxruntime.quantization's
        # shape inference. Retry without preprocessing; if still fails, skip.
        if ! uv run python -m mito_train.export.quantize \
                --input "${fp32_path}" \
                --output "${OUT_DIR}/${prefix}-int8.onnx" 2>&1 | tail -1
        then
            echo "[export_all] int8 quantization failed for ${prefix} — skipping" >&2
        fi
    fi
}

if [[ -z "${SKIP_DETECTOR}" ]]; then
    if [[ -f "${DETECTOR_CKPT}" ]]; then
        export_one "board-detector-mnv3s-w384" "${DETECTOR_CKPT}" "board_detector"
    else
        echo "[export_all] detector ckpt not found: ${DETECTOR_CKPT} (skipping)" >&2
    fi
fi

for bb in ${BACKBONES}; do
    ckpt="${CKPT_ROOT}/board-ocr-${bb}/latest.pt"
    if [[ ! -f "${ckpt}" ]]; then
        echo "[export_all] OCR ckpt not found: ${ckpt} (skipping ${bb})" >&2
        continue
    fi
    export_one "board-ocr-${bb}-w384" "${ckpt}" "board_ocr"
done

echo
echo "[export_all] === summary ==="
ls -lh "${OUT_DIR}"/*.onnx 2>/dev/null | awk '{printf "  %-8s  %s\n", $5, $9}'
