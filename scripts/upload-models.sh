#!/usr/bin/env bash
# 学習済み ONNX モデルを R2 に push する。
set -euo pipefail

REMOTE=${REMOTE:-r2}
BUCKET=${BUCKET:-mito-datasets}
MODEL_DIR=${MODEL_DIR:-./models}

echo "Uploading ONNX models to ${REMOTE}:${BUCKET}/models/ ..."
rclone copy "${MODEL_DIR}" "${REMOTE}:${BUCKET}/models/" \
  --progress \
  --include "*.onnx"

echo "Done. Latest models on remote:"
rclone ls "${REMOTE}:${BUCKET}/models/"
