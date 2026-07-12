#!/usr/bin/env bash
set -euo pipefail

REMOTE=${REMOTE:-r2}
BUCKET=${BUCKET:-mito-datasets}
DATA_DIR=${DATA_DIR:-./data}

mkdir -p "$DATA_DIR"

echo "Pulling piyo-train from ${REMOTE}:${BUCKET}/piyo-train/ ..."
rclone sync "${REMOTE}:${BUCKET}/piyo-train/" "${DATA_DIR}/piyo-train/" \
  --progress \
  --transfers 8 \
  --checkers 16 \
  --fast-list

echo "Pulling piyo-test-realistic from ${REMOTE}:${BUCKET}/piyo-test-realistic/ ..."
rclone sync "${REMOTE}:${BUCKET}/piyo-test-realistic/" "${DATA_DIR}/piyo-test-realistic/" \
  --progress \
  --transfers 8 \
  --checkers 16 \
  --fast-list

echo "Done. Local data:"
find "${DATA_DIR}" -type f | head -10
du -sh "${DATA_DIR}"/*
