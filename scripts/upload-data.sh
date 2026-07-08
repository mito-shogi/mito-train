#!/usr/bin/env bash
set -euo pipefail

REMOTE=${REMOTE:-r2}
BUCKET=${BUCKET:-mito-datasets}
SOURCE=${SOURCE:-./assets/train}
MANIFEST=${MANIFEST:-./assets/manifest.jsonl}

echo "Uploading ${SOURCE}/*.png to ${REMOTE}:${BUCKET}/piyo-train/ ..."
rclone sync "${SOURCE}" "${REMOTE}:${BUCKET}/piyo-train/" \
  --progress \
  --transfers 8 \
  --checkers 16 \
  --exclude "*.json*" \
  --include "*.png"

echo "Uploading manifest ..."
rclone copy "${MANIFEST}" "${REMOTE}:${BUCKET}/piyo-train/"
if [ -f "./assets/manifest-train.jsonl" ]; then
  rclone copy "./assets/manifest-train.jsonl" "${REMOTE}:${BUCKET}/piyo-train/"
fi
if [ -f "./assets/manifest-valid.jsonl" ]; then
  rclone copy "./assets/manifest-valid.jsonl" "${REMOTE}:${BUCKET}/piyo-train/"
fi

echo "Done."
