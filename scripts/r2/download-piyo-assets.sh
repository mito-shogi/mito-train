#!/usr/bin/env bash
# R2 → ./assets/piyo を rclone sync で pull する。
# ぴよ将棋 駒テンプレ画像 (14 デザイン × 30 駒 = 420 枚) を取得。
# 命名規則は docs/piyo-piece-templates.md §「ファイル名フォーマット」参照。
set -euo pipefail

REMOTE=${REMOTE:-r2}
BUCKET=${BUCKET:-mito-datasets}
ASSETS_DIR=${ASSETS_DIR:-./assets/piyo}

mkdir -p "$ASSETS_DIR"

echo "Pulling piyo templates from ${REMOTE}:${BUCKET}/piyo-assets/ ..."
rclone sync "${REMOTE}:${BUCKET}/piyo-assets/" "${ASSETS_DIR}/" \
  --progress \
  --transfers 8 \
  --checkers 16 \
  --fast-list \
  --include "k*/*.png"

echo "Done. Local assets:"
find "${ASSETS_DIR}" -type d -maxdepth 1 | sort
count=$(find "${ASSETS_DIR}" -type f -name "*.png" | wc -l)
echo "Total PNG count: ${count} (expected 420)"
