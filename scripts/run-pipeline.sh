#!/usr/bin/env bash
# end-to-end 学習パイプライン (PoC 段階)。
set -euo pipefail

./scripts/download-data.sh

python scripts/verify-gpu.py

python -m mito_train.training.train_piece \
  --data ./data/piyo-train \
  --epochs "${EPOCHS:-5}" \
  --out ./runs/piece-cnn

python -m mito_train.export.to_onnx \
  --target piece \
  --checkpoint ./runs/piece-cnn/best.pt \
  --out ./models/piece-classifier-v1.onnx

./scripts/upload-models.sh
