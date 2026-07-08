#!/usr/bin/env bash
# end-to-end 学習パイプライン: データ確認 → 学習 → ONNX 出力 → (任意) R2 push
set -euo pipefail

DATA_DIR=${DATA_DIR:-./data/piyo-train}
EPOCHS=${EPOCHS:-5}

echo "== 1. データ確認 =="
python -m mito_train.training.train_piece --data "${DATA_DIR}" --epochs "${EPOCHS}"

echo "== 2. ONNX 出力 =="
python -m mito_train.export.to_onnx --model piece

echo "== 3. (任意) R2 へ push =="
echo "  ./scripts/upload-models.sh  # 認証済みなら実行"
echo "Done."
