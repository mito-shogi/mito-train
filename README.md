# mito-train

ぴよ将棋OCR の 3 モデル (board-detector / piece-classifier / hand-classifier) を
PyTorch で学習し、契約 (`docs/ocr-model-interface.md`) 準拠の ONNX を R2 に配置するレポ。

- **学習・ONNX 出力** … このレポ (Python / PyTorch)
- **データ生成** … piyo-hook (別レポ)
- **推論・UI・デプロイ** … MITO (別レポ, TypeScript)

## セットアップ

```bash
# 1. rclone (データ pull 用)
brew install rclone            # macOS
# sudo apt install rclone      # Linux
rclone config                  # r2 remote を設定 (.rclone.conf.example 参照)

# 2. Python 環境 (uv)
uv sync                        # pyproject.toml から依存を解決し .venv に構築

# 3. データ pull (R2 → data/)
./scripts/r2/download-data.sh

# 4. device 動作確認
uv run python scripts/verify_gpu.py

# 5. PoC 学習を試す
uv run python -m mito_train.training.train_piece --data ./data/piyo-train --epochs 5

# 6. ONNX 出力
uv run python -m mito_train.export.to_onnx --model piece

# 7. モデルを R2 に push
./scripts/r2/upload-models.sh
```

依存の追加は `uv add <pkg>` で行う (pyproject.toml を直接編集しない)。

## ディレクトリ

| パス | 役割 |
|---|---|
| `mito_train/datasets/` | manifest を読む Dataset / augmentation |
| `mito_train/models/` | 3 モデル定義 |
| `mito_train/training/` | 学習エントリポイント |
| `mito_train/export/` | ONNX 出力 / INT8 量子化 |
| `mito_train/eval/` | 評価指標 / test-realistic 評価 |
| `scripts/` | データ・モデルの R2 同期、GPU 確認 |
| `data/` | R2 から pull (Git 管理外) |
| `docs/` | MITO から同期する仕様書 (Git 管理) |

## データの状態 (2026-07-07 時点)

- `data/d0/` に PNG が約 29,920 枚 (22GB) 配置済み。
- **`manifest-*.jsonl` / `annotations.json` は未配信**。piyo-hook 側の生成・アップロード待ち。
  manifest が届くまで学習ループ本体は動かせない (モデル生成の疎通確認までは可能)。

## 環境メモ

- この devcontainer は **aarch64 Linux + GPU 無し**。torch は CPU 版。
  実学習は 4070 Ti / 5060 / M3 Max などの GPU マシンで行う想定。
