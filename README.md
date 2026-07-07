# mito-train

MITO (ぴよ将棋 OCR) 用の PyTorch 学習パイプライン。

- **board-detector** — 盤面 4 隅回帰
- **piece-classifier** — 駒 29 クラス分類
- **hand-classifier** — 持ち駒 multi-head 分類

学習後は ONNX にエクスポートし、Cloudflare R2 経由で MITO 本体に配布する。

DevContainer は別途整備予定 (Python 3.11+ / PyTorch CUDA 12.4 想定)。

## クイックスタート

```bash
# 依存インストール (uv 推奨)
uv venv .venv
source .venv/bin/activate
uv sync --extra dev

# R2 認証設定
cp .rclone.conf.example .rclone.conf
# → access_key_id / secret_access_key / endpoint を埋める
export RCLONE_CONFIG=./.rclone.conf

# データ・アセット pull
./scripts/download-data.sh          # data/piyo-train, data/piyo-test-realistic
./scripts/download-piyo-assets.sh   # assets/piyo (14 デザイン × 30 駒)

# GPU 動作確認
python scripts/verify-gpu.py

# 学習 → ONNX 出力 → R2 push (現状 stub)
./scripts/run-pipeline.sh
```

## ドキュメント

| ファイル | 内容 |
|---|---|
| [`docs/mito-train-setup.md`](docs/mito-train-setup.md) | セットアップ全体 |
| [`docs/asset-layout.md`](docs/asset-layout.md) | assets/ と data/ の配置ルール |
| [`docs/ocr-model-interface.md`](docs/ocr-model-interface.md) | MITO ↔ mito-train モデル契約 (**遵守必須**) |
| [`docs/ocr-metrics.md`](docs/ocr-metrics.md) | 評価指標定義 |
| [`docs/piyo-data-generation-spec.md`](docs/piyo-data-generation-spec.md) | piyo-hook 側でのデータ生成仕様 |
| [`docs/piyo-piece-templates.md`](docs/piyo-piece-templates.md) | 駒テンプレの命名規則 |

## ディレクトリ構造

```
mito-train/
├── mito_train/          Python パッケージ (datasets / models / training / export / eval)
├── scripts/             R2 との pull/push shell + GPU 確認
├── notebooks/           Jupyter 実験用
├── tests/               pytest
├── docs/                仕様書 (MITO 側と同期)
├── assets/piyo/         駒テンプレ画像 (R2 から pull、Git 管理外)
├── data/                学習・評価データ (R2 から pull、Git 管理外)
└── models/              学習済み ONNX の出力先 (R2 に push、Git 管理外)
```

## 依存の方針

- `torch` / `torchvision` は `pyproject.toml` の `[tool.uv.sources]` で PyTorch CUDA 12.4 index を linux 限定で参照
- macOS (M3 Max) は PyPI 版で MPS が使えるので特別な設定不要
- `uv sync` で環境を作る前提。`pip` でも動くが動作確認は uv ベース

## Milestone

1. **M1** — 703 SFEN で piece-classifier の PoC 学習 (loss curve 確認)
2. **M2** — 3,000 SFEN で Piece Accuracy 95%+
3. **M3** — Twitter 劣化 augmentation スイープ
4. **M4** — test-realistic 100 枚に対する Exact-Match Rate 測定
5. **M5** — MITO 側 mock を実モデルに差し替え

詳細は `docs/mito-train-setup.md` §10 と `docs/ocr-phase2-plan.md` 参照。

## ライセンス

MIT
