# Model export & release パイプライン

BoardOCR / BoardDetector の PyTorch checkpoint を ONNX に export し、GitHub Release として配布するまでの手順。ブラウザ WebGPU 用途を主眼にしている。realistic 精度は [`w384-sweep-analysis.md`](./w384-sweep-analysis.md#realistic-評価-2026-07-13)、端末別推論見積は [`browser-inference-outlook.md`](./browser-inference-outlook.md)、client 側の入力仕様は [`ocr-model-interface.md`](./ocr-model-interface.md) を参照。

## 出力する成果物

`models/` に置く ONNX 一式:

| ファイル | 用途 |
|---|---|
| `board-detector-mnv3s-w384-{fp32,fp16,int8}.onnx` | 盤面検出 (親矩形 bbox 予測) |
| `board-ocr-mobilenet_v3_large-w384-{fp32,fp16}.onnx` | edge / WASM fallback tier |
| `board-ocr-efficientnet_b1-w384-{fp32,fp16}.onnx` | ブラウザ本命 (realistic 99.67%) |
| `board-ocr-convnext_nano-w384-fp32.onnx` | デスクトップ精度優先 tier |

`dist/` に置くリリース成果物:

| ファイル | 内容 |
|---|---|
| `models-vX.Y.Z.tar.gz` | 上記 ONNX を一括 tar.gz |
| `models-vX.Y.Z-MANIFEST.json` | ファイル → SHA256 / realistic 精度 / 推奨 tier のメタデータ |
| `models-vX.Y.Z-SHA256SUMS` | `sha256sum -c` 用の整合性チェック |
| `release-notes-vX.Y.Z.md` | GitHub Release body として使う Markdown |

## パイプライン全体

```
runs/board-*/latest.pt (PyTorch ckpt)
        │
        │  scripts/export/export_all.sh
        ▼
models/*.onnx (fp32 + fp16 + int8)
        │
        │  scripts/export/verify_onnx.py  (対 PyTorch numerics)
        ▼
確認済み ONNX
        │
        │  scripts/export/build_release.sh <version>
        │    ├─ write_manifest.py       (MANIFEST.json)
        │    ├─ sha256sum              (SHA256SUMS)
        │    ├─ tar -czf               (models-*.tar.gz)
        │    └─ write_release_notes.py (release-notes-*.md)
        ▼
dist/models-vX.Y.Z-*
        │
        │  gh release create (手動 or .github/workflows/deployment.yml)
        ▼
GitHub Release
```

## 個別スクリプト

### `mito_train/export/to_onnx.py`

PyTorch checkpoint を ONNX にエクスポート。

```bash
uv run python -m mito_train.export.to_onnx \
  --checkpoint runs/board-ocr-efficientnet_b1/latest.pt \
  --model board_ocr \
  --out models/board-ocr-efficientnet_b1-w384-fp32.onnx
```

- `--model` は `board_detector | board_ocr | piece | hand`。ckpt から backbone / image_size / hand_mode を読んで自動でモデル構築、state_dict load。
- **BoardOCR は dynamo exporter を使用** (AdaptiveAvgPool2d((9,9)) が trace exporter 非対応のため、`board_head` 内で入力 12x12 → 出力 9x9 に非整数比で pooling する構造がある)。
- **BoardDetector は trace exporter を使用** (dynamic batch を保ちつつ quantizer に食わせるため)。
- デフォルトは **batch=1 固定** (WebGPU 単画像推論前提)。`--dynamic-batch` で `batch` を動的軸に。

### `mito_train/export/to_fp16.py`

fp32 ONNX を fp16 に変換。

```bash
uv run python -m mito_train.export.to_fp16 \
  --input models/board-ocr-efficientnet_b1-w384-fp32.onnx \
  --keep-io-types \
  --output models/board-ocr-efficientnet_b1-w384-fp16.onnx
```

- `--keep-io-types`: 入出力 tensor は fp32 のまま、weights + 中間 activation のみ fp16 化。client 側で dtype 変換不要。
- **convnext_nano は変換自体は成功するが load 時に `Node (Loop) [TypeInferenceError]`**。onnxruntime の Loop op fp16 handling が未成熟。cvnano のみ fp32 提供に留める。

### `mito_train/export/quantize.py`

fp32 ONNX を int8 に動的量子化。

```bash
uv run python -m mito_train.export.quantize \
  --input models/board-detector-mnv3s-w384-fp32.onnx
```

- **OCR モデルへの動的量子化は実用不可**。weight のみ int8、activation は fp32 のまま計算するので、多層モデルで誤差蓄積 (mnv3l で board argmax 53/81、effb1 で 63/81 に崩壊)。
- detector int8 は生成できるが verify で max_diff 0.29 (bbox 正規化 [0,1] 空間で 29% 誤差)、実データでの IoU 未検証。**参考同梱** に留める。
- 実用 int8 が要る場合は **QDQ 静的量子化 + realistic 相当の calibration set** を別途実装 (未着手)。

### `scripts/export/export_all.sh`

detector + 3 OCR backbone を fp32/fp16/int8 で一括 export。

```bash
./scripts/export/export_all.sh

# サブセット
BACKBONES="efficientnet_b1" ./scripts/export/export_all.sh

# fp32 + fp16 のみ (int8 skip)
SKIP_INT8=1 ./scripts/export/export_all.sh
```

環境変数:

| 変数 | デフォルト | 意味 |
|---|---|---|
| `BACKBONES` | `"mobilenet_v3_large efficientnet_b1 convnext_nano"` | OCR backbone のスペース区切り |
| `CKPT_ROOT` | `./runs` | checkpoint 検索ルート |
| `OUT_DIR` | `./models` | 出力ディレクトリ |
| `DETECTOR_CKPT` | `runs/board-detector-v1/latest.pt` | detector checkpoint |
| `SKIP_DETECTOR` | (空) | 非空で detector export をスキップ |
| `SKIP_INT8` | (空) | 非空で int8 量子化をスキップ |
| `SKIP_FP16` | (空) | 非空で fp16 変換をスキップ |

### `scripts/export/verify_onnx.py`

PyTorch と ONNX の numerics 比較。

```bash
uv run python scripts/export/verify_onnx.py

# 一部だけ
uv run python scripts/export/verify_onnx.py \
  --backbones efficientnet_b1 --precisions fp32 fp16
```

出力: 各 (backbone, precision) について max_diff、argmax 一致数 (`board 81/81 hand 14/14` が全問正解)、ファイルサイズ。int8 の argmax 崩壊もここで検知できる。

### `scripts/export/build_release.sh`

`models/` の ONNX を tar + MANIFEST + release notes に packaging。

```bash
./scripts/export/build_release.sh v0.1.0
# または
VERSION=v0.1.0 ./scripts/export/build_release.sh
```

出力:
- `dist/models-v0.1.0.tar.gz`
- `dist/models-v0.1.0-MANIFEST.json` (`write_manifest.py` 経由)
- `dist/models-v0.1.0-SHA256SUMS`
- `dist/release-notes-v0.1.0.md` (`write_release_notes.py` 経由)

MANIFEST に含めるフィールド:
- 各ファイルの `size_bytes` / `sha256`
- OCR は `realistic_sfen_mean` + `realistic_sfen_per_device`
- detector は `detector_val_iou_mean`
- `recommendation` (browser 用途の tier)

Release notes は MANIFEST を単一ソースとして生成 → **精度数値は 1 箇所にしか書かない** ので drift しない。

### GitHub Actions: `.github/workflows/deployment.yml`

Release 発行を自動化。

**トリガー**:
- `workflow_dispatch` (手動): `version` input で `v0.1.0` 等を指定
- `push tags: models-v*.*.*`: タグ push で自動

**内部フロー**:
1. `actions/checkout@v4`
2. `astral-sh/setup-uv@v5`
3. version 解決 (dispatch input or tag suffix)
4. **`models_artifact` input が指定されていれば `actions/download-artifact` で `models/` を復元** (runner 上に checkpoints がない前提)
5. `uv sync`
6. `verify_onnx.py` (source ckpt 無くても warn で継続)
7. `build_release.sh <version>`
8. `gh release create models-<version>` with tar + MANIFEST + SHA256SUMS、body に release notes

**checkpoint の扱い**: workflow は runs/board-*/latest.pt を持たない前提で組んである。開発者が **ローカルで export_all.sh を回して確認 → models/ を actions/upload-artifact でアップロード → workflow_dispatch でリリース発行** のフローが基本。将来 HF Hub に weights を push すれば download step を足すだけで完全自動化できる。

## 手動フロー (今すぐやる場合)

CI に流さず手元で完結させる手順:

```bash
# 1. export (models/ に出力)
./scripts/export/export_all.sh

# 2. verify (argmax 一致確認)
uv run python scripts/export/verify_onnx.py

# 3. release bundle (dist/ に出力)
./scripts/export/build_release.sh v0.1.0

# 4. GitHub Release
gh release create models-v0.1.0 \
  --title "Models v0.1.0" \
  --notes-file dist/release-notes-v0.1.0.md \
  dist/models-v0.1.0.tar.gz \
  dist/models-v0.1.0-MANIFEST.json \
  dist/models-v0.1.0-SHA256SUMS
```

## Release の位置づけ

**GitHub Release は archive / traceability 用**、production 配信は **Cloudflare R2** 経由 (別インフラ、別レポ)。Release は:

- 版ごとに immutable な weights を残す (再現性)
- MANIFEST の精度数値と一緒に置くことで「この version の重みで realistic 何%」が後から追える
- 監査・比較のためのリファレンス

MITO client / R2 配信については本レポの範囲外。

## 現状の shipping 対応表 (2026-07-13)

| Model | fp32 | fp16 | int8 |
|---|:-:|:-:|:-:|
| detector (mnv3s) | ✓ | ✓ | △ (max_diff 0.29、参考同梱) |
| OCR mobilenet_v3_large | ✓ | ✓ | ✗ (argmax 53/81 崩壊) |
| OCR efficientnet_b1 | ✓ | ✓ | ✗ (argmax 63/81 崩壊) |
| OCR convnext_nano | ✓ | ✗ (Loop TypeInferenceError) | ✗ (LayerNorm shape inference 失敗) |

**推奨組み合わせ**:
- ブラウザ本命: detector fp16 + effb1 fp16 (合計 ~16 MB int8 相当のサイズ)
- 精度優先: detector fp16 + convnext_nano fp32 (合計 ~60 MB)
- 低スペック fallback: detector int8 (要 IoU 検証後) + mnv3l fp16 (合計 ~8 MB)
