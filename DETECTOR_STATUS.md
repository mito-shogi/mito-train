# board-detector: 現状と仕様書の差分

`docs/ocr-model-interface.md` §2 の仕様と、現在 `mito_train` にある `BoardDetector` 実装の差分を解説する。ONNX 出力・MITO 側との interface 交渉・拡張タスクの見積もりを行う際の一次資料。

**作成: 2026-07-13 / 対象実装: `runs/board-detector-v1/latest.pt` (100 epoch, 強 aug 版)**

---

## 1. 一言で

| 側面 | 仕様書 (§2) | 現実装 |
|---|---|---|
| 出力領域 | 3 領域を個別に返す (盤 4 隅 / 先手駒台 / 後手駒台) | **ShogiBoardView 全体**を単一 axis-aligned bbox で返す |
| 出力次数 | 16 (float32) | 4 (float32) |
| 盤面表現 | 4 隅 (perspective 復元可能) | axis-aligned bbox |
| 持ち駒領域 | 先手/後手それぞれ独立 bbox | (含まれない、親矩形内に相対配置) |
| 入力サイズ | 224×224 | 384×384 (224 でも並行学習予定) |

**現状の設計思想**: スクショは常に axis-aligned で描画され、盤と駒台の相対位置は device 固有に決定的なので、「親矩形 1 個」→「client 側で固定比率で分割」で必要情報を提供できる。仕様書は写真撮影・perspective 対応まで踏み込んだ将来ビジョン。

---

## 2. 仕様書 (`docs/ocr-model-interface.md` §2) の要求

### 2.1 入力
- Tensor 名: `image`
- Shape: `(1, 3, 224, 224)`
- Dtype: `float32`
- 値域: `[0.0, 1.0]` (RGB / 255)
- チャネル順: RGB、レイアウト: NCHW
- 前処理: アス比保持で長辺 224 に fit、余白は黒 (0.0) padding

### 2.2 出力
- Tensor 名: `regions`
- Shape: `(1, 16)`
- 値域: `[0.0, 1.0]` (letterbox 後 224×224 空間の正規化座標)
- 16 要素の内訳 (順序固定):

| index | 内容 |
|---|---|
| 0, 1 | 盤面 左上 (x, y) |
| 2, 3 | 盤面 右上 (x, y) |
| 4, 5 | 盤面 右下 (x, y) |
| 6, 7 | 盤面 左下 (x, y) |
| 8, 9 | 先手持ち駒 左上 (x, y) |
| 10, 11 | 先手持ち駒 右下 (x, y) |
| 12, 13 | 後手持ち駒 左上 (x, y) |
| 14, 15 | 後手持ち駒 右下 (x, y) |

持ち駒が写っていない場合、該当 4 要素すべてに `-1.0`。

### 2.3 仕様書が 4 隅を要求する理由
> 「透視変換で正規化するため 4 隅すべて必要」

= 将来 **写真撮影 (画面を斜めから撮った画像)** に対応する余地を残すため。axis-aligned bbox では perspective distortion を復元できない。

---

## 3. 現実装 (`mito_train`)

### 3.1 コード配置
- モデル: `mito_train/models/board_detector.py::BoardDetector`
- 学習: `mito_train/training/train_detector.py`
- Dataset (local): `mito_train/datasets/detector_dataset.py::DetectorDataset`
- Dataset (HF): `mito_train/datasets/hf_detector_dataset.py::HFDetectorDataset`
- Aug: `build_detector_transform(mode, image_size)` (同ファイル内)
- Export: `mito_train/export/to_onnx.py` (現状 shape `(1, 3, 256, 256)` で 384/224 いずれとも不整合)

### 3.2 モデル構造 (`BoardDetector`)
```
features:  torchvision.models.mobilenet_v3_small (pretrained)
pool:      AdaptiveAvgPool2d(1)
head:      Linear(576, 128) -> ReLU -> Linear(128, 4) -> Sigmoid
```
- 出力: `(B, 4)` = 正規化 bbox `(x1, y1, x2, y2)` in `[0, 1]`
- 座標系: 入力画像 (letterbox 済み正方形) の絶対比率
- パラメータ: 約 1.1M (mobilenet_v3_small 由来)

### 3.3 入力
- 学習時 `--image-size 384` (default) / `--image-size 224` (仕様書一致想定)
- 前処理: `A.LongestMaxSize(image_size) + A.PadIfNeeded(image_size, image_size, black)`
  - 仕様書の letterbox と等価
- Normalization: ImageNet mean/std を適用 → 仕様書の `[0, 1]` とは異なる
  - ONNX 出力時に仕様書と合わせるなら、前処理から Normalize を外す (or 前処理を分離) 必要あり

### 3.4 学習データ (HF `ultemica/piyoshogi::detector_paired`)
- 行構造: `{sfen, hash, images: [PIL×4], devices: [str×4], bboxes: [[x1,y1,x2,y2]×4]}`
- 各行 1 SFEN を 4 device (iPhone10,1 / 11,8 / 15,4 / iPad14,10) で render
- サイズ: train 800 行 × 4 device = 3200 sample、val 200 × 4 = 800 sample
- **`bboxes` = ShogiBoardView 全体の axis-aligned bbox** (後手駒台 + 盤面 9×9 + 先手駒台 を含む親矩形)
- 4 隅 label / 持ち駒個別 bbox label は存在しない

### 3.5 学習実績 (2026-07-13, `wandb: mito-train / detector / os5vtwqm`)
- 設定: `mobilenet_v3_small` / w384 / epochs=100 / batch=32 / lr=3e-4 / cosine + no warmup
- Aug: アス比歪め (CropAndPad positive-only) + 縦横独立 Affine scale + BboxSafeCrop + 強画質劣化 + HSV
  - 回転・shear・perspective なし (スクショは軸並行前提)
- 最終指標 (val):
  - `iou_mean` peak **0.9906** (epoch 094), final **0.9870** (epoch 099)
  - `@0.5 / @0.75 / @0.9` すべて **1.000** (末尾 20 epoch すべて)

---

## 4. 差分の詳細

### 4.1 入力サイズ (224 vs 384)
- 224 と 384 の両方で学習予定 → 実質的に問題なし
- モデル構造上、mobilenet backbone は任意解像度対応
- ONNX 出力時に対応する image_size で export すればよい

### 4.2 出力次数 (4 vs 16)
- 現実装 4 値: 単一 bbox `(x1, y1, x2, y2)`
- 仕様書 16 値: 盤面 4 隅 (8) + 先手持ち駒 bbox (4) + 後手持ち駒 bbox (4)
- **情報の質的差**:
  - 盤面 4 隅 → 現状の axis-aligned スクショでは bbox の 4 頂点と数学的に等価。写真撮影対応時に差が出る
  - 持ち駒 bbox → **現ラベルには存在しない情報**。piyo-hook 側で追加 label 生成が必要

### 4.3 出力する矩形の意味
| | 仕様書 | 現実装 |
|---|---|---|
| 盤面領域 | 9×9 grid の外接矩形 | ShogiBoardView 全体 (駒台 + 盤 込み) |
| 分離アプローチ | モデル出力で 3 領域を分離 | 親矩形内を client 側で固定比率分割 |

**現実装の親矩形は、内部を device 固有の比率で切ることで、仕様書の 3 領域を導出可能** (device_bboxes.json などから相対比率を保持する前提)。

### 4.4 値域と Normalization
- 仕様書: 入力 `[0.0, 1.0]` (÷255 のみ)
- 現実装: ImageNet mean/std で正規化 → モデル入力は centered around 0
- ONNX 出力時に前処理を仕様書に合わせる場合、モデルの最終形態を再検討する必要あり (fine-tune or Normalize を吸収する層を追加)

### 4.5 座標系
- 仕様書: letterbox 後 224×224 空間の正規化座標 `[0, 1]`
- 現実装: letterbox 後 `{image_size}×{image_size}` 空間の正規化座標 `[0, 1]`
- **値の意味は同じ** (letterbox 済み正方形での比率)

---

## 5. なぜ現状で(実務上)足りているか

1. **スクショは常に axis-aligned** — piyo将棋アプリの UI は矩形描画のみ、写真撮影を伴わない限り透視歪みなし
2. **`board_view_px` は駒台+盤の親矩形** — 3 領域すべてがこの中に収まる
3. **device 固有の内部比率は決定的** — iPhone10,1 の駒台高さ / 盤高さの比は UI コード由来で固定。データからも学習可能だが、静的テーブルで持つ方が確実

つまり **単一親矩形 + 静的な内部比率テーブル** = 仕様書の 3 領域出力と情報量的に等価。

---

## 6. 拡張が必要になる条件

| 条件 | 必要な追加 | コスト |
|---|---|---|
| 写真撮影 (斜め撮り) 対応 | 盤面 4 隅 label + perspective 歪みを含む学習データ | piyo-hook 側で写真撮影データ収集、または synthetic 生成 |
| Android / 他プラットフォーム対応 | device 数を増やす、UI レイアウトの多様化 | データ収集 |
| device 固有比率テーブルが破綻する UI 変更 (piyo将棋アプリ更新等) | 持ち駒 bbox の直接学習 | piyo-hook 側で持ち駒 bbox label 生成 |

現状これらは **すべて発生していない**。したがって現実装での運用が可能。

---

## 7. 仕様書に合わせて拡張する場合の作業

### 7.1 出力 head の拡張
- `BoardDetector.head` の最終 Linear を `Linear(128, 4)` → `Linear(128, 16)` に変更
- 学習 loss:
  - 盤面 4 隅 (8 値): SmoothL1 (現行と同じ設計)
  - 持ち駒 bbox (8 値): mask 付き SmoothL1 (未検出時は `-1.0` sentinel で drop)

### 7.2 label 側の準備
- 盤面 4 隅: 現データの bbox から機械的に導出可能 (axis-aligned なので `(x1,y1), (x2,y1), (x2,y2), (x1,y2)`)
- 持ち駒 bbox: **piyo-hook 側での新規 label 生成が必要**。または device 固有比率テーブルから合成

### 7.3 ONNX 出力
- 入力 shape を仕様書に合わせて `(1, 3, 224, 224)` に固定
- 出力 name を `regions`、shape `(1, 16)` に
- `to_onnx.py::MODEL_SPECS["board"]` を更新 (現状 256×256 で不整合)
- Normalize 層をモデルに含めるか、interface 側で吸収するかを決める

### 7.4 コスト概算
- head 拡張 + 学習コード修正: 半日
- piyo-hook 側の持ち駒 label 生成: piyo-hook の作業次第、要相談
- 再学習: 100 epoch = 約 40 分 (A100 1 枚)
- ONNX 出力 + 契約検証: 半日

---

## 8. 参考: 現行学習済みモデルの位置づけ

`runs/board-detector-v1/latest.pt` は仕様書に対する **プロトタイプ / MVP**。in-distribution 4 iOS device の ShogiBoardView 検出タスクで `iou_mean 0.99+ / @0.9=1.000` を達成しており、実装確認・パイプライン疎通・下流モデルの入力供給には十分。

仕様書 §2 準拠版に拡張するタイミングは、以下いずれかが発生してから:
- 写真撮影対応が必要になった
- 持ち駒領域を個別に検出したい要件が出た
- 未見 device での親矩形→3 領域分割精度が問題化した
