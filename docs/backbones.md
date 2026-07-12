# Backbones

BoardOCR がサポートする 9 個のバックボーンを、パラメータ数・想定用途・配信先ごとに整理する。全モデル `BoardOCR` から `--backbone <name>` で切替可能。

## 一覧

| Backbone | Params | fp32 | fp16 | int8 | 出典 | 特徴 |
|---|---|---|---|---|---|---|
| [mobilenet_v3_small](#mobilenet_v3_small) | 1.1M | 4.4 MB | 2.2 MB | ~1.1 MB | torchvision | 最軽量 |
| [mobilenet_v3_large](#mobilenet_v3_large) | 3.3M | 13 MB | 6.5 MB | ~3.3 MB | torchvision | small の上位 |
| [convnext_atto](#convnext_atto) | 3.5M | 14 MB | 7 MB | ~3.5 MB | timm | ConvNeXt最小 |
| [efficientnet_b0](#efficientnet_b0) | 4.4M | 18 MB | 9 MB | ~4.4 MB | torchvision | 定番中庸 |
| [convnext_femto](#convnext_femto) | 4.9M | 20 MB | 10 MB | ~4.9 MB | timm | ConvNeXt 次点 |
| [efficientnet_b1](#efficientnet_b1) | 6.9M | 28 MB | 14 MB | ~6.9 MB | torchvision | b0 の上位 |
| [convnext_pico](#convnext_pico) | 8.7M | 35 MB | 17 MB | ~8.7 MB | timm | ConvNeXt 小 |
| [convnext_nano](#convnext_nano) | 15.1M | 61 MB | 30 MB | ~15.1 MB | timm | Pareto中庸 |
| [convnext_tiny](#convnext_tiny) | 28.0M | 112 MB | 56 MB | ~28.0 MB | torchvision | モバイル上限 |

数値は `pretrained=False` で `BoardOCR(backbone=...)` を組んだときのパラメータ総数。int8 は「近似バイト数」であり、実際は量子化スキームで前後する。

## 用途マトリクス

| 配信先 | 推奨 backbone | 理由 |
|---|---|---|
| ブラウザ（初回DL重視） | mobilenet_v3_small / large, convnext_atto | int8 で数MB、初回ロード軽い |
| ブラウザ（精度重視） | convnext_nano | int8 15MB、精度/サイズのバランス最良 |
| ネイティブアプリ（iOS/Android） | convnext_tiny | 精度上限を狙える現実的サイズ |
| サーバ推論のみ | convnext_tiny | 拡張余地あり（convnext_small まで検討可） |
| 高速オンデバイス推論 | efficientnet_b0 | Compound scaling で少パラでも精度良 |
| 実験のベースライン | mobilenet_v3_small | 学習速度・容量ともに軽く鉄板 |

## モデル別詳細

### mobilenet_v3_small

- **Params**: 1.1M / **out channels**: 576
- **出典**: `torchvision.models.mobilenet_v3_small`
- **アーキ**: Inverted residuals + Squeeze-and-Excite + h-swish activation
- **特徴**: 2019年 Google 設計。Neural Architecture Search で mobile 向けに直接最適化。最軽量クラス。
- **強み**: 学習/推論ともに超高速、int8 で 1MB切る、ブラウザで一瞬でロード。
- **弱み**: 特徴容量が小さく、細部識別（成香/成桂/と金 の漢字区別）が伸びにくい。
- **想定用途**: パイプライン検証、smoke test、ブラウザで「とりあえず動く」ライン。
- **注意**: 現状 sweep で val cell_acc 96.9% 頭打ち報告あり。容量律速の可能性大。

### mobilenet_v3_large

- **Params**: 3.3M / **out channels**: 960
- **出典**: `torchvision.models.mobilenet_v3_large`
- **アーキ**: small と同系統、より広いチャネル・深い block 構成
- **特徴**: mobilenet_v3_small のスケールアップ版。同じ設計思想で 3 倍の容量。
- **強み**: small の学習パイプラインそのまま流用可、int8 で 3MB とまだ軽い。
- **弱み**: efficientnet_b0 と比べると精度/パラの効率で若干劣る（新しい設計思想の差）。
- **想定用途**: small の精度で足りない時の**最小限のスケールアップ**。

### convnext_atto

- **Params**: 3.5M
- **出典**: `timm.create_model("convnext_atto")`
- **アーキ**: 4-stage ConvNeXt を極限まで縮小（depth=[2,2,6,2], dims=[40,80,160,320]）
- **特徴**: 2022年 Facebook AI の ConvNeXt を極小化。LayerNorm + GELU + depthwise 7x7 conv の現代アーキ。
- **強み**: ConvNeXt 系の効率良い設計を最軽量帯に持ってきたもの。BN 不使用なので DDP でも SyncBN 気にせず。
- **弱み**: timm 依存。
- **想定用途**: mobilenet_v3_large と同じサイズ帯で ConvNeXt アーキを試したい時。

### efficientnet_b0

- **Params**: 4.4M / **out channels**: 1280
- **出典**: `torchvision.models.efficientnet_b0`
- **アーキ**: Compound scaling で width/depth/resolution を同時に最適化した MBConv ベース
- **特徴**: 2019年 Google、当時 SOTA の精度/パラ効率。後継の EfficientNet-V2 もあるが b0 の完成度が高く定番として残る。
- **強み**: 4.4M で ImageNet top-1 77% 級。この容量帯のリファレンス実装。
- **弱み**: SE ブロックの計算がやや重い。DWConv の並列度が若干控えめ。
- **想定用途**: 「小さいけど精度も欲しい」の第一候補。ブラウザ配信に強い。

### convnext_femto

- **Params**: 4.9M
- **出典**: `timm.create_model("convnext_femto")`
- **アーキ**: ConvNeXt をやや大きくした版（depth=[2,2,6,2], dims=[48,96,192,384]）
- **特徴**: atto と pico の間のサイズ。ConvNeXt 系の精度/パラのスイートスポット候補。
- **強み**: LayerNorm ベースで DDP フレンドリー。int8 で 5MB。
- **想定用途**: efficientnet_b0 のカウンタパート実験。ConvNeXt vs EfficientNet の比較の重要ピボット。

### efficientnet_b1

- **Params**: 6.9M
- **出典**: `torchvision.models.efficientnet_b1`
- **アーキ**: b0 を compound scaling で 1 段引き上げた版
- **特徴**: 入力解像度 240x240 で pretrained（BoardOCR では 224 リサイズで使用）。
- **強み**: b0 より数%高い精度、それでも 7M で mobile 圏。
- **弱み**: b0 との差が僅かで、コスト比の効率悪化傾向。
- **想定用途**: b0 で足りず nano まで行きたくない時のギャップ埋め。

### convnext_pico

- **Params**: 8.7M
- **出典**: `timm.create_model("convnext_pico")`
- **アーキ**: ConvNeXt（depth=[2,2,6,2], dims=[64,128,256,512]）
- **特徴**: 10M 前後の中量級 ConvNeXt。
- **強み**: femto の 2 倍の容量、精度がぐっと上がる帯。
- **想定用途**: mobile と server の中間帯を狙うとき。ブラウザだと少し重いが実用範囲。

### convnext_nano

- **Params**: 15.1M
- **出典**: `timm.create_model("convnext_nano")`
- **アーキ**: ConvNeXt（depth=[2,2,8,2], dims=[80,160,320,640]）
- **特徴**: ConvNeXt-Tiny を「肉薄しつつ半分のサイズ」に絞ったもの。
- **強み**: **Pareto Front の中央**。int8 で 15MB、convnext_tiny の精度に大きく譲らない可能性。
- **弱み**: timm 依存。
- **想定用途**: **ブラウザ配信の本命候補**。精度が求められて 30MB の初回DLが許容できるならこれ。

### convnext_tiny

- **Params**: 28.0M / **out channels**: 768
- **出典**: `torchvision.models.convnext_tiny`
- **アーキ**: 4-stage ConvNeXt（depth=[3,3,9,3], dims=[96,192,384,768]）
- **特徴**: 2022年 Facebook AI の ConvNeXt-Tiny。Transformer 系に迫る精度を ConvNet で達成。
- **強み**: 精度上限。ImageNet top-1 82%。torchvision 標準実装で ecosystem 対応◎。
- **弱み**: int8 でも 28MB でモバイル配信の実用上限、旧機種でメモリ厳しめ。fp16 だと 56MB。
- **想定用途**: **モバイル配信の上限**。ネイティブアプリなら現実解。ブラウザは fp16/int8 前提。
- **注意**: これより大きい convnext_small (50M) はブラウザには重い。ネイティブでも flagship 機限定。

## 選び方の判断樹

```
用途は？
├─ 実験/検証        → mobilenet_v3_small
├─ ブラウザ配信
│   ├─ 精度優先     → convnext_nano (int8 15MB)
│   └─ サイズ優先   → mobilenet_v3_large or efficientnet_b0 (int8 3-5MB)
├─ ネイティブアプリ  → convnext_tiny (int8 28MB)
└─ サーバ推論のみ
    ├─ 十分なら     → convnext_tiny
    └─ さらに精度   → 蒸留 or image_size 増 (backboneはtiny固定)
```

## パラメータ以外に効く要素

同じ backbone でも以下で精度が変わる：

- **image_size**: 224 → 384 で cell_acc 数%改善する可能性。将棋盤 9x9 の識別は解像度律速。
- **hand_mode**: `classification`（デフォ）vs `regression`。持ち駒枚数のオーディナル情報を活かすなら regression。
- **augmentation**: SNS 圧縮/ノイズを想定して調整可（`capture_dataset.py` の `build_transform`）。
- **preload**: 学習速度に効くが精度には無影響。

## 蒸留 (Knowledge Distillation) の余地

「convnext_tiny では足りないがサイズは上げたくない」場合の常道：

1. サーバで convnext_small / base を teacher として学習
2. mobile 向け convnext_nano / tiny を student として、teacher の soft label で学習
3. パラメータ数を保ったまま数%精度向上

BoardOCR では未実装だが、モバイル配信で精度を詰めたい場合の次のカードとして有力。

## 実測データ（今後追加）

各 backbone の val cell_acc / sfen_full_acc / wall-clock は sweep 完了後に W&B で並び、ここに引用予定。

- W&B project: `mito-train-board-ocr`
- run 名は各 backbone 名そのまま（例: `mobilenet_v3_small`, `convnext_tiny`）
