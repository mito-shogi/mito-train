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
| Cloudflare Workers + workers-wonnx (WebGPU) | mobilenet_v3_small〜convnext_atto | R2 にモデル置いて WebGPU 経由。CPU 推論より現実的 |
| Cloudflare Workers 純 CPU 推論 | mobilenet_v3_small (int8) | GPU 無しで実用ラインギリ、Paid tier 必須 |
| Cloudflare Workers AI（ホスト済み） | 該当なし（現状） | カタログ限定、BYOM は 2026 半ば時点で未 GA |
| ブラウザ（初回DL重視） | mobilenet_v3_small / large, convnext_atto | int8 で数MB、初回ロード軽い |
| ブラウザ（精度重視） | convnext_nano | int8 15MB、精度/サイズのバランス最良 |
| 2024+ フラグシップスマホ | convnext_tiny | NPU性能余裕、精度上限狙える |
| 2024+ ミッドレンジスマホ | convnext_nano | 旧機種互換も考慮した実用ライン |
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

## デプロイ先別の詳細

### Cloudflare Workers（自前ONNXモデル推論）

2026年時点、Cloudflareで自前 ONNX を動かす選択肢は3つ。それぞれ制約が違うので使い分けが必要〜。

#### ルート A: `workers-wonnx` パターン（推奨）

公式サンプル [`cloudflare/workers-wonnx`](https://github.com/cloudflare/workers-wonnx) が示す方法。WebGPU ベースの ONNX ランタイム WONNX を Worker 上で動かし、モデル本体は R2 バケットから fetch する。

- **GPU 使えるので推論はそこそこ速い**（WebGPU 経由）
- **モデルサイズ制約回避**: R2 に置くから Worker のバンドルサイズ制限に縛られない
- **BoardOCR の場合の実用度**: convnext_tiny クラスは重すぎるので、**mobilenet_v3_small〜convnext_atto が現実的**
- 推論時間目安: mobilenet_v3_small int8 で **20〜50ms**、convnext_atto で **80〜200ms**（WebGPU 経由）

#### ルート B: Workers AI カタログ（該当モデル無し）

Cloudflare 側で GPU 実行してくれる仕組み。ただし利用可能モデルはカタログ限定で、**カスタム ONNX は現状 GA していない**（BYOM がロードマップにはいるがまだ未提供、2026年半ば時点）。

- カタログにあるのは Llama, Whisper, Stable Diffusion 系
- **将棋盤 OCR 用途は該当なし** → 使えない
- 将来 BYOM 対応したら convnext_tiny クラスまで載る想定

過去に「Constellation」というカスタム ONNX 実行サービスがあったが、Workers AI に統合され独立プロダクトとしては終息。

#### ルート C: Worker CPU 上で ONNX Runtime Web / WASM

昔ながらの方法。**GPU 使えないので実用性はかなり厳しい**。

- **CPU 時間**: Free 10ms（推論不可）/ Paid tier で 30秒 (bundled) / 5分 (unbound)
- **メモリ**: 128MB
- **推論時間の見積もり**（Worker CPU で 224x224 1枚推論）
  - mobilenet_v3_small int8: **200〜400ms** — Paid tier ならぎりぎり
  - efficientnet_b0 int8: **500ms〜1s** — 苦しい
  - convnext_atto 以上: **1s超** — Unbound のみ実用

### Workers 料金の目安（2026年時点）

- **Workers Paid**: 月$5、10M req + 30M CPU-ms 込み
- 超過分: 1M req あたり$0.30、1M CPU-ms あたり$0.02
- **Workers AI**: 1,000 Neurons あたり$0.011（無料枠1日10,000 Neurons）
  - Neurons = リクエスト実行に必要な GPU コンピュートの単位
  - カスタム ONNX（`workers-wonnx`）は Workers AI ではなく Workers 側の課金体系
- **R2 ストレージ**: モデルファイル置き場、10GB/月まで無料、以降 $0.015/GB/月

**BoardOCR のケーススタディ**
- mobilenet_v3_small int8 (1.1MB) を R2 に置いて workers-wonnx で提供
- 1回の推論: ~30ms CPU + R2 fetch（キャッシュ効くから初回のみ）
- 1M リクエスト/月 想定: Worker Paid $5 + CPU 超過分ほぼゼロ = **~$5/月で運用可能**
- ただし GPU コンピュート単価は今後変動可能性あり、公式ドキュメント要確認

### 2024年以降のスマートフォン

処理性能が急速に上がり、モデル選択の自由度が上がった帯。

#### iPhone 15 Pro / iPhone 16（A17/A18/A18 Pro）

- **Neural Engine**: 17〜38 TOPS
- **RAM**: 8GB
- **推論経路**: CoreML → Neural Engine or Metal GPU
- **推論時間**（BoardOCR 224x224 fp16 想定）
  - convnext_tiny: **~15〜30ms**
  - convnext_nano: **~10〜20ms**
  - mobilenet_v3_small: **~3〜8ms**
- **上限モデル**: convnext_tiny 余裕、**convnext_small (50M) も動く**
- **推奨**: convnext_tiny（精度優先） or convnext_nano（アプリ容量抑えたい）

#### iPhone 15 / 16（A16/A18 non-Pro）

- **Neural Engine**: 15.8〜17 TOPS
- **RAM**: 6〜8GB
- Pro とほぼ同等の推論時間
- 上限は convnext_tiny、convnext_small はメモリ的にギリ

#### Snapdragon 8 Gen 3 / 8 Gen 4（2024年 Android フラグシップ）

- Galaxy S24, Xiaomi 14 系列
- **NPU (Hexagon)**: 45+ TOPS
- **推論経路**: NNAPI / QNN / ONNX Runtime Mobile + XNNPACK
- **推論時間**（BoardOCR 224x224 fp16 想定）
  - convnext_tiny: **~20〜40ms**
  - convnext_nano: **~15〜25ms**
- iPhone 15 Pro クラスと同程度の能力
- 推奨: convnext_tiny

#### Google Pixel 9 / 9 Pro（Tensor G4）

- **TPU (Edge TPU 派生)**: 高い量子化推論性能
- int8 量子化と相性◎
- convnext_tiny int8: **~30〜50ms**
- 推奨: convnext_tiny (int8)

#### 2024年ミッドレンジ Android（Snapdragon 7 Gen 3 / Dimensity 7300 級）

- **NPU**: 10〜20 TOPS
- **推論時間**
  - convnext_tiny: **~80〜150ms**（許容範囲だが体感重くなる）
  - convnext_nano: **~40〜80ms**
  - efficientnet_b0: **~20〜40ms**
- **推奨**: convnext_nano。convnext_tiny は動くが体感速度が犠牲になるケースあり

#### まとめ表（2024+ スマホ想定）

| 端末クラス | 推奨 backbone | 推論時間 | 備考 |
|---|---|---|---|
| iPhone 15/16 Pro | convnext_tiny | ~20ms | 精度優先で余裕 |
| iPhone 15/16 (無印) | convnext_tiny | ~30ms | 標準的選択 |
| Snapdragon 8 Gen 3+ | convnext_tiny | ~30ms | Android旗艦 |
| Pixel 9 (Tensor G4) | convnext_tiny (int8) | ~40ms | int8 との相性◎ |
| ミッドレンジ2024 | convnext_nano | ~40〜80ms | tiny は体感重い |
| 旧機種互換重視 | convnext_atto / mobilenet_v3_large | ~30〜100ms | 2020年頃のミッド機種でも動く |

**共通の実装ポイント**
- **CoreML / ONNX 変換**: `torch.onnx.export` → 端末側 SDK でロード
- **fp16 量子化**: モバイル NPU/GPU で最も相性が良い、精度低下は無視できる
- **int8 量子化**: サイズと速度に効くが、キャリブレーションデータ必要
- **画像入力の pre-processing**: `LongestMaxSize + PadIfNeeded + Normalize` を端末側で再現。ImageNet mean/std をハードコード

## 推論時間の比較表（横断）

224x224 入力・1画像あたりの推論時間の**目安**。数値は他モデルの公開ベンチと BoardOCR のパラメータ規模から推定した見積もりで、実測値ではない。

| Backbone | Worker CPU 推論 | Worker + workers-wonnx (WebGPU) | iPhone 15/16 Pro (NE) | Snapdragon 8 Gen 3 (NPU) | Pixel 9 (TPU int8) | ミッドレンジ Android 2024 |
|---|---|---|---|---|---|---|
| mobilenet_v3_small | **200〜400ms** | 20〜50ms | 3〜8ms | 5〜10ms | 5〜10ms | 15〜30ms |
| mobilenet_v3_large | 400〜800ms | 40〜100ms | 5〜12ms | 8〜15ms | 8〜15ms | 25〜50ms |
| convnext_atto | 800ms〜2s | 80〜200ms | 6〜12ms | 10〜18ms | 10〜18ms | 30〜60ms |
| efficientnet_b0 | 500ms〜1s | 50〜120ms | 5〜10ms | 8〜15ms | 8〜15ms | 20〜40ms |
| convnext_femto | 1〜3s | 100〜250ms | 8〜15ms | 12〜20ms | 12〜20ms | 40〜80ms |
| efficientnet_b1 | 800ms〜1.5s | 80〜180ms | 8〜15ms | 12〜20ms | 12〜20ms | 30〜60ms |
| convnext_pico | 2〜5s | 200〜500ms | 10〜18ms | 15〜25ms | 15〜25ms | 60〜120ms |
| convnext_nano | 5〜10s | 400ms〜1s | 10〜20ms | 15〜25ms | 15〜30ms | 40〜80ms |
| convnext_tiny | 10〜30s | 800ms〜2s | 15〜30ms | 20〜40ms | 30〜50ms | 80〜150ms |

**読み取りポイント**
- **Worker CPU 推論**: convnext_atto より重いモデルはリクエスト毎の CPU 時間制約でほぼ実用不可
- **workers-wonnx (WebGPU)**: CPU 推論の 10〜20倍速。convnext_atto までは実用範囲
- **モバイル NPU/TPU 経由**: convnext_tiny でも 30〜50ms、体感 60fps は無理でも 15〜20fps は出る
- **ミッドレンジスマホでのラインは convnext_nano**: tiny だと 100ms 超で操作もっさり
- サーバ GPU 推論なら全 backbone が数ms〜十数ms で完結（比較対象外だが参考として）

## 精度と実用性の関係

「val sfen_full が何%なら実用に耐えるか」は用途で決まる。以下は将棋盤 OCR で想定される要件レベル。

### 用途別の実用ライン

| 用途 | 必要 sfen_full acc | 理由 |
|---|---|---|
| **カジュアル閲覧**（画像から盤面表示、参考程度） | **~85%以上** | 誤認識してもユーザーが手動確認できる |
| **プレイ再現・棋譜記録** | **~95%以上** | 手が進むごとに再入力面倒、8割成功でも運用崩壊 |
| **エンジン解析への入力** | **~99%以上** | 盤面1マス違えば局面全く変わる。プロ棋士は絶対に許容しない |
| **公式棋譜化・研究用途** | **~99.5%以上** | 人手校正コスト削減の意味を成すライン |

### cell_acc と sfen_full の関係（重要）

`sfen_full` は「盤面81マス全部 + 持ち駒14スロット全部」正解の率。**セル単位の精度がわずかに下がるだけで sfen_full は急落する**。

理論値（独立仮定）：

| board cell_acc | 理論 sfen_full (81マス独立仮定) |
|---|---|
| 95% | 1.5% |
| 97% | 8.5% |
| 99% | 44% |
| 99.5% | **66%** |
| 99.8% | **85%** |
| 99.9% | **92%** |

**実測はこの理論値より高くなる**（同じ画像内での間違いが空間相関するため）。ただし cell_acc 97% と 99% では sfen_full が 数十%〜数倍差になる。

**教訓**: cell_acc を 99%台後半まで押し上げないと、実用ラインに乗らない。

### 何が精度を押し上げるか

sfen_full を伸ばす手段の効き順（BoardOCR での経験則）：

1. **image_size の増（224 → 288 → 384）**: 一番効きやすい。9x9 マス識別に解像度が直接効く
2. **モデル容量の増（backbone を上げる）**: 特に細部識別（成香 vs 成桂、と金 vs 金）で効く
3. **augmentation の適正化**: SNS 劣化への頑健性（cell_acc への直接寄与は小さいが実運用差で効く）
4. **hand_mode = "regression"** への切替: 持ち駒枚数のオーディナル情報を活かす
5. **エポック数の増**: 逓減あるが、cell_acc 97% → 98% への最後の詰めで効く
6. **蒸留**: 上限に近付いた後の最後の押し上げ手段

### 実用的な妥協点

| 想定シナリオ | 現実的な組み合わせ |
|---|---|
| 「まず動くもの」 | mobilenet_v3_small, image_size=224, sfen_full ~50%狙い |
| 「棋譜記録に使える」 | convnext_nano, image_size=288, sfen_full ~90%狙い |
| 「エンジン解析まで」 | convnext_tiny, image_size=384, sfen_full ~99%狙い |
| 「公式棋譜化」 | convnext_tiny + 蒸留, image_size=384, sfen_full 99.5%+ |

### 実運用での妥協策

100% は理論上不可能に近い。運用側で吸収する仕組みも視野に：

- **信頼度スコア表示**: 「このマスは確信度低め」と UI で示し、人手校正を促す
- **候補提示**: argmax だけでなく top-3 を出す、間違い時に選び直しやすく
- **文脈補正**: 「将棋のルール上、この位置に相手の玉は無い」等の後処理
- **A/B の複数モデル投票**: convnext_tiny + efficientnet_b0 の合議で頑健性↑

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
