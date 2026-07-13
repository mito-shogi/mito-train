# ブラウザ WebGPU 推論時間の見積もり (2026-07-13)

**目的**: 5 backbone × 各種端末での **単一 forward 推定時間** を並べ、ブラウザ WebGPU 展開時の backbone 選定材料にする。全て **推定値** で実測ではない。realistic 精度は [`w384-sweep-analysis.md#realistic-評価-2026-07-13`](./w384-sweep-analysis.md#realistic-評価-2026-07-13) を参照。

前提の指標定義は [`ocr-metrics.md`](./ocr-metrics.md)、モデル仕様は [`backbones.md`](./backbones.md)、Cloudflare Workers 側 (server WebGPU) は [`backbones.md#推論時間の比較表横断`](./backbones.md#推論時間の比較表横断) にある。ここでは **ユーザー端末のブラウザ WebGPU** に絞る。

## 前提と限界

- 数字は **単一 forward 推定** (fp32、w384 入力、ort-web の WebGPU backend or 同等)
- **画像 decode + preprocessing (30〜100ms)** と **WebGPU 初期化 warmup (初回 200ms〜2s)** は別途乗る
- ort-web WebGPU backend / workers-wonnx / Safari WebGPU の実装成熟度は 2026 現在も進化中で **同じ端末でもブラウザで 2〜3 倍ズレる**
- **ConvNeXt 系は FLOPs 比より 20〜50% 遅め**。LayerNorm + 7x7 DWConv の WebGPU kernel 最適化が甘い
- **Safari WebGPU** は Chrome 系より 20〜40% 遅い。iOS 17 以下は WebGPU 未対応 → WASM fallback で 5〜10 倍遅くなる
- **int8 動的量子化** はモデルサイズ半減だが速度メリットは薄い (WebGPU int8 kernel 未対応が多い)、**fp16 export** の方が 30〜40% 高速化に効く (対応環境限定)
- Android の WebGPU 品質は **Adreno > Mali > PowerVR** の順、Mali は Adreno より 30〜50% 遅い場合が多い

## 対象モデル一覧

| Model | Params | fp32 size | int8 size | realistic SFEN |
|---|---:|---:|---:|---:|
| detector (mnv3s w384) | 1.1M | 4.4 MB | ~1.1 MB | — (iou 0.99+) |
| mobilenet_v3_small | 1.1M | 4.4 MB | ~1.1 MB | 95.37% |
| mobilenet_v3_large | 3.3M | 13 MB | ~3.3 MB | 97.92% |
| **efficientnet_b1** | **6.9M** | **28 MB** | **~7 MB** | **99.67%** |
| convnext_nano | 15.1M | 61 MB | ~15 MB | 99.92% |
| convnext_tiny | 28.0M | 112 MB | ~28 MB | 99.92% |

## iGPU (デスクトップ・ラップトップ統合 GPU / Chrome or Edge WebGPU)

| Model | Apple M4 (10-core+) | Apple M2/M3 | Apple M1 | Intel Iris Xe / Arc | AMD 780M (RDNA3) | Intel UHD (旧世代) |
|---|---:|---:|---:|---:|---:|---:|
| detector (mnv3s) | 30〜50ms | 40〜80ms | 60〜100ms | 80〜150ms | 60〜120ms | 200〜400ms |
| mnv3s (OCR) | 30〜50ms | 40〜80ms | 60〜100ms | 80〜150ms | 60〜120ms | 200〜400ms |
| mnv3l | 50〜100ms | 80〜150ms | 120〜200ms | 200〜350ms | 150〜250ms | 500〜900ms |
| **effb1** | **80〜150ms** | **120〜220ms** | **180〜300ms** | **300〜500ms** | **220〜380ms** | **700〜1300ms** |
| convnext_nano | 250〜450ms | 400〜700ms | 600ms〜1.1s | 1.0〜1.8s | 800ms〜1.5s | 2.5〜5s |
| convnext_tiny | 400〜700ms | 600ms〜1.1s | 1.0〜1.7s | 1.5〜2.8s | 1.2〜2.2s | 4〜8s |

## iOS (Safari WebGPU / iOS 18+)

Safari の WebGPU は Chrome 系より 20〜40% 遅め、iOS 17 は制限付き/フラグ必須。iPhone/iPad は WebKit 縛りで Chrome iOS も内部は WebKit。

| Model | iPad Pro M4 | iPad Air M2 | iPad Pro M1 | iPad std (A16) | iPhone 16 Pro (A18 Pro) | iPhone 15 Pro (A17 Pro) | iPhone 14/13 Pro | iPhone 12 / SE3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| detector (mnv3s) | 30〜60ms | 50〜100ms | 80〜150ms | 150〜300ms | 60〜120ms | 100〜200ms | 150〜300ms | 300〜600ms |
| mnv3s (OCR) | 30〜60ms | 50〜100ms | 80〜150ms | 150〜300ms | 60〜120ms | 100〜200ms | 150〜300ms | 300〜600ms |
| mnv3l | 60〜120ms | 100〜200ms | 150〜300ms | 300〜600ms | 120〜250ms | 200〜400ms | 300〜600ms | 600ms〜1.2s |
| **effb1** | **100〜200ms** | **150〜300ms** | **250〜500ms** | **500ms〜1s** | **200〜400ms** | **300〜600ms** | **500ms〜1s** | **1〜2s** |
| convnext_nano | 300〜600ms | 500ms〜1s | 800ms〜1.6s | 1.6〜3.2s | 700ms〜1.4s | 1〜2s | 1.6〜3.2s | 3〜6s |
| convnext_tiny | 500ms〜1s | 800ms〜1.6s | 1.3〜2.6s | 2.5〜5s | 1.1〜2.2s | 1.6〜3.3s | 2.6〜5.2s | 5〜10s |

## Android (Chrome WebGPU / Android 12+)

Chrome Android の WebGPU は 2024 年正式リリースだが Adreno > Mali > PowerVR の順で品質差が大きい。

| Model | Snapdragon 8 Elite (2024+) | Snapdragon 8 Gen 3 / Tensor G4 | Snapdragon 8 Gen 2 / Tensor G3 | Snapdragon 7 Gen 3 (ミッド) | Snapdragon 6/4 系 (低スペ) |
|---|---:|---:|---:|---:|---:|
| detector (mnv3s) | 50〜100ms | 80〜150ms | 100〜200ms | 200〜400ms | 500ms〜1s |
| mnv3s (OCR) | 50〜100ms | 80〜150ms | 100〜200ms | 200〜400ms | 500ms〜1s |
| mnv3l | 100〜200ms | 150〜300ms | 200〜400ms | 400〜800ms | 1〜2s |
| **effb1** | **150〜300ms** | **250〜450ms** | **350〜650ms** | **700ms〜1.3s** | **1.5〜3s** |
| convnext_nano | 500ms〜1s | 800ms〜1.6s | 1.2〜2.4s | 2〜4s | 5〜10s |
| convnext_tiny | 800ms〜1.6s | 1.3〜2.6s | 2〜4s | 3〜6s | 8〜15s |

## 全パイプライン合計 = detector + OCR

上表は **単一モデル** の推定。実運用は detector (mnv3s) + OCR の 2 段推論なので、**OCR モデル時間に detector 時間 (同 mnv3s 行) を足す**。

| 端末例 | Backbone | detector | OCR | 合計/画像 |
|---|---|---:|---:|---:|
| iPhone 15 Pro (A17 Pro) | effb1 | 100〜200ms | 300〜600ms | **400〜800ms** |
| iPhone 16 Pro (A18 Pro) | convnext_nano | 60〜120ms | 700ms〜1.4s | **760ms〜1.5s** |
| Snapdragon 8 Gen 3 | effb1 | 80〜150ms | 250〜450ms | **330〜600ms** |
| iPad Pro M4 | convnext_nano | 30〜60ms | 300〜600ms | **330〜660ms** |
| MacBook M2 | effb1 | 40〜80ms | 120〜220ms | **160〜300ms** |
| Snapdragon 7 Gen 3 (ミッド) | mnv3l | 200〜400ms | 400〜800ms | **600ms〜1.2s** |

## 読み取り

1. **effb1 (7M)** は M-series iGPU / A17 Pro+ / SD 8 Gen 3+ で **フル pipeline < 1 秒**、UX 許容範囲。ミッドレンジ Android や旧 iPhone/iPad で 1〜3s で体感やや遅い。
2. **convnext_nano (15M)** は M-series と SD 8 Elite / iPhone 16 Pro 級でようやく **< 1 秒**。それ以下の端末では 2〜6s、**モバイル運用は厳しい**。
3. **mnv3l (3.3M)** は全体的にサクサク、旧端末でも 1s 以下で完結。realistic 97.92% を許容できる用途なら「どこでも動く」の魅力。
4. **cvtiny** はどの mobile でも重すぎ、実質デスクトップ M-series/dGPU 専用。
5. **mnv3s (OCR)** は realistic 95.37% で 99% ラインに届かない、**サイズは detector と同じなのに精度差 -4pt**、OCR 用としては落選。

## backbone 選定 (browser WebGPU 用途)

| 用途 | 推奨 backbone | 理由 |
|---|---|---|
| **本命 (デフォルト)** | **efficientnet_b1** | realistic 99.67% (99% ライン超え)、モバイル ミドル帯以上で < 1s、fp32 28 MB / int8 7 MB |
| デスクトップ精度優先 | convnext_nano | realistic 99.92% で iPad 100%、M-series では 400〜700ms で十分実用 |
| 低スペック端末 fallback | mobilenet_v3_large | realistic 97.92% で「棋譜記録」ライン、どこでも動く |
| 落選 | mnv3s / cvtiny | 前者は精度不足、後者はサイズと速度で mobile 不可 |

**2 段構え推奨**: WebGPU adapter info でデバイス能力判定 → M-series/フラグシップは **effb1** or **cvnano**、それ以下は **mnv3l** に自動切替。3 モデル全部を R2 に置いても int8 で合計 26 MB、CDN キャッシュ前提で許容範囲。

## 数字の前提

FLOPs @ w384 (推定):

| Model | GFLOPs |
|---|---:|
| detector (mnv3s) | 0.44 |
| mnv3s | 0.44 |
| mnv3l | 1.3 |
| effb1 | 1.8 |
| cvnano | 7.3 |
| cvtiny | 13 |

各デバイスの WebGPU 実効 FLOPS 目安 (推定):

| デバイス | 実効 GFLOPS |
|---|---:|
| Apple M4 GPU (Metal WebGPU) | ~2500 |
| Apple M2/M3 | ~1500 |
| Apple M1 | ~1000 |
| Intel Iris Xe / Arc iGPU | ~600 |
| AMD 780M (RDNA3) | ~900 |
| Intel UHD (旧世代) | ~150 |
| iPhone 16 Pro (A18 Pro / Safari) | ~500 |
| iPhone 15 Pro (A17 Pro / Safari) | ~350 |
| iPhone 14/13 Pro | ~250 |
| iPhone 12 / SE3 | ~120 |
| iPad Pro M4 | ~2500 |
| iPad Air M2 | ~1500 |
| iPad std A16 | ~250 |
| Snapdragon 8 Elite / Adreno 830 | ~1200 |
| Snapdragon 8 Gen 3 / Adreno 750 | ~800 |
| Snapdragon 8 Gen 2 / Adreno 740 | ~500 |
| Snapdragon 7 Gen 3 | ~250 |
| Snapdragon 6/4 系 | ~80 |

**推論時間 = FLOPs / 実効 GFLOPS + オーバーヘッド 20〜100ms** で概算。ConvNeXt は kernel 最適化不足でこの計算より 1.3〜1.5 倍遅く、Safari は Chrome 比 1.2〜1.4 倍遅い補正込み。

## 数字を上書きする方法 (実測)

これらは全て推定値で、実測とは 2〜3 倍ズレる可能性がある。実測手順:

1. 各 backbone を ONNX export (`mito_train/export/to_onnx.py --model board_ocr`)
2. ort-web を組んだ最小ページを用意し、各端末の Chrome/Safari WebGPU で計測
3. 100 回 warmup + 100 回本計測、中央値と 90 パーセンタイルを記録
4. 本 doc の該当セルを実測値で置換

実測次第で ConvNeXt が思ったより速い/遅い、Safari WebGPU がまだ制限的、といった判断が変わる可能性がある。**本 doc の数字は「発注の目安」であって「確定値」ではない** ことに注意。
