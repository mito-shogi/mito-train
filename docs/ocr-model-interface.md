# MITO ↔ mito-train モデル契約書 (ONNX Interface Spec)

**発行日**: 2026-07-06
**版**: v1.0
**適用範囲**: MITO (推論・UI) と mito-train (学習・ONNX 出力) の間の **唯一の契約**
**関連文書**:
- `docs/ocr-phase2-plan.md` — Phase 2/3 全体計画
- `docs/ocr-metrics.md` — 精度指標
- `docs/piyo-data-generation-spec.md` — 教師データ生成仕様

---

## 0. この文書の目的

MITO と mito-train は別リポジトリで独立に開発する。両者を疎結合に保ちつつ確実に噛み合わせるため、**ONNX モデルの入出力形式** を厳密に定義する。この契約を守れば:

- mito-train はモデルの中身を自由に設計・変更できる
- MITO は推論コードを一度書けば、モデルが更新されても差し替えるだけで動く
- 両レポの CI が独立に回せる

**契約に変更が必要なときは本文書を版更新** (v1.1, v1.2...) し、両レポの実装を同時に更新する。

---

## 1. モデル一覧

| モデル ID | 役割 | Phase |
|---|---|---|
| `board-detector` | 画像から盤面 + 持ち駒領域の位置を検出 | Phase 2b |
| `piece-classifier` | 盤面 81 マス分の駒を分類 | Phase 2a |
| `hand-classifier` | 持ち駒領域から駒種と個数を検出 | Phase 2c |

3 モデルとも **ONNX Runtime Web (WebGPU EP)** で動くこと、および **INT8 量子化済みで配信** することが前提。

---

## 2. board-detector (盤面 + 持ち駒領域検出)

### 2.1 責務

任意の入力画像から、以下の 3 領域の座標を検出する:

1. **盤面 9×9** の 4 隅座標 (透視変換で正規化するため 4 隅すべて必要)
2. **先手持ち駒領域** の矩形 (axis-aligned bbox で十分)
3. **後手持ち駒領域** の矩形

### 2.2 入力仕様

| 項目 | 値 |
|---|---|
| Tensor 名 | `image` |
| Shape | `(1, 3, 224, 224)` |
| Dtype | `float32` |
| 値の範囲 | `[0.0, 1.0]` (RGB を 255 で割った値) |
| チャネル順 | RGB (Red, Green, Blue) の順 |
| メモリレイアウト | NCHW (batch, channel, height, width) |
| リサイズ方法 | アスペクト比を保持して長辺 224 に fit、余白は黒 (0.0) パディング |

**前処理コード例 (client 側)**:
```typescript
function preprocessBoardInput(bitmap: ImageBitmap): Float32Array {
  const canvas = new OffscreenCanvas(224, 224)
  const ctx = canvas.getContext('2d')!
  // 黒背景に画像を letterbox で貼る
  ctx.fillStyle = 'black'
  ctx.fillRect(0, 0, 224, 224)
  const scale = Math.min(224 / bitmap.width, 224 / bitmap.height)
  const w = bitmap.width * scale
  const h = bitmap.height * scale
  const dx = (224 - w) / 2
  const dy = (224 - h) / 2
  ctx.drawImage(bitmap, dx, dy, w, h)
  const imgData = ctx.getImageData(0, 0, 224, 224)
  // RGBA → RGB float32 [0,1] NCHW
  const out = new Float32Array(3 * 224 * 224)
  for (let y = 0; y < 224; y++) {
    for (let x = 0; x < 224; x++) {
      const i = (y * 224 + x) * 4
      out[0 * 224 * 224 + y * 224 + x] = imgData.data[i + 0] / 255
      out[1 * 224 * 224 + y * 224 + x] = imgData.data[i + 1] / 255
      out[2 * 224 * 224 + y * 224 + x] = imgData.data[i + 2] / 255
    }
  }
  return out
}
```

### 2.3 出力仕様

**単一出力テンソル** で 3 領域すべてを返す。

| 項目 | 値 |
|---|---|
| Tensor 名 | `regions` |
| Shape | `(1, 16)` |
| Dtype | `float32` |
| 値の範囲 | `[0.0, 1.0]` (letterbox 後の 224×224 空間における正規化座標) |

**16 要素の意味 (順序固定、この順で書き込むこと)**:

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

**画像に持ち駒が写っていない場合**: 該当領域の 4 要素すべてに `-1.0` を書き込む (client 側で「持ち駒未検出」として扱う)。

### 2.4 後処理 (client 側)

- 座標を letterbox 逆変換で原画像座標系に戻す
- 盤面の 4 隅から **透視変換行列** を計算し、288×288 の正規化盤面画像を作る
- 持ち駒領域は矩形として切り出し、後段の `hand-classifier` に渡す

### 2.5 精度目標

- **盤面 IoU@0.75**: 95%+ (test-realistic 100枚)
- **持ち駒領域 IoU@0.5**: 90%+
- 推論時間: WebGPU で **10ms 以下** (M3 Max)、WASM で **50ms 以下**

### 2.6 モデルサイズ制約

- FP32: **1.5MB 以下**
- INT8 量子化後: **400KB 以下**

---

## 3. piece-classifier (盤面 81 マス駒分類)

### 3.1 責務

透視変換後の 288×288 正規化盤面画像を 9×9 に分割した各マス (32×32) を、29 クラスに分類する。

### 3.2 入力仕様

| 項目 | 値 |
|---|---|
| Tensor 名 | `patches` |
| Shape | `(N, 3, 32, 32)` (N は動的、通常 81) |
| Dtype | `float32` |
| 値の範囲 | `[0.0, 1.0]` |
| チャネル順 | RGB |
| メモリレイアウト | NCHW |

**MITO 側は N=81 で 81 マス一括推論** を想定。dynamic axes として ONNX の `batch` 軸を可変にすること。

**前処理**:
- 288×288 の正規化盤面画像を 9×9 = 81 マス (各 32×32) に等分
- 順序: `patches[rank * 9 + fileIdx]` (rank=0..8 上→下、fileIdx=0..8 左→右 = 9筋→1筋)

### 3.3 出力仕様

| 項目 | 値 |
|---|---|
| Tensor 名 | `logits` |
| Shape | `(N, 29)` |
| Dtype | `float32` |
| 値の意味 | ソフトマックス前の生の logits |

**29 クラスの順序 (index → 意味、この順序を厳守)**:

| index | クラス | SFEN 表記 |
|---|---|---|
| 0 | 空マス | (無) |
| 1 | 先手 歩 | P |
| 2 | 先手 香 | L |
| 3 | 先手 桂 | N |
| 4 | 先手 銀 | S |
| 5 | 先手 金 | G |
| 6 | 先手 角 | B |
| 7 | 先手 飛 | R |
| 8 | 先手 王 | K |
| 9 | 先手 と | +P |
| 10 | 先手 成香 | +L |
| 11 | 先手 成桂 | +N |
| 12 | 先手 成銀 | +S |
| 13 | 先手 馬 | +B |
| 14 | 先手 龍 | +R |
| 15 | 後手 歩 | p |
| 16 | 後手 香 | l |
| 17 | 後手 桂 | n |
| 18 | 後手 銀 | s |
| 19 | 後手 金 | g |
| 20 | 後手 角 | b |
| 21 | 後手 飛 | r |
| 22 | 後手 王 | k |
| 23 | 後手 と | +p |
| 24 | 後手 成香 | +l |
| 25 | 後手 成桂 | +n |
| 26 | 後手 成銀 | +s |
| 27 | 後手 馬 | +b |
| 28 | 後手 龍 | +r |

このマッピングは **MITO 側の `src/ocr/model-classes.ts` にも定数として実装** され、両レポで同期する。

### 3.4 後処理 (client 側)

- 各マスの logits に `softmax` を適用して信頼度 (0..1) を得る
- `argmax` で最終予測クラス
- 信頼度が閾値 (例: 0.5) 未満の場合、UI 上で「要確認」ハイライト
- 予測結果を 9×9 の SFEN 盤面配列に組み立て

### 3.5 精度目標

- **Piece Accuracy** (81マス平均、test-realistic): **99%+**
- **Exact-Match Rate** (盤面 9×9 全マス一致): 60% (Phase 2a 単独)、90% (Phase 2c 完了時)
- 推論時間: WebGPU で **8ms 以下** (81マスバッチ)、WASM で **50ms 以下**

### 3.6 モデルサイズ制約

- FP32: **1.6MB 以下**
- INT8 量子化後: **400KB 以下**

---

## 4. hand-classifier (持ち駒認識)

### 4.1 責務

`board-detector` で切り出された先手/後手の持ち駒領域画像から、7 種類の駒 (飛/角/金/銀/桂/香/歩) の **存在有無 + 個数 (0〜18)** を検出する。

### 4.2 入力仕様

| 項目 | 値 |
|---|---|
| Tensor 名 | `hand_image` |
| Shape | `(1, 3, 96, 384)` |
| Dtype | `float32` |
| 値の範囲 | `[0.0, 1.0]` |
| チャネル順 | RGB |
| メモリレイアウト | NCHW |

**画像形状**: 高さ 96、幅 384 の横長画像。持ち駒 UI は 7 スロットが横に並ぶ想定でこのアスペクト比。

**前処理**:
- `board-detector` の出力する持ち駒領域 bbox を切り出し
- アスペクト比を保持して 96×384 に letterbox リサイズ (余白は黒)

### 4.3 出力仕様

**2 つの出力テンソル** を返す (multi-head 分類):

#### 4.3.1 駒種存在

| 項目 | 値 |
|---|---|
| Tensor 名 | `presence` |
| Shape | `(1, 7)` |
| Dtype | `float32` |
| 値の意味 | 各駒種の存在有無 logits (sigmoid 前) |

**7 要素の順序**: `[R, B, G, S, N, L, P]` (飛, 角, 金, 銀, 桂, 香, 歩)。SFEN 慣習の駒順に合わせる。

#### 4.3.2 駒個数

| 項目 | 値 |
|---|---|
| Tensor 名 | `counts` |
| Shape | `(1, 7, 19)` |
| Dtype | `float32` |
| 値の意味 | 各駒種の個数 (0〜18) を 19 クラス分類する logits |

**存在しない駒種の counts は無視** (`presence` の sigmoid が閾値未満なら個数 0 として扱う)。

### 4.4 後処理 (client 側)

```typescript
const presenceSigmoid = sigmoid(presenceOutput)  // (7,)
const counts = argmax(countsOutput, dim=-1)      // (7,) の整数値

const hand: Record<string, number> = {}
const kinds = ['R', 'B', 'G', 'S', 'N', 'L', 'P']
for (let i = 0; i < 7; i++) {
  if (presenceSigmoid[i] > 0.5) {
    hand[kinds[i]] = counts[i]
  }
}
```

### 4.5 精度目標

- 駒種存在 F1: **99%+**
- 駒個数 exact match: **98%+**
- 持ち駒領域全体一致率: **95%+**
- 推論時間: WebGPU で **5ms 以下** (2 領域 = 先後)

### 4.6 モデルサイズ制約

- FP32: **800KB 以下**
- INT8 量子化後: **200KB 以下**

---

## 5. 配信仕様

### 5.1 ファイル配置

MITO の Cloudflare Pages で以下のパスに配置:

```
public/models/
├── board-detector-v1.onnx    (INT8 量子化済み)
├── piece-classifier-v1.onnx  (INT8 量子化済み)
└── hand-classifier-v1.onnx   (INT8 量子化済み)
```

### 5.2 バージョニング

- ファイル名に `-v{メジャー}` を含める
- **メジャーバージョン更新** = 入出力形式変更 = 本契約書の版更新
- **マイナー更新** (再学習で精度向上のみ) = ファイル差し替え、ファイル名変更なし
- MITO 側の推論コードはメジャーバージョンをハードコード

### 5.3 総サイズ制約

3 モデル合計 (INT8 量子化後):

- 合計: **1MB 以下** (gzip 圧縮後 700KB 目安)
- 初回ロード体験: 4G で 2秒以下

### 5.4 モデル配布フロー

1. mito-train が `.onnx` を生成
2. mito-train の CI が Cloudflare R2 の `models/` prefix に push
3. MITO の CI (or 手動デプロイ) が R2 → `public/models/` に pull
4. MITO のデプロイ (Cloudflare Pages) でユーザーに配布

---

## 6. ONNX 変換要件 (mito-train 側の実装制約)

### 6.1 対応 opset

- **opset_version = 17** を使用 (2023年時点のスタンダード、ONNX Runtime Web WebGPU EP が完全対応)

### 6.2 使ってよい Op

**推奨 (WebGPU EP で高速)**:
- `Conv`, `ConvTranspose`
- `BatchNormalization`
- `Relu`, `LeakyRelu`, `Sigmoid`, `Tanh`
- `MaxPool`, `AveragePool`, `GlobalAveragePool`
- `Gemm`, `MatMul`
- `Add`, `Mul`, `Sub`, `Div`
- `Softmax`, `LogSoftmax`
- `Reshape`, `Transpose`, `Concat`, `Split`
- `Cast`
- `Resize` (bilinear のみ)

**避けるべき (WebGPU EP 未対応 or 低速)**:
- `LayerNormalization` (WASM fallback になる)
- `GRU`, `LSTM`, `Attention`
- 独自 op / TorchScript の未対応 op
- `NonMaxSuppression` (盤面検出で bbox 使う場合は後処理を client 側でやる)

### 6.3 dynamic axes

- `piece-classifier` の batch 軸は **動的** ({0: "batch"})
- `board-detector` の batch 軸は **固定** (常に 1)
- `hand-classifier` の batch 軸は **固定** (常に 1)

### 6.4 量子化

- **`onnxruntime.quantization.quantize_dynamic`** で INT8 動的量子化
- `weight_type=QuantType.QUInt8`
- 量子化後に **精度低下 0.5% 以内** を維持
- 精度落ちる場合は **static quantization** (calibration data 必要) に切り替え

### 6.5 検証項目 (mito-train の CI で実施)

以下をすべて満たしたモデルのみ配信:

- [ ] `onnx.checker.check_model` が pass
- [ ] `onnxruntime.InferenceSession` でロードできる (CPU EP)
- [ ] MITO と同じ前処理・後処理を経て正解率が train/val に近い値
- [ ] ファイルサイズ制約 (Section 2.6, 3.6, 4.6) を満たす
- [ ] 推論時間 (Node.js の onnxruntime WASM で計測、Section 2.5, 3.5, 4.5) を満たす

---

## 7. MITO 側の実装契約

### 7.1 実装するモジュール

```
src/ocr/
├── model-classes.ts        29クラスの順序定数
├── inference/
│   ├── board-detector.ts   board-detector 呼び出し
│   ├── piece-classifier.ts piece-classifier 呼び出し
│   ├── hand-classifier.ts  hand-classifier 呼び出し
│   └── session-manager.ts  ONNX Runtime Web セッション共通管理
├── preprocess/
│   ├── letterbox.ts        board-detector 用の letterbox
│   ├── perspective.ts      透視変換 (Canvas 2D)
│   └── slice-cells.ts      288x288 → 81 x 32x32
└── postprocess/
    ├── decode-regions.ts   board-detector 出力 → ROI
    ├── decode-pieces.ts    piece-classifier 出力 → 盤面配列
    └── decode-hands.ts     hand-classifier 出力 → 持ち駒
```

### 7.2 テストデータ

MITO のリポジトリに `src/ocr/__fixtures__/` を用意し、以下を配置:

- **dummy モデル**: 契約通りの入出力形式で常に固定値を返すダミー ONNX
  - `board-detector-dummy.onnx`
  - `piece-classifier-dummy.onnx`
  - `hand-classifier-dummy.onnx`
- **テストベクター**: 入力画像 → 期待出力 (JSON)

これで **mito-train のモデル完成を待たずに MITO 側の実装が進められる**。

### 7.3 契約違反の検出

MITO の CI で以下を検証:

- モデルロード時に `session.inputNames` と `session.outputNames` が本契約と一致するか
- shape が契約通りか
- モデルサイズが Section 5.3 の制約を満たすか

不一致なら **配信をブロック** して mito-train 側に通知。

---

## 8. 変更管理

### 8.1 契約変更のプロセス

1. 変更提案を GitHub Issue で共有 (MITO と mito-train の両レポ)
2. 両レポの担当者が合意
3. 本文書を更新 (版番号インクリメント)
4. 両レポの実装を更新
5. 両レポの CI が pass することを確認してから main にマージ

### 8.2 マイナー変更 (契約に影響しない)

以下は契約変更にあたらない (勝手にやってよい):

- モデルの内部構造変更 (入出力形式は同じ)
- 学習データ増量、augmentation 調整
- INT8 量子化パラメータ調整
- 精度改善 (数値の向上のみ)

### 8.3 破壊的変更 (契約に影響する)

以下は必ず本文書の版更新が必要:

- 入出力テンソルの名前・shape・dtype 変更
- クラス順序の変更
- 新モデルの追加
- 前処理・後処理の変更

---

## 9. 変更履歴

| 版 | 日付 | 変更内容 |
|---|---|---|
| v1.0 | 2026-07-06 | 初版作成。3 モデルの入出力形式を確定。|

---

## 10. 参考: 実装イメージ (MITO 側)

```typescript
// src/ocr/pipeline.ts (簡略化)
import * as ort from 'onnxruntime-web/webgpu'
import { preprocessBoardInput } from './preprocess/letterbox'
import { perspectiveTransform } from './preprocess/perspective'
import { sliceCells } from './preprocess/slice-cells'
import { decodeRegions } from './postprocess/decode-regions'
import { decodePieces } from './postprocess/decode-pieces'
import { decodeHands } from './postprocess/decode-hands'

export async function recognize(bitmap: ImageBitmap) {
  const [boardSession, pieceSession, handSession] = await loadSessions()

  // Stage 1: 盤面 + 持ち駒領域検出
  const boardInput = preprocessBoardInput(bitmap)
  const boardOut = await boardSession.run({
    image: new ort.Tensor('float32', boardInput, [1, 3, 224, 224])
  })
  const regions = decodeRegions(boardOut.regions.data as Float32Array, bitmap)

  // Stage 2: 透視変換 + マス分割 + 駒認識
  const normalizedBoard = perspectiveTransform(bitmap, regions.board, 288)
  const patches = sliceCells(normalizedBoard)  // Float32Array of (81, 3, 32, 32)
  const pieceOut = await pieceSession.run({
    patches: new ort.Tensor('float32', patches, [81, 3, 32, 32])
  })
  const board = decodePieces(pieceOut.logits.data as Float32Array)

  // Stage 3: 持ち駒認識 (先後それぞれ)
  const hands = { b: {}, w: {} }
  if (regions.handBlack) {
    const handInput = extractHandPatch(bitmap, regions.handBlack)
    const handOut = await handSession.run({
      hand_image: new ort.Tensor('float32', handInput, [1, 3, 96, 384])
    })
    hands.b = decodeHands(
      handOut.presence.data as Float32Array,
      handOut.counts.data as Float32Array
    )
  }
  // (後手も同様)

  return { board, hands, regions }
}
```
