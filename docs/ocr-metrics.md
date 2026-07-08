# OCR 目的関数と評価指標

将棋盤スクショから SFEN を抽出するパイプラインの **精度指標** と **最適化目的** を明示する文書。今後のスコア報告・改善効果測定・モデル比較はすべて本文書の定義に沿う。

## 原則

- **プライマリ目的関数**: **SFEN 完全一致率** (手数を除く正規化後)
- **訓練時の loss**: 駒個別クロスエントロピー (勾配が安定して乗る)
- **モニタリング指標**: piece accuracy, IoU, 混同行列 (訓練中の様子見)
- **報告する数字**: 常に **プライマリ + セカンダリ複数** を並記する。単一の数字で「精度 X%」と言わない

## パイプライン全体像と段階別指標

```
入力画像
  ↓
[Stage 1] 盤面検出        → IoU, corner MSE
  ↓
[Stage 2] マス切り出し     → grid alignment error (px)
  ↓
[Stage 3] 駒個別分類 x81   → piece accuracy, 混同行列, per-class F1
  ↓
[Stage 4] 手番/持ち駒推定  → 手番一致率, 持ち駒一致率
  ↓
[Stage 5] SFEN 組み立て   → SFEN 完全一致率, legal SFEN 率
```

各 stage で独立に測る + end-to-end でも測る。

## プライマリ目的関数: SFEN 完全一致率

### 定義

正規化後の SFEN 文字列が bit exact で一致した比率。

```
Exact-Match Rate = (SFEN 一致サンプル数) / (全サンプル数)
```

### 正規化ルール

- **手数を除く**: SFEN の最後のフィールド (moveNumber) は無視して比較
- **持ち駒順序を正規化**: SFEN 慣習の `R,B,G,S,N,L,P` 順 (先手→後手) に統一
- **空白の扱い**: フィールド区切りのスペースは 1 個で正規化

つまり比較対象は：

```
<board> <turn> <hands_normalized>
```

の 3 フィールド。

### なぜこれをプライマリにするか

- ユーザーの実利用は「認識 → 解析エンジン投入」の流れ
- 1マスでも間違うと **手が全部変わって解析が的外れ**
- 「81マスのうち 80マス合ってた」は実運用では **不合格**
- SNS 実運用の期待精度と、この指標が一番リンクする

### 目標値

| ライン | Exact-Match Rate |
|---|---|
| **実用ライン** (エンジン投入して意味ある) | **90%** |
| 快適ライン (SNS 実運用で信頼される) | 95% |
| 完成ライン | 98% |

## セカンダリ指標

### S1. Piece Accuracy (駒個別分類 accuracy)

```
Piece Accuracy = (81マス中で正解したマス数) / (81 × サンプル数)
```

各マスを独立に見て、以下を「正解」とする：

- **空マス** → 予測も空マス
- **駒あり** → 種類・向き (先手/後手)・成/不成 全部一致
- **一部間違い** (種類は合ってるが成/不成が違う) → 不正解

**用途**: 訓練 loss と直結。局所改善が数値に出るので、改善のフィードバックが早い。ただし **これが 99% でも Exact-Match は 40〜60% の可能性**があるので、単独では実用性能を語れない。

**目標値**: 99.5% 以上 (Exact-Match 90% ライン到達に必要)

### S2. IoU (盤面検出)

```
IoU = area(pred ∩ gt) / area(pred ∪ gt)
```

- **IoU@0.75**: バイナリ hit 判定に使う (現在の hit rate 定義)
- **IoU@0.9**: より精密。Stage 2 のマス切り出し精度に効く
- **mean IoU**: 連続指標。改善の方向性が読める

**目標値**: IoU@0.9 で 95% 以上

### S3. 混同行列 (Confusion Matrix)

29 クラス (先手14種 + 後手14種 + 空マス) の confusion matrix。**どの駒がどれと間違えられるか** を可視化。

想定される混同パターン:
- **歩 ↔ と** (成/不成)
- **銀 ↔ 金** (書体近い)
- **成銀 (全) ↔ と金** (成駒の似た字体)
- **香 ↔ 成香 (杏)**
- **先手駒 ↔ 後手駒** (向き取り違え、稀だが致命的)

**用途**: augmentation の効き所を特定 (「歩の判別だけ落ちてる → 歩に特化した劣化を強める」)

### S4. 手番一致率

```
Turn Accuracy = (手番一致サンプル数) / (全サンプル数)
```

盤面認識が完璧でも手番を間違えれば解析エンジンが逆転して読む。独立に測る必要あり。

### S5. 持ち駒一致率

```
Hand Accuracy = (先手持ち駒完全一致 && 後手持ち駒完全一致) の比率
```

**先手 3P, 後手 B・P** のような組み合わせが 1駒でもずれたら不正解。

## 参考指標 (訓練中の様子見のみ)

### R1. 駒個数一致率

「先手金は 4枚以内」「歩+と金は 18枚以内」等の物理制約を満たしているか。**認識ミスの物理的整合性チェック**。

- 制約違反率: 「先手金 5枚」等の非合法カウントが出ている比率
- **目標**: 制約違反率 1% 未満 (99% は物理的に成立する結果)

### R2. Legal SFEN 率

出力した SFEN が **将棋のルール的に到達可能な局面** か。

- 40駒制約 (成駒含む)
- 二歩なし (先後それぞれ同筋に歩は1枚まで)
- 玉が両陣営に 1枚ずつ
- 王手放置なし (これは弱制約、判定コスト高いので任意)

tsshogi の `Position.isValid()` で自動判定可能。

### R3. 一手着手可能率

認識結果に対して、実際に 1手指せる合法手が 1手以上ある比率。これが 0 だと **完全に壊れた局面** を出したことになる。

## Stage 別の詳細指標

### Stage 1 (盤面検出) 単独評価

現在測定中の指標。`scripts/ocr/eval-iou.ts` が算出:

- hit rate (IoU >= 0.75 の比率)
- mean IoU
- median IoU
- 難度別 hit rate (easy/medium/hard)

**現状ベースライン (2026-07-06)**:
- アンサンブル (color→grid) hit rate 57.1%, mean IoU 0.655

### Stage 3 (駒認識) 単独評価

**盤面検出は完璧という条件下**で駒認識だけを見る。GT の盤面 ROI を使って直接切り出し → 分類。

- piece accuracy
- 混同行列
- per-class F1

これで **駒認識モデル単体の実力** が測れる。Stage 1 の失敗と分離して評価できる。

### End-to-End 評価

Stage 1 → Stage 3 → Stage 5 まで通す。

- Exact-Match Rate (プライマリ)
- 混同行列 (全 stage の誤りが混ざる)
- 難度別 Exact-Match Rate

## 訓練時の設定

### Loss 関数

**駒個別クロスエントロピー** を主。29 クラス分類問題として:

```
L = -Σ_i Σ_c y_{i,c} log(p_{i,c})
```

盤面検出モデルには **IoU loss (or GIoU loss)** + 4隅座標の smooth L1 loss。

### モニタリング (per epoch)

- train loss / val loss
- val piece accuracy
- val IoU
- 週次で val Exact-Match Rate (計算コスト高いので毎エポックはしない)

### Early stopping

**val Exact-Match Rate** の 5 epoch 移動平均で判定。piece accuracy で止めない (piece accuracy が上がっても Exact-Match が下がる過学習パターンあり)。

## データセット構成

### train / val / test の分離

| セット | 内容 | 用途 |
|---|---|---|
| train | JB hook 生成のクリーン画像 + augmentation | モデル訓練 |
| val (in-distribution) | JB hook 生成のクリーン画像 (train と別 SFEN) | 訓練中のモニタ |
| val (out-of-distribution) | Twitter 実データ 30枚 (アノテ済) | augmentation の効き測定 |
| **test-realistic** | **Twitter 実データ 100枚 (厳密アノテ)** | **最終評価、変更禁止** |

**test-realistic は絶対に触らない**。ここで測った数字が「本当の実力」。この数字をハイパラチューニングに使うと leakage が起きる。

### 難度別分割

test-realistic の内訳:
- easy 40%
- medium 40%
- hard 20% (ダーク、盤回転、極小、劣化強)

**hard での Exact-Match Rate** も別途報告する。「easy 95% だが hard 40%」は実運用でユーザーが不満持つ状態。

## 数字の報告フォーマット

任意の実験結果は最低限これを並記する:

```
[実験名]
  Exact-Match Rate: XX.X% (95% CI: XX.X - XX.X)
  Piece Accuracy:   XX.X%
  IoU@0.75:         XX.X%
  Hand Accuracy:    XX.X%
  Turn Accuracy:    XX.X%
  難度別 Exact:     easy XX% / medium XX% / hard XX%
  混同 top-3:       歩→と (X.X%), 銀→金 (X.X%), ...
  平均推論時間:     XX ms (p95 XX ms)
```

「精度 X%」だけの報告は **禁止**。何の指標か分からない発言は今後認めない。

## 現状のベースライン (2026-07-06)

`scripts/ocr/eval-iou.ts` の実行結果:

| 手法 | IoU@0.75 hit rate | mean IoU |
|---|---|---|
| 色ベース | 8.6% | 0.095 |
| 格子検出 (調整後) | 45.7% | 0.553 |
| **アンサンブル (color+grid)** | **57.1%** | **0.655** |

- Piece Accuracy: **未測定** (駒認識モデル未実装)
- Exact-Match Rate: **未測定** (end-to-end 未接続)

次のマイルストーン: **eval-e2e.ts** で Exact-Match Rate のベースライン算出。

## FAQ

**Q. Piece Accuracy 99% と Exact-Match 90% はどっちが厳しい？**

A. Exact-Match のほうがずっと厳しい。仮に 81マスが完全独立と仮定すると Exact-Match = Piece Accuracy^81 になる:

- Piece Acc 99.0% → Exact-Match 44.3%
- Piece Acc 99.5% → Exact-Match 66.6%
- Piece Acc 99.7% → Exact-Match 78.2%
- Piece Acc 99.9% → Exact-Match 92.2%

実際は独立じゃなくて相関がある (盤面検出が当たれば周辺マスも当たる) ので、上記より緩い。それでも **Exact-Match 90% を狙うなら Piece Accuracy 99.5% 以上必須**。

**Q. legal SFEN 率は目的関数にしないの？**

A. しない。**プライマリを 2 つ持つと最適化が壊れる**。legal 率は R2 の参考指標に留めて、Exact-Match の副産物として見る (Exact-Match が高ければ legal 率は自動で高い)。

**Q. 加重評価 (王の間違いは重い、歩は軽い) はやらないの？**

A. やらない。重みの設計に恣意性が入って比較が難しくなる。Exact-Match の 0/1 判定はシンプルで、駒種別の詳細は混同行列で見る、で切り分ける。

**Q. 難易度別のスコアが違い過ぎたらどうする？**

A. hard の Exact-Match が easy より 30pt 以上低い場合、**hard カテゴリ向けの augmentation を強化 or 追加データ収集**。easy を優先的に伸ばす方向にはしない (SNS 実運用では hard も出現する)。
