# OCR スケーリング見通し（エポック数・画像サイズを増やしたら？）

feat/ddp-and-backbone-sweep で回した 9 バックボーンの sweep 結果（2026-07-12）を材料に、エポックを伸ばした場合・入力解像度を上げた場合に精度がどこまで伸びそうかを見積もる。

前提の指標定義は [`ocr-metrics.md`](./ocr-metrics.md)、モデル選択の全体像は [`backbones.md`](./backbones.md) を参照。ここでは sweep から出た **数字だけ** を根拠に、次のイテレーションの方針を出すのがゴール。

## sweep の共通条件

`scripts/train_backbones.sh` のデフォルト（2026-07-12 現行）：

| 項目 | 値 |
|---|---|
| epochs | 50 |
| batch size | 32 |
| image size | **224** |
| lr | 3e-4 (AdamW 固定、scheduler 無し) |
| num workers | 16, prefetch 8, `--preload` |
| dataset | `ultemica/piyoshogi` (mode=full) |
| hand head | logit マスキング済み (v2)、hand_weight=1.0、class weight sqrt+clip |
| val 測定間隔 | 2 epoch ごと |

同一条件下で backbone だけを差し替えた ちょい厳密な比較になっている。lr scheduler も EMA も distillation も入っていない **素の baseline**。

## 各バックボーンの val 数値 (epoch 50 時点)

| Backbone | Params | val cell_acc | val sfen_full | train sfen_full | train loss |
|---|---:|---:|---:|---:|---:|
| mobilenet_v3_small | 1.1M | 0.966 | 0.385 | 0.185 | 0.4549 |
| mobilenet_v3_large | 3.3M | 0.989 | 0.593 | 0.428 | 0.1288 |
| convnext_atto | 3.5M | 0.991 | 0.611 | 0.541 | 0.0828 |
| efficientnet_b0 | 4.4M | 0.991 | 0.661 | 0.483 | 0.1128 |
| convnext_femto | 4.9M | 0.992 | 0.645 | 0.589 | 0.0635 |
| efficientnet_b1 | 6.9M | 0.992 | 0.663 | 0.475 | 0.1093 |
| convnext_pico | 8.7M | 0.992 | 0.634 | 0.632 | 0.0520 |
| **convnext_tiny** | **28.0M** | **0.994** | **0.716** | 0.659 | 0.0560 |
| convnext_nano | 15.1M | ― | ― | (途中終了、8 epoch まで) | ― |

- `val cell_acc` は **どのモデルも 0.966〜0.994** で頭打ち感。差は 3pt 弱。
- `val sfen_full` は **0.385〜0.716** で **33pt** も開く。プライマリ指標はまだまだ動く。
- convnext_nano は sweep 途中で止まっているので今回は評価対象外（8 epoch で val cell 0.989 / sfen 0.503 まで来ていたので、走り切れば pico〜femto と同格になりそう）。

train と val のギャップは全モデルで **val > train**（board head は val が優勢、hand head は val の方が難しい局面が来る）。過学習の兆候は現時点ゼロで、まだエポックを伸ばしてよい状態。

## val sfen_full の epoch 別トラジェクトリ

val は 2 epoch おき測定なので、代表点として ep10 / 20 / 30 / 40 / 50 を並べる。

| Backbone | ep10 | ep20 | ep30 | ep40 | ep50 | ep40→50 差分 |
|---|---:|---:|---:|---:|---:|---:|
| mobilenet_v3_small | 0.241 | 0.334 | 0.367 | 0.388 | 0.385 | **-0.003** |
| mobilenet_v3_large | 0.389 | 0.491 | 0.537 | 0.568 | 0.593 | +0.025 |
| convnext_atto | 0.482 | 0.562 | 0.599 | 0.611 | 0.611 | 0.000 |
| efficientnet_b0 | 0.447 | 0.570 | 0.611 | 0.640 | 0.661 | **+0.021** |
| convnext_femto | 0.529 | 0.577 | 0.612 | 0.643 | 0.645 | +0.002 |
| efficientnet_b1 | 0.428 | 0.526 | 0.596 | 0.637 | 0.663 | **+0.026** |
| convnext_pico | 0.512 | 0.598 | 0.618 | 0.641 | 0.634 | -0.007 |
| convnext_tiny | 0.620 | 0.690 | 0.710 | 0.699 | 0.716 | +0.017 |

読み取れること：

- **cell_acc は epoch 6〜10 で 0.98 台に到達、以後の伸びしろは 1pt 未満**。ボトルネックは常に hand head 経由の sfen_full。
- **mobilenet_v3_small は 40 epoch で頭打ち**。1.1M パラの容量律速でここから伸ばしても線形改善は期待できない。
- **convnext_atto と convnext_pico はサイズ違いなのに ep50 で 0.611 / 0.634 でほぼ団子**。ep40 でも同傾向。ConvNeXt 系はこの容量帯（3〜9M）で頭打ちしている可能性大。
- **EfficientNet 系（b0, b1）は 40→50 で +2pt 以上伸びており、直近 10 epoch でも上げ足を残している**。
- **convnext_tiny は ep30 で 0.710 に到達し ep40 でノイズで下がったが ep50 で再度 0.716**。5 epoch 移動平均を取ればまだ緩やかに上向き。飽和はまだ。

## エポックを伸ばしたときの見込み

各モデルの ep30→50 の 20 epoch 分の増分から、**ep50→100 の追加 50 epoch でどれくらい伸びるか** を粗く外挿する。

前提：
- 現状は AdamW `lr=3e-4` **固定**（scheduler 無し）。後半になるほど lr が高すぎて振動する典型パターン。
- v1 baseline（TRAINING_PLAN.md）でも同種の後半停滞が観察されている（v1 40 ep で sfen 0.486 → v2 hand 改善で ep50 0.385、条件が違う）。
- ep30→50 で伸びた幅の **1/2〜1/3** が ep50→100 の追加改善の目安（logistic 型に減衰する経験則）。ここに lr scheduler を入れると +2〜3pt が乗る、というのが `TRAINING_PLAN.md#D` の期待値。

| Backbone | ep30→50 増分 | ep100 予想 (据置lr) | +cosine LR で狙える上振れ | 備考 |
|---|---:|---:|---:|---|
| mobilenet_v3_small | +0.018 | ~0.395 | ~0.42 | 容量律速。飽和目前 |
| mobilenet_v3_large | +0.056 | ~0.62 | ~0.65 | まだ上げ足あり |
| convnext_atto | +0.012 | ~0.62 | ~0.65 | 30 epoch で飽和気配 |
| efficientnet_b0 | +0.050 | ~0.69 | ~0.72 | 直近も上向き、伸びる |
| convnext_femto | +0.033 | ~0.66 | ~0.69 | 中庸 |
| efficientnet_b1 | +0.067 | **~0.70** | **~0.73** | 直近の勾配が最良 |
| convnext_pico | +0.016 | ~0.65 | ~0.67 | atto と同水準に張り付き |
| convnext_tiny | +0.006 | ~0.73 | **~0.76** | ばらつきあり、平均で微増 |

これは **「同じ lr のまま 100 epoch まで回した場合」** の見込みで、実運用の伸び余地としては **cosine annealing / ReduceLROnPlateau を入れたときの +0.02〜0.03** をそこに乗せた側が現実的な上限。

### エポック増でも越えられない壁

sfen_full が上記予想値で頭打ちする理由：

1. **cell_acc が既に 0.99 台に張り付いている**。ここから 1pt 押し上げても sfen_full にはあまり効かない（[`ocr-metrics.md` の Q1 表](./ocr-metrics.md)、および [`backbones.md` の cell_acc→sfen_full 表](./backbones.md#cell_acc-と-sfen_full-の関係重要)）。
2. **hand_full_acc がまだ 0.7 前後**。sfen_full ≈ board_correct × hand_correct なので、hand を伸ばさない限り sfen は上に抜けない。詳細は下記「hand の失敗パターン」を参照。

つまり **エポックを 2 倍に伸ばしても sfen_full の到達値は +3〜5pt**。実用ライン 90% には遠く、エポック単体の追加投資では届かない。

## hand の失敗パターン

hand head は構造上 **駒の種類を間違えない**（`mito_train/models/board_ocr.py` の `hand_head` は 14 スロット固定で、スロット index が piece × side をエンコード）。だから「hand の間違い = 枚数の間違い」だけ。

残っている失敗パターンは以下の 2 つ：

### 1. 枚数の隣接ミス（本命）

`hand_logits: (B, 14, 19)` の 19-way CrossEntropy を使っているため、**「5 vs 6」の間違いも「5 vs 18」の間違いも同じ loss**。序数情報を完全に捨てている。argmax が隣接クラスで揺れやすいのはこの構造由来。sfen 落ちの支配的要因のはず。

対策は `TRAINING_PLAN.md#A` の **regression head**（SmoothL1、推論時 round+clamp）。

### 2. 0 バイアス（分布の偏り）

学習データの分布が極端に 0 に寄っている（`scripts/inspect/analyze_hand_distribution.py` の出力を参照）：

- ラベル 0（空スロット）が過半
- 飛スロットは 0/1 でほぼ完結（非ゼロ率 19〜20%）
- 高枚数（3〜9）は希少で、under-count しやすい

対症療法として v2 で `class weight sqrt+clip` を導入済み（`TRAINING_PLAN.md#B`）。ただし本丸は上記 1 の regression 化なので、これだけでは頭打ち。

### 解消済みの構造欠陥（参考）

以前は歩スロットで **count=10 が学習データに 1 件も無い** ギャップがあり（piyo-hook 側の SFEN 出力バグ由来）、歩 10 枚を毎回 8/9/11 に誤読していた。現在は `ultemica/piyoshogi` の最新スナップショットで歩 count=10 局面が収録済みで、この構造欠陥は解消されている。分布としては依然として裾で希少なので、対策 2（class weight）の対象には残っている。

### 診断コマンド

「どのスロットで、どの枚数を、何枚に間違えたか」を数字で出す：

- `scripts/inspect/diagnose_hand.py` — val 全体で per-slot accuracy + true count → pred count の confusion matrix
- `scripts/inspect/analyze_hand_failures.py` — 端末別（iPhone XR / iPhone 15 等）で mismatch 局面をダンプ

sweep 直後の checkpoint に対して回せば、隣接ミス / 0 バイアスのどちらが支配的か切り分けられる。

## 画像サイズを大きくしたときの見込み

現状 sweep は **image_size=224**。9x9 マス識別に対して 1マスあたり `224/9 ≈ 24.9px`。24x24 の画像で「歩 / と / 銀 / 金」を漢字で判別する状態。マス内の細部（成香=杏、成桂=圭 等の点画）が潰れる領域に片足入っている。

`docs/backbones.md#何が精度を押し上げるか` に **「image_size の増（224 → 288 → 384）: 一番効きやすい」** と明記されており、経験則としても解像度は cell_acc に直接効く。

### 参考: v1 baseline との落差

`TRAINING_PLAN.md` の v1 baseline は mobilenet_v3_small を epoch 40 まで回して **sfen_acc = 0.486**（batch 128, lr 6e-4, image 288 系）。今回の sweep 同 backbone ep40 で **sfen 0.388**。**batch と lr が違うので直接比較はできない**が、image_size の効きが大きな要因の 1 つなのは確か。

### 224 → 288 / 384 に上げた場合の予想

1マスあたり px 数と、cell_acc / sfen_full の伸び幅の見立て：

| image_size | 1マスあたり px | cell_acc の伸び (mobilenet_v3_small 想定) | sfen_full 期待伸び幅 |
|---:|---:|---:|---:|
| 224 (現状) | ~24.9 | ― (baseline 0.966) | ― (baseline 0.385) |
| 288 | ~32.0 | +0.5〜1.0pt | **+0.05〜+0.10** |
| 384 | ~42.7 | +0.8〜1.5pt | **+0.10〜+0.20** |

sfen_full 側の増分が大きいのは、cell_acc の 1pt 改善が **81 マス独立仮定で ~ (0.99/0.98)^81 ≈ 2.3倍** の効きになるため。

**image_size を上げる方が、追加エポックを回すよりコストパフォーマンスが良い**。VRAM と学習時間は概ね `(size/224)^2` で増えるが、`convnext_atto` くらいまでのモデルなら 288 / 384 は現実的。

### backbone × image_size の組み合わせ予想 (ep50, cosine lr 併用時)

epoch 50 相当で回したときの val sfen_full の見込みレンジ：

| Backbone | 224 (実測) | 288 予想 | 384 予想 |
|---|---:|---:|---:|
| mobilenet_v3_small | 0.385 | 0.45〜0.50 | 0.55〜0.60 |
| mobilenet_v3_large | 0.593 | 0.65〜0.70 | 0.72〜0.78 |
| convnext_atto | 0.611 | 0.67〜0.72 | 0.74〜0.80 |
| efficientnet_b0 | 0.661 | 0.72〜0.76 | 0.78〜0.83 |
| convnext_femto | 0.645 | 0.70〜0.75 | 0.77〜0.82 |
| efficientnet_b1 | 0.663 | 0.72〜0.77 | 0.79〜0.84 |
| convnext_pico | 0.634 | 0.71〜0.75 | 0.77〜0.82 |
| **convnext_tiny** | **0.716** | **0.77〜0.82** | **0.83〜0.88** |

この見込みは **image_size 効果 (backbones.md の想定)** + **既にある epoch トラジェクトリ** + **hand head の残り改善** を足したもの。convnext_tiny × 384 で **0.85 台**、実用ライン 90% には **もう一押し必要** な位置。

## 「エポック増」と「画像サイズ増」の効き順ランキング

sweep 結果から見えるコスパ順位（1 sample あたりの学習 wall-clock 増加を考慮）：

1. **image_size 224 → 288**（コスト ~1.65倍、期待 +5〜10pt）
2. **hand head の regression 化 + class weight 継続**（コスト実装のみ、期待 +3〜5pt）
3. **image_size 288 → 384**（コスト ~1.78倍、期待 +5〜10pt）
4. **cosine LR + epoch 100 まで延長**（コスト 2倍、期待 +2〜3pt）
5. **backbone を 1〜2 段上げる**（コストは backbone 依存、期待 +2〜5pt）

**エポック単独増は 4 番目**。sweep の各 backbone の後半勾配を見る限り、`convnext_atto` `convnext_pico` は既に飽和気味で、100 epoch まで回しても sfen +2pt がせいぜい。

一方 **画像サイズ増と hand head 改修が上位で、これを先に済ませないと後段の投資が全部 5割引き** になる。hand が枚数の隣接ミスで落ちている限り、cell_acc を上げても sfen_full には抜けない。

## 90% ライン到達までの推奨経路

`ocr-metrics.md` の **プライマリ目標 = SFEN Exact-Match 90%** に到達するための現実的な階段：

```
[現状 baseline]
convnext_tiny × 224 × 50ep → 0.716

  ↓ image_size を 288 に (+cosine LR)
convnext_tiny × 288 × 60ep → 0.77 前後  (2026-07 内で到達目安)

  ↓ image_size を 384 に + hand regression head
convnext_tiny × 384 × 80ep → 0.83〜0.86 (2026-08 内で到達目安)

  ↓ 蒸留 (convnext_small teacher) + 追加 aug
convnext_tiny × 384 × distill → 0.88〜0.92 (実用ライン到達)

  ↓ SNS aug ハード側の追加 + 手番/持ち駒後処理
0.93+ (SNS 実運用ライン)
```

**convnext_tiny × 384 に到達しても素の sfen_full は 0.85 前後で、90% には蒸留 or 後処理が要る** というのが今回の sweep から出る現実的な見立て。ブラウザ配信の本命 convnext_nano は今回未完走なので、次の sweep で必ず走らせて Pareto を埋める。

## 直近の推奨アクション

sweep 結果を踏まえた次のイテレーション優先順位：

1. **hand head の regression 化**（`TRAINING_PLAN.md#A`）。cell_acc は据置きで sfen_full が跳ねる可能性大。実装コスト最小、まず切って効果測定。
2. **convnext_nano の再学習**（今回未完走）。tiny の 15MB 版として実運用第一候補。
3. **image_size=288 での再 sweep**。まずは mobilenet_v3_small / convnext_atto / convnext_tiny の 3 点で効果測定。
4. **cosine annealing の導入**（`TRAINING_PLAN.md#D`）。resume 対応と同時に。
5. **image_size=384 は convnext_atto / tiny のみ、蒸留と一緒に**。VRAM とのトレードで動く backbone だけ選ぶ。
6. **hand 分布の再集計**（`scripts/inspect/analyze_hand_distribution.py`）。歩 count=10 収録後の最新分布を確認し、class weight の clip 値を再調整するか判断する。

「エポック増だけで頑張る」路線は **ROI が悪い**。sweep の後半勾配は既に落ちている。次の投資は **hand head + 解像度 + scheduler の三点セット**、そこに convnext_tiny を据えるのが最短。

## 再現用コマンド

sweep を同条件で再現する場合：

```bash
# 単一 GPU で全 backbone を順番に
EPOCHS=50 IMAGE_SIZE=224 ./scripts/train_backbones.sh

# image_size=288 版
EPOCHS=50 IMAGE_SIZE=288 BACKBONES="mobilenet_v3_small convnext_atto convnext_tiny" \
  ./scripts/train_backbones.sh

# resume-friendly な全 backbone sweep
RESUME_INCOMPLETE=1 EPOCHS=100 IMAGE_SIZE=288 ./scripts/train_backbones.sh
```

W&B project: `mito-train-board-ocr`。今回の sweep run 一覧：

| Backbone | Run ID |
|---|---|
| mobilenet_v3_small | vhhpiwgm |
| mobilenet_v3_large | k2mol6pl |
| convnext_atto | o68bjpih |
| convnext_femto | sw2z4m7f |
| efficientnet_b0 | 3p1kubgl |
| convnext_pico | vw4mk54g |
| efficientnet_b1 | 5h2kp29v |
| convnext_tiny | r4c7icav |
| convnext_nano | nyrsx25c (未完走) |
