# w384 backbone sweep 考察 (2026-07-13)

`scripts/train_backbones.sh` を **IMAGE_SIZE=384 / 最大 200 epoch** で 5 backbone 実施した結果 (wandb project: `mito-train-board-ocr-w384-v0.3.1`) から読み取れる **収束速度・過剰 epoch・train↔val ギャップ** をまとめる。

前提の指標定義は [`ocr-metrics.md`](./ocr-metrics.md)、w224 版の予測は [`ocr-scaling-outlook.md`](./ocr-scaling-outlook.md)、最終値のサマリと今後のイテレーション優先度は [`../TRAINING_PLAN.md`](../TRAINING_PLAN.md) を参照。ここでは sweep のログから拾える **数字と現象** の記録がゴール。

## sweep の共通条件

| 項目 | 値 |
|---|---|
| epochs | 200 (max) |
| batch size | 32 |
| image size | **384** |
| lr | 3e-4 peak, `cosine` schedule + warmup 5ep, min_lr=3e-6 |
| num workers / prefetch | 16 / 8, `--preload` |
| dataset | `ultemica/piyoshogi` (mode=full, train=20,000 / val=4,500) |
| hand head | logit マスキング (v2)、`hand_mode=classification`、hand_weight=1.0、class weight sqrt+clip[0.5, 10.0] |
| val 測定間隔 | 2 epoch ごと |
| 起動形態 | `PARALLEL_GPU=1` で 5 GPU 並走 (1 backbone / 1 GPU) |
| 総 wall time | 3h35m (04:26:40 起動 → 08:01 最遅 convnext_tiny 完了) |

## 最終値と best 到達 epoch

| Backbone | Params | val cell | val board | val hand | val hand_mae | val sfen (ep200) | **best val sfen** | best @ep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mobilenet_v3_small | 1.1M | 1.000 | 0.983 | 0.992 | 0.001 | 0.976 | 0.977 | 184 |
| mobilenet_v3_large | 3.3M | 1.000 | 0.995 | 1.000 | 0.000 | 0.995 | 0.995 | 200 |
| efficientnet_b1    | 6.9M | 1.000 | 0.999 | 1.000 | 0.000 | 0.999 | **1.000** | 148 |
| convnext_nano      | 15.1M | 1.000 | 0.999 | 1.000 | 0.000 | 0.999 | 0.999 | 92 |
| convnext_tiny      | 28.0M | 1.000 | 0.999 | 1.000 | 0.000 | 0.999 | 0.999 | 72 |

## 1. 200 epoch はほぼ全 backbone で過剰

val sfen が閾値を越えた epoch (2 epoch 刻み測定):

| Backbone | ≥0.90 | ≥0.95 | ≥0.98 | ≥0.99 | ≥0.995 |
|---|---:|---:|---:|---:|---:|
| mobilenet_v3_small | 50 | 84 | — | — | — |
| mobilenet_v3_large | 18 | 26 | 68 | 94 | 136 |
| efficientnet_b1    | 14 | 16 | 24 | 30 | 54 |
| convnext_nano      |  6 |  6 | 20 | 30 | 34 |
| convnext_tiny      |  6 |  8 | 10 | 14 | 20 |

- **convnext_tiny は 20ep で 0.995 到達**、以降 180ep はほぼ全部 waste。
- efficientnet_b1 も 54ep で 0.995、best は ep 148 の **1.000** で以降は 0.999 に微減 → 軽度の late-epoch overfit or val noise (val ≈ 2,000 SFEN で 1〜2 局面の差)。
- **200ep 使い切って伸びたのは mobilenet_v3_small のみ** (ep 184 で best 0.977、まだ僅かに右肩上がり)。
- 次回同構成の sweep をやるなら **convnext 系と effb1 は 60〜80ep** で十分。mobilenet_v3_large は 100ep、mobilenet_v3_small だけ 200ep フル。総 GPU 時間を約 1/3 に圧縮できる。

## 2. mobilenet_v3_small は容量限界に張り付いた

- val sfen が ep 120 以降 +0.005 しか伸びていない (0.972 → 0.977)。
- val hand_acc は 0.992〜0.994 で頭打ち。val board_acc 0.983 で、大型 backbone の 0.995 級と 1.3pt 差。
- **hand ではなく board 側の残差** が limiter。§B の hand class weight で救う話ではない。
- 一方 mobilenet_v3_large は val sfen 0.995 に達しているので、**edge 用の下限は v3_large が妥当**。v3_small は「動くだけの下限」枠。

## 3. train sfen < val sfen (逆転) は augmentation で説明可能

最終 epoch の train/val sfen ギャップ:

| Backbone | train sfen | val sfen | gap |
|---|---:|---:|---:|
| mobilenet_v3_small | 0.846 | 0.976 | **-0.130** |
| mobilenet_v3_large | 0.984 | 0.995 | -0.011 |
| efficientnet_b1    | 0.982 | 0.999 | -0.017 |
| convnext_nano      | 0.998 | 0.999 | -0.001 |
| convnext_tiny      | 0.995 | 0.999 | -0.004 |

`mito_train/datasets/capture_dataset.py:34-91` を確認する限り、train 側は `Affine` / `RandomBrightnessContrast` / `HueSaturationValue` / `GaussNoise` / `ImageCompression(quality 40〜92)` / `Downscale` をフルにかけていて、val 側は `LongestMaxSize` + `PadIfNeeded` + `Normalize` のみ。**train は「実運用に近い荒れた画像」、val は「HF paired の綺麗な画像」** なので、train < val は augmentation ノイズが乗った素の accuracy と綺麗版 val の差であって、過学習ではない (val は上限に張り付いている)。

ただし **mnv3s の -0.130 は augmentation ロバスト性が低い証拠** でもある。realistic 側で他 backbone より落差が大きく出る可能性が高いので、realistic 評価で「mnv3s だけ大きく劣化」なら容量よりロバスト性側の問題として切り分けるべき。

## 4. hand cold-start は warmup 内に解ける (大型のみ)

val hand_acc の初期挙動:

| Backbone | ep2 | ep4 | ep6 | ep8 | ep10 | ep20 |
|---|---:|---:|---:|---:|---:|---:|
| mobilenet_v3_small | 0.102 | 0.102 | 0.108 | 0.171 | 0.247 | 0.631 |
| mobilenet_v3_large | 0.102 | 0.135 | 0.402 | 0.782 | 0.944 | 0.989 |
| efficientnet_b1    | 0.102 | 0.112 | 0.218 | 0.504 | 0.773 | 0.995 |
| convnext_nano      | 0.161 | 0.951 | 0.986 | 0.997 | 0.998 | 0.998 |
| convnext_tiny      | 0.117 | 0.414 | 0.947 | 0.990 | 0.996 | 0.998 |

- 全 backbone とも初期 (ep1〜3) は val hand_acc ≒ 0.10、これは random init が 19-way logit をほぼ一様に出す状態 (常に 0 を返すベースラインの 0.66 を **下回っている**)。sqrt+clip class weight で count≥1 側にも勾配を流している副作用で、常に 0 を返すだけの局所最適には落ちない。
- **convnext_tiny は ep 6→8 で 0.947→0.990** (1 val step で解消)、convnext_nano は ep 4 で既に 0.951。**warmup 5ep + cosine の設計は大型 backbone に対しては綺麗にハマっている** (warmup 終了直後に hand が跳ねる)。
- mnv3s だけは warmup 内に解けず 40ep 引きずる (ep20 で 0.631)。mnv3s だけ狙って warmup を伸ばす価値はあるかもだが、そもそも上限が 0.977 なので工数対効果は薄い。

## 5. w224 版の予測との照合

[`ocr-scaling-outlook.md#backbone--image_size-の組み合わせ予想-ep50-cosine-lr-併用時`](./ocr-scaling-outlook.md#backbone--image_size-の組み合わせ予想-ep50-cosine-lr-併用時) の 384 予想値 vs 今回の実測 (ep50 相当と ep200 実測):

| Backbone | w224 doc の 384 予想 (ep50 cosine 想定) | 実測 val sfen @ ep50 | 実測 val sfen @ ep200 |
|---|---|---:|---:|
| mobilenet_v3_small | 0.55〜0.60 | 0.904 | 0.976 |
| mobilenet_v3_large | 0.72〜0.78 | 0.976 | 0.995 |
| efficientnet_b1    | 0.79〜0.84 | 0.994 | 0.999 |
| convnext_nano      | 0.78〜0.83 | 0.991 | 0.999 |
| convnext_tiny      | 0.83〜0.88 | 0.997 | 0.999 |

**予想は全 backbone で大幅に低め**。理由は 2 つ:
1. w224 の予想は「解像度効果は cell_acc に +1.5pt 程度」で見積もっていたが、実測では w384 で **cell_acc が全 backbone 1.000 に張り付く**まで詰まった。1マスあたり 42.7px は駒種判別に十分な解像度で、cell head 側の頭打ちが 0.994 → 1.000 に上がった影響が sfen exact-match に **81 マス掛け算** で乗った。
2. hand head 側も w224 では 0.66 前後だったのが、cell_acc が上がると盤上検出精度の連鎖で hand の切り出し部も安定して、hand_acc が 0.99+ に飽和した。予想時点では hand class weight と cosine の複合効果を過小評価していた。

**含意**: `ocr-scaling-outlook.md` の予想は数字ではもう古い (現実は全 backbone 0.99+ に到達済み)。ただし **「hand が sfen のボトルネックである」「convnext_nano は Pareto 上効率が悪い」** といった構造読みは今も有効。sweep 数字は本ドキュメント側を正、`ocr-scaling-outlook.md` は方法論と失敗パターンの参考として読む。

## 6. val 分布の穴 (再掲、評価解釈用)

`TRAINING_PLAN.md#hand-ラベル分布の観測realistic-評価の解釈用` と重複するが、上記数字を読むときの注意点として再掲:

- val の 0 比率が **75.10%** (train 65.68%)。常に 0 ベースラインの val slot_acc が train より 10pt 底上げされる。
- **count=11 以上が val に 1 件も無い**。regression head の高枚数改善効果を val slot_acc では評価不能。
- 飛の非ゼロ率は val で 9.4% / 7.2% (train は 17.8% / 17.1%)。val でさらに希少。

→ **val sfen 0.999 は「稀な hand 構成が測れていない状態の 0.999」**。realistic / 実運用キャプチャで真の壁を測るまでは backbone 間の 0.001 差を序列に使わない。

## 7. 次のイテレーション (TRAINING_PLAN と同期)

`TRAINING_PLAN.md#次のイテレーションの優先順位2026-07-13-更新` と重複するが、この sweep から直接落ちる推奨:

1. **realistic 再測定を最優先**。特に mnv3s は augmentation gap が大きい (-0.130) ので、realistic での落差を必ず測る。
2. **backbone 選定**: 数字だけ見ると **efficientnet_b1 (7M) が本命候補**。ep 148 で val 1.000、ep 200 でも 0.999、convnext_nano (15M) と実質同性能で params 半分。edge 下限は mobilenet_v3_large (3.3M) で妥当。**convnext_tiny (28M) は落選** でよさそう (nano と同点)。
3. **次回 sweep があるなら epoch を絞る**: convnext 系と effb1 は 60〜80ep、mnv3l は 100ep、mnv3s だけ 200ep フル。総 GPU 時間を約 1/3 に圧縮。
4. **best checkpoint 選定**: effb1 は ep 148 の 1.000 を採用したいなら latest.pt ではなく best を選ぶ。現状 `epoch-XXX.pt` を 5ep 刻みで残しているので、evaluation スクリプトで per-epoch を舐めれば済む。

## 再現用コマンド

```bash
# 今回の sweep と同条件を再現
EPOCHS=200 IMAGE_SIZE=384 PARALLEL_GPU=1 \
  BACKBONES="mobilenet_v3_small mobilenet_v3_large efficientnet_b1 convnext_nano convnext_tiny" \
  ./scripts/train_backbones.sh

# 短縮版 (次回推奨)
EPOCHS=80 IMAGE_SIZE=384 PARALLEL_GPU=1 \
  BACKBONES="mobilenet_v3_large efficientnet_b1 convnext_nano convnext_tiny" \
  ./scripts/train_backbones.sh
# mnv3s だけ別途 200ep で回す
EPOCHS=200 IMAGE_SIZE=384 BACKBONES="mobilenet_v3_small" \
  ./scripts/train_backbones.sh
```

W&B project: `mito-train-board-ocr-w384-v0.3.1`。今回の sweep run 一覧:

| Backbone | Run ID |
|---|---|
| mobilenet_v3_small | vgtolxsm |
| mobilenet_v3_large | sjp40vgn |
| efficientnet_b1    | dcuucjfx |
| convnext_nano      | c4p8plsk |
| convnext_tiny      | ypxo5yvz |

ローカルログは `runs/logs/<backbone>.log`、checkpoint は `runs/board-ocr-<backbone>/epoch-{005,010,...,200}.pt`。

---

# realistic 評価 (2026-07-13)

`ultemica/piyoshogi-eval` (`paired` config, 1,000 SFEN × 4 device = 4,000 画像、train/val と leak なし) を使い、5 backbone × 4 device で val の realistic 転移を実測した。2 段階で実施:

1. **OCR 単体 eval** (`scripts/eval/eval_realistic.py`): piyoshogi-eval の GT `bboxes` で crop → OCR。
2. **end-to-end eval** (`scripts/eval/eval_realistic_e2e.py`): 生スクショ → **detector 予測 bbox で crop** → OCR (`runs/board-detector-v1/latest.pt`、mobilenet_v3_small / w384 / val iou_mean 0.99+)。

## 結論 (先に)

- **efficientnet_b1 (7M) / convnext_nano (15M) / convnext_tiny (28M) は 4 device 全てで realistic SFEN 99.60〜100.00% 到達**。val ep200 の 0.999 は誇張ではなかった。
- **convnext_nano / convnext_tiny は iPad14,10 で 1,000 局面全問正解 (100.00%)**、mean sfen **99.92%** で val を実測が上回った。
- **mobilenet_v3_large (3.3M) は 97.60〜98.20% で edge 用の実用下限**、mobilenet_v3_small (1.1M) は 93.89〜96.30% で容量律速が実運用でも顕在化。
- **iPad14,10 は OCR 単体 eval で board=0.00% に落ちた** が、これは piyoshogi-eval の GT bbox が OCR 学習側 crop 規約と一致していなかったため。detector 予測 bbox で crop すると **全 backbone 93.89〜100.00% に復活**。iPad の失敗は「モデルの iPad domain gap」ではなく **「評価データ側の bbox 規約ズレ」** だった。

## 1. OCR 単体 eval (GT bbox → crop → OCR)

`scripts/eval/eval_realistic.py --backbones <bb> --wandb`
wandb project: `mito-train-board-ocr-w384-v0.3.1-eval-realistic`

**SFEN Exact-Match**:

| Backbone | iPhone10,1 | iPhone11,8 | iPhone15,4 | iPad14,10 | val ep200 |
|---|---:|---:|---:|---:|---:|
| mobilenet_v3_small | 0.956 | 0.951 | 0.957 | **0.000** | 0.976 |
| mobilenet_v3_large | 0.985 | 0.985 | 0.985 | **0.000** | 0.995 |
| efficientnet_b1    | 0.997 | 0.996 | 0.996 | **0.006** | 0.999 |
| convnext_nano      | 0.999 | 0.999 | 1.000 | **0.000** | 0.999 |
| convnext_tiny      | 0.999 | 0.999 | 0.999 | **0.001** | 0.999 |

**読み取り**:

- **iPhone 3 機種は val と実質同水準** (val 0.999 の backbone は realistic も 0.996〜1.000)。realistic domain gap は iPhone に対しては ≤ 0.3pt。
- **iPad14,10 の board_full が全 backbone で 0.00%** (effb1 0.6%、cvtiny 0.1% のみ非零)、一方 hand_full=100% / slot_acc=100%、cell_acc=83〜92%。「マス単位では 8-9 割合うが 81 マス連続一致が絶対に取れない」= 系統的な誤読 → データ側の crop convention 不一致を疑う。
- ラベル 0 (=空スロット) が 75% の val 分布と違って、realistic は駒種頻度が train により近い → hand の realistic 転移は本当に強い (hand_full=100.0% × 4 backbone)。

## 2. end-to-end eval (detector 予測 bbox → crop → OCR)

`scripts/eval/eval_realistic_e2e.py --backbones <bb> --wandb`
wandb project: `mito-train-board-ocr-w384-v0.3.1-eval-realistic-e2e`

**SFEN Exact-Match**:

| Backbone | iPhone10,1 | iPhone11,8 | iPhone15,4 | iPad14,10 | mean |
|---|---:|---:|---:|---:|---:|
| mobilenet_v3_small | 0.9610 | 0.9630 | 0.9520 | 0.9389 | 0.9537 |
| mobilenet_v3_large | 0.9800 | 0.9790 | 0.9820 | 0.9760 | 0.9792 |
| efficientnet_b1    | 0.9970 | 0.9970 | 0.9960 | 0.9970 | **0.9967** |
| convnext_nano      | 0.9990 | 0.9990 | 0.9990 | **1.0000** | **0.9992** |
| convnext_tiny      | 0.9990 | 0.9990 | 0.9990 | **1.0000** | **0.9992** |

**detector iou_mean (予測 bbox ↔ piyoshogi-eval GT bbox)**:

| Device | iou |
|---|---:|
| iPhone10,1 | 0.991 |
| iPhone11,8 | 0.991 |
| iPhone15,4 | 0.989 |
| **iPad14,10** | **0.842** |

**読み取り**:

- **iPad14,10 の board 全崩壊は消滅**。OCR 単体 0.00% → end-to-end 93.89〜100.00% に。全 backbone で iPad SFEN ≥ 93.89%。
- iPad の iou 0.842 は **piyoshogi-eval GT bbox の annotation が detector 学習側 (=OCR 学習側) 規約と食い違っている** ことを示している。iPad だけ bbox の切り方 (余白の取り方、盤+持ち駒の含み方) が違う。iPhone 3 機種は iou 0.99+ で規約一致。
- **detector を挟むと iPhone は「学習側 crop に normalize される効果」で val とほぼ同じ**。iPhone e2e SFEN は val -0.02〜-0.30pt 圏内。
- **convnext_nano/tiny は iPad で 100.00% (1000/1000)**。cell_acc も全 device で 100.00%。**val 分布の穴 (count≥11 が 0 件、飛の非ゼロ率が train の約半分) で val 0.999 に張り付いていた 2 個の backbone が、realistic 側で val 越えを達成**。

## 3. realistic - val ギャップまとめ

| Backbone | val ep200 | realistic e2e mean | gap |
|---|---:|---:|---:|
| mobilenet_v3_small | 0.976 | 0.9537 | -2.23pt |
| mobilenet_v3_large | 0.995 | 0.9792 | -1.58pt |
| efficientnet_b1    | 0.999 | 0.9967 | -0.23pt |
| convnext_nano      | 0.999 | 0.9992 | **+0.02pt** |
| convnext_tiny      | 0.999 | 0.9992 | **+0.02pt** |

- **backbone サイズが大きいほど realistic 転移が良い**。mnv3s は -2.23pt (augmentation ロバスト性が低い + 容量律速)、cvnano/cvtiny は val を超える。
- **mnv3s の realistic 落差 -2.23pt** は sweep 時点で予測していた augmentation gap (train sfen 0.846 vs val 0.976 の -0.13 差) の予兆通り。ただし iPad 特有の追加落差ではない (iPad 0.939 vs iPhone 平均 0.958 = -1.9pt 差)。

## 4. backbone 選定の結論 (realistic 反映後)

`TRAINING_PLAN.md` の「本命 backbone の確定」を realistic 数値で書き直す:

| 用途 | 推奨 backbone | 根拠 |
|---|---|---|
| クラウド API (精度優先) | **convnext_nano (15M) or convnext_tiny (28M)** | realistic 99.92% 到達、iPad で 100.00%、val 上回り |
| クラウド標準 (Pareto 効率) | **efficientnet_b1 (7M)** | realistic 99.67%、cvnano 半分の params で -0.25pt 差 |
| Edge / WASM | **mobilenet_v3_large (3.3M)** | realistic 97.92%、mnv3s より +2.5pt 差でここが下限 |
| 落選 | mobilenet_v3_small (1.1M) | realistic 95.37% は用途限定、量子化しても mnv3l に劣る |
| 落選 | convnext_tiny (28M) | cvnano と同点、params 倍のメリット無し |

**本命 = efficientnet_b1、精度上限狙い = convnext_nano、edge = mobilenet_v3_large の 3 枚看板** で確定。

## 5. detector 側への含意

- **piyoshogi-eval `paired` config の iPad14,10 `bboxes` は annotation を見直す価値がある**。detector iou_mean が iPhone 0.99+ に対し iPad だけ 0.84 まで落ちるのは、eval 側の annotation 規約が detector 学習側と一致していないから。iPad の GT bbox を detector 学習側規約 (`ocr_paired` の crop 由来) に合わせて再アノテすれば OCR 単体 eval の iPad も救われる。ただし現状 e2e で iPad 100% 取れているので優先度は中。
- 詳細は [`../DETECTOR_STATUS.md`](../DETECTOR_STATUS.md) 参照。

## 6. 再現用コマンド

```bash
# OCR 単体 eval (5 backbone 並列、GPU 1/3/4/5/6)
for gpu_bb in "1:mobilenet_v3_small" "3:mobilenet_v3_large" "4:efficientnet_b1" "5:convnext_nano" "6:convnext_tiny"; do
  gpu=${gpu_bb%%:*}; bb=${gpu_bb##*:}
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/eval/eval_realistic.py \
    --backbones $bb --wandb --out runs/eval/realistic-w384-20260713-$bb.json &
done; wait

# end-to-end eval (同上)
for gpu_bb in "1:mobilenet_v3_small" "3:mobilenet_v3_large" "4:efficientnet_b1" "5:convnext_nano" "6:convnext_tiny"; do
  gpu=${gpu_bb%%:*}; bb=${gpu_bb##*:}
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/eval/eval_realistic_e2e.py \
    --backbones $bb --wandb --out runs/eval/e2e-realistic-w384-20260713-$bb.json &
done; wait
```

出力 JSON: `runs/eval/realistic-w384-20260713-<bb>.json` / `runs/eval/e2e-realistic-w384-20260713-<bb>.json`。
