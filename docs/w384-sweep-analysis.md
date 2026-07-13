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
