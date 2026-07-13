# 学習改善計画

board_ocr の学習を回しながら気づいた改善候補と、次のイテレーションの優先度メモ。
`docs/` は MITO 側から同期する仕様書用なので、内部計画はここに置く。

## 現状の観測（2026-07-13 w384 backbone sweep）

`scripts/train_backbones.sh` を **IMAGE_SIZE=384 / 最大 200 epoch** で 5 backbone 実施（wandb project: `mito-train-board-ocr-w384-v0.3.1`）。§A の hand regression、§B の class weight sqrt+clip、§C の hand_weight=1.0、§D の cosine + warmup、X-1 の hand logit マスキングは全て入った状態での測定。詳細な収束速度・過剰 epoch・train↔val ギャップの分析は [`docs/w384-sweep-analysis.md`](./docs/w384-sweep-analysis.md) に集約。

sweep のサマリ（val, 最終エポック）：

| Backbone | Params | val cell | val board | val hand | val hand_mae | val sfen |
|---|---:|---:|---:|---:|---:|---:|
| mobilenet_v3_small | 1.1M | 1.000 | 0.983 | 0.992 | 0.001 | 0.976 |
| mobilenet_v3_large | 3.3M | 1.000 | 0.995 | 1.000 | 0.000 | 0.995 |
| efficientnet_b1 | 6.9M | 1.000 | 0.998 | 1.000 | 0.000 | 0.998（max 1.000） |
| convnext_nano | 15.1M | 1.000 | 0.999 | 1.000 | 0.000 | **0.999** |
| convnext_tiny | 28.0M | 1.000 | 0.998 | 1.000 | 0.000 | 0.998 |

### 読み取れる状態

- **hand head も飽和した**。w224 sweep 時点で 0.385〜0.716 だった val sfen_full が、w384 では **0.976〜0.999** まで詰まった。上記の hand regression / class weight を入れる前に、純粋な解像度スケール（224→384 = ピクセル約 2.9 倍）だけで解ける範囲に入った。
- **backbone のスケール効果は残るが差は縮小**。convnext_nano (15M) と efficientnet_b1 (7M) はほぼ同等 (0.999 vs 0.998)。mobilenet_v3_large (3.3M) までは実用圏。**convnext_tiny (28M) はオーバースペック気味**。
- **val 数字を鵜呑みにしない**。下の「hand ラベル分布の観測」に書いた通り、val 側は count=11+ が 0 件、count=10 が 1 件、飛の非ゼロ率が train の約半分。**val sfen_acc=0.999 は稀な hand 構成での性能が測れていない状態の 0.999**。realistic 側（`test-realistic` 相当）や実運用キャプチャで真の壁を見る必要がある。
- 次の意思決定は **backbone 選定 → realistic 評価** であって、hand head の追加チューニングではない。詳細は「次のイテレーションの優先順位」を参照。

---

## 現状の観測（2026-07-12 backbone sweep, w224）

**（履歴用）** 上の w384 sweep 以前に取った、9 backbone × 50 epoch × image_size=224 の sweep。詳細な数字と考察は [`docs/ocr-scaling-outlook.md`](./docs/ocr-scaling-outlook.md) に集約。

sweep のサマリ（val, epoch 50）：

| Backbone | Params | val cell_acc | val sfen_full |
|---|---:|---:|---:|
| mobilenet_v3_small | 1.1M | 0.966 | 0.385 |
| mobilenet_v3_large | 3.3M | 0.989 | 0.593 |
| convnext_atto | 3.5M | 0.991 | 0.611 |
| efficientnet_b0 | 4.4M | 0.991 | 0.661 |
| convnext_femto | 4.9M | 0.992 | 0.645 |
| efficientnet_b1 | 6.9M | 0.992 | 0.663 |
| convnext_pico | 8.7M | 0.992 | 0.634 |
| convnext_nano | 15.1M | 0.993 | 0.660 |
| convnext_tiny | 28.0M | 0.994 | 0.716 |

分布可視化ツール：`scripts/inspect/analyze_hand_distribution.py`（出力: `runs/hand-distribution.png`）

### 読み取れる状態（w224 時点、履歴）

- **board head は epoch 5〜10 前後で飽和**（cell_acc 0.99+ で伸びしろほぼ無し）。全 backbone 共通。
- **hand head がボトルネック**。sfen_full は 0.385〜0.716 と 33pt 開いており、backbone 容量と学習余地の両方が効いている。
- **↑ この時点で「hand がボトルネック」だったが、w384 sweep（上）でその症状は val 上ほぼ解消**。

## 実装済みの改善（履歴）

- **§A hand の枚数予測を regression 化**: `train_board_ocr.py --hand-mode {classification,regression}` で切替可能（`f3e927d`, `3a56344`）。SmoothL1 の regression head。w384 sweep は default の `classification` で回して val hand_acc=1.000。regression 側の効果測定は val 分布に穴があり困難、realistic 側で差が出たときに再検討。
- **§B hand class weight**: `compute_hand_class_weights` で `sqrt(1/freq)` + `clip[0.5, 10.0]` を default 有効。`--hand-class-weight` / `--class-weight-clip-min/max` / `--no-hand-class-weight` で制御可能。focal loss / effective number / oversampling は未実装だが、w384 で val hand が飽和したため優先度は下がった。
- **§C hand_weight**: default 1.0（board head 飽和済みなので勾配を hand 側に寄せる意図）。動的スケジュールは未実装（現状不要）。
- **§D LR scheduler**: `--scheduler cosine --warmup-epochs 5` が default（`c813caf`）。per-step で `LinearLR(1e-3→1.0, warmup_epochs steps)` → `CosineAnnealingLR(T_max=残り steps, eta_min)`。ckpt に scheduler state を含め `--resume` で復元。ReduceLROnPlateau は未実装。
- **§E resume 対応**: `train_board_ocr.py` / `train_detector.py` / `train_piece.py` の 3 script 全部（`955ac7d`）。含める state: `model`, `optimizer`, `epoch`, `scheduler`。
- **X-1 hand logit マスキング**: `board_ocr.py` の `PIECE_MAX_PER_SLOT` バッファでスロット別 (piece_max+1) 以降を `-inf`。飛/角は 3 以上、香桂銀金は 5 以上、歩以外は 10 以上を推論時に絶対に出さない。

## hand ラベル分布の観測（realistic 評価の解釈用）

`scripts/inspect/analyze_hand_distribution.py` を `ultemica/piyoshogi` `ocr_paired` train n=18,000 SFEN = 252,000 スロットラベルに対して 2026-07-12 実測。w384 で val が飽和した後も、realistic 側の per-count recall を読むときの参照分布として使う。

- ラベル `0`（=空スロット）が **65.68%**。`1` が 21.20%、`2` が 6.39%、`3` 以降は 3% 未満に急落し、`10` は 0.12%、`18` に至っては 0.003%。
- モデルが常に「0」と答えても素の accuracy が **65.68%** 出せる構造。旧集計（55.94%）より上振れしており、0 バイアスは寧ろ強まっている。
- 駒種別の非ゼロ率（先手/後手はほぼ対称）：
  - 歩（S:P / G:P）：**68.03% / 63.95%**（唯一 count=1〜18 全域に散らばる。count=10 も 148/145 件、count=18 も 4/4 件）
  - 角（S:B / G:B）：42.94% / 41.51%
  - 金（S:G / G:G）：32.48% / 32.34%
  - 銀（S:S / G:S）：31.42% / 30.69%
  - 桂（S:N / G:N）：29.28% / 28.84%
  - 香（S:L / G:L）：22.24% / 21.79%
  - **飛（S:R / G:R）：17.84% / 17.13%**（ほぼ 0 or 1 枚、稀に 2 枚）
- 同じ hand head で 14 スロット一括処理しているが、実質**駒種ごとに難易度が全く違うタスク**。
- 物理制約（飛/角は最大 2、香桂銀金は最大 4、歩のみ 18 まで）はデータ上も遵守されており、`PIECE_MAX_PER_SLOT` の logit マスクは正しく効いている。

**現行データでの raw class weight（`grand_total / (HAND_MAX_COUNT * v)`）**：
count=0 → 0.080、count=6 → 10.27、count=10 → 45.3、count=15 → 221、count=18 → 1658。5〜6 桁のレンジ。

**val 分布の穴（2026-07-12 実測, n=2,000 SFEN）**：
val は train と分布形が違う。
- ラベル 0 の比率が **75.10%**（train 65.68%）。「常に 0」ベースラインの val slot_acc が **10pt 底上げ**される。
- **count=11 以上が val に 1 件も無い**。歩 count=11〜18 は train には計 500 件超あるが val では見えないため、**regression head の高枚数改善効果を val slot_acc で評価できない**。
- count=10 は val 全体で 1 件のみ（S:P）。
- 飛の非ゼロ率は val で **9.4% / 7.2%**（train は 17.8% / 17.1%）。val でさらに希少。
- 香/桂/銀/金は val 側でスロット別の max が偶発的に非対称（例：S:L max=2、G:L max=4）。per-slot accuracy の左右比較に要注意。

→ **hand の改善評価は `val/hand/slot_acc` だけ見ない**。per-count recall（count ≥ 3）や、高枚数用の合成 mini-eval セット、`test-realistic` 側の実測を併用する。

## 未着手 TODO

### F. `train_piece.py` の manifest モード実装（優先度：中）

現状 `run_manifest()` は TODO：

```python
train_ds = PiyoDataset(...)
print(f"[manifest] train entries={len(train_ds)}")
# TODO: implement DataLoader + training loop
```

必要な追加：

- DataLoader（train / val 分割 + augmentation）
- 学習ループ（loss / metrics / logging）
- resume（E 参照）
- wandb 統合（`train_board_ocr.py` と同じ形式）

**設計判断**：board_ocr の board head が既に駒種別分類を高精度で担っているので、piece_classifier 単体の位置づけを整理する必要がある。
- 切り出し済み 1 枚画像に対する軽量エッジ推論用？
- board_ocr の下流検証用？
- 目的が固まらないうちに実装だけ進めるのは避ける。

### G. wandb run 連結（優先度：低）

**現状**：`--resume latest` すると新しい run が立つ。config に `start_epoch` `resumed_from` を書き込んでいるので追跡はできるが、ダッシュボード上で曲線が分断される。

**案**：`wandb.init(id=<prev_id>, resume="allow")` で同じ run に追記。ただし失敗した run を含めた履歴管理が複雑になるので、必要になるまで放置でよさそう。

### H. 序盤の early stopping 判定（優先度：低）

epoch を長めに指定して回すぶん、val 指標が K エポック改善しなかったら止めるガードがあると事故が減る。ReduceLROnPlateau と併用する場合は patience 設計に注意。

## 次のイテレーションの優先順位（2026-07-13 更新）

**背景**: §A/§B/§C/§D/§E は実装済み。w384 sweep で val sfen_acc が 0.976〜0.999 まで飽和。**realistic 評価も 2026-07-13 に実施済み** ([`docs/w384-sweep-analysis.md`](./docs/w384-sweep-analysis.md#realistic-評価-2026-07-13))、end-to-end (detector → OCR) で **cvnano/cvtiny が 99.92%、effb1 が 99.67%、mnv3l が 97.92%、mnv3s が 95.37%** を実測。以降は本番運用整備。

1. **~~realistic / 実運用キャプチャでの再測定~~**（済み）
   - piyoshogi-eval で 5 backbone × 4 device × 1000 SFEN を実測。上記結論を参照。
2. **本番 backbone の確定**（realistic 反映済み）
   - **クラウド API 精度優先**: convnext_nano (15M) — realistic 99.92%、iPad で 100.00%
   - **クラウド標準 (Pareto)**: efficientnet_b1 (7M) — realistic 99.67%、cvnano の半分 params で -0.25pt 差
   - **Edge / WASM**: mobilenet_v3_large (3.3M) — realistic 97.92%、これが下限
   - **落選**: mobilenet_v3_small (95.37%、mnv3l に劣る) / convnext_tiny (nano と同点、params 倍)
3. **realistic で hand tail が崩れた場合の再開手段**（発動条件未達）
   - 手段 A: `--hand-mode regression` を w384 で再学習（既に実装、config 変えるだけ）。
   - 手段 B: v3 データマージ（下記 v3 セクション、scripts 未実装 / データ未生成の状態から再開）。
   - 手段 C: focal loss / effective number（未実装）を §B に追加。
   - **realistic 側で hand_full=100.00% × 4 backbone を達成**しているため、上記手段は不要と判断。
4. **piyoshogi-eval iPad annotation の見直し**（優先度：中）
   - detector 予測 bbox で救えるので緊急ではないが、OCR 単体 eval で iPad を測れないのは資産として不便。annotation 再生成 or annotation 規約を `detector_paired` 側に合わせる。
5. **F（piece manifest 実装）** — 使途を整理してから。board_ocr の board head で駒種分類は解決している状態なので、piece_classifier 単体の位置づけは要再定義。
6. **G, H** — あると便利、なくても致命的ではない。

## 既知のデータ制約

### 駒種別の理論最大枚数（モデル設計に反映すべき制約）

学習データの観測範囲はゲームの自然な制約と整合している：

| 駒種 | 理論最大 | 観測範囲（train） |
|---|---|---|
| 歩 (P) | 18 | 0-9, 11-16 |
| 香 (L) | 4 | 0-4 |
| 桂 (N) | 4 | 0-4 |
| 銀 (S) | 4 | 0-4 |
| 金 (G) | 4 | 0-4 |
| 角 (B) | **2** | 0-2 |
| 飛 (R) | **2** | 0-2 |

現状 `hand_logits: (B, 14, 19)` は**全スロットに 19 クラス**割り当てているが、**大半のクラスは物理的に選ばれ得ない**。

- 飛スロットの logit[3..18]：16 個の無効クラス
- 角スロットの logit[3..18]：16 個の無効クラス
- 香桂銀金の logit[5..18]：14 個の無効クラス
- 歩以外の logit[10..18]：デッドコード

**X-1: logit マスキング（実装済み）** — `board_ocr.py` で softmax 直前にスロット別 (piece_max+1) 以降を `-inf` バッファでマスク。学習・推論両方で「あり得ない出力を出さない」保証が付いている。X-2（スロット別出力ヘッド）は将来的な最適化として保留。

## 過去 baseline（履歴）

| 世代 | image_size / backbone | 適用改善 | 最終 sfen_acc |
|---|---|---|---:|
| v1 | 288 / mobilenet_v3_small | (baseline) | epoch 40 で 0.486 |
| v2 | 288 / mobilenet_v3_small | X-1 マスキング, hand_weight=1.0, class weight sqrt+clip | epoch 60 目標 sfen_acc > 0.55 |
| w224 sweep | 224 / 9 backbone × 50ep | + regression head 未使用 | 0.385〜0.716 |
| **w384 sweep** | **384 / 5 backbone × 最大 200ep** | + cosine + warmup + resume | **0.976〜0.999** |

## realistic で hand tail が崩れた場合の fallback: v3 データマージ（保留中プラン）

> **状態**: w384 sweep で val hand が飽和したため保留。`scripts/data/subsample_synthetic_sfens.py` / `scripts/data/build_v3_manifest.py` は未コミット、`data/ocr_v3/` も未生成。実運用キャプチャで hand tail が崩れた場合の再開ポインタとして方針だけ残す。

**方針**: HF 18k は捨てず、そこに **裾を厚くした synthetic 10k を足す**。単純ユニオンだと HF 側の 0 バイアスがそのまま重みで持ち込まれるので、**HF 側も rare-preserving で 10k に間引き、10k + 10k = 20k の均等 mix** に落ち着かせる。synth 側の SFEN は piyo-hook で HF と同じ 4 端末（iPhone10,1 / 11,8 / 15,4 / iPad14,10）で撮影して **画像スタイルを統一**する（合成レンダラは使わない）。

**再開時の手順**:

1. **`scripts/data/generate_synthetic_sfens.py`**（既存）: `python-shogi` でランダム対局 → 途中局面から piece type ごとに Uniform[0, PIECE_MAX] で hand をサンプリング。盤上の駒を hand に移すので nifu / per-type 総数保存 / logit マスクとの整合は自動で満たされる。
2. **`scripts/data/subsample_synthetic_sfens.py`**（未実装）: 「どこかの slot で count ≥ 3」OR「飛/角が hand に入っている」の OR 条件で rare 判定 → rare 内 uniform で 10k に間引き。
3. **piyo-hook で 4 端末撮影**: synth 分の SFEN を 4 端末でキャプチャ。
4. **`scripts/data/build_v3_manifest.py`**（未実装）: HF cache の parquet から `sfen, hash, type` を pyarrow で吸い出し、HF train 18k を rare-preserving で 10k に間引き、synth 10k と concat して `data/ocr_v3/{train,val}.jsonl` を書き出す。hash 重複は natural 側優先。
5. **HF publish** (`build_paired_to_hf.py` を v3 入力に切替 → `ocr_paired_v3` として push) or **`--extra-train-jsonl` の入口を追加**。

**再開時の学習パラメータ調整の指針**:

- `--class-weight-clip-max` を 10 → 15 程度に緩める（synth 追加後は count≥7 の raw weight が桁で伸びるため、10 clip だと個別分離できない）。
- 実 epoch 数はサンプル数増（18k → 20k）を考慮して再計算。
- `--hand-mode regression` の 2 系統並行実験が理想（分布に count=3〜18 が実在するので効果測定できる）。
