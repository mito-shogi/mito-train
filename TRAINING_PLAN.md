# 学習改善計画

board_ocr の学習を回しながら気づいた改善候補と、次のイテレーションの優先度メモ。
`docs/` は MITO 側から同期する仕様書用なので、内部計画はここに置く。

## 現状の観測（2026-07-12 backbone sweep）

`scripts/train_backbones.sh` で 9 backbone × 50 epoch × image_size=224 の sweep 実施済み。
詳細な数字と考察は [`docs/ocr-scaling-outlook.md`](./docs/ocr-scaling-outlook.md) に集約。

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

### 読み取れる状態

- **board head は epoch 5〜10 前後で飽和**（cell_acc 0.99+ で伸びしろほぼ無し）。全 backbone 共通。
- **hand head がボトルネック**。sfen_full は 0.385〜0.716 と 33pt 開いており、backbone 容量と学習余地の両方が効いている。
- val > train の傾向で過学習は現状ゼロ。エポックはさらに伸ばせるが、後半勾配は落ち始めている backbone も多い。
- **参考: 旧 v1/v2 の観測**（image_size=288, batch=128, lr=6e-4）は本ファイル末尾の「v1 baseline」「v2」節に残す。sweep は条件が違うので直接比較不可。

## 改善候補

### A. hand の枚数予測を分類から見直す（優先度：高）

**現状**：`hand_logits: (B, 14, 19)` の 19-way CrossEntropy。0〜18 の枚数を独立クラスとして扱っている＝「5 枚と 6 枚を間違える」ことと「5 枚と 18 枚を間違える」ことに同じロスがつく。序数情報を完全に捨てている構造。

**案**：

1. **Regression head**（Huber / SmoothL1）
   - 出力を単一スカラーにして `pred_count` を回帰。
   - 近い枚数を近いと理解できる。
   - 推論時は `round + clamp(0, 18)`。
2. **Ordinal regression**（CORAL 等）
   - 分類の枠は保ちつつ、隣接クラス間で単調性を仮定。
3. **暫定策：CE のまま label smoothing / distance-aware label**
   - 教師を `one_hot(k)` から `softmax(-|i - k| / τ)` に置き換え、近距離クラスに確率をリーク。

**検証**：どれか 1 案で epoch 20 前後まで学習し、既存 baseline と `val/sfen/full_acc` で比較。

### B. データ分布の偏り対応（優先度：高）

**観測**（`scripts/inspect/analyze_hand_distribution.py` を現行 `ultemica/piyoshogi` `ocr_paired` train n=18,000 SFEN = 252,000 スロットラベルに対して 2026-07-12 実測）：

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

**素朴な `1 / freq` の危険性**：クラス重みが 5〜6 桁開く（count=0 で 0.08、count=18 で 1658）。希少クラスの勾配が暴発して学習不安定になるので、そのままは NG。

**対応候補**：

- **class weight（現実解）**：`w_k = min(sqrt(1 / freq_k), clip_max)`。`clip_max` は 10〜30 程度で試す。
- **focal loss**（`γ=2` 程度）：easy negative（大半の 0 枚予測）の寄与を自動で下げる。class weight とも併用可。
- **effective number of samples**（Cui et al. 2019）：`β=0.999` 前後で滑らかな重み付け。
- **oversampling**：希少枚数を含む局面を重複サンプリング。ただし augmentation の相性次第で汎化が下がる。
- **スロット別モデル分割** or **スロット別重み**：飛と歩を同じヘッドで扱うのが本当に妥当か再検討。

**検証**：class weight（sqrt + clip）→ focal loss の順に足していき、`val/hand/slot_acc` の per-count recall（特に count ≥ 3）で改善を確認。全体の `slot_acc` は下がる可能性があるが、希少枚数を当てられることの方が実運用上重要。

### C. マルチタスク重み調整（優先度：中）

**現状**：`hand_weight = 0.5`。board head が既に飽和しているのに 0.5 の重みが残っているので、勾配のうち board 側の寄与が無駄。

**案**：

- `hand_weight = 1.0` に上げる。
- あるいはエポック進行に応じて動的に上げる（`hand_weight = min(1.0, 0.5 + 0.05 * epoch)` など）。
- GradNorm や uncertainty weighting（Kendall et al.）で自動化するのは overkill 気味。

### D. LR スケジューラ導入（優先度：中）

**現状**：`AdamW(lr=6e-4)` 固定。後半になると同じ lr で更新が振動しがち。

**案**：

- **CosineAnnealingLR(T_max=epochs)** ：シンプルで効果が読みやすい。
- **ReduceLROnPlateau(patience=3)** ：val 指標が停滞したら自動で半減。
- resume 時に scheduler state もチェックポイントに含めること。

### E. resume 対応の横展開（優先度：中）

`--resume` は `train_board_ocr.py`, `train_detector.py`, `train_piece.py` の 3 script に横展開済み。

含める state：`model`, `optimizer`, `epoch`, `scheduler`（導入時）, `rng`（厳密再現したいとき）。

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

## 次のイテレーションの優先順位

1. **A（regression / ordinal 試作）** ＋ **B（分布可視化 → class weight）**
   - hand の学習効率がここで決まる。まずここに集中。
2. **C（hand_weight 引き上げ）**
   - コード変更が最小、効果も見やすい。A/B と並行可。
3. **D（LR scheduler）**
   - resume 対応と一緒に入れると綺麗。
4. **E（resume 横展開）**
   - F の前提。
5. **F（piece manifest 実装）**
   - 使途を整理してから。
6. **G, H**
   - あると便利、なくても致命的ではない。

## 既知のデータ制約

### 持ち駒 count=10 のギャップ（解消済み）

以前は `data/train.jsonl` に **どのスロットにも count=10 が 1 件も現れない**問題があった（piyo-hook 側の SFEN 出力バグに起因）。特に歩スロット（S:P, G:P）は実局面で 10 枚以上が普通に発生するため、学習データの穴が推論時に「10 枚 → 8/9/11 に丸める」挙動を起こしていた。

**現状**：`ultemica/piyoshogi` の最新スナップショットには歩 count=10 の局面が含まれており、この構造的な穴は解消済み。歩の高枚数（10 以上）は依然として希少（分布としては裾）だが、logit が「完全ゼロ学習」状態ではなくなっている。

**残っている hand 側の課題**（詳細は [`docs/ocr-scaling-outlook.md`](./docs/ocr-scaling-outlook.md) の「hand の失敗パターン」を参照）：

1. **枚数の隣接ミス**（本命）: 19-way CE のため「5 vs 6」も「5 vs 18」も同じ loss。序数情報が捨てられている。→ 対策は本ファイル **§A の regression head**。
2. **0 バイアス**: ラベル 0（空スロット）が過半、飛スロットは 0/1 でほぼ完結。希少枚数（3〜9）で under-count しやすい。→ v2 で入れた **§B の class weight sqrt+clip** が対症療法。本丸は §A。

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

**影響**：CE 学習では教師が有効範囲内に収まるため無効クラスの logit は自然に低くなるが、
1. 推論時に理論上あり得ない値が argmax で選ばれる可能性がゼロにはならない
2. softmax の正規化に無効クラスが混ざる分だけ確率が薄まる
3. 出力層のパラメータが無駄

**対策候補**：

- **X-1. 学習時に無効クラスを `-inf` マスク**：`hand_logits` を softmax する直前に、スロットごとに (piece_max+1) 以降を `-inf` で埋める。実装コスト最小、推論の物理的整合性を保証。
- **X-2. スロット別の出力ヘッド**：スロットごとに `nn.Linear(dim, piece_max+1)` を持たせる。パラメータ効率は最良だが実装が煩雑。
- **X-3. 現状維持 + `ignore_index`**：教師側だけ制約、logit は 19 のまま。副作用少だが推論時の整合性保証は無い。

**推奨**：**X-1（logit マスキング）を短期で導入**。学習・推論両方で「あり得ない出力を出さない」保証が付き、コード変更もモデル出力直前の 1 段追加のみ。X-2 への移行は将来的な最適化として保留。

## v1 baseline（比較用）

- ckpt: `runs/board-ocr-v1-baseline/` に凍結保存
- 学習: 40 epoch まで実施（当初 20 → resume で 40 まで）
- epoch 20: sfen_acc = 0.324
- epoch 30: sfen_acc = 0.417
- epoch 40: sfen_acc = 0.486
- epoch 40 で中断、v2 に移行

## v2（現在学習中）

**適用した改善**：

- **X-1: hand logit マスキング**（`board_ocr.py`）
  - `PIECE_MAX_PER_SLOT` 定義済み、スロット別に無効クラスを `-inf` バッファでマスク
  - 飛/角は 3 以上、香桂銀金は 5 以上、歩以外は 10 以上を推論時に絶対に出さない
- **C: hand_weight 0.5 → 1.0**（`train_board_ocr.py`）
  - board head が飽和済みなので勾配を hand 側に寄せる
- **B: class weight sqrt+clip[0.5, 10.0]**（`train_board_ocr.py`）
  - 起動時に train manifest から実測、count=0 は 1.34、count≥7 は 10 でキャップ
  - `--no-hand-class-weight` で無効化可能

**config**：`EPOCHS=60 BATCH_SIZE=128 LR=6e-4 IMAGE_SIZE=288 BACKBONE=mobilenet_v3_small`
**ckpt出力**：`runs/board-ocr-v2/`
**目標**：epoch 60 で sfen_acc > 0.55（v1 の 45 epoch 相当を超える）

## v3（HF natural + synthetic マージ版）

### 見つけた偏り

v2 は HF `ultemica/piyoshogi` `ocr_paired` train（18,000 SFEN、`build_manifests.py` で作った natural 40% + opening 20% + synthetic 40% + existing のミックス）を教師に使っていた。そこには **hand 側に強い分布バイアス** があった（2026-07-12 実測）：

| 指標 | v2 データ (HF 18k) |
|---|---:|
| slot ラベル 0（空スロット）の比率 | **65.68%** |
| slot ラベル 1 の比率 | 21.20% |
| slot ラベル 3 以上の合計 | ~4% |
| 歩 count=7 以上の合計 | ほぼゼロ |
| 歩 count=18 の観測 | 4 件（train 全体で） |

- 常に 0 を答えるだけで slot_acc **65.68%** が出る構造。hand head の勾配が「0 を当てにいく」方向に寄る。
- 高枚数（3〜18）は裾でしか観測されず、隣接ミス（5 vs 6, 10 vs 11）の学習信号が薄い。
- 飛 (R) / 角 (B) は非ゼロ率が **17〜42%** に留まる。
- val 側（2,000 SFEN）は train より更に極端で、**count=11 以上が 1 件も無い**。「regression head を入れたい」って言っても val で効果測定ができない。

詳細は `docs/ocr-scaling-outlook.md#hand-の失敗パターン` に集約。

### 偏りの修正手順

**方針**：HF 18k は捨てず、そこに **裾を厚くした synthetic 10k を足す**。ただし単純ユニオンだと HF 側の 0 バイアスがそのまま重みで持ち込まれるので、**HF 側も rare-preserving で 10k に間引き、10k + 10k = 20k の均等 mix** に落ち着かせる。synth 側の SFEN は piyo-hook で HF と同じ 4 端末（iPhone10,1 / 11,8 / 15,4 / iPad14,10）で撮影して、**画像スタイルは 20k 全件で統一**する。合成レンダラは使わない（画像スタイルと分布の相関を学習するリスクを避ける）。

1. **`scripts/data/generate_synthetic_sfens.py`**: `python-shogi` でランダム対局を回し、途中局面から **piece type ごとに Uniform[0, PIECE_MAX] で hand をサンプリング**。盤上の駒を hand に移すだけで生成するので、nifu / per-type 総数保存 / logit マスクとの整合はすべて自動で満たされる。作成: `data/synth_train_sfens.jsonl` (15,000)、`data/synth_val_sfens.jsonl` (2,500)。
2. **`scripts/data/subsample_synthetic_sfens.py`**: 15,000 は多いので **希少局面を残したまま 10,000 に間引き**。希少判定は「どこかの slot で count ≥ 3」OR「飛 / 角が hand に入っている」の OR 条件。実測では 14,906 件（99.4%）が rare 判定に入り、残り 94 件の「完全に静かな局面」だけがマジョリティから抜けた。rare 判定内は uniform で subsample。原本は `data/synth_train_sfens.jsonl.bak` に退避済み。
3. **piyo-hook で 4 端末撮影**: `data/synth_train_sfens.jsonl` / `data/synth_val_sfens.jsonl` の全 SFEN を 4 端末でキャプチャ。1 SFEN × 4 端末 = 40,000 webp（train）+ 10,000 webp（val）。HF 既存分と合わせて `data/ocr/<device>/<hash>.webp` に配置。
4. **`scripts/data/build_v3_manifest.py`**: HF cache の parquet から `sfen, hash, type` だけを pyarrow で吸い出し（画像列は触らない）、HF train 18k を同じ rare-preserving ルールで 10k に間引き、synth 10k と concat して `data/ocr_v3/train.jsonl` (20,000 行) を書き出す。val は subsample せず HF 2k + synth 2.5k = 4,500 行を全部残す。hash 重複は natural 側優先（実機 webp を持っている方を残す）。

### 修正結果（v3 train 20k、`data/ocr_v3/` の実測値）

| 指標 | v2 (HF 18k) | **v3 train 20k (実測)** | v3 - v2 |
|---|---:|---:|---:|
| count=0 の比率 | 65.68% | **54.03%** | **-11.6pt** |
| count=1 の比率 | 21.20% | 24.25% | +3.0pt |
| count=3 以上の合計 | ~4% | **11.41%** | +7.4pt |
| count=7 の比率 | ~0% | 0.55% | +0.55pt |
| count=10 の比率 | 0.12% | 0.28% | +0.16pt |
| count=15 の比率 | ~0 | 0.07% | +0.07pt |
| count=18 の観測 | 4 件 | **17 件** | +13 |
| 飛の非ゼロ率 (S:R / G:R) | 17.8% / 17.1% | **30.1% / 28.5%** | 約 +12pt |
| 角の非ゼロ率 (S:B / G:B) | 42.9% / 41.5% | 44.5% / 43.6% | ほぼ同 |
| 歩の非ゼロ率 (S:P / G:P) | 68.0% / 64.0% | **78.4% / 77.1%** | +10〜13pt |

「常に 0 ベースライン」の slot_acc が **65.7% → 54.0%**（-11.6pt）。pure synth 10k で得られる -20pt には届かないが、**サンプル数は 20k に増え、画像スタイルは実機で統一**。当初の見込み ~58.5% より 4.5pt 良かったのは、HF 18k の rare-preserving subsample がよく効いたため（18k 中 15,678 件（87%）が rare 判定に入り、common 側の「完全に静かな 8000 件」だけが落ちた）。

**v3 val の分布（4,500 行、`data/ocr_v3/val.jsonl` 実測）**：

| 指標 | v2 val (HF 2k) | **v3 val 4.5k (実測)** |
|---|---:|---:|
| count=0 の比率 | 75.10% | **58.88%** |
| count=3 以上の合計 | 2.98% | 9.61% |
| count=11 以上の観測 | **0 件** | 通算 335 件（count=11 で 120、count=18 で 5） |
| count=18 の観測 | 0 件 | **5 件** |

**val の count=11+ 問題が解消**され、regression head や高枚数向け class weight の効果を `val/hand/slot_acc` の per-count recall で直接評価できるようになった。

### 20k の内訳（provenance タグ、`type` field）

`build_v3_manifest.py` の実行結果より：

| type | 件数 | 由来 |
|---|---:|---|
| `synthetic` | 12,770 | 今回作った synth 10k + HF に元々含まれていた synth の残り |
| `existing` | 3,658 | HF の既存撮影済み `data/detector/*` reuse 分 |
| `natural` | 2,643 | HF の mate 系（`assets/mate{3,5,7,9,11}.sfen`）由来 |
| `opening` | 929 | HF の opening 系（`assets/start_sfens_ply{24,32}.txt`）由来 |
| **合計** | **20,000** | |

per-source per-count recall を取れば、synth が hand tail の学習にどれだけ効いたかを直接測れる。

### v3 で調整するパラメータ

**必要な変更（データマージに伴う）**：

- **train データソース**: `--hf-repo-id` の HF loader だけでは synth 分が読めない。実装コストの低い順に：
  - **(a) 新 HF split を publish**: `build_paired_to_hf.py` を synth 撮影後の webp も拾うよう拡張し、`ocr_paired_v3` として push。既存の HF loader をそのまま使える。**推奨**。`data/ocr_v3/train.jsonl` と `data/ocr_v3/val.jsonl` はこの入口に渡す形。
  - **(b) train_board_ocr.py に extra jsonl の入口**: `--extra-train-jsonl` / `--extra-val-jsonl` を追加、HF pool と concat する `ConcatDataset` にする。HF publish 手間を省ける代わりに loader 側に merge ロジックが増える。
- **`--class-weight-clip-max 10.0 → 15.0`**: v3 20k 実測の raw class weight は count=10 で 19.1、count=15 で 78.4、count=18 で 867。旧 clip=10 だと **count≥7 が全部同じ重み**でキャップされる。**clip を 15 に緩める** と sqrt 後 count=7 で 3.10、count=10 で 4.37、count=13 で 6.62、count=15 で 8.85、count=17 以上が 15 でキャップ、と個別に立ち上がる。それより上は継続キャップ。
- **`EPOCHS=60 → 50`**: サンプル数が 18k → 20k で 11% 増、batch=128 では **epoch あたり 141 → 156 ステップ**、v2 の 60 epoch (8460 steps) 相当は約 **54 epoch** で並ぶ。**50 epoch（v2 比 -8% 総ステップ、cosine scheduler で後半の実効 lr を下げるぶんはトントン）** に設定して cosine と組で回す。

**推奨追加（既存の TODO を v3 で入れると綺麗）**：

- **§D LR scheduler の導入**: `AdamW(lr=6e-4)` 固定を `CosineAnnealingLR(T_max=EPOCHS)` に。resume 時は scheduler state も ckpt に含める。データが変わって baseline を再取得する v3 は、scheduler も同時投入して次の baseline に組み込む好機。
- **`--hand-mode regression` の CLI 露出**: model 側の実装 (`board_ocr.py`) は既にあるが `train_board_ocr.py` に CLI 引数が無い。マージ後の分布は count=3〜18 が実在するので、regression head の効果測定は v2 データより格段にやりやすい。**別実験として v3-cls / v3-reg の 2 系統を並行して回すのが理想**。

**据置き**：

- `BATCH_SIZE=128` / `IMAGE_SIZE=288` / `BACKBONE=mobilenet_v3_small`: v2 との direct delta を測るため、まずはこの 3 点を据え置き、データ差 + class weight + epoch + scheduler の効果を分離する。backbone / image_size の bump は v3 の結果を見てから、v4 で入れる。
- **`hand_weight=1.0`**: v2 で入れた設定、そのまま。
- **X-1 hand logit マスキング**: そのまま。

### v3 想定 config

```bash
# 前提: HF に ocr_paired_v3 が publish 済み、あるいは train_board_ocr.py に extra jsonl 入口がある。
EPOCHS=50 \
BATCH_SIZE=128 \
LR=6e-4 \
IMAGE_SIZE=288 \
BACKBONE=mobilenet_v3_small \
CKPT_DIR=./runs/board-ocr-v3 \
HF_REPO_ID=ultemica/piyoshogi \  # または v3 split の repo
./scripts/train.sh \
    --class-weight-clip-max 15.0 \
    --lr-scheduler cosine  # 実装後
```

**目標**：sfen_full > 0.60（v2 baseline を追加 hand 分布で越える）。同時に per-count recall（特に count=5〜18）を `scripts/inspect/diagnose_hand.py` で計測して、v2 との hand 側改善差を数字で押さえる。**per-source (natural / synthetic) の per-count recall** も並行して見ると、synth の効きを直接評価できる。

### 開始前チェックリスト

1. **piyo-hook 撮影完了待ち**: `data/synth_train_sfens.jsonl` + `data/synth_val_sfens.jsonl` の全 SFEN が 4 端末で撮影されて `data/ocr/<device>/<hash>.webp` に落ちること。所要時間はデータ量次第（10k+2.5k SFEN × 4 端末 = 50,000 webp）。
2. **manifest 統合（済）**: `scripts/data/build_v3_manifest.py` を実行済み。`data/ocr_v3/train.jsonl` (20,000 行) と `data/ocr_v3/val.jsonl` (4,500 行) が生成済み。この 2 本を parquet 化担当に渡して `ocr_paired_v3` に育ててもらう。
3. **HF publish or extra jsonl 対応**: (a) `build_paired_to_hf.py` を `data/ocr_v3/` 入力に切り替えて回し `ocr_paired_v3` として push、または (b) `train_board_ocr.py` に `--extra-train-jsonl` の入口を追加。
4. §D CosineAnnealingLR の実装 + resume 対応。
5. `--hand-mode` CLI 引数の追加（別実験線として）。
6. `--class-weight-clip-max` を 15 に上げても勾配が暴発しないことを 5 epoch で確認。
7. baseline 比較のため v2 の epoch 60 ckpt を凍結（`runs/board-ocr-v2/` を触らない）。
