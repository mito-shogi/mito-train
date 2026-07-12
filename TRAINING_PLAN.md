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
| convnext_tiny | 28.0M | 0.994 | 0.716 |
| convnext_nano | 15.1M | (未完走) | (未完走) |

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

**観測**（`scripts/inspect/analyze_hand_distribution.py` を旧スナップショット n=27,000 局面 = 378,000 スロットラベルに対して実行した結果。傾向は現行データでも変わらないはずだが、最新の数字は再集計推奨）：

- ラベル `0`（=空スロット）が **過半（旧集計で 55.94%）**。`1` が 26%、`2` が 9%、以降 3% 以下まで急落。
- モデルが常に「0」と答えても素の accuracy が過半（旧集計で 56%）出せる構造。
- 駒種別の非ゼロ率：
  - 歩：**90%以上**（唯一 10枚超まで散らばる。最新データで歩 10 枚も収録済み）
  - 金・銀・桂：43〜46%
  - 香・角：30〜34%
  - **飛：19〜20%**（ほぼ 0 or 1 枚）
- 同じ hand head で 14 スロット一括処理しているが、実質**駒種ごとに難易度が全く違うタスク**。

**素朴な `1 / freq` の危険性**：クラス重みが 0.094（count=0）〜 9947（count=16）まで**5 桁開く**。希少クラスの勾配が暴発して学習不安定になるので、そのままは NG。

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
