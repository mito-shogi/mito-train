# 学習改善計画

board_ocr の学習を回しながら気づいた改善候補と、次のイテレーションの優先度メモ。
`docs/` は MITO 側から同期する仕様書用なので、内部計画はここに置く。

## 現状の観測（2026-07-08 board_ocr full run）

前回 20 エポック → 追加 40 エポック（`RESUME=latest EPOCHS=60`）で継続学習中。

分布可視化ツール：`scripts/inspect/analyze_hand_distribution.py`（出力: `runs/hand-distribution.png`）

| epoch | loss | board_loss | hand_loss | cell_acc | sfen_acc |
|------:|-----:|-----------:|----------:|---------:|---------:|
| 5     | 0.329 | 0.025 | 0.607 | 0.993 | 0.046 |
| 20    | 0.120 | 0.007 | 0.226 | 0.998 | 0.324 |
| 30    | 0.099 | 0.006 | 0.187 | 0.998 | 0.417 |

epoch 20 時点の val 側：
- `val/board/cell_acc = 0.999`（train より高い＝過学習なし）
- `val/hand/slot_acc = 0.939`
- `val/hand/full_acc = 0.385`
- `val/sfen/full_acc = 0.369`

### 読み取れる状態

- **board head は epoch 5 前後で飽和**（cell_acc 0.99+ で伸びしろほぼ無し）。
- **hand head はまだ学習中**（hand_loss / sfen_acc とも右肩下がり継続）。ボトルネックはこちら。
- val > train の傾向で過学習は現状ゼロ。エポックはさらに伸ばせる。

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

**観測**（`scripts/inspect/analyze_hand_distribution.py`、`data/train.jsonl` n=27,000 局面 = 378,000 スロットラベル）：

- ラベル `0`（=空スロット）が **55.94%**。`1` が 26.09%、`2` が 8.65%、以降 3%以下まで急落。
- モデルが常に「0」と答えても素の accuracy が 56% 出せる構造。
- 駒種別の非ゼロ率：
  - 歩：**90%以上**（唯一 10枚超まで散らばる）
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

### 持ち駒 count=10 のギャップ（要対応）

`data/train.jsonl` の全 27,000 局面を集計しても、**どのスロットにも count=10 が 1 件も現れない**（count=9 は 585 件、count=11 は 158 件あるのに）。加えて count=17, 18 も完全に不在。

**原因**：ぴよ将棋（データ生成元）側で 10 枚持ちの局面が SFEN 出力の段階で生成されない既知のバグ。mito-train 側で修正できるものではなく、入力データの制約として顕在化している。

**⚠️ 歩は要注意**：**実局面では歩 10 枚（およびそれ以上）の局面は普通に発生する**。飛・角・金・銀・桂・香は 10 枚以上を持てないので count=10 のギャップは実害ゼロだが、**歩のスロット（S:P, G:P）だけは実運用推論で count=10 を要求される場面がある**。学習データに 10 枚が 1 件も無い状態だと、モデルは歩 10 枚を 9 or 11 に誤読する挙動になる。

**推論への影響**：

- 歩以外のスロット：学習時に count=10 が来ないだけでなく実運用でも来ないので影響なし。
- **歩スロット**：実局面で count=10 が来たとき、そのラベルの logit は全く学習されていない → 8/9/11 に丸める挙動。
- count=17, 18 は理論上歩でのみあり得るが実局面での出現頻度は極めて低い。優先度は 10 より低い。

**対策候補**：

- **A. 歩スロットだけデータ合成で 10 枚局面を追加**（本命）。
  - piyo-hook 修正を待つ間の暫定策として、既存の 9 枚 / 11 枚局面から歩コマ画像を貼り替えて 10 枚版を合成。
  - 数百件でも「count=10 の logit が完全ゼロ学習」状態は解消できる。
  - 対象を歩スロットに限定すれば副作用が少ない。
- **B. piyo-hook 側のバグ修正を issue 化**（本筋）。上流で直せば A は不要になる。

優先度：**A を短期で、B を並行で issue 化**。C（駒種別の出力次元制約）は下記の別項で扱う。

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
