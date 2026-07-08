# ぴよ将棋 アセット・データ配置ルール

**発行日**: 2026-07-07
**目的**: mito-train レポ内で「どこに何を置くか」を一意に決める。
**関連**:
- `docs/mito-train-setup.md` — セットアップ全体
- `docs/piyo-piece-templates.md` — 駒テンプレの命名規則 (14 デザイン × 30 駒)
- `docs/piyo-data-generation-spec.md` — piyo-hook 側でのデータ生成仕様
- `docs/ocr-model-interface.md` — MITO ↔ mito-train モデル契約

---

## 0. 3 種類の「静的リソース」を混同しないこと

mito-train が扱う画像素材は大きく分けて 3 種類ある。**これらは絶対に同じディレクトリに置かない。**

| 種別 | 例 | 置き場所 | Git 管理 | 由来 |
|---|---|---|---|---|
| **piyo 駒テンプレ** | `k1_000_Normal.png` (420 枚固定) | `assets/piyo/` | 除外 | ぴよ将棋アプリの内部データ (piyo-hook 経由) |
| **学習データ (クリーン)** | `d0/xxx.png` + `manifest-*.jsonl` | `data/piyo-train/` | 除外 | piyo-hook で生成 |
| **評価データ (実写系)** | Twitter 劣化を模した実写風画像 + `annotations.json` | `data/piyo-test-realistic/` | 除外 | 手作業 or piyo-hook |

全部 R2 バケット `mito-datasets` に一元管理し、mito-train はそれを **pull するだけ**。

---

## 1. R2 バケットの中身

**バケット名**: `mito-datasets`

```
r2://mito-datasets/
├── piyo-assets/                     ← 駒テンプレ (14 デザイン × 30 駒 = 420 枚)
│   ├── k1/
│   │   ├── k1_000_Normal.png        先手・王将
│   │   ├── k1_001_Normal.png        先手・飛車
│   │   ├── ...
│   │   └── k1_117_Normal.png        後手・と金
│   ├── k2/
│   ├── ...
│   ├── k7/
│   ├── k101/                        ダークモード対応
│   ├── ...
│   └── k107/
├── piyo-train/                      ← クリーン学習データ
│   ├── manifest.jsonl               (全件、参考用)
│   ├── manifest-train.jsonl         train split (~85%)
│   ├── manifest-valid.jsonl         valid split (~15%)
│   ├── manifest-split-meta.json     split 生成メタ (seed 等)
│   └── *.png                        画像本体 (basename でフラット配置)
├── piyo-test-realistic/             ← Twitter 劣化再現などの評価用
│   ├── annotations.json
│   └── *.png
└── models/                          ← 学習済み ONNX の配布元
    ├── board-detector-v1.onnx
    ├── piece-classifier-v1.onnx
    └── hand-classifier-v1.onnx
```

**運用ルール**:
- `piyo-assets/` の書き込み権限は piyo-hook 側の CI のみが持つ (テンプレはめったに更新しない)。
- `piyo-train/` は piyo-hook が定期更新、mito-train は read-only で pull。
- `piyo-test-realistic/` は手作業更新、mito-train は read-only で pull。
- `models/` は mito-train のみが push、MITO 側 CI が pull。

---

## 2. mito-train ローカルの配置

```
mito-train/
├── assets/
│   └── piyo/                        ← R2:piyo-assets/ をここに sync
│       ├── .gitkeep
│       ├── k1/…                     (Git 管理外、.gitignore で除外)
│       └── …
├── data/                            ← R2:piyo-train/, piyo-test-realistic/ をここに sync
│   ├── .gitkeep
│   ├── piyo-train/                  (Git 管理外)
│   └── piyo-test-realistic/         (Git 管理外)
└── models/                          ← 学習の出力先、R2:models/ に push
    ├── .gitkeep
    ├── board-detector-v1.onnx       (Git 管理外)
    ├── piece-classifier-v1.onnx     (Git 管理外)
    └── hand-classifier-v1.onnx      (Git 管理外)
```

### .gitignore による除外

`assets/piyo/*`、`data/`、`models/` はすべて `.gitignore` で追跡外。
`.gitkeep` だけ Git 管理してディレクトリ構造は保証する。

これにより:
- レポジトリを clone しても大きな binary は入ってこない (clone が速い)
- R2 が single source of truth になる (「local が古い」問題を避けられる)
- 誤って画像を commit するリスクが下がる

---

## 3. 同期スクリプト

### 3.1 データを pull する

```bash
# 学習・評価データ (data/)
./scripts/download-data.sh

# 駒テンプレ (assets/piyo/)
./scripts/download-piyo-assets.sh
```

環境変数で挙動を調整可能:
- `REMOTE` — rclone remote 名 (default `r2`)
- `BUCKET` — バケット名 (default `mito-datasets`)
- `DATA_DIR` / `ASSETS_DIR` — 出力先

### 3.2 モデルを push する

```bash
./scripts/upload-models.sh
```

`./models/*.onnx` を `r2://mito-datasets/models/` に copy する (sync ではなく copy にして意図しない削除を防ぐ)。

---

## 4. パスの参照ルール

コード内で asset / data を参照するときは **必ずルート相対**。絶対パスや `~` は使わない。

推奨:
```python
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets"

train_ds = PiyoDataset(DATA_ROOT / "piyo-train", split="train")
```

非推奨:
```python
train_ds = PiyoDataset("/home/user/mito-train/data/piyo-train", split="train")  # NG
```

環境変数で差し替えたい場合は `MITO_TRAIN_DATA_ROOT` を read するヘルパを追加する (将来対応)。

---

## 5. なぜこの配置にしたか

- **`assets/` と `data/` を分けた理由**: 「変更頻度」と「所有者」が違う。テンプレは piyo-hook 側の CI が管理する固定リソース、data は piyo-hook が育て続ける動的リソース。混ざると .gitignore ルールがぶれる。
- **piyo-train の画像を flat 配置にした理由**: `manifest-*.jsonl` の `file` フィールドは `d0/xxx.png` 形式だが、実体は basename のみで解決する既存の運用に合わせる (`PiyoDataset` 実装参照)。
- **`models/` を .gitignore に含めた理由**: 学習中は複数 experiment の ONNX が並ぶことがあり、その全部を Git 管理する意味は薄い。配布 tag は R2 で運用する。もし少数の公式版だけ Git 管理したくなったら、`models/official/` を except-rule で include すればよい。

---

## 6. 変更履歴

| 日付 | 変更内容 |
|---|---|
| 2026-07-07 | 初版作成。R2 `mito-datasets` バケットの 4 プレフィックス構成を確定 |
