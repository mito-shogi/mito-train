# mito-train レポ セットアップガイド

**発行日**: 2026-07-07
**対象読者**: mito-train レポの初期セットアップ担当 LLM / 開発者
**関連文書**:
- `docs/ocr-phase2-plan.md` — Phase 2/3 全体計画
- `docs/ocr-model-interface.md` — MITO ↔ mito-train モデル契約
- `docs/piyo-data-generation-spec.md` — piyo-hook 向けデータ生成仕様

---

## 0. mito-train の位置付け

3 レポ体制:

| レポ | 役割 | 言語 | 実行環境 |
|---|---|---|---|
| **MITO** (この repo) | 推論・UI・デプロイ | TypeScript | Cloudflare Pages + ブラウザ |
| **piyo-hook** (別レポ) | ipa 解析・Frida hook・データ生成 | ? | JB 端末 + 母艦 |
| **mito-train** (これから作る) | 学習・ONNX 出力 | Python / PyTorch | GPU マシン (4070 Ti / 5060 / M3 Max) |

**mito-train が担当すること**:
- PyTorch で 3 モデル (board-detector, piece-classifier, hand-classifier) を学習
- augmentation パイプライン (Twitter 劣化再現)
- ONNX エクスポート + INT8 量子化
- 契約 (`docs/ocr-model-interface.md`) 準拠のモデルを R2 に配置

**mito-train が担当しないこと**:
- データ生成 (piyo-hook の担当)
- 推論・UI (MITO の担当)

---

## 1. リポジトリ構造 (推奨)

```
mito-train/
├── README.md                     初回セットアップ手順
├── pyproject.toml                Python 依存
├── .gitignore                    data/, models/*.onnx, .venv/, __pycache__/
├── .rclone.conf.example          R2 認証設定の雛形
├── data/                         ← Git 管理外、R2 から pull
│   ├── piyo-train/
│   │   ├── manifest-train.jsonl
│   │   ├── manifest-valid.jsonl
│   │   └── *.png
│   └── piyo-test-realistic/
│       ├── annotations.json
│       └── *.png
├── docs/                         MITO 側から同期する仕様書
│   ├── ocr-model-interface.md    (MITO からコピー)
│   ├── piyo-data-generation-spec.md
│   └── ocr-metrics.md
├── mito_train/                   Python パッケージ本体
│   ├── __init__.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── piyo_dataset.py       manifest*.jsonl を読む PyTorch Dataset
│   │   └── augment.py            albumentations パイプライン
│   ├── models/
│   │   ├── __init__.py
│   │   ├── board_detector.py     tiny UNet or 4隅回帰
│   │   ├── piece_classifier.py   29クラス tiny CNN
│   │   └── hand_classifier.py    持ち駒 multi-head
│   ├── training/
│   │   ├── train_board.py
│   │   ├── train_piece.py
│   │   └── train_hand.py
│   ├── export/
│   │   ├── to_onnx.py            PyTorch → ONNX
│   │   └── quantize.py           INT8 量子化
│   └── eval/
│       ├── metrics.py            piece accuracy, IoU, exact-match
│       └── evaluate.py           test-realistic に対する評価
├── notebooks/                    Jupyter 実験用 (Git 管理)
│   ├── 01-explore-data.ipynb
│   ├── 02-augment-preview.ipynb
│   └── 03-loss-curves.ipynb
├── scripts/                      運用 shell script
│   ├── download-data.sh          R2 → data/ を pull
│   ├── upload-data.sh            data/ → R2 に push (通常は piyo-hook 側で使用)
│   ├── upload-models.sh          学習済み ONNX を R2 に配置
│   └── run-pipeline.sh           end-to-end 学習パイプライン
└── models/                       (option) 学習済み ONNX を Git 管理
    ├── board-detector-v1.onnx
    ├── piece-classifier-v1.onnx
    └── hand-classifier-v1.onnx
```

---

## 2. Cloudflare R2 セットアップ (初回のみ、GUI 操作)

**ユーザー側で実施が必要な作業**:

### 2.1 R2 バケット作成

1. Cloudflare Dashboard → R2 → Create bucket
2. Bucket 名: `mito-datasets`
3. Location: 最寄りリージョン (自動でよい)
4. Public access: **無効** (private のまま)

### 2.2 API トークン発行

1. Cloudflare Dashboard → R2 → Manage R2 API Tokens → Create API Token
2. 権限: **Object Read & Write** (mito-train 用) と **Object Read only** (MITO CI 用) を分けて 2 つ作る
3. 対象 bucket: `mito-datasets`
4. TTL: **90 日** (定期ローテーション)
5. 発行された `Access Key ID` と `Secret Access Key` を安全な場所に保管

**注意**: シークレットは 1 度しか表示されない。パスワードマネージャに保存推奨。

### 2.3 Cloudflare Account ID とエンドポイント確認

R2 バケットの詳細画面から:
- **Account ID**: `1234567890abcdef...`
- **S3 API endpoint**: `https://<account-id>.r2.cloudflarestorage.com`

---

## 3. mito-train 側の環境構築

### 3.1 rclone インストール

各学習マシン (4070 Ti / 5060 / M3 Max) で:

```bash
# macOS (M3 Max)
brew install rclone

# Linux (4070 Ti / 5060 マシン)
sudo apt install rclone   # or curl https://rclone.org/install.sh | sudo bash
```

### 3.2 rclone remote 設定

```bash
rclone config
```

対話プロンプト:
- `n` (new remote)
- name: `r2`
- Storage: `s3` (Amazon S3 Compliant Storage)
- provider: `Cloudflare` (v4.15+ にプリセットあり、なければ `Other`)
- env_auth: `false` (キーを設定ファイルに保存)
- access_key_id: (Section 2.2 で発行した Access Key ID)
- secret_access_key: (同上)
- region: `auto`
- endpoint: `https://<account-id>.r2.cloudflarestorage.com`
- location_constraint: 空
- acl: 空
- 残りはデフォルト

設定確認:

```bash
rclone lsd r2:
# → mito-datasets が見えれば成功
```

### 3.3 Python 環境

```bash
# Python 3.11+ を推奨
uv venv .venv
source .venv/bin/activate

# PyTorch (CUDA 12.x 対応版、4070 Ti / 5060 用)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# M3 Max ユーザーは MPS 対応の通常版
# uv pip install torch torchvision

# 共通パッケージ
uv pip install albumentations opencv-python-headless
uv pip install onnx onnxruntime onnxruntime-tools
uv pip install pillow numpy pyyaml tqdm
uv pip install wandb  # optional
```

または `pyproject.toml` に依存を書いて `uv sync` する。

### 3.4 GPU 動作確認

```python
# scripts/verify-gpu.py として保存
import torch

if torch.cuda.is_available():
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Capability: {torch.cuda.get_device_capability(0)}")
elif torch.backends.mps.is_available():
    print("MPS (Apple Silicon)")
else:
    print("CPU only")

# 簡単な matmul で計測
import time
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
x = torch.randn(4096, 4096, device=device)
y = torch.randn(4096, 4096, device=device)
if device == 'cuda': torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    z = x @ y
if device == 'cuda': torch.cuda.synchronize()
print(f"{device}: {(time.perf_counter() - t0) * 100:.1f} ms/matmul")
```

期待値:
- RTX 4070 Ti: **2-3 ms/matmul**
- RTX 5060: **3-5 ms/matmul**
- M3 Max: **5-10 ms/matmul**
- CPU: 500 ms+

---

## 4. データ pull スクリプト

### `scripts/download-data.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

REMOTE=${REMOTE:-r2}
BUCKET=${BUCKET:-mito-datasets}
DATA_DIR=${DATA_DIR:-./data}

mkdir -p "$DATA_DIR"

echo "Pulling piyo-train from ${REMOTE}:${BUCKET}/piyo-train/ ..."
rclone sync "${REMOTE}:${BUCKET}/piyo-train/" "${DATA_DIR}/piyo-train/" \
  --progress \
  --transfers 8 \
  --checkers 16 \
  --fast-list

echo "Pulling piyo-test-realistic from ${REMOTE}:${BUCKET}/piyo-test-realistic/ ..."
rclone sync "${REMOTE}:${BUCKET}/piyo-test-realistic/" "${DATA_DIR}/piyo-test-realistic/" \
  --progress \
  --transfers 8 \
  --checkers 16 \
  --fast-list

echo "Done. Local data:"
find "${DATA_DIR}" -type f | head -10
du -sh "${DATA_DIR}"/*
```

使い方:

```bash
chmod +x scripts/download-data.sh
./scripts/download-data.sh
```

**運用ルール**:
- **`rclone sync` を使う** (`copy` ではなく)。remote で削除されたファイルはローカルからも消す
- `--transfers 8` は並列度、帯域が細いなら 4 に落とす
- 初回のみ大きい (現状 738MB) が、以降は差分のみ

### `scripts/upload-data.sh` (piyo-hook 側で使用)

```bash
#!/usr/bin/env bash
set -euo pipefail

REMOTE=${REMOTE:-r2}
BUCKET=${BUCKET:-mito-datasets}
SOURCE=${SOURCE:-./assets/train}
MANIFEST=${MANIFEST:-./assets/manifest.jsonl}

echo "Uploading ${SOURCE}/*.png to ${REMOTE}:${BUCKET}/piyo-train/ ..."
rclone sync "${SOURCE}" "${REMOTE}:${BUCKET}/piyo-train/" \
  --progress \
  --transfers 8 \
  --checkers 16 \
  --exclude "*.json*" \
  --include "*.png"

echo "Uploading manifest ..."
rclone copy "${MANIFEST}" "${REMOTE}:${BUCKET}/piyo-train/"
if [ -f "./assets/manifest-train.jsonl" ]; then
  rclone copy "./assets/manifest-train.jsonl" "${REMOTE}:${BUCKET}/piyo-train/"
fi
if [ -f "./assets/manifest-valid.jsonl" ]; then
  rclone copy "./assets/manifest-valid.jsonl" "${REMOTE}:${BUCKET}/piyo-train/"
fi

echo "Done."
```

### `scripts/upload-models.sh` (mito-train 側で使用)

```bash
#!/usr/bin/env bash
set -euo pipefail

REMOTE=${REMOTE:-r2}
BUCKET=${BUCKET:-mito-datasets}
MODEL_DIR=${MODEL_DIR:-./models}

echo "Uploading ONNX models to ${REMOTE}:${BUCKET}/models/ ..."
rclone copy "${MODEL_DIR}" "${REMOTE}:${BUCKET}/models/" \
  --progress \
  --include "*.onnx"

echo "Done. Latest models:"
rclone ls "${REMOTE}:${BUCKET}/models/"
```

---

## 5. .gitignore の中身

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/

# 学習データ (R2 から pull するので Git 管理外)
data/

# 学習中間物
runs/
wandb/
checkpoints/
*.pt
*.pth

# ONNX モデル (小さいので Git 管理してもよいが、大きくなるなら除外)
# models/*.onnx

# rclone 認証設定 (機密)
.rclone.conf

# Jupyter checkpoint
.ipynb_checkpoints/
```

**注意**: `docs/` は Git 管理対象。MITO 側からコピーしてくる仕様書は追跡する。

---

## 6. データ読み込みコード雛形

### `mito_train/datasets/piyo_dataset.py`

```python
"""ぴよ将棋クリーン画像用の PyTorch Dataset."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Literal

import torch
from torch.utils.data import Dataset
from PIL import Image


class PiyoDataset(Dataset):
    """
    manifest-train.jsonl / manifest-valid.jsonl を読んで
    (image_tensor, sfen_str) を返す。

    board / piece / hand ごとに前処理が違うので、transform 関数を注入する。
    """

    def __init__(
        self,
        data_root: Path,
        split: Literal['train', 'valid'],
        transform: Callable | None = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        manifest_path = self.data_root / f'manifest-{split}.jsonl'
        self.entries: list[dict] = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.entries.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        entry = self.entries[idx]
        # manifest の "file" は "d0/xxx.png" 形式だが、実体は flat 命名
        # → basename だけ取って data_root 直下から読む
        file_basename = Path(entry['file']).name
        image_path = self.data_root / file_basename
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, entry['sfen_normalized']
```

### 使い方

```python
from mito_train.datasets.piyo_dataset import PiyoDataset

train_ds = PiyoDataset(
    data_root='./data/piyo-train',
    split='train',
    transform=None,  # 後で augmentation 差し込む
)
print(f"train: {len(train_ds)} entries")
img, sfen = train_ds[0]
```

---

## 7. 初回セットアップ手順まとめ

新しいマシンで mito-train を初めて動かすとき:

```bash
# 1. リポジトリ clone
git clone git@github.com:<owner>/mito-train.git
cd mito-train

# 2. rclone インストール & 設定 (Section 3.1 / 3.2)
brew install rclone   # or apt install rclone
rclone config          # r2 remote 設定

# 3. Python 環境
uv venv .venv
source .venv/bin/activate
uv sync   # または pyproject.toml から個別 install

# 4. データ pull (R2 → data/)
./scripts/download-data.sh

# 5. GPU 動作確認
python scripts/verify-gpu.py

# 6. 最初の学習を試す (データ 703枚あれば PoC 学習は動く)
python -m mito_train.training.train_piece --data ./data/piyo-train --epochs 5

# 7. ONNX 出力
python -m mito_train.export.to_onnx --checkpoint runs/piece-cnn/best.pt

# 8. モデルを R2 に push (MITO 側に配信する)
./scripts/upload-models.sh
```

---

## 8. MITO 側との連携

### モデル配信の流れ

```
[mito-train] 学習完了
    ↓
models/piece-classifier-v1.onnx (300KB)
    ↓ upload-models.sh (rclone)
[R2] r2://mito-datasets/models/
    ↓ MITO の CI で pull
[MITO] public/models/piece-classifier-v1.onnx
    ↓ Cloudflare Pages デプロイ
[ブラウザ] ONNX Runtime Web で読む
```

### 契約遵守の CI 検証 (推奨)

mito-train の GitHub Actions で:

1. 学習コード完了時に ONNX 出力
2. `onnxruntime` で読み込みテスト
3. `session.inputNames` と `session.outputNames` を契約 (`docs/ocr-model-interface.md` Section 2/3/4) と比較
4. 入力 shape、出力 shape、dtype を検証
5. 不一致なら R2 upload をブロック

これで **契約違反したモデルが本番に流れない**。

---

## 9. トラブルシューティング

### rclone 認証エラー

```
rclone: error 403: Forbidden
```

- API トークンの権限確認 (Read & Write が付いてるか)
- Access Key ID / Secret Access Key の typo チェック
- `rclone config show r2` で設定を確認

### rclone が遅い

```bash
# 並列度を上げる
rclone sync --transfers 16 --checkers 32 --fast-list ...
```

- `--transfers` は同時ダウンロード数
- `--checkers` は同時に差分チェックする数
- `--fast-list` はディレクトリリスティングをまとめて取得 (R2 は list コストが高いので有効)

### CUDA out of memory

```python
# バッチサイズを下げる、または gradient accumulation を使う
batch_size = 64  # デフォルトから半分
grad_accum_steps = 2  # 実効バッチ 128
```

### M3 Max で MPS が遅い

- MPS は特定の Op で CPU fallback する
- `PYTORCH_ENABLE_MPS_FALLBACK=1` で fallback 許可
- 一部モデルは `torch.compile` を無効化 (MPS で不安定)

---

## 10. 次にやること (mito-train 立ち上げ後)

1. **Milestone 1**: 703 SFEN の training で PoC 学習 (loss curve 確認)
2. **Milestone 2**: 3,000 SFEN で Piece Accuracy 95%+ 到達
3. **Milestone 3**: Twitter 劣化 augmentation スイープ
4. **Milestone 4**: test-realistic 100 枚に対する Exact-Match Rate 測定
5. **Milestone 5**: MITO 側 Mock を実モデルに差し替え → end-to-end 動作確認

詳細は `docs/ocr-phase2-plan.md` の Milestone 一覧を参照。

---

## 11. 変更履歴

| 日付 | 変更内容 |
|---|---|
| 2026-07-07 | 初版作成 (Cloudflare R2 採用) |
