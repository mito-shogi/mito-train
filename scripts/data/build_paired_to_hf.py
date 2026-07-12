"""Build paired (horizontal, SFEN-per-row) parquet for ultemica/piyoshogi.

Reads:
    data/ocr/{train,val}.jsonl                     — SFEN + hash + type
    data/ocr/<device>/<hash>.webp                  — 4-device webp for OCR
    data/detector/{train,val}.jsonl                — SFEN + hash + device + type
    data/detector/<device>/<hash>.webp             — 4-device webp for detector
    data/test/device_bboxes.json                   — per-device board_view_px

Emits:
    uploads/piyoshogi_paired/ocr_paired/{train,val}-*.parquet
    uploads/piyoshogi_paired/detector_paired/{train,val}-*.parquet
    uploads/piyoshogi_paired/README.md

Schema:
    ocr_paired      : {sfen, hash, type, images: list<struct{bytes,path}>, devices: list<string>}
    detector_paired : {sfen, hash, images, devices, bboxes: list<list<int32>>}

Env:
    HF_TOKEN                       required for push
    HF_HUB_ENABLE_HF_TRANSFER=1    recommended
    HF_XET_HIGH_PERFORMANCE=1      recommended
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

DEVICES = ["iPhone10,1", "iPhone11,8", "iPhone15,4", "iPad14,10"]
DATA_ROOT = Path("data")
STAGING = Path("uploads/piyoshogi_paired")

OCR_SCHEMA = pa.schema([
    ("sfen", pa.string()),
    ("hash", pa.string()),
    ("type", pa.string()),
    ("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
    ("devices", pa.list_(pa.string())),
])

DETECTOR_SCHEMA = pa.schema([
    ("sfen", pa.string()),
    ("hash", pa.string()),
    ("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
    ("devices", pa.list_(pa.string())),
    ("bboxes", pa.list_(pa.list_(pa.int32()))),
])


class _ShardedParquetWriter:
    """pq.ParquetWriter with byte-based rotation and post-close rename."""

    def __init__(self, out_dir: Path, prefix: str, schema: pa.Schema,
                 max_shard_bytes: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.schema = schema
        self.max_shard_bytes = max_shard_bytes
        out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_paths: list[Path] = []
        self._writer: pq.ParquetWriter | None = None
        self._sink = None
        self._current_path: Path | None = None
        self._open_new()

    def _open_new(self) -> None:
        idx = len(self.shard_paths)
        # placeholder name — renamed on finalize
        p = self.out_dir / f"{self.prefix}-{idx:05d}-of-XXXXX.parquet"
        self._current_path = p
        self._sink = pa.OSFile(str(p), "wb")
        self._writer = pq.ParquetWriter(self._sink, self.schema, compression="snappy")
        self.shard_paths.append(p)

    def write_batch(self, batch: list[dict]) -> None:
        if not batch:
            return
        tbl = pa.Table.from_pylist(batch, schema=self.schema)
        self._writer.write_table(tbl)
        # rotate if the on-disk file (which is what upload_folder sees) exceeded threshold.
        # write_table flushes at row-group boundary, so tell() lags a bit, but 90% threshold
        # gives comfortable margin.
        if self._current_path.stat().st_size >= int(self.max_shard_bytes * 0.9):
            self._writer.close()
            self._sink.close()
            self._open_new()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._sink.close()
            self._writer = None
        # if last shard is empty (no rows written), remove it
        for p in list(self.shard_paths):
            if p.exists() and p.stat().st_size < 1024:  # empty parquet is a few hundred bytes
                pf = pq.ParquetFile(p)
                if pf.metadata.num_rows == 0:
                    p.unlink()
                    self.shard_paths.remove(p)

    def finalize_shard_names(self) -> None:
        total = len(self.shard_paths)
        renamed = []
        for i, p in enumerate(self.shard_paths):
            final = p.with_name(f"{self.prefix}-{i:05d}-of-{total:05d}.parquet")
            p.rename(final)
            renamed.append(final)
        self.shard_paths = renamed


def _load_bboxes() -> dict[str, list[int]]:
    with open(DATA_ROOT / "test/device_bboxes.json") as f:
        cfg = json.load(f)
    out = {}
    for dev in DEVICES:
        b = cfg["devices"][dev]["board_view_px"]
        out[dev] = [int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])]
    return out


def _build_ocr(split: str, out_dir: Path, max_shard_bytes: int, flush_batch: int,
               limit: int | None) -> int:
    manifest = DATA_ROOT / "ocr" / f"{split}.jsonl"
    writer = _ShardedParquetWriter(out_dir, prefix=split, schema=OCR_SCHEMA,
                                    max_shard_bytes=max_shard_bytes)
    batch: list[dict] = []
    total = 0
    with open(manifest) as f:
        for line in f:
            if limit is not None and total >= limit:
                break
            e = json.loads(line)
            images: list[dict] = []
            devs: list[str] = []
            for dev in DEVICES:
                p = DATA_ROOT / "ocr" / dev / f"{e['hash']}.webp"
                if not p.exists():
                    continue
                images.append({"bytes": p.read_bytes(), "path": p.name})
                devs.append(dev)
            if not images:
                continue
            batch.append({
                "sfen": e["sfen"],
                "hash": e["hash"],
                "type": e.get("type", "unknown"),
                "images": images,
                "devices": devs,
            })
            total += 1
            if len(batch) >= flush_batch:
                writer.write_batch(batch)
                batch = []
    if batch:
        writer.write_batch(batch)
    writer.close()
    writer.finalize_shard_names()
    return total


def _build_detector(split: str, out_dir: Path, bboxes: dict[str, list[int]],
                    max_shard_bytes: int, flush_batch: int,
                    limit: int | None) -> int:
    manifest = DATA_ROOT / "detector" / f"{split}.jsonl"
    # dedup by hash (jsonl is 1 row per (hash, device))
    by_hash: dict[str, dict] = {}
    with open(manifest) as f:
        for line in f:
            e = json.loads(line)
            by_hash.setdefault(e["hash"], e)  # first-seen row per hash

    writer = _ShardedParquetWriter(out_dir, prefix=split, schema=DETECTOR_SCHEMA,
                                    max_shard_bytes=max_shard_bytes)
    batch: list[dict] = []
    total = 0
    for h, e in by_hash.items():
        if limit is not None and total >= limit:
            break
        images: list[dict] = []
        devs: list[str] = []
        bbs: list[list[int]] = []
        for dev in DEVICES:
            p = DATA_ROOT / "detector" / dev / f"{h}.webp"
            if not p.exists():
                continue
            images.append({"bytes": p.read_bytes(), "path": p.name})
            devs.append(dev)
            bbs.append(bboxes[dev])
        if not images:
            continue
        batch.append({
            "sfen": e["sfen"],
            "hash": h,
            "images": images,
            "devices": devs,
            "bboxes": bbs,
        })
        total += 1
        if len(batch) >= flush_batch:
            writer.write_batch(batch)
            batch = []
    if batch:
        writer.write_batch(batch)
    writer.close()
    writer.finalize_shard_names()
    return total


def _write_readme(path: Path) -> None:
    """Paired-only README (ocr_paired + detector_paired). Overwrites Hub README."""
    path.write_text("""\
---
license: mit
task_categories:
- image-classification
- image-to-text
language:
- ja
tags:
- shogi
- board-recognition
- piyoshogi
- multi-device
pretty_name: PiyoShogi Multi-Device Dataset
configs:
- config_name: ocr_paired
  data_files:
  - split: train
    path: ocr_paired/train-*.parquet
  - split: val
    path: ocr_paired/val-*.parquet
- config_name: detector_paired
  data_files:
  - split: train
    path: detector_paired/train-*.parquet
  - split: val
    path: detector_paired/val-*.parquet
dataset_info:
- config_name: ocr_paired
  features:
  - name: sfen
    dtype: string
  - name: hash
    dtype: string
  - name: type
    dtype: string
  - name: images
    sequence: image
  - name: devices
    sequence: string
  splits:
  - name: train
    num_examples: 18000
  - name: val
    num_examples: 2000
- config_name: detector_paired
  features:
  - name: sfen
    dtype: string
  - name: hash
    dtype: string
  - name: images
    sequence: image
  - name: devices
    sequence: string
  - name: bboxes
    sequence:
      sequence: int32
  splits:
  - name: train
    num_examples: 800
  - name: val
    num_examples: 200
---

# PiyoShogi Multi-Device Dataset

ぴよ将棋の盤面認識モデル訓練用データセット。**4機種の実機スクリーンショットを SFEN 単位で束ねた横持ち形式**で提供。盤面OCR (`ocr_paired`) と盤面検出 (`detector_paired`) の 2 config 構成。

## 対応機種

| 機種名 | 識別子 (devices 列) | 画面解像度 |
| --- | --- | --- |
| iPhone 8 | `iPhone10,1` | 750 × 1334 |
| iPhone XR | `iPhone11,8` | 828 × 1792 |
| iPhone 15 | `iPhone15,4` | 1179 × 2556 |
| iPad Air M3 | `iPad14,10` | 1640 × 2360 |

## `ocr_paired`

盤面領域のみクロップした画像で、SFEN予測モデル用。**1 行 = 1 SFEN**、4 機種分の画像を list で持つ。

| Column | Type | 説明 |
| --- | --- | --- |
| `sfen` | `string` | SFEN形式の局面文字列 |
| `hash` | `string` | SFEN の SHA-256 |
| `type` | `string` | `existing` / `natural` / `synthetic` / `opening` |
| `images` | `Sequence(Image)` | 各機種の盤面クロップ (lossless WebP) |
| `devices` | `Sequence(string)` | images と同じ順序の機種識別子 |

行数: train **18,000 SFEN** (existing 7,000 + natural 4,400 + synthetic 4,400 + opening 2,200), val **2,000 SFEN** (existing)。

```python
from datasets import load_dataset
ds = load_dataset("ultemica/piyoshogi", "ocr_paired", split="train")
row = ds[0]
for img, dev in zip(row["images"], row["devices"]):
    print(dev, img.size)
```

## `detector_paired`

フルスクリーンショット + 盤面 bbox。ぴよ将棋アプリのUI上で盤面領域を検出するモデル用。

| Column | Type | 説明 |
| --- | --- | --- |
| `sfen` | `string` | SFEN形式の局面文字列 |
| `hash` | `string` | SFEN の SHA-256 |
| `images` | `Sequence(Image)` | 各機種のフルスクリーンショット |
| `devices` | `Sequence(string)` | 機種識別子 |
| `bboxes` | `Sequence(Sequence(int32))` | 各機種の盤面領域 `[x1, y1, x2, y2]` (retina px) |

行数: train **800 SFEN**, val **200 SFEN**。全機種で同一 SFEN セットを共有。

## 評価データ

train/val と leak なしの独立 test 1,000 SFEN は
[ultemica/piyoshogi-eval](https://huggingface.co/datasets/ultemica/piyoshogi-eval)
に paired 形式で分離管理。

## 局面ソース

- `existing`: ぴよ将棋アプリのプリセット局面
- `natural`: [やねうら王 BalancedPositions2025](https://github.com/yaneurao/YaneuraOu/releases/tag/BalancedPositions2025) からランダム抽出
- `synthetic`: 各駒種を Uniform[0, その駒の最大枚数] でサンプリングして合成
- `opening`: 定跡局面

## キャプチャ方法

[IPA-Patch/PiyoShot](https://github.com/IPA-Patch/PiyoShot) を用いて、指定SFENを
ぴよ将棋アプリに描画し、画面全体を自動でスクリーンショット。

## ライセンス

MIT
""")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="ultemica/piyoshogi")
    p.add_argument("--staging-dir", type=Path, default=STAGING)
    p.add_argument("--configs", nargs="+", default=["ocr", "detector"],
                   choices=["ocr", "detector"])
    p.add_argument("--max-shard-bytes", type=int, default=800_000_000)
    p.add_argument("--flush-batch", type=int, default=32)
    p.add_argument("--limit-rows", type=int, default=None,
                   help="For quick testing — limit each split to N rows")
    p.add_argument("--save-only", action="store_true", help="Skip push")
    p.add_argument("--push-only", action="store_true", help="Skip build, push staging as-is")
    args = p.parse_args()

    staging = args.staging_dir
    staging.mkdir(parents=True, exist_ok=True)

    if not args.push_only:
        bboxes = _load_bboxes()
        if "ocr" in args.configs:
            for split in ["train", "val"]:
                out_dir = staging / "ocr_paired"
                print(f"[build] ocr_paired/{split} …")
                n = _build_ocr(split, out_dir, args.max_shard_bytes,
                               args.flush_batch, args.limit_rows)
                shards = sorted(out_dir.glob(f"{split}-*.parquet"))
                total_size = sum(p.stat().st_size for p in shards) / 1024**3
                print(f"  {n} rows, {len(shards)} shard(s), {total_size:.2f} GB")
        if "detector" in args.configs:
            for split in ["train", "val"]:
                out_dir = staging / "detector_paired"
                print(f"[build] detector_paired/{split} …")
                n = _build_detector(split, out_dir, bboxes, args.max_shard_bytes,
                                     args.flush_batch, args.limit_rows)
                shards = sorted(out_dir.glob(f"{split}-*.parquet"))
                total_size = sum(p.stat().st_size for p in shards) / 1024**3
                print(f"  {n} rows, {len(shards)} shard(s), {total_size:.2f} GB")

        print("[readme] writing …")
        _write_readme(staging / "README.md")

    if args.save_only:
        print("[save-only] skipping push.")
        return

    print(f"[upload] pushing {staging} → {args.repo_id}")
    api = HfApi()
    api.upload_folder(
        folder_path=str(staging),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add paired configs (ocr_paired, detector_paired) and updated README",
    )
    print("[upload] done.")


if __name__ == "__main__":
    main()
