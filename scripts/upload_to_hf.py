"""Convert local capture dataset to Parquet shards and push to Hugging Face Hub.

Reads:
  data/train.jsonl, data/val.jsonl  (each line: {"hash": ..., "sfen": ...})
  data/captures/d0/{hash}.png

Uploads as a Hugging Face DatasetDict with two splits (train, val), each row being
{"image": <PNG bytes>, "sfen": <str>, "hash": <str>}.

Env vars:
  HF_TOKEN  (required)

Usage:
  HF_HUB_ENABLE_HF_TRANSFER=1 uv run --group upload python scripts/upload_to_hf.py \
      --repo-id ultemica/piyoshogi
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value
from datasets import Image as DsImage


def make_generator(manifest_path: Path, image_root: Path):
    def gen():
        with open(manifest_path) as f:
            for line in f:
                e = json.loads(line)
                yield {
                    "image": str(image_root / f"{e['hash']}.png"),
                    "sfen": e["sfen"],
                    "hash": e["hash"],
                }
    return gen


def _shard_bytes_estimate(n_shards: int, total_estimate_gb: float) -> str:
    return f"~{total_estimate_gb / n_shards:.1f}GB/shard"


def _save_split_to_parquet(ds: Dataset, out_dir: Path, split: str, n_shards: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_shards):
        shard = ds.shard(n_shards, i, contiguous=True)
        shard_path = out_dir / f"{split}-{i:05d}-of-{n_shards:05d}.parquet"
        print(f"[hf] writing {shard_path.name} ({len(shard)} rows)")
        shard.to_parquet(str(shard_path))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="ultemica/piyoshogi")
    p.add_argument("--train-manifest", type=Path, default=Path("data/train.jsonl"))
    p.add_argument("--val-manifest", type=Path, default=Path("data/val.jsonl"))
    p.add_argument("--image-root", type=Path, default=Path("data/captures/d0"))
    p.add_argument("--max-shard-size", default="1GB",
                   help="Per-shard size for parquet (e.g. 500MB, 1GB). Push mode only.")
    p.add_argument("--commit-message", default="Upload piyoshogi board captures as parquet")
    p.add_argument("--save-only", type=Path, default=None,
                   help="Save parquet shards under this dir instead of pushing. "
                        "Useful when the container's upload bandwidth is capped: "
                        "build here, then run `huggingface-cli upload` from the host.")
    p.add_argument("--train-shards", type=int, default=20,
                   help="Number of parquet shards for the train split (save-only mode).")
    p.add_argument("--val-shards", type=int, default=1,
                   help="Number of parquet shards for the val split (save-only mode).")
    args = p.parse_args()

    if not args.save_only and not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set (required for push mode).")

    features = Features({
        "image": DsImage(),
        "sfen": Value("string"),
        "hash": Value("string"),
    })

    print(f"[hf] building train split from {args.train_manifest}")
    train_ds = Dataset.from_generator(
        make_generator(args.train_manifest, args.image_root),
        features=features,
    )
    print(f"[hf] train rows={len(train_ds)}")

    print(f"[hf] building val split from {args.val_manifest}")
    val_ds = Dataset.from_generator(
        make_generator(args.val_manifest, args.image_root),
        features=features,
    )
    print(f"[hf] val rows={len(val_ds)}")

    if args.save_only:
        out = args.save_only
        print(f"[hf] save-only mode: writing parquet shards to {out}")
        _save_split_to_parquet(train_ds, out / "data", "train", args.train_shards)
        _save_split_to_parquet(val_ds, out / "data", "val", args.val_shards)
        print(f"[hf] done. Upload with:")
        print(f"  huggingface-cli upload {args.repo_id} {out}/data --repo-type dataset --commit-message='{args.commit_message}'")
        return

    dd = DatasetDict({"train": train_ds, "val": val_ds})
    print(f"[hf] pushing to {args.repo_id} (shard_size={args.max_shard_size})")
    dd.push_to_hub(
        repo_id=args.repo_id,
        max_shard_size=args.max_shard_size,
        commit_message=args.commit_message,
    )
    print("[hf] done.")


if __name__ == "__main__":
    main()
