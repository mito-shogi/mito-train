"""Evaluate BoardOCR checkpoints against ultemica/piyoshogi-eval (real device screenshots).

For each backbone × device pair, crop the full screenshot with its GT bbox and
feed the crop to BoardOCR. Reports SFEN Exact-Match and its sub-metrics per
(backbone, device). Isolates the domain gap between training-set images (paired
tight crops fed through the training augmentation) and untouched device
screenshots.

Dataset: `ultemica/piyoshogi-eval` (paired config)
  1,000 SFEN × 4 devices = 4,000 images, GT bbox per (sfen, device).
  Devices: iPhone10,1 / iPhone11,8 / iPhone15,4 / iPad14,10.
  Leak-free vs the train/val split in `ultemica/piyoshogi`.

Usage:
    # default: 5 w384 sweep backbones, all 4 devices, all 1000 SFEN
    uv run python scripts/eval/eval_realistic.py

    # subset for smoke tests
    uv run python scripts/eval/eval_realistic.py \\
        --backbones mobilenet_v3_small --devices iPhone10,1 --limit 100

    # on a specific GPU (avoid the one detector training is holding)
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/eval/eval_realistic.py
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

from mito_train.datasets import build_transform
from mito_train.datasets.sfen_utils import parse_sfen
from mito_train.models import BoardOCR
from mito_train.training.train_piece import _init_wandb

load_dotenv(override=True)

REPO_ID = "ultemica/piyoshogi-eval"
CONFIG = "paired"
PARQUET_FILE = "paired/test-00000-of-00001.parquet"
DEFAULT_BACKBONES = [
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b1",
    "convnext_nano",
    "convnext_tiny",
]
DEFAULT_DEVICES = ["iPhone10,1", "iPhone11,8", "iPhone15,4", "iPad14,10"]

HEADER = (
    f"{'backbone':<20} {'device':<12} {'n':>5} {'cell':>7} {'board':>7}"
    f" {'slot':>7} {'hand':>7} {'SFEN':>7} {'mae':>7} {'sec':>6}"
)


def fmt_row(bb: str, dev: str, n: int, m: dict[str, float], secs: float) -> str:
    return (
        f"{bb:<20} {dev:<12} {n:>5} {m['cell'] * 100:>6.2f}% {m['board_full'] * 100:>6.2f}%"
        f" {m['slot'] * 100:>6.2f}% {m['hand_full'] * 100:>6.2f}% {m['sfen'] * 100:>6.2f}%"
        f" {m['hand_mae']:>7.4f} {secs:>5.1f}s"
    )


def new_acc() -> dict:
    return {"cell_ok": 0, "cell_tot": 0, "slot_ok": 0, "slot_tot": 0,
            "b_full": 0, "h_full": 0, "sfen": 0, "n": 0, "mae_sum": 0.0}


def accumulate(bp: torch.Tensor, hp: torch.Tensor, bt: torch.Tensor, ht: torch.Tensor,
               acc: dict) -> None:
    acc["cell_ok"] += (bp == bt).sum().item()
    acc["cell_tot"] += bt.numel()
    acc["slot_ok"] += (hp == ht).sum().item()
    acc["slot_tot"] += ht.numel()
    b_ok = (bp == bt).all(dim=(1, 2))
    h_ok = (hp == ht).all(dim=1)
    acc["b_full"] += b_ok.sum().item()
    acc["h_full"] += h_ok.sum().item()
    acc["sfen"] += (b_ok & h_ok).sum().item()
    acc["n"] += bp.size(0)
    acc["mae_sum"] += (hp.float() - ht.float()).abs().sum().item()


def finalize(acc: dict) -> dict[str, float]:
    return {
        "cell": acc["cell_ok"] / acc["cell_tot"],
        "board_full": acc["b_full"] / acc["n"],
        "slot": acc["slot_ok"] / acc["slot_tot"],
        "hand_full": acc["h_full"] / acc["n"],
        "sfen": acc["sfen"] / acc["n"],
        "hand_mae": acc["mae_sum"] / acc["slot_tot"],
    }


def load_eval_rows(devices: list[str], limit: int | None) -> dict[str, list[dict]]:
    """Download the eval parquet once, flatten into per-device row lists.

    Returns {device: [{sfen, bbox, image_bytes}, ...]}. Each device gets up
    to `limit` rows (the same first `limit` SFEN indexes across devices, so
    per-device comparisons stay apples-to-apples).
    """
    print(f"[load] downloading {REPO_ID}:{PARQUET_FILE} …")
    path = hf_hub_download(REPO_ID, PARQUET_FILE, repo_type="dataset")
    tbl = pq.read_table(path)
    print(f"[load] parquet rows = {tbl.num_rows}")

    n_rows = tbl.num_rows if limit is None else min(limit, tbl.num_rows)
    sfens = tbl.column("sfen").to_pylist()[:n_rows]
    devs_col = tbl.column("devices").to_pylist()[:n_rows]
    imgs_col = tbl.column("images").to_pylist()[:n_rows]
    bboxes_col = tbl.column("bboxes").to_pylist()[:n_rows]

    per_dev: dict[str, list[dict]] = {d: [] for d in devices}
    for sfen, dev_list, img_list, bbox_list in zip(sfens, devs_col, imgs_col, bboxes_col):
        for dev, img, bbox in zip(dev_list, img_list, bbox_list):
            if dev in per_dev:
                per_dev[dev].append({"sfen": sfen, "bbox": bbox, "image": img})
    for d in devices:
        print(f"[load]   {d}: {len(per_dev[d])} rows")
    return per_dev


def preprocess_row(row: dict, tf) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Decode -> crop by GT bbox -> tf(image=crop) -> tensor + parsed targets.

    Returns None if the source bytes are missing or the image cannot be decoded;
    caller skips and increments a skip counter.
    """
    im = row["image"]
    buf = im["bytes"] if isinstance(im, dict) else im
    if not buf:
        return None
    try:
        img = Image.open(io.BytesIO(buf))
        img = img.convert("RGB") if img.mode != "RGB" else img
    except UnidentifiedImageError:
        return None
    arr = np.array(img)
    x1, y1, x2, y2 = row["bbox"]
    crop = arr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = arr
    x = tf(image=crop)["image"]
    parsed = parse_sfen(row["sfen"])
    bt = torch.tensor(parsed.board, dtype=torch.long)
    ht = torch.tensor(parsed.hand, dtype=torch.long)
    return x, bt, ht


def eval_backbone_device(model, tf, device_str: str, rows: list[dict],
                         batch_size: int) -> dict:
    acc = new_acc()
    xs, bts, hts = [], [], []
    skipped = 0

    def flush() -> None:
        nonlocal xs, bts, hts
        if not xs:
            return
        xb = torch.stack(xs).to(device_str)
        bl, hl = model(xb)
        accumulate(
            bl.argmax(dim=1).cpu(), hl.argmax(dim=-1).cpu(),
            torch.stack(bts), torch.stack(hts), acc,
        )
        xs, bts, hts = [], [], []

    with torch.no_grad():
        for row in rows:
            got = preprocess_row(row, tf)
            if got is None:
                skipped += 1
                continue
            x, bt, ht = got
            xs.append(x); bts.append(bt); hts.append(ht)
            if len(xs) == batch_size:
                flush()
        flush()
    return finalize(acc) | {"n": acc["n"], "skipped": skipped}


def load_ocr_ckpt(ckpt_path: Path, device_str: str) -> tuple[torch.nn.Module, int, str]:
    ck = torch.load(ckpt_path, map_location=device_str, weights_only=False)
    bb = ck["backbone"]
    S = ck["image_size"]
    model = BoardOCR(backbone=bb, pretrained=False).to(device_str).eval()
    model.load_state_dict(ck["model"])
    return model, S, bb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES,
                   help="Backbone names; ckpts loaded from <ckpt-root>/board-ocr-<bb>/<ckpt-name>.")
    p.add_argument("--devices", nargs="+", default=DEFAULT_DEVICES)
    p.add_argument("--ckpt-root", type=Path, default=Path("runs"))
    p.add_argument("--ckpt-name", default="latest.pt",
                   help="File under runs/board-ocr-<bb>/ (default latest.pt).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap SFEN rows (applied before per-device flatten).")
    p.add_argument("--out", type=Path, default=None,
                   help="JSON output path; parent dir auto-created.")
    p.add_argument("--wandb", action="store_true",
                   help="Log per-(backbone, device) metrics to W&B.")
    p.add_argument("--wandb-project", default="mito-train-board-ocr-w384-v0.3.1-eval-realistic")
    p.add_argument("--wandb-run-name", default=None,
                   help="Defaults to 'realistic-YYYYMMDDTHHMMSS'.")
    args = p.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device_str} backbones={args.backbones} devices={args.devices}")

    wandb_run = None
    if args.wandb:
        run_name = args.wandb_run_name or time.strftime("realistic-%Y%m%dT%H%M%S")
        wandb_run = _init_wandb(
            project=args.wandb_project,
            run_name=run_name,
            config={
                "dataset": REPO_ID,
                "config": CONFIG,
                "backbones": list(args.backbones),
                "devices": list(args.devices),
                "batch_size": args.batch_size,
                "limit": args.limit,
                "ckpt_name": args.ckpt_name,
            },
        )

    per_dev = load_eval_rows(args.devices, args.limit)

    results: dict[str, dict[str, dict]] = {}
    print("\n" + HEADER)
    for bb in args.backbones:
        ckpt_path = args.ckpt_root / f"board-ocr-{bb}" / args.ckpt_name
        if not ckpt_path.exists():
            print(f"[{bb}] [skip] ckpt not found: {ckpt_path}")
            continue
        model, S, bb_actual = load_ocr_ckpt(ckpt_path, device_str)
        if bb_actual != bb:
            print(f"[{bb}] [warn] ckpt reports backbone={bb_actual}, using dir name {bb}")
        tf = build_transform("val", image_size=S)
        results[bb] = {"_ckpt": str(ckpt_path), "_image_size": S}
        for dev in args.devices:
            if not per_dev[dev]:
                continue
            t0 = time.time()
            m = eval_backbone_device(model, tf, device_str, per_dev[dev], args.batch_size)
            results[bb][dev] = m
            print(fmt_row(bb, dev, m["n"], m, time.time() - t0))
            if wandb_run is not None:
                dev_slug = dev.replace(",", "_")
                wandb_run.log({
                    f"{bb}/{dev_slug}/cell": m["cell"],
                    f"{bb}/{dev_slug}/board_full": m["board_full"],
                    f"{bb}/{dev_slug}/slot": m["slot"],
                    f"{bb}/{dev_slug}/hand_full": m["hand_full"],
                    f"{bb}/{dev_slug}/sfen": m["sfen"],
                    f"{bb}/{dev_slug}/hand_mae": m["hand_mae"],
                    f"{bb}/{dev_slug}/n": m["n"],
                    f"{bb}/{dev_slug}/skipped": m["skipped"],
                })
        del model
        if device_str == "cuda":
            torch.cuda.empty_cache()

    print("\n=== per-backbone average across devices (unweighted mean of sfen) ===")
    summary: dict[str, float] = {}
    for bb, dev_map in results.items():
        sfens = [v["sfen"] for k, v in dev_map.items() if not k.startswith("_")]
        if sfens:
            avg = sum(sfens) / len(sfens)
            summary[bb] = avg
            print(f"  {bb:<20} mean sfen = {avg * 100:6.2f}%")

    if wandb_run is not None:
        wandb_run.log({"summary/mean_sfen": summary})
        for bb, avg in summary.items():
            wandb_run.summary[f"{bb}/mean_sfen"] = avg
        wandb_run.finish()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
