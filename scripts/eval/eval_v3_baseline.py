"""Evaluate v3 board-OCR per device on two settings.

(1) board-only val:
    data/ocr/val.jsonl (2000 samples) × each device's data/ocr/<device>/*.webp
    → same distribution the model was trained on (tight-cropped board+stands).
    4 devices: iPhone10,1 / iPhone11,8 / iPhone15,4 / iPad14,10

(2) screenshot eval:
    For each device, crop with the GT bbox from device_bboxes.json then feed
    the crop to OCR. Isolates the domain gap between Frida-hooked tight crops
    and real device screenshots.

    - iPhone10,1 / iPhone11,8 / iPhone15,4: uploads/eval/data/*.parquet
      (10000 raw screenshots per device from the piyoshogi-eval dataset).
    - iPad14,10: parquet not published yet, so fall back to
      data/detector/iPad14,10/*.webp (1000 raw screenshots that were
      collected for board-detector training but share the same SFEN pool).

Usage:
    uv run python scripts/eval/eval_v3_baseline.py --screenshot-limit 2000
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
from PIL import Image
from torch.utils.data import DataLoader

from mito_train.datasets import CaptureDataset, build_transform
from mito_train.datasets.sfen_utils import parse_sfen
from mito_train.models import BoardOCR

CKPT = Path("runs/board-ocr-v3/latest.pt")
VAL_MANIFEST = Path("data/ocr/val.jsonl")
OCR_ROOT = Path("data/ocr")
EVAL_PARQUET_DIR = Path("uploads/eval/data")
DETECTOR_ROOT = Path("data/detector")
DETECTOR_MANIFESTS = [
    Path("data/detector/train.jsonl"),
    Path("data/detector/val.jsonl"),
]
BBOX_JSON = Path("data/test/device_bboxes.json")
POSITION_JSONL = Path("data/test/test.jsonl")

DEV_LIST_VAL = ["iPhone10,1", "iPhone11,8", "iPhone15,4", "iPad14,10"]
# key: device id used in ocr_train/detector_train (comma form),
# value: parquet split prefix (underscore form) — None means "no parquet, fall
# back to detector_train screenshots".
DEV_LIST_EVAL: list[tuple[str, str | None]] = [
    ("iPhone10,1", "iPhone10_1"),
    ("iPhone11,8", "iPhone11_8"),
    ("iPhone15,4", "iPhone15_4"),
    ("iPad14,10", None),
]

HEADER = (
    f"{'device':<14} {'n':>6} {'cell':>8} {'board_full':>12}"
    f" {'slot':>8} {'hand_full':>11} {'SFEN':>8} {'sec':>7}"
)


def fmt_row(dev: str, n: int, m: dict[str, float], secs: float) -> str:
    return (
        f"{dev:<14} {n:>6} {m['cell'] * 100:>7.2f}% {m['board_full'] * 100:>11.2f}%"
        f" {m['slot'] * 100:>7.2f}% {m['hand_full'] * 100:>10.2f}%"
        f" {m['sfen'] * 100:>7.2f}% {secs:>6.1f}s"
    )


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


def finalize(acc: dict) -> dict[str, float]:
    return {
        "cell": acc["cell_ok"] / acc["cell_tot"],
        "board_full": acc["b_full"] / acc["n"],
        "slot": acc["slot_ok"] / acc["slot_tot"],
        "hand_full": acc["h_full"] / acc["n"],
        "sfen": acc["sfen"] / acc["n"],
    }


def eval_board_val(model, tf, device, batch_size: int, num_workers: int) -> dict:
    print("\n=== (1) board-only val (val.jsonl × ocr_train/<device>/*.webp) ===")
    print(HEADER)
    all_rows: dict[str, dict] = {}
    for dev in DEV_LIST_VAL:
        img_root = OCR_ROOT / dev
        if not img_root.exists():
            print(f"{dev:<14} [skip] {img_root} not found")
            continue
        ds = CaptureDataset(VAL_MANIFEST, img_root, transform=tf)
        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False,
        )
        acc = {"cell_ok": 0, "cell_tot": 0, "slot_ok": 0, "slot_tot": 0,
               "b_full": 0, "h_full": 0, "sfen": 0, "n": 0}
        t0 = time.time()
        for x, bt, ht in dl:
            x = x.to(device)
            with torch.no_grad():
                bl, hl = model(x)
            accumulate(bl.argmax(dim=1).cpu(), hl.argmax(dim=-1).cpu(), bt, ht, acc)
        m = finalize(acc)
        all_rows[dev] = {"n": acc["n"], **m}
        print(fmt_row(dev, acc["n"], m, time.time() - t0))
    return all_rows


def _iter_parquet_batches(shard_paths: list[Path], batch_size: int, limit: int | None):
    yielded = 0
    for shard in shard_paths:
        t = pq.read_table(shard)
        imgs = t.column("image").to_pylist()
        sfens = t.column("sfen").to_pylist()
        bboxes = t.column("bbox").to_pylist()
        n_rows = t.num_rows
        for i in range(0, n_rows, batch_size):
            end = min(i + batch_size, n_rows)
            if limit is not None and yielded >= limit:
                return
            if limit is not None:
                end = min(end, i + (limit - yielded))
            yield imgs[i:end], sfens[i:end], bboxes[i:end]
            yielded += end - i
            if limit is not None and yielded >= limit:
                return


def _load_and_crop(pngs: list[dict], bboxes: list[list[int]], tf, sfens: list[str]):
    xs, bts, hts = [], [], []
    for im, bbox, sfen in zip(pngs, bboxes, sfens):
        img = Image.open(io.BytesIO(im["bytes"]))
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert("RGB")
        arr = np.array(img)
        x1, y1, x2, y2 = bbox
        crop = arr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = arr
        xs.append(tf(image=crop)["image"])
        parsed = parse_sfen(sfen)
        bts.append(torch.tensor(parsed.board, dtype=torch.long))
        hts.append(torch.tensor(parsed.hand, dtype=torch.long))
    return torch.stack(xs), torch.stack(bts), torch.stack(hts)


def _load_detector_screenshots(dev_comma: str) -> tuple[list[Path], dict[str, str]]:
    """Return (webp paths, sfen-by-hash) for iPad screenshots that live under
    data/detector/<device>/. detector_train.jsonl + detector_val.jsonl
    together enumerate the split; SFENs come from data/test/test.jsonl."""
    hashes: set[str] = set()
    for manifest in DETECTOR_MANIFESTS:
        if not manifest.exists():
            continue
        with open(manifest) as f:
            for line in f:
                e = json.loads(line)
                if e.get("device") == dev_comma:
                    hashes.add(e["hash"])
    sfen_by_hash: dict[str, str] = {}
    with open(POSITION_JSONL) as f:
        for line in f:
            e = json.loads(line)
            if e["hash"] in hashes:
                sfen_by_hash[e["hash"]] = e["sfen"]
    img_dir = DETECTOR_ROOT / dev_comma
    paths = [img_dir / f"{h}.webp" for h in sorted(sfen_by_hash)]
    paths = [p for p in paths if p.exists()]
    return paths, sfen_by_hash


def _iter_detector_batches(paths: list[Path], sfen_by_hash: dict[str, str],
                           bbox: list[int], batch_size: int, limit: int | None):
    yielded = 0
    for i in range(0, len(paths), batch_size):
        end = min(i + batch_size, len(paths))
        if limit is not None:
            if yielded >= limit:
                return
            end = min(end, i + (limit - yielded))
        pngs = [{"bytes": p.read_bytes()} for p in paths[i:end]]
        sfens = [sfen_by_hash[p.stem] for p in paths[i:end]]
        bboxes = [bbox] * (end - i)
        yield pngs, sfens, bboxes
        yielded += end - i
        if limit is not None and yielded >= limit:
            return


def eval_screenshot(model, tf, device, batch_size: int, limit: int | None) -> dict:
    print("\n=== (2) screenshot eval (GT bbox → crop → OCR) ===")
    if limit is not None:
        print(f"    limit per device: {limit}")
    print(HEADER)
    with open(BBOX_JSON) as f:
        bbox_cfg = json.load(f)
    all_rows: dict[str, dict] = {}
    for dev_comma, parquet_prefix in DEV_LIST_EVAL:
        acc = {"cell_ok": 0, "cell_tot": 0, "slot_ok": 0, "slot_tot": 0,
               "b_full": 0, "h_full": 0, "sfen": 0, "n": 0}
        t0 = time.time()
        source_note: str
        if parquet_prefix is not None:
            shards = sorted(EVAL_PARQUET_DIR.glob(f"{parquet_prefix}-*-of-00010.parquet"))
            if not shards:
                shards = sorted(EVAL_PARQUET_DIR.glob(f"{parquet_prefix}-*.parquet"))
            batches = _iter_parquet_batches(shards, batch_size, limit)
            source_note = f"parquet (n≤{len(shards) * 1000})"
        else:
            paths, sfen_by_hash = _load_detector_screenshots(dev_comma)
            b = bbox_cfg["devices"][dev_comma]["board_view_px"]
            bbox = [int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])]
            batches = _iter_detector_batches(paths, sfen_by_hash, bbox, batch_size, limit)
            source_note = f"detector_train (n={len(paths)})"
        for pngs, sfens, bboxes in batches:
            x, bt, ht = _load_and_crop(pngs, bboxes, tf, sfens)
            x = x.to(device)
            with torch.no_grad():
                bl, hl = model(x)
            accumulate(bl.argmax(dim=1).cpu(), hl.argmax(dim=-1).cpu(), bt, ht, acc)
        if acc["n"] == 0:
            print(f"{dev_comma:<14} [skip] no source ({source_note})")
            continue
        m = finalize(acc)
        all_rows[dev_comma] = {"n": acc["n"], "source": source_note, **m}
        print(fmt_row(dev_comma, acc["n"], m, time.time() - t0))
    return all_rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--screenshot-limit", type=int, default=None,
                   help="Per-device sample cap for the screenshot eval. None = all 10k.")
    p.add_argument("--skip-val", action="store_true")
    p.add_argument("--skip-screenshot", action="store_true")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional JSON output path for both result tables.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    S = ck["image_size"]
    print(f"[v3] ckpt={CKPT} device={device} backbone={ck['backbone']}"
          f" S={S} epoch={ck['epoch']}")
    model = BoardOCR(backbone=ck["backbone"], pretrained=False).to(device).eval()
    model.load_state_dict(ck["model"])
    tf = build_transform("val", image_size=S)

    results: dict[str, dict] = {}
    if not args.skip_val:
        results["board_val"] = eval_board_val(
            model, tf, device, args.batch_size, args.num_workers,
        )
    if not args.skip_screenshot:
        results["screenshot"] = eval_screenshot(
            model, tf, device, args.batch_size, args.screenshot_limit,
        )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nsaved → {args.out}")


if __name__ == "__main__":
    main()
