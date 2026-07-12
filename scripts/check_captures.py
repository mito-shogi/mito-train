"""Sanity-check raw device screenshots produced by piyo-hook.

Runs cheap file-level checks against `<root>/<device>/*.{png,webp,jpg}`:
    - file size >= --min-size (default 5 KiB)
    - PIL can decode the image
    - decoded dimensions match device_bboxes.json's `screen_px`

By default also runs a board-region content check to catch cases where the
screenshot got taken during an AirDrop / share-sheet / system modal that
covers the board — those files pass the file-level checks but the training
data is silently wrong. The heuristic: crop the `board_view_px` bbox and
require its grayscale std-dev to sit in [--min-std, --max-std]. Real shogi
boards land around 40–90; a white overlay or blank screen sits under 15.
Pass --no-content-check to skip this pass.

Reports per-device counts and prints a few examples per failure category.
Exits 1 if anything looked broken.

Usage (defaults to data/captures/ once piyo-hook has produced it):
    uv run python scripts/check_captures.py
    uv run python scripts/check_captures.py --root data/detector    # audit existing set
    uv run python scripts/check_captures.py --no-content-check      # fast, file-level only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

BBOX_JSON_DEFAULT = Path("data/test/device_bboxes.json")
ROOT_DEFAULT = Path("data/captures")
IMAGE_EXTS = {".png", ".webp", ".jpg", ".jpeg"}


def check_one(path: Path, expected_wh: tuple[int, int], min_size: int,
              bbox: tuple[int, int, int, int] | None,
              min_std: float, max_std: float
              ) -> tuple[str, str | None]:
    """Return (category, detail). category is 'ok' when fine."""
    try:
        size = path.stat().st_size
    except OSError as e:
        return "unreadable", f"stat: {e}"
    if size < min_size:
        return "too_small", f"{size}B"
    try:
        with Image.open(path) as img:
            img.load()
            wh = img.size  # (w, h)
            if bbox is not None:
                crop_arr = np.asarray(img.convert("RGB").crop(bbox), dtype=np.uint8)
            else:
                crop_arr = None
    except (UnidentifiedImageError, OSError) as e:
        return "unreadable", str(e)[:60]
    if wh != expected_wh:
        return "wrong_dims", f"{wh[0]}x{wh[1]}"
    if crop_arr is not None:
        if crop_arr.size == 0:
            return "bad_content", "bbox empty"
        gray = crop_arr.mean(axis=-1)
        std = float(gray.std())
        if std < min_std:
            return "bad_content", f"std={std:.1f} (below {min_std})"
        if std > max_std:
            return "bad_content", f"std={std:.1f} (above {max_std})"
    return "ok", None


def check_device(image_dir: Path, expected_wh: tuple[int, int], min_size: int,
                 bbox: tuple[int, int, int, int] | None,
                 min_std: float, max_std: float) -> dict:
    counts = {"ok": 0, "too_small": 0, "unreadable": 0,
              "wrong_dims": 0, "bad_content": 0}
    bad: dict[str, list[tuple[str, str]]] = {
        "too_small": [], "unreadable": [], "wrong_dims": [], "bad_content": [],
    }
    for p in sorted(image_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        cat, detail = check_one(p, expected_wh, min_size, bbox, min_std, max_std)
        counts[cat] += 1
        if cat != "ok":
            bad[cat].append((p.name, detail or ""))
    return {"counts": counts, "bad": bad}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT_DEFAULT,
                   help="dir containing <device>/ subdirs of screenshots")
    p.add_argument("--bboxes", type=Path, default=BBOX_JSON_DEFAULT)
    p.add_argument("--min-size", type=int, default=5000,
                   help="bytes; files below this are flagged as too_small")
    p.add_argument("--content-check", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="crop board bbox and check its std-dev range "
                        "(default on; --no-content-check to skip)")
    p.add_argument("--min-std", type=float, default=15.0,
                   help="board bbox grayscale std-dev must be >= this "
                        "(blank / overlay = uniform → low std)")
    p.add_argument("--max-std", type=float, default=110.0,
                   help="board bbox grayscale std-dev must be <= this "
                        "(mid-capture with unrelated UI can spike variance)")
    p.add_argument("--show", type=int, default=3,
                   help="print up to N example bad files per category per device")
    args = p.parse_args()

    with args.bboxes.open() as f:
        cfg = json.load(f)
    devices = list(cfg["devices"].keys())

    total_bad = 0
    total_ok = 0
    for dev in devices:
        cfg_dev = cfg["devices"][dev]
        expected_wh = tuple(cfg_dev["screen_px"])  # [w, h]
        bbox_tuple: tuple[int, int, int, int] | None = None
        if args.content_check:
            b = cfg_dev["board_view_px"]
            bbox_tuple = (int(b["x1"]), int(b["y1"]),
                          int(b["x2"]), int(b["y2"]))
        image_dir = args.root / dev
        if not image_dir.exists():
            print(f"{dev:<14} [skip] {image_dir} not found")
            continue
        r = check_device(image_dir, expected_wh, args.min_size, bbox_tuple,
                         args.min_std, args.max_std)
        c = r["counts"]
        bad_here = (c["too_small"] + c["unreadable"]
                    + c["wrong_dims"] + c["bad_content"])
        total_bad += bad_here
        total_ok += c["ok"]
        status = "OK" if bad_here == 0 else "BAD"
        print(f"{dev:<14} exp={expected_wh[0]}x{expected_wh[1]}  "
              f"ok={c['ok']:>5}  too_small={c['too_small']:>3}  "
              f"unreadable={c['unreadable']:>3}  wrong_dims={c['wrong_dims']:>3}  "
              f"bad_content={c['bad_content']:>3}  {status}")
        for cat, samples in r["bad"].items():
            if not samples:
                continue
            for name, detail in samples[:args.show]:
                print(f"    {cat:<12} {name[:16]}…  {detail}")
            if len(samples) > args.show:
                print(f"    {cat:<12} … +{len(samples) - args.show} more")

    print()
    if total_bad == 0:
        print(f"all captures pass sanity check (checked {total_ok} files).")
        return 0
    print(f"FAIL: {total_bad} broken capture(s); {total_ok} OK.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
