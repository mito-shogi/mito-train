"""Analyze hand-piece OCR failures on eval devices.

Motivation: on iPhone XR / iPhone 15, hand_full accuracy drops to ~91-92% while
board_full stays at ~99%. This script figures out *which* hand slots are the
culprits and *how* they fail (over/under-count, which piece type).

Runs OCR with GT bbox (device_bboxes.json), collects every hand mismatch, and
prints:
  - per-slot mismatch rate
  - top confusion pairs (piece, gt_count -> pred_count)
  - a sample of hashes for visual follow-up (optionally saves crops)

Usage:
    python scripts/analyze_hand_failures.py \
        --devices iPhone11,8 iPhone15,4 \
        --dump-samples 12 --dump-dir runs/hand-fail-samples
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from mito_train.datasets.sfen_utils import parse_sfen
from mito_train.models import BoardOCR

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

HAND_PIECES = ["P", "L", "N", "S", "G", "B", "R"]  # slot 0..6 sente, 7..13 gote


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    ph, pw = size - nh, size - nw
    top, bottom = ph // 2, ph - ph // 2
    left, right = pw // 2, pw - pw // 2
    r = cv2.copyMakeBorder(r, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return r, (top, left, nh, nw)


def normalize(img: np.ndarray) -> torch.Tensor:
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def slot_label(slot: int) -> str:
    side = "sente" if slot < 7 else "gote"
    piece = HAND_PIECES[slot % 7]
    return f"{side}:{piece}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", type=Path, default=Path("data/test"))
    p.add_argument("--ocr-ckpt", type=Path, default=Path("runs/board-ocr-v3/latest.pt"))
    p.add_argument("--devices", nargs="+", default=["iPhone11,8", "iPhone15,4"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dump-samples", type=int, default=0,
                   help="Save this many mismatched crops per device for visual inspection.")
    p.add_argument("--dump-dir", type=Path, default=Path("runs/hand-fail-samples"))
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ocr_ck = torch.load(args.ocr_ckpt, map_location=device, weights_only=False)
    ocr = BoardOCR(backbone=ocr_ck["backbone"], pretrained=False).to(device).eval()
    ocr.load_state_dict(ocr_ck["model"])
    S = ocr_ck["image_size"]

    gt = {json.loads(l)["hash"]: json.loads(l)["sfen"]
          for l in open(args.eval_root / "position.jsonl")}
    with open(args.eval_root / "device_bboxes.json") as f:
        bbox_cfg = json.load(f)

    for dev in args.devices:
        b = bbox_cfg["devices"][dev]["board_view_px"]
        gt_bbox = (int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]))
        img_dir = args.eval_root / dev
        paths = sorted(img_dir.glob("*.png"))
        print(f"\n[hand-fail] === {dev} ({len(paths)}) bbox={gt_bbox} ===")

        # Aggregators
        slot_mismatch = Counter()             # slot -> #images where that slot is wrong
        conf: dict[int, Counter] = defaultdict(Counter)  # slot -> Counter((gt, pred))
        n_hand_wrong = 0
        sample_hashes: list[str] = []         # up to dump-samples per device

        for i in range(0, len(paths), args.batch_size):
            batch = paths[i:i + args.batch_size]
            crops_orig = []
            crops_tensor = []
            for p_ in batch:
                bgr = cv2.imread(str(p_))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                crop = rgb[gt_bbox[1]:gt_bbox[3], gt_bbox[0]:gt_bbox[2]]
                crops_orig.append(crop)
                lb, _ = letterbox(crop, S)
                crops_tensor.append(normalize(lb))
            xs = torch.stack(crops_tensor).to(device)
            with torch.no_grad():
                _, hl = ocr(xs)
            hp_all = hl.argmax(dim=-1).cpu().tolist()

            for j, p_ in enumerate(batch):
                gtp = parse_sfen(gt[p_.stem])
                pred_hand = hp_all[j]
                bad_slots = [s for s in range(14) if pred_hand[s] != gtp.hand[s]]
                if not bad_slots:
                    continue
                n_hand_wrong += 1
                for s in bad_slots:
                    slot_mismatch[s] += 1
                    conf[s][(gtp.hand[s], pred_hand[s])] += 1
                if len(sample_hashes) < args.dump_samples:
                    sample_hashes.append(p_.stem)
                    if args.dump_samples > 0:
                        out_dir = args.dump_dir / dev.replace(",", "_")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        # save the cropped board+stands area with a summary sidecar
                        cv2.imwrite(
                            str(out_dir / f"{p_.stem}.png"),
                            cv2.cvtColor(crops_orig[j], cv2.COLOR_RGB2BGR),
                        )
                        with open(out_dir / f"{p_.stem}.json", "w") as f:
                            json.dump({
                                "hash": p_.stem,
                                "sfen": gt[p_.stem],
                                "gt_hand": gtp.hand,
                                "pred_hand": pred_hand,
                                "bad_slots": [
                                    {
                                        "slot": s,
                                        "label": slot_label(s),
                                        "gt": gtp.hand[s],
                                        "pred": pred_hand[s],
                                    } for s in bad_slots
                                ],
                            }, f, indent=2)

        print(f"[hand-fail] {dev}: {n_hand_wrong}/{len(paths)} images have "
              f"at least one hand-slot mismatch "
              f"({n_hand_wrong/len(paths)*100:.2f}%)")

        # Per-slot ranking
        print(f"[hand-fail] top mismatched slots (slot: mismatched images):")
        for s, c in slot_mismatch.most_common():
            print(f"    {slot_label(s):>10}  slot={s:2d}   {c:>5} imgs "
                  f"({c/len(paths)*100:.2f}%)")

        # Top confusions per slot (only for the top-3 problem slots)
        print(f"[hand-fail] top confusions on worst 3 slots (GT -> Pred, count):")
        for s, _ in slot_mismatch.most_common(3):
            print(f"    {slot_label(s)} (slot {s}):")
            for (gt_c, pr_c), cnt in conf[s].most_common(6):
                print(f"        {gt_c:>2} -> {pr_c:<2}   {cnt}")

        if sample_hashes:
            print(f"[hand-fail] dumped {len(sample_hashes)} sample crops to "
                  f"{args.dump_dir / dev.replace(',', '_')}")


if __name__ == "__main__":
    main()
