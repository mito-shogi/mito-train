"""End-to-end eval: raw screenshot -> BoardDetector -> crop -> BoardOCR -> SFEN.

The whole point: no GT bbox at inference time. Detector predicts the bbox, the
crop is fed to v3, and we measure Exact-Match on the resulting SFEN against
data/test/test.jsonl.

Usage:
    python scripts/eval/eval_end_to_end.py \
        --detector-ckpt runs/board-detector-v1/latest.pt \
        --ocr-ckpt runs/board-ocr-v3/latest.pt \
        --devices iPhone10,1 iPhone11,8 iPhone15,2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from mito_train.datasets.sfen_utils import encode_sfen, parse_sfen
from mito_train.models import BoardDetector, BoardOCR

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def detector_bbox_to_original(
    pred: np.ndarray,          # (4,) normalized (x1, y1, x2, y2) in letterboxed frame
    letter_meta: tuple[int, int, int, int],
    orig_hw: tuple[int, int],
    letter_size: int,
) -> tuple[int, int, int, int]:
    top, left, nh, nw = letter_meta
    x1n, y1n, x2n, y2n = pred
    x1p, y1p, x2p, y2p = (
        x1n * letter_size, y1n * letter_size,
        x2n * letter_size, y2n * letter_size,
    )
    x1p -= left; x2p -= left
    y1p -= top; y2p -= top
    oh, ow = orig_hw
    sx, sy = ow / nw, oh / nh
    x1 = int(round(x1p * sx)); x2 = int(round(x2p * sx))
    y1 = int(round(y1p * sy)); y2 = int(round(y2p * sy))
    return max(0, x1), max(0, y1), min(ow, x2), min(oh, y2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", type=Path, default=Path("data/test"))
    p.add_argument("--detector-ckpt", type=Path, default=None)
    p.add_argument("--ocr-ckpt", type=Path, default=Path("runs/board-ocr-v3/latest.pt"))
    p.add_argument("--devices", nargs="+", default=["iPhone10,1"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--use-gt-bbox", action="store_true",
                   help="Skip the detector and crop with device_bboxes.json. "
                        "OCR-only upper-bound; not an end-to-end number.")
    args = p.parse_args()

    if not args.use_gt_bbox and args.detector_ckpt is None:
        raise SystemExit("--detector-ckpt is required unless --use-gt-bbox is set.")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    det = det_S = None
    if not args.use_gt_bbox:
        det_ck = torch.load(args.detector_ckpt, map_location=device, weights_only=False)
        det = BoardDetector(backbone=det_ck["backbone"], pretrained=False).to(device).eval()
        det.load_state_dict(det_ck["model"])
        det_S = det_ck["image_size"]

    ocr_ck = torch.load(args.ocr_ckpt, map_location=device, weights_only=False)
    ocr = BoardOCR(backbone=ocr_ck["backbone"], pretrained=False).to(device).eval()
    ocr.load_state_dict(ocr_ck["model"])
    ocr_S = ocr_ck["image_size"]

    if args.use_gt_bbox:
        print("[e2e] mode=OCR-only (GT bbox from device_bboxes.json)")
    else:
        print(f"[e2e] detector={args.detector_ckpt} (S={det_S})")
    print(f"[e2e] ocr={args.ocr_ckpt} (S={ocr_S})")

    gt = {json.loads(l)["hash"]: json.loads(l)["sfen"]
          for l in open(args.eval_root / "position.jsonl")}

    with open(args.eval_root / "device_bboxes.json") as f:
        bbox_cfg = json.load(f)  # only consulted when --use-gt-bbox is set

    per_device: dict[str, dict[str, float | int]] = {}
    for dev in args.devices:
        img_dir = args.eval_root / dev
        paths = sorted(img_dir.glob("*.png"))
        print(f"[e2e] === {dev} ({len(paths)}) ===")
        gt_bbox = None
        if args.use_gt_bbox:
            b = bbox_cfg["devices"][dev]["board_view_px"]
            gt_bbox = (int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]))
        hits = board_perfect = hand_perfect = 0
        cell_correct = cell_total = 0
        hand_correct = hand_total = 0
        t0 = time.time()
        for i in range(0, len(paths), args.batch_size):
            batch = paths[i:i + args.batch_size]
            imgs_orig, imgs_det, metas, hws = [], [], [], []
            for p_ in batch:
                bgr = cv2.imread(str(p_))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                imgs_orig.append(rgb)
                hws.append(rgb.shape[:2])
                if not args.use_gt_bbox:
                    lb, meta = letterbox(rgb, det_S)
                    imgs_det.append(normalize(lb))
                    metas.append(meta)
            if args.use_gt_bbox:
                bbox = None
            else:
                xs = torch.stack(imgs_det).to(device)
                with torch.no_grad():
                    bbox = det(xs).cpu().numpy()

            crops_batch = []
            for j, p_ in enumerate(batch):
                if args.use_gt_bbox:
                    x1, y1, x2, y2 = gt_bbox
                else:
                    x1, y1, x2, y2 = detector_bbox_to_original(bbox[j], metas[j], hws[j], det_S)
                crop = imgs_orig[j][y1:y2, x1:x2]
                if crop.size == 0:
                    crop = imgs_orig[j]
                lb, _ = letterbox(crop, ocr_S)
                crops_batch.append(normalize(lb))
            xs2 = torch.stack(crops_batch).to(device)
            with torch.no_grad():
                bl, hl = ocr(xs2)
            bp_all = bl.argmax(dim=1).cpu().tolist()
            hp_all = hl.argmax(dim=-1).cpu().tolist()
            for j, p_ in enumerate(batch):
                gtp = parse_sfen(gt[p_.stem])
                bp, hp = bp_all[j], hp_all[j]
                pred = encode_sfen(bp, hp, gtp.turn)
                gts = encode_sfen(gtp.board, gtp.hand, gtp.turn)
                hits += (pred == gts)
                b_ok = True
                for r in range(9):
                    for c in range(9):
                        cell_total += 1
                        if bp[r][c] == gtp.board[r][c]:
                            cell_correct += 1
                        else:
                            b_ok = False
                board_perfect += b_ok
                h_ok = all(hp[s] == gtp.hand[s] for s in range(14))
                hand_perfect += h_ok
                for s in range(14):
                    hand_total += 1
                    if hp[s] == gtp.hand[s]:
                        hand_correct += 1
        dt = time.time() - t0
        n = len(paths)
        per_device[dev] = {
            "n": n, "exact_match": hits / n, "board_perfect": board_perfect / n,
            "cell_acc": cell_correct / cell_total, "hand_perfect": hand_perfect / n,
            "slot_acc": hand_correct / hand_total, "seconds": dt,
        }
        print(
            f"[e2e] {dev}: EM={hits}/{n} ({hits/n*100:.2f}%)  "
            f"board_full={board_perfect/n*100:.2f}%  "
            f"cell={cell_correct/cell_total*100:.2f}%  "
            f"hand_full={hand_perfect/n*100:.2f}%  "
            f"slot={hand_correct/hand_total*100:.2f}%  "
            f"({dt:.1f}s)"
        )
    print("\n[e2e] summary:", json.dumps(per_device, indent=2))


if __name__ == "__main__":
    main()
