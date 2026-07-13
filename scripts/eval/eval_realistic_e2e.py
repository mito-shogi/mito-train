"""End-to-end realistic eval: raw screenshot -> BoardDetector -> crop -> BoardOCR -> SFEN.

Same dataset (ultemica/piyoshogi-eval, 1,000 SFEN × 4 devices) and same metric
schema as scripts/eval/eval_realistic.py, but the crop bbox is *predicted by the
detector*, not read from the GT `bboxes` column. Removes the crop-convention
mismatch between piyoshogi-eval's `bboxes` and what the OCR was trained on;
the detector was trained on the same crop convention as the OCR training data.

Contract:
- one wandb run per backbone (run name = backbone; section = device)
- CUDA_VISIBLE_DEVICES=<gpu> per invocation to fan out across GPUs
- --wandb requires exactly one backbone per invocation

Usage:
    # smoke (single backbone, single device)
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/eval/eval_realistic_e2e.py \\
        --backbones mobilenet_v3_small --devices "iPad14,10" --limit 100 --wandb

    # full run for one backbone
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/eval/eval_realistic_e2e.py \\
        --backbones convnext_nano --wandb \\
        --out runs/eval/e2e-realistic-w384-20260713-convnext_nano.json
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

load_dotenv("/home/vscode/app/.env", override=True)

import os as _os  # noqa: E402
import json as _json  # noqa: E402
_cf_id = _os.environ.get("CF_ACCESS_CLIENT_ID")
_cf_secret = _os.environ.get("CF_ACCESS_CLIENT_SECRET")
if _cf_id and _cf_secret and not _os.environ.get("WANDB__EXTRA_HTTP_HEADERS"):
    _os.environ["WANDB__EXTRA_HTTP_HEADERS"] = _json.dumps({
        "CF-Access-Client-Id": _cf_id,
        "CF-Access-Client-Secret": _cf_secret,
    })

from mito_train.datasets import build_transform  # noqa: E402
from mito_train.datasets.sfen_utils import parse_sfen  # noqa: E402
from mito_train.models import BoardDetector, BoardOCR  # noqa: E402
from mito_train.training.train_piece import _init_wandb  # noqa: E402

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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

HEADER = (
    f"{'backbone':<20} {'device':<12} {'n':>5} {'cell':>7} {'board':>7}"
    f" {'slot':>7} {'hand':>7} {'SFEN':>7} {'iou':>6} {'sec':>6}"
)


def fmt_row(bb: str, dev: str, n: int, m: dict[str, float], secs: float) -> str:
    return (
        f"{bb:<20} {dev:<12} {n:>5} {m['cell'] * 100:>6.2f}% {m['board_full'] * 100:>6.2f}%"
        f" {m['slot'] * 100:>6.2f}% {m['hand_full'] * 100:>6.2f}% {m['sfen'] * 100:>6.2f}%"
        f" {m['iou_mean']:>6.3f} {secs:>5.1f}s"
    )


def new_acc() -> dict:
    return {"cell_ok": 0, "cell_tot": 0, "slot_ok": 0, "slot_tot": 0,
            "b_full": 0, "h_full": 0, "sfen": 0, "n": 0, "iou_sum": 0.0,
            "mae_sum": 0.0}


def accumulate(bp: torch.Tensor, hp: torch.Tensor, bt: torch.Tensor, ht: torch.Tensor,
               iou_batch: list[float], acc: dict) -> None:
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
    acc["iou_sum"] += float(sum(iou_batch))


def finalize(acc: dict) -> dict[str, float]:
    return {
        "cell": acc["cell_ok"] / acc["cell_tot"],
        "board_full": acc["b_full"] / acc["n"],
        "slot": acc["slot_ok"] / acc["slot_tot"],
        "hand_full": acc["h_full"] / acc["n"],
        "sfen": acc["sfen"] / acc["n"],
        "hand_mae": acc["mae_sum"] / acc["slot_tot"],
        "iou_mean": acc["iou_sum"] / acc["n"],
    }


def load_eval_rows(devices: list[str], limit: int | None) -> dict[str, list[dict]]:
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
        for dev, im, bbox in zip(dev_list, img_list, bbox_list):
            if dev in per_dev:
                per_dev[dev].append({"sfen": sfen, "gt_bbox": bbox, "image": im})
    for d in devices:
        print(f"[load]   {d}: {len(per_dev[d])} rows")
    return per_dev


def _decode_row(row: dict) -> np.ndarray | None:
    im = row["image"]
    buf = im["bytes"] if isinstance(im, dict) else im
    if not buf:
        return None
    try:
        img = Image.open(io.BytesIO(buf))
        img = img.convert("RGB") if img.mode != "RGB" else img
    except UnidentifiedImageError:
        return None
    return np.array(img)


def letterbox(arr: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Aspect-preserving resize to `size` + center pad with 0. Returns (frame, (top, left, nh, nw))."""
    h, w = arr.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    r = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
    ph, pw = size - nh, size - nw
    top, bottom = ph // 2, ph - ph // 2
    left, right = pw // 2, pw - pw // 2
    r = cv2.copyMakeBorder(r, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return r, (top, left, nh, nw)


def normalize_imagenet(arr: np.ndarray) -> torch.Tensor:
    x = (arr.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def detector_bbox_to_original(
    pred: np.ndarray,  # (4,) normalized (x1, y1, x2, y2) in letterboxed frame
    letter_meta: tuple[int, int, int, int],
    orig_hw: tuple[int, int],
    letter_size: int,
) -> tuple[int, int, int, int]:
    top, left, nh, nw = letter_meta
    x1n, y1n, x2n, y2n = pred
    x1p = x1n * letter_size - left
    x2p = x2n * letter_size - left
    y1p = y1n * letter_size - top
    y2p = y2n * letter_size - top
    oh, ow = orig_hw
    sx, sy = ow / nw, oh / nh
    x1 = int(round(x1p * sx)); x2 = int(round(x2p * sx))
    y1 = int(round(y1p * sy)); y2 = int(round(y2p * sy))
    return max(0, x1), max(0, y1), min(ow, x2), min(oh, y2)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def eval_backbone_device(
    ocr_model, ocr_tf, det_model, det_S: int,
    device_str: str, rows: list[dict], batch_size: int,
) -> dict:
    acc = new_acc()
    skipped = 0
    # Per-row bbox prediction happens on decoded arrays; OCR batching happens
    # after crop. Keeping it row-by-row for clarity — 4k rows total is trivial.
    det_batch_imgs: list[torch.Tensor] = []
    det_batch_meta: list[tuple[int, int, int, int]] = []
    det_batch_orig: list[np.ndarray] = []
    det_batch_gt: list[tuple[int, int, int, int]] = []
    det_batch_targets: list[tuple[torch.Tensor, torch.Tensor]] = []
    pending_ocr_x: list[torch.Tensor] = []
    pending_ocr_bt: list[torch.Tensor] = []
    pending_ocr_ht: list[torch.Tensor] = []
    pending_ocr_iou: list[float] = []

    def flush_ocr() -> None:
        nonlocal pending_ocr_x, pending_ocr_bt, pending_ocr_ht, pending_ocr_iou
        if not pending_ocr_x:
            return
        xb = torch.stack(pending_ocr_x).to(device_str)
        with torch.no_grad():
            bl, hl = ocr_model(xb)
        accumulate(
            bl.argmax(dim=1).cpu(), hl.argmax(dim=-1).cpu(),
            torch.stack(pending_ocr_bt), torch.stack(pending_ocr_ht),
            pending_ocr_iou, acc,
        )
        pending_ocr_x, pending_ocr_bt, pending_ocr_ht, pending_ocr_iou = [], [], [], []

    def flush_det() -> None:
        nonlocal det_batch_imgs, det_batch_meta, det_batch_orig, det_batch_gt, det_batch_targets
        if not det_batch_imgs:
            return
        xs = torch.stack(det_batch_imgs).to(device_str)
        with torch.no_grad():
            preds = det_model(xs).cpu().numpy()
        for i in range(len(det_batch_imgs)):
            orig = det_batch_orig[i]
            meta = det_batch_meta[i]
            gt = det_batch_gt[i]
            bt_target, ht_target = det_batch_targets[i]
            oh, ow = orig.shape[:2]
            pred_bbox = detector_bbox_to_original(preds[i], meta, (oh, ow), det_S)
            iou = bbox_iou(pred_bbox, gt)
            x1, y1, x2, y2 = pred_bbox
            crop = orig[y1:y2, x1:x2]
            if crop.size == 0:
                crop = orig
            x = ocr_tf(image=crop)["image"]
            pending_ocr_x.append(x)
            pending_ocr_bt.append(bt_target)
            pending_ocr_ht.append(ht_target)
            pending_ocr_iou.append(iou)
            if len(pending_ocr_x) >= batch_size:
                flush_ocr()
        det_batch_imgs, det_batch_meta, det_batch_orig, det_batch_gt, det_batch_targets = [], [], [], [], []

    for row in rows:
        arr = _decode_row(row)
        if arr is None:
            skipped += 1
            continue
        lb, meta = letterbox(arr, det_S)
        det_batch_imgs.append(normalize_imagenet(lb))
        det_batch_meta.append(meta)
        det_batch_orig.append(arr)
        det_batch_gt.append(tuple(row["gt_bbox"]))
        parsed = parse_sfen(row["sfen"])
        det_batch_targets.append((
            torch.tensor(parsed.board, dtype=torch.long),
            torch.tensor(parsed.hand, dtype=torch.long),
        ))
        if len(det_batch_imgs) >= batch_size:
            flush_det()
    flush_det()
    flush_ocr()
    return finalize(acc) | {"n": acc["n"], "skipped": skipped}


def load_ocr_ckpt(ckpt_path: Path, device_str: str) -> tuple[torch.nn.Module, int, str]:
    ck = torch.load(ckpt_path, map_location=device_str, weights_only=False)
    bb = ck["backbone"]
    S = ck["image_size"]
    model = BoardOCR(backbone=bb, pretrained=False).to(device_str).eval()
    model.load_state_dict(ck["model"])
    return model, S, bb


def load_detector_ckpt(ckpt_path: Path, device_str: str) -> tuple[torch.nn.Module, int]:
    ck = torch.load(ckpt_path, map_location=device_str, weights_only=False)
    model = BoardDetector(backbone=ck["backbone"], pretrained=False).to(device_str).eval()
    model.load_state_dict(ck["model"])
    return model, ck["image_size"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES,
                   help="OCR backbone names; ckpts loaded from <ckpt-root>/board-ocr-<bb>/<ckpt-name>.")
    p.add_argument("--devices", nargs="+", default=DEFAULT_DEVICES)
    p.add_argument("--ckpt-root", type=Path, default=Path("runs"))
    p.add_argument("--ckpt-name", default="latest.pt")
    p.add_argument("--detector-ckpt", type=Path,
                   default=Path("runs/board-detector-v1/latest.pt"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap SFEN rows (applied before per-device flatten).")
    p.add_argument("--out", type=Path, default=None,
                   help="JSON output path; parent dir auto-created.")
    p.add_argument("--wandb", action="store_true",
                   help="Log per-device metrics to W&B (one run per backbone).")
    p.add_argument("--wandb-project", default="mito-train-board-ocr-w384-v0.3.1-eval-realistic-e2e")
    p.add_argument("--wandb-run-name", default=None,
                   help="Wandb run display name; defaults to backbone name.")
    p.add_argument("--wandb-group", default=None,
                   help="Wandb group id; defaults to 'e2e-realistic-YYYYMMDD'.")
    args = p.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device_str} backbones={args.backbones} devices={args.devices}")
    print(f"[eval] detector={args.detector_ckpt}")

    if args.wandb and len(args.backbones) != 1:
        raise SystemExit(
            "[eval] --wandb requires exactly one --backbones entry per invocation."
        )

    det_model, det_S = load_detector_ckpt(args.detector_ckpt, device_str)
    print(f"[eval] detector image_size={det_S}")

    wandb_run = None
    if args.wandb:
        bb_for_wandb = args.backbones[0]
        run_name = args.wandb_run_name or bb_for_wandb
        group = args.wandb_group or time.strftime("e2e-realistic-%Y%m%d")
        _os.environ["WANDB_RUN_GROUP"] = group
        _os.environ["WANDB_JOB_TYPE"] = "eval-realistic-e2e"
        wandb_run = _init_wandb(
            project=args.wandb_project,
            run_name=run_name,
            config={
                "dataset": REPO_ID,
                "config": CONFIG,
                "backbone": bb_for_wandb,
                "devices": list(args.devices),
                "batch_size": args.batch_size,
                "limit": args.limit,
                "ckpt_name": args.ckpt_name,
                "detector_ckpt": str(args.detector_ckpt),
                "detector_image_size": det_S,
                "group": group,
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
        ocr_model, S, bb_actual = load_ocr_ckpt(ckpt_path, device_str)
        tf = build_transform("val", image_size=S)
        results[bb] = {
            "_ocr_ckpt": str(ckpt_path), "_ocr_image_size": S,
            "_detector_ckpt": str(args.detector_ckpt), "_detector_image_size": det_S,
        }
        for dev in args.devices:
            if not per_dev[dev]:
                continue
            t0 = time.time()
            m = eval_backbone_device(
                ocr_model, tf, det_model, det_S,
                device_str, per_dev[dev], args.batch_size,
            )
            results[bb][dev] = m
            print(fmt_row(bb, dev, m["n"], m, time.time() - t0))
            if wandb_run is not None:
                dev_slug = dev.replace(",", "_")
                wandb_run.log({
                    f"{dev_slug}/cell": m["cell"],
                    f"{dev_slug}/board_full": m["board_full"],
                    f"{dev_slug}/slot": m["slot"],
                    f"{dev_slug}/hand_full": m["hand_full"],
                    f"{dev_slug}/sfen": m["sfen"],
                    f"{dev_slug}/hand_mae": m["hand_mae"],
                    f"{dev_slug}/iou_mean": m["iou_mean"],
                    f"{dev_slug}/n": m["n"],
                    f"{dev_slug}/skipped": m["skipped"],
                })
        del ocr_model
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
        if len(summary) == 1:
            avg = next(iter(summary.values()))
            wandb_run.log({"summary/mean_sfen": avg})
            wandb_run.summary["summary/mean_sfen"] = avg
        wandb_run.finish()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
