"""Verify ONNX exports match PyTorch numerics.

For each (backbone, precision) pair under ./models/, load both the ONNX and
the source PyTorch checkpoint and compare:
  - forward output shapes
  - board argmax (per-cell class) equality
  - hand argmax (per-slot count) equality
  - max absolute numerical diff (for regression checks and detector bbox)

Detects silent quantization damage (e.g. int8 that broke argmax) before the
model gets shipped.

Usage:
    uv run python scripts/export/verify_onnx.py                    # all found
    uv run python scripts/export/verify_onnx.py --backbones efficientnet_b1
    uv run python scripts/export/verify_onnx.py --precisions fp32 fp16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from mito_train.models import BoardDetector, BoardOCR

CKPTS = {
    "detector": ("board_detector", "runs/board-detector-v1/latest.pt"),
    "mobilenet_v3_small": ("board_ocr", "runs/board-ocr-mobilenet_v3_small/latest.pt"),
    "mobilenet_v3_large": ("board_ocr", "runs/board-ocr-mobilenet_v3_large/latest.pt"),
    "efficientnet_b1": ("board_ocr", "runs/board-ocr-efficientnet_b1/latest.pt"),
    "convnext_nano": ("board_ocr", "runs/board-ocr-convnext_nano/latest.pt"),
    "convnext_tiny": ("board_ocr", "runs/board-ocr-convnext_tiny/latest.pt"),
}

# Maps ONNX filename middle segment -> (backbone_key, prefix used in filename).
FILENAME_PATTERNS = [
    ("board-detector-mnv3s-w384", "detector"),
    ("board-ocr-mobilenet_v3_small-w384", "mobilenet_v3_small"),
    ("board-ocr-mobilenet_v3_large-w384", "mobilenet_v3_large"),
    ("board-ocr-efficientnet_b1-w384", "efficientnet_b1"),
    ("board-ocr-convnext_nano-w384", "convnext_nano"),
    ("board-ocr-convnext_tiny-w384", "convnext_tiny"),
]


def build_pt_model(kind: str, ckpt_path: Path) -> tuple[torch.nn.Module, int, str]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bb = ck.get("backbone", "mobilenet_v3_small")
    S = int(ck.get("image_size", 384))
    if kind == "board_detector":
        model = BoardDetector(backbone=bb, pretrained=False).eval()
    else:
        hm = ck.get("hand_mode", "classification")
        model = BoardOCR(backbone=bb, hand_mode=hm, pretrained=False).eval()
    model.load_state_dict(ck["model"])
    return model, S, bb


def verify(prefix: str, key: str, precisions: list[str], models_dir: Path) -> None:
    kind, ckpt_path = CKPTS[key]
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"[{key}] PT ckpt missing: {ckpt_path}")
        return

    pt_model, S, _ = build_pt_model(kind, ckpt_path)
    torch.manual_seed(42)
    x = torch.randn(1, 3, S, S)
    with torch.no_grad():
        pt_out = pt_model(x)

    for prec in precisions:
        onnx_path = models_dir / f"{prefix}-{prec}.onnx"
        if not onnx_path.exists():
            continue
        try:
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            outs = sess.run(None, {"input": x.numpy().astype(np.float32)})
        except Exception as e:
            print(f"[{key}:{prec}] load/run failed: {e}")
            continue

        size_mb = onnx_path.stat().st_size / 1024 / 1024
        if kind == "board_detector":
            pt_b = pt_out.numpy()
            diff = float(np.abs(pt_b - outs[0]).max())
            print(f"[{key:<20}:{prec:<4}] {size_mb:>7.2f} MB  bbox max_diff={diff:.6f}  (regression)")
        else:
            pt_b, pt_h = pt_out
            db = float(np.abs(pt_b.numpy() - outs[0]).max())
            pt_ba = pt_b.argmax(dim=1).numpy()
            onnx_ba = outs[0].argmax(axis=1)
            pt_ha = pt_h.argmax(dim=-1).numpy()
            onnx_ha = outs[1].argmax(axis=-1)
            board_ok = int((pt_ba == onnx_ba).sum())
            hand_ok = int((pt_ha == onnx_ha).sum())
            board_total = pt_ba.size
            hand_total = pt_ha.size
            print(
                f"[{key:<20}:{prec:<4}] {size_mb:>7.2f} MB  "
                f"board max_diff={db:.5f}  argmax board {board_ok}/{board_total}  hand {hand_ok}/{hand_total}"
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", type=Path, default=Path("./models"))
    p.add_argument("--backbones", nargs="+", default=None,
                   help="Filter by backbone key (matches CKPTS keys).")
    p.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "int8"])
    args = p.parse_args()

    for prefix, key in FILENAME_PATTERNS:
        if args.backbones and key not in args.backbones:
            continue
        found = any((args.models_dir / f"{prefix}-{p}.onnx").exists() for p in args.precisions)
        if not found:
            continue
        verify(prefix, key, args.precisions, args.models_dir)


if __name__ == "__main__":
    main()
