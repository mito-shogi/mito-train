"""Emit MANIFEST.json for a models release.

Bakes in realistic-eval accuracy (from docs/w384-sweep-analysis.md) so a
downstream consumer can pick a model by (target-accuracy, target-size) without
re-running eval.

Usage (invoked by scripts/export/build_release.sh):
    uv run python scripts/export/write_manifest.py \\
        --version v0.1.0 --models-dir ./models --out MANIFEST.json \\
        board-detector-mnv3s-w384-fp32.onnx board-ocr-efficientnet_b1-w384-fp32.onnx ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

# realistic (end-to-end SFEN) from docs/w384-sweep-analysis.md — mean of 4 devices.
REALISTIC_SFEN_MEAN = {
    "mobilenet_v3_small": 0.9537,
    "mobilenet_v3_large": 0.9792,
    "efficientnet_b1":    0.9967,
    "convnext_nano":      0.9992,
    "convnext_tiny":      0.9992,
}
# Per-device realistic SFEN for the 3 backbones we ship.
REALISTIC_SFEN_PER_DEVICE = {
    "mobilenet_v3_large": {
        "iPhone10,1": 0.9800, "iPhone11,8": 0.9790, "iPhone15,4": 0.9820, "iPad14,10": 0.9760,
    },
    "efficientnet_b1": {
        "iPhone10,1": 0.9970, "iPhone11,8": 0.9970, "iPhone15,4": 0.9960, "iPad14,10": 0.9970,
    },
    "convnext_nano": {
        "iPhone10,1": 0.9990, "iPhone11,8": 0.9990, "iPhone15,4": 0.9990, "iPad14,10": 1.0000,
    },
}
DETECTOR_VAL_IOU = 0.99  # detector val iou_mean, from DETECTOR_STATUS.md

# Backbone-tier recommendation from docs/browser-inference-outlook.md.
TIER_RECOMMENDATION = {
    "board-detector-mnv3s": "detector (required)",
    "board-ocr-mobilenet_v3_large": "edge / WASM fallback tier",
    "board-ocr-efficientnet_b1": "browser default — 99% ライン超え",
    "board-ocr-convnext_nano": "desktop precision-preferred tier",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_filename(name: str) -> dict:
    """board-ocr-efficientnet_b1-w384-fp16.onnx ->
       {kind: board_ocr, backbone: efficientnet_b1, image_size: 384, precision: fp16}
    """
    m = re.match(r"^(board-detector|board-ocr)-([a-z0-9_]+)-w(\d+)-(fp32|fp16|int8)\.onnx$", name)
    if not m:
        return {"kind": None, "backbone": None, "image_size": None, "precision": None}
    kind_key, backbone, size, prec = m.groups()
    return {
        "kind": "board_detector" if kind_key == "board-detector" else "board_ocr",
        "backbone": backbone if kind_key == "board-ocr" else "mobilenet_v3_small",
        "image_size": int(size),
        "precision": prec,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--models-dir", type=Path, default=Path("./models"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("files", nargs="+")
    args = p.parse_args()

    entries = []
    for name in args.files:
        path = args.models_dir / name
        info = parse_filename(name)
        e = {
            "file": name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_of(path),
            **info,
        }
        if info["kind"] == "board_ocr" and info["backbone"] in REALISTIC_SFEN_MEAN:
            e["realistic_sfen_mean"] = REALISTIC_SFEN_MEAN[info["backbone"]]
            per_dev = REALISTIC_SFEN_PER_DEVICE.get(info["backbone"])
            if per_dev:
                e["realistic_sfen_per_device"] = per_dev
        if info["kind"] == "board_detector":
            e["detector_val_iou_mean"] = DETECTOR_VAL_IOU

        tier_key = name.rsplit("-w", 1)[0]  # strip -w384-{prec}.onnx tail
        if tier_key in TIER_RECOMMENDATION:
            e["recommendation"] = TIER_RECOMMENDATION[tier_key]
        entries.append(e)

    manifest = {
        "version": args.version,
        "generated_from": "runs/board-*/latest.pt (see docs/w384-sweep-analysis.md)",
        "input_contract": {
            "shape": [1, 3, "image_size", "image_size"],
            "dtype": "float32 (fp32 model) or float16 (fp16 model)",
            "layout": "NCHW, RGB",
            "preprocessing": "aspect-preserving LongestMaxSize -> center Pad to (image_size, image_size) with black, then ImageNet mean/std normalize",
            "note": "See docs/ocr-model-interface.md for the full client-side pipeline.",
        },
        "outputs": {
            "board_detector": {
                "bbox": {"shape": [1, 4], "meaning": "normalized [x1,y1,x2,y2] in the letterboxed image_size x image_size frame; un-letterbox back to original px on the client."},
            },
            "board_ocr": {
                "board_logits": {"shape": [1, 29, 9, 9], "meaning": "per-cell 29-class logits. argmax over dim=1 gives cell class."},
                "hand_logits": {"shape": [1, 14, 19], "meaning": "per-slot piece-count logits. argmax over dim=-1 gives count. Impossible (slot, count) combos are pre-masked with -inf."},
            },
        },
        "files": entries,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    print(f"[write_manifest] wrote {args.out}")


if __name__ == "__main__":
    main()
