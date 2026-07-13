"""PyTorch checkpoint -> ONNX export.

Usage:
    # Board OCR (uses checkpoint's backbone + image_size)
    python -m mito_train.export.to_onnx \\
        --checkpoint runs/board-ocr-efficientnet_b1/latest.pt \\
        --model board_ocr \\
        --out models/board-ocr-efficientnet_b1-w384-fp32.onnx

    # Board detector
    python -m mito_train.export.to_onnx \\
        --checkpoint runs/board-detector-v1/latest.pt \\
        --model board_detector \\
        --out models/board-detector-w384-fp32.onnx

Contract: input/output names and shapes must match docs/ocr-model-interface.md.
Model-specific I/O:

  piece:          (1, 3, 32, 32)          -> logits (1, C_piece)
  board_detector: (1, 3, image_size, ...) -> bbox (1, 4)
  board_ocr:      (1, 3, image_size, ...) -> board_logits (1, 29, 9, 9),
                                             hand_logits  (1, 14, 19)
  hand:           (1, 3, 64, 128)         -> counts (1, C_hand)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch

from mito_train.models.board_detector import BoardDetector
from mito_train.models.board_ocr import BoardOCR
from mito_train.models.hand_classifier import HandClassifier
from mito_train.models.piece_classifier import PieceClassifier


def _spec_piece(_ck: dict | None) -> tuple[torch.nn.Module, tuple[int, ...], list[str], list[str]]:
    return PieceClassifier(), (1, 3, 32, 32), ["input"], ["logits"]


def _spec_hand(_ck: dict | None) -> tuple[torch.nn.Module, tuple[int, ...], list[str], list[str]]:
    return HandClassifier(), (1, 3, 64, 128), ["input"], ["counts"]


def _spec_board_detector(ck: dict | None) -> tuple[torch.nn.Module, tuple[int, ...], list[str], list[str]]:
    bb = (ck or {}).get("backbone", "mobilenet_v3_small")
    S = int((ck or {}).get("image_size", 384))
    return BoardDetector(backbone=bb, pretrained=False), (1, 3, S, S), ["input"], ["bbox"]


def _spec_board_ocr(ck: dict | None) -> tuple[torch.nn.Module, tuple[int, ...], list[str], list[str]]:
    bb = (ck or {}).get("backbone", "mobilenet_v3_small")
    S = int((ck or {}).get("image_size", 384))
    hand_mode = (ck or {}).get("hand_mode", "classification")
    model = BoardOCR(backbone=bb, hand_mode=hand_mode, pretrained=False)
    return model, (1, 3, S, S), ["input"], ["board_logits", "hand_logits"]


# Factory-per-model so checkpoint-side settings (backbone, image_size, hand_mode)
# survive into the exported graph without CLI having to know them.
MODEL_SPECS: dict[str, Callable[[dict | None], tuple[torch.nn.Module, tuple[int, ...], list[str], list[str]]]] = {
    "piece": _spec_piece,
    "hand": _spec_hand,
    "board_detector": _spec_board_detector,
    # Legacy alias — old CLI used --model board for the detector.
    "board": _spec_board_detector,
    "board_ocr": _spec_board_ocr,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=False)
    p.add_argument("--model", choices=list(MODEL_SPECS), default="piece")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    ck: dict | None = None
    if args.checkpoint and args.checkpoint.exists():
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        print(f"[to_onnx] loaded checkpoint {args.checkpoint}")
        for k in ("backbone", "image_size", "hand_mode", "epoch"):
            if k in ck:
                print(f"[to_onnx]   {k}={ck[k]}")

    model, dummy_shape, in_names, out_names = MODEL_SPECS[args.model](ck)
    if ck is not None:
        state = ck.get("model", ck)
        model.load_state_dict(state)
    else:
        print("[to_onnx] WARNING: no checkpoint. Exporting from random init (pipeline sanity check only).")
    model.eval()

    out = args.out or Path(f"./models/{args.model}-v1.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(*dummy_shape)

    dynamic_axes = {n: {0: "batch"} for n in in_names + out_names}
    torch.onnx.export(
        model, dummy, str(out),
        input_names=in_names, output_names=out_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
    )

    # torch 2.x dynamo exporter can externalize weights to <name>.onnx.data.
    # Inline them back into a single file for simple R2 delivery / MITO loading.
    import onnx
    sidecar = out.with_name(out.name + ".data")
    if sidecar.exists():
        m = onnx.load(str(out))  # also loads the external data
        onnx.save_model(m, str(out), save_as_external_data=False)
        sidecar.unlink()
        print(f"[to_onnx] inlined external weights, removed {sidecar.name}")

    print(f"[to_onnx] wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
