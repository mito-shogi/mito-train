"""PyTorch checkpoint -> ONNX export.

Usage:
    python -m mito_train.export.to_onnx --checkpoint runs/piece-cnn/best.pt --model piece

Contract: input/output names and shapes must match docs/ocr-model-interface.md exactly.
CI runs contract validation via onnxruntime (see README section 8).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch

from mito_train.models.piece_classifier import PieceClassifier
from mito_train.models.board_detector import BoardDetector
from mito_train.models.hand_classifier import HandClassifier

# Per-model tuple: (constructor, dummy input shape, input name, output name)
MODEL_SPECS = {
    "piece": (PieceClassifier, (1, 3, 32, 32), "input", "logits"),
    "board": (BoardDetector, (1, 3, 256, 256), "input", "corners"),
    "hand": (HandClassifier, (1, 3, 64, 128), "input", "counts"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=False)
    p.add_argument("--model", choices=list(MODEL_SPECS), default="piece")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    ctor, dummy_shape, in_name, out_name = MODEL_SPECS[args.model]
    model = ctor()
    if args.checkpoint and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        print(f"[to_onnx] loaded checkpoint {args.checkpoint}")
    else:
        print("[to_onnx] WARNING: no checkpoint. Exporting from random init (pipeline sanity check only).")
    model.eval()

    out = args.out or Path(f"./models/{args.model}-classifier-v1.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(*dummy_shape)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=[in_name], output_names=[out_name],
        dynamic_axes={in_name: {0: "batch"}, out_name: {0: "batch"}},
        opset_version=args.opset,
    )

    # torch 2.x dynamo exporter externalizes weights to <name>.onnx.data.
    # The model is small, so inline them back into a single file
    # (simplifies R2 delivery and MITO loading).
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
