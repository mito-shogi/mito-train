"""ONNX INT8 quantization.

Usage:
    python -m mito_train.export.quantize --input models/piece-classifier-v1.onnx

TODO: consider static quantization (QDQ) with a representative dataset. Start with dynamic quantization to prove the pipeline works.
"""
from __future__ import annotations
import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    from onnxruntime.quantization import quantize_dynamic, QuantType

    out = args.output or args.input.with_name(args.input.stem + "-int8.onnx")
    quantize_dynamic(str(args.input), str(out), weight_type=QuantType.QInt8)
    print(f"[quantize] wrote {out}")


if __name__ == "__main__":
    main()
