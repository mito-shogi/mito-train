"""ONNX INT8 dynamic quantization.

Usage:
    python -m mito_train.export.quantize --input models/board-ocr-efficientnet_b1-w384-fp32.onnx

Runs `quant_pre_process` first (shape inference, symbolic shape, optimization)
so that quantize_dynamic doesn't trip on the dynamo exporter's dynamic shapes.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Skip quant_pre_process. Older ONNX exports may not need it.")
    args = p.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out = args.output or args.input.with_name(args.input.stem + "-int8.onnx")

    if args.skip_preprocess:
        quantize_dynamic(str(args.input), str(out), weight_type=QuantType.QInt8)
    else:
        # Pre-process into a temp file, then quantize. quant_pre_process runs
        # symbolic shape inference + graph optimizations that quantize_dynamic
        # otherwise assumes are already done. Falls back to skipping symbolic
        # shape inference if it errors out (ConvNeXt has empty-name inputs the
        # inferrer can't handle) — the quantizer itself will re-check.
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tf:
            preproc_path = Path(tf.name)
        try:
            try:
                quant_pre_process(str(args.input), str(preproc_path), skip_symbolic_shape=False)
            except Exception as e:
                print(f"[quantize] symbolic shape inference failed ({type(e).__name__}), retrying with skip_symbolic_shape=True")
                quant_pre_process(str(args.input), str(preproc_path), skip_symbolic_shape=True)
            quantize_dynamic(str(preproc_path), str(out), weight_type=QuantType.QInt8)
        finally:
            if preproc_path.exists():
                preproc_path.unlink()

    print(f"[quantize] wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
