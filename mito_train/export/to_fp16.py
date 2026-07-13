"""ONNX fp32 -> fp16 conversion.

fp16 is often the sweet spot for browser WebGPU inference: half the file
size vs fp32 (better for CDN/R2 delivery), same speed or slightly faster on
adapters that support fp16 shaders (M-series, Snapdragon 8 Elite, etc.),
and unlike int8 the ort-web WebGPU backend has full fp16 kernel coverage.

Usage:
    python -m mito_train.export.to_fp16 --input models/board-ocr-efficientnet_b1-w384-fp32.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--keep-io-types", action="store_true",
                   help="Keep input/output tensors as fp32; only weights and "
                        "intermediates get converted to fp16. Simpler for the "
                        "client (no dtype conversion at the boundary).")
    args = p.parse_args()

    import onnx
    from onnxconverter_common import float16

    out = args.output or args.input.with_name(args.input.stem.replace("-fp32", "") + "-fp16.onnx")

    model = onnx.load(str(args.input))
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=args.keep_io_types,
    )
    onnx.save(fp16_model, str(out))
    print(f"[to_fp16] wrote {out} ({out.stat().st_size} bytes, keep_io_types={args.keep_io_types})")


if __name__ == "__main__":
    main()
