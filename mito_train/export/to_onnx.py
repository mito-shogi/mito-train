"""PyTorch checkpoint → ONNX。契約は docs/ocr-model-interface.md 参照。"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--target",
        choices=["board", "piece", "hand"],
        required=True,
        help="どのモデル契約でエクスポートするか",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"to_onnx not yet implemented. args={args}")


if __name__ == "__main__":
    main()
