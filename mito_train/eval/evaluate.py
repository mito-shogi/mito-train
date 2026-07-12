"""End-to-end evaluation over the 100 test-realistic images (skeleton).

Usage:
    python -m mito_train.eval.evaluate --data ./data/piyo-test-realistic

TODO: chain the 3 models, build the SFEN, and compare against annotations.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mito_train.eval.metrics import exact_match


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("./data/piyo-test-realistic"))
    args = p.parse_args()

    ann = args.data / "annotations.json"
    if not ann.exists():
        print(f"[evaluate] WARNING: {ann} missing. Waiting for test-realistic delivery.")
        return

    annotations = json.loads(ann.read_text())
    print(f"[evaluate] {len(annotations)} entries queued for evaluation (TODO: wire up inference pipeline).")
    _ = exact_match  # used once implemented


if __name__ == "__main__":
    main()
