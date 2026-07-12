"""Training entry point for the board-detector (skeleton).

Usage:
    python -m mito_train.training.train_board --data ./data/piyo-train --epochs 5

TODO: implement the training loop once corner annotations or mask generation are ready.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mito_train.models.board_detector import BoardDetector
from mito_train.training.train_piece import get_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("./data/piyo-train"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("./runs/board-detector"))
    p.parse_args()

    device = get_device()
    model = BoardDetector().to(device)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"[train_board] device={device} BoardDetector params={n_params:,}")
    print("[train_board] TODO: implement training loop once annotations are ready.")


if __name__ == "__main__":
    main()
