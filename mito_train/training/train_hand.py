"""Training entry point for the hand-classifier (skeleton).

Usage:
    python -m mito_train.training.train_hand --data ./data/piyo-train --epochs 5

TODO: implement the multi-head training loop once hand labels are ready.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mito_train.models.hand_classifier import HandClassifier
from mito_train.training.train_piece import get_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("./data/piyo-train"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("./runs/hand-classifier"))
    p.parse_args()

    device = get_device()
    model = HandClassifier().to(device)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"[train_hand] device={device} HandClassifier params={n_params:,}")
    print("[train_hand] TODO: implement training loop once hand labels are ready.")


if __name__ == "__main__":
    main()
