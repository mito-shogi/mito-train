"""piece-classifier 学習エントリポイント。

使い方:
    python -m mito_train.training.train_piece --data ./data/piyo-train --epochs 5
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("./data/piyo-train"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("./runs/piece-cnn"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: Milestone 1 で実装
    raise NotImplementedError(f"train_piece not yet implemented. args={args}")


if __name__ == "__main__":
    main()
