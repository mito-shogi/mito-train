"""持ち駒 (14 slots × 0..18 counts) の分布を train.jsonl から集計して可視化する。

使い方:
    uv run python scripts/inspect/analyze_hand_distribution.py \
        --data ./data/ocr/train.jsonl --out ./runs/hand-distribution.png
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from mito_train.datasets.sfen_utils import (
    HAND_MAX_COUNT,
    HAND_SLOTS,
    parse_sfen,
)

SLOT_LABELS = [
    "S:P", "S:L", "S:N", "S:S", "S:G", "S:B", "S:R",
    "G:P", "G:L", "G:N", "G:S", "G:G", "G:B", "G:R",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("./data/ocr/train.jsonl"))
    p.add_argument("--out", type=Path, default=Path("./runs/hand-distribution.png"))
    p.add_argument("--no-plot", action="store_true", help="skip saving PNG, only print table")
    args = p.parse_args()

    per_slot: list[Counter[int]] = [Counter() for _ in range(HAND_SLOTS)]
    total = 0

    with args.data.open() as f:
        for line in f:
            row = json.loads(line)
            hand = parse_sfen(row["sfen"]).hand
            for slot_idx, count in enumerate(hand):
                per_slot[slot_idx][count] += 1
            total += 1

    print(f"parsed {total} positions from {args.data}")
    print()

    print(f"{'slot':<6}", end="")
    for k in range(HAND_MAX_COUNT):
        print(f"{k:>6}", end="")
    print(f"{'nonzero%':>10}")

    global_counter: Counter[int] = Counter()
    for label, counter in zip(SLOT_LABELS, per_slot):
        global_counter.update(counter)
        print(f"{label:<6}", end="")
        for k in range(HAND_MAX_COUNT):
            v = counter.get(k, 0)
            print(f"{v:>6}", end="")
        nonzero_pct = 100 * (1 - counter.get(0, 0) / total)
        print(f"{nonzero_pct:>9.2f}%")

    print()
    print("=== global class distribution across all 14 slots ===")
    grand_total = total * HAND_SLOTS
    for k in range(HAND_MAX_COUNT):
        v = global_counter.get(k, 0)
        if v == 0:
            continue
        pct = 100 * v / grand_total
        weight = grand_total / (HAND_MAX_COUNT * v) if v > 0 else float("inf")
        print(f"  count={k:>2}  n={v:>8}  {pct:>6.2f}%  suggested_class_weight={weight:.3f}")

    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed. skipping plot.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 7, figsize=(21, 6), sharey=True)
    for slot_idx, ax in enumerate(axes.flat):
        counter = per_slot[slot_idx]
        xs = list(range(HAND_MAX_COUNT))
        ys = [counter.get(k, 0) for k in xs]
        ax.bar(xs, ys)
        ax.set_yscale("log")
        ax.set_title(SLOT_LABELS[slot_idx])
        ax.set_xlabel("count")
    fig.suptitle(f"Hand-count distribution per slot (n={total}, log-y)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
