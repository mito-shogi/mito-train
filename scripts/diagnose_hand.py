"""Diagnose hand-head accuracy on val split.

Loads a board-ocr checkpoint and reports:
  - board/cell_acc, board/full_acc (reference)
  - hand/slot_acc, hand/full_acc (overall)
  - per-slot accuracy (14 slots)
  - confusion matrix: true count -> predicted count (aggregated across slots)

Usage:
    uv run python scripts/diagnose_hand.py \
        --ckpt ./runs/board-ocr-v2/latest.pt \
        --val-manifest ./data/ocr/val.jsonl \
        --image-root ./data/ocr/iPhone10,1
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mito_train.datasets import CaptureDataset, build_transform
from mito_train.datasets.sfen_utils import HAND_MAX_COUNT, HAND_SLOTS
from mito_train.models import BoardOCR
from mito_train.training.train_piece import get_device

PIECE_LABELS = ["sP", "sL", "sN", "sS", "sG", "sB", "sR",
                "gP", "gL", "gN", "gS", "gG", "gB", "gR"]


@torch.no_grad()
def diagnose(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"[diag] device={device} ckpt={args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "mobilenet_v3_small")
    image_size = ckpt.get("image_size", args.image_size)
    print(f"[diag] backbone={backbone} image_size={image_size} epoch={ckpt.get('epoch')}")

    val_ds = CaptureDataset(
        manifest_path=args.val_manifest,
        image_root=args.image_root,
        transform=build_transform("val", image_size=image_size),
    )
    print(f"[diag] val samples={len(val_ds)}")
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    model = BoardOCR(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    board_cell_correct = 0
    board_cell_total = 0
    board_full_correct = 0
    board_full_total = 0
    hand_slot_correct = 0
    hand_slot_total = 0
    hand_full_correct = 0
    hand_full_total = 0

    per_slot_correct = torch.zeros(HAND_SLOTS, dtype=torch.long)
    per_slot_total = torch.zeros(HAND_SLOTS, dtype=torch.long)
    confusion = torch.zeros(HAND_MAX_COUNT, HAND_MAX_COUNT, dtype=torch.long)

    for img, board, hand in val_loader:
        img = img.to(device)
        board_logits, hand_logits = model(img)  # (B,29,9,9), (B,14,19)
        board_pred = board_logits.argmax(dim=1).cpu()  # (B,9,9)
        hand_pred = hand_logits.argmax(dim=-1).cpu()   # (B,14)

        # Board
        board_cell_correct += (board_pred == board).sum().item()
        board_cell_total += board.numel()
        full_ok = (board_pred == board).view(board.size(0), -1).all(dim=1)
        board_full_correct += full_ok.sum().item()
        board_full_total += board.size(0)

        # Hand
        slot_ok = (hand_pred == hand)  # (B, 14)
        hand_slot_correct += slot_ok.sum().item()
        hand_slot_total += hand.numel()
        hand_full_correct += slot_ok.all(dim=1).sum().item()
        hand_full_total += hand.size(0)

        # Per-slot
        per_slot_correct += slot_ok.sum(dim=0)
        per_slot_total += torch.tensor([hand.size(0)] * HAND_SLOTS)

        # Confusion
        for t, p in zip(hand.view(-1), hand_pred.view(-1)):
            confusion[t.item(), p.item()] += 1

    print()
    print("=== overall ===")
    print(f"  board/cell_acc = {board_cell_correct / board_cell_total:.4f}")
    print(f"  board/full_acc = {board_full_correct / board_full_total:.4f}")
    print(f"  hand/slot_acc  = {hand_slot_correct / hand_slot_total:.4f}")
    print(f"  hand/full_acc  = {hand_full_correct / hand_full_total:.4f}")

    print()
    print("=== per-slot accuracy (worst first) ===")
    slot_acc = (per_slot_correct.float() / per_slot_total.clamp(min=1).float()).tolist()
    slot_order = sorted(range(HAND_SLOTS), key=lambda i: slot_acc[i])
    for i in slot_order:
        print(f"  slot {i:>2} ({PIECE_LABELS[i]}): {slot_acc[i]:.4f}  ({per_slot_correct[i].item()}/{per_slot_total[i].item()})")

    print()
    print("=== confusion matrix (true row -> pred col), classes with any presence ===")
    active = [c for c in range(HAND_MAX_COUNT) if confusion[c].sum() > 0 or confusion[:, c].sum() > 0]
    header = "true\\pred " + " ".join(f"{c:>5}" for c in active) + "   total  err%"
    print(header)
    for t in active:
        row = confusion[t, active]
        total = confusion[t].sum().item()
        err = (total - confusion[t, t].item()) / total * 100 if total > 0 else 0.0
        cells = " ".join(f"{v.item():>5}" for v in row)
        print(f"{t:>10} {cells}   {total:>5}  {err:>5.1f}%")

    print()
    print("=== top misclassifications (true != pred, by frequency) ===")
    pairs = []
    for t in range(HAND_MAX_COUNT):
        for p in range(HAND_MAX_COUNT):
            if t != p and confusion[t, p] > 0:
                pairs.append((confusion[t, p].item(), t, p))
    pairs.sort(reverse=True)
    for n, t, p in pairs[:15]:
        print(f"  true={t:>2} -> pred={p:>2}: {n}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("./runs/board-ocr-v2/latest.pt"))
    p.add_argument("--val-manifest", type=Path, default=Path("./data/ocr/val.jsonl"))
    p.add_argument("--image-root", type=Path, default=Path("./data/ocr/iPhone10,1"))
    p.add_argument("--image-size", type=int, default=288)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    diagnose(args)


if __name__ == "__main__":
    main()
