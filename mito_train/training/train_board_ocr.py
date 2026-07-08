"""Training entry point for BoardOCR (image -> 9x9 board grid + 14-slot hand counts).

Currently only smoke mode is available:
    python -m mito_train.training.train_board_ocr --mode smoke --limit 100 --epochs 5

Real training (intended to run on a GPU machine) uses --mode full to iterate over all 27k samples:
    python -m mito_train.training.train_board_ocr --mode full --epochs 20
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mito_train.datasets import CaptureDataset, build_transform
from mito_train.datasets.sfen_utils import parse_sfen
from mito_train.models import BoardOCR
from mito_train.training.train_piece import _init_wandb, get_device


def compute_hand_class_weights(
    entries: list[dict],
    num_classes: int,
    clip_min: float = 0.5,
    clip_max: float = 10.0,
) -> torch.Tensor:
    """sqrt(1/freq) class weights for hand CE, computed from the training manifest.

    Rare hand-counts (e.g. 8+ pieces) get up-weighted; the dominant 0 class
    gets down-weighted. The sqrt + clip[clip_min, clip_max] combo tames the
    ~5-orders-of-magnitude raw frequency spread so gradients stay stable.
    Classes never observed keep clip_max (still upweighted, no direct signal).
    """
    counts = torch.zeros(num_classes, dtype=torch.long)
    for e in entries:
        for c in parse_sfen(e["sfen"]).hand:
            counts[c] += 1
    total = counts.sum().clamp(min=1)
    freq = counts.float() / total
    weights = (1.0 / freq.clamp(min=1e-9)).sqrt()
    weights = torch.where(counts > 0, weights, torch.full_like(weights, clip_max))
    return weights.clamp(clip_min, clip_max)


def compute_loss(
    board_logits: torch.Tensor,  # (B, 29, 9, 9)
    hand_logits: torch.Tensor,   # (B, 14, 19)
    board_target: torch.Tensor,  # (B, 9, 9)
    hand_target: torch.Tensor,   # (B, 14)
    hand_weight: float = 1.0,
    hand_class_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted sum of board CE + hand CE."""
    board_loss = F.cross_entropy(board_logits, board_target)
    B, S, C = hand_logits.shape
    hand_loss = F.cross_entropy(
        hand_logits.reshape(B * S, C),
        hand_target.reshape(B * S),
        weight=hand_class_weight,
    )
    total = board_loss + hand_weight * hand_loss
    return total, board_loss, hand_loss


def compute_metrics(
    board_logits: torch.Tensor,
    hand_logits: torch.Tensor,
    board_target: torch.Tensor,
    hand_target: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        board_pred = board_logits.argmax(dim=1)  # (B, 9, 9)
        board_cell_acc = (board_pred == board_target).float().mean().item()
        # Fraction of images where all 81 cells are correct
        board_full = (board_pred == board_target).all(dim=(1, 2)).float().mean().item()

        hand_pred = hand_logits.argmax(dim=-1)  # (B, 14)
        hand_slot_acc = (hand_pred == hand_target).float().mean().item()
        hand_full = (hand_pred == hand_target).all(dim=1).float().mean().item()

        sfen_full = (
            ((board_pred == board_target).all(dim=(1, 2)))
            & ((hand_pred == hand_target).all(dim=1))
        ).float().mean().item()
    return {
        "board/cell_acc": board_cell_acc,
        "board/full_acc": board_full,
        "hand/slot_acc": hand_slot_acc,
        "hand/full_acc": hand_full,
        "sfen/full_acc": sfen_full,
    }


def run(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"[board_ocr] device={device} mode={args.mode} backbone={args.backbone}")

    train_tf = build_transform("train", image_size=args.image_size)
    val_tf = build_transform("val", image_size=args.image_size)

    limit_train = args.limit if args.mode == "smoke" else None
    limit_val = min(args.limit, 20) if args.mode == "smoke" else None

    if args.hf_repo_id:
        # HF-hosted parquet dataset (embedded PNG bytes). First run downloads
        # the shards into HF cache (~21GB); subsequent runs are cache-hits.
        from mito_train.datasets import HFCaptureDataset
        print(f"[board_ocr] loading from HF repo: {args.hf_repo_id}")
        train_ds = HFCaptureDataset(
            repo_id=args.hf_repo_id, split="train",
            transform=train_tf, limit=limit_train,
        )
        val_ds = HFCaptureDataset(
            repo_id=args.hf_repo_id, split="val",
            transform=val_tf, limit=limit_val,
        )
    else:
        train_ds = CaptureDataset(
            manifest_path=args.train_manifest,
            image_root=args.image_root,
            transform=train_tf,
            limit=limit_train,
        )
        val_ds = CaptureDataset(
            manifest_path=args.val_manifest,
            image_root=args.image_root,
            transform=val_tf,
            limit=limit_val,
        )
    print(f"[board_ocr] train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    model = BoardOCR(backbone=args.backbone, pretrained=args.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[board_ocr] {args.backbone} params={n_params:,}")

    if args.hand_class_weight:
        hand_class_weight = compute_hand_class_weights(
            train_ds.entries,
            num_classes=model.hand_max,
            clip_min=args.class_weight_clip_min,
            clip_max=args.class_weight_clip_max,
        ).to(device)
        print(
            "[board_ocr] hand class weights (sqrt+clip): "
            + ", ".join(f"{i}:{w:.2f}" for i, w in enumerate(hand_class_weight.tolist()))
        )
    else:
        hand_class_weight = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    resumed_from: str | None = None
    resumed_wandb_id: str | None = None
    if args.resume is not None:
        ckpt_path = args.ckpt_dir / "latest.pt" if str(args.resume) == "latest" else args.resume
        print(f"[board_ocr] resume from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        resumed_from = str(ckpt_path)
        resumed_wandb_id = ckpt.get("wandb_run_id")
        print(f"[board_ocr] resumed at epoch={start_epoch} (ckpt was epoch {ckpt['epoch']})")
        if resumed_wandb_id:
            print(f"[board_ocr] will resume wandb run id={resumed_wandb_id}")

    if args.wandb_run_id:
        resumed_wandb_id = args.wandb_run_id
        print(f"[board_ocr] wandb run id overridden by CLI: {resumed_wandb_id}")

    run_name = f"board-ocr-{args.mode}-{args.backbone}"
    wandb_run = _init_wandb(
        project="mito-train-board-ocr",
        run_name=run_name,
        run_id=resumed_wandb_id,
        resume="allow" if resumed_wandb_id else None,
        config={
            "mode": args.mode,
            "backbone": args.backbone,
            "pretrained": args.pretrained,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "start_epoch": start_epoch,
            "resumed_from": resumed_from,
            "hand_weight": args.hand_weight,
            "hand_class_weight": args.hand_class_weight,
            "class_weight_clip_min": args.class_weight_clip_min,
            "class_weight_clip_max": args.class_weight_clip_max,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "n_params": n_params,
            "device": device,
        },
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_board_loss = 0.0
        running_hand_loss = 0.0
        running_metrics = {
            "board/cell_acc": 0.0, "board/full_acc": 0.0,
            "hand/slot_acc": 0.0, "hand/full_acc": 0.0, "sfen/full_acc": 0.0,
        }
        n_batches = 0
        for img, board, hand in train_loader:
            img = img.to(device)
            board = board.to(device)
            hand = hand.to(device)
            board_logits, hand_logits = model(img)
            total, bl, hl = compute_loss(
                board_logits, hand_logits, board, hand,
                hand_weight=args.hand_weight,
                hand_class_weight=hand_class_weight,
            )
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            running_loss += total.item()
            running_board_loss += bl.item()
            running_hand_loss += hl.item()
            m = compute_metrics(board_logits, hand_logits, board, hand)
            for k, v in m.items():
                running_metrics[k] += v
            n_batches += 1

        avg = {k: v / n_batches for k, v in running_metrics.items()}
        avg_loss = running_loss / n_batches
        avg_bl = running_board_loss / n_batches
        avg_hl = running_hand_loss / n_batches

        print(
            f"[board_ocr] epoch={epoch:3d} "
            f"loss={avg_loss:.4f} (board={avg_bl:.4f} hand={avg_hl:.4f}) "
            f"cell_acc={avg['board/cell_acc']:.3f} "
            f"sfen_acc={avg['sfen/full_acc']:.3f}"
        )

        log = {
            "train/loss": avg_loss,
            "train/board_loss": avg_bl,
            "train/hand_loss": avg_hl,
            "epoch": epoch,
        }
        log.update({f"train/{k}": v for k, v in avg.items()})

        # Simple val (in smoke mode this is only a same-distribution sanity check against train)
        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            val_metrics = {k: 0.0 for k in running_metrics}
            n_val_batches = 0
            with torch.no_grad():
                for img, board, hand in val_loader:
                    img = img.to(device)
                    board = board.to(device)
                    hand = hand.to(device)
                    bl_out, hl_out = model(img)
                    m = compute_metrics(bl_out, hl_out, board, hand)
                    for k, v in m.items():
                        val_metrics[k] += v
                    n_val_batches += 1
            if n_val_batches > 0:
                v_avg = {k: v / n_val_batches for k, v in val_metrics.items()}
                print(
                    f"[board_ocr]           val cell_acc={v_avg['board/cell_acc']:.3f} "
                    f"sfen_acc={v_avg['sfen/full_acc']:.3f}"
                )
                log.update({f"val/{k}": v for k, v in v_avg.items()})

        if wandb_run is not None:
            wandb_run.log(log)

        ckpt_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "backbone": args.backbone,
            "image_size": args.image_size,
            "hand_weight": args.hand_weight,
            "wandb_run_id": wandb_run.id if wandb_run is not None else None,
        }
        torch.save(ckpt_payload, args.ckpt_dir / "latest.pt")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(ckpt_payload, args.ckpt_dir / f"epoch-{epoch:03d}.pt")

    if wandb_run is not None:
        wandb_run.finish()

    # Smoke check
    if args.mode == "smoke":
        final_cell = avg["board/cell_acc"]
        if final_cell < 0.6:
            print(f"[board_ocr] WARNING: did not overfit (cell_acc={final_cell:.3f}).")
        else:
            print(f"[board_ocr] smoke OK (cell_acc={final_cell:.3f}). Design pipeline verified.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--backbone", default="mobilenet_v3_small",
                   choices=["mobilenet_v3_small", "convnext_tiny"])
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--train-manifest", type=Path, default=Path("./data/train.jsonl"))
    p.add_argument("--val-manifest", type=Path, default=Path("./data/val.jsonl"))
    p.add_argument("--image-root", type=Path, default=Path("./data/captures/d0"))
    p.add_argument("--hf-repo-id", type=str, default=None,
                   help="Load from a HF dataset repo (e.g. 'ultemica/piyoshogi') instead of local manifests.")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hand-weight", type=float, default=1.0)
    p.add_argument("--hand-class-weight", action="store_true", default=True,
                   help="Use sqrt(1/freq) class weights on hand CE.")
    p.add_argument("--no-hand-class-weight", dest="hand_class_weight", action="store_false")
    p.add_argument("--class-weight-clip-min", type=float, default=0.5)
    p.add_argument("--class-weight-clip-max", type=float, default=10.0)
    p.add_argument("--val-every", type=int, default=2)
    p.add_argument("--ckpt-dir", type=Path, default=Path("./runs/board-ocr"))
    p.add_argument("--save-every", type=int, default=5,
                   help="Interval for saving epoch-{N}.pt snapshots. latest.pt is saved every epoch.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Checkpoint path. Pass 'latest' to load --ckpt-dir/latest.pt.")
    p.add_argument("--wandb-run-id", type=str, default=None,
                   help="Force resume this W&B run id (overrides ckpt's stored id).")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
