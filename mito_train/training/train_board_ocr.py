"""Training entry point for BoardOCR (image -> 9x9 board grid + 14-slot hand counts).

Single GPU (smoke):
    python -m mito_train.training.train_board_ocr --mode smoke --limit 100 --epochs 5

Single GPU (full):
    python -m mito_train.training.train_board_ocr --mode full --epochs 20

Multi-GPU (single node, N GPUs) via torchrun:
    torchrun --standalone --nproc_per_node=N \\
        -m mito_train.training.train_board_ocr --mode full --epochs 20

The same script handles both launch modes: `setup_distributed()` no-ops outside
torchrun, so rank/world detection, DDP wrapping, sampler shard rotation, and
metric all_reduce all activate only when actually distributed.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version as _pkg_version
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from mito_train.datasets import CaptureDataset, build_transform
from mito_train.datasets.sfen_utils import parse_sfen
from mito_train.models import BoardOCR
from mito_train.training.dist_utils import (
    all_reduce_mean,
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    is_main,
    resolve_device,
    setup_distributed,
    teardown_distributed,
)
from mito_train.training.train_piece import _init_wandb


def _unwrap(model: nn.Module) -> nn.Module:
    """Strip DDP and torch.compile wrappers to reach the underlying module.

    Needed when saving/loading state_dict: DDP prefixes with `module.` and
    compile prefixes with `_orig_mod.` — unwrapping both makes checkpoints
    portable across launch modes.
    """
    m = model.module if isinstance(model, DDP) else model
    return getattr(m, "_orig_mod", m)


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


METRIC_KEYS = (
    "board/cell_acc", "board/full_acc",
    "hand/slot_acc", "hand/full_acc", "sfen/full_acc",
)


def compute_metrics(
    board_logits: torch.Tensor,
    hand_logits: torch.Tensor,
    board_target: torch.Tensor,
    hand_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-batch accuracy metrics as 0-dim GPU tensors.

    Returns tensors (not floats) so the hot loop can accumulate on-device
    without forcing a CPU<->GPU sync every step; call .item() only when
    actually logging/printing.
    """
    with torch.no_grad():
        board_pred = board_logits.argmax(dim=1)  # (B, 9, 9)
        board_ok = board_pred == board_target
        board_cell_acc = board_ok.float().mean()
        # Fraction of images where all 81 cells are correct
        board_all = board_ok.all(dim=(1, 2))
        board_full = board_all.float().mean()

        hand_pred = hand_logits.argmax(dim=-1)  # (B, 14)
        hand_ok = hand_pred == hand_target
        hand_slot_acc = hand_ok.float().mean()
        hand_all = hand_ok.all(dim=1)
        hand_full = hand_all.float().mean()

        sfen_full = (board_all & hand_all).float().mean()
    return {
        "board/cell_acc": board_cell_acc,
        "board/full_acc": board_full,
        "hand/slot_acc": hand_slot_acc,
        "hand/full_acc": hand_full,
        "sfen/full_acc": sfen_full,
    }


def run(args: argparse.Namespace) -> None:
    # torchrun sets WORLD_SIZE/RANK/LOCAL_RANK; single-process runs skip this cleanly.
    setup_distributed()
    device = resolve_device()
    use_cuda = device.startswith("cuda")
    world_size = get_world_size()
    ddp = is_distributed()

    def log_main(msg: str) -> None:
        if is_main():
            print(msg)

    if use_cuda:
        # Input size is fixed (image_size x image_size) -> let cuDNN autotune kernels.
        torch.backends.cudnn.benchmark = True
        # TF32 matmuls on Ampere+: big speedup, negligible precision impact here.
        torch.set_float32_matmul_precision("high")

    # Mixed precision: bf16 on Ampere+ (no scaler needed), fp16 + GradScaler otherwise.
    amp_dtype = (
        torch.bfloat16
        if use_cuda and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_cuda and amp_dtype == torch.float16,
    )
    log_main(
        f"[board_ocr] device={device} mode={args.mode} backbone={args.backbone} "
        f"amp={amp_dtype if use_cuda else 'off'} "
        f"world_size={world_size} rank={get_rank()}"
    )

    train_tf = build_transform("train", image_size=args.image_size)
    val_tf = build_transform("val", image_size=args.image_size)

    limit_train = args.limit if args.mode == "smoke" else None
    limit_val = min(args.limit, 20) if args.mode == "smoke" else None

    if args.hf_repo_id:
        # HF-hosted parquet dataset (embedded PNG bytes). First run downloads
        # the shards into HF cache (~21GB); subsequent runs are cache-hits.
        from mito_train.datasets import HFCaptureDataset
        log_main(f"[board_ocr] loading from HF repo: {args.hf_repo_id}")
        preload_kwargs = dict(
            preload=args.preload,
            preload_image_size=args.image_size,
            preload_workers=max(args.num_workers, 8),
        )
        train_ds = HFCaptureDataset(
            repo_id=args.hf_repo_id, split="train",
            transform=train_tf, limit=limit_train,
            **preload_kwargs,
        )
        val_ds = HFCaptureDataset(
            repo_id=args.hf_repo_id, split="val",
            transform=val_tf, limit=limit_val,
            device_index=0,  # deterministic device pick for stable val curves
            **preload_kwargs,
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
    log_main(f"[board_ocr] train={len(train_ds)} val={len(val_ds)}")

    # DistributedSampler shards the dataset across ranks and re-shuffles per
    # epoch (via set_epoch). Its shuffle replaces DataLoader's, so we pass
    # shuffle=False when a sampler is used.
    train_sampler = (
        DistributedSampler(train_ds, shuffle=True, drop_last=False) if ddp else None
    )
    val_sampler = (
        DistributedSampler(val_ds, shuffle=False, drop_last=False) if ddp else None
    )
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        shuffle=False, **loader_kwargs,
    )

    model = BoardOCR(backbone=args.backbone, pretrained=args.pretrained).to(device)
    if use_cuda:
        # NHWC layout: faster conv kernels under cuDNN/TensorCores.
        model = model.to(memory_format=torch.channels_last)
    if ddp:
        # Sync BatchNorm stats across ranks so BN-heavy backbones (mobilenet,
        # efficientnet) don't diverge on per-rank stats. Harmless for LayerNorm
        # backbones (convnext) since no BN layers get replaced.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[get_local_rank()])
    if args.compile:
        model = torch.compile(model)
        log_main("[board_ocr] torch.compile enabled (first steps will be slow while compiling)")
    n_params = sum(p.numel() for p in _unwrap(model).parameters())
    log_main(f"[board_ocr] {args.backbone} params={n_params:,}")

    if args.hand_class_weight:
        hand_class_weight = compute_hand_class_weights(
            train_ds.entries,
            num_classes=_unwrap(model).hand_max,
            clip_min=args.class_weight_clip_min,
            clip_max=args.class_weight_clip_max,
        ).to(device)
        log_main(
            "[board_ocr] hand class weights (sqrt+clip): "
            + ", ".join(f"{i}:{w:.2f}" for i, w in enumerate(hand_class_weight.tolist()))
        )
    else:
        hand_class_weight = None

    # fused=True runs the whole AdamW update in one CUDA kernel per dtype group.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4, fused=use_cuda,
    )

    if is_main():
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    resumed_from: str | None = None
    resumed_wandb_id: str | None = None
    if args.resume is not None:
        ckpt_path = args.ckpt_dir / "latest.pt" if str(args.resume) == "latest" else args.resume
        log_main(f"[board_ocr] resume from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        _unwrap(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        resumed_from = str(ckpt_path)
        resumed_wandb_id = ckpt.get("wandb_run_id")
        log_main(f"[board_ocr] resumed at epoch={start_epoch} (ckpt was epoch {ckpt['epoch']})")
        if resumed_wandb_id:
            log_main(f"[board_ocr] will resume wandb run id={resumed_wandb_id}")

    if args.wandb_run_id:
        resumed_wandb_id = args.wandb_run_id
        log_main(f"[board_ocr] wandb run id overridden by CLI: {resumed_wandb_id}")

    # Only rank 0 talks to W&B; other ranks keep wandb_run=None and log nothing.
    run_name = args.backbone
    wandb_project = f"mito-train-board-ocr-w{args.image_size}-v{_pkg_version('mito-train')}"
    wandb_run = _init_wandb(
        project=wandb_project,
        run_name=run_name,
        run_id=resumed_wandb_id,
        resume="allow" if resumed_wandb_id else None,
        config={
            "mode": args.mode,
            "backbone": args.backbone,
            "pretrained": args.pretrained,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "effective_batch_size": args.batch_size * world_size,
            "world_size": world_size,
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
    ) if is_main() else None

    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        # DistributedSampler.set_epoch(epoch) rotates the shuffle seed per epoch
        # so shards see different orderings; skipping it makes every epoch
        # identical inside a single training run.
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        # Accumulate losses/metrics as 0-dim tensors on-device; .item() forces a
        # GPU sync, so we only touch host values every --log-every steps and at
        # epoch end. This keeps the CUDA queue full between steps.
        running_loss = torch.zeros((), device=device)
        running_board_loss = torch.zeros((), device=device)
        running_hand_loss = torch.zeros((), device=device)
        running_metrics = {k: torch.zeros((), device=device) for k in METRIC_KEYS}
        n_batches = 0
        pbar = tqdm(
            train_loader, desc=f"epoch {epoch:3d}/{args.epochs}",
            dynamic_ncols=True, leave=False, disable=not is_main(),
        )
        for img, board, hand in pbar:
            img = img.to(device, non_blocking=True)
            if use_cuda:
                img = img.contiguous(memory_format=torch.channels_last)
            board = board.to(device, non_blocking=True)
            hand = hand.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
                board_logits, hand_logits = model(img)
                total, bl, hl = compute_loss(
                    board_logits, hand_logits, board, hand,
                    hand_weight=args.hand_weight,
                    hand_class_weight=hand_class_weight,
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += total.detach()
            running_board_loss += bl.detach()
            running_hand_loss += hl.detach()
            m = compute_metrics(board_logits, hand_logits, board, hand)
            for k, v in m.items():
                running_metrics[k] += v
            n_batches += 1
            global_step += 1
            if global_step % args.log_every == 0:
                # Single sync point: pull host values for tqdm + W&B together.
                pbar.set_postfix(
                    loss=f"{running_loss.item() / n_batches:.3f}",
                    cell=f"{running_metrics['board/cell_acc'].item() / n_batches:.3f}",
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "step/loss": total.item(),
                        "step/board_loss": bl.item(),
                        "step/hand_loss": hl.item(),
                        "step/board/cell_acc": m["board/cell_acc"].item(),
                        "step/hand/slot_acc": m["hand/slot_acc"].item(),
                        "step/global_step": global_step,
                        "step/epoch_frac": epoch - 1 + n_batches / len(train_loader),
                    })

        # Reduce per-rank means to a single global mean per metric before
        # printing / logging. DistributedSampler shards are equal-size so an
        # unweighted mean is unbiased.
        avg = {
            k: all_reduce_mean(t / n_batches).item()
            for k, t in running_metrics.items()
        }
        avg_loss = all_reduce_mean(running_loss / n_batches).item()
        avg_bl = all_reduce_mean(running_board_loss / n_batches).item()
        avg_hl = all_reduce_mean(running_hand_loss / n_batches).item()

        log_main(
            f"[board_ocr] epoch={epoch:3d} "
            f"loss={avg_loss:.4f} (board={avg_bl:.4f} hand={avg_hl:.4f}) "
            f"cell_acc={avg['board/cell_acc']:.3f} "
            f"sfen_acc={avg['sfen/full_acc']:.3f}"
        )

        wandb_log = {
            "train/loss": avg_loss,
            "train/board_loss": avg_bl,
            "train/hand_loss": avg_hl,
            "train/epoch": epoch,
        }
        wandb_log.update({f"train/{k}": v for k, v in avg.items()})

        # Simple val (in smoke mode this is only a same-distribution sanity check against train)
        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            val_metrics = {k: torch.zeros((), device=device) for k in METRIC_KEYS}
            n_val_batches = 0
            with torch.no_grad():
                val_pbar = tqdm(
                    val_loader, desc=f"  val {epoch:3d}",
                    dynamic_ncols=True, leave=False, disable=not is_main(),
                )
                for img, board, hand in val_pbar:
                    img = img.to(device, non_blocking=True)
                    if use_cuda:
                        img = img.contiguous(memory_format=torch.channels_last)
                    board = board.to(device, non_blocking=True)
                    hand = hand.to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
                        bl_out, hl_out = model(img)
                    m = compute_metrics(bl_out, hl_out, board, hand)
                    for k, v in m.items():
                        val_metrics[k] += v
                    n_val_batches += 1
            if n_val_batches > 0:
                v_avg = {
                    k: all_reduce_mean(t / n_val_batches).item()
                    for k, t in val_metrics.items()
                }
                log_main(
                    f"[board_ocr]           val cell_acc={v_avg['board/cell_acc']:.3f} "
                    f"sfen_acc={v_avg['sfen/full_acc']:.3f}"
                )
                wandb_log.update({f"val/{k}": v for k, v in v_avg.items()})

        if wandb_run is not None:
            wandb_run.log(wandb_log)

        # Checkpoint save is main-only to avoid concurrent writes clobbering
        # each other; the state_dict is identical across ranks after all_reduce
        # in the backward pass (DDP guarantee).
        if is_main():
            ckpt_payload = {
                "epoch": epoch,
                # Unwrap DDP and torch.compile so ckpt keys stay stable across
                # launch modes (single-GPU or torchrun, --compile or not).
                "model": _unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
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
    if args.mode == "smoke" and is_main():
        final_cell = avg["board/cell_acc"]
        if final_cell < 0.6:
            print(f"[board_ocr] WARNING: did not overfit (cell_acc={final_cell:.3f}).")
        else:
            print(f"[board_ocr] smoke OK (cell_acc={final_cell:.3f}). Design pipeline verified.")

    teardown_distributed()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--backbone", default="mobilenet_v3_small",
                   choices=[
                       "mobilenet_v3_small",
                       "mobilenet_v3_large",
                       "efficientnet_b0",
                       "efficientnet_b1",
                       "convnext_atto",
                       "convnext_femto",
                       "convnext_pico",
                       "convnext_nano",
                       "convnext_tiny",
                   ])
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
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="DataLoader prefetch queue depth per worker.")
    p.add_argument("--preload", action="store_true", default=False,
                   help="Decode + resize every image once at init and hold in RAM. "
                        "Removes WebP decode + resize from the hot loop entirely. "
                        "Costs ~image_size^2*3*N*num_devices bytes "
                        "(~10 GB for 18k rows x 4 devices at 224).")
    p.add_argument("--compile", action="store_true", default=False,
                   help="torch.compile the model. Worth measuring for full runs; "
                        "compile overhead usually not worth it for smoke runs.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hand-mode", choices=["classification", "regression"],
                   default="classification",
                   help="classification: 14x19 logits + CE. regression: 14 scalars + SmoothL1.")
    p.add_argument("--hand-weight", type=float, default=1.0)
    p.add_argument("--hand-class-weight", action="store_true", default=True,
                   help="Use sqrt(1/freq) class weights on hand CE (ignored under --hand-mode=regression).")
    p.add_argument("--no-hand-class-weight", dest="hand_class_weight", action="store_false")
    p.add_argument("--class-weight-clip-min", type=float, default=0.5)
    p.add_argument("--class-weight-clip-max", type=float, default=10.0)
    p.add_argument("--val-every", type=int, default=2)
    p.add_argument("--log-every", type=int, default=20,
                   help="Log per-step train metrics to W&B every N batches.")
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
