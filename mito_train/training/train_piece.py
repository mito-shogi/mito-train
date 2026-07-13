"""Training entry point for the piece-classifier.

Usage:
    # Smoke (overfit pieces/k1): sanity-checks device / W&B / training loop wiring
    python -m mito_train.training.train_piece --mode smoke

    # Resume from the last checkpoint in --ckpt-dir (bumps --epochs to keep going)
    python -m mito_train.training.train_piece --mode smoke --resume latest --epochs 80

    # Real training (via manifest): not yet implemented
    python -m mito_train.training.train_piece --mode manifest --data ./data/piyo-train

The manifest mode currently lacks a DataLoader / training loop (waiting on the piyo-hook side to deliver a manifest).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from mito_train.datasets import PiecesDataset, PiyoDataset
from mito_train.models.piece_classifier import PieceClassifier


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _init_wandb(
    project: str,
    run_name: str,
    config: dict,
    run_id: str | None = None,
    resume: str | None = None,
):
    """Initialize W&B. Returns None (skipped) when WANDB_API_KEY is unset.

    The self-hosted instance (wandb.tkgstrator.work) sits behind Cloudflare Access,
    so we inject the CF Access headers into the SDK.

    Passing run_id + resume="allow" continues an existing W&B run rather than
    creating a new one — used to keep curves contiguous across ckpt resumes.
    """
    if not os.environ.get("WANDB_API_KEY"):
        print("[wandb] WANDB_API_KEY not set, skipping.")
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] wandb is not installed (`uv sync --extra experiment`), skipping.")
        return None

    settings_kwargs = {}
    cf_id = os.environ.get("CF_ACCESS_CLIENT_ID")
    cf_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET")
    if cf_id and cf_secret:
        settings_kwargs["_extra_http_headers"] = {
            "CF-Access-Client-Id": cf_id,
            "CF-Access-Client-Secret": cf_secret,
        }
        print("[wandb] injected CF Access headers.")

    settings = wandb.Settings(**settings_kwargs) if settings_kwargs else None
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    init_kwargs = dict(project=project, name=run_name, config=config, settings=settings, dir=str(log_dir))
    if run_id is not None:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = resume or "allow"
    run = wandb.init(**init_kwargs)
    if run_id is not None:
        print(f"[wandb] resumed run: {run.url}")
    else:
        print(f"[wandb] run initialized: {run.url}")
    return run


def run_smoke(args: argparse.Namespace) -> None:
    """Smoke test that overfits on pieces/k1.

    (1) forward/backward runs
    (2) train_acc reaches close to 1.0 (model / loss / optimizer wiring works)
    (3) metrics arrive at W&B (env vars and CF Access wiring works)
    """
    device = get_device()
    print(f"[smoke] device={device} pieces={args.pieces} epochs={args.epochs}")

    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
    ])
    ds = PiecesDataset(root=args.pieces, themes=args.themes, transform=transform)
    print(f"[smoke] dataset len={len(ds)} num_classes={ds.num_classes}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = PieceClassifier(num_classes=ds.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] PieceClassifier params={n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    resumed_from: str | None = None
    resumed_wandb_id: str | None = None
    if args.resume is not None:
        ckpt_path = args.ckpt_dir / "latest.pt" if str(args.resume) == "latest" else args.resume
        print(f"[smoke] resume from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        resumed_from = str(ckpt_path)
        resumed_wandb_id = ckpt.get("wandb_run_id")
        print(f"[smoke] resumed at epoch={start_epoch} (ckpt was epoch {ckpt['epoch']})")
        if resumed_wandb_id:
            print(f"[smoke] will resume wandb run id={resumed_wandb_id}")

    if args.wandb_run_id:
        resumed_wandb_id = args.wandb_run_id
        print(f"[smoke] wandb run id overridden by CLI: {resumed_wandb_id}")

    run = _init_wandb(
        project="mito-train-smoke",
        run_name="piece-smoke",
        run_id=resumed_wandb_id,
        resume="allow" if resumed_wandb_id else None,
        config={
            "mode": "smoke",
            "themes": args.themes,
            "input_size": args.input_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "start_epoch": start_epoch,
            "resumed_from": resumed_from,
            "num_classes": ds.num_classes,
            "n_params": n_params,
            "device": device,
        },
    )

    avg_loss = 0.0
    acc = 0.0
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
        avg_loss = total_loss / total
        acc = correct / total
        if epoch % args.log_every == 0 or epoch in (start_epoch, args.epochs):
            print(f"[smoke] epoch={epoch:3d} loss={avg_loss:.4f} acc={acc:.4f}")
        if run is not None:
            run.log({"train/loss": avg_loss, "train/acc": acc, "train/epoch": epoch})

        ckpt_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "num_classes": ds.num_classes,
            "input_size": args.input_size,
            "wandb_run_id": run.id if run is not None else None,
        }
        torch.save(ckpt_payload, args.ckpt_dir / "latest.pt")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(ckpt_payload, args.ckpt_dir / f"epoch-{epoch:03d}.pt")

    print(f"[smoke] final acc={acc:.4f} loss={avg_loss:.4f}")
    if acc < 0.95:
        print("[smoke] WARNING: overfit did not converge. Suspect model / loss / lr.")
    else:
        print("[smoke] overfit OK. Pipeline sanity check passed.")

    if run is not None:
        run.finish()


def run_manifest(args: argparse.Namespace) -> None:
    """Real training via manifest (not yet implemented)."""
    device = get_device()
    print(f"[manifest] device={device} data={args.data} epochs={args.epochs}")

    model = PieceClassifier().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[manifest] PieceClassifier params={n_params:,}")

    manifest = args.data / "manifest-train.jsonl"
    if not manifest.exists():
        print(f"[manifest] WARNING: {manifest} missing. Waiting for delivery from piyo-hook.")
        print("[manifest] Only model construction verified; skipping training.")
        return

    train_ds = PiyoDataset(data_root=args.data, split="train", transform=None)
    print(f"[manifest] train entries={len(train_ds)}")
    # TODO: implement DataLoader + training loop


def main() -> None:
    # Pull WANDB_API_KEY / CF_* / HF_TOKEN out of .env into os.environ before
    # any wandb or HF call resolves credentials. No-op if .env is missing.
    # override=True so devcontainer.json's ${localEnv:...} forwards that expand
    # to an empty string on hosts without those vars don't win over .env.
    load_dotenv(override=True)
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "manifest"], default="smoke")
    # smoke-mode args
    p.add_argument("--pieces", type=Path, default=Path("./data/pieces"))
    p.add_argument("--themes", nargs="+", default=["k1"])
    p.add_argument("--input-size", type=int, default=32)
    p.add_argument("--log-every", type=int, default=5)
    # manifest-mode args
    p.add_argument("--data", type=Path, default=Path("./data/piyo-train"))
    # shared
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=Path, default=Path("./runs/piece-cnn"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("./runs/piece-cnn"),
                   help="Directory for latest.pt / epoch-{N}.pt checkpoints.")
    p.add_argument("--save-every", type=int, default=10,
                   help="Interval for saving epoch-{N}.pt snapshots. latest.pt is saved every epoch.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Checkpoint path. Pass 'latest' to load --ckpt-dir/latest.pt.")
    p.add_argument("--wandb-run-id", type=str, default=None,
                   help="Force resume this W&B run id (overrides ckpt's stored id).")
    args = p.parse_args()

    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_manifest(args)


if __name__ == "__main__":
    main()
