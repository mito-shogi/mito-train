---
name: train-backbones
description: Launch the BoardOCR backbone sweep. Asks the user for launch mode, image size, epoch count, and optional backbone subset; auto-tunes batch_size/num_workers/prefetch_factor from detected GPU count + VRAM + CPU cores via scripts/autotune.py; then runs scripts/train_backbones.sh with the resolved env vars. Use for the "start / kick off / run the training sweep" ask on this project.
---

# train-backbones

Kicks off training of BoardOCR across one or more backbones. Handles both the
1-GPU dev box (RTX 4070 Ti) and the 8-GPU A100 target uniformly by delegating
to `scripts/train_backbones.sh` (which supports three launch modes) and letting
`scripts/autotune.py` size DataLoader/batch based on the detected hardware.

## What this skill does

1. Ask the user (via `AskUserQuestion`) for the choices they should actually
   own — launch mode, image size, epochs, optionally the backbone subset. Do
   NOT ask for anything that autotune can infer (workers, batch, VRAM, etc.).
2. Detect hardware and derived hyperparams by running
   `uv run python scripts/autotune.py --backbone <bb> --image-size <sz> --mode <mode> [--nprocs <n>]`.
   Parse its `KEY=VALUE` lines. The backbone passed to autotune should be the
   *heaviest* one in the sweep, so batch_size holds for every backbone in the run.
3. Print a compact plan back to the user (mode, backbones, image_size, epochs,
   resolved batch/workers) and then launch the sweep by exporting the env vars
   inline with the shell invocation:
   `EPOCHS=... BATCH_SIZE=... NUM_WORKERS=... PREFETCH_FACTOR=... IMAGE_SIZE=... ./scripts/train_backbones.sh`
4. Stream stdout so the user sees the sweep script's progress log.

## The three launch modes

Present them via a single `AskUserQuestion` with these labels/descriptions so
the user picks based on their goal, not implementation:

- **Sequential (1 backbone × 1 GPU)** — simplest, dev-box friendly. Runs each
  backbone in turn. Env: default.
- **DDP (1 backbone × N GPUs)** — one heavy backbone, split across all GPUs
  via torchrun. Env: `NPROC_PER_NODE=<ngpus>`. Useful for training a single
  chosen model as fast as possible.
- **Parallel (N backbones × 1 GPU each)** — best for a full comparison sweep
  on a multi-GPU node. Each backbone runs on its own GPU concurrently. Env:
  `PARALLEL_GPU=1`. Waves of `NGPUS` at a time.

If autotune reports `NGPUS_DETECTED=1`, only Sequential makes sense — skip the
question and note in the plan why.

## Image size options

- **224** (default) — matches ImageNet pretraining, smallest cache (~10 GB),
  fastest per epoch. Each 9x9 board cell gets ~25 px.
- **288** — ~32 px/cell, moderate cache (~17 GB), ~1.5x epoch time.
- **384** — ~42 px/cell, big cache (~31 GB), ~2.5x epoch time. Best for
  distinguishing fine-detail characters like 成香/成桂.

## Epoch options

Present 30 / 50 / 100 / 200 as anchor choices. Note that first epoch of a
cold-cache run pays the preload build cost (~30-60 s for full dataset).

## Backbones

Default = all 9 (small → large): `mobilenet_v3_small mobilenet_v3_large
convnext_atto convnext_femto efficientnet_b0 convnext_pico efficientnet_b1
convnext_nano convnext_tiny`. Ask only if the user hints at a subset. Pass via
the `BACKBONES` env var, space-separated.

## VRAM cache — not implemented

The user may ask about loading the preload cache into VRAM (an A100 can hold
the whole 10 GB cache easily). This is a known followup, NOT implemented today.
Reasons:

- The current pipeline decodes preload → DRAM → CPU DataLoader workers → GPU.
- Moving the cache to VRAM only pays off if augmentation also runs on GPU,
  because otherwise workers still have to copy back to CPU per sample.
- That requires porting the Albumentations chain to Kornia — a real refactor,
  not a one-line change.

If asked, explain the tradeoff honestly and defer. The DRAM cache is already
near-instant on cache hit (mmap) and DataLoader isn't currently the bottleneck.

## Autotune calling convention

`scripts/autotune.py` prints eval-able env assignments. Call it once with the
*heaviest* backbone in the sweep so batch_size is safe for every backbone
(smaller ones will fit trivially).

Examples:
```bash
# Parallel mode on the 8-GPU node, all backbones:
uv run python scripts/autotune.py --backbone convnext_tiny --image-size 224 --mode parallel

# DDP mode, 8 GPUs, single backbone:
uv run python scripts/autotune.py --backbone convnext_tiny --image-size 224 --mode ddp --nprocs 8
```

## Full launch invocation

Compose the final shell command. Mode → env var mapping:
- sequential: no extra env
- ddp: `NPROC_PER_NODE=${NGPUS_DETECTED}`
- parallel: `PARALLEL_GPU=1`

Then run (adjust the trailing script name if the user wants to change it):

```bash
IMAGE_SIZE=<sz> EPOCHS=<n> BATCH_SIZE=<b> NUM_WORKERS=<w> PREFETCH_FACTOR=<p> \
  [BACKBONES="<subset>"] [PARALLEL_GPU=1 | NPROC_PER_NODE=<n>] \
  ./scripts/train_backbones.sh
```

Show it in the plan so the user can copy/rerun without going through the
skill later.

## Things not to ask about (skill owns these)

- Learning rate — leave the script default (3e-4). If asked, mention linear
  scaling for larger effective batch and that the user can pass `LR=<...>` env.
- Preload — always on (`--preload`). Cache is on-disk mmap; zero cost after
  first build.
- Checkpoint dir / W&B project name — script defaults are fine.
- `RESUME_INCOMPLETE` — leave off unless the user asks to resume; not a
  hyperparameter question.
