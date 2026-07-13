"""Autotune training hyperparams from detected hardware.

Prints shell-eval-able env var assignments to stdout — the calling script
does `eval "$(python scripts/autotune.py ...)"` to import them.

Emitted vars:
  NUM_WORKERS       DataLoader workers per training run
  BATCH_SIZE        per-GPU batch size
  PREFETCH_FACTOR   DataLoader prefetch depth per worker
  NGPUS_DETECTED    number of CUDA devices visible
  VRAM_GB_PER_GPU   VRAM on the first GPU (all assumed same)
  CPU_COUNT         host CPU count

Sizing rules (deliberately conservative — tuner should adjust up if VRAM
util stays low mid-run):
  - batch = ~ vram_gb * scale / footprint(backbone, image_size)
    where footprint approximates activation + optimizer memory per sample
  - workers = min(cpu_count / concurrent_runs, 16)
    with concurrent_runs being 1 in sequential/DDP modes, NGPUS in parallel mode
"""
from __future__ import annotations
import argparse
import multiprocessing
import sys


# Per-sample VRAM footprint at 224x224 batch=1 (GB), rough empirical estimate.
# Scales quadratically with image_size (H*W).
BACKBONE_FOOTPRINT_GB = {
    "mobilenet_v3_small": 0.020,
    "mobilenet_v3_large": 0.035,
    "efficientnet_b0":    0.045,
    "efficientnet_b1":    0.055,
    "convnext_atto":      0.040,
    "convnext_femto":     0.050,
    "convnext_pico":      0.070,
    "convnext_nano":      0.090,
    "convnext_tiny":      0.130,
}


def detect_gpu() -> tuple[int, float]:
    try:
        import torch
    except ImportError:
        return 0, 0.0
    if not torch.cuda.is_available():
        return 0, 0.0
    n = torch.cuda.device_count()
    if n == 0:
        return 0, 0.0
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    return n, vram_gb


def suggest_batch(backbone: str, image_size: int, vram_gb: float,
                  concurrent_on_gpu: int = 1) -> int:
    """Round to a nearby power of 2 for kernel friendliness."""
    if vram_gb <= 0:
        return 32  # CPU fallback
    footprint = BACKBONE_FOOTPRINT_GB.get(backbone, 0.10)
    footprint *= (image_size / 224) ** 2
    # ~60% of VRAM budgeted for activations+batch (rest: model, optimizer, preload)
    usable_vram = vram_gb * 0.60 / concurrent_on_gpu
    raw = usable_vram / footprint
    # Snap to nearest power of two below raw, clamped to a sane range.
    b = 32
    while b * 2 <= raw and b < 1024:
        b *= 2
    return max(16, b)


def suggest_workers(cpu_count: int, concurrent_runs: int) -> int:
    """Leave a couple cores free for the main + DataLoader main; cap at 16."""
    per_run = max(1, (cpu_count - 2) // max(1, concurrent_runs))
    return min(per_run, 16)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True,
                   help="Backbone name; used to look up per-sample footprint.")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--mode", choices=["sequential", "ddp", "parallel"],
                   default="sequential",
                   help="Launch mode. Affects workers division and concurrent-per-GPU count.")
    p.add_argument("--nprocs", type=int, default=None,
                   help="For DDP: GPUs used per run. For parallel: GPUs sharing the sweep. "
                        "Default = auto-detected GPU count.")
    args = p.parse_args()

    ngpus, vram_gb = detect_gpu()
    cpu_count = multiprocessing.cpu_count()

    if args.mode == "parallel":
        concurrent_runs = args.nprocs or ngpus or 1
        concurrent_on_gpu = 1  # one backbone per GPU
    elif args.mode == "ddp":
        concurrent_runs = 1
        concurrent_on_gpu = 1  # single backbone across many GPUs
    else:
        concurrent_runs = 1
        concurrent_on_gpu = 1

    batch = suggest_batch(args.backbone, args.image_size, vram_gb, concurrent_on_gpu)
    workers = suggest_workers(cpu_count, concurrent_runs)
    prefetch = 4 if workers <= 4 else 8

    print(f"NUM_WORKERS={workers}")
    print(f"BATCH_SIZE={batch}")
    print(f"PREFETCH_FACTOR={prefetch}")
    print(f"NGPUS_DETECTED={ngpus}")
    print(f"VRAM_GB_PER_GPU={vram_gb:.1f}")
    print(f"CPU_COUNT={cpu_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
