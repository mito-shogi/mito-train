"""Small helpers around torch.distributed so single-GPU and DDP share the same
training script.

Usage pattern (see train_board_ocr.py):
    setup_distributed()          # no-op unless launched via torchrun
    if is_main():                # log/save/wandb only on rank 0
        ...
    dataset  ... DistributedSampler(...)
    model    ... DDP(model, device_ids=[local_rank])
    metrics  ... all_reduce_mean(tensor)  # aggregate across ranks
    teardown_distributed()

Detection is env-driven: torchrun exports LOCAL_RANK / RANK / WORLD_SIZE.
When those aren't set the helpers degrade to single-process behavior, so the
same code path runs on the 1-GPU dev box and the 8-GPU A100 node."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main() -> bool:
    return get_rank() == 0


def setup_distributed() -> None:
    """Initialize the NCCL process group and bind this process to its GPU.

    Safe to call unconditionally; it's a no-op unless torchrun set WORLD_SIZE.
    """
    if not is_distributed():
        return
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())


def teardown_distributed() -> None:
    if is_distributed() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """Rendezvous point across ranks. No-op outside DDP."""
    if is_distributed() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across ranks (in place). No-op outside DDP.

    Use this to aggregate per-rank running metrics into a global mean at
    epoch end; each rank saw len(dataset)/world_size samples so a plain
    mean-of-per-rank-means is unweighted-correct only when the sampler
    hands out equal-size shards, which DistributedSampler does by design
    (it drops or pads to make shards equal).
    """
    if is_distributed() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / get_world_size()
    return tensor


def resolve_device() -> str:
    """Return the device string this rank should train on.

    Under DDP each process pins to its LOCAL_RANK GPU; outside DDP fall back
    to whatever accelerator the harness's get_device() picks.
    """
    if is_distributed() and torch.cuda.is_available():
        return f"cuda:{get_local_rank()}"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
