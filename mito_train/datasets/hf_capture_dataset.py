"""HF Hub-backed variant of CaptureDataset.

Loads the dataset from a Hugging Face dataset repo where each row is
{"image": <PIL image>, "sfen": <str>, "hash": <str>} — same schema uploaded by
scripts/upload_to_hf.py.

Usage (in training):
    from mito_train.datasets import HFCaptureDataset, build_transform
    train_ds = HFCaptureDataset(
        repo_id="ultemica/piyoshogi",
        split="train",
        transform=build_transform("train", image_size=288),
    )

First call downloads the parquet shards into `~/.cache/huggingface/datasets/`.
Subsequent runs are cache-hits.
"""
from __future__ import annotations
from typing import Callable

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .capture_dataset import build_transform
from .sfen_utils import parse_sfen


class HFCaptureDataset(Dataset):
    """CaptureDataset backed by a Hugging Face Hub dataset repo."""

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        transform: Callable | A.Compose | None = None,
        limit: int | None = None,
        cache_dir: str | None = None,
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        repo_id:    Hugging Face dataset repo (e.g. "ultemica/piyoshogi")
        split:      "train" or "val"
        transform:  Albumentations Compose or callable. Defaults to val transform.
        limit:      Truncate to first N rows (smoke tests).
        cache_dir:  Override HF cache location. None = ~/.cache/huggingface/datasets
        streaming:  If True, stream from Hub (no local cache). Random access breaks;
                    __getitem__ becomes O(n). Use only for one-shot iteration.
        """
        from datasets import load_dataset

        self.transform = transform if transform is not None else build_transform("val")
        self.streaming = streaming

        ds = load_dataset(
            repo_id,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )
        if limit is not None and not streaming:
            ds = ds.select(range(min(limit, len(ds))))
        self.ds = ds

        if streaming:
            # Streaming datasets are one-shot iterables; expose an iterator only.
            self._length = limit
        else:
            self._length = len(ds)

    @property
    def entries(self):
        """Sfen-only entries list, compatible with compute_hand_class_weights."""
        if self.streaming:
            raise RuntimeError(
                "entries is not available in streaming mode. "
                "Disable streaming to compute class weights."
            )
        # Only pulls the sfen column (fast; ~4MB for 27k rows), no image decode.
        return [{"sfen": s} for s in self.ds["sfen"]]

    def __len__(self) -> int:
        if self._length is None:
            raise RuntimeError("Streaming dataset without --limit has unknown length.")
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.streaming:
            raise RuntimeError("Streaming datasets do not support random access; iterate instead.")
        row = self.ds[idx]
        img = row["image"]  # PIL Image (decoded by HF Image feature)

        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img_np = np.array(img)

        parsed = parse_sfen(row["sfen"])
        board = torch.tensor(parsed.board, dtype=torch.long)  # (9,9)
        hand = torch.tensor(parsed.hand, dtype=torch.long)    # (14,)

        if isinstance(self.transform, A.Compose):
            out = self.transform(image=img_np)
            image_tensor = out["image"]
        else:
            image_tensor = self.transform(img)

        return image_tensor, board, hand
