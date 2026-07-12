"""HF Hub-backed paired-format dataset for BoardOCR training.

Loads `ultemica/piyoshogi` `ocr_paired` config where each row is
    {"sfen": str, "hash": str, "type": str,
     "images": list<PIL.Image>, "devices": list<str>}

with 4 images per row (one per device: iPhone10,1 / iPhone11,8 / iPhone15,4 / iPad14,10).

The training pipeline consumes single (image, board, hand) triples, so this class
flattens paired rows into per-device examples by default: N paired rows → 4N examples.

Usage (drop-in replacement for HFCaptureDataset):
    from mito_train.datasets import HFPairedDataset, build_transform
    train_ds = HFPairedDataset(
        repo_id="ultemica/piyoshogi",
        split="train",
        transform=build_transform("train", image_size=384),
    )

First call downloads the parquet shards into `~/.cache/huggingface/datasets/`.
Subsequent runs are cache hits.
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


class HFPairedDataset(Dataset):
    """Paired-config BoardOCR dataset backed by HF Hub.

    Each paired row is expanded to `len(devices)` examples so that the model
    still sees (image, board, hand) triples. `entries` and `__len__` reflect
    the expanded count (e.g. 18000 paired rows → 72000 examples for 4 devices).

    Parameters
    ----------
    repo_id:      HF dataset repo (default "ultemica/piyoshogi").
    split:        "train" or "val".
    config:       HF config name (default "ocr_paired").
    transform:    Albumentations Compose or callable. Defaults to val transform.
    devices:      Filter to specific device identifiers (e.g. ["iPhone15,4"]).
                  None = keep all devices found in each row.
    limit:        Truncate to first N paired rows (before device expansion).
    cache_dir:    Override HF cache location.
    """

    def __init__(
        self,
        repo_id: str = "ultemica/piyoshogi",
        split: str = "train",
        config: str = "ocr_paired",
        transform: Callable | A.Compose | None = None,
        devices: list[str] | None = None,
        limit: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        from datasets import load_dataset

        self.transform = transform if transform is not None else build_transform("val")
        self.device_filter = set(devices) if devices else None

        ds = load_dataset(repo_id, config, split=split, cache_dir=cache_dir)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        self.ds = ds

        # Build a flat index: (row_idx, device_idx_within_row) for each expanded example.
        # Reads only the `devices` column (fast; ~few MB even for 18k rows) so we don't
        # decode any images here.
        devices_col = ds["devices"]
        index: list[tuple[int, int]] = []
        for row_i, devs in enumerate(devices_col):
            for dev_i, dev in enumerate(devs):
                if self.device_filter is not None and dev not in self.device_filter:
                    continue
                index.append((row_i, dev_i))
        self._index = index

    @property
    def entries(self):
        """Sfen-only entries list (one per expanded example), for class-weight calc."""
        # Pull only the sfen column; each SFEN gets repeated by number of kept devices per row.
        sfens = self.ds["sfen"]
        # Count kept device slots per row via the built index for correct expansion.
        per_row: dict[int, int] = {}
        for row_i, _ in self._index:
            per_row[row_i] = per_row.get(row_i, 0) + 1
        out = []
        for row_i, sfen in enumerate(sfens):
            out.extend({"sfen": sfen} for _ in range(per_row.get(row_i, 0)))
        return out

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_i, dev_i = self._index[idx]
        row = self.ds[row_i]
        img = row["images"][dev_i]  # PIL.Image (decoded by HF Image feature)

        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img_np = np.array(img)

        parsed = parse_sfen(row["sfen"])
        board = torch.tensor(parsed.board, dtype=torch.long)  # (9, 9)
        hand = torch.tensor(parsed.hand, dtype=torch.long)    # (14,)

        if isinstance(self.transform, A.Compose):
            out = self.transform(image=img_np)
            image_tensor = out["image"]
        else:
            image_tensor = self.transform(img)

        return image_tensor, board, hand
