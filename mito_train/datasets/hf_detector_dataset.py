"""HF Hub-backed variant of DetectorDataset.

Loads the dataset from a Hugging Face dataset repo where each row is
{"sfen": str, "hash": str, "images": [PIL, ...], "devices": [str, ...],
 "bboxes": [[x1, y1, x2, y2], ...]}. Each row bundles the same SFEN rendered on
multiple devices; we flatten to one (image, bbox) sample per (row, device).

Contract matches DetectorDataset:
    image: (3, S, S) float32, ImageNet-normalized (after transform)
    bbox:  (4,)      float32, normalized (x1, y1, x2, y2) in [0, 1] w.r.t. S,
                     or (-1, -1, -1, -1) sentinel if augmentation removed the
                     board from frame.

Bboxes travel per-sample in the parquet, so `device_bboxes.json` is not needed
on this path (`.bboxes` on the returned instance is kept as a compatibility
mapping for logging only — device -> (x1, y1, x2, y2) of any sample from that
device).
"""
from __future__ import annotations

from typing import Callable

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .detector_dataset import build_detector_transform


class HFDetectorDataset(Dataset):
    """DetectorDataset backed by a Hugging Face Hub dataset repo."""

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        transform: Callable | A.Compose | None = None,
        limit: int | None = None,
        cache_dir: str | None = None,
        config_name: str = "detector_paired",
        image_size: int = 384,
    ) -> None:
        from datasets import Image as HFImage
        from datasets import Sequence, load_dataset

        mode = "train" if split == "train" else "val"
        self.transform = transform if transform is not None else build_detector_transform(mode, image_size)

        ds = load_dataset(repo_id, config_name, split=split, cache_dir=cache_dir)
        ds = ds.cast_column("images", Sequence(HFImage(decode=False)))
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        self.ds = ds

        first = ds[0]
        self.num_devices = len(first["devices"])
        self.bboxes: dict[str, tuple[int, int, int, int]] = {}
        for i, dev in enumerate(first["devices"]):
            b = first["bboxes"][i]
            self.bboxes[dev] = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

    def __len__(self) -> int:
        return len(self.ds) * self.num_devices

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row_idx, dev_idx = divmod(idx, self.num_devices)
        row = self.ds[row_idx]
        entry = row["images"][dev_idx]
        buf = np.frombuffer(entry["bytes"], dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR_RGB)
        h, w = img.shape[:2]
        b = row["bboxes"][dev_idx]
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))

        out = self.transform(image=img, bboxes=[[x1, y1, x2, y2]], labels=[0])
        image = out["image"]
        S = image.shape[-1]
        if out["bboxes"]:
            bx1, by1, bx2, by2 = out["bboxes"][0]
            bbox = torch.tensor(
                [bx1 / S, by1 / S, bx2 / S, by2 / S], dtype=torch.float32
            ).clamp(0.0, 1.0)
        else:
            bbox = torch.full((4,), -1.0, dtype=torch.float32)
        return image, bbox
