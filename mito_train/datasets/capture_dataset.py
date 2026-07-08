"""Dataset of piyo-shogi captures (board + hand, 1 image) paired with sfen labels.

Each line of the manifest jsonl is {"hash": ..., "sfen": ...}. Actual images are
expected to be flat-laid out under `data/captures/{subdir}/{hash}.png`.

`__getitem__` returns (image_tensor, board_tensor, hand_tensor):
- image_tensor: (3, H, W) float32, normalized
- board_tensor: (9, 9) long, 29-class ID per cell
- hand_tensor:  (14,) long, piece count per slot (0..18)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Literal

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from .sfen_utils import parse_sfen

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(
    mode: Literal["train", "val"],
    image_size: int = 384,
) -> A.Compose:
    """Return the augmentation pipeline.

    Training data already frames the board + piece stands tightly, so cropping
    would clip the piece-stand region (especially the pawn slots at the edges).
    We use LongestMaxSize + PadIfNeeded to guarantee full board+stands survive,
    then Affine for position/scale jitter that only shifts within padding.

    Handled variations:
      - Screenshot resolution differences causing size / position shifts -> Affine (translate + scale)
      - SNS re-encoding (JPEG) -> ImageCompression
      - Low-resolution devices -> Downscale
      - Device-side color/brightness differences -> RandomBrightnessContrast, HueSaturationValue
      - Screen capture noise -> GaussNoise
    """
    if mode == "train":
        return A.Compose([
            # Fit longest side to image_size, pad shorter side to square with black bars.
            # This preserves the entire board + both piece stands without clipping.
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size, min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT, fill=0,
            ),
            # Simulate position/scale jitter within the padded canvas (no content loss).
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.0),
                border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.7,
            ),
            # Color / brightness / saturation (device differences)
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.3),
            # Noise / degradation
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
            A.ImageCompression(quality_range=(50, 90), p=0.7),
            A.Downscale(scale_range=(0.5, 0.9), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size, min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT, fill=0,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


class CaptureDataset(Dataset):
    """Dataset of capture images + sfen labels."""

    def __init__(
        self,
        manifest_path: Path,
        image_root: Path,
        transform: Callable | A.Compose | None = None,
        limit: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        manifest_path: path to train.jsonl / val.jsonl
        image_root:    directory holding the images (e.g. data/captures/d0)
        transform:     Albumentations Compose or callable. When None, defaults to val transform.
        limit:         use only the first N entries (for smoke tests)
        """
        self.image_root = Path(image_root)
        self.transform = transform if transform is not None else build_transform("val")

        entries: list[dict] = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        if limit is not None:
            entries = entries[:limit]
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entry = self.entries[idx]
        h = entry["hash"]
        image_path = self.image_root / f"{h}.png"

        img = Image.open(image_path)
        # RGBA -> RGB (composited on white background)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert("RGB")
        img_np = np.array(img)  # HxWxC uint8

        parsed = parse_sfen(entry["sfen"])
        board = torch.tensor(parsed.board, dtype=torch.long)  # (9,9)
        hand = torch.tensor(parsed.hand, dtype=torch.long)    # (14,)

        if isinstance(self.transform, A.Compose):
            out = self.transform(image=img_np)
            image_tensor = out["image"]  # (C,H,W) via ToTensorV2
        else:
            image_tensor = self.transform(img)  # also accept a PIL-based callable

        return image_tensor, board, hand
