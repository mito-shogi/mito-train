"""Dataset for board-detector training: (screenshot, bbox) pairs.

Sources:
    <image_root>/{device}/{hash}.webp        raw device screenshots
    data/detector/train.jsonl / detector_val.jsonl
        one row per sample:
            {"path": "detector_train/{device}/{hash}.webp",
             "device": "{device}", "hash": "{hash}"}
    data/test/device_bboxes.json             per-device ShogiBoardView bbox (retina px)

Returns per __getitem__:
    image: (3, S, S) float32, ImageNet-normalized
    bbox:  (4,)      float32, normalized (x1, y1, x2, y2) in [0, 1] w.r.t. S x S

Augmentation strategy:
    - Random Scale + Pad on a black canvas so the board can shrink / off-center
      arbitrarily -> forces the network to learn "where is the board" instead
      of memorizing the fixed frame.
    - RandomCrop (larger than input then crop) simulates users cropping around
      the board when reposting on SNS.
    - Perspective / Rotate simulate photo-of-screen and hand-held tilt.
    - JPEG / Downscale / noise / brightness for Twitter degradation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_detector_transform(
    mode: Literal["train", "val"],
    image_size: int = 384,
) -> A.Compose:
    if mode == "train":
        return A.Compose(
            [
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(
                    min_height=image_size, min_width=image_size,
                    border_mode=cv2.BORDER_CONSTANT, fill=0,
                ),
                A.Affine(
                    scale=(0.7, 1.05),
                    translate_percent=(-0.08, 0.08),
                    rotate=(-6, 6),
                    shear=(-2, 2),
                    fit_output=False,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.9,
                ),
                A.Perspective(scale=(0.02, 0.05), p=0.3),
                A.RandomSizedBBoxSafeCrop(
                    height=image_size, width=image_size, erosion_rate=0.15, p=0.5
                ),
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(
                    min_height=image_size, min_width=image_size,
                    border_mode=cv2.BORDER_CONSTANT, fill=0,
                ),
                A.ImageCompression(quality_range=(50, 95), p=0.7),
                A.Downscale(scale_range=(0.5, 0.9), p=0.4),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["labels"],
                min_visibility=0.5, filter_invalid_bboxes=True,
            ),
        )
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size, min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT, fill=0,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"], filter_invalid_bboxes=False,
        ),
    )


class DetectorDataset(Dataset):
    """jsonl-driven dataset of (screenshot, bbox) samples."""

    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        bboxes_json: Path,
        transform: A.Compose,
    ) -> None:
        """
        Parameters
        ----------
        manifest_path: path to detector_train.jsonl / detector_val.jsonl
        data_root:     directory that manifest paths are resolved against
                       (e.g. `data/`, so a manifest entry
                       "detector_train/{device}/{hash}.webp" resolves to
                       `data/detector/{device}/{hash}.webp`).
        bboxes_json:   device_bboxes.json giving per-device board bbox in px.
        transform:     Albumentations Compose with BboxParams.
        """
        self.data_root = Path(data_root)
        self.transform = transform
        with open(bboxes_json) as f:
            cfg = json.load(f)
        cfg_devices: dict = cfg["devices"]

        entries: list[dict] = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        self.entries = entries

        self.bboxes: dict[str, tuple[int, int, int, int]] = {}
        for entry in entries:
            dev = entry["device"]
            if dev in self.bboxes:
                continue
            if dev not in cfg_devices:
                raise KeyError(f"device_bboxes.json missing entry for {dev!r}")
            b = cfg_devices[dev]["board_view_px"]
            self.bboxes[dev] = (b["x1"], b["y1"], b["x2"], b["y2"])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.entries[idx]
        dev = entry["device"]
        path = self.data_root / entry["path"]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"detector image not readable: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = self.bboxes[dev]
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
            # Aug wiped the board out of frame; use a sentinel and let train loop mask it.
            bbox = torch.full((4,), -1.0, dtype=torch.float32)
        return image, bbox
