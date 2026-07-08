"""Twitter degradation augmentation pipeline (albumentations).

TODO: implement per the degradation spec in docs/piyo-data-generation-spec.md.
- JPEG re-compression (quality sweep)
- Downscale -> upscale
- Mild blur / noise / brightness-contrast jitter
"""
from __future__ import annotations

import albumentations as A


def build_twitter_degradation(train: bool = True) -> A.Compose:
    """Return an augmentation pipeline that simulates Twitter-caused image degradation (skeleton)."""
    if not train:
        return A.Compose([])

    return A.Compose([
        A.ImageCompression(quality_lower=40, quality_upper=90, p=0.8),
        A.Downscale(scale_min=0.5, scale_max=0.9, p=0.5),
        A.GaussNoise(var_limit=(2.0, 12.0), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
    ])
