"""Board detector model: tiny UNet or 4-corner regression.

Contract: see docs/ocr-model-interface.md section 2.
Input: normalized image (1x3xHxW). Output: board 4-corner coordinates or a segmentation mask.
TODO: finalize and implement per the contract.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BoardDetector(nn.Module):
    """Lightweight CNN that regresses the 4 board corners (skeleton)."""

    def __init__(self, out_dim: int = 8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, out_dim)  # (x, y) for each of 4 corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x).flatten(1)
        return self.head(z)
