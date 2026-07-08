"""Piece classifier model: 29-class tiny CNN.

Contract: see docs/ocr-model-interface.md section 3.
Input: single-square crop. Output: 29-class logits (empty + each piece type x side + promoted).
TODO: keep the class definition strictly aligned with the contract.
"""
from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASSES = 29


class PieceClassifier(nn.Module):
    """Lightweight CNN that classifies a single-square image into 29 classes (skeleton)."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x).flatten(1)
        return self.classifier(z)
