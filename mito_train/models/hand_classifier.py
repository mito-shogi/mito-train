"""Hand-piece classifier model: multi-head.

Contract: see docs/ocr-model-interface.md section 4.
Input: hand-region image. Output: per-piece count (multi-head regression or classification).
TODO: align the piece-type heads and count ranges with the contract.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# 7 piece types (pawn, lance, knight, silver, gold, bishop, rook) x 2 sides
HAND_PIECE_TYPES = 7


class HandClassifier(nn.Module):
    """Multi-head CNN that predicts per-piece counts from the hand region (skeleton)."""

    def __init__(self, num_piece_types: int = HAND_PIECE_TYPES, max_count: int = 18):
        super().__init__()
        self.max_count = max_count
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        # Per piece type, (max_count+1)-way classification (0..max_count pieces)
        self.heads = nn.ModuleList(
            [nn.Linear(64, max_count + 1) for _ in range(num_piece_types)]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        z = self.features(x).flatten(1)
        return [head(z) for head in self.heads]
