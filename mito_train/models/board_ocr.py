"""End-to-end board OCR model: capture image -> (9x9 board grid, 14-slot hand counts).

A design that integrates the 3-model pipeline (board_detector / piece_classifier /
hand_classifier). Does not require corner labels for the board; only sfen is used as the label.

The backbone is swappable. For smoke runs use MobileNetV3-small (~2M) for speed;
for real training swap in ConvNeXt-Tiny (~28M).

Outputs:
    board_logits: (B, 29, 9, 9)  per-cell 29-class logits
    hand_logits:  (B, 14, 19)    per hand-slot 19-class logits (0..18 pieces)
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

BackboneName = Literal["mobilenet_v3_small", "convnext_tiny"]

# Theoretical max count per hand slot (sente P,L,N,S,G,B,R then gote same order).
# Pawn = 18, minor pieces = 4, bishop/rook = 2.
PIECE_MAX_PER_SLOT: tuple[int, ...] = (18, 4, 4, 4, 4, 2, 2, 18, 4, 4, 4, 4, 2, 2)


def _build_backbone(name: BackboneName, pretrained: bool) -> tuple[nn.Module, int]:
    """Return (features_module, out_channels)."""
    if name == "mobilenet_v3_small":
        from torchvision.models import (
            MobileNet_V3_Small_Weights,
            mobilenet_v3_small,
        )
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = mobilenet_v3_small(weights=weights)
        return m.features, 576
    if name == "convnext_tiny":
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = convnext_tiny(weights=weights)
        return m.features, 768
    raise ValueError(f"unknown backbone: {name}")


class BoardOCR(nn.Module):
    """Two-head model: capture image -> (board grid logits, hand slot logits)."""

    def __init__(
        self,
        backbone: BackboneName = "mobilenet_v3_small",
        num_board_classes: int = 29,
        hand_slots: int = 14,
        hand_max: int = 19,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.features, feat_dim = _build_backbone(backbone, pretrained)
        self.num_board_classes = num_board_classes
        self.hand_slots = hand_slots
        self.hand_max = hand_max

        # board head: AdaptivePool feature map to 9x9 -> 1x1 conv to 29 channels
        self.board_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((9, 9)),
            nn.Conv2d(feat_dim, num_board_classes, kernel_size=1),
        )
        # hand head: global pool -> Linear to 14*19 -> reshape
        self.hand_pool = nn.AdaptiveAvgPool2d(1)
        self.hand_head = nn.Linear(feat_dim, hand_slots * hand_max)

        # Physically impossible slot/count combinations get -inf added to their
        # logit so both training softmax and inference argmax ignore them.
        # Shape (1, hand_slots, hand_max) for broadcast over batch.
        if len(PIECE_MAX_PER_SLOT) != hand_slots:
            raise ValueError(
                f"PIECE_MAX_PER_SLOT length {len(PIECE_MAX_PER_SLOT)} != hand_slots {hand_slots}"
            )
        mask = torch.zeros(1, hand_slots, hand_max)
        for slot_idx, piece_max in enumerate(PIECE_MAX_PER_SLOT):
            if piece_max + 1 < hand_max:
                mask[0, slot_idx, piece_max + 1 :] = float("-inf")
        self.register_buffer("hand_logit_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)  # (B, feat_dim, H', W')
        board_logits = self.board_head(feat)  # (B, 29, 9, 9)
        hand_feat = self.hand_pool(feat).flatten(1)  # (B, feat_dim)
        hand_logits = self.hand_head(hand_feat).view(
            -1, self.hand_slots, self.hand_max
        )  # (B, 14, 19)
        hand_logits = hand_logits + self.hand_logit_mask
        return board_logits, hand_logits
