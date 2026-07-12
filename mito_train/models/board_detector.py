"""Board detector: raw screenshot -> normalized ShogiBoardView bbox.

Contract:
    Input:  (B, 3, H, W) float32, ImageNet-normalized.
    Output: (B, 4) float32, normalized bbox (x1, y1, x2, y2) in [0, 1] w.r.t. input HxW.

The backbone is MobileNetV3-small (same family as BoardOCR v3) so the whole
pipeline stays uniform in weight class. A sigmoid on the head keeps outputs in
[0, 1] and stabilizes early training.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

BackboneName = Literal["mobilenet_v3_small", "convnext_tiny"]


def _build_backbone(name: BackboneName, pretrained: bool) -> tuple[nn.Module, int]:
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


class BoardDetector(nn.Module):
    def __init__(
        self,
        backbone: BackboneName = "mobilenet_v3_small",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.features, feat_dim = _build_backbone(backbone, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = self.pool(f).flatten(1)
        return self.head(f)  # (B, 4) normalized (x1, y1, x2, y2)
