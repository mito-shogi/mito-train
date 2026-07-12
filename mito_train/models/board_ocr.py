"""End-to-end board OCR model: capture image -> (9x9 board grid, 14-slot hand counts).

A design that integrates the 3-model pipeline (board_detector / piece_classifier /
hand_classifier). Does not require corner labels for the board; only sfen is used as the label.

The backbone is swappable. For smoke runs use MobileNetV3-small (~2M) for speed;
for real training swap in ConvNeXt-Tiny (~28M).

Outputs (hand_mode dependent):
    board_logits: (B, 29, 9, 9)  per-cell 29-class logits
    hand_out:
      - hand_mode="classification" (default) : (B, 14, 19) per hand-slot 19-class logits
      - hand_mode="regression"                : (B, 14) raw scalar counts (round + clamp at inference)
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

BackboneName = Literal[
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_b1",
    "convnext_atto",
    "convnext_femto",
    "convnext_pico",
    "convnext_nano",
    "convnext_tiny",
]
HandMode = Literal["classification", "regression"]

# timm's convnext family (atto/femto/pico/nano) — same architecture as
# torchvision's convnext_tiny, scaled down. Loaded uniformly via _TimmFeatureWrapper.
_TIMM_CONVNEXT_VARIANTS: tuple[str, ...] = (
    "convnext_atto",
    "convnext_femto",
    "convnext_pico",
    "convnext_nano",
)

# Theoretical max count per hand slot (sente P,L,N,S,G,B,R then gote same order).
# Pawn = 18, minor pieces = 4, bishop/rook = 2.
PIECE_MAX_PER_SLOT: tuple[int, ...] = (18, 4, 4, 4, 4, 2, 2, 18, 4, 4, 4, 4, 2, 2)


def _build_backbone(name: BackboneName, pretrained: bool) -> tuple[nn.Module, int]:
    """Return (features_module, out_channels).

    torchvision backbones expose feature maps directly via `.features`.
    timm backbones use `features_only=True` and return a list of feature
    maps at each stride; we take the deepest (last) one.
    """
    if name == "mobilenet_v3_small":
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = mobilenet_v3_small(weights=weights)
        return m.features, 576
    if name == "mobilenet_v3_large":
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        m = mobilenet_v3_large(weights=weights)
        return m.features, 960
    if name == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = efficientnet_b0(weights=weights)
        return m.features, 1280
    if name == "efficientnet_b1":
        from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1
        weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
        m = efficientnet_b1(weights=weights)
        return m.features, 1280
    if name == "convnext_tiny":
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = convnext_tiny(weights=weights)
        return m.features, 768
    if name in _TIMM_CONVNEXT_VARIANTS:
        import timm
        m = timm.create_model(name, pretrained=pretrained, features_only=True)
        feat_dim = m.feature_info.channels()[-1]
        return _TimmFeatureWrapper(m), feat_dim
    raise ValueError(f"unknown backbone: {name}")


class _TimmFeatureWrapper(nn.Module):
    """Adapts timm's features_only output (list of feature maps) to expose only
    the deepest map, matching the torchvision `.features` contract used elsewhere.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return feats[-1]


class BoardOCR(nn.Module):
    """Two-head model: capture image -> (board grid logits, hand slot counts).

    The hand head has two modes:
    - "classification" (default): 14 x 19 logits; CE loss, argmax at inference.
    - "regression": 14 scalars; SmoothL1 loss, `round + clamp(0, piece_max_per_slot)` at
      inference. Motivated by TRAINING_PLAN §A — the classification head treats
      "5 vs 6" and "5 vs 18" equally, discarding ordinal info between counts.
    """

    def __init__(
        self,
        backbone: BackboneName = "mobilenet_v3_small",
        num_board_classes: int = 29,
        hand_slots: int = 14,
        hand_max: int = 19,
        hand_mode: HandMode = "classification",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.features, feat_dim = _build_backbone(backbone, pretrained)
        self.num_board_classes = num_board_classes
        self.hand_slots = hand_slots
        self.hand_max = hand_max
        self.hand_mode: HandMode = hand_mode

        # board head: AdaptivePool feature map to 9x9 -> 1x1 conv to 29 channels
        self.board_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((9, 9)),
            nn.Conv2d(feat_dim, num_board_classes, kernel_size=1),
        )
        self.hand_pool = nn.AdaptiveAvgPool2d(1)

        if len(PIECE_MAX_PER_SLOT) != hand_slots:
            raise ValueError(
                f"PIECE_MAX_PER_SLOT length {len(PIECE_MAX_PER_SLOT)} != hand_slots {hand_slots}"
            )

        # Per-slot theoretical max, exposed for the training loop / inference to
        # clamp regression predictions and for downstream code to introspect.
        self.register_buffer(
            "piece_max_per_slot",
            torch.tensor(PIECE_MAX_PER_SLOT, dtype=torch.long),
            persistent=False,
        )

        if hand_mode == "classification":
            # global pool -> Linear to 14*19 -> reshape
            self.hand_head = nn.Linear(feat_dim, hand_slots * hand_max)
            # Physically impossible slot/count combinations get -inf added to their
            # logit so both training softmax and inference argmax ignore them.
            mask = torch.zeros(1, hand_slots, hand_max)
            for slot_idx, piece_max in enumerate(PIECE_MAX_PER_SLOT):
                if piece_max + 1 < hand_max:
                    mask[0, slot_idx, piece_max + 1 :] = float("-inf")
            self.register_buffer("hand_logit_mask", mask, persistent=False)
        elif hand_mode == "regression":
            # global pool -> Linear to 14 raw scalars
            self.hand_head = nn.Linear(feat_dim, hand_slots)
        else:
            raise ValueError(f"unknown hand_mode: {hand_mode!r}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)  # (B, feat_dim, H', W')
        board_logits = self.board_head(feat)  # (B, 29, 9, 9)
        hand_feat = self.hand_pool(feat).flatten(1)  # (B, feat_dim)
        if self.hand_mode == "classification":
            hand_out = self.hand_head(hand_feat).view(
                -1, self.hand_slots, self.hand_max
            )  # (B, 14, 19)
            hand_out = hand_out + self.hand_logit_mask
        else:  # regression
            hand_out = self.hand_head(hand_feat)  # (B, 14)
        return board_logits, hand_out

    def predict_hand(self, hand_out: torch.Tensor) -> torch.Tensor:
        """Convert raw head output to discrete count predictions (B, hand_slots) long.

        - classification: argmax over the 19 count classes.
        - regression: round + clamp to per-slot theoretical max.
        """
        if self.hand_mode == "classification":
            return hand_out.argmax(dim=-1)
        # regression
        rounded = hand_out.round()
        max_per_slot = self.piece_max_per_slot.to(rounded.device).unsqueeze(0)  # (1, 14)
        return rounded.clamp(min=0).minimum(max_per_slot.expand_as(rounded)).long()
