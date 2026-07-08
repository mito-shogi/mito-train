"""Evaluation metrics: piece accuracy / IoU / exact-match.

See docs/ocr-metrics.md for full definitions.
"""
from __future__ import annotations


def piece_accuracy(pred_board: list[str], gt_board: list[str]) -> float:
    """Per-board piece match rate over 81 squares."""
    if not gt_board:
        return 0.0
    correct = sum(p == g for p, g in zip(pred_board, gt_board))
    return correct / len(gt_board)


def exact_match(pred_sfen: str, gt_sfen: str) -> bool:
    """Exact match on normalized SFEN."""
    return pred_sfen.strip() == gt_sfen.strip()


def bbox_iou(a: tuple[float, float, float, float],
             b: tuple[float, float, float, float]) -> float:
    """IoU between two bboxes (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0
