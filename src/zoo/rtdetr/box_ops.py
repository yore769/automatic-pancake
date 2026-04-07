"""Bounding-box utility functions used throughout RT-DETR."""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Format conversions
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h)  →  (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """(x1, y1, x2, y2)  →  (cx, cy, w, h)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


# ---------------------------------------------------------------------------
# IoU family
# ---------------------------------------------------------------------------

def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Area of (x1, y1, x2, y2) boxes."""
    return (boxes[:, 2] - boxes[:, 0]).clamp(0) * (boxes[:, 3] - boxes[:, 1]).clamp(0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """Pairwise IoU between two sets of boxes (xyxy format).

    Returns:
        iou  – shape (N, M)
        union – shape (N, M)
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalised IoU (GIoU) between two sets of boxes (xyxy).

    Returns shape (N, M).
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1 has negative dimensions"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2 has negative dimensions"

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    enclosing_area = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (enclosing_area - union) / (enclosing_area + 1e-6)
    return giou


# ---------------------------------------------------------------------------
# NWD – Normalised Wasserstein Distance for small objects
# ---------------------------------------------------------------------------

def normalized_wasserstein_distance(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    C: float = 2.0,
) -> torch.Tensor:
    """Compute NWD between matched pairs of boxes (cxcywh format).

    NWD = exp(-W₂ / C) where W₂ is the Wasserstein-2 distance between the
    2-D Gaussian representations of the boxes.

    Args:
        pred_boxes:   (N, 4) predicted boxes in cxcywh, **normalised** [0, 1].
        target_boxes: (N, 4) target boxes in cxcywh, **normalised** [0, 1].
        C:            Normalisation factor.  The original paper uses C=12.8 for
                      un-normalised pixel coordinates on 800×800 images.  For
                      normalised [0, 1] coordinates (used in RT-DETR) the
                      appropriate value is C=2.0 — this scales the sensitivity
                      to match the coordinate range and makes NWD informative
                      for small objects (w,h ~ 0.01–0.05 in normalised coords).

    Returns:
        nwd: (N,) values in (0, 1].  Loss = 1 − nwd.
    """
    px, py, pw, ph = pred_boxes.unbind(-1)
    tx, ty, tw, th = target_boxes.unbind(-1)

    # Squared Wasserstein-2 distance between Gaussians:
    #   W₂² = ||μ₁ − μ₂||² + ||Σ₁^½ − Σ₂^½||²_F
    # For axis-aligned Gaussians with σᵢ = dᵢ/2:
    #   = (px-tx)² + (py-ty)² + (pw/2 - tw/2)² + (ph/2 - th/2)²
    w2_sq = (
        (px - tx) ** 2
        + (py - ty) ** 2
        + ((pw - tw) / 2) ** 2
        + ((ph - th) / 2) ** 2
    )
    w2 = w2_sq.sqrt()
    nwd = torch.exp(-w2 / C)
    return nwd


def nwd_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    C: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """Loss = 1 − NWD (lower is better).

    Args:
        pred_boxes:   (N, 4) cxcywh normalised.
        target_boxes: (N, 4) cxcywh normalised.
        C:            Normalisation factor (default 2.0 for normalised coords).
        reduction:    'none' | 'mean' | 'sum'.
    """
    loss = 1.0 - normalized_wasserstein_distance(pred_boxes, target_boxes, C)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
