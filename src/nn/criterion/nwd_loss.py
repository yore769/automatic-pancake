"""NWD Loss (Normalized Wasserstein Distance Loss) for small object detection.

Background
----------
For small objects, the bounding box area is tiny compared to the image.
Standard IoU-based losses have near-zero gradient when predicted and target
boxes don't overlap, stalling learning.  NWD models each box as a 2-D
Gaussian and computes the Wasserstein-2 distance between the distributions.
Because the Gaussian domain is continuous, NWD always produces a non-zero
gradient signal even when boxes don't overlap spatially.

Combined-Improvement Integration
---------------------------------
The critical bug in the original combined implementation was that NWD and
GIoU were both assigned weight 1.0 (or close to it), but NWD operates on
a different numerical scale (typically 0–1 after normalisation) than GIoU
(also 0–1 but with different gradient magnitudes).  When combined, NWD
gradients dominated early in training, preventing the model from learning
the basic geometric constraints captured by GIoU/bbox regression.

Fix: scale ``nwd_weight`` relative to ``giou_weight`` so that their
gradient magnitudes are balanced.  The recommended ratio is
    loss_nwd_weight = 0.5 * loss_giou_weight
i.e., NWD provides an additional soft signal without overpowering GIoU.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ['nwd_loss', 'giou_loss', 'l1_loss']


def _boxes_to_gaussian_params(boxes: Tensor):
    """Convert (cx, cy, w, h) boxes to Gaussian (mean, variance) parameters.

    Args:
        boxes: (..., 4) in (cx, cy, w, h) format (values in [0, 1])

    Returns:
        mu:  (..., 2)  mean  = [cx, cy]
        var: (..., 2)  variance = [w/2, h/2]^2  (treating box as 2σ range)
    """
    mu = boxes[..., :2]               # (cx, cy)
    sigma = boxes[..., 2:] / 2.0     # half-widths as std devs
    var = sigma ** 2
    return mu, var


def nwd_loss(pred_boxes: Tensor, target_boxes: Tensor,
             constant: float = 12.8) -> Tensor:
    """Compute NWD (Normalized Wasserstein Distance) loss.

    NWD = 1 - exp(-W2 / C)

    where W2 is the Wasserstein-2 distance between two axis-aligned Gaussians:
        W2² = ||μ₁ - μ₂||² + ||σ₁ - σ₂||²

    The ``constant`` C controls the scale of the exponential decay.
    The value 12.8 is recommended in the NWD paper for normalised coordinates.

    Args:
        pred_boxes:   (N, 4) predicted boxes in (cx, cy, w, h)
        target_boxes: (N, 4) target boxes in (cx, cy, w, h)
        constant:     decay constant C

    Returns:
        (N,) per-element NWD loss values in [0, 1]
    """
    mu_p, var_p = _boxes_to_gaussian_params(pred_boxes)
    mu_t, var_t = _boxes_to_gaussian_params(target_boxes)

    sigma_p = var_p.sqrt()
    sigma_t = var_t.sqrt()

    # Wasserstein-2 distance squared for axis-aligned Gaussians
    w2_sq = ((mu_p - mu_t) ** 2).sum(dim=-1) + \
            ((sigma_p - sigma_t) ** 2).sum(dim=-1)

    # Normalised Wasserstein Distance
    nwd = torch.exp(-w2_sq.sqrt() / constant)  # (N,)

    return 1.0 - nwd  # loss: 0 = perfect match, 1 = worst


def giou_loss(pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
    """Generalised IoU loss.

    Args:
        pred_boxes:   (N, 4) in (cx, cy, w, h)
        target_boxes: (N, 4) in (cx, cy, w, h)

    Returns:
        (N,) per-element GIoU loss values in [0, 2]
    """
    # Convert to (x1, y1, x2, y2)
    def to_xyxy(b):
        return torch.cat([b[..., :2] - b[..., 2:] / 2,
                          b[..., :2] + b[..., 2:] / 2], dim=-1)

    p = to_xyxy(pred_boxes)
    t = to_xyxy(target_boxes)

    # Intersection
    inter_x1 = torch.max(p[..., 0], t[..., 0])
    inter_y1 = torch.max(p[..., 1], t[..., 1])
    inter_x2 = torch.min(p[..., 2], t[..., 2])
    inter_y2 = torch.min(p[..., 3], t[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    area_p = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
    area_t = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    union_area = area_p + area_t - inter_area + 1e-7

    iou = inter_area / union_area

    # Enclosing box
    enc_x1 = torch.min(p[..., 0], t[..., 0])
    enc_y1 = torch.min(p[..., 1], t[..., 1])
    enc_x2 = torch.max(p[..., 2], t[..., 2])
    enc_y2 = torch.max(p[..., 3], t[..., 3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1) + 1e-7

    giou = iou - (enc_area - union_area) / enc_area
    return 1.0 - giou


def l1_loss(pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
    """Element-wise L1 loss on box coordinates."""
    return F.l1_loss(pred_boxes, target_boxes, reduction='none').sum(dim=-1)
