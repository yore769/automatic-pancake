"""Hungarian Matcher for RT-DETR."""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.core import register
from .box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    normalized_wasserstein_distance,
)


@register()
class HungarianMatcher(torch.nn.Module):
    """Optimal bipartite matching between predictions and ground-truth.

    Supports three cost terms:
        cost_class – focal-style classification cost.
        cost_bbox  – L1 distance between box coordinates (cxcywh).
        cost_giou  – negative GIoU.
    """

    def __init__(
        self,
        weight_dict: dict = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
        use_nwd: bool = False,
        nwd_weight: float = 2.0,
        nwd_C: float = 2.0,
    ):
        super().__init__()
        if weight_dict is None:
            weight_dict = {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}
        self.cost_class = weight_dict.get("cost_class", 2)
        self.cost_bbox = weight_dict.get("cost_bbox", 5)
        self.cost_giou = weight_dict.get("cost_giou", 2)
        self.alpha = alpha
        self.gamma = gamma
        self.use_nwd = use_nwd
        self.nwd_weight = nwd_weight
        self.nwd_C = nwd_C

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list):
        """Compute optimal assignment.

        Args:
            outputs: dict with 'pred_logits' (B, Q, C) and 'pred_boxes' (B, Q, 4).
            targets: list of dicts, each with 'labels' (Nᵢ,) and 'boxes' (Nᵢ, 4).

        Returns:
            List of (index_i, index_j) pairs – one per batch element.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten batch dimension for batch cost computation
        pred_logits = outputs["pred_logits"].flatten(0, 1).sigmoid()  # (B*Q, C)
        pred_boxes = outputs["pred_boxes"].flatten(0, 1)               # (B*Q, 4)

        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_boxes = torch.cat([t["boxes"] for t in targets])

        # --- Classification cost (focal) ---
        neg_cost_class = (1 - self.alpha) * (pred_logits ** self.gamma) * (
            -(1 - pred_logits + 1e-8).log()
        )
        pos_cost_class = self.alpha * ((1 - pred_logits) ** self.gamma) * (
            -(pred_logits + 1e-8).log()
        )
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # --- Box L1 cost ---
        cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)

        # --- IoU / NWD cost ---
        if self.use_nwd:
            # NWD cost: shape (B*Q, Σ Nᵢ)
            # Use pairwise NWD between every prediction and every target.
            n_pred = pred_boxes.shape[0]
            n_tgt = tgt_boxes.shape[0]
            pred_exp = pred_boxes.unsqueeze(1).expand(n_pred, n_tgt, 4)
            tgt_exp = tgt_boxes.unsqueeze(0).expand(n_pred, n_tgt, 4)
            nwd_vals = normalized_wasserstein_distance(
                pred_exp.reshape(-1, 4), tgt_exp.reshape(-1, 4), C=self.nwd_C
            ).reshape(n_pred, n_tgt)
            cost_iou = -nwd_vals * self.nwd_weight
        else:
            cost_iou = -generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes),
                box_cxcywh_to_xyxy(tgt_boxes),
            )

        # --- Total cost matrix ---
        C_matrix = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_iou
        )
        C_matrix = C_matrix.view(bs, num_queries, -1).cpu()

        sizes = [len(t["boxes"]) for t in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C_matrix.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
