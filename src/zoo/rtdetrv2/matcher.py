"""Hungarian Matcher for RT-DETRv2."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.core.config import register
from .utils import box_cxcywh_to_xyxy, generalized_box_iou


@register
class HungarianMatcher(nn.Module):
    """
    Optimal bipartite matching between predicted and target boxes.

    Args:
        weight_dict: dict with 'cost_class', 'cost_bbox', 'cost_giou'
        alpha, gamma: focal loss params for classification cost
    """

    def __init__(
        self,
        weight_dict: dict = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        if weight_dict is None:
            weight_dict = {'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2}
        self.cost_class = weight_dict.get('cost_class', 2)
        self.cost_bbox = weight_dict.get('cost_bbox', 5)
        self.cost_giou = weight_dict.get('cost_giou', 2)
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list):
        """
        Args:
            outputs: dict with 'pred_logits' [B, N, C] and 'pred_boxes' [B, N, 4]
            targets: list of dicts with 'labels' [Ni] and 'boxes' [Ni, 4]
        Returns:
            list of (row_indices, col_indices) tuples, one per batch element
        """
        B, N, C = outputs['pred_logits'].shape
        pred_logits = outputs['pred_logits'].flatten(0, 1)  # [B*N, C]
        pred_boxes = outputs['pred_boxes'].flatten(0, 1)    # [B*N, 4]

        # Compute class cost using focal-like scoring
        pred_scores = torch.sigmoid(pred_logits)  # [B*N, C]
        alpha = self.alpha
        gamma = self.gamma
        neg_cost = (1 - alpha) * pred_scores ** gamma * (-(1 - pred_scores + 1e-8).log())
        pos_cost = alpha * (1 - pred_scores) ** gamma * (-(pred_scores + 1e-8).log())

        indices = []
        for b in range(B):
            tgt = targets[b]
            if len(tgt['labels']) == 0:
                indices.append((torch.zeros(0, dtype=torch.long),
                                 torch.zeros(0, dtype=torch.long)))
                continue

            tgt_labels = tgt['labels']  # [Ni]
            tgt_boxes = tgt['boxes']    # [Ni, 4] cxcywh

            n = len(tgt_labels)
            src_start = b * N

            # Class cost: [N, Ni]
            cost_class = (
                pos_cost[src_start:src_start + N, tgt_labels]
                - neg_cost[src_start:src_start + N, tgt_labels]
            )

            # Box L1 cost: [N, Ni]
            src_boxes_b = pred_boxes[src_start:src_start + N]
            cost_bbox = torch.cdist(src_boxes_b, tgt_boxes, p=1)

            # GIoU cost: [N, Ni]
            src_xyxy = box_cxcywh_to_xyxy(src_boxes_b)
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
            cost_giou = -generalized_box_iou(src_xyxy, tgt_xyxy)

            # Total cost
            C_mat = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )

            row_ind, col_ind = linear_sum_assignment(C_mat.cpu().numpy())
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long),
                torch.as_tensor(col_ind, dtype=torch.long),
            ))

        return indices
