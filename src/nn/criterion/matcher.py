"""Hungarian Matcher for RT-DETR v2.

Performs optimal bipartite matching between predicted and ground-truth boxes.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from src.core.workspace import register
from .nwd_loss import giou_loss

__all__ = ['HungarianMatcher']


@register
class HungarianMatcher(nn.Module):
    """Hungarian bipartite matcher.

    Args:
        weight_dict: {cost_class, cost_bbox, cost_giou}
        alpha, gamma: focal-loss parameters for class cost
    """

    def __init__(self, weight_dict=None, alpha=0.25, gamma=2.0):
        super().__init__()
        if weight_dict is None:
            weight_dict = {'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2}
        self.cost_class = weight_dict.get('cost_class', 2)
        self.cost_bbox = weight_dict.get('cost_bbox', 5)
        self.cost_giou = weight_dict.get('cost_giou', 2)
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, pred_logits, pred_boxes, targets):
        """Compute bipartite matching.

        Args:
            pred_logits: (B, Nq, num_classes)
            pred_boxes:  (B, Nq, 4) in (cx, cy, w, h)
            targets:     list of dicts with 'labels' and 'boxes'

        Returns:
            list of (row_idx, col_idx) tuples per image
        """
        B, Nq, C = pred_logits.shape
        indices = []

        for b in range(B):
            tgt = targets[b]
            Nt = len(tgt['labels'])
            if Nt == 0:
                indices.append((torch.zeros(0, dtype=torch.long),
                                torch.zeros(0, dtype=torch.long)))
                continue

            pred_prob = pred_logits[b].sigmoid()  # (Nq, C)
            tgt_labels = tgt['labels']            # (Nt,)
            tgt_boxes = tgt['boxes']              # (Nt, 4)

            # Classification cost (focal-loss style)
            neg_cost_class = (1 - self.alpha) * (pred_prob ** self.gamma) * \
                             (-(1 - pred_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - pred_prob) ** self.gamma) * \
                             (-(pred_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_labels] - \
                         neg_cost_class[:, tgt_labels]  # (Nq, Nt)

            # L1 box cost
            cost_bbox = torch.cdist(pred_boxes[b], tgt_boxes, p=1)  # (Nq, Nt)

            # GIoU cost
            pred_b = pred_boxes[b].unsqueeze(1).expand(-1, Nt, -1).reshape(-1, 4)
            tgt_b = tgt_boxes.unsqueeze(0).expand(Nq, -1, -1).reshape(-1, 4)
            cost_giou = giou_loss(pred_b, tgt_b).reshape(Nq, Nt)

            C_mat = (self.cost_class * cost_class +
                     self.cost_bbox * cost_bbox +
                     self.cost_giou * cost_giou)

            row_idx, col_idx = linear_sum_assignment(C_mat.cpu().numpy())
            indices.append((
                torch.as_tensor(row_idx, dtype=torch.long),
                torch.as_tensor(col_idx, dtype=torch.long),
            ))

        return indices
