"""RT-DETRv2 criterion with NWD auxiliary loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.config import register
from .utils import (
    box_cxcywh_to_xyxy, generalized_box_iou,
    varifocal_loss, sigmoid_focal_loss,
)


def wasserstein_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7,
    constant: float = 12.8,
) -> torch.Tensor:
    """
    Normalized Wasserstein Distance loss for bounding boxes.

    Represents each box (cxcywh, normalized) as a 2D Gaussian:
        N(mu=(cx, cy), sigma=diag(w/2, h/2))

    W2^2 = ||mu1 - mu2||^2 + ||sigma1 - sigma2||_F^2
         = center_dist + wh_dist   (diagonal case)

    NWD = exp(-sqrt(W2^2) / constant)
    loss = 1 - NWD

    Args:
        pred_boxes: [N, 4] cxcywh normalized
        target_boxes: [N, 4] cxcywh normalized
        eps: numerical stability
        constant: normalization constant (empirically tuned to 12.8)

    Returns:
        loss: [N] per-box NWD loss
    """
    cx1, cy1, w1, h1 = pred_boxes.unbind(-1)
    cx2, cy2, w2, h2 = target_boxes.unbind(-1)

    # Center distance
    center_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Frobenius norm of diagonal sigma difference:
    # sigma = diag(w/2, h/2), so:
    # ||sigma1 - sigma2||_F^2 = (w1/2 - w2/2)^2 + (h1/2 - h2/2)^2
    #                         = (sqrt(w1) - sqrt(w2))^2/4 + (sqrt(h1) - sqrt(h2))^2/4
    # We absorb the constant 1/4 into the normalization constant `constant`.
    wh_dist = (w1.sqrt() - w2.sqrt()) ** 2 + (h1.sqrt() - h2.sqrt()) ** 2

    wasserstein_2 = center_dist + wh_dist
    nwd = torch.exp(-wasserstein_2.sqrt() / constant)
    return 1 - nwd


@register
class RTDETRCriterionv2(nn.Module):
    """
    RT-DETRv2 detection criterion.

    Losses:
        - vfl: VariFocal Loss for classification
        - boxes: L1 + GIoU + NWD (auxiliary, weight 0.5) for regression

    Weight dict keys: loss_vfl, loss_bbox, loss_giou, loss_nwd
    """

    def __init__(
        self,
        num_classes: int = 80,
        matcher=None,
        weight_dict: dict = None,
        losses: list = None,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.alpha = alpha
        self.gamma = gamma

        if weight_dict is None:
            weight_dict = {
                'loss_vfl': 1,
                'loss_bbox': 5,
                'loss_giou': 2,
                'loss_nwd': 0.5,
            }
        # Ensure NWD is in weight_dict with default weight 0.5
        if 'loss_nwd' not in weight_dict:
            weight_dict['loss_nwd'] = 0.5
        self.weight_dict = weight_dict

        self.losses = losses or ['vfl', 'boxes']

        # Build auxiliary weight_dict for intermediate decoder layers
        self._build_aux_weight_dict()

    def _build_aux_weight_dict(self):
        """Extend weight_dict for intermediate decoder outputs and denoising."""
        aux_weight = {}
        for k, v in self.weight_dict.items():
            # intermediate decoder layers (up to 5 auxiliary sets)
            for i in range(5):
                aux_weight[f'{k}_aux{i}'] = v
            # denoising
            aux_weight[f'{k}_dn'] = v
        self.weight_dict.update(aux_weight)

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def loss_vfl(self, outputs, targets, indices, num_boxes):
        """VariFocal Loss for classification."""
        src_logits = outputs['pred_logits']  # [B, N, C]
        B, N, C = src_logits.shape

        # Build soft targets: background = 0, matched = iou_score
        target_classes = torch.zeros(B, N, C, device=src_logits.device)

        src_boxes = outputs['pred_boxes']  # [B, N, 4]

        for b, (row_idx, col_idx) in enumerate(indices):
            if len(row_idx) == 0:
                continue
            tgt = targets[b]
            tgt_labels = tgt['labels'][col_idx]
            tgt_boxes_b = tgt['boxes'][col_idx]  # [Ni, 4] cxcywh
            pred_boxes_b = src_boxes[b, row_idx].detach()  # [Ni, 4]

            # Compute IoU score for soft label
            with torch.no_grad():
                pred_xyxy = box_cxcywh_to_xyxy(pred_boxes_b)
                tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes_b)
                iou = torch.diag(generalized_box_iou(pred_xyxy, tgt_xyxy)).clamp(0)

            target_classes[b, row_idx, tgt_labels] = iou

        # Normalised VFL loss
        loss = varifocal_loss(
            src_logits.flatten(0, 1),
            target_classes.flatten(0, 1),
            alpha=self.alpha,
            gamma=self.gamma,
            reduction='sum',
        )
        return {'loss_vfl': loss / max(num_boxes, 1)}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """L1 + GIoU + NWD losses for bounding boxes."""
        src_boxes_list = []
        tgt_boxes_list = []

        for b, (row_idx, col_idx) in enumerate(indices):
            if len(row_idx) == 0:
                continue
            src_boxes_list.append(outputs['pred_boxes'][b][row_idx])
            tgt_boxes_list.append(targets[b]['boxes'][col_idx])

        if not src_boxes_list:
            zero = outputs['pred_boxes'].sum() * 0
            return {'loss_bbox': zero, 'loss_giou': zero, 'loss_nwd': zero}

        src_boxes = torch.cat(src_boxes_list, dim=0)  # [M, 4]
        tgt_boxes = torch.cat(tgt_boxes_list, dim=0)  # [M, 4]

        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='sum') / max(num_boxes, 1)

        # GIoU loss
        src_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
        giou = torch.diag(generalized_box_iou(src_xyxy, tgt_xyxy))
        loss_giou = (1 - giou).sum() / max(num_boxes, 1)

        # NWD auxiliary loss (weight 0.5 in weight_dict)
        # Uses lightweight Wasserstein distance - complements GIoU without conflicting
        loss_nwd = wasserstein_loss(src_boxes, tgt_boxes).sum() / max(num_boxes, 1)

        return {
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
            'loss_nwd': loss_nwd,
        }

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _compute_losses(self, outputs, targets, indices, num_boxes):
        losses = {}
        for loss_name in self.losses:
            if loss_name == 'vfl':
                losses.update(self.loss_vfl(outputs, targets, indices, num_boxes))
            elif loss_name == 'boxes':
                losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses

    def forward(self, outputs: dict, targets: list) -> dict:
        """
        Compute losses.

        Args:
            outputs: dict with 'pred_logits', 'pred_boxes', optionally
                     'aux_outputs' (list) and 'dn_outputs'
            targets: list of target dicts per batch element

        Returns:
            dict of losses
        """
        # Number of matched boxes (across all batch elements)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = max(num_boxes, 1)

        # ---- Main output ----
        main_outputs = {
            'pred_logits': outputs['pred_logits'],
            'pred_boxes': outputs['pred_boxes'],
        }
        indices = self.matcher(main_outputs, targets)
        losses = self._compute_losses(main_outputs, targets, indices, num_boxes)

        # ---- Auxiliary outputs (intermediate decoder layers) ----
        if 'aux_outputs' in outputs:
            for i, aux in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher(aux, targets)
                aux_losses = self._compute_losses(aux, targets, aux_indices, num_boxes)
                losses.update({f'{k}_aux{i}': v for k, v in aux_losses.items()})

        # ---- Denoising outputs ----
        if 'dn_outputs' in outputs and outputs['dn_outputs'] is not None:
            dn_out = outputs['dn_outputs']
            dn_meta = outputs.get('dn_meta', {})
            dn_losses = self._compute_dn_losses(dn_out, targets, dn_meta, num_boxes)
            losses.update({f'{k}_dn': v for k, v in dn_losses.items()})

        return losses

    def _compute_dn_losses(self, dn_outputs, targets, dn_meta, num_boxes):
        """Compute losses for denoising queries using known ground-truth matching."""
        if dn_meta is None or 'gt_boxes' not in dn_meta:
            return {}

        B = dn_outputs['pred_logits'].shape[0]
        gt_boxes = dn_meta['gt_boxes']    # [B, G*N*2, 4]
        gt_labels = dn_meta['gt_labels']  # [B, G*N*2]
        pad_mask = dn_meta['pad_mask']    # [B, G*N*2] True=padding

        N_dn = dn_outputs['pred_logits'].shape[1]

        # Build direct assignment: DN query i <-> GT i (1-to-1 by construction)
        dn_targets = []
        dn_indices = []
        for b in range(B):
            valid = (~pad_mask[b]).nonzero(as_tuple=True)[0]  # valid GT indices
            n = min(len(valid), N_dn)
            if n == 0:
                dn_targets.append({
                    'boxes': torch.zeros(0, 4, device=gt_boxes.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=gt_labels.device),
                })
                dn_indices.append((torch.zeros(0, dtype=torch.long),
                                    torch.zeros(0, dtype=torch.long)))
                continue
            valid_n = valid[:n]
            dn_targets.append({
                'boxes': gt_boxes[b][valid_n],
                'labels': gt_labels[b][valid_n],
            })
            # DN query indices: the valid positions in the DN output
            dn_query_idx = valid_n.clamp(max=N_dn - 1)
            dn_gt_idx = torch.arange(n, device=valid_n.device)
            dn_indices.append((dn_query_idx, dn_gt_idx))

        return self._compute_losses(dn_outputs, dn_targets, dn_indices, num_boxes)
