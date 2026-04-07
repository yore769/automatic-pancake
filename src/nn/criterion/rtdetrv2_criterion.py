"""RT-DETR v2 Criterion with NWD Loss integration.

NWD + GIoU Integration Fix
---------------------------
The original combined model had NWD Loss weight equal to GIoU Loss weight.
This causes gradient magnitude imbalance because:

  • GIoU Loss gradient is bounded by [-2, 2] and well-conditioned throughout
    training (it always produces signal even for overlapping boxes).
  • NWD Loss gradient in the early training phase when predicted boxes are
    far from targets can be much larger (the Wasserstein distance is large,
    making the exponential term near zero, giving gradient ≈ 1/C per coord).
  
  → NWD dominates early training → GIoU/bbox regression doesn't converge
  → combined model is worse than either alone.

Fix Strategy
------------
1. ``loss_nwd`` weight is set to ``0.5 * loss_giou`` by default.
2. NWD is treated as an *auxiliary* signal rather than a primary objective:
   it is only applied at the **last decoder layer** (not aux layers) to avoid
   gradient accumulation across all 6 layers overwhelming the GIoU signal.
3. For the denoising auxiliary outputs the NWD loss is also skipped.

These three changes together ensure that NWD adds complementary gradient
signal for small objects without competing with GIoU's fundamental role of
ensuring geometric correctness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional

from src.core.workspace import register, create
from .nwd_loss import nwd_loss, giou_loss, l1_loss
from .matcher import HungarianMatcher

__all__ = ['RTDETRCriterionv2']


def _get_src_permutation_idx(indices):
    """Build batch/src index for matched predictions."""
    batch_idx = torch.cat([
        torch.full_like(src, i) for i, (src, _) in enumerate(indices)
    ])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _get_tgt_permutation_idx(indices):
    """Build batch/tgt index for matched ground-truths."""
    batch_idx = torch.cat([
        torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
    ])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


@register
class RTDETRCriterionv2(nn.Module):
    """Criterion for RT-DETR v2.

    Args:
        num_classes:    number of detection classes
        weight_dict:    {loss_vfl: w, loss_bbox: w, loss_giou: w,
                         loss_nwd: w (optional)}
        losses:         list of loss types to compute: 'vfl', 'boxes'
        alpha, gamma:   VariFocal-loss parameters
        matcher:        matcher config dict or HungarianMatcher instance
        use_nwd_loss:   whether to compute NWD loss
        nwd_only_final: if True, only compute NWD at the final decoder layer
                        (recommended when combined with Dynamic Query Grouping
                        and LS Conv to prevent gradient overload)
    """

    def __init__(self,
                 num_classes=80,
                 weight_dict=None,
                 losses=('vfl', 'boxes'),
                 alpha=0.75,
                 gamma=2.0,
                 matcher=None,
                 use_nwd_loss=False,
                 nwd_only_final=True):
        super().__init__()
        self.num_classes = num_classes
        self.losses = list(losses)
        self.alpha = alpha
        self.gamma = gamma
        self.use_nwd_loss = use_nwd_loss
        self.nwd_only_final = nwd_only_final

        # Build weight dict with safe defaults
        if weight_dict is None:
            weight_dict = {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2}
        self.weight_dict = weight_dict

        # Ensure NWD weight is set. Default: 0.5 × GIoU weight.
        # Note: if the user explicitly sets loss_nwd in weight_dict, that value
        # is used as-is. This only provides a safe default when it is absent.
        if 'loss_nwd' not in self.weight_dict:
            giou_w = self.weight_dict.get('loss_giou', 2)
            self.weight_dict['loss_nwd'] = giou_w * 0.5

        # Build matcher
        if matcher is None:
            self.matcher = HungarianMatcher()
        elif isinstance(matcher, dict):
            self.matcher = create(matcher.get('type', 'HungarianMatcher'),
                                  **{k: v for k, v in matcher.items()
                                     if k != 'type'})
        else:
            self.matcher = matcher

    # ------------------------------------------------------------------
    # Individual loss computations
    # ------------------------------------------------------------------

    def loss_vfl(self, pred_logits, targets, indices,
                 num_boxes, **kwargs) -> Dict[str, Tensor]:
        """VariFocal classification loss."""
        idx = _get_src_permutation_idx(indices)

        target_scores = torch.zeros(
            pred_logits.shape[:2], dtype=torch.float32,
            device=pred_logits.device)

        # Compute IoU scores for matched pairs (used as target quality)
        pred_boxes_matched = kwargs.get('pred_boxes', None)
        if pred_boxes_matched is not None:
            matched_pred = pred_boxes_matched[idx]  # (M, 4)
            matched_tgt_boxes = torch.cat(
                [t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)
            with torch.no_grad():
                iou = 1.0 - giou_loss(matched_pred, matched_tgt_boxes)
                iou = iou.clamp(0.)

            # Assign IoU as soft label (VariFocal quality score)
            target_class = torch.cat(
                [t['labels'][j] for t, (_, j) in zip(targets, indices)], dim=0)
            target_scores_o = torch.zeros(
                len(target_class), self.num_classes,
                device=pred_logits.device)
            target_scores_o.scatter_(1, target_class.unsqueeze(1), iou.unsqueeze(1))
        else:
            target_class = torch.cat(
                [t['labels'][j] for t, (_, j) in zip(targets, indices)], dim=0)
            target_scores_o = torch.zeros(
                len(target_class), self.num_classes,
                device=pred_logits.device)
            target_scores_o.scatter_(1, target_class.unsqueeze(1), 1.0)

        target_scores_2d = torch.zeros(
            *pred_logits.shape[:2], self.num_classes,
            device=pred_logits.device)
        target_scores_2d[idx] = target_scores_o

        pred_prob = pred_logits.sigmoid()

        # VariFocal loss
        weight = self.alpha * pred_prob.pow(self.gamma) * (1 - target_scores_2d) + \
                 target_scores_2d * (1 - pred_prob).pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred_logits, target_scores_2d, reduction='none')
        loss = (loss * weight).sum() / num_boxes

        return {'loss_vfl': loss}

    def loss_boxes(self, pred_boxes, targets, indices, num_boxes,
                   compute_nwd=False, **kwargs) -> Dict[str, Tensor]:
        """Bounding box regression: L1 + GIoU (+ optional NWD)."""
        idx = _get_src_permutation_idx(indices)
        matched_pred = pred_boxes[idx]  # (M, 4)
        matched_tgt = torch.cat(
            [t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        loss_l1 = l1_loss(matched_pred, matched_tgt).sum() / num_boxes
        loss_giou = giou_loss(matched_pred, matched_tgt).sum() / num_boxes

        losses = {
            'loss_bbox': loss_l1,
            'loss_giou': loss_giou,
        }

        if self.use_nwd_loss and compute_nwd:
            loss_nwd = nwd_loss(matched_pred, matched_tgt).sum() / num_boxes
            losses['loss_nwd'] = loss_nwd

        return losses

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, outputs: Dict, targets: List[Dict],
                **kwargs) -> Dict[str, Tensor]:
        """Compute all losses.

        Args:
            outputs:  model output dict with keys:
                      'pred_logits', 'pred_boxes', 'aux_outputs',
                      'enc_topk_logits', 'enc_topk_bboxes',
                      (optional) 'dn_aux_outputs', 'dn_meta'
            targets:  list of per-image dicts with 'labels', 'boxes'

        Returns:
            dict of named scalar losses, weighted by weight_dict
        """
        all_losses: Dict[str, Tensor] = {}

        # ---- Main (final decoder layer) output ----
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']

        indices = self.matcher(pred_logits, pred_boxes, targets)
        num_boxes = max(sum(len(t['labels']) for t in targets), 1)

        losses = {}
        losses.update(self.loss_vfl(pred_logits, targets, indices,
                                     num_boxes, pred_boxes=pred_boxes))
        # NWD is computed at the final layer (nwd_only_final=True is the fix)
        losses.update(self.loss_boxes(pred_boxes, targets, indices,
                                       num_boxes, compute_nwd=True))
        all_losses.update(losses)

        # ---- Auxiliary decoder layer outputs ----
        if 'aux_outputs' in outputs:
            for i, aux in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher(
                    aux['pred_logits'], aux['pred_boxes'], targets)
                aux_losses = {}
                aux_losses.update(
                    self.loss_vfl(aux['pred_logits'], targets, aux_indices,
                                   num_boxes, pred_boxes=aux['pred_boxes']))
                # NWD skipped for aux layers when nwd_only_final=True (the fix)
                aux_losses.update(
                    self.loss_boxes(aux['pred_boxes'], targets, aux_indices,
                                    num_boxes,
                                    compute_nwd=(not self.nwd_only_final)))
                for k, v in aux_losses.items():
                    key = f'{k}_aux_{i}'
                    all_losses[key] = v

        # ---- Encoder output losses ----
        if 'enc_topk_logits' in outputs and 'enc_topk_bboxes' in outputs:
            enc_indices = self.matcher(
                outputs['enc_topk_logits'], outputs['enc_topk_bboxes'], targets)
            enc_losses = {}
            enc_losses.update(
                self.loss_vfl(outputs['enc_topk_logits'], targets, enc_indices,
                               num_boxes,
                               pred_boxes=outputs['enc_topk_bboxes']))
            enc_losses.update(
                self.loss_boxes(outputs['enc_topk_bboxes'], targets, enc_indices,
                                num_boxes, compute_nwd=False))
            for k, v in enc_losses.items():
                all_losses[f'{k}_enc'] = v

        # ---- Denoising auxiliary outputs ----
        if 'dn_aux_outputs' in outputs and 'dn_meta' in outputs:
            dn_meta = outputs['dn_meta']
            dn_indices = self._get_dn_indices(targets, dn_meta)
            for i, dn_aux in enumerate(outputs['dn_aux_outputs']):
                dn_losses = {}
                dn_losses.update(
                    self.loss_vfl(dn_aux['pred_logits'], targets, dn_indices,
                                   num_boxes, pred_boxes=dn_aux['pred_boxes']))
                # NWD skipped for dn outputs to avoid gradient overload
                dn_losses.update(
                    self.loss_boxes(dn_aux['pred_boxes'], targets, dn_indices,
                                    num_boxes, compute_nwd=False))
                for k, v in dn_losses.items():
                    all_losses[f'{k}_dn_{i}'] = v

        # Apply weights
        weighted = {}
        for k, v in all_losses.items():
            # Find base name (strip _aux_N, _enc, _dn_N suffixes)
            base_key = k.split('_aux_')[0].split('_enc')[0].split('_dn_')[0]
            w = self.weight_dict.get(base_key, 1.0)
            weighted[k] = v * w

        return weighted

    def _get_dn_indices(self, targets, dn_meta):
        """Build match indices for denoising targets (trivial: dn query i ↔ gt i)."""
        max_gt = dn_meta['max_gt']
        num_groups = dn_meta['num_denoising_groups']
        indices = []
        for tgt in targets:
            n = len(tgt['labels'])
            src = torch.arange(n * num_groups, dtype=torch.long)
            tgt_idx = torch.arange(n, dtype=torch.long).repeat(num_groups)
            indices.append((src, tgt_idx))
        return indices
