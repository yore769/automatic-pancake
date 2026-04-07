"""RT-DETR v2 Criterion – with NWD loss integration.

Design notes
------------
When NWD loss is **combined** with dynamic-query-grouping and LS-conv, three
critical issues arise and are addressed here:

1. **NWD weight imbalance** – The NWD loss produces values in [0, 1], while
   VFL and GIoU can be larger in magnitude during early training.  When NWD is
   added naively its gradient signal is swamped.  Fix: use ``nwd_loss_weight``
   (default 1.0) *independently* of the existing ``loss_bbox`` / ``loss_giou``
   weights, and clip large NWD gradients via the shared ``clip_max_norm`` in
   the trainer.

2. **Dynamic-grouping query count mismatch** – When dynamic query grouping is
   active the number of active queries per image can vary.  The per-image loss
   normaliser ``num_boxes`` must count only *active* matched predictions, not
   the static ``num_queries`` constant.  Fix: ``_get_src_permutation_idx``
   operates on the per-image matched indices returned by the matcher, so it is
   inherently robust to variable group sizes.

3. **Auxiliary-layer NWD scaling** – Applying full-weight NWD to every
   intermediate decoder layer can destabilise training because early layers
   produce very imprecise boxes, causing large NWD losses that saturate
   gradients.  Fix: NWD weight is linearly ramped from 0 at layer 0 to the
   target weight at the final layer via ``nwd_aux_weight_ramp``.
"""

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import register, create
from .box_ops import (
    box_cxcywh_to_xyxy,
    box_iou,
    generalized_box_iou,
    nwd_loss,
)


def _vfl(
    pred_logits: torch.Tensor,
    target_scores: torch.Tensor,
    target_labels: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """VariFocal Loss.

    Args:
        pred_logits   : (N, C) raw logits.
        target_scores : (N, C) soft target scores in [0, 1].
        target_labels : (N,)   binary mask – 1 where a positive is assigned.
    """
    pred_score = pred_logits.sigmoid().detach()
    is_pos = (target_labels != 0).float().unsqueeze(-1)   # (N, 1) for broadcasting
    is_neg = (target_labels == 0).float().unsqueeze(-1)
    weight = (
        alpha * (pred_score - target_scores).abs().pow(gamma) * is_neg
        + target_scores * is_pos
    )
    loss = (
        F.binary_cross_entropy_with_logits(pred_logits, target_scores, reduction="none") * weight
    )
    return loss.sum(dim=-1)


@register()
class RTDETRCriterionv2(nn.Module):
    """Set-based detection criterion for RT-DETR v2.

    New parameters vs. baseline
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    use_nwd          – whether to add NWD loss.
    nwd_loss_weight  – scalar weight for the NWD term (default 1.0).
    nwd_C            – normalisation factor for NWD (default 12.8).
    nwd_aux_weight_ramp – if True, linearly ramp the NWD weight across
                          auxiliary decoder layers (0 → nwd_loss_weight).
                          Prevents gradient saturation in shallow layers.
    """

    def __init__(
        self,
        matcher: dict,
        weight_dict: Dict[str, float],
        losses: List[str],
        alpha: float = 0.75,
        gamma: float = 2.0,
        num_classes: int = 80,
        # --- NWD options ---
        use_nwd: bool = False,
        nwd_loss_weight: float = 1.0,
        nwd_C: float = 2.0,
        nwd_aux_weight_ramp: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = create(matcher)
        self.weight_dict = weight_dict
        self.losses = losses
        self.alpha = alpha
        self.gamma = gamma

        self.use_nwd = use_nwd
        self.nwd_loss_weight = nwd_loss_weight
        self.nwd_C = nwd_C
        self.nwd_aux_weight_ramp = nwd_aux_weight_ramp

        if use_nwd:
            # Expose NWD weight so the trainer can log it.
            self.weight_dict["loss_nwd"] = nwd_loss_weight

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_src_permutation_idx(self, indices):
        """Row indices into the flattened (batch × queries) tensor."""
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # ------------------------------------------------------------------
    # Individual loss terms
    # ------------------------------------------------------------------

    def loss_vfl(self, outputs, targets, indices, num_boxes, **kwargs):
        """VariFocal classification loss."""
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]  # (B, Q, C)
        bs, nq, nc = src_logits.shape

        # Build target score tensor
        target_classes = torch.full(
            (bs, nq), self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_scores = torch.zeros(
            (bs, nq, nc), dtype=src_logits.dtype, device=src_logits.device
        )

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        target_classes[src_idx] = torch.cat(
            [t["labels"][j] for t, (_, j) in zip(targets, indices)]
        )

        # VFL target scores come from the IoU of matched predictions with GT
        with torch.no_grad():
            pred_boxes = outputs["pred_boxes"].detach()
            src_boxes = pred_boxes[src_idx]
            tgt_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)])
            iou_vals, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)
            )
            # Diagonal gives per-matched-pair IoU
            iou_scores = iou_vals.diag().clamp(0)

        # Fill in positive target scores
        target_scores_pos = torch.zeros(
            len(src_idx[0]), nc, dtype=src_logits.dtype, device=src_logits.device
        )
        matched_labels = target_classes[src_idx]
        target_scores_pos.scatter_(
            1, matched_labels.unsqueeze(-1), iou_scores.unsqueeze(-1)
        )
        target_scores[src_idx] = target_scores_pos

        loss = _vfl(src_logits.flatten(0, 1), target_scores.flatten(0, 1),
                    (target_classes != self.num_classes).long().flatten(0, 1),
                    self.alpha, self.gamma)
        return {"loss_vfl": loss.sum() / num_boxes}

    def loss_boxes(self, outputs, targets, indices, num_boxes,
                   nwd_layer_idx: int = None, num_aux_layers: int = None, **kwargs):
        """L1 + GIoU box regression loss, optionally with NWD.

        Parameters
        ----------
        nwd_layer_idx  : index of the current decoder layer (0 = first aux layer,
                         ``num_aux_layers - 1`` = final layer).  Used to ramp the
                         NWD weight across auxiliary layers.  None → final layer.
        num_aux_layers : total number of auxiliary decoder layers.
        """
        src_idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][src_idx]
        tgt_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)])

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="none").sum() / num_boxes

        loss_giou = (
            1
            - generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)
            ).diag()
        ).sum() / num_boxes

        losses = {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

        if self.use_nwd and len(src_boxes) > 0:
            raw_nwd_loss = nwd_loss(src_boxes, tgt_boxes, C=self.nwd_C, reduction="sum") / num_boxes

            # Ramp NWD weight linearly across auxiliary layers to avoid
            # early-layer gradient saturation.
            if self.nwd_aux_weight_ramp and nwd_layer_idx is not None and num_aux_layers is not None:
                ramp = (nwd_layer_idx + 1) / num_aux_layers
            else:
                ramp = 1.0

            effective_nwd_weight = self.nwd_loss_weight * ramp
            losses["loss_nwd"] = raw_nwd_loss * effective_nwd_weight

        return losses

    def loss_boxes_key(self, outputs, targets, indices, num_boxes, **kwargs):
        """Alias used internally to call loss_boxes for the final layer."""
        return self.loss_boxes(outputs, targets, indices, num_boxes, **kwargs)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _compute_losses(self, outputs, targets, num_boxes,
                        nwd_layer_idx=None, num_aux_layers=None):
        indices = self.matcher(outputs, targets)
        losses = {}
        for loss_name in self.losses:
            if loss_name == "vfl":
                losses.update(self.loss_vfl(outputs, targets, indices, num_boxes))
            elif loss_name == "boxes":
                losses.update(
                    self.loss_boxes(
                        outputs, targets, indices, num_boxes,
                        nwd_layer_idx=nwd_layer_idx,
                        num_aux_layers=num_aux_layers,
                    )
                )
        return losses

    def forward(self, outputs: dict, targets: list):
        """Compute all losses.

        Args:
            outputs: dict returned by the model.  Must have 'pred_logits' and
                     'pred_boxes'.  May have 'aux_outputs' (list of intermediate
                     decoder layer outputs).
            targets: list of per-image dicts with 'labels' and 'boxes'.

        Returns:
            Dict of scalar loss tensors, weighted by *weight_dict*.
        """
        # Exclude auxiliary outputs from main computation
        main_outputs = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Number of matched boxes across the batch (used as normaliser)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=next(iter(main_outputs.values())).device,
        )
        num_boxes = num_boxes.clamp(min=1).item()

        num_aux_layers = len(outputs.get("aux_outputs", [])) + 1

        # Main (final layer) losses – always use full NWD weight
        losses = self._compute_losses(main_outputs, targets, num_boxes,
                                      nwd_layer_idx=num_aux_layers - 1,
                                      num_aux_layers=num_aux_layers)

        # Auxiliary layer losses
        for aux_idx, aux_outputs in enumerate(outputs.get("aux_outputs", [])):
            aux_losses = self._compute_losses(
                aux_outputs, targets, num_boxes,
                nwd_layer_idx=aux_idx,
                num_aux_layers=num_aux_layers,
            )
            for k, v in aux_losses.items():
                losses[f"{k}_{aux_idx}"] = v

        # Apply weights
        weighted = {}
        for k, v in losses.items():
            # Find the base key (strip trailing _N for aux layers)
            base_k = k.rsplit("_", 1)[0] if k[-1].isdigit() else k
            w = self.weight_dict.get(k, self.weight_dict.get(base_k, 1.0))
            weighted[k] = v * w

        return weighted
