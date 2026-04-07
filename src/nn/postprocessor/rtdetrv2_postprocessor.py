"""RT-DETR v2 Postprocessor."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.workspace import register

__all__ = ['RTDETRPostProcessor']


@register
class RTDETRPostProcessor(nn.Module):
    """Convert decoder outputs to final detection results.

    Args:
        num_top_queries: maximum detections to return per image
        num_classes:     number of detection classes
    """

    def __init__(self, num_top_queries=300, num_classes=80):
        super().__init__()
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes

    def forward(self, outputs, orig_target_sizes):
        """Post-process model outputs.

        Args:
            outputs:           model output dict
            orig_target_sizes: (B, 2) original image sizes (H, W)

        Returns:
            list of dicts with 'boxes', 'scores', 'labels'
        """
        pred_logits = outputs['pred_logits']   # (B, Nq, C)
        pred_boxes = outputs['pred_boxes']     # (B, Nq, 4) (cx,cy,w,h) normalised

        scores = pred_logits.sigmoid()
        scores, pred_labels = scores.max(dim=-1)  # (B, Nq)

        # Top-K selection
        topk = min(self.num_top_queries, scores.shape[-1])
        topk_scores, topk_idx = scores.topk(topk, dim=1)
        topk_labels = pred_labels.gather(1, topk_idx)
        topk_boxes_norm = pred_boxes.gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))

        # Denormalise boxes
        # ---- Unused scale variable removed; denormalization done per-coordinate ----
        results = []
        for i in range(len(orig_target_sizes)):
            h, w = orig_target_sizes[i].unbind(-1)
            cx, cy, bw, bh = topk_boxes_norm[i].unbind(-1)
            boxes = torch.stack([
                (cx - bw / 2) * w,
                (cy - bh / 2) * h,
                (cx + bw / 2) * w,
                (cy + bh / 2) * h,
            ], dim=-1)
            results.append({
                'boxes': boxes,
                'scores': topk_scores[i],
                'labels': topk_labels[i],
            })

        return results

    def deploy(self):
        """Return a deploy-mode version (same in this implementation)."""
        return self
