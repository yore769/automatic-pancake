"""RT-DETR post-processor."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import register


@register()
class RTDETRPostProcessor(nn.Module):
    """Convert raw decoder outputs to final detections.

    Args:
        num_top_queries : keep the top-k queries by score.
        num_classes     : number of detection classes.
    """

    def __init__(self, num_top_queries: int = 300, num_classes: int = 80):
        super().__init__()
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes

    def forward(self, outputs: dict, orig_target_sizes: torch.Tensor):
        """
        Args:
            outputs          : dict with 'pred_logits' (B, Q, C) and
                               'pred_boxes' (B, Q, 4, cxcywh normalised).
            orig_target_sizes: (B, 2) with original (height, width) of each image.

        Returns:
            List of dicts, each with 'scores', 'labels', 'boxes' (xyxy, pixel).
        """
        logits = outputs["pred_logits"]  # (B, Q, C)
        boxes = outputs["pred_boxes"]    # (B, Q, 4)

        scores = logits.sigmoid()
        scores, labels = scores.max(-1)  # (B, Q)

        # Select top-k by score
        topk = min(self.num_top_queries, scores.shape[1])
        topk_scores, topk_idx = torch.topk(scores, topk, dim=1)
        topk_labels = labels.gather(1, topk_idx)
        topk_boxes = boxes.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))

        # Convert normalised cxcywh → xyxy in pixel space
        results = []
        for b_score, b_label, b_box, size in zip(
            topk_scores, topk_labels, topk_boxes, orig_target_sizes
        ):
            h, w = size.unbind(-1)
            cx, cy, bw, bh = b_box.unbind(-1)
            x1 = (cx - 0.5 * bw) * w
            y1 = (cy - 0.5 * bh) * h
            x2 = (cx + 0.5 * bw) * w
            y2 = (cy + 0.5 * bh) * h
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
            results.append({
                "scores": b_score,
                "labels": b_label,
                "boxes": boxes_xyxy,
            })
        return results
