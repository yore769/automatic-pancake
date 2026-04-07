"""RT-DETRv2 post-processor."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.config import register
from .utils import box_cxcywh_to_xyxy


@register
class RTDETRPostProcessor(nn.Module):
    """
    Convert model outputs to COCO-format detections.

    Args:
        num_classes: number of object classes
        use_focal_loss: if True, use sigmoid for scores; else softmax
        num_top_queries: keep this many top detections per image
    """

    def __init__(
        self,
        num_classes: int = 80,
        use_focal_loss: bool = True,
        num_top_queries: int = 300,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries

    @torch.no_grad()
    def forward(self, outputs: dict, orig_sizes: torch.Tensor) -> list:
        """
        Args:
            outputs: dict with 'pred_logits' [B, N, C] and 'pred_boxes' [B, N, 4]
            orig_sizes: [B, 2] (H, W) of original images

        Returns:
            list of dicts with 'scores', 'labels', 'boxes' per image
        """
        logits = outputs['pred_logits']  # [B, N, C]
        boxes = outputs['pred_boxes']    # [B, N, 4] cxcywh in [0,1]

        if self.use_focal_loss:
            scores = torch.sigmoid(logits)  # [B, N, C]
            scores, labels = scores.max(-1)  # [B, N]
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(-1)

        # Keep top-k
        if self.num_top_queries < scores.shape[1]:
            top_scores, top_idx = scores.topk(self.num_top_queries, dim=1)
            scores = top_scores
            labels = labels.gather(1, top_idx)
            boxes = boxes.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, 4))

        results = []
        for i in range(len(orig_sizes)):
            h, w = orig_sizes[i].tolist()
            # Convert cxcywh [0,1] to xyxy in pixel coords
            box_xyxy = box_cxcywh_to_xyxy(boxes[i])
            scale = torch.tensor([w, h, w, h], device=box_xyxy.device, dtype=box_xyxy.dtype)
            box_xyxy = box_xyxy * scale
            results.append({
                'scores': scores[i],
                'labels': labels[i],
                'boxes': box_xyxy,
            })
        return results
