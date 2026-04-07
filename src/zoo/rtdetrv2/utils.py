"""Utility functions for RT-DETRv2."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise GIoU between two sets of boxes (xyxy format)."""
    assert boxes1.shape[-1] == 4 and boxes2.shape[-1] == 4

    # Intersection
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    iou = inter / union.clamp(min=1e-6)

    # Enclosing box
    enc_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enc_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enc_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    enc_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    enc_area = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0)

    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return giou


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
) -> torch.Tensor:
    """Sigmoid focal loss."""
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal = alpha_t * (1 - p_t) ** gamma * ce
    if reduction == 'mean':
        return focal.mean()
    elif reduction == 'sum':
        return focal.sum()
    return focal


def varifocal_loss(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    reduction: str = 'sum',
) -> torch.Tensor:
    """VariFocal Loss."""
    pred_score = torch.sigmoid(pred_logits)
    weight = alpha * (pred_score - targets).abs().pow(gamma) * (targets == 0).float() \
           + targets
    loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none') * weight
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    return loss


def get_contrastive_denoising_training_group(
    targets,
    num_classes: int,
    num_queries: int,
    class_embed: nn.Module,
    num_denoising: int = 100,
    label_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
):
    """
    Build contrastive denoising queries and targets.

    Returns:
        input_query_label: [B, num_dn_queries, hidden_dim]
        input_query_bbox: [B, num_dn_queries, 4]
        attn_mask: [total_queries, total_queries]
        dn_meta: dict with denoising metadata
    """
    if num_denoising <= 0:
        return None, None, None, None

    device = targets[0]['boxes'].device if targets and len(targets[0]['boxes']) > 0 else None

    gt_boxes = [t['boxes'] for t in targets]  # list of [Ni, 4] cxcywh
    gt_labels = [t['labels'] for t in targets]  # list of [Ni]

    batch_size = len(targets)
    max_gt = max((len(b) for b in gt_boxes), default=0)
    if max_gt == 0:
        return None, None, None, None

    num_groups = max(1, num_denoising // max_gt)
    num_dn_queries = num_groups * max_gt * 2  # positive + negative

    # ---- Build padded GT tensors ----
    pad_labels = torch.zeros(batch_size, max_gt, dtype=torch.long, device=device)
    pad_boxes = torch.zeros(batch_size, max_gt, 4, device=device)
    pad_mask = torch.ones(batch_size, max_gt, dtype=torch.bool, device=device)

    for i, (lbl, box) in enumerate(zip(gt_labels, gt_boxes)):
        n = len(lbl)
        if n == 0:
            continue
        pad_labels[i, :n] = lbl
        pad_boxes[i, :n] = box
        pad_mask[i, :n] = False  # valid

    # Repeat for groups
    pad_labels = pad_labels.unsqueeze(1).repeat(1, num_groups, 1)   # [B, G, N]
    pad_boxes = pad_boxes.unsqueeze(1).repeat(1, num_groups, 1, 1)  # [B, G, N, 4]
    pad_mask = pad_mask.unsqueeze(1).repeat(1, num_groups, 1)        # [B, G, N]

    # ---- Positive group: noised labels / boxes ----
    known_labels_pos = pad_labels.clone()
    known_boxes_pos = pad_boxes.clone()

    if label_noise_ratio > 0:
        noise_mask = torch.rand_like(pad_labels.float()) < label_noise_ratio
        rand_labels = torch.randint_like(pad_labels, 0, num_classes)
        known_labels_pos = torch.where(noise_mask, rand_labels, known_labels_pos)

    if box_noise_scale > 0:
        diff = torch.zeros_like(known_boxes_pos)
        diff[..., :2] = known_boxes_pos[..., 2:] / 2   # half wh
        diff[..., 2:] = known_boxes_pos[..., 2:]
        noise = (torch.rand_like(known_boxes_pos) * 2 - 1) * box_noise_scale * diff
        known_boxes_pos = (known_boxes_pos + noise).clamp(0, 1)

    # ---- Negative group: same labels but further noised boxes ----
    known_labels_neg = pad_labels.clone()
    known_boxes_neg = pad_boxes.clone()
    if box_noise_scale > 0:
        diff = torch.zeros_like(known_boxes_neg)
        diff[..., :2] = known_boxes_neg[..., 2:] / 2
        diff[..., 2:] = known_boxes_neg[..., 2:]
        noise = (torch.rand_like(known_boxes_neg) * 2 - 1) * box_noise_scale * 2.0 * diff
        known_boxes_neg = (known_boxes_neg + noise).clamp(0, 1)

    # Interleave pos/neg per GT instance: ordering is [G0_N0_pos, G0_N0_neg, G0_N1_pos, ...]
    # Use stack on last dim so both labels and boxes have consistent ordering.

    # Labels: [B, G, N] → stack pos/neg → [B, G, N, 2] → [B, G*N*2]
    known_labels = torch.stack([known_labels_pos, known_labels_neg], dim=-1)
    known_labels = known_labels.reshape(batch_size, -1)  # [B, G*N*2]

    # Boxes: [B, G, N, 4] → stack pos/neg → [B, G, N, 2, 4] → [B, G*N*2, 4]
    known_boxes = torch.stack([known_boxes_pos, known_boxes_neg], dim=-2)
    known_boxes = known_boxes.reshape(batch_size, -1, 4)  # [B, G*N*2, 4]

    # Padding mask: [B, G, N] → [B, G, N, 2] → [B, G*N*2]
    pad_mask_flat = pad_mask.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(batch_size, -1)

    # Clean GT boxes/labels for DN loss (same clean target for both pos and neg)
    clean_gt_boxes = pad_boxes.unsqueeze(-2).expand(-1, -1, -1, 2, -1).reshape(batch_size, -1, 4)
    clean_gt_labels = pad_labels.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(batch_size, -1)

    # Embed labels for query content
    label_embed = class_embed(known_labels)  # [B, G*N*2, hidden]

    # Attention mask: dn queries can attend to each other only within same group
    total_queries = num_dn_queries + num_queries
    attn_mask = torch.zeros(total_queries, total_queries, dtype=torch.bool, device=device)

    group_size = max_gt * 2  # pos+neg per group
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        # Block cross-group attention in DN region
        attn_mask[start:end, :start] = True
        attn_mask[start:end, end:num_dn_queries] = True
        # Block DN attending to matching queries
        attn_mask[start:end, num_dn_queries:] = True
    # Block matching queries attending to DN queries
    attn_mask[num_dn_queries:, :num_dn_queries] = True

    dn_meta = {
        'dn_num_split': [num_dn_queries, num_queries],
        'dn_num_group': num_groups,
        'dn_num_queries': num_dn_queries,
        'pad_mask': pad_mask_flat,           # [B, G*N*2] True=padding
        'gt_boxes': clean_gt_boxes,          # [B, G*N*2, 4] clean GT for loss
        'gt_labels': clean_gt_labels,        # [B, G*N*2] clean GT labels
    }

    return label_embed, known_boxes, attn_mask, dn_meta
