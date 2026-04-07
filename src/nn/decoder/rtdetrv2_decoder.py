"""RT-DETR v2 Transformer Decoder with Dynamic Query Grouping.

Dynamic Query Grouping
----------------------
In standard RT-DETR each decoder query attends to all encoder positions
independently.  For dense-small-object datasets (VisDrone) many queries
address the same spatial location, wasting capacity and producing redundant
detections.

Dynamic Query Grouping clusters the initial query embeddings at the start of
each forward pass into G groups based on their predicted reference-point
locations.  Queries within the same group share a group-level context
embedding that is added to (not replacing) the individual query embedding.
This encourages diversity between groups while maintaining intra-group
coherence.

Integration Fixes (why the original combined model underperformed)
------------------------------------------------------------------
1. **Gradient isolation**: group embeddings are computed from detached
   reference-point predictions so that grouping does not compete with the
   NWD/GIoU optimisation objectives.
2. **Warm-up period**: grouping is disabled for the first ``warmup_epochs``
   epochs to let the individual query embeddings stabilise before grouping
   is imposed.  This prevents the gradient interference that plagued the
   original combined training.
3. **Additive (not substitutive) context**: group context is *added* to
   queries rather than replacing them, preserving per-query identity which
   is crucial for the NWD Loss Gaussian modelling.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List

from src.core.workspace import register

__all__ = ['RTDETRTransformerv2']


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _build_mlp(input_dim, hidden_dim, output_dim, num_layers):
    layers = []
    for i in range(num_layers):
        in_d = input_dim if i == 0 else hidden_dim
        out_d = output_dim if i == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(in_d, out_d))
        if i < num_layers - 1:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class MultiScaleDeformableAttention(nn.Module):
    """Simplified multi-scale deformable cross-attention."""

    def __init__(self, d_model, n_heads, n_levels, n_points):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, reference_points, value, value_spatial_shapes,
                value_level_start_index, value_padding_mask=None):
        B, Lq, C = query.shape
        B, Lv, C = value.shape

        value_proj = self.value_proj(value)
        sampling_offsets = self.sampling_offsets(query).view(
            B, Lq, self.n_heads, self.n_levels, self.n_points, 2)
        attn_weights = self.attention_weights(query).view(
            B, Lq, self.n_heads, self.n_levels * self.n_points)
        attn_weights = F.softmax(attn_weights, dim=-1).view(
            B, Lq, self.n_heads, self.n_levels, self.n_points)

        # Compute sampling locations
        if reference_points.shape[-1] == 2:
            ref = reference_points[:, :, None, :, None, :]
            sampling_locations = ref + sampling_offsets / (
                torch.stack([s[1] for s in value_spatial_shapes] +
                            [s[0] for s in value_spatial_shapes], dim=-1
                            ).to(query) if False else
                value_spatial_shapes[0][0].float()
            )
        else:
            ref_xy = reference_points[:, :, None, :, None, :2]
            ref_wh = reference_points[:, :, None, :, None, 2:] * 0.5
            sampling_locations = ref_xy + sampling_offsets * ref_wh

        # Gather values at sampling locations (bilinear interpolation)
        output = self._ms_deform_attn_sample(
            value_proj, value_spatial_shapes, value_level_start_index,
            sampling_locations, attn_weights)
        return self.output_proj(output)

    def _ms_deform_attn_sample(self, value, spatial_shapes,
                                level_start_index, sampling_locations,
                                attention_weights):
        """Fallback pure-PyTorch implementation."""
        B, Lv, _ = value.shape
        B, Lq, H, L, P, _ = sampling_locations.shape

        value_list = []
        for l_idx, (H_l, W_l) in enumerate(spatial_shapes):
            start = level_start_index[l_idx]
            end = start + H_l * W_l
            v_l = value[:, start:end, :].reshape(B, H_l, W_l, -1)
            value_list.append(v_l.permute(0, 3, 1, 2))

        sampling_grids = 2 * sampling_locations - 1  # [-1, 1]
        output = torch.zeros(B, Lq, self.n_heads * self.head_dim,
                             device=value.device, dtype=value.dtype)

        for l_idx, feat_l in enumerate(value_list):
            # feat_l: (B, C, H_l, W_l)
            # sampling_grids for level l: (B, Lq, H, P, 2)
            grid = sampling_grids[:, :, :, l_idx, :, :].reshape(
                B, Lq * H, P, 2)  # (B, Lq*H, P, 2)
            # Expand feat_l per head
            feat_exp = feat_l.unsqueeze(1).expand(
                -1, H, -1, -1, -1).reshape(B * H, self.head_dim, *feat_l.shape[-2:])
            grid_exp = grid.reshape(B * H, Lq, P, 2)
            sampled = F.grid_sample(feat_exp, grid_exp,
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=False)
            # sampled: (B*H, head_dim, Lq, P)
            sampled = sampled.reshape(B, H, self.head_dim, Lq, P)
            weights_l = attention_weights[:, :, :, l_idx, :]  # (B, Lq, H, P)
            weights_l = weights_l.permute(0, 2, 3, 1)  # (B, H, P, Lq)
            weighted = (sampled * weights_l.unsqueeze(2)).sum(dim=-1)
            # weighted: (B, H, head_dim, Lq)
            output += weighted.permute(0, 3, 1, 2).reshape(B, Lq, -1)

        return output


class DecoderLayer(nn.Module):
    """Single RT-DETR v2 decoder layer."""

    def __init__(self, d_model=256, n_heads=8, n_levels=3, n_points=4,
                 dim_feedforward=1024, dropout=0.):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (deformable)
        self.cross_attn = MultiScaleDeformableAttention(
            d_model, n_heads, n_levels, n_points)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # FFN
        self.ffn1 = nn.Linear(d_model, dim_feedforward)
        self.ffn2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, tgt, reference_points, memory, memory_spatial_shapes,
                memory_level_start_index, attn_mask=None,
                memory_key_padding_mask=None, query_pos=None):
        # Self-attn
        q = k = tgt if query_pos is None else tgt + query_pos
        sa_out, _ = self.self_attn(q, k, tgt, attn_mask=attn_mask)
        tgt = self.norm1(tgt + self.dropout1(sa_out))

        # Cross-attn
        ca_out = self.cross_attn(
            tgt if query_pos is None else tgt + query_pos,
            reference_points, memory, memory_spatial_shapes,
            memory_level_start_index, memory_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout2(ca_out))

        # FFN
        ff = self.ffn2(self.dropout3(self.act(self.ffn1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(ff))
        return tgt


# ---------------------------------------------------------------------------
# Dynamic Query Grouping
# ---------------------------------------------------------------------------

class DynamicQueryGrouping(nn.Module):
    """Dynamic Query Grouping module.

    Clusters decoder queries into G groups based on their reference-point
    proximity and assigns a shared group context embedding to each query.

    Integration Fixes Applied Here
    --------------------------------
    * Reference points used for grouping are **detached** from the computation
      graph (``ref.detach()``).  This breaks the gradient path between the
      grouping decision and the NWD/GIoU losses, preventing the gradient
      competition that caused the combined model to underperform.
    * Group context is *added* to queries (not substituted) so the individual
      query identity is preserved for per-object loss computation.
    * Grouping is skipped during the warm-up phase (controlled externally by
      the decoder via ``self.enable_grouping``).
    * Group context embedding is initialised near zero (small normal) so it
      starts as a near-identity operation and learns progressively.

    Args:
        d_model:    query embedding dimension
        num_groups: number of groups G
    """

    def __init__(self, d_model: int, num_groups: int = 10):
        super().__init__()
        self.num_groups = num_groups
        self.d_model = d_model

        # Per-group context embeddings
        self.group_embed = nn.Embedding(num_groups, d_model)
        # Near-zero init: group context starts as small perturbation
        nn.init.normal_(self.group_embed.weight, mean=0.0, std=0.01)

        # Light projection to get group logits from reference points
        self.group_proj = nn.Linear(2, num_groups, bias=False)
        nn.init.normal_(self.group_proj.weight, std=0.02)

    def forward(self, queries: Tensor, reference_points: Tensor) -> Tensor:
        """Add group context embeddings to queries.

        Args:
            queries:          (B, Nq, d_model)  decoder query embeddings
            reference_points: (B, Nq, 2 or 4)  predicted reference points

        Returns:
            queries with group context added: (B, Nq, d_model)
        """
        # Use only x,y of reference points; detach to isolate gradient flow
        ref_xy = reference_points[..., :2].detach()  # (B, Nq, 2)

        # Compute soft group assignments via similarity to group prototypes
        logits = self.group_proj(ref_xy)  # (B, Nq, G)
        soft_assign = F.softmax(logits, dim=-1)  # (B, Nq, G)

        # Retrieve weighted group context
        all_ctx = self.group_embed.weight  # (G, d_model)
        ctx = torch.einsum('bng,gd->bnd', soft_assign, all_ctx)  # (B, Nq, d_model)

        return queries + ctx  # additive: preserves individual query identity


# ---------------------------------------------------------------------------
# Denoising helpers
# ---------------------------------------------------------------------------

def get_contrastive_denoising_training_group(targets, num_classes, num_queries,
                                              label_noise_ratio=0.5,
                                              box_noise_scale=1.0,
                                              num_denoising=100):
    """Create denoising training group for RT-DETR v2."""
    if num_denoising <= 0 or targets is None:
        return None, None, None, None

    device = targets[0]['boxes'].device
    max_gt = max(len(t['labels']) for t in targets) if targets else 0
    if max_gt == 0:
        return None, None, None, None

    B = len(targets)
    num_dn_groups = num_denoising // max(max_gt, 1)
    num_dn_groups = max(num_dn_groups, 1)
    dn_queries = num_dn_groups * max_gt

    # Build dn labels and boxes
    dn_labels = torch.zeros(B, dn_queries, dtype=torch.long, device=device)
    dn_boxes = torch.zeros(B, dn_queries, 4, device=device)

    for i, tgt in enumerate(targets):
        n = len(tgt['labels'])
        if n == 0:
            continue
        rep_labels = tgt['labels'].repeat(num_dn_groups)[:dn_queries]
        rep_boxes = tgt['boxes'].repeat(num_dn_groups, 1)[:dn_queries]
        dn_labels[i, :len(rep_labels)] = rep_labels
        dn_boxes[i, :len(rep_boxes)] = rep_boxes

    # Add noise
    if label_noise_ratio > 0:
        mask = torch.rand(B, dn_queries, device=device) < label_noise_ratio
        dn_labels[mask] = torch.randint(0, num_classes, [mask.sum()],
                                         device=device)
    if box_noise_scale > 0:
        noise = (torch.rand_like(dn_boxes) - 0.5) * box_noise_scale
        dn_boxes = (dn_boxes + noise).clamp(0., 1.)

    # Attention mask: dn queries cannot see regular queries
    total = num_queries + dn_queries
    attn_mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    attn_mask[:dn_queries, dn_queries:] = True  # dn cannot see regular
    attn_mask[dn_queries:, :dn_queries] = True  # regular cannot see dn

    dn_meta = {
        'num_denoising': dn_queries,
        'num_denoising_groups': num_dn_groups,
        'max_gt': max_gt,
    }
    return dn_labels, dn_boxes, attn_mask, dn_meta


# ---------------------------------------------------------------------------
# RTDETRTransformerv2
# ---------------------------------------------------------------------------

@register
class RTDETRTransformerv2(nn.Module):
    """RT-DETR v2 Transformer Decoder.

    Key args for combined improvements:
        use_dynamic_grouping (bool): enable Dynamic Query Grouping
        num_groups (int):            number of query groups
        grouping_warmup_iters (int): number of training iterations before
                                     grouping is activated (default 2000).
                                     Keeping grouping off during warmup prevents
                                     early interference with the NWD gradient.
    """

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=(256, 256, 256),
                 feat_strides=(8, 16, 32),
                 num_levels=3,
                 num_layers=6,
                 num_points=(4, 4, 4),
                 num_heads=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 eval_idx=-1,
                 # query selection
                 cross_attn_method='default',
                 query_select_method='default',
                 # Dynamic Query Grouping
                 use_dynamic_grouping=False,
                 num_groups=10,
                 grouping_warmup_iters=2000):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_levels = num_levels
        self.num_layers = num_layers
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

        # ---- Dynamic Query Grouping ----
        self.use_dynamic_grouping = use_dynamic_grouping
        self.grouping_warmup_iters = grouping_warmup_iters
        # _iter tracks training steps for warm-up.  It is reset by reset_iter()
        # when training is restarted in the same process (e.g. fine-tuning).
        self._iter = 0
        if use_dynamic_grouping:
            self.dqg = DynamicQueryGrouping(hidden_dim, num_groups)
        else:
            self.dqg = None

        # ---- Decoder layers ----
        layer = DecoderLayer(
            hidden_dim, num_heads, num_levels,
            num_points[0] if isinstance(num_points, (list, tuple)) else num_points,
            dim_feedforward, dropout)
        self.layers = _get_clones(layer, num_layers)

        # ---- Prediction heads ----
        # Shared heads replicated per layer
        self.class_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_layers)
        ])
        self.bbox_head = nn.ModuleList([
            _build_mlp(hidden_dim, hidden_dim, 4, 3)
            for _ in range(num_layers)
        ])

        # ---- Embeddings ----
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.level_embed = nn.Embedding(num_levels, hidden_dim)

        # ---- Encoder output projection ----
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = _build_mlp(hidden_dim, hidden_dim, 4, 3)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # init class heads with bias for focal-loss stability
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for head in self.class_head:
            nn.init.constant_(head.bias, bias_value)
        nn.init.constant_(self.enc_score_head.bias, bias_value)

    def reset_iter(self):
        """Reset the training iteration counter.

        Call this before restarting training (e.g. fine-tuning) so that the
        Dynamic Query Grouping warm-up period is re-applied from the start of
        the new training run.
        """
        self._iter = 0

    def _get_encoder_feats(self, feats, feat_strides):
        """Flatten multi-scale encoder features."""
        spatial_shapes = []
        level_start_index = []
        flat_feats = []
        level_pos = []

        start = 0
        for i, feat in enumerate(feats):
            B, C, H, W = feat.shape
            spatial_shapes.append((H, W))
            level_start_index.append(start)
            start += H * W
            flat = feat.flatten(2).permute(0, 2, 1)  # (B, HW, C)
            flat_feats.append(flat)
            # Level embedding
            lv_emb = self.level_embed.weight[i].view(1, 1, -1).expand(B, H * W, -1)
            level_pos.append(lv_emb)

        memory = torch.cat(flat_feats, dim=1)  # (B, sum(HW), C)
        level_pos = torch.cat(level_pos, dim=1)
        level_start_index = torch.tensor(level_start_index,
                                          dtype=torch.long,
                                          device=memory.device)
        return memory, spatial_shapes, level_start_index, level_pos

    def _get_reference_points(self, memory, spatial_shapes):
        """Compute initial reference points from top-K encoder predictions."""
        enc_out = self.enc_output(memory)
        scores = self.enc_score_head(enc_out)  # (B, L, num_classes)
        boxes = self.enc_bbox_head(enc_out)    # (B, L, 4)
        boxes = boxes.sigmoid()

        # Select top-K queries
        topk_scores = scores.max(dim=-1)[0]  # (B, L)
        topk_idx = topk_scores.topk(self.num_queries, dim=1)[1]  # (B, Nq)

        ref_points = boxes.gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))  # (B, Nq, 4)
        query_feat = enc_out.gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, enc_out.shape[-1]))

        return ref_points, query_feat, enc_out, scores, boxes

    def forward(self, feats, targets=None):
        B = feats[0].shape[0]
        device = feats[0].device

        memory, spatial_shapes, level_start_index, level_pos = \
            self._get_encoder_feats(feats, self.num_layers)

        ref_points, query_feat, enc_out, enc_scores, enc_boxes = \
            self._get_reference_points(memory, spatial_shapes)

        # Memory + level positional embedding
        memory = memory + level_pos

        # ---- Denoising training group ----
        dn_labels = dn_boxes = attn_mask = dn_meta = None
        if self.training and targets is not None and self.num_denoising > 0:
            dn_labels, dn_boxes, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(
                    targets, self.num_classes, self.num_queries,
                    self.label_noise_ratio, self.box_noise_scale,
                    self.num_denoising)

        if dn_labels is not None and dn_boxes is not None:
            dn_queries = dn_labels.shape[1]
            dn_query_embed = self.query_embed.weight[:dn_queries].unsqueeze(0) \
                .expand(B, -1, -1)
            queries = torch.cat([dn_query_embed, query_feat], dim=1)
            ref = torch.cat([
                dn_boxes,
                ref_points,
            ], dim=1)
        else:
            queries = query_feat
            ref = ref_points
            dn_queries = 0

        # Reference points for cross-attention (x,y only)
        ref_pts_xy = ref[..., :2]

        # ---- Decoder layers ----
        all_cls = []
        all_box = []

        tgt = queries
        cur_ref = ref

        for i, layer in enumerate(self.layers):
            # Apply Dynamic Query Grouping after warm-up
            if (self.use_dynamic_grouping and self.dqg is not None
                    and self._iter >= self.grouping_warmup_iters
                    and self.training):
                # Separate dn and regular parts for grouping
                if dn_queries > 0:
                    tgt_regular = tgt[:, dn_queries:]
                    ref_regular = cur_ref[:, dn_queries:, :2]
                    tgt_regular = self.dqg(tgt_regular, ref_regular)
                    tgt = torch.cat([tgt[:, :dn_queries], tgt_regular], dim=1)
                else:
                    tgt = self.dqg(tgt, cur_ref[..., :2])

            # Multi-scale reference points (one per level)
            ref_pts_per_level = cur_ref[..., :2].unsqueeze(2).expand(
                -1, -1, self.num_levels, -1)

            tgt = layer(
                tgt=tgt,
                reference_points=ref_pts_per_level,
                memory=memory,
                memory_spatial_shapes=spatial_shapes,
                memory_level_start_index=level_start_index,
                attn_mask=attn_mask,
            )

            # Predict from current layer
            cls_logits = self.class_head[i](tgt)
            delta_box = self.bbox_head[i](tgt)

            # Update reference points iteratively.
            # cur_ref always carries 4 coordinates (cx, cy, w, h) after the
            # first layer; assert this to catch shape bugs early.
            assert cur_ref.shape[-1] == 4, (
                f'Expected ref points of size 4, got {cur_ref.shape[-1]}')
            cur_ref_xy = (_inverse_sigmoid(cur_ref[..., :2]) +
                          delta_box[..., :2]).sigmoid()
            cur_ref_wh = (delta_box[..., 2:]).sigmoid()
            cur_box = torch.cat([cur_ref_xy, cur_ref_wh], dim=-1)

            all_cls.append(cls_logits)
            all_box.append(cur_box)
            cur_ref = cur_box.detach()

        if self.training:
            self._iter += 1

        # Split dn / regular outputs
        if dn_queries > 0:
            dn_cls = [c[:, :dn_queries] for c in all_cls]
            dn_box = [b[:, :dn_queries] for b in all_box]
            all_cls = [c[:, dn_queries:] for c in all_cls]
            all_box = [b[:, dn_queries:] for b in all_box]
        else:
            dn_cls = dn_box = None

        out = {
            'pred_logits': all_cls[self.eval_idx],
            'pred_boxes': all_box[self.eval_idx],
            'aux_outputs': [
                {'pred_logits': c, 'pred_boxes': b}
                for c, b in zip(all_cls[:-1], all_box[:-1])
            ],
            'enc_topk_logits': enc_scores,
            'enc_topk_bboxes': enc_boxes,
        }
        if dn_cls is not None:
            out['dn_aux_outputs'] = [
                {'pred_logits': c, 'pred_boxes': b}
                for c, b in zip(dn_cls, dn_box)
            ]
            out['dn_meta'] = dn_meta

        return out
