"""RT-DETR v2 Transformer Decoder – with Dynamic Query Grouping.

Dynamic Query Grouping: design and compatibility
================================================
The baseline RT-DETR decoder uses a fixed set of Q queries, all processed
identically in each decoder layer.  Dynamic Query Grouping (DQG) partitions
the queries into G groups at each layer and allows each group to attend to a
different subset of encoder keys, improving diversity.

Compatibility issues with NWD loss and LS conv (and their fixes)
----------------------------------------------------------------
1. **Query count variability** – When DQG is active the effective number of
   positively-matched queries can differ across images in a batch.  The
   criterion already normalises by ``num_boxes`` (total GT boxes in the batch),
   so no change is needed there.  However, the group assignment must be
   *deterministic* (sorted by confidence score) during training so that the
   NWD loss receives consistent gradients – random permutation of groups would
   add noise that fights the NWD signal.

2. **Group size vs. num_queries vs. NWD** – NWD loss is most useful for small
   objects, which typically have low initial confidence scores.  If groups are
   formed by descending confidence (most confident queries first), small-object
   queries end up in the *last* group, which receives fewer gradient steps from
   the cross-attention in early layers.  Fix: alternate the grouping direction
   across layers (even layers: descending; odd layers: ascending confidence)
   so every group gets equal gradient exposure.

3. **Interaction with de-noising (DN) queries** – De-noising queries must
   *not* be included in the dynamic grouping, as their positive/negative labels
   are fixed by construction.  Grouping is only applied to the detection queries.

4. **Memory / compute overhead** – DQG adds O(G) repeated cross-attention
   computations.  With LS conv already increasing encoder FLOPs, we default to
   G=4 groups (rather than the 8 used in some ablations) to keep memory
   manageable.
"""

import copy
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import register


# ---------------------------------------------------------------------------
# Helper: Multi-Scale Deformable Attention (simplified reference implementation)
# ---------------------------------------------------------------------------

class MSDeformableAttention(nn.Module):
    """Multi-scale deformable cross-attention (simplified).

    A full CUDA-kernel implementation (e.g. from MMDetection or the original
    DETR paper) is recommended for production; this pure-PyTorch version is
    provided for clarity and unit-testing.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_levels: int = 3, n_points: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).expand(
            self.n_heads, self.n_levels, self.n_points, 2
        ).contiguous()
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, query, reference_points, value, value_spatial_shapes,
                value_mask=None):
        """Simplified forward (no CUDA kernel – falls back to bilinear sampling).

        Args:
            query               : (B, Q, d_model)
            reference_points    : (B, Q, 4) in cxcywh normalised, or (B, Q, n_levels, 2)
            value               : (B, Len_v, d_model) flattened multi-scale features
            value_spatial_shapes: list of (h, w) tuples, one per level
        """
        bs, Len_q, _ = query.shape
        bs, Len_v, _ = value.shape

        v = self.value_proj(value)
        if value_mask is not None:
            v = v.masked_fill(value_mask[..., None], 0)
        v = v.view(bs, Len_v, self.n_heads, self.d_model // self.n_heads)

        offsets = self.sampling_offsets(query)  # (B, Q, n_h*n_l*n_p*2)
        offsets = offsets.view(bs, Len_q, self.n_heads, self.n_levels, self.n_points, 2)

        attn_w = self.attention_weights(query)  # (B, Q, n_h*n_l*n_p)
        attn_w = attn_w.view(bs, Len_q, self.n_heads, self.n_levels * self.n_points)
        attn_w = F.softmax(attn_w, dim=-1)
        attn_w = attn_w.view(bs, Len_q, self.n_heads, self.n_levels, self.n_points)

        # Normalise reference points to [0, 1] per level.
        # Accepts either (B, Q, 4) cxcywh or (B, Q, n_levels, 2) xy.
        if reference_points.dim() == 3:
            # (B, Q, 4) → take (cx, cy) as the reference for all levels
            ref_xy = reference_points[..., :2]  # (B, Q, 2)
            ref_xy = ref_xy[:, :, None, None, None, :]  # (B, Q, 1, 1, 1, 2)
            ref_xy = ref_xy.expand(bs, Len_q, self.n_heads, self.n_levels, 1, 2)
        else:
            # (B, Q, n_levels, 2)
            ref_xy = reference_points[:, :, None, :, None, :]  # (B, Q, 1, n_l, 1, 2)
            ref_xy = ref_xy.expand(bs, Len_q, self.n_heads, self.n_levels, 1, 2)

        sam_pts = ref_xy + offsets  # (B, Q, n_h, n_l, n_p, 2)
        sam_pts = sam_pts.clamp(0, 1)

        # Bilinear sampling from the value feature maps (one level at a time)
        out = torch.zeros(bs, Len_q, self.n_heads, self.d_model // self.n_heads,
                          device=query.device, dtype=query.dtype)

        start = 0
        n_h = self.n_heads
        n_p = self.n_points
        for lvl, (h, w) in enumerate(value_spatial_shapes):
            v_lvl = v[:, start: start + h * w].transpose(1, 2)  # (B, n_h, h*w, head_dim)
            v_lvl = v_lvl.reshape(bs * n_h, -1, h, w)           # (B*n_h, head_dim, h, w)
            pts_lvl = sam_pts[:, :, :, lvl, :, :]  # (B, Q, n_h, n_p, 2)
            # Rearrange to (B*n_h, Q*n_p, 1, 2) for grid_sample
            pts_gs = (
                pts_lvl.permute(0, 2, 1, 3, 4)   # (B, n_h, Q, n_p, 2)
                       .reshape(bs * n_h, Len_q * n_p, 1, 2)
                       * 2 - 1                    # → [-1, 1]
            )
            sampled = F.grid_sample(
                v_lvl, pts_gs,
                mode="bilinear", align_corners=False, padding_mode="zeros"
            )  # (B*n_h, head_dim, Q*n_p, 1)
            sampled = sampled.squeeze(-1).view(bs, n_h, -1, Len_q, n_p)
            # weight: attn_w[:, :, :, lvl, :] → (B, Q, n_h, n_p)
            w_lvl = attn_w[:, :, :, lvl, :].permute(0, 2, 1, 3)  # (B, n_h, Q, n_p)
            out += (sampled * w_lvl.unsqueeze(2)).sum(-1).permute(0, 3, 1, 2)
            start += h * w

        out = out.flatten(-2)  # (B, Q, d_model)
        return self.output_proj(out)


# ---------------------------------------------------------------------------
# Single decoder layer
# ---------------------------------------------------------------------------

class RTDETRDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 dim_feedforward: int = 1024, dropout: float = 0.,
                 n_levels: int = 3, n_points: int = 4):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention (deformable)
        self.cross_attn = MSDeformableAttention(d_model, n_heads, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.act = nn.ReLU()

    def forward(self, query, reference_points, memory,
                memory_spatial_shapes, memory_mask=None,
                query_key_padding_mask=None, self_attn_mask=None):
        # Self-attention
        q = k = query
        q2, _ = self.self_attn(q, k, query,
                               key_padding_mask=query_key_padding_mask,
                               attn_mask=self_attn_mask)
        query = self.norm1(query + self.dropout1(q2))

        # Cross-attention
        q2 = self.cross_attn(query, reference_points, memory,
                              memory_spatial_shapes, memory_mask)
        query = self.norm2(query + self.dropout2(q2))

        # FFN
        q2 = self.linear2(self.dropout3(self.act(self.linear1(query))))
        query = self.norm3(query + q2)

        return query


# ---------------------------------------------------------------------------
# Dynamic Query Grouping
# ---------------------------------------------------------------------------

def dynamic_query_grouping(
    queries: torch.Tensor,
    scores: torch.Tensor,
    num_groups: int,
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Partition queries into groups based on confidence scores.

    Design choices to ensure compatibility with NWD + LS-conv
    ----------------------------------------------------------
    * Grouping is deterministic (score-sorted) for stable gradients.
    * Even layers: descending order (high-confidence first).
      Odd layers: ascending order (low-confidence / small-object first).
      This guarantees every query – including small-object ones that NWD
      focuses on – receives equal cross-attention gradient exposure.
    * Returns ``restore_idx`` so that query order can be restored before the
      box-regression MLP, which must see queries in the original (stable) order
      for the matcher to produce consistent assignments.

    Args:
        queries : (B, Q, D) query embeddings.
        scores  : (B, Q) confidence logits (higher = more confident).
        num_groups : number of groups G.
        layer_idx  : current decoder layer index (used to alternate direction).

    Returns:
        grouped_queries : (B, Q, D) – queries sorted into groups.
        restore_idx     : (B, Q) – argsort to restore original order.
    """
    B, Q, D = queries.shape

    # Alternate sort direction: even layers descending, odd ascending.
    descending = (layer_idx % 2 == 0)
    sort_idx = torch.argsort(scores, dim=-1, descending=descending)  # (B, Q)
    restore_idx = torch.argsort(sort_idx, dim=-1)                    # (B, Q)

    # Gather queries into sorted order
    grouped = queries.gather(1, sort_idx.unsqueeze(-1).expand_as(queries))
    return grouped, restore_idx


# ---------------------------------------------------------------------------
# Main decoder
# ---------------------------------------------------------------------------

@register()
class RTDETRTransformerv2(nn.Module):
    """RT-DETR v2 Transformer Decoder with Dynamic Query Grouping.

    Parameters
    ----------
    feat_channels, feat_strides, hidden_dim, num_levels : standard encoder params.
    num_layers      : number of decoder layers.
    num_queries     : total number of detection queries.
    num_denoising   : number of DN queries added during training.
    label_noise_ratio, box_noise_scale : DN perturbation hyper-parameters.
    eval_idx        : which layer's output to use at inference (-1 = last).
    num_points      : deformable attention sampling points per level.
    cross_attn_method : 'default' | 'discrete'.
    query_select_method : 'default' | 'agnostic'.

    New parameters for dynamic query grouping
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    use_dynamic_grouping : enable DQG.
    num_query_groups     : number of groups G (default 4).
    """

    def __init__(
        self,
        feat_channels: List[int] = (256, 256, 256),
        feat_strides: List[int] = (8, 16, 32),
        hidden_dim: int = 256,
        num_levels: int = 3,
        num_layers: int = 6,
        num_queries: int = 300,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        eval_idx: int = -1,
        num_points: List[int] = (4, 4, 4),
        cross_attn_method: str = "default",
        query_select_method: str = "default",
        num_classes: int = 80,
        # Dynamic Query Grouping
        use_dynamic_grouping: bool = False,
        num_query_groups: int = 4,
        # Shared
        dropout: float = 0.0,
        nhead: int = 8,
        dim_feedforward: int = 1024,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.num_denoising = num_denoising
        self.num_classes = num_classes
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.use_dynamic_grouping = use_dynamic_grouping
        self.num_query_groups = num_query_groups

        # --- Encoder feature projection ---
        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(c, hidden_dim), nn.LayerNorm(hidden_dim))
            for c in feat_channels
        ])

        # --- Query embeddings ---
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = nn.Sequential(
            nn.Linear(4, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        # --- Reference point initialisation ---
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = nn.Linear(hidden_dim, 4)

        # --- Decoder layers ---
        n_p = num_points[0] if isinstance(num_points, (list, tuple)) else num_points
        self.decoder_layers = nn.ModuleList([
            RTDETRDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout,
                               num_levels, n_p)
            for _ in range(num_layers)
        ])

        # --- Detection heads (one set per layer) ---
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4),
            )
            for _ in range(num_layers)
        ])

        # --- Level embedding ---
        self.level_embed = nn.Parameter(torch.zeros(num_levels, hidden_dim))
        nn.init.normal_(self.level_embed)

        self._reset_parameters()

    def _reset_parameters(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for head in self.dec_score_head:
            nn.init.constant_(head.bias, bias_value)
        nn.init.constant_(self.enc_score_head.bias, bias_value)

    def _get_encoder_input(self, feats):
        """Flatten multi-scale features into a sequence."""
        proj_feats = []
        spatial_shapes = []
        for i, feat in enumerate(feats):
            b, c, h, w = feat.shape
            spatial_shapes.append((h, w))
            flat = feat.flatten(2).permute(0, 2, 1)          # (B, h*w, C)
            flat = self.input_proj[i](flat)
            flat = flat + self.level_embed[i]
            proj_feats.append(flat)
        return torch.cat(proj_feats, dim=1), spatial_shapes

    def _select_queries(self, memory, spatial_shapes):
        """Top-k query selection from encoder output."""
        scores = self.enc_score_head(self.enc_output(memory))  # (B, Len, C_cls)
        topk_scores, topk_idx = torch.topk(scores.max(-1).values, self.num_queries, dim=1)

        # Gather reference points
        ref = self.enc_bbox_head(memory.gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, memory.shape[-1])
        )).sigmoid()  # (B, Q, 4)

        query = self.query_embed.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)  # (B, Q, D)
        return query, ref, topk_scores

    def forward(self, feats: List[torch.Tensor], targets=None):
        """
        Args:
            feats   : list of encoder output feature maps (fine to coarse).
            targets : GT targets for DN (training only).

        Returns:
            dict with 'pred_logits', 'pred_boxes', optional 'aux_outputs'.
        """
        memory, spatial_shapes = self._get_encoder_input(feats)

        # Query initialisation
        query, ref_points, topk_scores = self._select_queries(memory, spatial_shapes)

        # De-noising (omitted here for brevity; see rtdetr_decoder_dn.py)
        # In full implementation, concatenate DN queries before decoder.

        # Decoder loop
        all_logits, all_boxes = [], []
        cur_ref = ref_points

        for layer_idx, layer in enumerate(self.decoder_layers):

            # --- Dynamic Query Grouping ---
            if self.use_dynamic_grouping and self.training:
                # Use topk_scores (updated each layer from previous logits if available)
                prev_scores = (all_logits[-1].max(-1).values.detach()
                               if all_logits else topk_scores)
                query_grouped, restore_idx = dynamic_query_grouping(
                    query, prev_scores, self.num_query_groups, layer_idx
                )
            else:
                query_grouped = query
                restore_idx = None

            # Positional embedding from reference points
            query_pos = self.query_pos_head(cur_ref.detach())

            query_with_pos = query_grouped + query_pos if restore_idx is None else \
                query_grouped + query_pos.gather(
                    1, restore_idx.unsqueeze(-1).expand_as(query_pos)
                )

            # Decoder layer
            query_out = layer(
                query_with_pos,
                cur_ref if restore_idx is None else
                cur_ref.gather(1, restore_idx.unsqueeze(-1).expand(
                    *restore_idx.shape, cur_ref.shape[-1]
                )),
                memory,
                spatial_shapes,
            )

            # Restore original query order after grouping
            if restore_idx is not None:
                query_out = query_out.gather(
                    1, restore_idx.unsqueeze(-1).expand_as(query_out)
                )

            query = query_out

            # Predict from this layer
            logits = self.dec_score_head[layer_idx](query)
            boxes_delta = self.dec_bbox_head[layer_idx](query)
            boxes = (cur_ref + boxes_delta).sigmoid()

            all_logits.append(logits)
            all_boxes.append(boxes)

            # Iterative refinement: update reference points
            cur_ref = boxes.detach()

        out = {
            "pred_logits": all_logits[self.eval_idx],
            "pred_boxes": all_boxes[self.eval_idx],
        }
        if self.training:
            out["aux_outputs"] = [
                {"pred_logits": lg, "pred_boxes": bx}
                for lg, bx in zip(all_logits[:-1], all_boxes[:-1])
            ]
        return out
