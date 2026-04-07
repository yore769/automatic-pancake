"""RT-DETRv2 transformer decoder with Dynamic Query Grouping."""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.config import register
from src.nn.utils import (
    MLP, inverse_sigmoid, bias_init_with_prob,
    deformable_attention_core_func,
)
from .utils import get_contrastive_denoising_training_group


__all__ = ['RTDETRTransformerv2']


# ---------------------------------------------------------------------------
# Dynamic Query Grouping
# ---------------------------------------------------------------------------

class DynamicQueryGrouping(nn.Module):
    """
    Groups decoder content queries dynamically during training.

    During training, queries are softly assigned to groups based on their
    hidden-state similarity. A small group-conditioned scale factor (0.1)
    encourages specialisation without conflicting with NWD loss optimization
    (which needs consistent box-space gradients).

    During inference, hard assignment is used but the scale factor is clamped
    small so it does not distort predictions.

    Args:
        num_queries: total number of queries
        num_groups: number of query groups
        hidden_dim: query dimension
    """

    def __init__(self, num_queries: int, num_groups: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.num_groups = num_groups
        self.group_embed = nn.Linear(hidden_dim, num_groups, bias=False)
        # Temperature parameter: learnable, clamped to ≥0.1
        self.temperature = nn.Parameter(torch.ones(1))

        nn.init.normal_(self.group_embed.weight, std=0.01)

    def forward(self, query_embeds: torch.Tensor, boxes: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query_embeds: [B, N, D] content query embeddings
            boxes: [B, N, 4] predicted boxes (optional, kept for future use)

        Returns:
            query_embeds: [B, N, D] scaled query embeddings
        """
        logits = self.group_embed(query_embeds)  # [B, N, G]
        temp = self.temperature.clamp(min=0.1)

        if self.training:
            # Soft assignment with temperature scaling
            assignments = torch.softmax(logits / temp, dim=-1)  # [B, N, G]
        else:
            # Hard assignment at inference (still small scale factor)
            assignments = torch.zeros_like(logits).scatter_(
                -1, logits.argmax(-1, keepdim=True), 1.0
            )

        # Group-conditioned scale: projection back to query space
        # Use a small coefficient (0.1) to avoid competing with NWD/GIoU gradients
        group_scale = (assignments @ self.group_embed.weight)  # [B, N, D]
        group_scale = group_scale.mean(-1, keepdim=True)       # [B, N, 1] scalar per query

        return query_embeds * (1 + group_scale * 0.1)


# ---------------------------------------------------------------------------
# Multi-scale deformable attention
# ---------------------------------------------------------------------------

class MSDeformableAttention(nn.Module):
    """Multi-scale deformable attention."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 num_levels: int = 3, num_points: int = 4,
                 method: str = 'default'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        self.method = method

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels, self.num_points, 1
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.flatten())
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        value_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, Lq, _ = query.shape
        B, Lv, _ = value.shape

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask.unsqueeze(-1), 0.)
        value = value.view(B, Lv, self.num_heads, self.head_dim)

        offsets = self.sampling_offsets(query)
        offsets = offsets.view(B, Lq, self.num_heads, self.num_levels, self.num_points, 2)

        attn_weights = self.attention_weights(query)
        attn_weights = attn_weights.view(
            B, Lq, self.num_heads, self.num_levels * self.num_points
        )
        attn_weights = F.softmax(attn_weights, dim=-1).view(
            B, Lq, self.num_heads, self.num_levels, self.num_points
        )

        # reference_points: [B, Lq, num_levels, 2] or [B, Lq, num_levels, 4]
        if reference_points.shape[-1] == 2:
            ref = reference_points[:, :, None, :, None, :]  # [B, Lq, 1, L, 1, 2]
            norm = torch.stack(
                [value_spatial_shapes[:, 1], value_spatial_shapes[:, 0]], dim=-1
            ).float()
            norm = norm[None, None, None, :, None, :]  # [1, 1, 1, L, 1, 2]
            sampling_locs = ref + offsets / norm
        elif reference_points.shape[-1] == 4:
            ref = reference_points[:, :, None, :, None, :2]
            wh = reference_points[:, :, None, :, None, 2:] * 0.5
            sampling_locs = ref + offsets / self.num_points * wh
        else:
            raise ValueError(f"Unexpected reference_points shape: {reference_points.shape}")

        output = deformable_attention_core_func(
            value, value_spatial_shapes, sampling_locs, attn_weights
        )
        return self.output_proj(output)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class RTDETRDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = 'relu',
        num_levels: int = 3,
        num_points: int = 4,
        cross_attn_method: str = 'default',
    ):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention (deformable)
        self.cross_attn = MSDeformableAttention(
            d_model, n_head, num_levels, num_points, method=cross_attn_method
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        tgt: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        memory_spatial_shapes: torch.Tensor,
        attn_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        query_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        # Self-attention
        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt, attn_mask=attn_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # Cross-attention
        tgt2 = self.cross_attn(
            tgt if query_pos is None else tgt + query_pos,
            reference_points,
            memory,
            memory_spatial_shapes,
            memory_mask,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # FFN
        tgt2 = self.ff2(self.dropout3(self.act(self.ff1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


# ---------------------------------------------------------------------------
# RTDETRTransformerv2
# ---------------------------------------------------------------------------

@register
class RTDETRTransformerv2(nn.Module):
    """
    RT-DETRv2 Transformer Decoder.

    Features:
    - Multi-scale deformable attention
    - Contrastive denoising training
    - Dynamic Query Grouping applied to content queries

    Args:
        feat_channels: encoder output channels per level
        feat_strides: encoder output strides per level
        hidden_dim: hidden dimension
        num_levels: number of feature levels
        num_layers: number of decoder layers
        num_queries: number of object queries
        num_denoising: number of denoising queries
        label_noise_ratio: label noise in denoising
        box_noise_scale: box noise in denoising
        eval_idx: which decoder layer to use at eval (-1 = last)
        num_points: deformable attention points per level
        cross_attn_method: 'default' or 'discrete'
        query_select_method: 'default' or 'agnostic'
    """

    def __init__(
        self,
        feat_channels: list = None,
        feat_strides: list = None,
        hidden_dim: int = 256,
        num_levels: int = 3,
        num_layers: int = 6,
        num_queries: int = 300,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        eval_idx: int = -1,
        num_points: list = None,
        cross_attn_method: str = 'default',
        query_select_method: str = 'default',
        # Decoder layer params
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        # Dynamic Query Grouping
        num_query_groups: int = 4,
        # Number of object classes - must match criterion/config
        num_classes: int = 80,
    ):
        super().__init__()
        if feat_channels is None:
            feat_channels = [256, 256, 256]
        if feat_strides is None:
            feat_strides = [8, 16, 32]
        if num_points is None:
            num_points = [4] * num_levels

        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.query_select_method = query_select_method

        # Level embed
        self.level_embed = nn.Parameter(torch.zeros(num_levels, hidden_dim))

        # Number of classes (set now, not lazily)
        self.num_classes = num_classes

        # Decoder layers
        _points = num_points if isinstance(num_points[0], int) else [p[0] for p in num_points]
        self.decoder_layers = nn.ModuleList([
            RTDETRDecoderLayer(
                d_model=hidden_dim,
                n_head=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_levels=num_levels,
                num_points=sum(_points),
                cross_attn_method=cross_attn_method,
            )
            for _ in range(num_layers)
        ])

        # Query embedding (positional)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # Content queries (learned)
        self.content_queries = nn.Embedding(num_queries, hidden_dim)

        # Prediction heads - linear layers shared across layers
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Dynamic Query Grouping
        self.dynamic_query_grouping = DynamicQueryGrouping(
            num_queries=num_queries,
            num_groups=num_query_groups,
            hidden_dim=hidden_dim,
        )

        # Build prediction heads with the known num_classes
        self._build_heads(num_classes)

        # Initialise positional and query params
        self._init_params()

    def _build_heads(self, num_classes: int):
        """Build prediction heads."""
        self.num_classes = num_classes
        self.enc_score_head = nn.Linear(self.hidden_dim, num_classes)
        self.enc_bbox_head = MLP(self.hidden_dim, self.hidden_dim, 4, num_layers=3)

        self.dec_score_head = nn.ModuleList(
            [nn.Linear(self.hidden_dim, num_classes) for _ in range(self.num_layers)]
        )
        self.dec_bbox_head = nn.ModuleList(
            [MLP(self.hidden_dim, self.hidden_dim, 4, num_layers=3)
             for _ in range(self.num_layers)]
        )

        # Denoising label embed
        self.dn_label_embed = nn.Embedding(num_classes + 1, self.hidden_dim)

        self._init_heads()

    def _init_params(self):
        nn.init.normal_(self.level_embed)
        nn.init.normal_(self.content_queries.weight)

    def _init_heads(self):
        prior_prob = 0.01
        bias = bias_init_with_prob(prior_prob)
        nn.init.constant_(self.enc_score_head.bias, bias)
        for layer in self.dec_score_head:
            nn.init.constant_(layer.bias, bias)
        # Init bbox heads to predict identity (output near zero offset)
        for head in [self.enc_bbox_head] + list(self.dec_bbox_head):
            nn.init.constant_(head.layers[-1].weight, 0.)
            nn.init.constant_(head.layers[-1].bias, 0.)

    # ------------------------------------------------------------------
    # Memory preparation
    # ------------------------------------------------------------------

    def _prepare_memory(self, encoder_feats: list):
        """
        Flatten multi-scale encoder features into a single memory tensor.

        Returns:
            memory: [B, sum(Hi*Wi), hidden_dim]
            spatial_shapes: [num_levels, 2]
        """
        B = encoder_feats[0].shape[0]
        memory_list = []
        spatial_shapes = []
        for l, feat in enumerate(encoder_feats):
            B, C, H, W = feat.shape
            spatial_shapes.append([H, W])
            feat_flat = feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            feat_flat = feat_flat + self.level_embed[l]
            memory_list.append(feat_flat)
        memory = torch.cat(memory_list, dim=1)  # [B, L, C]
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long,
                                       device=memory.device)
        return memory, spatial_shapes

    def _get_reference_points(self, memory: torch.Tensor,
                               spatial_shapes: torch.Tensor) -> tuple:
        """
        Compute initial reference points from encoder output.
        Returns top-K query proposals.
        """
        B, L, C = memory.shape
        # Encoder output
        output = self.enc_output(memory)  # [B, L, C]
        enc_logits = self.enc_score_head(output)  # [B, L, num_classes]
        enc_boxes = self.enc_bbox_head(output).sigmoid()  # [B, L, 4]

        # Select top-K queries
        if self.query_select_method == 'agnostic':
            scores = enc_logits.max(-1)[0]  # [B, L]
        else:
            scores = enc_logits.sigmoid().max(-1)[0]  # [B, L]

        _, topk_idx = scores.topk(self.num_queries, dim=1)  # [B, K]
        ref_points = enc_boxes.gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, 4)
        )  # [B, K, 4]
        topk_feat = output.gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, C)
        )  # [B, K, C]

        # Encoder aux output
        topk_logits = enc_logits.gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, enc_logits.shape[-1])
        )

        return ref_points, topk_feat, topk_logits

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, encoder_feats: list, targets: list = None):
        """
        Args:
            encoder_feats: list of [B, C, Hi, Wi]
            targets: list of target dicts (only used in training)

        Returns:
            dict with 'pred_logits', 'pred_boxes', optionally
            'aux_outputs', 'dn_outputs', 'dn_meta'
        """
        B = encoder_feats[0].shape[0]
        device = encoder_feats[0].device

        # Prepare memory
        memory, spatial_shapes = self._prepare_memory(encoder_feats)

        # Get query proposals from encoder
        ref_points, topk_feat, _ = self._get_reference_points(memory, spatial_shapes)
        # ref_points: [B, K, 4], topk_feat: [B, K, C]

        # Content queries (learned + encoder-proposed feature)
        content_q = self.content_queries.weight.unsqueeze(0).expand(B, -1, -1)
        content_q = content_q + topk_feat  # Blend learned + encoder proposals

        # Reference points for positional encoding
        ref_sig = ref_points.detach()  # [B, K, 4]

        # Denoising queries
        dn_label_embed = None
        dn_bbox = None
        attn_mask = None
        dn_meta = None

        if self.training and targets is not None and self.num_denoising > 0:
            dn_label_embed, dn_bbox, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(
                    targets,
                    num_classes=self.num_classes,
                    num_queries=self.num_queries,
                    class_embed=self.dn_label_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                )

        if dn_label_embed is not None and dn_bbox is not None:
            # Concatenate DN queries with matching queries
            content_q = torch.cat([dn_label_embed, content_q], dim=1)
            ref_sig = torch.cat([dn_bbox, ref_sig], dim=1)

        # Apply Dynamic Query Grouping to content queries (matching part only)
        # We apply it only to the matching queries (not DN) to avoid conflicting
        # with the structured DN training signal
        n_dn = dn_label_embed.shape[1] if dn_label_embed is not None else 0
        if n_dn > 0:
            matching_q = content_q[:, n_dn:]
            matching_q = self.dynamic_query_grouping(matching_q, ref_sig[:, n_dn:])
            content_q = torch.cat([content_q[:, :n_dn], matching_q], dim=1)
        else:
            content_q = self.dynamic_query_grouping(content_q, ref_sig)

        # Build reference points per level for deformable attention
        # [B, N, num_levels, 2] from [B, N, 4] cxcywh
        ref_cxcywh = ref_sig
        ref_xy = ref_cxcywh[..., :2]
        ref_multi_level = ref_xy.unsqueeze(2).expand(
            -1, -1, self.num_levels, -1
        )  # [B, N, L, 2]

        # Decoder loop
        tgt = content_q
        aux_outputs = []
        dn_outputs_list = []

        for i, layer in enumerate(self.decoder_layers):
            # Query positional encoding from reference points
            query_pos = self.query_pos_head(ref_cxcywh)

            tgt = layer(
                tgt,
                ref_multi_level,
                memory,
                spatial_shapes,
                attn_mask=attn_mask,
                query_pos=query_pos,
            )

            # Predict boxes and classes
            ref_cxcywh_detach = ref_cxcywh.detach()
            delta_box = self.dec_bbox_head[i](tgt)
            new_ref = (inverse_sigmoid(ref_cxcywh_detach) + delta_box).sigmoid()

            # Update ref for next layer
            ref_cxcywh = new_ref.detach()
            ref_multi_level = ref_cxcywh[..., :2].unsqueeze(2).expand(
                -1, -1, self.num_levels, -1
            )

            logits = self.dec_score_head[i](tgt)

            if i < self.num_layers - 1 or i == self.eval_idx:
                # Split DN / matching
                if n_dn > 0:
                    dn_out = {
                        'pred_logits': logits[:, :n_dn],
                        'pred_boxes': new_ref[:, :n_dn],
                    }
                    match_out = {
                        'pred_logits': logits[:, n_dn:],
                        'pred_boxes': new_ref[:, n_dn:],
                    }
                    dn_outputs_list.append(dn_out)
                else:
                    match_out = {'pred_logits': logits, 'pred_boxes': new_ref}

                if i < self.num_layers - 1:
                    aux_outputs.append(match_out)

        # Final output
        final_logits = logits[:, n_dn:] if n_dn > 0 else logits
        final_boxes = new_ref[:, n_dn:] if n_dn > 0 else new_ref

        out = {
            'pred_logits': final_logits,
            'pred_boxes': final_boxes,
        }
        if aux_outputs:
            out['aux_outputs'] = aux_outputs
        if dn_outputs_list:
            out['dn_outputs'] = dn_outputs_list[-1]  # use last DN output
            out['dn_meta'] = dn_meta

        return out
