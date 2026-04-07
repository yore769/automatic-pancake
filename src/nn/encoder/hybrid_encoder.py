"""HybridEncoder: AIFI + CCFM for RT-DETR v2.

The HybridEncoder fuses multi-scale backbone features through:
  - AIFI  (Attention-based Intra-scale Feature Interaction)
  - CCFM  (Cross-scale Channel Feature Merge)

LS Conv Integration
-------------------
LS Conv is applied selectively **only inside CCFM RepBlocks** (the
intermediate fusion layers), NOT on the initial 1×1 projection convolutions.
This design choice is critical for maintaining performance when combined with
NWD Loss and Dynamic Query Grouping:

  • The 1×1 projections need full-rank channel mixing to produce the
    per-object feature representations that NWD Loss models as Gaussians.
  • The CCFM fusion layers are good candidates for LS Conv because their
    role is purely spatial aggregation across scales, not fine-grained
    channel discrimination.

Setting ``use_ls_conv=True`` replaces only those CCFM 3×3 convolutions.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.workspace import register
from src.nn.backbone.ls_conv import ConvNormAct, make_conv

__all__ = ['HybridEncoder']


# ---------------------------------------------------------------------------
# Transformer components for AIFI
# ---------------------------------------------------------------------------

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        return self.attn(q, k, v, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask)[0]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.,
                 activation='gelu', normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout)
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(inplace=True)
        self.normalize_before = normalize_before

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None):
        attn_out = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout(attn_out))
        ff_out = self.ff2(self.dropout(self.act(self.ff1(src))))
        src = self.norm2(src + self.dropout(ff_out))
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        return self.forward_post(src, src_mask, src_key_padding_mask)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)
        return output


# ---------------------------------------------------------------------------
# CCFM building blocks
# ---------------------------------------------------------------------------

class RepBlock(nn.Module):
    """Repeated Conv block used in CCFM.

    When ``use_ls_conv=True`` each 3×3 conv is replaced with an LSConv,
    reducing parameters while keeping gradient flow via skip connections.
    """

    def __init__(self, in_ch, out_ch, num_repeats=3, act='silu',
                 use_ls_conv=False):
        super().__init__()
        blocks = [make_conv(in_ch, out_ch, 3, act=act,
                             use_ls_conv=use_ls_conv)]
        for _ in range(num_repeats - 1):
            blocks.append(make_conv(out_ch, out_ch, 3, act=act,
                                    use_ls_conv=use_ls_conv))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class CCFM(nn.Module):
    """Cross-scale Channel Feature Merge module."""

    def __init__(self, in_channels, hidden_dim, num_blocks=3,
                 act='silu', use_ls_conv=False):
        super().__init__()
        # Top-down path
        self.reduce_p5 = ConvNormAct(in_channels[2], hidden_dim, 1, act=act)
        self.reduce_p4 = ConvNormAct(in_channels[1], hidden_dim, 1, act=act)
        self.reduce_p3 = ConvNormAct(in_channels[0], hidden_dim, 1, act=act)

        # Fusion blocks (these use LS Conv when enabled)
        self.rep_p4 = RepBlock(hidden_dim * 2, hidden_dim, num_blocks,
                               act=act, use_ls_conv=use_ls_conv)
        self.rep_p3 = RepBlock(hidden_dim * 2, hidden_dim, num_blocks,
                               act=act, use_ls_conv=use_ls_conv)
        # Bottom-up path
        self.downsample_p3 = ConvNormAct(hidden_dim, hidden_dim, 3, stride=2, act=act)
        self.rep_n4 = RepBlock(hidden_dim * 2, hidden_dim, num_blocks,
                               act=act, use_ls_conv=use_ls_conv)
        self.downsample_n4 = ConvNormAct(hidden_dim, hidden_dim, 3, stride=2, act=act)
        self.rep_n5 = RepBlock(hidden_dim * 2, hidden_dim, num_blocks,
                               act=act, use_ls_conv=use_ls_conv)

    def forward(self, feats):
        p3, p4, p5 = feats

        # Projections
        p5_r = self.reduce_p5(p5)
        p4_r = self.reduce_p4(p4)
        p3_r = self.reduce_p3(p3)

        # Top-down
        p5_up = F.interpolate(p5_r, size=p4_r.shape[-2:], mode='nearest')
        p4_td = self.rep_p4(torch.cat([p4_r, p5_up], dim=1))

        p4_up = F.interpolate(p4_td, size=p3_r.shape[-2:], mode='nearest')
        p3_out = self.rep_p3(torch.cat([p3_r, p4_up], dim=1))

        # Bottom-up
        p3_dn = self.downsample_p3(p3_out)
        p4_out = self.rep_n4(torch.cat([p4_td, p3_dn], dim=1))

        p4_dn = self.downsample_n4(p4_out)
        p5_out = self.rep_n5(torch.cat([p5_r, p4_dn], dim=1))

        return [p3_out, p4_out, p5_out]


# ---------------------------------------------------------------------------
# AIFI
# ---------------------------------------------------------------------------

class AIFI(nn.Module):
    """Attention-based Intra-scale Feature Interaction."""

    def __init__(self, hidden_dim, nhead, num_layers, dim_feedforward,
                 dropout=0., act='gelu'):
        super().__init__()
        layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, act)
        self.encoder = TransformerEncoder(layer, num_layers)
        self.hidden_dim = hidden_dim

    @staticmethod
    def build_2d_sincos_pos_embed(h, w, embed_dim, temperature=10000.,
                                  device='cpu'):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing='ij',
        )
        assert embed_dim % 4 == 0, 'embed_dim must be divisible by 4'
        d = embed_dim // 4
        omega = torch.arange(d, dtype=torch.float32, device=device) / d
        omega = 1.0 / (temperature ** omega)

        out_y = grid_y.flatten()[:, None] * omega[None, :]
        out_x = grid_x.flatten()[:, None] * omega[None, :]

        pos = torch.cat([out_x.sin(), out_x.cos(),
                         out_y.sin(), out_y.cos()], dim=1)
        return pos.unsqueeze(0)  # (1, H*W, embed_dim)

    def forward(self, feat):
        B, C, H, W = feat.shape
        pos = self.build_2d_sincos_pos_embed(H, W, C, device=feat.device)
        x = feat.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        x = self.encoder(x + pos)
        return x.permute(0, 2, 1).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# HybridEncoder
# ---------------------------------------------------------------------------

@register
class HybridEncoder(nn.Module):
    """Hybrid Encoder: AIFI on the highest-level feature + CCFM across scales.

    Args:
        in_channels:      channels from backbone for each returned feature
        feat_strides:     spatial strides for each feature
        hidden_dim:       inner dimension for encoder
        use_encoder_idx:  which feature levels to apply AIFI to (0-based idx
                          into in_channels list)
        num_encoder_layers: number of AIFI transformer layers
        nhead:            attention heads in AIFI
        dim_feedforward:  FFN dimension in AIFI
        dropout:          dropout rate
        enc_act:          AIFI activation
        expansion:        channel expansion for CCFM
        depth_mult:       depth multiplier for CCFM RepBlocks
        act:              CCFM activation
        use_ls_conv:      apply LSConv inside CCFM RepBlocks (LS Conv improvement)
    """

    def __init__(self,
                 in_channels=(512, 1024, 2048),
                 feat_strides=(8, 16, 32),
                 hidden_dim=256,
                 use_encoder_idx=(2,),
                 num_encoder_layers=1,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 enc_act='gelu',
                 expansion=1.0,
                 depth_mult=1,
                 act='silu',
                 use_ls_conv=False):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx

        # Input projections (always standard 1×1 conv — preserving full channel mixing)
        self.input_proj = nn.ModuleList([
            ConvNormAct(ch, hidden_dim, 1, act=act)
            for ch in in_channels
        ])

        # AIFI encoders (applied only to selected scales)
        self.encoders = nn.ModuleList()
        for _ in use_encoder_idx:
            layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward,
                                            dropout, enc_act)
            self.encoders.append(
                TransformerEncoder(layer, num_encoder_layers))

        # CCFM with optional LS Conv
        aifi_out = [hidden_dim] * len(in_channels)
        num_blocks = max(round(3 * depth_mult), 1)
        self.ccfm = CCFM(aifi_out, hidden_dim, num_blocks=num_blocks,
                         act=act, use_ls_conv=use_ls_conv)

        self.out_channels = [hidden_dim] * len(in_channels)
        self.out_strides = feat_strides

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [proj(f) for proj, f in zip(self.input_proj, feats)]

        # Apply AIFI to selected scales
        enc_iter = iter(self.encoders)
        for i in self.use_encoder_idx:
            B, C, H, W = proj_feats[i].shape
            pos = AIFI.build_2d_sincos_pos_embed(
                H, W, C, device=proj_feats[i].device)
            x = proj_feats[i].flatten(2).permute(0, 2, 1)
            enc = next(enc_iter)
            x = enc(x + pos)
            proj_feats[i] = x.permute(0, 2, 1).reshape(B, C, H, W)

        # CCFM fusion
        return self.ccfm(proj_feats)
