"""HybridEncoder with cross-scale feature fusion and LSConv."""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.config import register
from src.nn.utils import get_activation


__all__ = ['HybridEncoder']


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=1, s=1, p=0, g=1, act='relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = get_activation(act) if isinstance(act, str) else act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LSConv(nn.Module):
    """
    Learnable Scaling Convolution.

    Adds a per-channel learnable scale parameter after a standard conv-BN-Act.
    Applied selectively in the encoder bottleneck (cross-scale fusion) to avoid
    feature redundancy with Dynamic Query Grouping and NWD loss optimisation.

    Initialized with scale=1 so it does not disrupt early training.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 1, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        # Per-channel learnable scale, initialized near 1 so training is stable
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x))) * self.scale


class RepVGGBlock(nn.Module):
    """Re-parameterizable VGG-style block."""

    def __init__(self, in_c, out_c, act='relu'):
        super().__init__()
        self.conv1 = ConvBNAct(in_c, out_c, k=3, s=1, p=1, act=act)
        self.conv2 = ConvBNAct(in_c, out_c, k=1, s=1, p=0, act=act)
        self.shortcut = nn.BatchNorm2d(in_c) if in_c == out_c else None

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        if self.shortcut is not None:
            y = y + self.shortcut(x)
        return y


class CSPBlock(nn.Module):
    """Cross-Stage Partial bottleneck block."""

    def __init__(self, in_c, out_c, n=1, shortcut=True, expansion=0.5, act='silu',
                 use_ls_conv: bool = False):
        super().__init__()
        mid_c = int(out_c * expansion)
        self.cv1 = ConvBNAct(in_c, mid_c, k=1, act=act)
        self.cv2 = ConvBNAct(in_c, mid_c, k=1, act=act)
        self.cv3 = ConvBNAct(2 * mid_c, out_c, k=1, act=act)
        self.bottlenecks = nn.Sequential(
            *[self._make_bottleneck(mid_c, mid_c, shortcut, act, use_ls_conv=use_ls_conv)
              for _ in range(n)]
        )

    def _make_bottleneck(self, in_c, out_c, shortcut, act, use_ls_conv=False):
        return _Bottleneck(in_c, out_c, shortcut=shortcut, act=act, use_ls_conv=use_ls_conv)

    def forward(self, x):
        y1 = self.bottlenecks(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))


class _Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, shortcut=True, act='silu', use_ls_conv: bool = False):
        super().__init__()
        if use_ls_conv:
            # Use LSConv selectively in bottleneck to provide learnable scaling
            self.cv1 = ConvBNAct(in_c, out_c, k=3, s=1, p=1, act=act)
            self.cv2 = LSConv(out_c, out_c, kernel_size=3, stride=1, padding=1)
        else:
            self.cv1 = ConvBNAct(in_c, out_c, k=3, s=1, p=1, act=act)
            self.cv2 = ConvBNAct(out_c, out_c, k=3, s=1, p=1, act=act)
        self.shortcut = shortcut and in_c == out_c

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


# ---------------------------------------------------------------------------
# Transformer encoder (AIFI: Attention In Feature Interaction)
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024,
                 dropout: float = 0.0, activation: str = 'gelu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        q = k = src if pos is None else src + pos
        src2, _ = self.self_attn(q, k, src)
        src = self.norm1(src + self.dropout(src2))
        src2 = self.ff(src)
        return self.norm2(src + self.dropout(src2))


class AIFI(nn.Module):
    """Attention In Feature Interaction module (applied at highest-level feature)."""

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int = 1024, dropout: float = 0.0, act: str = 'gelu'):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, act)
             for _ in range(num_layers)]
        )
        self.d_model = d_model

    @staticmethod
    def build_2d_sincos_pos_embed(h: int, w: int, d_model: int,
                                   device: torch.device) -> torch.Tensor:
        """Build 2D sinusoidal positional embedding [1, H*W, d_model]."""
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing='ij',
        )
        pos_dim = d_model // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1.0 / (10000 ** omega)
        out_x = grid_x.flatten()[:, None] * omega[None, :]  # [H*W, D/4]
        out_y = grid_y.flatten()[:, None] * omega[None, :]
        pos = torch.cat([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=-1)
        return pos.unsqueeze(0)  # [1, H*W, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W]"""
        B, C, H, W = x.shape
        flat = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        pos = self.build_2d_sincos_pos_embed(H, W, self.d_model, x.device)
        for layer in self.layers:
            flat = layer(flat, pos)
        return flat.permute(0, 2, 1).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# HybridEncoder
# ---------------------------------------------------------------------------

@register
class HybridEncoder(nn.Module):
    """
    Hybrid Encoder: projects backbone features to uniform channels,
    applies AIFI transformer on the highest-level feature, then
    cross-scale feature fusion (CCFF) with CSP blocks.

    LSConv is applied selectively in the CSP bottleneck of the CCFF stage only,
    avoiding feature redundancy across the full network.

    Args:
        in_channels: backbone output channels per level
        feat_strides: backbone output strides per level
        hidden_dim: encoder hidden dimension
        use_encoder_idx: which levels get AIFI applied (0-indexed, highest first)
        num_encoder_layers: number of transformer encoder layers in AIFI
        nhead: number of attention heads
        dim_feedforward: feedforward dim in AIFI
        dropout: dropout in AIFI
        enc_act: activation in AIFI ('gelu')
        expansion: CSP block expansion ratio
        depth_mult: depth multiplier for CSP blocks
        act: activation for CSP blocks ('silu')
    """

    def __init__(
        self,
        in_channels: list,
        feat_strides: list,
        hidden_dim: int = 256,
        use_encoder_idx: list = None,
        num_encoder_layers: int = 1,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        enc_act: str = 'gelu',
        expansion: float = 1.0,
        depth_mult: int = 1,
        act: str = 'silu',
    ):
        super().__init__()
        if use_encoder_idx is None:
            use_encoder_idx = [2]
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        num_levels = len(in_channels)

        # 1x1 projection to hidden_dim
        self.input_proj = nn.ModuleList(
            ConvBNAct(in_c, hidden_dim, k=1, act=act)
            for in_c in in_channels
        )

        # AIFI transformer encoder on selected levels
        self.encoder = nn.ModuleList()
        for idx in use_encoder_idx:
            self.encoder.append(
                AIFI(hidden_dim, nhead, num_encoder_layers, dim_feedforward, dropout, enc_act)
            )

        # Top-down cross-scale fusion (PAN-style)
        # Uses LSConv selectively in CSP bottleneck of CCFF
        n_bottlenecks = max(round(3 * depth_mult), 1)
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(num_levels - 1):
            self.lateral_convs.append(ConvBNAct(hidden_dim, hidden_dim, k=1, act=act))
            self.fpn_blocks.append(
                CSPBlock(hidden_dim * 2, hidden_dim, n=n_bottlenecks,
                         expansion=expansion, act=act, use_ls_conv=True)
            )

        # Bottom-up path
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(num_levels - 1):
            self.downsample_convs.append(
                ConvBNAct(hidden_dim, hidden_dim, k=3, s=2, p=1, act=act)
            )
            self.pan_blocks.append(
                CSPBlock(hidden_dim * 2, hidden_dim, n=n_bottlenecks,
                         expansion=expansion, act=act, use_ls_conv=False)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: list) -> list:
        """
        Args:
            features: list of [B, Ci, Hi, Wi] from backbone

        Returns:
            list of [B, hidden_dim, Hi, Wi] per level
        """
        # Project all levels to hidden_dim
        proj = [self.input_proj[i](f) for i, f in enumerate(features)]

        # Apply AIFI to selected levels
        enc_idx = 0
        for level_idx in self.use_encoder_idx:
            proj[level_idx] = self.encoder[enc_idx](proj[level_idx])
            enc_idx += 1

        # Top-down FPN path
        inner_outs = [proj[-1]]
        for i in range(len(proj) - 1, 0, -1):
            lat = self.lateral_convs[len(proj) - 1 - i](inner_outs[-1])
            up = F.interpolate(lat, size=proj[i - 1].shape[2:], mode='nearest')
            fused = torch.cat([up, proj[i - 1]], dim=1)
            inner_outs.append(self.fpn_blocks[len(proj) - 1 - i](fused))

        inner_outs = inner_outs[::-1]  # [low, ..., high]

        # Bottom-up PAN path
        outs = [inner_outs[0]]
        for i in range(len(inner_outs) - 1):
            down = self.downsample_convs[i](outs[-1])
            fused = torch.cat([down, inner_outs[i + 1]], dim=1)
            outs.append(self.pan_blocks[i](fused))

        return outs  # [P3, P4, P5]
