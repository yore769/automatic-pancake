"""HybridEncoder for RT-DETR v2 – with optional LS Convolution.

Compatibility notes (combined training)
----------------------------------------
When LS conv, NWD loss, and dynamic query grouping are all enabled together:

* BN momentum is forced to ``ls_bn_momentum`` (default 0.03) for all
  RepCSP / LS-conv BN layers so that rapidly changing feature statistics
  (driven by NWD's sensitivity to small boxes) do not de-stabilise running
  mean/variance.

* The ``alpha`` gate in each LSConv starts at 0, giving NWD loss and dynamic
  grouping time to converge before the long-range features start contributing.
"""

import copy
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import register
from .ls_conv import LSConv, ConvBNAct


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------

class AIFI(nn.Module):
    """Attention-based Intra-scale Feature Interaction (transformer layer)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0., act: str = "gelu"):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        act_fn = nn.GELU() if act == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _build_2d_sin_pos_embed(h: int, w: int, d_model: int,
                                 device: torch.device) -> torch.Tensor:
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2-D sinusoidal PE"
        y = torch.arange(h, device=device).float()
        x = torch.arange(w, device=device).float()
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        # Use d_model // 4 frequencies for each axis × 2 (sin/cos) = d_model // 2 per axis
        # Total: 2 × (d_model // 2) = d_model
        dim_quarter = d_model // 4
        div = torch.exp(
            torch.arange(0, dim_quarter, device=device).float()
            * (-math.log(10000.0) / dim_quarter)
        )
        gy_flat = gy.flatten()  # (H*W,)
        gx_flat = gx.flatten()
        # each embed: (H*W, 2*dim_quarter) = (H*W, d_model//2)
        embed_y = torch.stack([torch.sin(gy_flat[:, None] * div),
                                torch.cos(gy_flat[:, None] * div)], dim=2).flatten(1)
        embed_x = torch.stack([torch.sin(gx_flat[:, None] * div),
                                torch.cos(gx_flat[:, None] * div)], dim=2).flatten(1)
        return torch.cat([embed_y, embed_x], dim=1)  # (H*W, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)          # (B, H*W, C)
        pos = self._build_2d_sin_pos_embed(h, w, c, x.device)   # (H*W, C)
        q = k = x_flat + pos.unsqueeze(0)
        x2, _ = self.attn(q, k, x_flat)
        x_flat = self.norm1(x_flat + x2)
        x_flat = self.norm2(x_flat + self.ffn(x_flat))
        return x_flat.permute(0, 2, 1).reshape(b, c, h, w)


class RepCSPLayer(nn.Module):
    """CSP-style fusion layer, with optional LS conv in the bottleneck."""

    def __init__(self, in_channels: int, out_channels: int,
                 num_blocks: int = 3, expansion: float = 1.0,
                 use_ls_conv: bool = False, ls_kernel: int = 3,
                 act: nn.Module = None, bn_momentum: float = 0.1):
        super().__init__()
        hidden = int(out_channels * expansion)
        act_fn = act if act is not None else nn.SiLU()

        self.cv1 = ConvBNAct(in_channels, hidden, k=1, act=act_fn, bn_momentum=bn_momentum)
        self.cv2 = ConvBNAct(in_channels, hidden, k=1, act=act_fn, bn_momentum=bn_momentum)
        self.cv3 = ConvBNAct(2 * hidden, out_channels, k=1, act=act_fn, bn_momentum=bn_momentum)

        blocks = []
        for _ in range(num_blocks):
            if use_ls_conv:
                blocks.append(LSConv(hidden, hidden, kernel_size=ls_kernel,
                                     act=act_fn, bn_momentum=bn_momentum))
            else:
                blocks.append(ConvBNAct(hidden, hidden, k=3, act=act_fn,
                                        bn_momentum=bn_momentum))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.blocks(self.cv1(x)), self.cv2(x)], dim=1))


# ---------------------------------------------------------------------------
# HybridEncoder
# ---------------------------------------------------------------------------

@register()
class HybridEncoder(nn.Module):
    """Hybrid Encoder: AIFI transformer on the finest used scale + RepCSP PANet.

    Parameters matching the YAML config
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    in_channels      : backbone output channels for each return index.
    feat_strides     : spatial strides of those feature maps.
    hidden_dim       : internal channel dimension.
    use_encoder_idx  : which feature level to apply AIFI to (e.g. [2] = largest stride).
    num_encoder_layers : number of stacked AIFI layers.
    nhead            : attention heads in AIFI.
    dim_feedforward  : FFN width in AIFI.
    dropout          : dropout in AIFI.
    enc_act          : activation in AIFI ('gelu' | 'relu').
    expansion        : CSP hidden-channel expansion factor.
    depth_mult       : multiplier for RepCSP block count.
    act              : activation in CSP / LS-conv ('silu' | 'relu').

    New parameters for LS conv
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    use_ls_conv      : replace standard 3×3 convs in RepCSP with LSConv.
    ls_kernel_size   : local kernel size for LSConv (default 3).
    ls_bn_momentum   : BN momentum for LS-conv layers (default 0.03).
    """

    def __init__(
        self,
        in_channels: List[int] = (512, 1024, 2048),
        feat_strides: List[int] = (8, 16, 32),
        hidden_dim: int = 256,
        use_encoder_idx: List[int] = (2,),
        num_encoder_layers: int = 1,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        enc_act: str = "gelu",
        expansion: float = 1.0,
        depth_mult: float = 1,
        act: str = "silu",
        # LS conv options
        use_ls_conv: bool = False,
        ls_kernel_size: int = 3,
        ls_bn_momentum: float = 0.03,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_levels = len(in_channels)
        self.use_ls_conv = use_ls_conv

        act_fn = {"silu": nn.SiLU(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(act, nn.SiLU())
        # Use a lower BN momentum when LS conv is active to improve stability.
        csp_bn_mom = ls_bn_momentum if use_ls_conv else 0.1

        # --- Input projections (backbone → hidden_dim) ---
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim, momentum=csp_bn_mom),
            )
            for c in in_channels
        ])

        # --- Transformer encoder (AIFI) applied to selected scales ---
        self.encoder = nn.ModuleList([
            nn.Sequential(*[
                AIFI(hidden_dim, nhead, dim_feedforward, dropout, enc_act)
                for _ in range(num_encoder_layers)
            ])
            for _ in use_encoder_idx
        ])

        # --- Top-down PANet ---
        num_blocks = max(round(3 * depth_mult), 1)
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(self.num_levels - 1):
            self.lateral_convs.append(
                ConvBNAct(hidden_dim, hidden_dim, k=1, act=copy.deepcopy(act_fn),
                          bn_momentum=csp_bn_mom)
            )
            self.fpn_blocks.append(
                RepCSPLayer(2 * hidden_dim, hidden_dim,
                            num_blocks=num_blocks, expansion=expansion,
                            use_ls_conv=use_ls_conv, ls_kernel=ls_kernel_size,
                            act=copy.deepcopy(act_fn), bn_momentum=csp_bn_mom)
            )

        # --- Bottom-up PANet ---
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(self.num_levels - 1):
            self.downsample_convs.append(
                ConvBNAct(hidden_dim, hidden_dim, k=3, s=2, act=copy.deepcopy(act_fn),
                          bn_momentum=csp_bn_mom)
            )
            self.pan_blocks.append(
                RepCSPLayer(2 * hidden_dim, hidden_dim,
                            num_blocks=num_blocks, expansion=expansion,
                            use_ls_conv=use_ls_conv, ls_kernel=ls_kernel_size,
                            act=copy.deepcopy(act_fn), bn_momentum=csp_bn_mom)
            )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            feats: list of backbone feature maps ordered from fine to coarse.

        Returns:
            list of encoder output feature maps (same order, fine to coarse).
        """
        proj = [self.input_proj[i](feats[i]) for i in range(self.num_levels)]

        # Apply AIFI to specified scales
        for i, idx in enumerate(self.use_encoder_idx):
            proj[idx] = self.encoder[i](proj[idx])

        # Top-down FPN
        inner = [proj[-1]]
        for i in range(self.num_levels - 2, -1, -1):
            lat = self.lateral_convs[self.num_levels - 2 - i](inner[-1])
            up = F.interpolate(lat, size=proj[i].shape[2:], mode="nearest")
            inner.append(self.fpn_blocks[self.num_levels - 2 - i](
                torch.cat([up, proj[i]], dim=1)
            ))
        inner = inner[::-1]  # fine to coarse

        # Bottom-up PAN
        outs = [inner[0]]
        for i in range(self.num_levels - 1):
            down = self.downsample_convs[i](outs[-1])
            outs.append(self.pan_blocks[i](torch.cat([down, inner[i + 1]], dim=1)))

        return outs
