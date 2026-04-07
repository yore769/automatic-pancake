"""Large Separable (LS) Convolution for RT-DETR HybridEncoder.

Design motivation
-----------------
The LS convolution replaces a standard k×k conv with a sequence of:
  1. Depth-wise k×k convolution  (captures local structure)
  2. Point-wise 1×1 convolution  (mixes channels)
  3. Optional extra depth-wise 1×k + k×1 conv  (captures long-range context)

When combined with NWD loss and dynamic query grouping three extra care-points
are required:

* **Batch-norm momentum** – three concurrent modifications all change the
  feature distribution seen by BN layers.  The default momentum of 0.1 can
  cause BN running-stats to lag, producing noisy normalisation during early
  training.  We default to momentum=0.03 (the EMA-style value used by YOLOv8)
  for LS-conv BN layers.

* **Initialisation** – the extra depth-wise branch (1×k + k×1) is initialised
  with identity-like weights so that at the start of training the LS conv
  behaves like the standard conv it replaces, allowing the other two
  modifications to converge first.

* **Residual path** – a learnable scalar ``alpha`` gates the long-range branch
  (initialised to 0), giving the network the ability to ignore it if the other
  modifications already capture that information.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    """Conv → BN → Activation."""

    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, g=1,
                 act: nn.Module = None, bn_momentum: float = 0.03,
                 bn_eps: float = 1e-3):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, momentum=bn_momentum, eps=bn_eps)
        self.act = act if act is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# LS Convolution
# ---------------------------------------------------------------------------

class LSConv(nn.Module):
    """Large Separable Convolution block.

    Architecture
    ~~~~~~~~~~~~
    Input
      ├─ DW k×k conv → BN → (held in ``local_branch``)
      └─ DW 1×k conv → DW k×1 conv → BN → gated by ``alpha``
    Concatenated, then PW 1×1 → BN → Act → output

    Parameters
    ----------
    in_channels  : input channel count.
    out_channels : output channel count.
    kernel_size  : local kernel size (e.g. 3 or 5).
    large_k      : size of the long-range 1D kernels (default = 2×kernel_size+1).
    act          : activation applied after the point-wise 1×1 conv.
    bn_momentum  : BatchNorm momentum. Use 0.03 to stabilise combined training.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        large_k: int = None,
        act: nn.Module = None,
        bn_momentum: float = 0.03,
        bn_eps: float = 1e-3,
    ):
        super().__init__()
        if large_k is None:
            large_k = 2 * kernel_size + 1

        # Local branch: standard depth-wise conv
        self.local_branch = ConvBNAct(
            in_channels, in_channels, k=kernel_size, g=in_channels,
            act=nn.Identity(), bn_momentum=bn_momentum, bn_eps=bn_eps
        )

        # Long-range branch: 1×k → k×1 depth-wise (no BN here; shared BN below)
        self.lr_dw_h = nn.Conv2d(
            in_channels, in_channels, (1, large_k),
            padding=(0, large_k // 2), groups=in_channels, bias=False
        )
        self.lr_dw_v = nn.Conv2d(
            in_channels, in_channels, (large_k, 1),
            padding=(large_k // 2, 0), groups=in_channels, bias=False
        )
        self.lr_bn = nn.BatchNorm2d(in_channels, momentum=bn_momentum, eps=bn_eps)

        # Learnable gate – initialised to 0 so long-range branch starts silent.
        # This is critical: it ensures that at epoch 0 the LS-conv behaves like
        # a regular DW conv, letting NWD loss and dynamic-grouping stabilise first.
        self.alpha = nn.Parameter(torch.zeros(1))

        # Point-wise mixing
        self.pw = ConvBNAct(
            in_channels, out_channels, k=1,
            act=act if act is not None else nn.SiLU(),
            bn_momentum=bn_momentum, bn_eps=bn_eps,
        )

        self._init_long_range_weights()

    def _init_long_range_weights(self):
        """Initialise 1D depth-wise kernels close to identity (centre = 1)."""
        for conv in (self.lr_dw_h, self.lr_dw_v):
            nn.init.zeros_(conv.weight)
            # Set the centre element to 1/√2 so the combined response starts at 1
            mid = conv.weight.shape[-1] // 2
            with torch.no_grad():
                if conv.weight.dim() == 4:
                    # shape: (C, 1, H, W)
                    h_mid = conv.weight.shape[-2] // 2
                    w_mid = conv.weight.shape[-1] // 2
                    conv.weight[:, 0, h_mid, w_mid] = 1.0 / (2 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local_branch(x)
        lr = self.lr_bn(self.lr_dw_v(self.lr_dw_h(x)))
        # Gate the long-range contribution; alpha grows during training
        fused = local + self.alpha * lr
        return self.pw(fused)
