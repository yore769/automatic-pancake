"""Lightweight Separable Convolution (LS Conv).

LS Conv decomposes a standard convolution into depthwise + pointwise components
with a skip connection, offering reduced parameters while preserving gradient flow.

Integration Fix
---------------
When combined with NWD Loss and Dynamic Query Grouping, LS Conv is applied
*selectively* only to the intermediate CCFM layers in the encoder, not to the
initial projection layers.  This preserves the feature richness that NWD Loss
needs for accurate Gaussian-distribution modelling of small objects, and that
Dynamic Query Grouping needs for reliable similarity computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LSConv', 'ConvNormAct', 'make_conv']


class ConvNormAct(nn.Module):
    """Standard Conv + BN + Act."""

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1,
                 padding=None, groups=1, dilation=1, act='relu'):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                              padding=padding, groups=groups,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif act == 'gelu':
            self.act = nn.GELU()
        elif act is None or act == 'none':
            self.act = nn.Identity()
        else:
            raise ValueError(f'Unknown activation: {act}')

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LSConv(nn.Module):
    """Lightweight Separable Convolution.

    Replaces a k×k conv with:
        DWConv(k×k, groups=C) + PWConv(1×1) + skip (if in_ch == out_ch)

    The skip connection ensures gradients can flow back even when the
    depthwise branch is under-optimised at the beginning of training.
    This is critical for stability when combined with NWD Loss gradients.

    Args:
        in_ch:      input channels
        out_ch:     output channels
        kernel_size: spatial kernel size for the depthwise conv
        stride:     stride (applied to depthwise conv)
        act:        activation type
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, act='silu'):
        super().__init__()
        # Depthwise spatial mixing
        self.dw = ConvNormAct(in_ch, in_ch, kernel_size, stride,
                              groups=in_ch, act=act)
        # Pointwise channel projection
        self.pw = ConvNormAct(in_ch, out_ch, 1, 1, act=act)
        # Identity skip when shapes match
        self.use_skip = (in_ch == out_ch and stride == 1)
        if not self.use_skip and in_ch != out_ch:
            # Projection skip for shape mismatch
            self.skip_proj = ConvNormAct(in_ch, out_ch, 1, stride, act='none')
            self.use_proj_skip = True
        else:
            self.use_proj_skip = False

    def forward(self, x):
        out = self.pw(self.dw(x))
        if self.use_skip:
            out = out + x
        elif self.use_proj_skip:
            out = out + self.skip_proj(x)
        return out


def make_conv(in_ch, out_ch, kernel_size=3, stride=1, act='silu',
              use_ls_conv=False):
    """Factory: returns LSConv or standard ConvNormAct.

    The ``use_ls_conv`` flag lets the caller decide whether to replace the
    standard conv.  In the combined-improvement setting, only intermediate
    fusion layers use LSConv to avoid degrading the initial feature maps.
    """
    if use_ls_conv and kernel_size >= 3:
        return LSConv(in_ch, out_ch, kernel_size, stride, act=act)
    return ConvNormAct(in_ch, out_ch, kernel_size, stride, act=act)
