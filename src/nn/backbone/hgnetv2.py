"""HGNetv2 Backbone for RT-DETR v2.

This is a re-implementation of the HGNetv2 backbone used in RT-DETR v2.
Reference: lyuwenyu/RT-DETR (Apache-2.0)
"""

import torch
import torch.nn as nn
from src.core.workspace import register
from .ls_conv import ConvNormAct

__all__ = ['HGNetv2']


# ---------------------------------- building blocks --------------------------

class HGStem(nn.Module):
    """Stem block: two ConvNormAct + max-pool."""

    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.stem1 = ConvNormAct(in_ch, mid_ch, 3, 2, act='relu')
        self.stem2 = ConvNormAct(mid_ch, out_ch, 3, 1, act='relu')
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.stem1(x)
        x = self.stem2(x)
        return self.pool(x)


class LightConvNormAct(nn.Module):
    """Depthwise + pointwise used inside HGBlock."""

    def __init__(self, in_ch, out_ch, kernel_size, act='relu'):
        super().__init__()
        self.dw = ConvNormAct(in_ch, in_ch, kernel_size, groups=in_ch, act='none')
        self.pw = ConvNormAct(in_ch, out_ch, 1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))


class HGBlock(nn.Module):
    """HG (High-Resolution Ghost) block."""

    def __init__(self, in_ch, mid_ch, out_ch, kernel_size=3,
                 num_convs=6, act='relu', use_lab=False):
        super().__init__()
        self.use_lab = use_lab
        convs = []
        for i in range(num_convs):
            ch_in = in_ch if i == 0 else mid_ch
            convs.append(LightConvNormAct(ch_in, mid_ch, kernel_size, act=act))
        self.convs = nn.ModuleList(convs)
        # aggregation: cat all intermediate outputs + input
        agg_in = in_ch + num_convs * mid_ch
        self.agg = ConvNormAct(agg_in, out_ch, 1, act=act)

    def forward(self, x):
        feats = [x]
        for conv in self.convs:
            feats.append(conv(feats[-1]))
        return self.agg(torch.cat(feats, dim=1))


class HGStage(nn.Module):
    """One stage = optional downsample + sequence of HGBlocks."""

    def __init__(self, in_ch, mid_ch, out_ch, num_blocks,
                 downsample=True, kernel_size=3, num_convs=6,
                 act='relu', use_lab=False):
        super().__init__()
        if downsample:
            self.downsample = ConvNormAct(in_ch, in_ch, 3, stride=2,
                                          groups=in_ch, act='none')
        else:
            self.downsample = None

        blocks = []
        for i in range(num_blocks):
            ch_in = in_ch if i == 0 else out_ch
            blocks.append(HGBlock(ch_in, mid_ch, out_ch, kernel_size,
                                   num_convs, act, use_lab))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return self.blocks(x)


# ---------------------------------- arch configs -----------------------------

# name: (stage_configs)
# stage_config: (in_ch, mid_ch, out_ch, num_blocks, downsample)
_ARCH_SETTINGS = {
    # L configuration used in the VisDrone experiments
    'L': {
        'stem': (3, 32, 64),
        'stages': [
            # stage1
            dict(in_ch=64, mid_ch=48, out_ch=256,
                 num_blocks=1, downsample=False),
            # stage2
            dict(in_ch=256, mid_ch=96, out_ch=512,
                 num_blocks=1, downsample=True),
            # stage3
            dict(in_ch=512, mid_ch=192, out_ch=1024,
                 num_blocks=3, downsample=True),
            # stage4
            dict(in_ch=1024, mid_ch=384, out_ch=2048,
                 num_blocks=1, downsample=True),
        ],
    },
    'B': {
        'stem': (3, 24, 32),
        'stages': [
            dict(in_ch=32, mid_ch=32, out_ch=128, num_blocks=1, downsample=False),
            dict(in_ch=128, mid_ch=64, out_ch=256, num_blocks=1, downsample=True),
            dict(in_ch=256, mid_ch=128, out_ch=512, num_blocks=2, downsample=True),
            dict(in_ch=512, mid_ch=256, out_ch=1024, num_blocks=1, downsample=True),
        ],
    },
}


@register
class HGNetv2(nn.Module):
    """HGNetv2 backbone.

    Args:
        name:        architecture name ('L', 'B')
        return_idx:  list of stage indices (0-based) to return as feature maps
        pretrained:  not used (placeholder for API compatibility)
        use_lab:     use Label-Aware Block (not implemented here, placeholder)
    """

    def __init__(self, name='L', return_idx=(1, 2, 3),
                 pretrained=True, use_lab=False):
        super().__init__()
        arch = _ARCH_SETTINGS[name]
        stem_cfg = arch['stem']
        self.stem = HGStem(*stem_cfg)

        self.stages = nn.ModuleList()
        for cfg in arch['stages']:
            self.stages.append(HGStage(**cfg, use_lab=use_lab))

        self.return_idx = list(return_idx)
        self.out_channels = [arch['stages'][i]['out_ch']
                             for i in self.return_idx]

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.return_idx:
                outs.append(x)
        return outs
