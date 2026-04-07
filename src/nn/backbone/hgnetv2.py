"""HGNetv2 backbone for RT-DETR v2."""

import torch
import torch.nn as nn

from src.core import register


class LearnableAffineBlock(nn.Module):
    """Learnable affine transform used in HGNetv2 stem."""
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale_value))
        self.bias = nn.Parameter(torch.tensor(bias_value))

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HGBlock(nn.Module):
    """High-Level Gather-Distribute block."""
    def __init__(self, in_ch, mid_ch, out_ch, k=3, n=6, use_lab=False):
        super().__init__()
        self.use_lab = use_lab
        self.conv1 = ConvBNAct(in_ch, mid_ch, 1)
        self.mid_convs = nn.ModuleList([
            ConvBNAct(mid_ch, mid_ch, k) for _ in range(n)
        ])
        agg_in = in_ch + (n + 1) * mid_ch
        self.conv2 = ConvBNAct(agg_in, out_ch, 1)
        if use_lab:
            self.lab = LearnableAffineBlock()

    def forward(self, x):
        y = [x, self.conv1(x)]
        for m in self.mid_convs:
            y.append(m(y[-1]))
        out = self.conv2(torch.cat(y, dim=1))
        if self.use_lab:
            out = self.lab(out)
        return out


class HGStage(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, n_blocks=1, stride=2,
                 k=3, n_mid=6, use_lab=False):
        super().__init__()
        layers = [ConvBNAct(in_ch, out_ch, k=3, s=stride)]
        for _ in range(n_blocks):
            layers.append(HGBlock(out_ch, mid_ch, out_ch, k=k, n=n_mid,
                                  use_lab=use_lab))
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)


# Channel and block configurations for different HGNetv2 variants
_HGNETV2_CFGS = {
    "B0": dict(stem_ch=16, stages=[
        (16, 16, 64, 1), (64, 32, 256, 1), (256, 64, 512, 2), (512, 128, 1024, 1)
    ]),
    "B1": dict(stem_ch=24, stages=[
        (24, 32, 96, 1), (96, 48, 384, 1), (384, 96, 768, 2), (768, 192, 1536, 1)
    ]),
    "B2": dict(stem_ch=32, stages=[
        (32, 64, 128, 1), (128, 64, 512, 1), (512, 128, 1024, 2), (1024, 256, 2048, 1)
    ]),
    "B3": dict(stem_ch=48, stages=[
        (48, 96, 192, 1), (192, 96, 768, 1), (768, 192, 1536, 2), (1536, 384, 3072, 1)
    ]),
    "B4": dict(stem_ch=64, stages=[
        (64, 128, 256, 1), (256, 128, 1024, 1), (1024, 256, 2048, 2), (2048, 512, 4096, 1)
    ]),
    "B5": dict(stem_ch=96, stages=[
        (96, 192, 384, 1), (384, 192, 1536, 1), (1536, 384, 3072, 2), (3072, 768, 6144, 1)
    ]),
    # HGNetv2-L as used in RT-DETR-L: produces [512, 1024, 2048] at return_idx [1, 2, 3],
    # which matches the HybridEncoder's in_channels: [512, 1024, 2048] in the config.
    "L": dict(stem_ch=32, stages=[
        (32, 64, 256, 1), (256, 128, 512, 1), (512, 256, 1024, 2), (1024, 512, 2048, 1)
    ]),
}

# Output channels for each return stage (stages 1-3 in 0-indexed → indices 1,2,3)
_HGNETV2_OUT_CHANNELS = {
    "B0": [64, 256, 512],
    "B1": [96, 384, 768],
    "B2": [128, 512, 1024],
    "B3": [192, 768, 1536],
    "B4": [256, 1024, 2048],
    "B5": [384, 1536, 3072],
    "L":  [512, 1024, 2048],  # stages [1,2,3] outputs; matches HybridEncoder in_channels
}


@register()
class HGNetv2(nn.Module):
    """HGNetv2 backbone.

    Parameters
    ----------
    name       : variant name ('B0' – 'B5', 'L').
    return_idx : list of stage indices (0-based) whose outputs to return.
    pretrained : load ImageNet-pretrained weights (requires network access).
    use_lab    : use Learnable Affine Block in HGBlocks.
    """

    def __init__(self, name: str = "L", return_idx=(1, 2, 3),
                 pretrained: bool = False, use_lab: bool = False):
        super().__init__()
        cfg = _HGNETV2_CFGS[name]
        stem_ch = cfg["stem_ch"]
        stages_cfg = cfg["stages"]

        # Stem: 3 → stem_ch
        self.stem = nn.Sequential(
            ConvBNAct(3, stem_ch // 2, k=3, s=2),
            ConvBNAct(stem_ch // 2, stem_ch, k=3, s=1),
            ConvBNAct(stem_ch, stem_ch, k=3, s=1),
        )

        # Build stages
        self.stages = nn.ModuleList()
        in_ch = stem_ch
        for (s_mid, _, s_out, n_blk) in stages_cfg:
            self.stages.append(
                HGStage(in_ch, s_mid, s_out, n_blocks=n_blk, stride=2, use_lab=use_lab)
            )
            in_ch = s_out

        self.return_idx = list(return_idx)
        self.out_channels = [stages_cfg[i][2] for i in return_idx]

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.return_idx:
                outs.append(x)
        return outs
