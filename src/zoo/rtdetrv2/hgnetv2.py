"""HGNetv2 backbone (PP-HGNetv2) for RT-DETRv2."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.config import register


__all__ = ['HGNetv2']


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=1, s=1, p=0, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LightConvBNAct(nn.Module):
    """Depthwise + pointwise convolution (lightweight conv)."""

    def __init__(self, in_c, out_c, k, act=True):
        super().__init__()
        self.dw = ConvBNAct(in_c, in_c, k=k, s=1, p=k // 2, g=in_c, act=False)
        self.pw = ConvBNAct(in_c, out_c, k=1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))


class HGBlock(nn.Module):
    """HGNet block: concat multiple LightConv outputs."""

    def __init__(self, in_c, mid_c, out_c, k=3, n=6, shortcut=False, act=True):
        super().__init__()
        self.m = nn.ModuleList(
            LightConvBNAct(in_c if i == 0 else mid_c, mid_c, k=k, act=act)
            for i in range(n)
        )
        total = in_c + mid_c * n
        self.sc = ConvBNAct(total, out_c // 2, k=1, act=act)
        self.ec = ConvBNAct(out_c // 2, out_c, k=1, act=act)
        self.shortcut = shortcut and in_c == out_c

    def forward(self, x):
        out = [x]
        y = x
        for m in self.m:
            y = m(y)
            out.append(y)
        y = self.sc(torch.cat(out, dim=1))
        y = self.ec(y)
        return x + y if self.shortcut else y


class HGStage(nn.Module):
    """One stage of HGNetv2."""

    def __init__(self, in_c, mid_c, out_c, n_blocks: int, k=3, stride=2, shortcut=True):
        super().__init__()
        self.downsample = ConvBNAct(in_c, in_c, k=3, s=stride, p=1, g=in_c, act=False)
        self.blocks = nn.Sequential(
            *[HGBlock(in_c if i == 0 else out_c, mid_c, out_c, k=k,
                      shortcut=shortcut and i > 0)
              for i in range(n_blocks)]
        )

    def forward(self, x):
        return self.blocks(self.downsample(x))


# ---------------------------------------------------------------------------
# Architecture configs: name -> (stages cfg)
# stages cfg: list of (in_c, mid_c, out_c, n_blocks, stride)
# ---------------------------------------------------------------------------

HGNetv2_CONFIGS = {
    'B0': dict(
        stem_channels=16,
        stages=[
            (16,  16,  64,  1, 2),
            (64,  32,  256, 1, 2),
            (256, 64,  512, 2, 2),
            (512, 128, 1024, 1, 2),
        ],
        out_channels=[64, 256, 512, 1024],
    ),
    'B1': dict(
        stem_channels=32,
        stages=[
            (32,  32,  64,  1, 2),
            (64,  48,  256, 1, 2),
            (256, 96,  512, 2, 2),
            (512, 192, 1024, 1, 2),
        ],
        out_channels=[64, 256, 512, 1024],
    ),
    'B2': dict(
        stem_channels=32,
        stages=[
            (32,  32,  96,  1, 2),
            (96,  64,  384, 1, 2),
            (384, 128, 768, 3, 2),
            (768, 256, 1536, 1, 2),
        ],
        out_channels=[96, 384, 768, 1536],
    ),
    'B3': dict(
        stem_channels=48,
        stages=[
            (48,  48,  128, 1, 2),
            (128, 96,  512, 1, 2),
            (512, 192, 1024, 3, 2),
            (1024, 384, 2048, 1, 2),
        ],
        out_channels=[128, 512, 1024, 2048],
    ),
    'B4': dict(
        stem_channels=64,
        stages=[
            (64,  64,  128, 1, 2),
            (128, 128, 512, 1, 2),
            (512, 256, 1024, 3, 2),
            (1024, 512, 2048, 1, 2),
        ],
        out_channels=[128, 512, 1024, 2048],
    ),
    'B5': dict(
        stem_channels=64,
        stages=[
            (64,  64,  128, 1, 2),
            (128, 128, 512, 1, 2),
            (512, 256, 1024, 4, 2),
            (1024, 512, 2048, 2, 2),
        ],
        out_channels=[128, 512, 1024, 2048],
    ),
    'B6': dict(
        stem_channels=64,
        stages=[
            (64,  64,  128, 1, 2),
            (128, 128, 512, 2, 2),
            (512, 256, 1024, 5, 2),
            (1024, 512, 2048, 3, 2),
        ],
        out_channels=[128, 512, 1024, 2048],
    ),
    'L': dict(
        stem_channels=96,
        stages=[
            (96,  96,   256, 1, 2),
            (256, 256,  512, 1, 2),
            (512, 512,  1024, 2, 2),
            (1024, 512, 2048, 1, 2),
        ],
        out_channels=[256, 512, 1024, 2048],
    ),
    'X': dict(
        stem_channels=128,
        stages=[
            (128, 128,  512, 1, 2),
            (512, 512,  1024, 2, 2),
            (1024, 512, 2048, 3, 2),
            (2048, 1024, 4096, 1, 2),
        ],
        out_channels=[512, 1024, 2048, 4096],
    ),
}


@register
class HGNetv2(nn.Module):
    """
    PP-HGNetv2 backbone.

    Args:
        name: model variant (e.g., 'L')
        return_idx: which stage outputs to return (0-indexed)
        pretrained: whether to load pretrained weights
        use_lab: use LAB activation (not implemented, kept for config compat)
    """

    def __init__(
        self,
        name: str = 'L',
        return_idx=None,
        pretrained: bool = False,
        use_lab: bool = False,
    ):
        super().__init__()
        if return_idx is None:
            return_idx = [1, 2, 3]
        self.return_idx = return_idx

        cfg = HGNetv2_CONFIGS[name]
        stem_c = cfg['stem_channels']
        stages_cfg = cfg['stages']
        self.out_channels = [cfg['out_channels'][i] for i in return_idx]

        # Stem: 3 -> stem_c at stride 4
        self.stem = nn.Sequential(
            ConvBNAct(3, stem_c // 2, k=3, s=2, p=1),
            ConvBNAct(stem_c // 2, stem_c, k=3, s=2, p=1),
        )

        self.stages = nn.ModuleList()
        in_c = stem_c
        for (stage_in, mid_c, out_c, n_blocks, stride) in stages_cfg:
            # Allow first stage to start from stem output
            self.stages.append(HGStage(in_c, mid_c, out_c, n_blocks, stride=stride))
            in_c = out_c

        if pretrained:
            self._load_pretrained(name)

    def _load_pretrained(self, name: str):
        """Attempt to load pretrained weights. Fails silently."""
        try:
            import torch.hub as hub
            url_map = {
                'L': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/PPHGNetV2_L_ssld_pretrained_from_paddle.pth',
                'X': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/PPHGNetV2_X_ssld_pretrained_from_paddle.pth',
            }
            if name in url_map:
                state = hub.load_state_dict_from_url(url_map[name], map_location='cpu')
                # Load matching keys only
                model_state = self.state_dict()
                filtered = {}
                for k, v in state.items():
                    # Remove potential 'model.' prefix
                    k2 = k[len('model.'):] if k.startswith('model.') else k
                    if k2 in model_state and model_state[k2].shape == v.shape:
                        filtered[k2] = v
                model_state.update(filtered)
                self.load_state_dict(model_state, strict=False)
                print(f"Loaded {len(filtered)}/{len(state)} pretrained params for HGNetv2-{name}")
        except Exception as e:
            print(f"Pretrained load failed for HGNetv2-{name}: {e}")

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return [outs[i] for i in self.return_idx]
