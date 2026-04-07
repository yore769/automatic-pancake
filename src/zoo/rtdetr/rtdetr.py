"""RTDETR main model."""

import torch
import torch.nn as nn

from src.core import register


@register()
class RTDETR(nn.Module):
    """RT-DETR: a real-time DEtection TRansformer.

    Assembles backbone, encoder, and decoder into a full detection model.
    """

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor, targets=None):
        feats = self.backbone(x)
        enc_feats = self.encoder(feats)
        outputs = self.decoder(enc_feats, targets)
        return outputs
