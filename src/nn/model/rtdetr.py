"""RT-DETR model wrapper."""

import torch
import torch.nn as nn

from src.core.workspace import register

__all__ = ['RTDETR']


@register
class RTDETR(nn.Module):
    """RT-DETR: Real-Time DEtection TRansformer.

    Combines backbone, encoder, and decoder into a single model.
    """

    def __init__(self, backbone, encoder, decoder):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, targets=None):
        feats = self.backbone(x)
        feats = self.encoder(feats)
        return self.decoder(feats, targets)

    def deploy(self):
        """Switch model to deploy mode."""
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
