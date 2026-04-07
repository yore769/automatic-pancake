"""RT-DETRv2 main model."""

import torch
import torch.nn as nn

from src.core.config import register


@register
class RTDETR(nn.Module):
    """
    RT-DETRv2 object detection model.

    Combines:
    - Backbone (HGNetv2)
    - Encoder (HybridEncoder with LSConv)
    - Decoder (RTDETRTransformerv2 with Dynamic Query Grouping)

    Args:
        backbone: feature extraction backbone
        encoder: cross-scale feature fusion encoder
        decoder: transformer decoder
        num_classes: number of object classes (optional, overridden by criterion)
    """

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int = 80,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, targets: list = None) -> dict:
        """
        Args:
            x: [B, 3, H, W] input images
            targets: list of target dicts (used during training for denoising)

        Returns:
            dict with 'pred_logits', 'pred_boxes', and optionally
            'aux_outputs', 'dn_outputs', 'dn_meta'
        """
        features = self.backbone(x)
        encoder_feats = self.encoder(features)
        return self.decoder(encoder_feats, targets=targets)

    def deploy(self):
        """Switch to deployment mode (no denoising, export-friendly)."""
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
