"""Common neural network utility functions and modules."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(act: str, inplace: bool = True):
    """Return an activation module by name."""
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1, inplace=inplace)
    elif act == 'silu' or act == 'swish':
        return nn.SiLU(inplace=inplace)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'hardswish':
        return nn.Hardswish(inplace=inplace)
    elif act == 'identity' or act == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation: {act}")


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse of sigmoid function."""
    x = x.clamp(min=0., max=1.)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def bias_init_with_prob(prior_prob: float) -> float:
    """Return bias value for classification head given prior probability."""
    return -math.log((1 - prior_prob) / prior_prob)


def deformable_attention_core_func(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Core function for deformable attention.

    Args:
        value: [B, Lv, nHead, headDim]
        value_spatial_shapes: [numLevels, 2]  (H, W) per level
        sampling_locations: [B, Lq, nHead, numLevels, numPoints, 2]
        attention_weights: [B, Lq, nHead, numLevels, numPoints]

    Returns:
        output: [B, Lq, nHead * headDim]
    """
    B, _, nHead, headDim = value.shape
    _, Lq, _, numLevels, numPoints, _ = sampling_locations.shape

    # Split value by level
    value_list = value.split(
        [H * W for H, W in value_spatial_shapes], dim=1
    )

    sampling_grids = 2 * sampling_locations - 1  # normalize to [-1, 1]
    output = []
    for level, (H, W) in enumerate(value_spatial_shapes):
        # [B, H*W, nHead, headDim] -> [B*nHead, headDim, H, W]
        v = value_list[level].permute(0, 2, 3, 1)  # [B, nHead, headDim, H*W]
        v = v.reshape(B * nHead, headDim, H, W)

        # [B, Lq, nHead, numPoints, 2] -> [B*nHead, Lq, numPoints, 2]
        grid = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4)
        grid = grid.reshape(B * nHead, Lq, numPoints, 2)

        # Sample: [B*nHead, headDim, Lq, numPoints]
        sampled = F.grid_sample(
            v, grid, mode='bilinear', padding_mode='zeros', align_corners=False
        )

        # [B*nHead, headDim, Lq, numPoints] -> [B, nHead, Lq, numPoints, headDim]
        sampled = sampled.view(B, nHead, headDim, Lq, numPoints)
        sampled = sampled.permute(0, 3, 1, 4, 2)  # [B, Lq, nHead, numPoints, headDim]

        output.append(sampled)

    # Stack levels: [B, Lq, nHead, numLevels, numPoints, headDim]
    output = torch.stack(output, dim=3)

    # Attention weights: [B, Lq, nHead, numLevels, numPoints] -> [..., 1]
    attn = attention_weights.unsqueeze(-1)

    # Weighted sum: [B, Lq, nHead, headDim]
    output = (output * attn).sum(dim=[3, 4])

    # [B, Lq, nHead * headDim]
    return output.reshape(B, Lq, nHead * headDim)


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
