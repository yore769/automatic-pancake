import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math


class SKAFunction(Function):
    @staticmethod
    def forward(ctx, x, w):
        """
        Selective Kernel Attention (SKA) - 高效向量化实现
        x: (B, C, H, W)
        w: (B, G, K*K, H, W)   # G = C // groups, 通常 groups=8 → G = C//8
        """
        B, C, H, W = x.shape
        G = w.shape[1]           # 权重通道数（分组数）
        K2 = w.shape[2]
        K = int(math.isqrt(K2))
        assert K * K == K2 and (K % 2) == 1
        pad = (K - 1) // 2

        # 1. unfold x → (B, C, K2, L)
        x_padded = F.pad(x, (pad, pad, pad, pad))
        x_unfolded = F.unfold(x_padded, kernel_size=K, stride=1)
        x_unfolded = x_unfolded.view(B, C, K2, H * W)

        # 2. w → (B, G, K2, L)
        w_flat = w.view(B, G, K2, H * W)

        # 3. 扩展 w 到 C 通道：每个 group 负责 C//G 个输入通道
        assert C % G == 0
        channel_per_group = C // G
        w_expanded = w_flat.repeat(1, channel_per_group, 1, 1)   # (B, C, K2, L)

        # 4. 加权求和
        out_flat = (x_unfolded * w_expanded).sum(dim=2)   # (B, C, L)
        out = out_flat.view(B, C, H, W)

        ctx.save_for_backward(x, w, x_unfolded, w_expanded)
        ctx.G, ctx.K2, ctx.L, ctx.channel_per_group = G, K2, H * W, channel_per_group
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w, x_unfolded, w_expanded = ctx.saved_tensors
        G, K2, L, channel_per_group = ctx.G, ctx.K2, ctx.L, ctx.channel_per_group
        B, C, H, W = x.shape

        go = grad_output.contiguous()
        go_flat = go.view(B, C, L)

        gx = gw = None

        # grad w.r.t x
        if ctx.needs_input_grad[0]:
            gx_flat = (go_flat.unsqueeze(2) * w_expanded).sum(dim=2)
            gx = gx_flat.view(B, C, H, W)

        # grad w.r.t w
        if ctx.needs_input_grad[1]:
            grad_w = x_unfolded * go_flat.unsqueeze(2)           # (B, C, K2, L)
            grad_w = grad_w.view(B, G, channel_per_group, K2, L)                # (B, G, C//G, K2, L)
            gw_flat = grad_w.sum(dim=2)                          # (B, G, K2, L)
            gw = gw_flat.view(B, G, K2, H, W)

        return gx, gw


class SKA(nn.Module):
    def forward(self, x, w):
        return SKAFunction.apply(x, w)