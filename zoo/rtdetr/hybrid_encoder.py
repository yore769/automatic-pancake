"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from .ska import SKA
from .utils import get_activation

from ...core import register


__all__ = ['HybridEncoder']



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size-1)//2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# LS卷积相关模块
class LKP(nn.Module):
    def __init__(self, dim, lks=5, sks=3, groups=8):
        super().__init__()
        assert dim % groups == 0, f"dim={dim} must be divisible by groups={groups}"
        self.groups = groups
        self.sks = sks

        self.cv1 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True)
        )
        # self.act = nn.ReLU()
        self.cv2 = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, lks, padding=(lks - 1) // 2, groups=dim // 2, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True)
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, 1, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True)
        )
        self.cv4 = nn.Conv2d( dim // 2, sks * sks  *  groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=sks ** 2 * groups)



    def forward(self, x):
        x = self.cv3(self.cv2(self.cv1(x)))
        w = self.norm(self.cv4(x))
        B, C, H, W = w.shape
        w = w.view(B, self.groups, self.sks ** 2, H, W)
        return w



class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

        self.gamma = nn.Parameter(1e-4 * torch.ones((dim, 1, 1)))
    def forward(self, x):
        return x + self.gamma * self.bn(self.ska(x, self.lkp(x)))

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.act = get_activation(act) if act is not None else nn.Identity()

        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)   # 3x3
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)   # 1x1
        self.identity = nn.BatchNorm2d(ch_in) if ch_in == ch_out else None

    def forward(self, x):
        if hasattr(self, 'reparam_conv'):
            return self.act(self.reparam_conv(x))

        y = self.conv1(x) + self.conv2(x)
        if self.identity is not None:
            y = y + self.identity(x)
        return self.act(y)

    def convert_to_deploy(self):
        if hasattr(self, 'reparam_conv'):
            return

        kernel, bias = self._fuse_bn_tensor(self.conv1)
        kernel1, bias1 = self._fuse_bn_tensor(self.conv2)
        kernel_id, bias_id = self._fuse_bn_tensor(self.identity)

        if kernel1 is not None:
            kernel = kernel + F.pad(kernel1, [1, 1, 1, 1])
            bias = bias + bias1
        if kernel_id is not None:
            kernel = kernel + F.pad(kernel_id, [1, 1, 1, 1])
            bias = bias + bias_id

        self.reparam_conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, 1, bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        for para in ['conv1', 'conv2', 'identity']:
            if hasattr(self, para):
                delattr(self, para)

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return None, None
        if isinstance(branch, nn.BatchNorm2d):
            # identity 分支只有 BN
            gamma = branch.weight
            std = (branch.running_var + branch.eps).sqrt()
            kernel = torch.diag(gamma.div(std))
            bias = branch.bias - branch.weight.mul(branch.running_mean).div(std)
            return kernel, bias
        else:
            # Conv + BN
            kernel = branch.conv.weight
            bias = branch.conv.bias if branch.conv.bias is not None else torch.zeros_like(branch.norm.bias)
            std = (branch.norm.running_var + branch.norm.eps).sqrt()
            t = (branch.norm.weight / std).reshape(-1, 1, 1, 1)
            return kernel * t, bias - branch.norm.running_mean * branch.norm.weight / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[128 , 256 , 512],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # channel projection
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim)
            )for c in in_channels
        ])

        # LS卷积模块用于处理低级特征
        self.ls_convs = nn.ModuleList([
            LSConv(hidden_dim) if i < len(in_channels) - 1 else nn.Identity()
            for i in range(len(in_channels))
        ])

        # # [消融实验二：所有层全加 LSConv]
        # self.ls_convs = nn.ModuleList([
        #     LSConv(hidden_dim)
        #     for i in range(len(in_channels))
        # ])

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # 应用LS卷积处理低级特征
        for i in range(len(proj_feats)):
            proj_feats[i] = self.ls_convs[i](proj_feats[i])

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                    if pos_embed is None:
                        pos_embed = self.build_2d_sincos_position_embedding(
                            w, h, self.hidden_dim, self.pe_temperature)
                    pos_embed = pos_embed.to(src_flatten.device)

                memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs
