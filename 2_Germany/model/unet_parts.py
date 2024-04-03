"""
Parts of the U-Net model
Original Source: https://github.com/milesial/Pytorch-UNet
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['BasicConv', 'Down', 'Up', 'OutConv', 'LearnablePosition', 'LCN2DLayer']


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, residual=True):
        super(BasicConv, self).__init__()

        # Main block
        self.basic_conv = nn.Sequential()
        self.basic_conv.add_module('basic_conv1',
                                   nn.Conv2d(in_channels, out_channels,
                                             kernel_size=3, padding=1, bias=False))
        self.basic_conv.add_module('basic_bn',
                                   nn.BatchNorm2d(out_channels, track_running_stats=False))
        self.basic_conv.add_module('basic_relu', nn.LeakyReLU(inplace=True))
        self.basic_conv.add_module('basic_conv2',
                                   nn.Conv2d(out_channels, out_channels,
                                             kernel_size=3, padding=1, bias=False))

        # Residual block
        self.residual = None
        if residual == True:
            self.residual = nn.Sequential()
            self.residual.add_module("res_conv",
                                     nn.Conv2d(in_channels, out_channels,
                                               kernel_size=1, padding=0, bias=False))
            self.residual.add_module("res_bn",
                                     nn.BatchNorm2d(out_channels, track_running_stats=False))

    def forward(self, x):
        out = self.basic_conv(x)

        # Residual connection
        if self.residual is not None:
            x = self.residual(x)
            out = torch.add(x, out)

        return out


class DoubleConv(nn.Module):
    """
    ([BN] => LReLU => convolution) * 2
    In downsampling, first convolution is changed to MaxPool
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, residual=True):
        super(DoubleConv, self).__init__()

        if not mid_channels:
            mid_channels = out_channels

        # Main block
        self.double_conv = nn.Sequential()
        self.double_conv.add_module("conv1",
                                    nn.Conv2d(in_channels, mid_channels,
                                              kernel_size=3, padding=1, bias=False))
        self.double_conv.add_module("bn1",
                                    nn.BatchNorm2d(mid_channels, track_running_stats=False))
        self.double_conv.add_module("relu1",
                                    nn.ReLU(inplace=True))
        self.double_conv.add_module("conv2",
                                    nn.Conv2d(mid_channels, out_channels,
                                              kernel_size=3, padding=1, bias=False))
        self.double_conv.add_module("bn2",
                                    nn.BatchNorm2d(out_channels, track_running_stats=False))
        self.double_conv.add_module("relu2",
                                    nn.ReLU(inplace=True))

        self.residual = None
        if residual == True:
            self.residual = nn.Conv2d(in_channels, out_channels,
                                      kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.double_conv(x)

        # Residual connection
        if self.residual is not None:
            x = self.residual(x)
            out = torch.add(x, out)

        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, residual=True):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.conv = DoubleConv(in_channels, out_channels, residual=residual)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,
                 learnable_pos=None, bilinear=False, attention=False, residual=True, skip=True):
        super(Up, self).__init__()

        # Learnable position setting
        pos_dim = in_channels - (out_channels * 2)
        if pos_dim == 0:
            assert learnable_pos == None

        self.learnable_pos = learnable_pos

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.upsample = nn.Sequential()
        if bilinear:
            self.upsample.add_module('upsample', nn.Upsample(scale_factor=2,
                                                             mode='bilinear',
                                                             align_corners=True))
            self.upsample.add_module('upsample_conv', nn.Conv2d(in_channels - pos_dim, out_channels,
                                                                kernel_size=3, stride=1,
                                                                padding=1, bias=False))
        else:
            self.upsample.add_module('upsample', nn.ConvTranspose2d(in_channels - pos_dim,
                                                                    out_channels,
                                                                    kernel_size=2,
                                                                    stride=2))

        self.attn_gate = None
        if attention:
            self.attn_gate = AttentionGate(out_channels, out_channels, out_channels // 2)

        self.skip = skip
        if not skip:  # Don't use skip connection
            in_channels = out_channels
        self.double_conv = DoubleConv(in_channels, out_channels, residual=residual)

    def forward(self, x, x_res):
        # Upsample
        x = self.upsample(x)

        # input is CHW
        diffY = torch.tensor([x_res.size()[2] - x.size()[2]])
        diffX = torch.tensor([x_res.size()[3] - x.size()[3]])

        x = F.pad(x, [torch.div(diffX, 2, rounding_mode='floor'), diffX - torch.div(diffX, 2, rounding_mode='floor'),
                      torch.div(diffY, 2, rounding_mode='floor'), diffY - torch.div(diffY, 2, rounding_mode='floor')])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        # Attention gate if any
        if self.attn_gate:
            x_res = self.attn_gate(x, x_res)

        # Concat skip connection with previous feature map
        if self.skip:
            out = torch.cat([x_res, x], dim=1)
        else:
            out = x

        # Concat with learnalble position if any
        if self.learnable_pos:
            out = self.learnable_pos(out)

        out = self.double_conv(out)

        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class LearnablePosition(nn.Module):
    def __init__(self, emb_dim, x_size, y_size):
        super(LearnablePosition, self).__init__()
        self.learnable_pos = nn.Parameter(torch.zeros(emb_dim, y_size, x_size), requires_grad=True)

    def forward(self, x):
        x = x + torch.vstack([self.learnable_pos.unsqueeze(0)] * x.shape[0])

        return x


class AttentionGate(nn.Module):
    def __init__(self, feature_x, feature_g, inter):
        super(AttentionGate, self).__init__()

        self.w_g = nn.Sequential(
            nn.Conv2d(feature_g, inter, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter, track_running_stats=False)
        )

        self.w_x = nn.Sequential(
            nn.Conv2d(feature_x, inter, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter, track_running_stats=False)
        )

        self.w_psi = nn.Sequential(
            nn.Conv2d(inter, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1, track_running_stats=False),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.w_psi(psi)

        return x * psi


class LCN2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, x_size, y_size, kernel=3, with_affine=False):
        super(LCN2DLayer, self).__init__()
        self.c = in_channels
        self.oc = out_channels
        self.w = x_size
        self.h = y_size

        weight_shape = [self.c, self.w, self.h, self.oc, kernel, kernel]
        bias_shape = [self.w, self.h, self.oc]
        self.weights = nn.Parameter(torch.zeros(weight_shape, dtype=torch.float32), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(bias_shape, dtype=torch.float32), requires_grad=True)
        self.kernel = kernel
        self.with_affine = with_affine

    def _symm_pad(self, x: torch.Tensor, padding: Tuple[int, int, int, int]):
        h, w = x.shape[-2:]
        left, right, top, bottom = padding

        x_idx = np.arange(-left, w + right)
        y_idx = np.arange(-top, h + bottom)

        def reflect(x, minx, maxx):
            """ Reflects an array around two points making a triangular waveform that ramps up
            and down,  allowing for pad lengths greater than the input length """
            rng = maxx - minx
            double_rng = 2 * rng
            mod = np.fmod(x - minx, double_rng)
            normed_mod = np.where(mod < 0, mod + double_rng, mod)
            out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx

            return np.array(out, dtype=x.dtype)

        x_pad = reflect(x_idx, -0.5, w - 0.5)
        y_pad = reflect(y_idx, -0.5, h - 0.5)
        xx, yy = np.meshgrid(x_pad, y_pad)

        return x[..., yy, xx]

    def forward(self, x):
        pad_size = (self.kernel - 1) // 2
        pad_x = self._symm_pad(x, padding=[pad_size, pad_size, pad_size, pad_size])

        result = [None] * self.oc
        for k in range(self.oc):
            for i in range(self.kernel):
                for j in range(self.kernel):
                    tensor_crop = pad_x[:, :, i:i + self.w, j:j + self.h]
                    weight_filter = self.weights[:, :, :, k, i, j]
                    lcn = torch.sum(weight_filter * tensor_crop, dim=1)

                    if i == 0 and j == 0:
                        result[k] = lcn + self.bias[:, :, k]
                    else:
                        result[k] = result[k] + lcn

        oc_result = torch.cat([torch.reshape(t, [-1, 1, self.w, self.h]) for t in result], dim=1)

        return oc_result
