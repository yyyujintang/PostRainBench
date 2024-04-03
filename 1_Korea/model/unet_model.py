"""
Full assembly of the parts to form the complete network
Original Source: https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
import numpy as np

# from model.unet_parts import *
from .unet_parts import *

__all__ = ['UNet']


class UNet(nn.Module):
    def __init__(self, input_data, window_size, embedding_dim, n_channels, n_classes, n_blocks, start_channels,
                 end_lead_time, residual, no_skip, batch_size=1, use_lcn=False, use_tte=False):
        super(UNet, self).__init__()

        # Model entrance block
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.input_channels = n_channels // window_size  # number of input variables
        self.embedding = nn.Conv2d(in_channels=self.input_channels,
                                   out_channels=embedding_dim,
                                   kernel_size=1)
        self.end_lead_time = end_lead_time
        self.use_tte = use_tte

        # Learnable position related
        if input_data == 'ldaps':
            x_size = 512
            y_size = 512
        elif input_data == 'gdaps_kim':
            x_size = 65
            y_size = 50
        elif input_data == 'gdaps_um':
            x_size = 130
            y_size = 151
        else:
            raise ValueError('Invalid `input_data` argument: {}'.format(input_data))

        if not use_tte:
            self.ste = nn.Sequential()
            for _ in range(self.end_lead_time):
                self.ste.add_module('pos{}'.format(_), LearnablePosition(embedding_dim, x_size, y_size))

        self.inc = nn.Sequential()
        self.inc.add_module('inc', BasicConv(embedding_dim * window_size + (1 if use_tte else 0), start_channels,
                                             residual=residual))

        # Create down blocks
        self.down = nn.ModuleList([])
        for i in range(n_blocks):
            cur_in_ch = start_channels << i
            self.down.append(Down(cur_in_ch, cur_in_ch * 2, residual=residual))

        # Create bridge block
        self.bridge = nn.Sequential()
        bridge_channels = start_channels << n_blocks
        self.bridge.add_module('bridge_conv', BasicConv(bridge_channels, bridge_channels, residual=residual))

        # Create up blocks
        self.up = nn.ModuleList([])
        skip = [(i not in no_skip) for i in range(n_blocks + 1)]  # Enable/disable skip connections
        for i in range(n_blocks, 0, -1):
            cur_in_ch = start_channels << i
            self.up.append(Up(cur_in_ch, cur_in_ch // 2, residual=residual, skip=skip[i]))

        # Create out convolution block
        self.outc = nn.Sequential()
        if use_lcn:
            self.outc.add_module('out_lcn',
                                 LCN2DLayer(in_channels=start_channels, out_channels=n_classes, x_size=x_size,
                                            y_size=y_size))
        else:
            self.outc.add_module('out_conv', OutConv(start_channels, n_classes))

    def forward(self, x, target_time=None):
        with torch.no_grad():
            x = torch.cat([x[:, i, :, :, :] for i in range(x.shape[1])], dim=1)
        embedding_lst = []
        for i in range(self.window_size):
            embedding_lst.append(self.embedding(x[:, i * self.input_channels: (i + 1) * self.input_channels, :, :]))
        x = torch.cat(embedding_lst, axis=1)

        # x shape : [batch_size, embedding_dim * window_size, y_size (세로), x_size (가로)]
        target_time = None
        if target_time is not None:
            target_h = target_time[0][-1]
            if self.use_tte:  # Use TTE
                B, C, H, W = x.shape
                last_channel = torch.tensor([[[[((target_h + 8) % 24 + 1) / 24] * W] * H]] * B).cuda().float()
                x = torch.cat([x, last_channel], dim=1)
            else:  # Use LPE as default
                for i, idx in enumerate(range(target_h - self.window_size + 1, target_h + 1)):
                    x[:, i * self.embedding_dim: (i + 1) * self.embedding_dim, :, :] += \
                        self.ste[idx](x[:, i * self.embedding_dim: (i + 1) * self.embedding_dim, :, :])

        out = self.inc(x)

        # Long residual list for Up phase
        long_residual = []
        long_residual.append(out.clone())

        # Down blocks
        for down_block in self.down:
            out = down_block(out)
            long_residual.append(out.clone())

        # Bridge block
        out = self.bridge(out)

        # Up blocks
        for i, up_block in enumerate(self.up):
            out = up_block(out, long_residual[-(i + 2)])

        logit = self.outc(out)

        return logit
    
class UNet_Two(nn.Module):
    def __init__(self, input_data, window_size, embedding_dim, n_channels, n_classes, n_blocks, start_channels,
                 end_lead_time, residual, no_skip, batch_size=1, use_lcn=False, use_tte=False):
        super(UNet_Two, self).__init__()

        # Model entrance block
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.input_channels = n_channels // window_size  # number of input variables
        self.embedding = nn.Conv2d(in_channels=self.input_channels,
                                   out_channels=embedding_dim,
                                   kernel_size=1)
        self.end_lead_time = end_lead_time
        self.use_tte = use_tte

        # Learnable position related
        if input_data == 'ldaps':
            x_size = 512
            y_size = 512
        elif input_data == 'gdaps_kim':
            x_size = 64
            y_size = 64
        elif input_data == 'gdaps_um':
            x_size = 130
            y_size = 151
        else:
            raise ValueError('Invalid `input_data` argument: {}'.format(input_data))

        if not use_tte:
            self.ste = nn.Sequential()
            for _ in range(self.end_lead_time):
                self.ste.add_module('pos{}'.format(_), LearnablePosition(embedding_dim, x_size, y_size))

        self.inc = nn.Sequential()
        self.inc.add_module('inc', BasicConv(embedding_dim * window_size + (1 if use_tte else 0), start_channels,
                                             residual=residual))

        # Create down blocks
        self.down = nn.ModuleList([])
        for i in range(n_blocks):
            cur_in_ch = start_channels << i
            self.down.append(Down(cur_in_ch, cur_in_ch * 2, residual=residual))

        # Create bridge block
        self.bridge = nn.Sequential()
        bridge_channels = start_channels << n_blocks
        self.bridge.add_module('bridge_conv', BasicConv(bridge_channels, bridge_channels, residual=residual))

        # Create up blocks
        self.up = nn.ModuleList([])
        skip = [(i not in no_skip) for i in range(n_blocks + 1)]  # Enable/disable skip connections
        for i in range(n_blocks, 0, -1):
            cur_in_ch = start_channels << i
            self.up.append(Up(cur_in_ch, cur_in_ch // 2, residual=residual, skip=skip[i]))

        # Create out convolution block
        self.outc = nn.Sequential()
        self.outc2 = nn.Sequential()
        if use_lcn:
            self.outc.add_module('out_lcn',
                                 LCN2DLayer(in_channels=start_channels, out_channels=n_classes, x_size=x_size,
                                            y_size=y_size))
        else:
            self.outc.add_module('out_conv', OutConv(start_channels, n_classes))
        self.outc2.add_module('out_conv', OutConv(start_channels, 1))
            

    def forward(self, x, target_time=None):
        with torch.no_grad():
            x = torch.cat([x[:, i, :, :, :] for i in range(x.shape[1])], dim=1)
        embedding_lst = []
        for i in range(self.window_size):
            embedding_lst.append(self.embedding(x[:, i * self.input_channels: (i + 1) * self.input_channels, :, :]))
        x = torch.cat(embedding_lst, axis=1)

        # x shape : [batch_size, embedding_dim * window_size, y_size (세로), x_size (가로)]

        if target_time is not None:
            target_h = target_time[0][-1]
            if self.use_tte:  # Use TTE
                B, C, H, W = x.shape
                last_channel = torch.tensor([[[[((target_h + 8) % 24 + 1) / 24] * W] * H]] * B).cuda().float()
                x = torch.cat([x, last_channel], dim=1)
            else:  # Use LPE as default
                for i, idx in enumerate(range(target_h - self.window_size + 1, target_h + 1)):
                    x[:, i * self.embedding_dim: (i + 1) * self.embedding_dim, :, :] += \
                        self.ste[idx](x[:, i * self.embedding_dim: (i + 1) * self.embedding_dim, :, :])

        out = self.inc(x)

        # Long residual list for Up phase
        long_residual = []
        long_residual.append(out.clone())

        # Down blocks
        for down_block in self.down:
            out = down_block(out)
            long_residual.append(out.clone())

        # Bridge block
        out = self.bridge(out)

        # Up blocks
        for i, up_block in enumerate(self.up):
            out = up_block(out, long_residual[-(i + 2)])

        logit = self.outc(out)
        logit2 = self.outc2(out)

        return logit,logit2

# input_data = torch.randn(4, 3, 12, 50, 65).cuda()
# model = UNet(input_data='gdaps_kim',
#             window_size=3,
#             embedding_dim=8,
#             n_channels=input_data.shape[0] * input_data.shape[1],
#             n_classes=3,
#             n_blocks=3,
#             start_channels=16,
#             batch_size=1,
#             end_lead_time=88,
#             residual=True,
#             no_skip=[],
#             use_tte=False).cuda()

# output = model(input_data)
# print(output.shape)
# parameters = filter(lambda p: p.requires_grad, model.parameters())
# parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
# print('Trainable Parameters: %.3fM' % parameters)
# 输入的是4个channel，输出的是4张图
