"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math
import torch
from .conv_lstm import ConvLSTM

__all__ = ['precipitation_point']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_1x1_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 1, stride=1, padding=0, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 1, stride=1, padding=0, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PointwiseMetNet(nn.Module):
    def __init__(self, input_data, window_size, num_cls, in_channels, start_dim, pred_hour, center_crop=False,
                 center=None):
        super(PointwiseMetNet, self).__init__()
        
        self.input_data = input_data
        self.window_size = window_size
        self.num_cls = num_cls
        self.in_channels = in_channels
        self.start_dim = start_dim
        self.pred_hour = pred_hour
        self.center_crop = center_crop
        self.center = center
        
        if input_data == 'gdaps_kim':
            self.h = 50
            self.w = 65
        elif input_data == 'gdaps_um':
            self.h = 151
            self.w = 130
        
        width_mult =1.
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [6,  32, 3, 1],
            [6,  64, 3, 1],
            # [6,  32, 2, 1],
            # [6,  64, 2, 1],
            # [6,  96, 2, 1],
            # [6, 160, 2, 1],
            # [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_1x1_bn(self.in_channels, input_channel)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(128 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 128
        self.conv = conv_1x1_bn(input_channel, output_channel)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Linear(output_channel, num_classes)
        # self.out_layer = nn.Conv2d(output_channel, self.pred_hour * self.num_cls, 1)


        self._initialize_weights()
        
        hidden_dim_list = [output_channel, output_channel, output_channel]
        self.temporal_encoder = ConvLSTM(
            input_data=self.input_data,  # Added
            window_size=self.window_size,  # Added
            input_dim=output_channel,
            hidden_dim=hidden_dim_list,
            kernel_size=(3, 3),
            num_layers=3,
            num_classes=self.num_cls * self.pred_hour,  # Actually, not rquired for MetNet
            batch_first=True,
            bias=True,
            return_all_layers=False,
            for_metnet=False,
        )

    def forward(self, x, target_time=None, gk2a=None, target_lead_t=60, current_t=None):
        
         # input data preprocessing
        N = x.shape[1]

        # spatial downsampling for tensors of each time stemp
        feat_t = []
        for _t in range(N):
            feat_t.append(self.conv(self.features(x[:, _t, ...])).unsqueeze(1))
        
        # x = self.features(x)
        # x = self.conv(x)
        # print(feat_t[0].shape)
        # print(feat_t.shape)
        feat_t = torch.cat(feat_t, dim=1)

        out = self.temporal_encoder(feat_t)
        # feat = feat[0][:, -1, ...]  # get last output        
        # print(feat.shape)
        # out = self.out_layer(feat)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def precipitation_point(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return PointwiseMetNet(**kwargs)