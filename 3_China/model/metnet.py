import torch
import torch.nn as nn

from .conv_lstm import ConvLSTM
#####################################################################
# for axial attention
# borrowed from https://github.com/lucidrains/axial-attention.git
#####################################################################
from .position_sensitive_attn import AxialBlock

__all__ = ['MetNet','MetNet_Two']


class MetNet(nn.Module):
    def __init__(self, input_data, window_size, num_cls, in_channels, start_dim, pred_hour, center_crop=False,
                 center=None):
        super(MetNet, self).__init__()
        self.input_data = input_data
        self.window_size = window_size
        self.num_cls = num_cls
        self.in_channels = in_channels
        self.start_dim = start_dim
        self.pred_hour = pred_hour
        self.center_crop = center_crop
        self.center = center

        if input_data == 'gdaps_kim':
            self.h = 64
            self.w = 64
        elif input_data == 'gdaps_um':
            self.h = 151
            self.w = 130

        self.spatial_downsampler = nn.Sequential(
            ConvBNAct(self.in_channels, start_dim, padding=1, use_batchnorm=True, use_dropout=False),
            nn.MaxPool2d(2),
            ConvBNAct(start_dim, 2 * start_dim, padding=1, use_batchnorm=True, use_dropout=False),
            # ConvBNAct(2*start_dim, 2*start_dim, padding=1, use_batchnorm=True, use_dropout=False),
            # ConvBNAct(2*start_dim, 2*start_dim, padding=1, use_batchnorm=True, use_dropout=False),
            # nn.MaxPool2d(2),

            # nn.Conv(self.in_channels, 160, 3),
            # nn.MaxPool2d(2),
            # nn.Conv2d(160, 256, 3),
            # nn.Conv2d(256, 256, 3),
            # nn.Conv2d(256, 256, 3),
            # nn.MaxPool2d(2),
        )

        self.temporal_encoder = ConvLSTM(
            input_data=self.input_data,  # Added
            window_size=self.window_size,  # Added
            input_dim=2 * start_dim,
            hidden_dim=[2 * start_dim, 2 * start_dim, 2 * start_dim],
            kernel_size=(3, 3),
            num_layers=3,
            num_classes=self.num_cls,  # Actually, not rquired for MetNet
            batch_first=True,
            bias=True,
            return_all_layers=False,
            for_metnet=True,
        )

        self.spatial_aggregator = nn.Sequential(
            AxialBlock(2 * start_dim, 2 * start_dim, groups=16, width=4 * start_dim,
                       kernel_size=(self.h // 2, self.w // 2)),
            AxialBlock(2 * start_dim, 2 * start_dim, groups=16, width=4 * start_dim,
                       kernel_size=(self.h // 2, self.w // 2)),
            # AxialBlock(384, 384, groups=16, width=2048, kernel_size=(90,72)),
            # AxialBlock(384, 384, groups=16, width=2048, kernel_size=(90,72)),
        )

        '''
        self.spatial_aggregator1 = nn.Sequential(
                axial_PE(dim=384, shape=(90, 72)),
                axial_attn(dim = 384, dim_index = 1, dim_heads = 2048 // 16, heads = 16, num_dimensions=2, sum_axial_out=False))
        self.spatial_aggregator2 = nn.Sequential(
                axial_PE(dim=384, shape=(90, 72)),
                axial_attn(dim = 384, dim_index = 1, dim_heads = 2048 // 16, heads = 16, num_dimensions=2, sum_axial_out=False))
        self.spatial_aggregator3 = nn.Sequential(
                axial_PE(dim=384, shape=(90, 72)),
                axial_attn(dim = 384, dim_index = 1, dim_heads = 2048 // 16, heads = 16, num_dimensions=2, sum_axial_out=False))
        self.spatial_aggregator4 = nn.Sequential(
                axial_PE(dim=384, shape=(90, 72)),
                axial_attn(dim = 384, dim_index = 1, dim_heads = 2048 // 16, heads = 16, num_dimensions=2, sum_axial_out=False))
        
        self.spatial_aggregator = nn.Sequential(
            axial_PE(dim=384, shape=(90, 72)),
            axial_attn(dim = 384, dim_index = 1, dim_heads = 2048 // 16, heads = 16, num_dimensio
            axial_PE(dim=384, shape=(90, 72)),
            axial_attn(dim = 384, dim_index = 1, dim_heads = 2048 // 16, heads = 16, num_dimensio
            axial_PE(dim=384, shape=(90, 72)),
            axial_attn(dim = 384, dim_index = 1, dim_heads = 2048 // 16, heads = 16, num_dimensio
            axial_PE(dim=384, shape=(90, 72)),
            axial_attn(dim = 384, dim_index = 1, dim_heads = 2048 // 16, heads = 16, num_dimensio
            )
        '''
        # self.upsampling = UpSampler(scale=(8, 8), dim_in=384, dim_out=32)

        self.upsampling = nn.Sequential(
            #                 nn.Upsample(size=(self.h//2 + 2,self.w//2 + 2), mode='bilinear', align_corners=True),
            #                 ConvBNAct(2*start_dim, start_dim, use_batchnorm=True, use_dropout=False),

            #                 nn.Upsample(size=(self.h + 2, self.w + 2), mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(2 * start_dim, 2 * start_dim, kernel_size=2, stride=2),  # learning-based upsample
            nn.Upsample(size=(self.h + 2, self.w + 2), mode='bilinear', align_corners=True),
            ConvBNAct(2 * start_dim, start_dim, use_batchnorm=True, use_dropout=False),

            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # ConvBNAct(128, 64, use_batchnorm=True, use_dropout=False),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # ConvBNAct(64, 32, use_batchnorm=True, use_dropout=False),

            # nn.ConvTranspose2d(384, 128, 2, stride=2),
            # nn.ConvTranspose2d(128, 64, 2, stride=2),
            # nn.ConvTranspose2d(64, 32, 2, stride=2),
        )
        self.out_layer = nn.Conv2d(start_dim, self.pred_hour * self.num_cls, 1)

        # weight initialization
        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x, target_time=None, gk2a=None, target_lead_t=60, current_t=None):

        # input data preprocessing
        N = x.shape[1]

        # spatial downsampling for tensors of each time stemp
        feat_t = []
        for _t in range(N):
            feat_t.append(self.spatial_downsampler(x[:, _t, ...]).unsqueeze(1))

        feat_t = torch.cat(feat_t, dim=1)

        # temporal encoder
        feat, _ = self.temporal_encoder(feat_t)
        feat = feat[0][:, -1, ...]  # get last output

        # spatial aggregator        
        '''
        out = feat + self.spatial_aggregator1(feat)
        out = out + self.spatial_aggregator2(out)
        out = out + self.spatial_aggregator3(out)
        out = out + self.spatial_aggregator4(out)
        '''
        out = self.spatial_aggregator(feat)

        # upsampling
        out = self.upsampling(out)
        out = self.out_layer(out)

        return out

class MetNet_Two(nn.Module):
    def __init__(self, input_data, window_size, num_cls, in_channels, start_dim, pred_hour, center_crop=False,
                 center=None):
        super(MetNet_Two, self).__init__()
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

        self.spatial_downsampler = nn.Sequential(
            ConvBNAct(self.in_channels, start_dim, padding=1, use_batchnorm=True, use_dropout=False),
            nn.MaxPool2d(2),
            ConvBNAct(start_dim, 2 * start_dim, padding=1, use_batchnorm=True, use_dropout=False),
            # ConvBNAct(2*start_dim, 2*start_dim, padding=1, use_batchnorm=True, use_dropout=False),
            # ConvBNAct(2*start_dim, 2*start_dim, padding=1, use_batchnorm=True, use_dropout=False),
            # nn.MaxPool2d(2),

            # nn.Conv(self.in_channels, 160, 3),
            # nn.MaxPool2d(2),
            # nn.Conv2d(160, 256, 3),
            # nn.Conv2d(256, 256, 3),
            # nn.Conv2d(256, 256, 3),
            # nn.MaxPool2d(2),
        )

        self.temporal_encoder = ConvLSTM(
            input_data=self.input_data,  # Added
            window_size=self.window_size,  # Added
            input_dim=2 * start_dim,
            hidden_dim=[2 * start_dim, 2 * start_dim, 2 * start_dim],
            kernel_size=(3, 3),
            num_layers=3,
            num_classes=self.num_cls,  # Actually, not rquired for MetNet
            batch_first=True,
            bias=True,
            return_all_layers=False,
            for_metnet=True,
        )

        self.spatial_aggregator = nn.Sequential(
            AxialBlock(2 * start_dim, 2 * start_dim, groups=16, width=4 * start_dim,
                       kernel_size=(self.h // 2, self.w // 2)),
            AxialBlock(2 * start_dim, 2 * start_dim, groups=16, width=4 * start_dim,
                       kernel_size=(self.h // 2, self.w // 2)),
            # AxialBlock(384, 384, groups=16, width=2048, kernel_size=(90,72)),
            # AxialBlock(384, 384, groups=16, width=2048, kernel_size=(90,72)),
        )

        '''
        self.spatial_aggregator1 = nn.Sequential(
                axial_PE(dim=384, shape=(90, 72)),
                axial_attn(dim = 384, dim_index = 1, dim_heads = 2048 // 16, heads = 16, num_dimensions=2, sum_axial_out=False))
        self.spatial_aggregator2 = nn.Sequential(
                axial_PE(dim=384, shape=(90, 72)),
                axial_attn(dim = 384, dim_index = 1, dim_heads = 2048 // 16, heads = 16, num_dimensions=2, sum_axial_out=False))
        self.spatial_aggregator3 = nn.Sequential(
                axial_PE(dim=384, shape=(90, 72)),
                axial_attn(dim = 384, dim_index = 1, dim_heads = 2048 // 16, heads = 16, num_dimensions=2, sum_axial_out=False))
        self.spatial_aggregator4 = nn.Sequential(
                axial_PE(dim=384, shape=(90, 72)),
                axial_attn(dim = 384, dim_index = 1, dim_heads = 2048 // 16, heads = 16, num_dimensions=2, sum_axial_out=False))
        
        self.spatial_aggregator = nn.Sequential(
            axial_PE(dim=384, shape=(90, 72)),
            axial_attn(dim = 384, dim_index = 1, dim_heads = 2048 // 16, heads = 16, num_dimensio
            axial_PE(dim=384, shape=(90, 72)),
            axial_attn(dim = 384, dim_index = 1, dim_heads = 2048 // 16, heads = 16, num_dimensio
            axial_PE(dim=384, shape=(90, 72)),
            axial_attn(dim = 384, dim_index = 1, dim_heads = 2048 // 16, heads = 16, num_dimensio
            axial_PE(dim=384, shape=(90, 72)),
            axial_attn(dim = 384, dim_index = 1, dim_heads = 2048 // 16, heads = 16, num_dimensio
            )
        '''
        # self.upsampling = UpSampler(scale=(8, 8), dim_in=384, dim_out=32)

        self.upsampling = nn.Sequential(
            #                 nn.Upsample(size=(self.h//2 + 2,self.w//2 + 2), mode='bilinear', align_corners=True),
            #                 ConvBNAct(2*start_dim, start_dim, use_batchnorm=True, use_dropout=False),

            #                 nn.Upsample(size=(self.h + 2, self.w + 2), mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(2 * start_dim, 2 * start_dim, kernel_size=2, stride=2),  # learning-based upsample
            nn.Upsample(size=(self.h + 2, self.w + 2), mode='bilinear', align_corners=True),
            ConvBNAct(2 * start_dim, start_dim, use_batchnorm=True, use_dropout=False),

            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # ConvBNAct(128, 64, use_batchnorm=True, use_dropout=False),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # ConvBNAct(64, 32, use_batchnorm=True, use_dropout=False),

            # nn.ConvTranspose2d(384, 128, 2, stride=2),
            # nn.ConvTranspose2d(128, 64, 2, stride=2),
            # nn.ConvTranspose2d(64, 32, 2, stride=2),
        )
        self.out_layer = nn.Conv2d(start_dim, self.pred_hour * self.num_cls, 1)
        self.out_layer2 = nn.Conv2d(start_dim, self.pred_hour * 1, 1)

        # weight initialization
        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x, target_time=None, gk2a=None, target_lead_t=60, current_t=None):

        # input data preprocessing
        N = x.shape[1]

        # spatial downsampling for tensors of each time stemp
        feat_t = []
        for _t in range(N):
            feat_t.append(self.spatial_downsampler(x[:, _t, ...]).unsqueeze(1))

        feat_t = torch.cat(feat_t, dim=1)

        # temporal encoder
        feat, _ = self.temporal_encoder(feat_t)
        feat = feat[0][:, -1, ...]  # get last output

        # spatial aggregator        
        '''
        out = feat + self.spatial_aggregator1(feat)
        out = out + self.spatial_aggregator2(out)
        out = out + self.spatial_aggregator3(out)
        out = out + self.spatial_aggregator4(out)
        '''
        out = self.spatial_aggregator(feat)

        # upsampling
        out = self.upsampling(out)
        out1 = self.out_layer(out)
        out2 = self.out_layer2(out)

        return out1,out2

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, act_fn=nn.ReLU, use_batchnorm=False, use_dropout=False):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, track_running_stats=False) if use_batchnorm else nn.Identity()
        self.act = act_fn()
        self.dropout = nn.Dropout(0.2) if use_dropout else nn.Identity()

    def forward(self, x):
        return self.dropout(self.act(self.bn(self.conv(x))))
