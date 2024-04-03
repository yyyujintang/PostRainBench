import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys,SwinTransformerSys_Two
import torch.nn.functional as F
from model.TSViT_module import ChannelAttention
from model.CBAM import CBAMBlock



class SwinUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.swin_unet = SwinTransformerSys(img_size=64,
                                patch_size=4,
                                in_chans=143,
                                num_classes=3,
                                embed_dim=96,
                                depths=[ 2, 2, 2, 2 ],
                                num_heads=[ 3, 6, 12, 24 ],
                                window_size=8,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

    def forward(self, x, target_time=None):
        # x = torch.flatten(x, start_dim=1, end_dim=2)
        # # print(x.shape)
        # target_size = (64, 64) # 224-7
        # x = torch.nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)       
        
        # print(x.shape)
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        logits = self.swin_unet(x)
        # target_size = (50, 65)
        # logits = torch.nn.functional.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)  
        return logits
    
class SwinUnet_CAM(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet_CAM, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.ca = ChannelAttention(in_planes=143, ratio=16)
        self.swin_unet = SwinTransformerSys(img_size=64,
                                patch_size=4,
                                in_chans=143,
                                num_classes=3,
                                embed_dim=96,
                                depths=[ 2, 2, 2, 2 ],
                                num_heads=[ 3, 6, 12, 24 ],
                                window_size=8,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

    def forward(self, x, target_time=None):
        B, T, C, H, W = x.shape
        # x = torch.flatten(x, start_dim=1, end_dim=2)
        # # print(x.shape)
        # target_size = (64, 64) # 224-7
        # x = torch.nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)       
        
        # print(x.shape)
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        residual = x
        out = self.ca(x)
        x_cam = x*out + residual       
        x = rearrange(x_cam, 'b (t c) h w -> b t c h w', t=T)       
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        logits = self.swin_unet(x)
        # target_size = (50, 65)
        # logits = torch.nn.functional.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)  
        return logits    
    
# class SwinUnet_Two(nn.Module):
#     def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
#         super(SwinUnet_Two, self).__init__()
#         self.num_classes = num_classes
#         self.zero_head = zero_head
#         self.swin_unet = SwinTransformerSys_Two(img_size=64,
#                                 patch_size=4,
#                                 in_chans=12,
#                                 num_classes=3,
#                                 embed_dim=96,
#                                 depths=[ 2, 2, 2, 2 ],
#                                 num_heads=[ 3, 6, 12, 24 ],
#                                 window_size=8,
#                                 mlp_ratio=4.,
#                                 qkv_bias=True,
#                                 qk_scale=None,
#                                 drop_rate=0.0,
#                                 drop_path_rate=0.0,
#                                 ape=False,
#                                 patch_norm=True,
#                                 use_checkpoint=False)

#     def forward(self, x, target_time=None):
#         B, T, C, H, W = x.shape
#         original_height, original_width = H, W
#         target_height, target_width  = 64, 64
           
#         # For Padding             
#         pad_height = target_height - original_height
#         pad_width = target_width - original_width
#         top = pad_height // 2
#         bottom = pad_height - top
#         left = pad_width // 2
#         right = pad_width - left
        
#         x = F.pad(x, (left, right, top, bottom), 'constant', 0)        
#         x = rearrange(x, 'b t c h w -> (b t) c h w')

             
#         if x.size()[1] == 1:
#             x = x.repeat(1,3,1,1)
            
#         logits1,logits2 = self.swin_unet(x)
        
#         # For Recover Index
#         original_height, original_width = 64, 64
#         target_height, target_width  = 50, 65
                   
#         # For Padding             
#         pad_height = target_height - original_height
#         pad_width = target_width - original_width
#         top = pad_height // 2
#         bottom = pad_height - top
#         left = pad_width // 2
#         right = pad_width - left
        
#         logits1 = F.pad(logits1, (left, right, top, bottom), 'constant', 0)  
#         logits2 = F.pad(logits2, (left, right, top, bottom), 'constant', 0)
#         logits1 = rearrange(logits1, '(b t) c h w -> b t c h w', c = 3, t = 6).mean(dim=1)
#         logits2 = rearrange(logits2, '(b t) c h w -> b t c h w', c = 1, t = 6).mean(dim=1)       
#         return logits1,logits2

   
# class SwinUnet_CAM_Two(nn.Module):
#     def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
#         super(SwinUnet_CAM_Two, self).__init__()
#         self.num_classes = num_classes
#         self.zero_head = zero_head
#         self.ca = ChannelAttention(in_planes=1*143, ratio=16)         
#         self.swin_unet = SwinTransformerSys_Two(img_size=64,
#                                 patch_size=4,
#                                 in_chans=143,
#                                 num_classes=3,
#                                 embed_dim=96,
#                                 depths=[ 2, 2, 2, 2 ],
#                                 num_heads=[ 3, 6, 12, 24 ],
#                                 window_size=8,
#                                 mlp_ratio=4.,
#                                 qkv_bias=True,
#                                 qk_scale=None,
#                                 drop_rate=0.0,
#                                 drop_path_rate=0.0,
#                                 ape=False,
#                                 patch_norm=True,
#                                 use_checkpoint=False)

#     def forward(self, x, target_time=None):
#         B, T, C, H, W = x.shape
#         # print(x.shape)
#         # original_height, original_width = H, W
#         # target_height, target_width  = 64, 64
           
#         # For Padding             
#         # pad_height = target_height - original_height
#         # pad_width = target_width - original_width
#         # top = pad_height // 2
#         # bottom = pad_height - top
#         # left = pad_width // 2
#         # right = pad_width - left
        
#         # x = F.pad(x, (left, right, top, bottom), 'constant', 0) 
        
#         x = rearrange(x, 'b t c h w -> b (t c) h w')
#         residual = x
#         out = self.ca(x)
#         x_cam = x*out + residual       
#         x = rearrange(x_cam, 'b (t c) h w -> b t c h w', t=T)
        
#         x = rearrange(x, 'b t c h w -> (b t) c h w')
             
#         if x.size()[1] == 1:
#             x = x.repeat(1,3,1,1)
            
#         logits1,logits2 = self.swin_unet(x)
        
#         # # For Recover Index
#         # original_height, original_width = 64, 64
#         # target_height, target_width  = 50, 65
                   
#         # # For Padding             
#         # pad_height = target_height - original_height
#         # pad_width = target_width - original_width
#         # top = pad_height // 2
#         # bottom = pad_height - top
#         # left = pad_width // 2
#         # right = pad_width - left
        
#         # logits1 = F.pad(logits1, (left, right, top, bottom), 'constant', 0)  
#         # logits2 = F.pad(logits2, (left, right, top, bottom), 'constant', 0)
#         logits1 = rearrange(logits1, '(b t) c h w -> b t c h w', c = 3, t = T).mean(dim=1)
#         logits2 = rearrange(logits2, '(b t) c h w -> b t c h w', c = 1, t = T).mean(dim=1)       
#         return logits1,logits2

class SwinUnet_Two(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet_Two, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.swin_unet = SwinTransformerSys_Two(img_size=64,
                                patch_size=4,
                                in_chans=12,
                                num_classes=3,
                                embed_dim=96,
                                depths=[ 2, 2, 2, 2 ],
                                num_heads=[ 3, 6, 12, 24 ],
                                window_size=8,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

    def forward(self, x, target_time=None):
        B, T, C, H, W = x.shape
        original_height, original_width = H, W
        target_height, target_width  = 64, 64
           
        # For Padding             
        pad_height = target_height - original_height
        pad_width = target_width - original_width
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        
        x = F.pad(x, (left, right, top, bottom), 'constant', 0)        
        x = rearrange(x, 'b t c h w -> (b t) c h w')

             
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
            
        logits1,logits2 = self.swin_unet(x)
        
        # For Recover Index
        original_height, original_width = 64, 64
        target_height, target_width  = 50, 65
                   
        # For Padding             
        pad_height = target_height - original_height
        pad_width = target_width - original_width
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        
        logits1 = F.pad(logits1, (left, right, top, bottom), 'constant', 0)  
        logits2 = F.pad(logits2, (left, right, top, bottom), 'constant', 0)
        logits1 = rearrange(logits1, '(b t) c h w -> b t c h w', c = 3, t = T).mean(dim=1)
        logits2 = rearrange(logits2, '(b t) c h w -> b t c h w', c = 1, t = T).mean(dim=1)       
        return logits1,logits2

   
class SwinUnet_CAM_Two(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet_CAM_Two, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.ca = ChannelAttention(in_planes=143, ratio=16)         
        self.swin_unet = SwinTransformerSys_Two(img_size=64,
                                patch_size=4,
                                in_chans=143,
                                num_classes=3,
                                embed_dim=96,
                                depths=[ 2, 2, 2, 2 ],
                                num_heads=[ 3, 6, 12, 24 ],
                                window_size=8,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

    def forward(self, x, target_time=None):
        B, T, C, H, W = x.shape
        # print(x.shape)
        # original_height, original_width = H, W
        # target_height, target_width  = 64, 64
           
        # # For Padding             
        # pad_height = target_height - original_height
        # pad_width = target_width - original_width
        # top = pad_height // 2
        # bottom = pad_height - top
        # left = pad_width // 2
        # right = pad_width - left
        
        # x = F.pad(x, (left, right, top, bottom), 'constant', 0) 
        
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        residual = x
        out = self.ca(x)
        x_cam = x*out + residual       
        x = rearrange(x_cam, 'b (t c) h w -> b t c h w', t=T)
        
        x = rearrange(x, 'b t c h w -> (b t) c h w')
             
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
            
        logits1,logits2 = self.swin_unet(x)
        
        # For Recover Index
        # original_height, original_width = 64, 64
        # target_height, target_width  = 50, 65
                   
        # # For Padding             
        # pad_height = target_height - original_height
        # pad_width = target_width - original_width
        # top = pad_height // 2
        # bottom = pad_height - top
        # left = pad_width // 2
        # right = pad_width - left
        
        # logits1 = F.pad(logits1, (left, right, top, bottom), 'constant', 0)  
        # logits2 = F.pad(logits2, (left, right, top, bottom), 'constant', 0)
        logits1 = rearrange(logits1, '(b t) c h w -> b t c h w', c = 3, t = T).mean(dim=1)
        logits2 = rearrange(logits2, '(b t) c h w -> b t c h w', c = 1, t = T).mean(dim=1)       
        return logits1,logits2
    
    

      
# model = SwinUnet().cuda()
# # print(model)
# input = torch.randn(4, 6, 12, 50, 65).cuda()
# output = model(input)
# print(output.shape)
# 输入的是4个channel，输出的是4张图

# model = SwinUnet_CAM_Two().cuda()
# parameters = filter(lambda p: p.requires_grad, model.parameters())
# parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
# print('Trainable Parameters: %.5fM' % parameters)
# input = torch.randn(4, 1, 143, 64, 64).cuda()
# output1,output2 = model(input)
# print(output1.shape)
# print(output2.shape)
