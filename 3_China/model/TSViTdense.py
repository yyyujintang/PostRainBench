import sys
import os
sys.path.insert(0, os.getcwd())
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.TSViT_module import Attention, PreNorm, FeedForward,ChannelAttention
from model.CBAM import CBAMBlock
import numpy as np
# import math

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)    

    
class SViT(nn.Module):
    """
    Spatial-Temporal ViT (used in ablation study, section 4.2)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.depth = model_config['depth']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = self.num_patches_1d ** 2
        patch_dim = model_config['num_channels'] * self.patch_size ** 2                    
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_frames, num_patches, self.dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes * self.patch_size**2))

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
        
        B, T, C, H, W = x.shape
        
        # print("ori", x.shape)
        x = self.to_patch_embedding(x)
        # print("patch embedding", x.shape)
        
        b, t, n, _ = x.shape
        
        x += self.pos_embedding#[:, :, :(n + 1)]
        # print("pos embedding", x.shape)
        x = rearrange(x, 'b t n d -> (b t) n d')
        # print("rearrange", x.shape)
        
        x = self.space_transformer(x) # [B 1024 128]
        # print("space transformer", x.shape)
        x = x.mean(dim=0) # [1024,128] 沿着时间维度求mean
        # print("mean", x.shape)
        
        x = self.mlp_head(x) # [1024 12]
        # print("mlp head", x.shape)
        
        x = x.reshape(B, self.num_patches_1d**2, self.patch_size**2, self.num_classes)
        x = x.reshape(B, H*W, self.num_classes)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        
        reshaped_x = x.reshape(-1, H, W)  # Combine the first three dimensions for resizing
        resized_x = torch.nn.functional.interpolate(reshaped_x.unsqueeze(0), size=(50, 65), mode='bilinear', align_corners=False)
        x = resized_x.view(B, self.num_classes, 50, 65)        
        return x
    
class TSViT(nn.Module):
    """
    Temporal-Spatial ViT5 (used in main results, section 4.3)
    For improved training speed, this implementation uses a (365 x dim) temporal position encodings indexed for
    each day of the year. Use TSViT_lookup for a slower, yet more general implementation of lookup position encodings
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.time_emd_dim = model_config['time_emd_dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  # -1 is set to exclude time feature
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        # self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.to_temporal_embedding_input = nn.Linear(self.time_emd_dim, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2)
        )

    def forward(self, x, target_time=None):
        # x = x.permute(0, 1, 4, 2, 3)    
        # (4, 6, 12, 50, 65)   
        # (1, 6, 12, 50, 65)
        # (4, 6, 64, 64, 12) 
        # x = x.permute(0, 1, 4, 2, 3)
        # B, T, C, H, W = x.shape

        # print(x.shape)
        B, T, C, H, W = x.shape

        reshaped_x = x.reshape(-1, H, W)  # Combine the first three dimensions for resizing
        resized_x = torch.nn.functional.interpolate(reshaped_x.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
        x = resized_x.view(B, T, C, 64, 64)
        B, T, C, H, W = x.shape

        # xt = x[:, :, -1, 0, 0] # [B T]
        xt = x[:, :, 6, H//2-1, W//2-1] # [B T]
        # print("xt ori",xt)
        x = x[:, :, :-1] # [B T C-1 H W]
        # print(x.shape)
        xt = (xt * (self.time_emd_dim-1+.0001)).to(torch.int64)
        # print(xt)
        # print(xt.shape)
        xt = F.one_hot(xt, num_classes=self.time_emd_dim).to(torch.float32)
        # print(xt)
        # print(xt.shape)
        xt = xt.reshape(-1, self.time_emd_dim)
        # print(xt)
        # print(xt.shape)
        
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)

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
        
        # print(x_cls)
        x = F.pad(x, (left, right, top, bottom), 'constant', 0)
        
        return x
    
class SViT_CAM(nn.Module):
    """
    Spatial-Temporal ViT (used in ablation study, section 4.2)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.depth = model_config['depth']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = self.num_patches_1d ** 2
        patch_dim = model_config['num_channels'] * self.patch_size ** 2              
        self.ca = ChannelAttention(in_planes=self.num_frames * model_config['num_channels'], ratio=16)               
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_frames, num_patches, self.dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes * self.patch_size**2))

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
        
        B, T, C, H, W = x.shape
        
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        residual = x
        out = self.ca(x)
        x_cam = x*out + residual       
        x = rearrange(x_cam, 'b (t c) h w -> b t c h w', t=self.num_frames)
        
        # print("recover", x.shape)
        x = self.to_patch_embedding(x)
        # print("patch embedding", x.shape)
        
        b, t, n, _ = x.shape
        
        x += self.pos_embedding#[:, :, :(n + 1)]
        # print("pos embedding", x.shape)
        x = rearrange(x, 'b t n d -> (b t) n d')
        # print("rearrange", x.shape)
        
        x = self.space_transformer(x) # [B 1024 128]
        # print("space transformer", x.shape)
        x = x.mean(dim=0) # [1024,128] 沿着时间维度求mean
        # print("mean", x.shape)
        
        x = self.mlp_head(x) # [1024 12]
        # print("mlp head", x.shape)
        
        x = x.reshape(B, self.num_patches_1d**2, self.patch_size**2, self.num_classes)
        x = x.reshape(B, H*W, self.num_classes)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        
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
        
        # print(x_cls)
        x = F.pad(x, (left, right, top, bottom), 'constant', 0)    
        return x
    
class SViT_Two(nn.Module):
    """
    Spatial-Temporal ViT (used in ablation study, section 4.2)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.depth = model_config['depth']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = self.num_patches_1d ** 2
        patch_dim = model_config['num_channels'] * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_frames, num_patches, self.dim))
        # print('pos embedding: ', self.pos_embedding.shape)
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        # print('space token: ', self.space_token.shape)
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        # self.temporal_token = nn.Parameter(torch.randn(1, 1, self.dim))
        # print('temporal token: ', self.temporal_token.shape)
        # self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head_cls = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes * self.patch_size**2))
        self.mlp_head_reg = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2))

    def forward(self, x, target_time=None):
        B, T, C, H, W = x.shape
        
        # print(x.shape)
        # print(x)
        
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
        
        # print(x)
        
        B, T, C, H, W = x.shape
     
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape
        x += self.pos_embedding#[:, :, :(n + 1)]
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x) # [T 1024 128]
        x_ebd = x.mean(dim=0) # [1024,128] 
        
        x_cls = self.mlp_head_cls(x_ebd) # [1024 12]
        x_cls = x_cls.reshape(B, self.num_patches_1d**2, self.patch_size**2, self.num_classes)
        x_cls = x_cls.reshape(B, H*W, self.num_classes)
        x_cls = x_cls.reshape(B, H, W, self.num_classes)
        x_cls = x_cls.permute(0, 3, 1, 2)
        
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
        
        # print(x_cls)
        x_cls = F.pad(x_cls, (left, right, top, bottom), 'constant', 0)
        # print(x_cls)
              
        x_reg = self.mlp_head_reg(x_ebd) # [1024 12]
        x_reg = x_reg.reshape(B, self.num_patches_1d**2, self.patch_size**2, 1)
        x_reg = x_reg.reshape(B, H*W, 1)
        x_reg = x_reg.reshape(B, H, W, 1)
        x_reg = x_reg.permute(0, 3, 1, 2)
        
        # print(x_reg)
        x_reg = F.pad(x_reg, (left, right, top, bottom), 'constant', 0)
        # print(x_reg)
        
        return x_cls, x_reg


        
class SViT_CAM_Two(nn.Module):
    """
    Spatial-Temporal ViT (used in ablation study, section 4.2)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.depth = model_config['depth']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = self.num_patches_1d ** 2
        patch_dim = 28 * self.patch_size ** 2
        # self.ca = ChannelAttention(in_planes=self.num_frames * model_config['num_channels'], ratio=16)
        self.ca = ChannelAttention(in_planes=28, ratio=16)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, num_patches, self.dim))
        # print('pos embedding: ', self.pos_embedding.shape)
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        # print('space token: ', self.space_token.shape)
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        # self.temporal_token = nn.Parameter(torch.randn(1, 1, self.dim))
        # print('temporal token: ', self.temporal_token.shape)
        # self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head_cls = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes * self.patch_size**2))
        self.mlp_head_reg = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2))

    def forward(self, x, target_time=None):
        # print(x.shape)
        B, T, C, H, W = x.shape

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
        
        # B, T, C, H, W = x.shape
        
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        residual = x
        out = self.ca(x)
        x_cam = x*out + residual       
        x = rearrange(x_cam, 'b (t c) h w -> b t c h w', t=1)
        
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape
        x += self.pos_embedding#[:, :, :(n + 1)]
        # print(x.shape)
        x = rearrange(x, 'b t n d -> (b t) n d')
        # print(x.shape)
        x = self.space_transformer(x) # [T 1024 128]
        # print(x.shape)
        # x_ebd = x.mean(dim=0) # [1024,128] 
        
        x_cls = self.mlp_head_cls(x) # [1024 12]
        # print(x_cls.shape)
        x_cls = x_cls.reshape(B, self.num_patches_1d**2, self.patch_size**2, self.num_classes)
        x_cls = x_cls.reshape(B, H*W, self.num_classes)
        x_cls = x_cls.reshape(B, H, W, self.num_classes)
        x_cls = x_cls.permute(0, 3, 1, 2)
        
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
        
        # # print(x_cls)
        # x_cls = F.pad(x_cls, (left, right, top, bottom), 'constant', 0)
        # print(x_cls)
              
        x_reg = self.mlp_head_reg(x) # [1024 12]
        x_reg = x_reg.reshape(B, self.num_patches_1d**2, self.patch_size**2, 1)
        x_reg = x_reg.reshape(B, H*W, 1)
        x_reg = x_reg.reshape(B, H, W, 1)
        x_reg = x_reg.permute(0, 3, 1, 2)
        
        # print(x_reg)
        # x_reg = F.pad(x_reg, (left, right, top, bottom), 'constant', 0) 
        
        return x_cls, x_reg



    
class SViT_CBAM_Two(nn.Module):
    """
    Spatial-Temporal ViT (used in ablation study, section 4.2)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.depth = model_config['depth']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        self.kernel_size = model_config['kernel_size']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = self.num_patches_1d ** 2
        patch_dim = model_config['num_channels'] * self.patch_size ** 2
        self.ca = CBAMBlock(channel=self.num_frames * model_config['num_channels'],reduction=16,kernel_size=self.kernel_size)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_frames, num_patches, self.dim))
        # print('pos embedding: ', self.pos_embedding.shape)
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        # print('space token: ', self.space_token.shape)
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        # self.temporal_token = nn.Parameter(torch.randn(1, 1, self.dim))
        # print('temporal token: ', self.temporal_token.shape)
        # self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head_cls = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes * self.patch_size**2))
        self.mlp_head_reg = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2))

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
        
        B, T, C, H, W = x.shape
        
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        residual = x
        x = self.ca(x)   
        x = rearrange(x, 'b (t c) h w -> b t c h w', t=self.num_frames)
        
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape
        x += self.pos_embedding#[:, :, :(n + 1)]
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x) # [T 1024 128]
        x_ebd = x.mean(dim=0) # [1024,128] 
        
        x_cls = self.mlp_head_cls(x_ebd) # [1024 12]
        x_cls = x_cls.reshape(B, self.num_patches_1d**2, self.patch_size**2, self.num_classes)
        x_cls = x_cls.reshape(B, H*W, self.num_classes)
        x_cls = x_cls.reshape(B, H, W, self.num_classes)
        x_cls = x_cls.permute(0, 3, 1, 2)
        
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
        
        # print(x_cls)
        x_cls = F.pad(x_cls, (left, right, top, bottom), 'constant', 0)
        # print(x_cls)
              
        x_reg = self.mlp_head_reg(x_ebd) # [1024 12]
        x_reg = x_reg.reshape(B, self.num_patches_1d**2, self.patch_size**2, 1)
        x_reg = x_reg.reshape(B, H*W, 1)
        x_reg = x_reg.reshape(B, H, W, 1)
        x_reg = x_reg.permute(0, 3, 1, 2)
        
        # print(x_reg)
        x_reg = F.pad(x_reg, (left, right, top, bottom), 'constant', 0)
        
        return x_cls, x_reg

if __name__ == "__main__":
    x = torch.rand(1, 6, 12, 50, 65).cuda()
    # targets = torch.rand(1, 50, 65).cuda()
    max_seq_len, num_channels = x.shape[1],x.shape[2]
    num_class = 3
    model_config = {'img_res': 64, 'patch_size': 8, 'patch_size_time': 1, 'patch_time': 4, 'num_classes': num_class,
                    'max_seq_len': max_seq_len, 'time_emd_dim': 6, 'dim': 128, 'temporal_depth': 4, 'spatial_depth': 4,
                    'heads': 4, 'pool': 'cls', 'num_channels': num_channels, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
                    'scale_dim': 4, 'depth': 4}

    model = SViT_CAM_Two(model_config).cuda() # 1.23578M

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.5fM' % parameters)

    # out1,out2 = model(x,x)

    # print("Shape of out :", out1.shape)  # [B, num_classes]
    # print("Shape of out :", out2.shape)  # [B, num_classes]
    
    out1,out2 = model(x,x)
    print("Shape of out :", out1.shape)  # [B, num_classes]
    print("Shape of out :", out2.shape)  # [B, num_classes]
    # print("Shape of out :", out3.shape)  # [B, num_classes]
    # print("Shape of out :", out4.shape)  # [B, num_classes]
    

# if __name__ == "__main__":
#     # res = 24
#     # x = torch.rand((1, 6, 50, 65, 12)).cuda()
#     x = torch.rand(1, 6, 12, 50, 65).cuda()
#     max_seq_len, num_channels = x.shape[1],x.shape[2]
#     num_class = 3
#     model_config = {'img_res': 64, 'patch_size': 8, 'patch_size_time': 1, 'patch_time': 4, 'num_classes': num_class,
#                     'max_seq_len': max_seq_len, 'time_emd_dim': 64, 'dim': 128, 'temporal_depth': 4, 'spatial_depth': 4,
#                     'heads': 4, 'pool': 'cls', 'num_channels': num_channels, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
#                     'scale_dim': 4, 'depth': 4,'kernel_size': 3}
#     # model_config = {'img_res': 64, 'patch_size_list': [2,4,8,16], 'patch_size_time': 1, 'patch_time': 4, 'num_classes': num_class,
#     #                 'max_seq_len': max_seq_len, 'time_emd_dim': 6, 'dim': 128, 'temporal_depth': 4, 'spatial_depth': 4,
#     #                 'heads': 4, 'pool': 'cls', 'num_channels': num_channels, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
#     #                 'scale_dim': 4, 'depth': 4}
#     # B T H W C
#     # x = torch.rand((3, 16, res, res, 14))
#     # x = torch.rand((3, 6, 64, 64, 12))

#     # model = TViT(model_config).cuda()
#     # model = CSViT(model_config).cuda()
#     # model = CSViT_Two(model_config).cuda()
#     # model = CSViT_CAM_Two(model_config).cuda()
#     # model = SViT_CAM(model_config).cuda()
#     # model = Multi_SViT(model_config)
#     # model = STViT(model_config).cuda()
#     # model = Multi_SViT(model_config)
#     # model = TSViT_global_attention_spatial_encoder(model_config)#.cuda()
#     # model = TSViT(model_config).cuda()
#     # model = TSViT(model_config).cuda()
#     # model = SViT_Two(model_config).cuda() # 1.235M
#     model = SViT_CAM_Two(model_config).cuda() # 1.23578M
#     # model = SViT_CBAM_Two(model_config).cuda() # 1.23588M

#     parameters = filter(lambda p: p.requires_grad, model.parameters())
#     parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
#     print('Trainable Parameters: %.5fM' % parameters)

#     # out = model(x,x)
#     out1,out2 = model(x,x)
    
#     # print(model)

#     # print("Shape of out :", out.shape)  # [B, num_classes]
#     print("Shape of out :", out1.shape)  # [B, num_classes]
#     print("Shape of out :", out2.shape)  # [B, num_classes]
    
#     # x1 = torch.rand((3, 3, 12))
#     # x2 = torch.rand((3, 12))
#     # weight = torch.where(x2 > 0.5, 5, torch.where(x2 > 0, 2, torch.ones_like(x2)))
#     # loss = nn.CrossEntropyLoss(weight=torch.tensor([1,2,5],dtype=torch.float))(x1, x2.long()) 