import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.TSViT_module import Attention, PreNorm, FeedForward
from mmcv.cnn import ConvModule


    
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
    
class Multi_SViT(nn.Module):
    """
    Spatial-Temporal ViT (used in ablation study, section 4.2)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size_list = model_config['patch_size_list']
        self.num_patches_1d_list = [self.image_size // patch_size for patch_size in self.patch_size_list]
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
        # assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches_list = [n ** 2 for n in self.num_patches_1d_list]
        self.patch_dim_list = [model_config['num_channels'] * self.patch_size ** 2 for self.patch_size in self.patch_size_list]
        self.to_patch_embedding = nn.ModuleList([
            nn.Sequential(
                Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size_list[index], p2=self.patch_size_list[index]),
                nn.Linear(self.patch_dim_list[index], self.dim))
            for index in range(len(self.patch_size_list))  # 不同尺度的 patch 大小
        ])
        self.pos_embedding_list = [nn.Parameter(torch.randn(1, self.num_frames, num_patches, self.dim)) for num_patches in num_patches_list]
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)     
        self.mlp_head = [nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes * self.patch_size**2)) for self.patch_size in self.patch_size_list]

    def forward(self, x, target_time=None):
        B, T, C, H, W = x.shape

        reshaped_x = x.reshape(-1, H, W)  # Combine the first three dimensions for resizing
        resized_x = torch.nn.functional.interpolate(reshaped_x.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
        x = resized_x.view(B, T, C, 64, 64)
        B, T, C, H, W = x.shape
        
        x_scale = []
        for index,patch_embedding in enumerate(self.to_patch_embedding):
            patch_x = patch_embedding(x)
            b, t, n, _ = patch_x.shape
            # patch_x += self.pos_embedding_list[index]  
            patch_x = rearrange(patch_x, 'b t n d -> (b t) n d')
            patch_x = self.space_transformer(patch_x)
            patch_x = patch_x.mean(dim=0)
            x_recov = self.mlp_head[index](patch_x)
            x_recov = x_recov.reshape(B, self.num_patches_1d_list[index]**2, self.patch_size_list[index]**2, self.num_classes)
            x_recov = x_recov.reshape(B, H*W, self.num_classes)
            x_recov = x_recov.reshape(B, H, W, self.num_classes)
            x_recov = x_recov.permute(0, 3, 1, 2)
            x_scale.append(x_recov)
                    

        # 将不同尺度的 patch embeddings 进行拼接或融合
        x = torch.cat(x_scale, dim=0)
        x = torch.mean(x, dim=0)
        # print(x.shape)
           
        reshaped_x = x.reshape(-1, H, W)  # Combine the first three dimensions for resizing
        resized_x = torch.nn.functional.interpolate(reshaped_x.unsqueeze(0), size=(50, 65), mode='bilinear', align_corners=False)
        x = resized_x.view(B, self.num_classes, 50, 65)      
                       
        return x
    
    
# class SViT(nn.Module):
#     """
#     Spatial-Temporal ViT (used in ablation study, section 4.2)
#     """
#     def __init__(self, model_config):
#         super().__init__()
#         self.image_size = model_config['img_res']
#         self.patch_size = model_config['patch_size']
#         self.num_patches_1d = self.image_size//self.patch_size
#         self.num_classes = model_config['num_classes']
#         self.num_frames = model_config['max_seq_len']
#         self.dim = model_config['dim']
#         self.depth = model_config['depth']
#         if 'temporal_depth' in model_config:
#             self.temporal_depth = model_config['temporal_depth']
#         else:
#             self.temporal_depth = model_config['depth']
#         if 'spatial_depth' in model_config:
#             self.spatial_depth = model_config['spatial_depth']
#         else:
#             self.spatial_depth = model_config['depth']
#         self.heads = model_config['heads']
#         self.dim_head = model_config['dim_head']
#         self.dropout = model_config['dropout']
#         self.emb_dropout = model_config['emb_dropout']
#         self.pool = model_config['pool']
#         self.scale_dim = model_config['scale_dim']
#         assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#         assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
#         num_patches = self.num_patches_1d ** 2
#         patch_dim = model_config['num_channels'] * self.patch_size ** 2
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
#             nn.Linear(patch_dim, self.dim))
#         self.pos_embedding = nn.Parameter(torch.randn(1, self.num_frames, num_patches, self.dim))
#         self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
#         self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
#         self.dropout = nn.Dropout(self.emb_dropout)     
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(self.dim),
#             nn.Linear(self.dim, self.num_classes * self.patch_size**2))
        
#     def forward(self, x, target_time=None):
#         B, T, C, H, W = x.shape

#         reshaped_x = x.reshape(-1, H, W)  # Combine the first three dimensions for resizing
#         resized_x = torch.nn.functional.interpolate(reshaped_x.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
#         x = resized_x.view(B, T, C, 64, 64)
#         B, T, C, H, W = x.shape
        
#         # 从patch dim到self.dim(128) 应该改进这里，成为逐channel的
#         x = self.to_patch_embedding(x)
        
#         b, t, n, _ = x.shape
        
#         x += self.pos_embedding#[:, :, :(n + 1)]
#         x = rearrange(x, 'b t n d -> (b t) n d')
        
#         x = self.space_transformer(x) # [B 1024 128]
#         x = x.mean(dim=0) # [1024,128] 沿着时间维度求mean
        
#         x = self.mlp_head(x) # [1024 12]
        
#         x = x.reshape(B, self.num_patches_1d**2, self.patch_size**2, self.num_classes)
#         x = x.reshape(B, H*W, self.num_classes)
#         x = x.reshape(B, H, W, self.num_classes)
#         x = x.permute(0, 3, 1, 2)
        
        
#         reshaped_x = x.reshape(-1, H, W)  # Combine the first three dimensions for resizing
#         resized_x = torch.nn.functional.interpolate(reshaped_x.unsqueeze(0), size=(50, 65), mode='bilinear', align_corners=False)
#         x = resized_x.view(B, self.num_classes, 50, 65)      
        
               
#         return x
    
# # Example usage

# num_class = 3
# num_channels = 1
# max_seq_len = 6
# model_config = {'img_res': 64, 'patch_size': 2, 'patch_size_time': 1, 'patch_time': 4, 'num_classes': num_class,
#                 'max_seq_len': max_seq_len, 'time_emd_dim': 6, 'dim': 128, 'temporal_depth': 4, 'spatial_depth': 4,
#                 'heads': 4, 'pool': 'cls', 'num_channels': num_channels, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
#                 'scale_dim': 4, 'depth': 4}

# model = MSViT(12, model_config).cuda()
# parameters = filter(lambda p: p.requires_grad, model.parameters())
# parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
# print('Trainable Parameters: %.3fM' % parameters)
# input_data = torch.rand(1, 6, 12, 50, 65).cuda()
# output = model(input_data)
# print(output.shape)  # Shape: (16, num_classes, num_channels, height, width)


if __name__ == "__main__":
    # res = 24
    # x = torch.rand((1, 6, 50, 65, 12)).cuda()
    x = torch.rand(1, 6, 12, 50, 65).cuda()
    max_seq_len, num_channels = x.shape[1],x.shape[2]
    num_class = 3
    # model_config = {'img_res': 64, 'patch_size': 2, 'patch_size_time': 1, 'patch_time': 4, 'num_classes': num_class,
    #                 'max_seq_len': max_seq_len, 'time_emd_dim': 6, 'dim': 128, 'temporal_depth': 4, 'spatial_depth': 4,
    #                 'heads': 4, 'pool': 'cls', 'num_channels': num_channels, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
    #                 'scale_dim': 4, 'depth': 4}
    model_config = {'img_res': 64, 'patch_size_list': [2,4], 'patch_size_time': 1, 'patch_time': 4, 'num_classes': num_class,
                    'max_seq_len': max_seq_len, 'time_emd_dim': 6, 'dim': 128, 'temporal_depth': 4, 'spatial_depth': 4,
                    'heads': 4, 'pool': 'cls', 'num_channels': num_channels, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
                    'scale_dim': 4, 'depth': 4}

    # B T H W C
    # x = torch.rand((3, 16, res, res, 14))
    # x = torch.rand((3, 6, 64, 64, 12))

    # model = TViT(model_config).cuda()
    # model = STViT(model_config).cuda()
    model = Multi_SViT(model_config)
    # model = TSViT_global_attention_spatial_encoder(model_config)#.cuda()
    # model = TSViT(model_config).cuda()
    # model = TSViT(model_config).cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out = model(x,x)

    print("Shape of out :", out.shape)  # [B, num_classes]
