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

# class ArcFace(Module):

#     def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.5):
#         super(ArcFace, self).__init__()
#         self.classnum = classnum
#         self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
#         # initial kernel
#         self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
#         self.m = m # the margin value, default is 0.5
#         self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369

#         self.eps = 1e-4

#     def forward(self, embbedings, norms, label):

#         kernel_norm = l2_norm(self.kernel,axis=0)
#         cosine = torch.mm(embbedings,kernel_norm)
#         cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

#         m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
#         m_hot.scatter_(1, label.reshape(-1, 1), self.m)

#         theta = cosine.acos()

#         theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi-self.eps)
#         cosine_m = theta_m.cos()
#         scaled_cosine_m = cosine_m * self.s

#         return scaled_cosine_m
    
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
        patch_dim = model_config['num_channels'] * self.patch_size ** 2
        self.ca = ChannelAttention(in_planes=self.num_frames * model_config['num_channels'], ratio=16)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_frames, num_patches, self.dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.output_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.dim * self.num_patches_1d**2 , 512),
                # nn.BatchNorm1d(512, affine=False)
                )
        
        # self.mlp_head_cls = nn.Sequential(
        #     nn.LayerNorm(self.dim),
        #     nn.Linear(self.dim, self.num_classes * self.patch_size**2))
        self.mlp_head_reg = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2))

    def forward(self, x, target_time=None):
        B, T, C, H, W = x.shape

        reshaped_x = x.reshape(-1, H, W)  # Combine the first three dimensions for resizing
        resized_x = torch.nn.functional.interpolate(reshaped_x.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
        x = resized_x.view(B, T, C, 64, 64)
        B, T, C, H, W = x.shape
        
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        residual = x
        out = self.ca(x)
        x_cam = x*out + residual       
        x = rearrange(x_cam, 'b (t c) h w -> b t c h w', t=self.num_frames)
        
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape
        x += self.pos_embedding#[:, :, :(n + 1)]
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x) # [T 1024 128]
        
        # print("befor embedding shape",x.shape)
        # 需要的embedding是[b,embedding]
        x_ebd = x.mean(dim=0) # [64,512-dim] [64,64]       
        print("embedding shape",x_ebd.shape)
        
        x_head = rearrange(x_ebd, '(b n) d -> b (n d)',b=B)
        # print("x_head shape",x_head.shape)
        x_head = self.output_layer(x_head)
        # print("embedding head shape",x_head.shape)
        
        norms = torch.norm(x_head, 2, 1, True)
        embeddings = torch.div(x_head, norms)
               
        # embeddings, norms = self.model(images)
        # 这里的label其实是没有给的，是送到criterion里面去的，因此这里也可以作为返回值
        # cos_thetas = self.head(embeddings, norms, labels)
        
        
        # x_cls = self.mlp_head_cls(x_ebd) # [1024 12]
        # x_cls = x_cls.reshape(B, self.num_patches_1d**2, self.patch_size**2, self.num_classes)
        # x_cls = x_cls.reshape(B, H*W, self.num_classes)
        # x_cls = x_cls.reshape(B, H, W, self.num_classes)
        # x_cls = x_cls.permute(0, 3, 1, 2)
        
        # reshaped_x_cls = x_cls.reshape(-1, H, W)  # Combine the first three dimensions for resizing
        # reshaped_x_cls = torch.nn.functional.interpolate(reshaped_x_cls.unsqueeze(0), size=(50, 65), mode='bilinear', align_corners=False)
        # x_cls = reshaped_x_cls.view(B, self.num_classes, 50, 65)        
        
        x_reg = self.mlp_head_reg(x_ebd) # [1024 12]
        x_reg = x_reg.reshape(B, self.num_patches_1d**2, self.patch_size**2, 1)
        x_reg = x_reg.reshape(B, H*W, 1)
        x_reg = x_reg.reshape(B, H, W, 1)
        x_reg = x_reg.permute(0, 3, 1, 2)
        
        reshaped_x_reg = x_reg.reshape(-1, H, W)  # Combine the first three dimensions for resizing
        reshaped_x_reg = torch.nn.functional.interpolate(reshaped_x_reg.unsqueeze(0), size=(50, 65), mode='bilinear', align_corners=False)
        x_reg = reshaped_x_reg.view(B, 1, 50, 65)   
        
        return x_reg, embeddings, norms


# if __name__ == "__main__":
#     x = torch.rand(1, 6, 12, 50, 65).cuda()
#     max_seq_len, num_channels = x.shape[1],x.shape[2]
#     num_class = 3
#     model_config = {'img_res': 64, 'patch_size': 8, 'patch_size_time': 1, 'patch_time': 4, 'num_classes': num_class,
#                     'max_seq_len': max_seq_len, 'time_emd_dim': 6, 'dim': 128, 'temporal_depth': 4, 'spatial_depth': 4,
#                     'heads': 4, 'pool': 'cls', 'num_channels': num_channels, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
#                     'scale_dim': 4, 'depth': 4}

#     model = SViT_CAM_Two(model_config).cuda() # 1.23578M

#     parameters = filter(lambda p: p.requires_grad, model.parameters())
#     parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
#     print('Trainable Parameters: %.5fM' % parameters)

#     # out1,out2 = model(x,x)

#     # print("Shape of out :", out1.shape)  # [B, num_classes]
#     # print("Shape of out :", out2.shape)  # [B, num_classes]
    
#     out1,out2,out3 = model(x,x)
#     print("Shape of out :", out1.shape)  # [B, num_classes]
#     print("Shape of out :", out2.shape)  # [B, num_classes]
#     print("Shape of out :", out3.shape)  # [B, num_classes]
#     # print("Shape of out :", out4.shape)  # [B, num_classes]
    
