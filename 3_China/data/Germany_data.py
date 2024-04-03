# from data.dataset import CustomTensorDataset
from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
from data.transforms import ClassifyByThresholds
from einops import rearrange


save_path = "/mnt/ssd1/yujint/GermanyData/"
# trn_x = np.load(save_path + '01_trn_x.npy')
trn_y = np.load(save_path + '01_trn_y.npy')
# tst_x = np.load(save_path + '01_tst_x.npy')
tst_y = np.load(save_path + '01_tst_y.npy')
# vld_x = np.load(save_path + '01_vld_x.npy')
vld_y = np.load(save_path + '01_vld_y.npy')
# print('Train: x =', trn_x.shape, '---> y =', trn_y.shape)
# print('Test: x =', tst_x.shape, '---> y =', tst_y.shape)
# print('Valid: x =', vld_x.shape, '---> y =', vld_y.shape)


# class CustomTensorDataset(Dataset):
#     def __init__(self, input_tensors,gt_tensors,size=(32,32)):
#         self.input_tensors = input_tensors
#         self.gt_tensors = gt_tensors
#         self.size = size

#     def __getitem__(self, index):
#         x = self.input_tensors[index]
#         y = self.gt_tensors[index]
#         y_resized = torch.nn.functional.interpolate(y.unsqueeze(0).unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)
#         y_resized = y_resized.squeeze()
#         y_one_hot = ClassifyByThresholds([0.00001,2])(y_resized) 
#         x = rearrange(x.unsqueeze(0), 't h w c-> t c h w')
#         y_one_hot = y_one_hot.unsqueeze(0)
#         return x,y_one_hot
#     def __len__(self):
#         return self.input_tensors.shape[0]
# class CustomTensorDataset(Dataset):
#     def __init__(self, input_tensors,gt_tensors,size=(32,32)):
#         self.input_tensors = input_tensors
#         self.gt_tensors = gt_tensors
#         self.size = size

#     def __getitem__(self, index):
#         x = self.input_tensors[index]
#         y = self.gt_tensors[index]
#         x = torch.nn.functional.interpolate(x.permute(2,0,1).unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)
#         x = x.squeeze().permute(1,2,0)
#         y_resized = torch.nn.functional.interpolate(y.unsqueeze(0).unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)
#         y_resized = y_resized.squeeze()
#         y_one_hot = ClassifyByThresholds([0.00001,2])(y_resized) 
#         x = rearrange(x.unsqueeze(0), 't h w c-> t c h w')
#         y_one_hot = y_one_hot.unsqueeze(0)
#         return x,y_one_hot,y_resized.unsqueeze(0)
#     def __len__(self):
#         return self.input_tensors.shape[0]

# batch_size = 64
# train_dataset = CustomTensorDataset(torch.from_numpy(trn_x),torch.from_numpy(trn_y),size=(36,36))
# val_dataset = CustomTensorDataset(torch.from_numpy(vld_x),torch.from_numpy(vld_y),size=(36,36))
# test_dataset = CustomTensorDataset(torch.from_numpy(tst_x),torch.from_numpy(tst_y),size=(36,36))
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # print(len(train_dataset),len(val_dataset),len(test_dataset))
# # print(len(train_dataloader),len(val_dataloader),len(test_dataloader))
# # print(train_dataloader[0].shape)
# for x,y,z in test_dataloader:
#     print(x.shape,y.shape)
#     # print(y)
#     # print(z)
#     # 在这里处理批量数据，例如输入模型进行训练
#     pass
# [64, 36, 36, 143]
# [64, 36, 36]
# [B T C H W]
tensor1 = torch.from_numpy(trn_y)
tensor2 = torch.from_numpy(tst_y)
tensor3 = torch.from_numpy(vld_y)

c1 = torch.concat([tensor1,tensor2,tensor3],dim=0)
# print(c1.shape)
# # 将张量展平为一维张量
flattened_tensor = c1.view(-1)

# 统计所有像素的数量
total_pixel_count = flattened_tensor.numel()  # 获取元素总数
# print(total_pixel_count)
# positive_pixel_count = (flattened_tensor ==0 ).sum().item()  # 统计大于0的像素数量
positive_pixel_count_1 = (flattened_tensor >=5 ).sum().item()  # 统计大于0的像素数量
positive_pixel_count_2 = (flattened_tensor >2 ).sum().item()  # 统计大于0的像素数量
positive_pixel_count = positive_pixel_count_2 - positive_pixel_count_1
# print(positive_pixel_count)
print(positive_pixel_count/total_pixel_count)
# positive_pixel_count = (0.2> flattened_tensor >= 0.1).sum().item()  # 统计大于0的像素数量
# print(positive_pixel_count)
# positive_pixel_count = (0.5> flattened_tensor >= 0.2).sum().item()  # 统计大于0的像素数量
# print(positive_pixel_count)
# positive_pixel_count = (1> flattened_tensor >= 0.5).sum().item()  # 统计大于0的像素数量
# print(positive_pixel_count)
# positive_pixel_count = (2> flattened_tensor >= 1).sum().item()  # 统计大于0的像素数量
# print(positive_pixel_count)
# positive_pixel_count = (5> flattened_tensor >= 2).sum().item()  # 统计大于0的像素数量
# print(positive_pixel_count)
# positive_pixel_count = (flattened_tensor > 5).sum().item()  # 统计大于0的像素数量
# print(positive_pixel_count)