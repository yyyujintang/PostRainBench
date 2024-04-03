import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import time
import cv2
import torch.nn.functional as F
import random
from pytorch_lightning import seed_everything
from data.dataset import get_dataset_class,CustomTensorDataset_SD

# seed_everything(0)

# class WeatherDataset(Dataset):
#     def __init__(self, root_dir, stage="train",ratio=0.2,size =(430,815),input_transform=None,output_transform=None):
#         self.root_dir = root_dir  # 数据存放的根目录
#         self.input_variables = ["cape","d2m","msl","pwat","r_500","r_700","r_850","r_925","rain",
#                                 "rain_big","rain_thud","t_500","t_700","t_850","t_925","t2m",
#                                 "u_10","u_200","u_500","u_700","u_850","u_925","v_10","v_200","v_500",
#                                 "v_700","v_850","v_925"]  
#         self.nwp_rain = ["rain"]  
#         self.output_variable = 'ob'  # 输出变量的名称
#         self.stage = stage
#         self.ratio = ratio
#         self.size = size
#         self.samples = []  # 样本列表
#         self.input_transform = input_transform
#         self.output_transform = output_transform
#         filenames = os.listdir(root_dir)
#         # filenames = sorted(filenames)
#         random.shuffle(filenames)
#         length = len(filenames)
        
#         # train_files = filenames
#         train_files = filenames[:int(length*0.6)]
#         val_files = filenames[int(length*0.6):int(length*0.8)]
#         test_files = filenames[int(length*0.8):]
       
#         if self.stage == "train":
#             files = train_files
#         elif self.stage == "val":
#             files = val_files
#         elif self.stage == "test":
#             files = test_files
#         for sample_id in files:  # 遍历所有日期的文件夹
#             sample_id_dir = os.path.join(root_dir, sample_id)  # 当前日期的完整路径
#             if not os.path.isdir(sample_id_dir):  # 如果不是文件夹，则跳过
#                 continue
#             for filename in os.listdir(sample_id_dir):  # 遍历当前日期文件夹里的所有文件
#                 if not filename.endswith('.png'):  # 如果不是png文件，则跳过
#                     continue
#                 if len(filename.split('_')) == 3:
#                     variable_name, sample_id, frame_num = filename.split('_')[0:3]  # 从文件名中提取样本ID和帧数
#                 elif len(filename.split('_')) == 4:
#                     variable_name, sample_id, frame_num = filename.split('_')[0]+'_'+filename.split('_')[1], filename.split('_')[2], filename.split('_')[3]
#                 else:
#                     continue
#                 sample_id = int(sample_id)  # 转换成整数类型
#                 frame_num = int(frame_num.split('.')[0])  # 去掉文件扩展名，并转换成整数类型
#             # self.samples.append((day, variable_name, sample_id, frame_num))  # 将当前样本加入列表
#             self.samples.append(sample_id)  # 将当前样本加入列表

#     def __len__(self):
#         # print(len(self.samples))
#         return len(self.samples)  # 返回样本数目

#     def __getitem__(self, idx):
#         sample_id = self.samples[idx]  # 获取指定索引的样本的日期、ID和帧数
#         sample_dir = os.path.join(self.root_dir, str(sample_id))  # 根据日期生成该样本的路径
#         input_data = []  # 输入数据列表
#         input_rain = []
#         output_data_24h = []
       
#         for hour in range(3,27,3):
#             var_name_3h_8_imgs = []
#             for var_name in self.input_variables:  # 遍历输入变量列表
#                 var_path = os.path.join(sample_dir, '{}_{}_{}.png'.format(var_name, str(sample_id), hour))  # 根据变量名称、样本ID和帧数生成该变量的完整路径
#                 var_data = cv2.imread(var_path, cv2.IMREAD_UNCHANGED)
#                 var_data = cv2.resize(var_data, self.size)               
#                 var_data = torch.tensor(var_data,dtype=float)                
#                 if self.input_transform is not None:
#                     var_data = self.input_transform(var_data) 
#                 var_name_3h_8_imgs.append(var_data)
#             channel_data = torch.stack(var_name_3h_8_imgs, dim=0) 
#             input_data.append(channel_data)  # 将该变量的数据加入输入列表
            
#         for hour in range(3,27,3):
#             var_rain_3h_8_imgs = []
#             for var_name in self.nwp_rain:  # 遍历输入变量列表
#                 var_path = os.path.join(sample_dir, '{}_{}_{}.png'.format(var_name, str(sample_id), hour))  # 根据变量名称、样本ID和帧数生成该变量的完整路径
#                 var_data = cv2.imread(var_path, cv2.IMREAD_UNCHANGED)
#                 var_data = cv2.resize(var_data, self.size)               
#                 var_data = torch.tensor(var_data,dtype=float)                
#                 if self.input_transform is not None:
#                     var_data = self.input_transform(var_data) 
#                 var_rain_3h_8_imgs.append(var_data)
#             channel_data = torch.stack(var_rain_3h_8_imgs, dim=0) 
#             input_rain.append(channel_data)  # 将该变量的数据加入输入列表
        
#         # for hour in range(1,25):
#         for hour in range(3,27,3):
#             output_data = os.path.join(sample_dir, '{}_{}_{}.png'.format(self.output_variable, str(sample_id), hour))  # 根据输出变量名称、样本ID和帧数生成该变量的完整路径
#             output_data = cv2.imread(output_data, cv2.IMREAD_UNCHANGED)
#             output_data = cv2.resize(output_data, self.size)
#             output_data = torch.tensor(output_data,dtype=torch.long)
#             if self.output_transform is not None:
#                 output_data = self.output_transform(output_data)
#             output_data_24h.append(output_data)
            
#         input_data = torch.stack(input_data)
#         input_data = torch.where(input_data > 250 ,0, input_data).squeeze()
#         input_rain = torch.stack(input_rain)
#         input_rain = torch.where(input_rain > 250 ,0, input_rain).squeeze()
#         output_data = torch.stack(output_data_24h)
#         output_data = torch.where(output_data > 250 ,0, output_data)   
#         return input_data, input_rain,output_data  # 返回输入列表和输出变量的数据
    

# # # input_transform = transforms.Compose([
# # #     transforms.Resize((64, 64)),
# # #     transforms.ToTensor(),
# # #     # transforms.Normalize((0.5,), (0.5,))
# # # ])
# # # output_transform = transforms.Compose([
# # #     transforms.Resize((64, 64)),
# # #     transforms.ToTensor(),
# # #     # transforms.Normalize((0.5,), (0.5,))
# # # ])
# input_transform = None
# output_transform = None

# # train_dataset = WeatherDataset('/mnt/ssd1/yujint/challenge/train', stage="train",ratio=0.2, size=(224,224),input_transform=input_transform,output_transform=output_transform)  # 创建天气预测数据集
# val_dataset = WeatherDataset('/mnt/ssd1/yujint/challenge/train', stage="val",ratio=0.2, size=(224,224),output_transform=output_transform)  # 创建天气预测数据集
# # test_dataset = WeatherDataset('/mnt/ssd1/yujint/challenge/train', stage="test",ratio=0.2, size=(224,224),output_transform=output_transform)  # 创建天气预测数据集

# # train_dataloader = DataLoader(train_dataset, batch_size=472, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=472, shuffle=True)
# # test_dataloader = DataLoader(test_dataset, batch_size=472, shuffle=True)
# # print(len(train_dataloader))
# # print(len(test_dataloader))

# for input_data,input_rain, output_data_24h in val_dataloader:
#     # print(input_data.shape)
#     # print('input_data max min mean std:',tensor_stats(input_data))
#     # input_data = torch.flatten(input_data, start_dim=0, end_dim=1)
#     # print(input_data.shape)
#     # print(input_data.shape)
#     # # print('input_data max min mean std:',tensor_stats(input_data))
#     # output_data_24h = output_data_24h.reshape(-1,3,815,430)
#     print(input_data.shape)
#     print(input_rain.shape)
#     print(output_data_24h.shape)
#     # print('output_data max min mean std:',tensor_stats(output_data_24h))
#     input_data = torch.flatten(input_data, start_dim=0, end_dim=1)
#     input_rain = torch.flatten(input_rain, start_dim=0, end_dim=1)
#     output_data_24h = torch.flatten(output_data_24h, start_dim=0, end_dim=1)
#     print(input_data.shape)
#     print(input_rain.shape)
#     print(output_data_24h.shape)
#     # 将PyTorch张量转换为NumPy数组
#     train_nwp = input_data.numpy()
#     train_nwp_rain = input_rain.numpy()
#     train_gt = output_data_24h.numpy()

#     # 保存为.npy文件
#     np.save('val_x_224.npy', train_nwp)
#     np.save('val_c_224.npy', train_nwp_rain)
#     np.save('val_y_224.npy', train_gt)
    
# # trn_x = np.load('train_nwp.npy')
trn_y = np.load('train_gt.npy')
# # tst_x = np.load('val_nwp.npy')
vld_y = np.load('val_gt.npy')
# # vld_x = np.load('test_nwp.npy')
tst_y = np.load('test_gt.npy')

# tensor1 = torch.from_numpy(trn_y)
# tensor2 = torch.from_numpy(tst_y)
# tensor3 = torch.from_numpy(vld_y)

tensor1 = torch.from_numpy(trn_y)
tensor2 = torch.from_numpy(vld_y)
tensor3 = torch.from_numpy(tst_y)

c1 = torch.concat([tensor1,tensor2,tensor3],dim=0)
flattened_tensor = c1.view(-1)


# print(c1.shape)
# # 将张量展平为一维张量
# flattened_tensor = c1.view(-1)

# 统计所有像素的数量
total_pixel_count = flattened_tensor.numel()  # 获取元素总数
print(total_pixel_count)
# positive_pixel_count = (flattened_tensor ==0 ).sum().item()  # 统计大于0的像素数量
# positive_pixel_count_1 = (flattened_tensor >=2 ).sum().item()  # 统计大于0的像素数量
positive_pixel_count_2 = (flattened_tensor >0 ).sum().item()  # 统计大于0的像素数量
positive_pixel_count = positive_pixel_count_2 - positive_pixel_count_1
# print(positive_pixel_count)
print(positive_pixel_count/total_pixel_count)

# trn_x = np.load('train_nwp.npy')
# trn_y = np.load('train_gt.npy')

# tst_x = np.load('test_nwp.npy')
# tst_y = np.load('test_gt.npy')

# vld_x = np.load('val_nwp.npy')
# vld_y = np.load('val_gt.npy')

# batch_size = 4
# # train_dataset = CustomTensorDataset_SD(torch.from_numpy(trn_x),torch.from_numpy(trn_y))
# val_dataset = CustomTensorDataset_SD(torch.from_numpy(vld_x),torch.from_numpy(vld_y))
# # test_dataset = CustomTensorDataset_SD(torch.from_numpy(tst_x),torch.from_numpy(tst_y))
# # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# for x,y,z in valid_loader:
#     print(x.shape,y.shape)