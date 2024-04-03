import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from einops import rearrange
from loss import *


__all__ = ['CrossEntropyLoss', 'CrossEntropyLoss_Two','DiceLoss']


class ClassificationStat(nn.Module):
    """
    Superclass that provides functionality for evaluation based on original station observation information.
    """
    def __init__(self, args, num_classes):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.reference = args.reference

    def get_stat(self, preds, targets, mode):
        """
        Get predictions and labels for original station observations.
        """
        if mode == 'train':
            _, pred_labels = preds.topk(1, dim=1, largest=True, sorted=True)
            b = pred_labels.shape[0]
            if b == 0:
                return

            pred_labels = pred_labels.squeeze(1).detach().reshape(b, -1)
            target_labels = targets.data.detach().reshape(b, -1)

        elif (mode == 'valid') or (mode == 'test'):
            # Old
            _, pred_labels = preds.topk(1, dim=1, largest=True, sorted=True)

            # Current
            # preds = F.softmax(preds, dim=1)
            # true_probs = preds[:, 1, :].unsqueeze(1)
            # pred_labels = torch.where(true_probs > 0.05,
            #                           torch.ones(true_probs.shape).to(0),
            #                           torch.zeros(true_probs.shape).to(0))
            
            b, _, num_stn = pred_labels.shape
            assert (b, num_stn) == targets.shape

        pred_labels = pred_labels.squeeze(1).detach()
        target_labels = targets.data.detach()

        return pred_labels, target_labels

    def remove_missing_station(self, targets):
        _targets = targets.squeeze(0)
        targets_idx = (_targets >= 0).nonzero().cpu().tolist()  # [(x, y)'s] - hardcode
        return np.array(targets_idx)

# class CrossEntropyLoss(ClassificationStat):
#     def __init__(self, args, device, num_classes, experiment_name=None):
#         super().__init__(args=args, num_classes=num_classes)
#         self.device = device
#         self.dataset_dir = args.dataset_dir
#         self.experiment_name = experiment_name
#         self.model_name = args.model
#         self.args = args

#     def forward(self, preds, targets, target_time, mode):
#         """
#         :param preds: model predictions in B x C x W x H format (batch size, channels, width, height)
#         :param targets: targets in B x W x H format
#         :param target_time:
#         :param mode:
#         :return:
#         """
#         if self.model_name == 'unet' or 'metnet' or 'swinunet':
#             assert preds.shape[0] == targets.shape[0] and preds.shape[2] == targets.shape[1] and preds.shape[3] == \
#                    targets.shape[2]
#         elif self.model_name == 'convlstm':
#             pass  # Chek the output size of convlstm model

#         targets_shape = targets.shape

#         if self.args.no_rain_ratio is not None:
#             if self.args.target_precipitation == 'rain':  # Rain Case
#                 unique, counts = np.unique(targets[0].cpu().numpy(), return_counts=True)
#                 rain_counts_dict = dict(zip(unique, counts))

#                 rain_cnt = 0

#                 for rain_index in rain_counts_dict.keys():
#                     if rain_index not in [0, -9999]:
#                         rain_cnt += rain_counts_dict[rain_index]

#                 no_rain_cnt = int(rain_cnt * self.args.no_rain_ratio)

#                 if no_rain_cnt == 0:
#                     return None, None, None
#                 elif no_rain_cnt < rain_counts_dict[0]:
#                     target_1d = targets[0].cpu().numpy().flatten()

#                     for idx in np.random.choice(np.where(target_1d == 0)[0], rain_counts_dict[0] - no_rain_cnt):
#                         target_1d[idx] = -9999

#                     targets = torch.from_numpy(target_1d).view(targets_shape).cuda()
#             else:
#                 # Counts rain points
#                 unique, counts = np.unique(targets[0].cpu().numpy(), return_counts=True)
#                 rain_counts_dict = dict(zip(unique, counts))

#                 target_1d = targets[0].cpu().numpy().flatten()

#                 for idx in np.random.choice(np.where(target_1d == 0)[0],
#                                             int(rain_counts_dict[0] * (1 - self.args.no_rain_resample_ratio)),
#                                             replace=False):
#                     target_1d[idx] = -9999

#                 targets = torch.from_numpy(target_1d).view(targets_shape).cuda()

#         stn_codi = self.remove_missing_station(targets)

#         stn_preds = preds[:, :, stn_codi[:, 0], stn_codi[:, 1]]
#         stn_targets = targets[:, stn_codi[:, 0], stn_codi[:, 1]]

#         pred_labels, target_labels = self.get_stat(stn_preds, stn_targets, mode=mode)

#         loss = F.cross_entropy(stn_preds, stn_targets, reduction='none')
#         loss = torch.mean(torch.mean(loss, dim=1))

#         return loss, pred_labels, target_labels


class CrossEntropyLoss(ClassificationStat):
    def __init__(self, args, device, num_classes, experiment_name=None):
        super().__init__(args=args, num_classes=num_classes)
        self.device = device
        self.dataset_dir = args.dataset_dir
        self.experiment_name = experiment_name
        self.model_name = args.model
        self.num_classes = num_classes
        self.args = args
        self.loss = args.loss
        self.weight_version = args.weight_version

    def forward(self, preds, targets, target_time, mode):
        """
        :param preds: model predictions in B x C x W x H format (batch size, channels, width, height)
        :param targets: targets in B x W x H format
        :param target_time:
        :param mode:
        :return:
        """
        # print('preds.shape',preds.shape)
        # print('targets.shape',targets.shape)
        # if self.model_name == 'unet' or 'metnet' or 'swinunet' or 'unet3d' or 'TSViT':
        #     assert preds.shape[0] == targets.shape[0] and preds.shape[2] == targets.shape[1] and preds.shape[3] == \
        #            targets.shape[2]
        # elif self.model_name == 'convlstm':
            # pass  # Chek the output size of convlstm model

        targets_shape = targets.shape

        if self.args.no_rain_ratio is not None:
            if self.args.target_precipitation == 'rain':  # Rain Case
                unique, counts = np.unique(targets[0].cpu().numpy(), return_counts=True)
                rain_counts_dict = dict(zip(unique, counts))

                rain_cnt = 0

                for rain_index in rain_counts_dict.keys():
                    if rain_index not in [0, -9999]:
                        rain_cnt += rain_counts_dict[rain_index]

                no_rain_cnt = int(rain_cnt * self.args.no_rain_ratio)

                if no_rain_cnt == 0:
                    return None, None, None
                elif no_rain_cnt < rain_counts_dict[0]:
                    target_1d = targets[0].cpu().numpy().flatten()

                    for idx in np.random.choice(np.where(target_1d == 0)[0], rain_counts_dict[0] - no_rain_cnt):
                        target_1d[idx] = -9999

                    targets = torch.from_numpy(target_1d).view(targets_shape).cuda()
            else:
                # Counts rain points
                unique, counts = np.unique(targets[0].cpu().numpy(), return_counts=True)
                rain_counts_dict = dict(zip(unique, counts))

                target_1d = targets[0].cpu().numpy().flatten()

                for idx in np.random.choice(np.where(target_1d == 0)[0],
                                            int(rain_counts_dict[0] * (1 - self.args.no_rain_resample_ratio)),
                                            replace=False):
                    target_1d[idx] = -9999

                targets = torch.from_numpy(target_1d).view(targets_shape).cuda()

        # stn_codi = self.remove_missing_station(targets)

        # stn_preds = preds[:, :, stn_codi[:, 0], stn_codi[:, 1]]
        # stn_targets = targets[:, stn_codi[:, 0], stn_codi[:, 1]]
        
        stn_preds,stn_targets = preds,targets.squeeze()
        
        if self.weight_version == 1:
            ce_weight = torch.tensor([1,2,30],dtype=torch.float)
        elif self.weight_version == 2:
            ce_weight = torch.tensor([1,5,30],dtype=torch.float)
        elif self.weight_version == 3:
            ce_weight = torch.tensor([1,5,50],dtype=torch.float)
        elif self.weight_version == 4:
            ce_weight = torch.tensor([1,8,30],dtype=torch.float)
        elif self.weight_version == 5:
            ce_weight = torch.tensor([1,8,50],dtype=torch.float)
        elif self.weight_version == 6:
            ce_weight = torch.tensor([1,2,100],dtype=torch.float)
        else:
            print('!!!!!!!!!!!!Weight Not Implement Yet!!!!!!!!!!!!')


        pred_labels, target_labels = self.get_stat(stn_preds, stn_targets, mode=mode)
        # stn_preds_trans = torch.transpose(stn_preds, 1, 2)
        if self.loss == 'ce':
            # loss = nn.CrossEntropyLoss(weight=ce_weight.to(self.device))(stn_preds, stn_targets) 
            loss = nn.CrossEntropyLoss()(stn_preds, stn_targets) 
        elif self.loss == 'Bal_ce':
            loss = Bal_CE_loss()(stn_preds, stn_targets)
        elif self.loss == 'focal':
            loss = FocalLoss(gamma=0.5)(stn_preds, stn_targets)
        elif self.loss == 'gce':
            loss = GeneralizedCrossEntropy(num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'mae':
            loss = MeanAbsoluteError(num_classes=self.num_classes)(stn_preds, stn_targets)      
        elif self.loss == 'ce+mae':
            loss = CEandMAE(beta = 1.0, num_classes=self.num_classes)(stn_preds, stn_targets)  
        elif self.loss == 'nce+mae':
            loss = NCEandMAE(alpha = 1.0, beta = 1.0, num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'nce+rce':
            loss = NCEandRCE(alpha = 1.0, beta = 1.0, num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'nce':
            loss = NormalizedCrossEntropy(num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'nfl+mae':
            loss = NFLandMAE(alpha = 1.0, beta = 1.0,num_classes=self.num_classes)(stn_preds, stn_targets)        
        elif self.loss == 'nfl+rce':
            loss = NFLandRCE(alpha = 1.0, beta = 1.0,num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'nfl':
            loss = NormalizedFocalLoss( scale=10.0, gamma=0.5, num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'ngce+mae':
            loss = NGCEandMAE(alpha = 1.0, beta = 1.0,num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'ngce+rce':
            loss = NGCEandRCE(alpha = 1.0, beta = 1.0,num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'ngce':
            loss = NormalizedGeneralizedCrossEntropy(num_classes=self.num_classes)(stn_preds, stn_targets)
        # elif args.loss == 'nlnl':
            # loss = NCEandMAE(num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'rce':
            loss = ReverseCrossEntropy(num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'sce':
            loss = SCELoss(alpha = 0.1, beta = 1.0,num_classes=self.num_classes)(stn_preds, stn_targets)
        else:
            print('!!!!!!!!!!!!Loss Not Implement Yet!!!!!!!!!!!!')
    
        return loss, pred_labels, target_labels
 

class CrossEntropyLoss_Two(ClassificationStat):
    def __init__(self, args, device, num_classes, experiment_name=None):
        super().__init__(args=args, num_classes=num_classes)
        self.device = device
        self.dataset_dir = args.dataset_dir
        self.experiment_name = experiment_name
        self.model_name = args.model
        self.num_classes = num_classes
        self.args = args
        self.loss = args.loss
        self.weight_version = args.weight_version
        self.alpha = args.alpha

    def forward(self, preds, preds2, targets, target_time, mode):
        """
        :param preds: model predictions in B x C x W x H format (batch size, channels, width, height)
        :param targets: targets in B x W x H format
        :param target_time:
        :param mode:
        :return:
        """
        # print('preds.shape',preds.shape)
        # print('targets.shape',targets.shape)
        # if self.model_name == 'unet' or 'metnet' or 'swinunet'or 'unet3d' or 'TSViT':
        #     assert preds.shape[0] == targets.shape[0] and preds.shape[2] == targets.shape[1] and preds.shape[3] == \
        #            targets.shape[2]
        # elif self.model_name == 'convlstm':
        #     pass  # Chek the output size of convlstm model

        targets_shape = targets.shape

        if self.args.no_rain_ratio is not None:
            if self.args.target_precipitation == 'rain':  # Rain Case
                unique, counts = np.unique(targets[0].cpu().numpy(), return_counts=True)
                rain_counts_dict = dict(zip(unique, counts))

                rain_cnt = 0

                for rain_index in rain_counts_dict.keys():
                    if rain_index not in [0, -9999]:
                        rain_cnt += rain_counts_dict[rain_index]

                no_rain_cnt = int(rain_cnt * self.args.no_rain_ratio)

                if no_rain_cnt == 0:
                    return None, None, None
                elif no_rain_cnt < rain_counts_dict[0]:
                    target_1d = targets[0].cpu().numpy().flatten()

                    for idx in np.random.choice(np.where(target_1d == 0)[0], rain_counts_dict[0] - no_rain_cnt):
                        target_1d[idx] = -9999

                    targets = torch.from_numpy(target_1d).view(targets_shape).cuda()
            else:
                # Counts rain points
                unique, counts = np.unique(targets[0].cpu().numpy(), return_counts=True)
                rain_counts_dict = dict(zip(unique, counts))

                target_1d = targets[0].cpu().numpy().flatten()

                for idx in np.random.choice(np.where(target_1d == 0)[0],
                                            int(rain_counts_dict[0] * (1 - self.args.no_rain_resample_ratio)),
                                            replace=False):
                    target_1d[idx] = -9999

                targets = torch.from_numpy(target_1d).view(targets_shape).cuda()

        stn_codi = self.remove_missing_station(targets)

        # stn_preds = preds[:, :, stn_codi[:, 0], stn_codi[:, 1]]
        # stn_preds2 = preds2[:, :, stn_codi[:, 0], stn_codi[:, 1]]
        # stn_targets = targets[:, stn_codi[:, 0], stn_codi[:, 1]]
        
        stn_preds,stn_preds2,stn_targets = preds,preds2,targets.squeeze(1)
        
        
        if self.weight_version == 1:
            ce_weight = torch.tensor([1,5,10],dtype=torch.float)
        elif self.weight_version == 2:
            ce_weight = torch.tensor([1,5,30],dtype=torch.float)
        elif self.weight_version == 3:
            ce_weight = torch.tensor([1,5,20],dtype=torch.float)
        elif self.weight_version == 4:
            ce_weight = torch.tensor([1,15,10],dtype=torch.float)
        elif self.weight_version == 5:
            ce_weight = torch.tensor([1,25,20],dtype=torch.float)
        elif self.weight_version == 6:
            ce_weight = torch.tensor([1,2,10],dtype=torch.float)
        else:
            print('!!!!!!!!!!!!Weight Not Implement Yet!!!!!!!!!!!!')

        pred_labels, target_labels = self.get_stat(stn_preds, stn_targets, mode=mode)
        # stn_preds_trans = torch.transpose(stn_preds, 1, 2)
        if self.loss == 'ce':
            loss = nn.CrossEntropyLoss()(stn_preds, stn_targets) 
        elif self.loss == 'ce+mse':
            loss = nn.CrossEntropyLoss(weight=ce_weight.to(self.device))(stn_preds, stn_targets) + self.alpha * nn.MSELoss(reduction='mean')(stn_preds2.squeeze(1).float(), stn_targets.float()).mean()
            # loss = nn.CrossEntropyLoss(weight=ce_weight.to(self.device))(stn_preds, stn_targets) + self.alpha * torch.mul(mse_weight, nn.MSELoss(reduction='none')(stn_preds2.squeeze(1).float(), stn_targets.float())).mean()
        elif self.loss == 'focal':
            loss = FocalLoss(gamma=0.5)(stn_preds, stn_targets)
        elif self.loss == 'focal+mse':
            loss = FocalLoss(gamma=0.5)(stn_preds, stn_targets)+ nn.MSELoss(reduction='none')(stn_preds2.squeeze(1).float(), stn_targets.float()).mean()
        elif self.loss == 'gce':
            loss = GeneralizedCrossEntropy(num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'mae':
            loss = MeanAbsoluteError(num_classes=self.num_classes)(stn_preds, stn_targets)      
        elif self.loss == 'ce+mae':
            loss = CEandMAE(beta = 1.0, num_classes=self.num_classes)(stn_preds, stn_targets)  
        elif self.loss == 'nce+mae':
            loss = NCEandMAE(alpha = 1.0, beta = 1.0, num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'nce+rce':
            loss = NCEandRCE(alpha = 1.0, beta = 1.0, num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'nce':
            loss = NormalizedCrossEntropy(num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'nfl+mae':
            loss = NFLandMAE(alpha = 1.0, beta = 1.0,num_classes=self.num_classes)(stn_preds, stn_targets)        
        elif self.loss == 'nfl+rce':
            loss = NFLandRCE(alpha = 1.0, beta = 1.0,num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'nfl+mse':
            loss = NormalizedFocalLoss( scale=10.0, gamma=0.5, num_classes=self.num_classes)(stn_preds, stn_targets)+ nn.MSELoss(reduction='none')(stn_preds2.squeeze(1).float(), stn_targets.float()).mean()
        elif self.loss == 'nfl':
            loss = NormalizedFocalLoss( scale=10.0, gamma=0.5, num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'ngce+mae':
            loss = NGCEandMAE(alpha = 1.0, beta = 1.0,num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'ngce+rce':
            loss = NGCEandRCE(alpha = 1.0, beta = 1.0,num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'ngce':
            loss = NormalizedGeneralizedCrossEntropy(num_classes=self.num_classes)(stn_preds, stn_targets)
        # elif args.loss == 'nlnl':
            # loss = NCEandMAE(num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'rce':
            loss = ReverseCrossEntropy(num_classes=self.num_classes)(stn_preds, stn_targets)
        elif self.loss == 'sce':
            loss = SCELoss(alpha = 0.1, beta = 1.0,num_classes=self.num_classes)(stn_preds, stn_targets)
        else:
            print('Not Implement Yet！')
    
        return loss, pred_labels, target_labels    

  
class DiceLoss(nn.Module):
    def __init__(self, args, device, num_classes, balance, experiment_name=None):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.reference = args.reference
        self.alpha = 0.75
        self.device = device
        self.balance = balance

        
    def forward(self, pred_labels, target_labels, device):
        confusion_matrix = Variable(torch.zeros((self.num_classes, self.num_classes)), requires_grad=True).to(self.device)
    
        for i in range(pred_labels.shape[1]):
            confusion_matrix[target_labels[0, i], pred_labels[0, i]] += (target_labels[0, i]==pred_labels[0, i]).sum()
        
        dice = 0.0
        for clas_num in range(1, self.num_classes):
            tp, fn, fp = confusion_matrix[clas_num, clas_num], confusion_matrix[clas_num, :].sum()-confusion_matrix[clas_num, clas_num], confusion_matrix[:, clas_num].sum()-confusion_matrix[clas_num, clas_num]
            dice += (2 * tp) / (2*tp + fn + fp + 1e-6)
        dice /= (self.num_classes-1)

        return self.balance * (1 - dice ** self.alpha)

# a = torch.randn(4,3,224,224)
# b = torch.randn(4,224,224)
# loss = CrossEntropyLoss
# print(loss)

# B = 1
# # a = torch.rand(B,3,50,65).cuda()
# b = torch.rand(B,1,50,65).cuda()
# c = torch.rand(B,50,65).cuda()
# d = torch.rand(B,1).cuda()
# e = torch.rand(B,512).cuda()
# # loss, pred_labels, target_labels = CrossEntropyLoss_Two()(a,b,c.long())
# loss = CrossEntropyLoss_Two()(b,c.long(),d,e)
# print(loss)