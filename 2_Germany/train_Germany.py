import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import time
from time import gmtime, strftime
import os
from data.data_split import cyclic_split
from data.dataset import get_dataset_class,CustomTensorDataset
from data.transforms import ClassifyByThresholds
from trainer import NIMSTrainer,NIMSTrainer_Two,NIMSTrainer_Germnay_Two,NIMSTrainer_Germany
from utils import *

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(args):
    device = set_device(args)
    fix_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    

    # Set experiment name and use it as process name if possible
    experiment_name = get_experiment_name(args)
    current_time = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    args.experiment_name = experiment_name+ "_" + current_time
    experiment_name = args.experiment_name
    
    print('Running Experiment'.center(30).center(80, "="))
    print(experiment_name)

    print("Using date intervals")
    print("#" * 80)
    for start, end in args.date_intervals:
        print("{} - {}".format(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    print("#" * 80)

    save_path = args.dataset_dir
    trn_x = np.load(save_path + '01_trn_x.npy')
    trn_y = np.load(save_path + '01_trn_y.npy')
    tst_x = np.load(save_path + '01_tst_x.npy')
    tst_y = np.load(save_path + '01_tst_y.npy')
    vld_x = np.load(save_path + '01_vld_x.npy')
    vld_y = np.load(save_path + '01_vld_y.npy')

    batch_size = args.batch_size
    train_dataset = CustomTensorDataset(torch.from_numpy(trn_x),torch.from_numpy(trn_y),size=(64,64))
    val_dataset = CustomTensorDataset(torch.from_numpy(vld_x),torch.from_numpy(vld_y),size=(64,64))
    test_dataset = CustomTensorDataset(torch.from_numpy(tst_x),torch.from_numpy(tst_y),size=(64,64))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    nwp_sample = torch.rand(1, 1, 143, 64, 64)
    model, criterion, dice_criterion = set_model(nwp_sample, device, args)
    normalization = None
    # Train model
    optimizer, scheduler = set_optimizer(model, args)
    
    if args.use_two:
        nims_trainer = NIMSTrainer_Germnay_Two(model, criterion, dice_criterion, optimizer, scheduler, device,
                                train_loader, valid_loader, test_loader, experiment_name,
                                args, normalization=normalization)
    else:
        nims_trainer = NIMSTrainer_Germany(model, criterion, dice_criterion, optimizer, scheduler, device,
                        train_loader, valid_loader, test_loader, experiment_name,
                        args, normalization=normalization)
        
    nims_trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
