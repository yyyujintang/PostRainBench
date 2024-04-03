import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import time
from time import gmtime, strftime
import os
from data.data_split import cyclic_split
from data.dataset import get_dataset_class
from data.transforms import InterpolateAWS, ToTensor, ClassifyByThresholds, UndersampleDry, UndersampleGlobal
from trainer import NIMSTrainer,NIMSTrainer_Two
from utils import *
# import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

    # Load transforms for x, y, and augmented y. By default, augmentation for x is not considered.
    transform = transforms.Compose([
        ToTensor(),
    ])
    target_transform_list = [
        ToTensor(),
        ClassifyByThresholds(args.rain_thresholds),
    ]
    target_transform = transforms.Compose(target_transform_list)
    augmented_target_transform_list = [
        ToTensor(),
        ClassifyByThresholds(args.rain_thresholds),
    ]
    if args.interpolate_aws:
        print("Using interpolated AWS targets")
        augmented_target_transform_list.insert(1, InterpolateAWS())
    if args.dry_sampling_rate < 1.0:
        print("Undersampling dry points")
        augmented_target_transform_list.append(UndersampleDry(args.dry_sampling_rate))
    if args.global_sampling_rate < 1.0:
        print("Undersampling all points")
        augmented_target_transform_list.append(UndersampleGlobal(args.global_sampling_rate))
    augmented_target_transform = transforms.Compose(augmented_target_transform_list)

    # Load `StandardDataset`s
    dataset = load_dataset_from_args(args, transform=transform, target_transform=target_transform)
    augmented_dataset = load_dataset_from_args(args, transform=transform, target_transform=augmented_target_transform)

    # Get normalization transform
    normalization = None
    if args.normalization:
        print()
        print('Normalization'.center(30).center(80, "="))
        max_values, min_values = get_min_max_values(augmented_dataset)

        normalization = {'max_values': max_values,
                         'min_values': min_values}

    nwp_sample, gt_sample, _ = augmented_dataset[0]  # samples to determine shape of tensor
    # print(nwp_sample.shape)
    model, criterion, dice_criterion = set_model(nwp_sample, device, args)

    # Apply data splits and load `DataLoader`
    train_dataset, _, _ = cyclic_split(augmented_dataset)
    _, valid_dataset, test_dataset = cyclic_split(dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker, generator=g,)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker, generator=g,)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker, generator=g,)

    # Apply heavy-rain specific strategies
    if args.target_precipitation == 'heavy':  # Heavy Rain Case
        point_cnt_dict = {i: 0 for i in list(range(1 + len(args.rain_thresholds))) + [-9999]}

        for data in train_loader:
            target = data[1].cpu().numpy()
            uni, cnt = np.unique(target, return_counts=True)

            rain_cnt_dict = dict(zip(uni, cnt))

            for rain_idx, cnt in rain_cnt_dict.items():
                point_cnt_dict[rain_idx] += cnt

        rain_cnt = 0

        for rain_idx, cnt in point_cnt_dict.items():
            if rain_idx not in [0, -9999]:
                rain_cnt += cnt

        if args.no_rain_ratio:
            args.no_rain_resample_ratio = (args.no_rain_ratio * rain_cnt) / point_cnt_dict[0]
        else:
            args.no_rain_resample_ratio = None
        print("No Rain Resample Ratio", args.no_rain_resample_ratio)

    # Train model
    optimizer, scheduler = set_optimizer(model, args)
    if args.use_two:
        nims_trainer = NIMSTrainer_Two(model, criterion, dice_criterion, optimizer, scheduler, device,
                                train_loader, valid_loader, test_loader, experiment_name,
                                args, normalization=normalization)
    else:
        nims_trainer = NIMSTrainer(model, criterion, dice_criterion, optimizer, scheduler, device,
                        train_loader, valid_loader, test_loader, experiment_name,
                        args, normalization=normalization)
        
    nims_trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
