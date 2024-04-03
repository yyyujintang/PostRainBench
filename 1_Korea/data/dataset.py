try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from datetime import datetime, timedelta
from typing import List, Tuple, Type, Union

import numpy as np
import torch
from data.base_dataset import *
from torch.utils.data import Dataset
from data.transforms import ClassifyByThresholds
from einops import rearrange


__all__ = ['StandardDataset', 'GdapsKimDataset', 'get_dataset_class']


class StandardDataset(Dataset):
    """
    NWP (feature), AWS (target)에 해당하는 BaseDataset 두 개를 토치 Dataset 형태로 감싸는 클래스입니다. 본 클래스는 추상 클래스로,
    자식 클래스를 만들고 `nwp|aws_base_dataset_class` 클래스 속성을 지정해서 사용해야 합니다. 아래 GdapsKimDataset 참고.

    본 클래스에서 담당하는 주요 역할은 주어진 학습 범위 (`start_date`, `end_date`)와 `lead_time` 범위를 바탕으로, 학습에 사용할 target 시점을
    산정하는 작업입니다. 산정된 target 시점은 `self.target_timestamps`에 저장됩니다.
    """
    nwp_base_dataset_class = None
    aws_base_dataset_class = None

    def __init__(self, utc: Union[int, List[int]], window_size: int, root_dir: str,
                 date_intervals: List[Tuple[datetime, datetime]],
                 variable_filter=None, start_lead_time=6, end_lead_time=25,
                 transform=None, target_transform=None, null_targets=False):
        """
        :param utc:
        :param window_size:
        :param root_dir:
        :param date_intervals:
        :param variable_filter:
        :param start_lead_time:
        :param end_lead_time:
        :param transform:
        :param target_transform:
        :param null_targets: If True, don't load/output targets. This is used when labels are not available, for example,
        realtime prediction. __getitem__ will return (nwp, None, timestamp) for code reusability.
        """
        if isinstance(utc, int):
            self.utc_list = [utc]
        elif isinstance(utc, Iterable):
            self.utc_list = list(utc)
        else:
            raise ValueError('Invalid `utc` argument: {}'.format(utc))
        self.window_size = window_size
        self.date_intervals = date_intervals
        self.transform = transform
        self.target_transform = target_transform
        self.start_lead_time = start_lead_time
        self.end_lead_time = end_lead_time
        self.null_targets = null_targets

        self.nwp_base_dataset: BaseDataset = self.nwp_base_dataset_class(root_dir, variable_filter)
        if self.null_targets:
            self.aws_base_dataset = None
        else:
            self.aws_base_dataset: AwsBaseDataset = self.aws_base_dataset_class(root_dir)
        self.target_timestamps = self._build_target_timestamps()

    def _build_target_timestamps(self) -> List[Tuple[datetime, int]]:
        target_timestamps = []
        target_timestamp_candidates = []
        for start_date, end_date in self.date_intervals:
            origin = start_date
            while origin < end_date + timedelta(days=1):
                for h in range(self.start_lead_time, self.end_lead_time):
                    target_timestamp_candidates.append((origin, h))
                origin += timedelta(days=1)

        for (origin, lead_time) in target_timestamp_candidates:
            target = origin + timedelta(hours=lead_time)
            found = True
            if not self.null_targets and (target, 0) not in self.aws_base_dataset.timestamps:
                found = False
                # print("Warning: AWS data missing for target time: {}".format(target.strftime("%Y-%m-%d %H:%M")))
            if (origin, lead_time) not in self.nwp_base_dataset.timestamps:
                found = False
                # print("Warning: NWP data missing for target timestamp: {} (+{}H)".format(origin.strftime("%Y-%m-%d %H:%M"), lead_time))

            if found:
                target_timestamps.append((origin, lead_time))

        print("Using total of {} target timestamps".format(len(target_timestamps)))
        if not target_timestamps:
            raise AssertionError("No target timestamps are available due to missing data")
        unused = len(target_timestamp_candidates) - len(target_timestamps)
        if unused:
            fmt = "WARNING: {} target timestamp candidates are unused due to missing data within the given range"
            print(fmt.format(unused))

        return target_timestamps

    def __len__(self):
        return len(self.target_timestamps)

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray, torch.tensor]:
        target_timestamp = self.target_timestamps[index]
        origin, lead_time = target_timestamp
        target = origin + timedelta(hours=lead_time)

        nwp = self.nwp_base_dataset.load_window(origin, lead_time, self.window_size)
        if self.transform:
            nwp = self.transform(nwp)

        if self.null_targets:
            aws = torch.tensor(0)
        else:
            aws = self.aws_base_dataset.load_array(target, 0)
            if self.target_transform:
                aws = self.target_transform(aws)

        return nwp, aws, torch.tensor([origin.year, origin.month, origin.day, origin.hour, lead_time])


class GdapsKimDataset(StandardDataset):
    nwp_base_dataset_class = GdapsKimBaseDataset
    aws_base_dataset_class = AwsBaseDatasetForGdapsKim


DATASET_CLASSES = {
    'gdaps_kim': GdapsKimDataset,
}


def get_dataset_class(key) -> Type[StandardDataset]:
    if key not in DATASET_CLASSES:
        raise ValueError('{} is not a valid dataset'.format(key))
    return DATASET_CLASSES[key]



class CustomTensorDataset(Dataset):
    def __init__(self, input_tensors,gt_tensors,size=(32,32)):
        self.input_tensors = input_tensors
        self.gt_tensors = gt_tensors
        self.size = size

    def __getitem__(self, index):
        x = self.input_tensors[index]
        y = self.gt_tensors[index]
        x = torch.nn.functional.interpolate(x.permute(2,0,1).unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)
        x = x.squeeze().permute(1,2,0)
        y_resized = torch.nn.functional.interpolate(y.unsqueeze(0).unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)
        y_resized = y_resized.squeeze()
        y_one_hot = ClassifyByThresholds([0.00001,2])(y_resized) 
        x = rearrange(x.unsqueeze(0), 't h w c-> t c h w')
        y_one_hot = y_one_hot.unsqueeze(0)
        return x,y_one_hot,y_resized.unsqueeze(0)
    def __len__(self):
        return self.input_tensors.shape[0]
    
class CustomTensorDataset_SD(Dataset):
    def __init__(self, input_tensors,gt_tensors):
        self.input_tensors = input_tensors
        self.gt_tensors = gt_tensors

    def __getitem__(self, index):
        x = self.input_tensors[index].float()
        y = self.gt_tensors[index].float()
        # print(y.shape)
        y_one_hot = ClassifyByThresholds([0.00001,2.0])(y) 
        # print(y_one_hot)
        # if (y_one_hot>=2).any():
        #     print("heavy")
        x = x.unsqueeze(0)
        # x = rearrange(x.unsqueeze(0), 't h w c-> t c h w')
        y_one_hot = y_one_hot.unsqueeze(0)
        return x,y_one_hot,y.unsqueeze(0)
    def __len__(self):
        return self.input_tensors.shape[0]
    