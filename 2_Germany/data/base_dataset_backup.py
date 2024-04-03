import glob
import os
import re
from abc import abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Union

import netCDF4 as nc
import numpy as np


__all__ = ['BaseDataset', 'GdapsKimBaseDataset', 'AwsBaseDataset', 'AwsBaseDatasetForGdapsKim']


class BaseDataset:
    """
    An abstract class for easy access to NWP and AWS data in the form of `np.ndarray`s. Main features include:

    - Scan data directories based on the following class attributes below and store paths as a dictionary
      in `self.data_path_dict`.
      - `data_path_glob`: glob expression to find (potential) data paths
      - `data_path_regex`: regular expression to extract year, month, day, hour, lead_time info from data path
    - `load_array()`: load NWP or AWS sample as `np.ndarray` based on the supplied target time.

    ## Some Terminology

   - `origin`: for NWPs, the time at which simulations are executed. For AWS, the time of the observation.
               I.e., `datetime(year, month, day, hour=utc)`.
   - `lead_time`: for NWPs, the time between origin and target time (in hours). For AWS, leave as 0.
   - `target`: the target time of the prediction or observation. I.e., `origin + timedelta(hours=lead_time)`.

    ## Usage Example (GdapsKimBaseDataset)

    ```python
    utc = 6
    ds = GdapsKimBaseDataset(root_dir='/home/osilab12/ssd4', variable_filter='rain, snow')
    feature_map = ds.load_array(origin=datetime(2020, 7, 1, hour=utc), lead_time=9)
    plt.imshow(feature_map[0])  # map of rain at 2020/07/01 3:00PM predicted at 6:00AM
    ```

    ## Major Attributes

    - `data_path_dict`: a dictionary that maps target timestamps to data paths

      - key: the target timestamp in the form of `(origin: datetime, lead_time: int)`
      - value: the list of all data paths that correspond to the key in the form of `List[str]`

        - Q. can there be multiple paths for a single timestamp?
        - A. in the case of GDAPS-KIM, Unis and Pres variables are saved in different files, therefore there are
             two files that correspond to a particular target timestamp.
        - A. 예를 들어, GDAPS-KIM 모델에서는 단일면, 등압면 데이터가 각기 다른 파일에 저장되어 있기 때문에, 특정 target 시점에 해당하는 path가
    """
    dataset_dir = None  # e.g., 'GDPSKIM'
    data_path_glob: str = None  # e.g., '**/*.nc'
    data_path_regex: str = None  # regex with named groups: 'year', 'month', 'day', 'hour', 'lead_time'

    def __init__(self, root_dir: str, variable_filter: str = None):
        """
        :param root_dir: the root directory that contains `dataset_dir`
        :param variable_filter: the string that specifies which variables to load. The format is determined by the
                                implementation of `load_array`.
        """
        self._verify_class_attributes()
        self.root_dir = root_dir
        self.variable_filter = variable_filter

        # Search paths and populate `self.data_path_dict`
        self._re_path = re.compile(self.data_path_regex)
        self.data_path_dict = self._generate_data_path_dict()
        self._verify_data_path_dict(self.data_path_dict)
        self.timestamps = list(self.data_path_dict.keys())

    _ret = Union[np.ndarray, Tuple[np.ndarray, List[str]]]

    @abstractmethod
    def load_array(self, origin: datetime, lead_time: int, return_variables: bool = False) -> _ret:
        """
        For NWPs, the feature map of the prediction made at `origin` targeting +`lead_time` hours is returned in
        `np.ndarray` format. For AWS, use `lead_time=0` to load observations made at `origin`.

        This is an abstract method. You must provide an implementation for child classes based on the format of the
        actual underlying dataset files. You should reference the `data_path_dict` attribute to obtain the
        data paths corresponding to the supplied target time and parse the files to extract feature maps in
        `np.ndarray` format.

        :param origin: for NWPs, the time at which simulations are executed. For AWS, the time of the observation.
                       I.e., `datetime(year, month, day, hour=utc)`.
        :param lead_time: for NWPs, the time between origin and target time (in hours). For AWS, leave as 0.
        :param return_variables: whether to return the list of variables included in the resulting feature map.
        :return:
        """
        raise NotImplementedError

    def load_window(self, origin: datetime, lead_time: int, window_size: int, zero_padding=False) -> np.ndarray:
        arrays = []
        assert lead_time >= 0
        for i in range(window_size):  # fill from end to start
            if lead_time - i < 0:  # padding
                if zero_padding:
                    arrays.insert(0, np.zeros_like(arrays[-1]))
                else:
                    raise Exception('`window_size` must be <= `lead_time + 1` when `zero_padding=False')
            else:
                arrays.insert(0, self.load_array(origin, lead_time - i))
        return np.stack(arrays, axis=0)

    def _verify_data_path_dict(self, data_path_dict):
        pass

    def _generate_data_path_dict(self) -> Dict[Tuple[datetime, int], List[str]]:
        all_paths = glob.glob(os.path.join(self.root_dir, self.dataset_dir, self.data_path_glob), recursive=True)
        all_paths.sort()
        data_path_dict: Dict[Tuple[datetime, int], List[str]] = defaultdict(list)
        for path in all_paths:
            origin_datetime, lead_time = self._parse_data_path(path)
            data_path_dict[(origin_datetime, lead_time)].append(path)
        return data_path_dict

    def _verify_class_attributes(self):
        try:
            assert self.dataset_dir is not None
            assert self.data_path_regex is not None
            assert self.data_path_glob is not None
        except AssertionError:
            raise NotImplementedError('Required class attributes have not been specified')

    def _parse_data_path(self, path):
        match = self._re_path.match(path)
        year = int(match.group('year'))
        month = int(match.group('month'))
        day = int(match.group('day'))
        hour = int(match.group('hour'))
        lead_time = int(match.group('lead_time'))
        origin_time = datetime(year=year, month=month, day=day, hour=hour)
        return origin_time, lead_time

    def _print_data_paths(self, n=10):
        items = list(self.data_path_dict.items())
        items = items[n] if n else items
        for dt, paths_by_lead_time in items:
            for lead_time, paths in paths_by_lead_time.items():
                print('{} (lead_time={})'.format(dt, lead_time))
                for p in paths:
                    print('\t' + p)


class GdapsKimBaseDataset(BaseDataset):
    """
    Implementation of BaseDataset corresponding to the data format of GDAPS-KIM.
    """
    dataset_dir = 'GDPS_KIM'
    data_path_glob = os.path.join('**', '*.nc')
    data_path_regex = '.*(?P<year>\d{4})(?P<month>\d{2})/(?P<day>\d{2})/(?P<hour>\d{2})/.*ft(?P<lead_time>\d{3})\.nc'

    pressure_level_dict = {'1000': 0, '975': 1, '950': 2, '925': 3, '900': 4, '875': 5, '850': 6, '800': 7,
                           '750': 8, '700': 9, '650': 10, '600': 11, '550': 12, '500': 13, '450': 14, '400': 15,
                           '350': 16, '300': 17, '250': 18, '200': 19, '150': 20, '100': 21}

    # default_variables = [
    #     'T:850', 'T:700', 'T:500',
    #     'rh_liq:850', 'rh_liq:700', 'rh_liq:500',
    #     'rain', 'q2m', 'rh2m', 't2m', 'tsfc', 'ps'
    # ]
    
    default_variables = 'rain, snow, hpbl, pbltype, q2m, rh2m, t2m, psl, tsfc, topo, u10m, v10m, \
                    T:100, T:150, T:200, T:250, T:300, T:350, T:400, T:450, T:500, T:550, T:600, T:650,T:700, T:750, T:800, T:850, T:875,T:900, T:925,T:950, T:975,T:1000, \
                    rh_liq:100, rh_liq:150, rh_liq:200, rh_liq:250, rh_liq:300, rh_liq:350, rh_liq:400, rh_liq:450, rh_liq:500, rh_liq:550, rh_liq:600, rh_liq:650, rh_liq:700, rh_liq:750, rh_liq:800, rh_liq:850, rh_liq:875, rh_liq:900, rh_liq:925, rh_liq:950, rh_liq:975, rh_liq:1000, \
                    uv:100, uv:150, uv:200, uv:250, uv:300, uv:350, uv:400, uv:450, uv:500, uv:550, uv:600, uv:650,uv:700, uv:750, uv:800, uv:850, uv:875, uv:900, uv:925, uv:950, uv:975, uv:1000, \
                    hgt:100, hgt:150, hgt:200, hgt:250, hgt:300, hgt:350, hgt:400, hgt:450, hgt:500, hgt:550, hgt:600, hgt:650,hgt:700, hgt:750, hgt:800, hgt:850, hgt:875, hgt:900, hgt:925, hgt:950, hgt:975, hgt:1000'


    def __init__(self, root_dir: str, variable_filter: str = None):
        if variable_filter is None:
            variable_filter = ','.join(self.default_variables)
        super().__init__(root_dir, variable_filter)

    def load_array(self, origin: datetime, lead_time: int, return_variables=False):
        feature_maps = []
        variables_used = []
        variables = self.variable_filter.split(',')
        variables = [v.strip() for v in variables]

        # Load dfp, dfs dataset files
        dfp_dataset = self._load_dfp_dataset(origin, lead_time)
        dfs_dataset = self._load_dfs_dataset(origin, lead_time)
        previous_dfs_dataset = None
        if lead_time > 0:  # load previous dfs_dataset to adjust for accumulated variables
            # TODO: optimize redundant file reads
            previous_dfs_dataset = self._load_dfs_dataset(origin, lead_time - 1)

        # Retrieve feature maps based on selected variables
        for v in variables:
            if ':' in v:
                name, pressure = v.split(":")
                pressure_level = self.pressure_level_dict[pressure]
                if name == 'uv':
                    u = dfp_dataset['u'][0][pressure_level]
                    v = dfp_dataset['v'][0][pressure_level]
                    ws = np.sqrt(u * u + v * v)
                    wd = 180 * np.arctan2(v, u) / np.pi
                    wd = np.where(wd >= 0, wd, wd + 360)
                    wd = np.where(wd < 270, 270 - wd, 270 - wd + 360)
                    feature_maps.append(ws)
                    feature_maps.append(wd)
                    variables_used.append('ws:{}'.format(pressure))
                    variables_used.append('wd:{}'.format(pressure))
                elif name in ['u', 'v', 'T', 'rh_liq', 'hgt']:
                    feature_maps.append(dfp_dataset[name][0][pressure_level])
                    variables_used.append(v)
                else:
                    raise Exception('Invalid variable: {}'.format(v))
            else:
                if v == 'rain':
                    if previous_dfs_dataset is None:
                        rainc = dfs_dataset['rainc_acc'][0]
                        rainl = dfs_dataset['rainl_acc'][0]
                    else:
                        rainc = dfs_dataset['rainc_acc'][0] - previous_dfs_dataset['rainc_acc'][0]
                        rainl = dfs_dataset['rainl_acc'][0] - previous_dfs_dataset['rainl_acc'][0]
                    feature_maps.append(rainc + rainl)
                    variables_used.append('rain')
                elif v == 'snow':
                    if previous_dfs_dataset is None:
                        snowc = dfs_dataset['snowc_acc'][0]
                        snowl = dfs_dataset['snowl_acc'][0]
                    else:
                        snowc = dfs_dataset['snowc_acc'][0] - previous_dfs_dataset['snowc_acc'][0]
                        snowl = dfs_dataset['snowl_acc'][0] - previous_dfs_dataset['snowl_acc'][0]
                    feature_maps.append(snowc + snowl)
                    variables_used.append('snow')
                elif v in ['hpbl', 'pbltype', 'psl', 'tsfc', 'topo', 'ps']:
                    feature_maps.append(dfs_dataset[v][0])
                    variables_used.append(v)
                elif v in ['q2m', 'rh2m', 't2m', 'u10m', 'v10m']:
                    feature_maps.append(dfs_dataset[v][0][0])
                    variables_used.append(v)
                else:
                    raise Exception('Invalid variable: {}'.format(v))

        dfp_dataset.close()
        dfs_dataset.close()
        previous_dfs_dataset.close() if previous_dfs_dataset else None

        # Unmask array (parity w/ previous implementation)
        array = np.stack(feature_maps, axis=0)
        _array = array.filled()
        # assert array.mask is False

        if return_variables:
            return _array, variables_used
        else:
            return _array

    def _verify_data_path_dict(self, data_path_dict):
        if len(data_path_dict) == 0:
            raise Exception('No data found (regardless of dates)')

        for (origin_time, lead_time), paths in data_path_dict.items():
            if len(paths) != 2:
                raise Exception('Invalid data paths found (there should be exactly two files): {}'.format(paths))

    def _load_dfp_dataset(self, origin, lead_time) -> nc.Dataset:
        paths = self.data_path_dict[(origin, lead_time)]
        paths = list(filter(lambda p: 'dfp' in p, paths))
        if len(paths) != 1:
            raise AssertionError(
                'Multiple or no dfp filepath for timestamp {}: \n{}'.format((origin, lead_time), paths))
        return nc.Dataset(paths[0])

    def _load_dfs_dataset(self, origin, lead_time) -> nc.Dataset:
        paths = self.data_path_dict[(origin, lead_time)]
        paths = list(filter(lambda p: 'dfs' in p, paths))
        if len(paths) != 1:
            raise AssertionError(
                'Multiple or no dfs filepath for timestamp {}: \n{}'.format((origin, lead_time), paths))
        return nc.Dataset(paths[0])


class AwsBaseDataset(BaseDataset):
    dataset_dir = None  # NWP model must be specified in subclass
    data_path_glob = os.path.join('**', '*.npy')
    data_path_regex = '.*AWS_HOUR_ALL_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})(?P<hour>\d{2})(?P<lead_time>\d{2})_\d*.npy'

    def __init__(self, root_dir: str, variable_filter: str = None):
        if variable_filter is not None:
            print('Warning: `variable_filter` argument is not used in `AWSBaseDataset`')
        if self.dataset_dir is None:
            raise NotImplementedError('You must specify the attribute `dataset_dir`')
        super().__init__(root_dir, variable_filter)

    def load_array(self, origin: datetime, lead_time: int, return_variables=False):
        path = self.data_path_dict[(origin, lead_time)][0]  # only one path per timestamp
        gt = np.load(path).astype(np.float32)
        gt[gt < 0] = -9999.0
        if return_variables:
            return gt, ['observed_rain']
        else:
            return gt

    def _verify_data_path_dict(self, data_path_dict):
        if len(data_path_dict) == 0:
            raise Exception('No data found (regardless of dates)')

        for (origin_time, lead_time), paths in data_path_dict.items():
            if len(paths) != 1:
                raise Exception(
                    'Invalid data paths found (there should be exactly one file for each timestamp): {}'.format(
                        paths))


class AwsBaseDatasetForGdapsKim(AwsBaseDataset):
    dataset_dir = 'AWS_GDPS_KIM_GRID'


def print_dataset_info(dataset: BaseDataset):
    print('Data Paths (subset):')
    print('-' * 80)
    for (origin, lead_time), paths in list(dataset.data_path_dict.items())[:5]:
        print('Timestamp: {} (lead_time={}):'.format(origin, lead_time))
        for path in paths:
            print('\t{}'.format(path))
    print('=' * 80)
    print('Data Format:')
    print('-' * 80)
    # array, variables = dataset.load_array(datetime(2020, 7, 1, 0), 6, return_variables=True)
    # 尝试把原来的0改为2，要去改y的load_array方式
    array, variables = dataset.load_array(datetime(2020, 7, 1, 0), 0, return_variables=True)
    print('Array shape: {}'.format(array.shape))
    print('Variables used: {}'.format(variables))
    print('=' * 80)


def main():
    root_dir = '/mnt/ssd1/yujint/KoMet/nims'
    # variable_filter = 'rain, hpbl, pbltype, psl, tsfc, topo, q2m, T:850, T:700, T:500, uv:850, uv:700, uv:500'
    variable_filter = 'rain, snow, hpbl, pbltype, q2m, rh2m, t2m, psl, tsfc, topo, u10m, v10m, \
                        T:100, T:150, T:200, T:250, T:300, T:350, T:400, T:450, T:500, T:550, T:600, T:650,T:700, T:750, T:800, T:850, T:875,T:900, T:925,T:950, T:975,T:1000, \
                        rh_liq:100, rh_liq:150, rh_liq:200, rh_liq:250, rh_liq:300, rh_liq:350, rh_liq:400, rh_liq:450, rh_liq:500, rh_liq:550, rh_liq:600, rh_liq:650, rh_liq:700, rh_liq:750, rh_liq:800, rh_liq:850, rh_liq:875, rh_liq:900, rh_liq:925, rh_liq:950, rh_liq:975, rh_liq:1000, \
                        uv:100, uv:150, uv:200, uv:250, uv:300, uv:350, uv:400, uv:450, uv:500, uv:550, uv:600, uv:650,uv:700, uv:750, uv:800, uv:850, uv:875, uv:900, uv:925, uv:950, uv:975, uv:1000, \
                        hgt:100, hgt:150, hgt:200, hgt:250, hgt:300, hgt:350, hgt:400, hgt:450, hgt:500, hgt:550, hgt:600, hgt:650,hgt:700, hgt:750, hgt:800, hgt:850, hgt:875, hgt:900, hgt:925, hgt:950, hgt:975, hgt:1000'
    # pressure_level_dict = {'1000': 0, '975': 1, '950': 2, '925': 3, '900': 4, '875': 5, '850': 6, '800': 7,
    #                    '750': 8, '700': 9, '650': 10, '600': 11, '550': 12, '500': 13, '450': 14, '400': 15,
    #                    '350': 16, '300': 17, '250': 18, '200': 19, '150': 20, '100': 21}
    dataset = GdapsKimBaseDataset(root_dir, variable_filter)
    print('GdapsKimBaseDataset Test'.center(40).center(80, '='))
    print_dataset_info(dataset)

    print()
    print()
    print()
    print()
    print()

    root_dir = '/mnt/ssd1/yujint/KoMet/nims'
    dataset = AwsBaseDatasetForGdapsKim(root_dir)
    print('AwsBaseDatasetForGdapsKim Test'.center(40).center(80, '='))
    print_dataset_info(dataset)


if __name__ == '__main__':
    main()
