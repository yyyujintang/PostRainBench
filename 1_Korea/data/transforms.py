"""
Custom transforms that are used in our framework.
"""
__all__ = ["InterpolateAWS", "ToTensor", "ClassifyByThresholds", "UndersampleDry", "UndersampleGlobal"]


import numpy as np
import torch
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
from torch.multiprocessing import set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class InterpolateAWS:
    def __call__(self, aws_grid: torch.Tensor) -> torch.Tensor:
        """
        :param aws_grid: AWS observation feature map in W x H format.
        :return:
        """
        # Since csr_matrix considers 0 as null, we temporarily set -9999 (null) to 0, and add 1 to the values.
        aws_grid[aws_grid == -9999] = -1
        aws_grid += 1
        sparse = csr_matrix(aws_grid)  # hotfix, since csr_matrix considers 0 as null

        w, h = aws_grid.shape
        grid_x, grid_y = np.mgrid[0:w:1, 0:h:1]

        rows, cols = sparse.nonzero()
        rows, cols = rows.tolist(), cols.tolist()
        # points: coordinates where AWS observations are valid in `(x, y)` format
        # values: values corresponding to the coordinates above
        points = np.array(list(zip(rows, cols)))
        values = sparse.data

        interpolated = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=-9998, rescale=False)
        interpolated = interpolated - 1  # shift back to -9999, [0, inf)

        # # plotting
        # from PIL import Image
        # interpolated2 = np.flipud(interpolated)
        # pyplot.imshow(interpolated2,interpolation='none')
        # pyplot.savefig('spline.png')

        return torch.Tensor(interpolated)


class ClassifyByThresholds:
    """
    Map AWS observation values to one-hot vectors according to `thresholds`.
    """

    def __init__(self, thresholds):
        self.thresholds = thresholds

    def __call__(self, grid: torch.Tensor) -> torch.Tensor:
        """
        :param grid: AWS observations in W x H format. Null values should be -9999.
        """
        thresholds = [0.0] + self.thresholds + [float('inf')]
        # print("thresholds",thresholds)
        result = grid
        for i, start in enumerate(thresholds[:-1]):
            end = thresholds[i + 1]
            result = torch.where((start <= grid) & (grid < end), i * torch.ones(grid.shape), result)
        # print("result",result)
        return result

    
class UndersampleDry:
    """
    First sampling strategy in KoMet paper. Undersample points that have class label of 0 with given `sampling_rate`.
    Unused values are set to -9999 (null).
    """

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def __call__(self, label: torch.Tensor) -> torch.Tensor:
        """
        :param grid: AWS observations in W x H format. Null values should be -9999.
        """
        label[(label == 0) & (torch.rand_like(label) >= self.sampling_rate)] = -9999
        return label


class UndersampleGlobal:
    """
    Undersample points with given `sampling_rate`. Unused values are set to -9999 (null).
    """

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def __call__(self, label: torch.Tensor) -> torch.Tensor:
        """
        :param grid: AWS observations in W x H format. Null values should be -9999.
        """
        label[torch.rand_like(label) >= self.sampling_rate] = -9999
        return label


class ToTensor:
    def __call__(self, images):
        return torch.from_numpy(images)
