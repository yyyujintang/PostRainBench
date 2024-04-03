from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from data.dataset import StandardDataset
from torch.utils.data import Subset


def compile_metrics(dataset: StandardDataset, predictions: np.ndarray, timestamps: List,
                    thresholds: List[float]) -> Tuple[np.ndarray, Dict[float, pd.DataFrame]]:
    """
    :param dataset:
    :param predictions: shape=(batch_size, height, width), dtype=int
    :param timestamps:
    :param thresholds:
    :return: confusion_matrix, binary_metrics_by_threshold

    confusion: (prediction, target)

    pred\target |   false  |   true
    false       |   cn     |   miss
    true        |   fa     |   hit
    """
    assert (predictions.shape[0] == len(timestamps))
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    if isinstance(predictions, torch.Tensor):
        raise ValueError("`predictions` must be a numpy array")

    n_classes = len(thresholds) + 1
    padded_thresholds = [float('-inf')] + thresholds + [float('inf')]
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    binary_metrics_by_threshold = defaultdict(lambda: defaultdict(list))
    for (origin, lead_time), prediction in zip(timestamps, predictions):
        local_confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        target = dataset.aws_base_dataset.load_array(origin + timedelta(hours=lead_time), 0)
        valid = target != -9999

        # Populate local confusion matrix
        interval_list = zip(padded_thresholds[:-1], padded_thresholds[1:])
        preds_by_interval = []
        targets_by_interval = []
        for i, (start, end) in enumerate(interval_list):
            preds_by_interval.append(prediction == i)
            targets_by_interval.append((start <= target) & (target < end))
        for i, p in enumerate(preds_by_interval):
            for j, t in enumerate(targets_by_interval):
                local_confusion_matrix[i, j] = (valid & p & t).sum()

        # Populate binary metric dict[list]s
        for i, threshold in enumerate(thresholds, start=1):
            metrics = binary_metrics_by_threshold[threshold]
            metrics["origin"].append(origin.strftime("%Y%m%d_%H%M"))
            metrics["lead_time"].append(lead_time)
            metrics["hit"].append(local_confusion_matrix[i:, i:].sum())
            metrics["miss"].append(local_confusion_matrix[:i, i:].sum())
            metrics["fa"].append(local_confusion_matrix[i:, :i].sum())
            metrics["cn"].append(local_confusion_matrix[:i, :i].sum())

        confusion_matrix += local_confusion_matrix

    binary_metrics_by_threshold = {t: pd.DataFrame(metrics) for t, metrics in binary_metrics_by_threshold.items()}
    return confusion_matrix, binary_metrics_by_threshold


def compile_metrics_Germany(dataset: StandardDataset, predictions: np.ndarray, targets: np.ndarray,
                    thresholds: List[float]) -> Tuple[np.ndarray, Dict[float, pd.DataFrame]]:
    """
    :param dataset:
    :param predictions: shape=(batch_size, height, width), dtype=int
    :param timestamps:
    :param thresholds:
    :return: confusion_matrix, binary_metrics_by_threshold

    confusion: (prediction, target)

    pred\target |   false  |   true
    false       |   cn     |   miss
    true        |   fa     |   hit
    """
    # assert (predictions.shape[0] == len(timestamps))
    # if isinstance(dataset, Subset):
    #     dataset = dataset.dataset
    # if isinstance(predictions, torch.Tensor):
    #     raise ValueError("`predictions` must be a numpy array")

    n_classes = len(thresholds) + 1
    padded_thresholds = [float('-inf')] + thresholds + [float('inf')]
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    binary_metrics_by_threshold = defaultdict(lambda: defaultdict(list))
    for target, prediction in zip(targets, predictions):
        local_confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        # target = dataset.aws_base_dataset.load_array(origin + timedelta(hours=lead_time), 0)
        valid = target != -9999

        # Populate local confusion matrix
        interval_list = zip(padded_thresholds[:-1], padded_thresholds[1:])
        preds_by_interval = []
        targets_by_interval = []
        for i, (start, end) in enumerate(interval_list):
            preds_by_interval.append(prediction == i)
            targets_by_interval.append((start <= target) & (target < end))
        for i, p in enumerate(preds_by_interval):
            for j, t in enumerate(targets_by_interval):
                local_confusion_matrix[i, j] = (valid & p & t).sum()

        # Populate binary metric dict[list]s
        for i, threshold in enumerate(thresholds, start=1):
            metrics = binary_metrics_by_threshold[threshold]
            metrics["hit"].append(local_confusion_matrix[i:, i:].sum())
            metrics["miss"].append(local_confusion_matrix[:i, i:].sum())
            metrics["fa"].append(local_confusion_matrix[i:, :i].sum())
            metrics["cn"].append(local_confusion_matrix[:i, :i].sum())

    confusion_matrix += local_confusion_matrix

    binary_metrics_by_threshold = {t: pd.DataFrame(metrics) for t, metrics in binary_metrics_by_threshold.items()}
    return confusion_matrix, binary_metrics_by_threshold


def compute_evaluation_metrics(df: pd.DataFrame):
    """
    Compute evaluation metrics in-place for dataframe containing binary confusion matrix.
    """
    df['acc'] = (df['hit'] + df['cn']) / (df['hit'] + df['miss'] + df['fa'] + df['cn'])  # accuracy
    df['pod'] = df['hit'] / (df['hit'] + df['miss'])  # probability of detection
    df['far'] = df['fa'] / (df['hit'] + df['fa'])  # false alarm ratio
    df['csi'] = df['hit'] / (df['hit'] + df['miss'] + df['fa'])  # critical success index
    df['bias'] = (df['hit'] + df['fa']) / (df['hit'] + df['miss'])  # bias
    df['hss'] = 2* (df['hit'] * df['cn']-df['miss']*df['fa']) / (df['miss']**2 + df['fa']**2 + 2*df['hit']*df['cn']+(df['miss']+df['fa'])*(df['hit']+df['cn']))  
    
    return df
