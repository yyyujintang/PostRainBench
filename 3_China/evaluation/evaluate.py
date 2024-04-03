"""
Evaluation module. Refer to `notebooks/evaluation_example.ipynb` on examples.
"""

from collections import defaultdict
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset
from tqdm import tqdm

from data.dataset import StandardDataset
from evaluation.metrics import compile_metrics,compile_metrics_Germany
from einops import rearrange


def evaluate_model(model: nn.Module, data_loader, thresholds, criterion, device,
                   normalization=None) -> Tuple[float, float, np.ndarray, Dict[float, pd.DataFrame]]:
    """
    :param model:
    :param data_loader:
    :param thresholds:
    :param criterion:
    :param device:
    :param normalization:
    :return: confusion, binary_metrics_by_threshold
    """
    dataset = data_loader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    if not isinstance(dataset, StandardDataset):
        raise ValueError('`data_loader` must contain a (subset of) StandardDataset')

    n_thresholds = len(thresholds)
    n_classes = n_thresholds + 1
    total_loss = 0
    total_samples = 0

    model.eval()

    # Run inference on single epoch
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    metrics_by_threshold = defaultdict(list)
    for i, (images, target, t) in enumerate(tqdm(data_loader)):
        # Note, StandardDataset retrieves timestamps in Tensor format due to collation issue, for now
        timestamps = []
        for e in t:
            origin = datetime(year=e[0], month=e[1], day=e[2], hour=e[3])
            lead_time = e[4].item()
            timestamps.append((origin, lead_time))

        if normalization:
            with torch.no_grad():
                for i, (max_val, min_val) in enumerate(zip(normalization['max_values'], normalization['min_values'])):
                    if min_val < 0:
                        images[:, :, i, :, :] = images[:, :, i, :, :] / max(-min_val, max_val)
                    else:
                        images[:, :, i, :, :] = (images[:, :, i, :, :] - min_val) / (max_val - min_val)

        images = images.float().to(device)
        target = target.long().to(device)
        output = model(images, t)
        loss, _, _ = criterion(output, target, timestamps, mode="train")
        if loss is None:  # hotfix for None return values from losses.CrossEntropyLoss
            continue

        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]

        predictions = output.detach().cpu().topk(1, dim=1, largest=True, sorted=True)[1]  # (batch_size, height, width)
        predictions = predictions.numpy()
        step_confusion, step_metrics_by_threshold = compile_metrics(data_loader.dataset, predictions, timestamps,
                                                                    thresholds)
        confusion += step_confusion
        for threshold, metrics in step_metrics_by_threshold.items():
            metrics_by_threshold[threshold].append(metrics)

    metrics_by_threshold = {t: pd.concat(metrics) for t, metrics in metrics_by_threshold.items()}
    correct = (confusion[np.diag_indices_from(confusion)]).sum()
    accuracy = correct / confusion.sum()
    loss = total_loss / total_samples

    return accuracy, loss, confusion, metrics_by_threshold

def evaluate_model_Germany(model: nn.Module, data_loader, thresholds, criterion, device,
                   normalization=None) -> Tuple[float, float, np.ndarray, Dict[float, pd.DataFrame]]:
    """
    :param model:
    :param data_loader:
    :param thresholds:
    :param criterion:
    :param device:
    :param normalization:
    :return: confusion, binary_metrics_by_threshold
    """
    # dataset = data_loader.dataset
    # if isinstance(dataset, Subset):
    #     dataset = dataset.dataset
    # if not isinstance(dataset, StandardDataset):
    #     raise ValueError('`data_loader` must contain a (subset of) StandardDataset')

    n_thresholds = len(thresholds)
    n_classes = n_thresholds + 1
    total_loss = 0
    total_samples = 0

    model.eval()

    # Run inference on single epoch
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    metrics_by_threshold = defaultdict(list)
    for i, (images, target, t) in enumerate(tqdm(data_loader)):
        # Note, StandardDataset retrieves timestamps in Tensor format due to collation issue, for now
        # timestamps = []
        # for e in t:
        #     origin = datetime(year=e[0], month=e[1], day=e[2], hour=e[3])
        #     lead_time = e[4].item()
        #     timestamps.append((origin, lead_time))

        if normalization:
            with torch.no_grad():
                for i, (max_val, min_val) in enumerate(zip(normalization['max_values'], normalization['min_values'])):
                    if min_val < 0:
                        images[:, :, i, :, :] = images[:, :, i, :, :] / max(-min_val, max_val)
                    else:
                        images[:, :, i, :, :] = (images[:, :, i, :, :] - min_val) / (max_val - min_val)

        images = images.float().to(device)
        target = target.long().to(device)
        output = model(images, t)
        timestamps = None
        loss, _, _ = criterion(output, target, timestamps, mode="train")
        if loss is None:  # hotfix for None return values from losses.CrossEntropyLoss
            continue

        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]

        predictions = output.detach().cpu().topk(1, dim=1, largest=True, sorted=True)[1]  # (batch_size, height, width)
        predictions = predictions.numpy()
        step_confusion, step_metrics_by_threshold = compile_metrics_Germany(data_loader.dataset, predictions, target.detach().cpu().numpy(), thresholds)
        confusion += step_confusion
        for threshold, metrics in step_metrics_by_threshold.items():
            metrics_by_threshold[threshold].append(metrics)

    metrics_by_threshold = {t: pd.concat(metrics) for t, metrics in metrics_by_threshold.items()}
    correct = (confusion[np.diag_indices_from(confusion)]).sum()
    accuracy = correct / confusion.sum()
    loss = total_loss / total_samples

    return accuracy, loss, confusion, metrics_by_threshold

#     return accuracy, loss, confusion, metrics_by_threshold

def evaluate_model_Germany_Two(model: nn.Module, data_loader, thresholds, criterion, device,
                   normalization=None) -> Tuple[float, float, np.ndarray, Dict[float, pd.DataFrame]]:
    """
    :param model:
    :param data_loader:
    :param thresholds:
    :param criterion:
    :param device:
    :param normalization:
    :return: confusion, binary_metrics_by_threshold
    """
    # dataset = data_loader.dataset
    # if isinstance(dataset, Subset):
    #     dataset = dataset.dataset
    # if not isinstance(dataset, StandardDataset):
    #     raise ValueError('`data_loader` must contain a (subset of) StandardDataset')

    n_thresholds = len(thresholds)
    n_classes = n_thresholds + 1
    total_loss = 0
    total_samples = 0

    model.eval()

    # Run inference on single epoch
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    metrics_by_threshold = defaultdict(list)
    for i, (images, target, t) in enumerate(tqdm(data_loader)):
        # Note, StandardDataset retrieves timestamps in Tensor format due to collation issue, for now
        # timestamps = []
        # for e in t:
        #     origin = datetime(year=e[0], month=e[1], day=e[2], hour=e[3])
        #     lead_time = e[4].item()
        #     timestamps.append((origin, lead_time))

        if normalization:
            with torch.no_grad():
                for i, (max_val, min_val) in enumerate(zip(normalization['max_values'], normalization['min_values'])):
                    if min_val < 0:
                        images[:, :, i, :, :] = images[:, :, i, :, :] / max(-min_val, max_val)
                    else:
                        images[:, :, i, :, :] = (images[:, :, i, :, :] - min_val) / (max_val - min_val)

        images = images.float().to(device)
        target = target.long().to(device)
        output,output2 = model(images, t)
        timestamps = None
        loss, _, _ = criterion(output, output2, target, timestamps, mode="train")
        if loss is None:  # hotfix for None return values from losses.CrossEntropyLoss
            continue

        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]

        predictions = output.detach().cpu().topk(1, dim=1, largest=True, sorted=True)[1]  # (batch_size, height, width)
        step_confusion, step_metrics_by_threshold = compile_metrics_Germany(data_loader.dataset, predictions.numpy(),
                                                                        target.detach().cpu().numpy(), thresholds)
        confusion += step_confusion
        for threshold, metrics in step_metrics_by_threshold.items():
            metrics_by_threshold[threshold].append(metrics)

    metrics_by_threshold = {t: pd.concat(metrics) for t, metrics in metrics_by_threshold.items()}
    correct = (confusion[np.diag_indices_from(confusion)]).sum()
    accuracy = correct / confusion.sum()
    loss = total_loss / total_samples

    return accuracy, loss, confusion, metrics_by_threshold

def evaluate_model_Two(model: nn.Module, data_loader, thresholds, criterion, device,
                   normalization=None) -> Tuple[float, float, np.ndarray, Dict[float, pd.DataFrame]]:
    """
    :param model:
    :param data_loader:
    :param thresholds:
    :param criterion:
    :param device:
    :param normalization:
    :return: confusion, binary_metrics_by_threshold
    """
    dataset = data_loader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    if not isinstance(dataset, StandardDataset):
        raise ValueError('`data_loader` must contain a (subset of) StandardDataset')

    n_thresholds = len(thresholds)
    n_classes = n_thresholds + 1
    total_loss = 0
    total_samples = 0

    model.eval()

    # Run inference on single epoch
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    metrics_by_threshold = defaultdict(list)
    for i, (images, target, t) in enumerate(tqdm(data_loader)):
        # Note, StandardDataset retrieves timestamps in Tensor format due to collation issue, for now
        timestamps = []
        for e in t:
            origin = datetime(year=e[0], month=e[1], day=e[2], hour=e[3])
            lead_time = e[4].item()
            timestamps.append((origin, lead_time))

        if normalization:
            with torch.no_grad():
                for i, (max_val, min_val) in enumerate(zip(normalization['max_values'], normalization['min_values'])):
                    if min_val < 0:
                        images[:, :, i, :, :] = images[:, :, i, :, :] / max(-min_val, max_val)
                    else:
                        images[:, :, i, :, :] = (images[:, :, i, :, :] - min_val) / (max_val - min_val)

        images = images.float().to(device)
        target = target.long().to(device)
        output,output2 = model(images, t)
        loss, _, _ = criterion(output, output2,  target, timestamps, mode="train")
        # loss, _, _ = criterion(output, output2, target, timestamps, mode="train")
        if loss is None:  # hotfix for None return values from losses.CrossEntropyLoss
            continue

        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]        
        predictions = output.detach().cpu().topk(1, dim=1, largest=True, sorted=True)[1]  # (batch_size, height, width)
        predictions = predictions.numpy()
        step_confusion, step_metrics_by_threshold = compile_metrics(data_loader.dataset, predictions, timestamps,
                                                                    thresholds)
        confusion += step_confusion
        for threshold, metrics in step_metrics_by_threshold.items():
            metrics_by_threshold[threshold].append(metrics)

    metrics_by_threshold = {t: pd.concat(metrics) for t, metrics in metrics_by_threshold.items()}
    correct = (confusion[np.diag_indices_from(confusion)]).sum()
    accuracy = correct / confusion.sum()
    loss = total_loss / total_samples

    return accuracy, loss, confusion, metrics_by_threshold
