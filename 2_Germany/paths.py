"""
Provides methods on
"""
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')


def get_model_tag(epoch: int = None, step: int = None, required=False):
    """
    Get tag for model trained until a specific epoch OR step
    :param epoch:
    :param step:
    :param required: Requires either epoch or step to be specified
    :return:
    """
    if epoch and step:
        raise ValueError('Cannot specify both epoch and step to get prediction path')
    if epoch:
        return 'epoch{:03d}'.format(epoch)
    elif step:
        return 'step{:08d}'.format(step)
    elif required:
        raise ValueError('Either epoch or step must be specified')
    return None


def get_trained_model_path(experiment: str, epoch: int = None, step: int = None, makedirs=False):
    # output/<exp_name>/trained_model/{epoch001,step00000001}.pt
    model_tag = get_model_tag(epoch, step, required=True)
    path = os.path.join(OUTPUT_DIR, experiment, 'trained_model', '{}.pt'.format(model_tag))
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_train_log_dir(experiment: str, makedirs=False):
    # output/<exp_name>/train_log/
    path = os.path.join(OUTPUT_DIR, experiment, 'train_log')
    if makedirs:
        os.makedirs(path, exist_ok=True)
    return path

def get_tensorboard_log_dir(experiment: str, makedirs=False):
    # output/<exp_name>/train_log/
    path = os.path.join(OUTPUT_DIR, experiment, 'tensorboard_log')
    if makedirs:
        os.makedirs(path, exist_ok=True)
    return path

def get_train_log_binary_metrics_path(experiment: str, epoch: int, threshold: float, makedirs=False):
    # output/<exp_name>/train_log/binary_metrics{,_heavy,_000.00}/{epoch001,step00000001}.csv
    if threshold == 0.1:
        threshold_suffix = ""
    elif threshold == 10.0:
        threshold_suffix = "_heavy"
    else:
        threshold_suffix = "_{:06.2f}".format(threshold)

    train_log_dir = get_train_log_dir(experiment, makedirs=makedirs)
    model_tag = get_model_tag(epoch, step=None, required=True)
    path = os.path.join(train_log_dir, 'binary_metrics' + threshold_suffix, '{}.csv'.format(model_tag))
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_train_log_confusion_path(experiment: str, epoch: int, makedirs=False):
    # output/<exp_name>/train_log/confusion/{epoch001,step00000001}.np
    train_log_dir = get_train_log_dir(experiment, makedirs=makedirs)
    model_tag = get_model_tag(epoch, step=None, required=True)
    path = os.path.join(train_log_dir, 'confusion', '{}.np'.format(model_tag))
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_train_log_history_path(experiment: str, makedirs=False):
    # output/<exp_name>/train_log/history.csv
    train_log_dir = get_train_log_dir(experiment, makedirs=makedirs)
    path = os.path.join(train_log_dir, 'history.csv')
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_train_log_step_history_path(experiment: str, makedirs=False):
    # output/<exp_name>/train_log/step_history.csv
    train_log_dir = get_train_log_dir(experiment, makedirs=makedirs)
    path = os.path.join(train_log_dir, 'step_history.csv')
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_train_log_json_path(experiment: str, makedirs=False):
    # output/<exp_name>/train_log/log.json
    train_log_dir = get_train_log_dir(experiment, makedirs=makedirs)
    path = os.path.join(train_log_dir, 'log.json')
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_train_log_txt_path(experiment: str, makedirs=False):
    # output/<exp_name>/train_log/log.txt
    train_log_dir = get_train_log_dir(experiment, makedirs=makedirs)
    path = os.path.join(train_log_dir, 'log.txt')
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_prediction_path(experiment: str, test_class: str = None, epoch: int = None, step: int = None,
                        origin_time: datetime = None, makedirs=False):
    # output/<exp_name>/prediction(/<test_class>)(/{epoch001,step00000001})(/AIW_xxx.nc)
    model_tag = get_model_tag(epoch, step)
    path = os.path.join(OUTPUT_DIR, experiment, 'prediction')
    dirname = path

    if test_class:
        path = os.path.join(path, test_class)
        dirname = path
    if test_class and model_tag:
        path = os.path.join(path, model_tag)
        dirname = path
    if test_class and model_tag and origin_time:
        basename = 'AIW_UNET_PRCP01_KR_V01_{ymd}_{utc:02d}0000.nc'.format(ymd=origin_time.strftime('%Y%m%d'),
                                                                          utc=origin_time.hour)
        path = os.path.join(path, basename)

    if makedirs:
        os.makedirs(dirname, exist_ok=True)

    return path


def get_evaluation_result_dir(experiment: str, epoch: int = None, step: int = None, makedirs=False,
                              require_model_tag=False):
    # output/<exp_name>/evaluation_result(/{epoch001,step00000001})
    model_tag = get_model_tag(epoch, step, required=require_model_tag)
    path = os.path.join(OUTPUT_DIR, experiment, 'evaluation_result')
    if model_tag is not None:
        path = os.path.join(path, model_tag)
    if makedirs:
        os.makedirs(path, exist_ok=True)
    return path


def get_evaluation_summary_path(experiment: str, epoch: int = None, step: int = None, makedirs=False):
    # output/<exp_name>/evaluation_result/{epoch001,step00000001}/summary.csv
    evaluation_result_dir = get_evaluation_result_dir(experiment, epoch, step, makedirs=makedirs,
                                                      require_model_tag=True)
    path = os.path.join(evaluation_result_dir, 'summary.csv')
    return path


"""
Output paths v3.0 (as of 5/9/2022)

output/<exp>/
|–– epoch001/
    |–– train_binary_metrics_rain.csv
    |–– train_binary_metrics_heavy.csv
    |–– train_binary_metrics_###.##.csv
    |–– train_confusion.npy
    |–– train_summary.txt
    |–– val_*
    |–– test_*
    |–– test_2022_*
    |–– <subset_name>_*
    |–– trained_model.pt  # not implementing yet
|–– epoch###/
"""


def get_epoch_directory(exp: str, epoch: int = None):
    return os.path.join(OUTPUT_DIR, exp, "epoch{:03d}".format(epoch))


def get_binary_metrics_path(exp: str, epoch: int, subset="train", threshold=0.1):
    if threshold == 0.1:
        t = "rain"
    elif threshold == 10.0:
        t = "heavy"
    else:
        t = "{:6.2f}".format(threshold)

    path = get_epoch_directory(exp, epoch)
    path = os.path.join(path, "{}_binary_metrics_{}.csv".format(subset, t))
    return path


def get_confusion_matrix_path(exp: str, epoch: int, subset="train"):
    path = get_epoch_directory(exp, epoch)
    path = os.path.join(path, "{}_confusion.npy".format(subset))
    return path


def get_summary_path(exp: str, epoch: int, subset="train"):
    path = get_epoch_directory(exp, epoch)
    path = os.path.join(path, "{}_summary.txt".format(subset))
    return path
