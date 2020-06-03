import json
from importlib import import_module
import numpy as np
import torch
import torch.nn.functional as F
import logging
import os


def check_mandatory_keys(dictionary, mandatory_keys, name):
    for key in mandatory_keys:
        if key not in dictionary.keys():
            raise KeyError(f'Mandatory key {key} not in dictionary keys {dictionary.keys()} from {name}')


def set_dict_value(dictionary, key, default_value=None):
    value = dictionary.get(key) or default_value
    dictionary[key] = value


def import_object(module, name, package=''):
    mod = import_module(module, package)
    return getattr(mod, name)


def parse_function_import(function_dict):
    function_name = function_dict.get('name')
    module = function_dict.get('module')
    package = function_dict.get('package') or ''

    return import_object(module, function_name, package)


def parse_object_import(object_dict):
    attributes = object_dict.get('attributes') or {}
    object_class = parse_function_import(object_dict)
    return object_class(**attributes), object_class


def generate_json_document(filename, **kwargs):
    with open(filename, 'w') as file:
        json.dump(kwargs, file)


def to_var(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.to(device)
    return x


def to_numpy(x):
    if not (isinstance(x, np.ndarray) or x is None):
        if hasattr(x, 'cuda') and x.is_cuda:
            x = x.data.cpu()
        if hasattr(x, 'numpy'):
            x = x.numpy()
        else:
            x = np.array(x)
    return x


def summary(epoch, i, nb_batch, loss, batch_time, average_loss, average_time, mode, granularity='Batch'):
    string = f'[{str(mode)}] Epoch: [{epoch}][{i}/{nb_batch}]\t'

    string += f'{granularity} Loss {loss:.4f} '
    string += f'(Average {average_loss:.4f}) \t'
    string += f'{granularity} Time {batch_time:.4f} '
    string += f'(Average {average_time:.4f}) \t'

    return string


def instantiate_logger(logger_name, log_level, log_filename):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    file_handler = logging.FileHandler(log_filename)
    console_handler = logging.StreamHandler()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_checkpoint(state, save_path, custom_save=False, model=None):
    """
    Save the current model.
    """

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state.get('epoch')
    val_loss = state.get('val_loss')
    iterations = state.get('iterations')

    filename = f'{save_path}/model_ep{epoch}'
    if iterations is not None:
        filename += f'_it{iterations}'
    if val_loss is not None:
        filename += f'_loss{val_loss:0.4f}'
    filename += '.pth.tar'

    if custom_save:
        model.save(filename)
    else:
        torch.save(state, filename)


def mean_metric(prediction, target, metric):
    if target.shape[1] == 1:
        return mean_metric_1d(prediction, target, metric)
    else:
        return mean_metric_nd(prediction, target, metric)


def mean_metric_1d(prediction, target, metric):
    prediction = torch.sigmoid(prediction)
    return metric(prediction[:, 0, ...], target[:, 0, ...])


def mean_metric_nd(prediction, target, metric):
    prediction = F.softmax(prediction, dim=1)
    channels = list(range(target.shape[1]))
    res = 0
    for channel in channels:
        res += metric(prediction[:, channel, ...], target[:, channel, ...])

    return res / len(channels)
