import json
from importlib import import_module
import numpy as np
import torch
import logging
import os


def import_object(module, name, package=''):
    mod = import_module(module, package)
    return getattr(mod, name)


def parse_object_import(object_dict):
    object_name = object_dict.get('name')
    module = object_dict.get('module')
    package = object_dict.get('package') or ''
    attributes = object_dict.get('attributes') or {}

    object_class = import_object(module, object_name, package)
    return object_class(**attributes), object_class


def parse_function_import(function_dict):
    function_name = function_dict.get('name')
    module = function_dict.get('module')
    package = function_dict.get('package') or ''

    return import_object(module, function_name, package)


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
        if x.is_cuda:
            x = x.data.cpu()
        x = x.numpy()
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

    epoch = state['epoch']
    val_loss = state['val_loss']
    filename = save_path + '/' + \
        'model.{:02d}--{:.3f}.pth.tar'.format(epoch, val_loss)
    if custom_save:
        model.save(filename)
    else:
        torch.save(state, filename)
