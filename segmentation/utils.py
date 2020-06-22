import commentjson as json
from importlib import import_module
import numpy as np
import torch
import torch.nn.functional as F
import logging
import sys
import os


def import_object(module, name, package=''):
    """
    Import an object from a given module.
    """
    mod = import_module(module, package)
    return getattr(mod, name)


def parse_function_import(function_dict):
    """
    Import a function from a dictionary that specifies where to find it.
    """
    function_name = function_dict.get('name')
    module = function_dict.get('module')
    package = function_dict.get('package') or ''

    return import_object(module, function_name, package)


def parse_object_import(object_dict):
    """
    Import a class and instantiate it from a dictionary that specifies where to find it.
    """
    attributes = object_dict.get('attributes') or {}
    object_class = parse_function_import(object_dict)
    return object_class(**attributes), object_class


def parse_method_import(method_dict):
    """
    Import a method from a class instance using a dictionary that specifies where to find it.
    """
    object_instance, _ = parse_object_import(method_dict)
    return getattr(object_instance, method_dict['method'])


def generate_json_document(filename, **kwargs):
    """
    Generate a json file from a dictionary.
    """
    with open(filename, 'w') as file:
        json.dump(kwargs, file, indent=4, sort_keys=True)


def to_var(x, device):
    """
    Applied to a NumPy array or a Torch tensor, it returns a Torch tensor on the given device.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.to(device)
    return x


def to_numpy(x):
    """
    Applied to a NumPy array or a Torch tensor, it returns a NumPy array.
    """
    if not (isinstance(x, np.ndarray) or x is None):
        if x.requires_grad:
            x = x.detach()
        if hasattr(x, 'cuda') and x.is_cuda:
            x = x.data.cpu()
        if hasattr(x, 'numpy'):
            x = x.numpy()
        else:
            x = np.array(x)
    return x


def summary(epoch, i, nb_batch, loss, batch_time, average_loss, average_time, mode, granularity='Batch'):
    """
    Generate a summary of the model performances on a batch.
    """
    string = f'[{str(mode)}] Epoch: [{epoch}][{i}/{nb_batch}]\t'

    if isinstance(loss, str):
        string += f'{granularity} Loss {loss} '
        string += f'(Average {average_loss}) \t'
    else:
        string += f'{granularity} Loss {loss:.4f} '
        string += f'(Average {average_loss:.4f}) \t'
    string += f'{granularity} Time {batch_time:.4f} '
    string += f'(Average {average_time:.4f}) \t'

    return string


def instantiate_logger(logger_name, log_level, log_filename, console=True):
    """
    Create a logger that will both write to console and to a log file.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)-2s: %(levelname)-2s : %(message)s")

    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(console_handler)

    return logger


def save_checkpoint(state, save_path, model):
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

    if hasattr(model, 'save'):
        model.save(filename)
    else:
        torch.save(state, filename)


def channel_metrics(prediction, target, metric):
    """
    Compute a given metric on every channel of the volumes.
    """
    if target.shape[1] == 1:
        target = torch.cat([target, 1 - target], dim=1)
    prediction = F.softmax(prediction, dim=1)
    channels = list(range(target.shape[1]))
    res = []
    for channel in channels:
        res.append(metric(prediction[:, channel, ...], target[:, channel, ...]))

    return torch.tensor(res)


def mean_metric(prediction, target, metric):
    """
    Compute a given metric on every channel of the volumes and average them.
    """
    if target.shape[1] == 1:
        target = torch.cat([target, 1 - target], dim=1)
    prediction = F.softmax(prediction, dim=1)
    channels = list(range(target.shape[1]))
    res = 0
    for channel in channels:
        res += metric(prediction[:, channel, ...], target[:, channel, ...])

    return res / len(channels)


def get_class_name_from_method(method):
    cls = method.__self__.__class__
    # str(cls) gives something like "<class 'the_class_name'>"
    cls_name = str(cls)[8:-2]
    return cls_name.split('.')[-1]
