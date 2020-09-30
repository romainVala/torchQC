import commentjson as json
from importlib import import_module
import numpy as np
import torch
import logging
import sys
import os
from torchio import SubjectsDataset


def identity_activation(x):
    return x


def custom_import(object_dict):
    """
    Import an object from a dictionary that specifies where to find it.
    """
    name = object_dict.get('name')
    module = object_dict.get('module')
    package = object_dict.get('package') or ''

    mod = import_module(module, package)
    return getattr(mod, name)


def parse_function_import(function_dict):
    """
    Import a function from a dictionary that specifies where to find it.
    """
    func = custom_import(function_dict)
    attributes = function_dict.get('attributes') or {}

    return lambda *args: func(*args, **attributes)


def parse_object_import(object_dict):
    """
    Import a class and instantiate it from a dictionary that specifies
    where to find it.
    """
    attributes = object_dict.get('attributes') or {}
    object_class = custom_import(object_dict)
    return object_class(**attributes), object_class


def parse_method_import(method_dict):
    """
    Import a method from a class instance using a dictionary that specifies
    where to find it.
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
    Applied to a NumPy array or a Torch tensor, it returns a Torch tensor
    on the given device.
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
        # if hasattr(x,'requires_grad'):
        #     if x.requires_grad:
        #         x = x.detach()
        if hasattr(x, 'cuda') and x.is_cuda:
            x = x.data.cpu()
        if hasattr(x, 'numpy'):
            x = x.numpy()
        else:
            x = np.array(x)
    return x


def summary(epoch, i, nb_batch, loss, batch_time, average_loss, average_time,
            mode, reporting_time, average_reporting_time, granularity=None,
            task='reporting'):
    """
    Generate a summary of the model performances on a batch.
    """
    string = f'[{str(mode)}] Epoch: [{epoch}][{i}/{nb_batch}]\t'

    if granularity is None:
        loss_prefix = 'Max'
        time_prefix = 'Batch'
    else:
        loss_prefix = time_prefix = granularity

    if isinstance(loss, str):
        string += f'{loss_prefix} Loss {loss} '
        string += f'(Average {average_loss}) \t'
    else:
        string += f'{loss_prefix} Loss {loss:.4f} '
        string += f'(Average {average_loss:.4f}) \t'
    string += f'{time_prefix} prediction time {batch_time:.4f} '
    string += f'(Average {average_time:.4f}) \t'
    string += f'{time_prefix} {task} time {reporting_time:.4f} '
    string += f'(Average {average_reporting_time:.4f}) \t'

    return string


def instantiate_logger(logger_name, log_level, log_filename, console=True):
    """
    Create a logger that will both write to console and to a log file.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)-2s: %(levelname)-2s : %(message)s")

    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(console_handler)

    return logger


def save_checkpoint(state, save_path, model):
    """
    Save the current model and optimizer state.
    """
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state.get('epoch')
    val_loss = state.get('val_loss')
    iterations = state.get('iterations')

    # Save model
    filename = f'{save_path}/model_ep{epoch}'
    if iterations is not None:
        filename += f'_it{iterations}'
    if val_loss is not None:
        filename += f'_loss{val_loss:0.4f}'
    filename += '.pth.tar'

    if hasattr(model, 'save'):
        model.save(filename)
    else:
        torch.save(state['state_dict'], filename)

    if state['optimizer'] is not None:
        filename = f'{save_path}/opt_ep{epoch}.pth.tar'
        torch.save(state['optimizer'], filename)

    if state['scheduler'] is not None:
        filename = f'{save_path}/sch_ep{epoch}.pth.tar'
        torch.save(state['scheduler'], filename)

    if 'amp' in state:
        filename = f'{save_path}/amp_ep{epoch}.pth.tar'
        torch.save(state['amp'], filename)


class CustomDataset(SubjectsDataset):
    def __init__(self, subjects, transform=None, epoch_length=100):
        super().__init__(subjects=subjects, transform=transform)
        self.epoch_length = epoch_length
        assert len(subjects) % epoch_length == 0
        self.current_index = 0
        self.current_epoch = 0
        self.n_epochs = len(subjects) // epoch_length

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, index: int) -> dict:
        idx = self.current_epoch * self.epoch_length + index
        self.current_index += 1
        if self.current_index == self.epoch_length:
            self.current_index = 0
            self.current_epoch = (self.current_epoch + 1) % self.n_epochs
        return super().__getitem__(idx)
