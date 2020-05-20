import json
from importlib import import_module
import torch


default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def import_model(module, name, package=''):
    mod = import_module(module, package)
    return getattr(mod, name)


def load_model(folder, model_filename='model.json', device=default_device):
    with open(folder + model_filename) as file:
        info = json.load(file)

    model = info.get('model')
    model_name = model.get('name')
    module = model.get('module')
    package = model.get('package') or ''
    attributes = model.get('attributes') or {}
    load = model.get('load')

    model_class = import_model(module, model_name, package)
    model = model_class(**attributes)

    if load is not None:
        if load.get('custom'):
            model, params = model_class.load(load.get('path'))
        else:
            model.load_state_dict(torch.load(load.get('path')))

    return model.to(device)
