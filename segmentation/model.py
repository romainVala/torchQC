""" Load model """

import json
import torch
from segmentation.utils import parse_object_import, set_dict_value, check_mandatory_keys


MODEL_KEYS = ['model']
MODEL_DICT_KEYS = ['name', 'module']


def load_model(folder, model_filename='model.json'):
    """ Load a model using the information in a json configuration file """
    with open(folder + model_filename) as file:
        info = json.load(file)

    check_mandatory_keys(info, MODEL_KEYS, folder + model_filename)
    check_mandatory_keys(info['model'], MODEL_DICT_KEYS, 'model dict')
    set_dict_value(info, 'load')
    set_dict_value(info, 'device', 'cuda')

    if info['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model, model_class = parse_object_import(info['model'])

    if info['load'] is not None:
        set_dict_value(info['load'], 'custom')
        if info['load']['custom']:
            model, params = model_class.load(info['load']['path'])
        else:
            model.load_state_dict(torch.load(info['load']['path']))

    return model.to(device)
