import json
from importlib import import_module


def import_object(module, name, package=''):
    mod = import_module(module, package)
    return getattr(mod, name)


def generate_json_document(filename, **kwargs):
    with open(filename, 'w') as file:
        json.dump(kwargs, file)
