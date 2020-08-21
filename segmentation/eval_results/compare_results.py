import pandas as pd
import collections
from pathlib import Path


def flatten(d, parent_key='', sep='_'):
    """ From https://stackoverflow.com/a/6027615"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def compare_results(results_dirs, filename, metrics, names=None):
    results = {}
    for metric in metrics:
        for i, results_dir in enumerate(results_dirs):
            if names is not None:
                name = names[i]
            else:
                name = Path(results_dir).parent.name
            results[f'{name}_{metric.__name__}'] = flatten(metric(results_dir))

    df = pd.DataFrame.from_dict(results)
    df.to_csv(filename)
