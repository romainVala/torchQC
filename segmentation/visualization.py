import json
from inspect import signature
from plot_dataset import PlotDataset


def parse_visualization_config_file(folder, visualization_filename='visualization.json'):
    with open(folder + visualization_filename) as file:
        info = json.load(file)

    sig = signature(PlotDataset)
    for key in info.keys():
        if key not in sig.parameters:
            del info[key]
    return info
