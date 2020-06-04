import json
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from inspect import signature
from operator import itemgetter
from segmentation.utils import check_mandatory_keys
from plot_dataset import PlotDataset


VISUALIZATION_KEYS = ['image_key_name', 'label_key_name']


def parse_visualization_config_file(folder, visualization_filename='visualization.json'):
    """
    Get PlotDataset arguments from a json configuration file.
    """
    with open(folder + visualization_filename) as file:
        info = json.load(file)

    check_mandatory_keys(info, VISUALIZATION_KEYS, folder + visualization_filename)

    sig = signature(PlotDataset)
    for key in info.keys():
        if key not in sig.parameters:
            del info[key]
    return info


def report_loss(results_folder):
    """
    Plot error curves from record files and save plot to jpeg file.
    """
    def get_number(number_type, string):
        number = re.findall(f'{number_type}[0-9]+', string)[0]
        return int(number[len(number_type):])

    def get_info(files):
        info = []
        for file in files:
            epoch = get_number('ep', file)
            iteration = get_number('it', file)
            df = pd.read_csv(file, index_col=0)
            losses = df['loss'].values
            info.append((epoch, iteration, losses))
        info.sort(key=itemgetter(0, 1))
        return info

    def get_losses(info):
        iteration_multiple = info[0][1]
        loss = []
        mean_loss = []
        var_loss = []

        for _, _, losses in info:
            loss.append(losses)
            mean_loss.append(np.mean(losses))
            var_loss.append(np.var(losses))

        loss = np.repeat(loss, iteration_multiple, axis=0)
        mean_loss = np.repeat(mean_loss, iteration_multiple)
        var_loss = np.repeat(var_loss, iteration_multiple)

        return loss, mean_loss, var_loss

    train_files = glob.glob(results_folder + '/Train_ep*.csv')
    val_files = glob.glob(results_folder + '/Val_ep*.csv')

    train_info = get_info(train_files)
    val_info = get_info(val_files)

    train_loss, train_mean_loss, train_var_loss = get_losses(train_info)
    val_loss, val_mean_loss, val_var_loss = get_losses(val_info)

    plt.plot(train_mean_loss, color='blue', label='train mean loss')
    plt.plot(train_mean_loss + 1.96 * np.sqrt(train_var_loss), '--', color='blue')
    plt.plot(train_mean_loss - 1.96 * np.sqrt(train_var_loss), '--', color='blue')

    plt.plot(val_mean_loss, color='green', label='val mean loss')
    plt.plot(val_mean_loss + 1.96 * np.sqrt(val_var_loss), '--', color='green')
    plt.plot(val_mean_loss - 1.96 * np.sqrt(val_var_loss), '--', color='green')

    n_iter_per_epoch = next(filter(lambda x: x[0] == 1, reversed(train_info)))[1]
    n_epochs = train_info[-1][0]
    for epoch in range(1, n_epochs):
        plt.axvline(x=n_iter_per_epoch * epoch, color='red')

    plt.xlabel('iterations')
    plt.legend()
    plt.title('Training and validation error curves')
    plt.show()

    plt.savefig(results_folder + '/error_curves.jpeg')
