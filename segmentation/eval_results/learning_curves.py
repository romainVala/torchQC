# TODO: Handle iterations

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def report_learning_curves(results_dir, save=False):
    """
    Plot error curves from record files and save plot to jpeg file.
    """
    def plot_losses(pattern, color, label):
        files = glob.glob(results_dir + pattern)
        files.sort(key=os.path.getmtime)
        losses = []
        for file in files:
            df = pd.read_csv(file, index_col=0)
            loss = df['loss'].values
            losses.append(loss)
        plt.plot(np.mean(losses, axis=1), color=color, label=label)
        plt.plot(np.quantile(losses, 0.95, axis=1), '--', color=color)
        plt.plot(np.quantile(losses, 0.05, axis=1), '--', color=color)

    plot_losses('/Train_ep*.csv', 'blue', 'train mean loss')
    plot_losses('/Val_ep*.csv', 'green', 'val mean loss')

    plt.xlabel('Epochs')
    plt.legend()
    plt.title('Training and validation error curves')
    plt.show()

    if save:
        plt.savefig(results_dir + '/error_curves.jpeg')