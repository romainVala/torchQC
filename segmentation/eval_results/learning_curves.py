# TODO: Handle iterations

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils_file import get_parent_path

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_df_column(df, col_name):
    for col in col_name:
        plt.figure()
        plt.title(col)
        y = df[col].values
        plt.plot(y)
        ys = smooth(y,20)
        plt.plot(ys)

def read_csv_in_data_frame(results_dir, pattern='Train*csv'):
    df_all = pd.DataFrame()
    files = glob.glob(results_dir +'/' + pattern)
    files.sort()  # alpha numeric order

    for file in files:
        df = pd.read_csv(file, index_col=0)
        df_all = pd.concat([df_all, df], axis=0, ignore_index=True)

    return df_all

def report_df_col(df, col_list, info_title=''):
    for col in col_list:
        plt.figure()
        plt.title(f'{col}_{info_title}')
        plt.ylabel(col)
        aa = df[col]
        plt.plot(aa)
        aas = smooth(aa,20)
        plt.plot(aas)


def report_learning_curves(results_dirs, save=False, sort_time=False):
    """
    Plot error curves from record files and save plot to jpeg file.
    """
    def plot_losses(pattern, color, label):
        files = glob.glob(results_dir + pattern)
        if sort_time:
            files.sort(key=os.path.getmtime)
        else:
            files.sort()  #alpha numeric order
        losses, ymean, qt05, qt95 = [], [], [], []
        nb_iter_list = []
        nb_iter_mem = -1
        for file in files:
            df = pd.read_csv(file, index_col=0)
            loss = df['loss'].values
            ymean.append(np.mean(loss))
            qt05.append(np.quantile(loss, 0.05))
            qt95.append(np.quantile(loss, 0.95))
            #losses.append(loss)
            nb_iter = df.shape[0]
            if nb_iter_mem >= 0:
                if not(nb_iter_mem == nb_iter):
                    print('\t {} \t {} iter previous {} '.format(os.path.basename(file), nb_iter, nb_iter_mem,))
            nb_iter_mem = nb_iter
            nb_iter_list.append(nb_iter)

        #plt.plot(np.mean(losses, axis=1), color=color, label=label)
        #plt.plot(np.quantile(losses, 0.95, axis=1), '--', color=color)
        #plt.plot(np.quantile(losses, 0.05, axis=1), '--', color=color)
        plt.plot(ymean, color=color, label=label)
        plt.plot(qt95, '--', color=color)
        plt.plot(qt05, '--', color=color, label='0.05 and 0.95 quantile')

        return [np.max(nb_iter_list), np.min(nb_iter_list), ymean]

    if isinstance(results_dirs,str):
        results_dirs = [results_dirs]

    train_loss, resname_list = [], []
    for results_dir in results_dirs:

        resname = get_parent_path(results_dir, level=2)[1]
        print(resname)
        resname_list.append(resname)
        plt.figure(resname)

        iter_max, iter_min, ymeans = plot_losses('/Train_ep*.csv', 'blue', 'average over the epochs')
        train_loss.append(ymeans)

        #plot_losses('/Val_ep*.csv', 'green', 'val mean loss')

        if iter_max==iter_min:
            plt.xlabel('epch ( {} iter)'.format(iter_max))
        else:
            plt.xlabel('epoch ( {} iter {})'.format(iter_max, iter_min))
        plt.legend()
        #plt.title('Training and validation error curves')
        plt.title('per epoch Training loss')
        plt.ylabel("Mean Dice Loss")
        plt.show()

        if save:
            plt.savefig(results_dir + '/{}loss_curves.jpeg'.format(resname))

    plt.figure('all')
    for tt in train_loss:
        plt.plot(tt)
    plt.legend(resname_list)
