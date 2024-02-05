# TODO: Handle iterations

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils_file import get_parent_path
import torch,numpy as np,  torchio as tio
from utils_metrics import compute_metric_from_list #get_tio_data_loader, predic_segmentation, load_model, computes_all_metric
from timeit import default_timer as timer
import json, os, seaborn as sns
from utils_file import gfile, gdir, get_parent_path, addprefixtofilenames
import pandas as pd
from nibabel.viewers import OrthoSlicer3D as ov
from utils_labels import remap_filelist, get_fastsurfer_remap
import matplotlib.pyplot as plt
plt.interactive(True)
sns.set_style("darkgrid")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import colorcet as cc

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

def report_training(results_dir, save=False, sort_time=False):
    #01 2024 let s redo with panda, so that on can easily get every thing
    files = glob.glob(results_dir + '/Train_ep*.csv')
    if sort_time:
        files.sort(key=os.path.getmtime)
    else:
        files.sort()  #alpha numeric order

    df = pd.concat([pd.read_csv(file, index_col=0) for file in files])
    df['iter'] = np.array(range(len(df)))

    if 'train_dice' in df.keys():
        if 'array' in df.train_dice.values[0]:
            df.train_dice.replace({'array\(':'',', dtype=float32\)':''}, inplace=True,regex=True)
        df["train_dice"] = df.train_dice.apply(lambda s: eval(s)) #string to dict
        df["train_dice"] = df.train_dice.apply(lambda s: { 'dice_'+k:v for k,v in s.items()}  )
        df = pd.concat([df, df['train_dice'].apply(pd.Series)], axis=1)
    #let say one epoch is 1000 volume (lines in df)
    dfg = df.groupby(np.arange(len(df)) // 1000)
    dfg_mean, dfg05, dfg95 = dfg.mean(),  dfg.quantile(q=0.05),  dfg.quantile(q=0.95)

    #training loss
    fig = plt.figure('training_los')
    plt.plot(dfg_mean.loss, color='b', label='mean'); plt.plot(dfg05.loss, '--', color='b'); plt.plot(dfg95.loss, '--', color='b', label='0.05 and 0.95 quantile')
    plt.legend(); plt.title('training_loss')
    plt.savefig(results_dir + '/training_loss.jpg');

    #dice order
    ymet=[]
    for k in df.keys():
        if 'dice_' in k:
            ymet.append(k)
    last_dic = {k: round(dfg_mean.iloc[-1,:][k]*100)/100 for k in ymet}
    last_dic = dict(sorted(last_dic.items(), key=lambda item: item[1],reverse=True))
    horder = [k for k in last_dic.keys()]; legend_hue = [f'{k}={v}' for k,v in last_dic.items()]
    dfmm = dfg_mean.melt(id_vars=[ 'iter'], value_vars=ymet, var_name='metric', value_name='dice')
    #fig = sns.catplot(data=dfmm,x='metric',y='y',kind="boxen")
    palette = sns.color_palette(cc.glasbey, n_colors=len(ymet))
    fig = sns.relplot(data=dfmm,hue='metric',y='dice',x='iter',kind='line', hue_order = horder, height=5,
                      palette=palette,aspect=3)
    for t, l in zip(fig._legend.texts, legend_hue):
        t.set_text(l)
    plt.savefig(results_dir + '/training_label_dice_loss.jpg');

    #volume métric on the first 6 epochs
    dfs = df.iloc[:2000].copy()
    dfs["epochs"] = [(k//1000)+1  for k in range(2000)]
    ymet=[]
    for k in df.keys():
        if  k.startswith('occupied_volume'):
            ymet.append(k)
    dfmm = (dfs/dfs.mean()).melt(id_vars=[ 'iter','epochs'], value_vars=ymet, var_name='metric', value_name='targetVol')
    dfmm.metric.replace({'occupied_volume': 'V'}, inplace=True, regex=True)

    fig = sns.catplot(data=dfmm,x='metric',y='targetVol',kind="violin", col='epochs', height=5, aspect=3,col_wrap=1)
    #fig = sns.relplot(data=dfmm,hue='metric',y='dice',x='iter',kind='line', height=5, aspect=3)
    #fig = sns.histplot(data=dfmm,hue='metric',x='dice', height=5, aspect=3)
    plt.savefig(results_dir + '/label_volume_2_epoch.jpg');


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

        dirpass = glob.glob(results_dir+'/pas*')
        dirpass.sort()
        if len(dirpass)>0:
            files_all = []
            for dirp in dirpass:
                print(f'including resdir {dirpass[0]}')
                files_more = glob.glob(dirp + pattern)
                files_more.sort()  # alpha numeric order
                files_all += files_more
            files = files_all + files

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
        resname = get_parent_path(results_dir, level=1)[1]
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
         #plt.show()

        if save:
            plt.savefig(results_dir + '/{}loss_curves.jpeg'.format(resname))

    plt.figure('allres')
    for tt in train_loss:
        plt.plot(tt)
    plt.legend(resname_list)
