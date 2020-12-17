import matplotlib.pyplot as plt, pandas as pd, seaborn as sns
import torchio as tio, torch
from segmentation.config import Config
from segmentation.run_model import RunModel
from nibabel.viewers import OrthoSlicer3D as ov
import glob, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
#pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1, 'display.width', 400)

import nibabel as nib

from read_csv_results import ModelCSVResults
def get_metric_key_from_df(df, label_list):
    mkeys = dict();
    metric_list = []
    for k in df.keys():
        if k.startswith('m_'):
            indl = k.find('_label_L')
            if indl < 0:
                #if len(metric_list) > 0:
                metric_list = [k]
                mkeys[k] = metric_list
            else:
                label_num = int( k[indl+8:] )
                new_name = metric_list[0] + '_' + label_list[label_num]
                df = df.rename(columns={k:new_name})
                metric_list.append(new_name)
    return df, mkeys

def my_mel_plot(df1):
    dfall = pd.DataFrame()
    for mkey, mlist in mkeys.items():
        print(mkey)
        dfsub = df1[mlist]
        mlist[0] += '_'
        x = [xx[len(mlist[0]):] for xx in mlist]
        x[0] = 'all'
        mlist[0] = mlist[0][:-1]
        for ii, xx in enumerate(x):
            if xx.startswith('both_R_'): x[ii] = xx[7:]
        dfsub = dfsub.rename(columns={on: nn for on, nn in zip(mlist, x)})

        dfm = dfsub.melt(id_vars=None, value_vars=x, var_name='label', value_name='m_val')
        dfm['m_name'] = mkey
        dfall = dfall.append(dfm)

        g = sns.catplot(data=dfm, x='label', y='m_val')
        plt.subplots_adjust(top=0.9);
        plt.gcf().set_size_inches([10, 4.5]);
        g.fig.suptitle(mkey)
        # sns.pairplot(dfsub, kind="scatter", corner=True)
    return dfall

result_dir = '/data/romain/PVsynth/ex_transfo/pve_synth_drop01_aug_mot/rrr'
file = '/data/romain/PVsynth/ex_transfo/pve_synth_drop01_aug_mot/main.json'
result_dir = '/data/romain/PVsynth/ex_transfo/pve_synth_drop01_aug_mot/rrr'
config = Config(file, result_dir, mode='eval')
config.init()
mr = config.get_runner()
label_list = mr.labels
label_list =["GM", "CSF", "WM", "both_R_Accu", "both_R_Amyg", "both_R_Caud", "both_R_Hipp", "both_R_Pall", "both_R_Puta", "both_R_Thal", "backNOpv"]
label_list =["GM","CSF","WM","both_R_Accu", "both_R_Amyg","both_R_Caud","both_R_Hipp","both_R_Pall","both_R_Puta", "both_R_Thal", "BrStem","cereb_GM", "cereb_WM","skin","skull","background"]

d= result_dir + '/'
f1 = d + 'Train_ep001.csv'
#f1 = d + 'Train_random_label_one_small_motion.csv'
f2 = d + 'Train_ep003.csv'
#f2 = d + 'Train_random_label_affine_one_small_motion.csv'
#f2 = d + 'Train_one_contrast_radom_motion_small.csv'

mres = ModelCSVResults(f1,  out_tmp="/tmp/rrr")
mres2 = ModelCSVResults(f2,  out_tmp="/tmp/rrr")

keys_unpack = ['T_LabelsToImage','T_Affine', 'T_RandomMotionFromTimeCourse', 'Tm__metrics', 't1']
suffix = ['Tl', 'Ta', 'Tm', '', 'm']
df1 = mres.normalize_dict_to_df(keys_unpack, suffix=suffix); df1 = df1.rename(columns = {"sample_time":"meanS"})
df2 = mres2.normalize_dict_to_df(keys_unpack, suffix=suffix); df2 = df2.rename(columns = {"sample_time":"meanS"})

sel_key = ['luminance_SSIM', 'structure_SSIM', 'contrast_SSIM', 'ssim_SSIM', 'L1','NCC' ]
sel_key = ['nL2e', 'pSNR', 'metric_ssim_old', 'L1_map', 'NCC', 'meanS', 'Tm_mean_DispP']
sel_key = ['nL2e', 'pSNRm', 'L1', 'L2', 'NCC', 'ssim_SSIM','ssim_SSIM_brain'] #'meanS'
sel_key = ['ssim_SSIM', 'ssim_SSIM_brain', 'NCC', 'NCC_brain','contrast_SSIM', 'contrast_SSIM_brain' ]
sel_key =  ['Tm_mean_DispP', 'Tm_rmse_Disp', 'Tm_meanDispP_wTF2', 'Tm_rmse_Disp_wTF2', 'NCC']
sel_key =  ['m_NCC', 'm_L1', 'm_L2',  'Tm_mean_DispP']

#sur le graphe one motion different contrast,
#L1 L2 pSNR tres correle
#nL2e nL2m pSNRm tres correle (nL2m et pSNRm sont tres tres correle) NCC plus proche de nL2e

dff, mkeys = get_metric_key_from_df(df2, label_list)
dfall = my_mel_plot(dff)
dfp = dfall.pivot(columns='m_name', values='m_val')
sns.pairplot(dfp, kind="scatter", corner=True)

sns.catplot(data=dfall, x='label', y='m_val', col='m_name')


sns.pairplot(df1[sel_key], kind="scatter", corner=True)
sns.pairplot(df2[sel_key], kind="scatter", corner=True)

plt.scatter(df1['L1_map'], df1['NCC'])
plt.figure();plt.scatter(df1['metric_ssim_old'], df1['ssim_SSIMr'])
plt.figure();plt.scatter(df1['SSIM_ssim_SSIM'], df1['NCC'])
plt.figure();plt.scatter(df1['SSIM_ssim_SSIM'], df1['SSIM_contrast_SSIM'])
plt.figure();plt.scatter(df1['SSIM_ssim_SSIM'], df1['SSIM_structure_SSIM'])
plt.figure();plt.scatter(df1['SSIM_ssim_SSIM'], df1['SSIM_luminance_SSIM'])
plt.figure();plt.scatter(df1['NCC'], df1['NCC_c'])
plt.figure();plt.scatter(df1['SSIM_contrast_SSIM'], df1['metric_ssim_old'])


mres.scatter('L1_map','NCC')




