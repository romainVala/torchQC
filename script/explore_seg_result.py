# from segmentation.eval_results.occupation_stats import occupation_stats, \
# abs_occupation_stats, dice_score_stats, bin_dice_score_stats
# from segmentation.eval_results.compare_results import compare_results
# from itertools import product

# model_prefixes = ['bin_synth', 'bin_dice_pve_synth', 'pve_synth']
# GM_levels = ['03', '06', '09']
# noise_levels = ['01', '05', '1']
# modes = ['t1', 't2']
# results_dirs, names = [], []
# root = '/home/fabien.girka/data/segmentation_tasks/RES_1.4mm/'
# filename = f'{root}compare_models.csv'
# metrics = [dice_score_stats, bin_dice_score_stats, occupation_stats, abs_occupation_stats]
#
# for mode, GM_level, noise, prefix in product(modes, GM_levels, noise_levels, model_prefixes):
#     results_dirs.append(f'{root}{prefix}_data_64_common_noise_no_gamma/eval_on_{mode}_like_data_GM_{GM_level}_{noise}_noise')
#     names.append(f'{prefix}_{mode}_GM_{GM_level}_{noise}_noise')
#     compare_results(results_dirs, filename, metrics, names)

from utils import reduce_name_list, remove_string_from_name_list
from utils_plot_results import get_ep_iter_from_res_name, plot_resdf, plot_train_val_results, \
    transform_history_to_factor, parse_history
from utils_file import get_parent_path, gfile, gdir

from segmentation.eval_results.compare_results import plot_value_vs_GM_level, aggregate_csv_files,aggregate_all_csv, plot_metric_against_GM_level
import glob
from segmentation.eval_results.learning_curves import  report_learning_curves
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt, pandas as pd
#manual ploting
import seaborn as sns
import pandas as pd
import json
sns.set_style("darkgrid")

results_dirs = glob.glob('/home/fabien.girka/data/segmentation_tasks/RES_1.4mm/eval_models_with_more_metrics/data_t*')
results_dirs += glob.glob('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_samseg/eval-14mm*')


plot_value_vs_GM_level(results_dirs, 'dice_loss', ylim=(0, 0.2),
                       save_fig='/home/romain.valabregue/datal/PVsynth/figure/dice_against_GM_res_14mm')

results_dirs = glob.glob('/home/fabien.girka/data/segmentation_tasks/RES_1mm/eval_models_with_more_metrics/data_t*')
results_dirs += glob.glob('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_samseg/eval-1mm*')

results_dirs = glob.glob('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_1mm/eval_metric_on_pv/data_t*')
results_dirs = glob.glob('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_1.4mm/eval_metric_on_pv/data_t*')
results_dirs = glob.glob('/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_1mm/eval_metric_on_pv/data_t*')
results_dirs = glob.glob('/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_1mm/eval_metric_on_bin/data_t*')

results_dirs = glob.glob('/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_1.4mm/eval_metric_on_pv/data_t*')
results_dirs = glob.glob('/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_1.4mm/eval_metric_on_bin/data_t*')

results_dirs += glob.glob('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_samseg/eval-1mm*')
results_dirs += glob.glob('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_samseg/eval-14mm*')


#Explore training curve
res='/home/fabien.girka/data/segmentation_tasks/RES_1.4mm/bin_synth_data_64_common_noise_no_gamma/results_cluster'
res = gdir('/home/fabien.girka/data/segmentation_tasks/RES_1mm/','data_64_common_noise_no_gamma')
res = gdir('/home/romain.valabregue/datal/PVsynth/training','pve')
res = gdir('/home/romain.valabregue/datal/PVsynth/jzay/training/RES28mm','data')
res = gdir('/home/romain.valabregue/datal/PVsynth/jzay/training/RES1mm','data')
res = gdir('/home/romain.valabregue/datal/PVsynth/training/RES_14mm_tissue','data')
res = gdir(res,'resul')
report_learning_curves(res)


#explore synthetic data histogram
results_dirs = glob.glob('/home/romain.valabregue/datal/PVsynth/RES_1.4mm/t*/513130')
f=gfile(results_dirs,'^5.*nii')
resname = get_parent_path(results_dirs,2)[1]

resfig = '/home/romain.valabregue/datal/PVsynth/figure/volume_synth/'
for i, ff in enumerate(f):
    img = nb.load(ff)
    data = img.get_fdata(dtype=np.float32)
    fig = plt.figure(resname[i])
    hh = plt.hist(data.flatten(), bins=500)
    axes = plt.gca()
    axes.set_ylim([0 , 40000])
    #fig.savefig(resfig + resname[i] + '.png')



#concat eval.csv
res = '/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_14mm_tissue/dataS*/*/eval.csv'
file = '/home/romain.valabregue/datal/PVsynth/eval_cnn/res14mm_tissue_all.csv'
res = '/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_14mm_tissue_onfly/dataS*/*/eval.csv'
file = '/home/romain.valabregue/datal/PVsynth/eval_cnn/res14mm_tissue_onfly_all.csv'

res = '/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_1mm_tissue_valid_models/dataS*/*/eval.csv'
file = '/home/romain.valabregue/datal/PVsynth/eval_cnn/res1mm_tissue_eval_models.csv'

res = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_1mm_tissue/prepare_job_test/dataS*/*csv'
file = '/home/romain.valabregue/datal/PVsynth/eval_cnn/res1mm_tissue_eval_models_pve_synth_drop01.csv'

res = '/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_1mm_tissue_eval_augment/dataS*/*/*csv'
file = '/home/romain.valabregue/datal/PVsynth/eval_cnn/res1mm_tissue_eval_augment_pve_bin_01.csv'
file = '/home/romain.valabregue/datal/PVsynth/eval_cnn/res1mm_tissue_eval_augment_pve_bin_02.csv'
res = '/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_1mm_tissue_eval_augment/dataS*mod3*/*/*csv'
file = '/home/romain.valabregue/datal/PVsynth/eval_cnn/res1mm_tissue_eval_augment_mod3_01.csv'
res = '/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_1mm_tissue_eval_augment/dataS*mod3*/*/*csv'
file = '/home/romain.valabregue/datal/PVsynth/eval_cnn/res1mm_tissue_eval_augment_mod3_01.csv'
file = '/home/romain.valabregue/datal/PVsynth/eval_cnn/res1mm_tissue_eval_augment_mod3_02.csv'
res = '/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_1mm_all_tissue/dataS*/*/*csv'
file = '/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_1mm_all_tissue/all.csv'

aggregate_csv_files(res, file, fragment_position=-3)

res = '/home/romain.valabregue/datal/PVsynth/jzay/eval/eval_cnn/RES_14mm/data_G*/*/eval.csv'
file = '/home/romain.valabregue/datal/PVsynth/jzay/eval/eval_cnn/res14mm_all.csv'
res = '/home/romain.valabregue/datal/PVsynth/jzay/eval/eval_cnn/RES_1mm_tissue/data_S*/*/eval.csv'
res = '/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_1mm_tissue/dataS*/*/eval.csv'
file = '/home/romain.valabregue/datal/PVsynth/eval_cnn/res1mm_tissue_all.csv'
file = '/home/romain.valabregue/datal/PVsynth/eval_cnn/res14mm_tissue_all.csv'
file = '/home/romain.valabregue/datal/PVsynth/eval_cnn/old_csv/res28mm_all.csv'

aggregate_csv_files(res, file, fragment_position=-3)

metrics = ['metric_dice_loss_GM', 'metric_bbin_dice_loss_GM', 'metric_bin_dice_loss_GM',
           'metric_l1_loss_GM', 'metric_l1_loss_GM_mask_GM', 'metric_l1_loss_GM_mask_NO_GM',
           'metric_bin_volume_ratio_GM','metric_volume_ratio_GM']
metrics = ['metric_dice_loss_GM',
           'metric_l1_loss_GM', 'metric_l1_loss_GM_mask_GM', 'metric_l1_loss_GM_mask_NO_GM',
           'metric_bin_volume_ratio_GM','metric_volume_ratio_GM']

#metrics += ['metric_l1_loss_on_band_GM','metric_l1_loss_on_band_GM_far']

filter = dict(col='model', str='mRes')
filter = dict(col='model', str='M40')
filter = dict(col='model', str='(bin_dice)|(bin_syn)|(pve_synth)')
filter = dict(col='model', str='(pve_mResDp_)|(pve_synth)')
filter = dict(col='model', str='(synDp)|(pve_synth)')

plot_metric_against_GM_level(file, metrics=metrics, filter=filter, remove_max=False, add_strip=False,
                             save_fig='/home/romain.valabregue/datal/PVsynth/figure/new3/tissue_28mm',
                             kind='box', enlarge=True, showfliers=False)

df = pd.read_csv(file, index_col=0)
rn = df['T_RandomLabelsToImage']
noisdic = [json.loads(r)['seed'] for r in rn]

df = pd.read_csv(file)

def split_model_name(s):
    s_list = s.split('_')
    return str.join('_',s_list[0:2])
def split_model_epoch(s):
    s_list = s.split('_')
    return int(s_list[-1][2:])
def gess_transfo(s):
    return 'affine' if 'RandomAffine' in s else 'bias' if 'BiasFiel' in s else 'motion' if 'Motion' in s else None

df['model_name'] = df['model'].apply(lambda s: split_model_name(s))
df['epoch'] = df['model'].apply(lambda s: split_model_epoch(s))
df['transfo'] = df['transfo_order'].apply(lambda s: gess_transfo(s))

from pathlib import PosixPath
if not isinstance(df['image_filename'][0], str):  # np.isnan(df['image_filename'][0]):
    ffarg = [eval(fff)[0] for fff in df['label_filename'].values]
    ff = [eval(fff)[0].parent.parent.parent.name for fff in ffarg] #df['label_filename'].values]
else:
    ff = [eval(fff)[0].parent.name for fff in df['image_filename'].values]
df['suj_name'] = ff

filter = dict(col='model', str='ep60')
if filter:
    rows = df[filter['col']].str.contains(filter['str'])
    df = df[ ~ rows]

sns.catplot(data=df, x='epoch', y='metric_dice_loss_GM', kind='box', col='model_name')
sns.catplot(data=df, x='epoch', y='metric_l1_loss_GM_mask_PV_GM', kind='box', col='model_name')
sns.catplot(data=df, x='suj_name', y='metric_dice_loss_GM', kind='box', col='model_name')
sns.catplot(data=df, x='suj_name', y='predicted_occupied_volume_GM', kind='box', col='model_name')

sns.catplot(data=df, x='transfo', y='predicted_occupied_volume_GM', kind='box', col='model')
sns.catplot(data=df2, x='transfo', y='metric_dice_loss_GM', kind='box', col='SNR', hue='GM')
sns.catplot(data=df, x='transfo', y='metric_dice_loss_GM', kind='box', col='model', hue='GM')
df1 = df[df.model=='dp_mod3_ep90']
df2 = df[df.model=='dp_mod3_Aug_ep90']
dff=pd.concat([df1,df2])
df['gm_snr'] = df['GM'].astype(str).str.cat(df['SNR'].astype(str), sep='_')
dff['gm_snr'] = dff['GM'].astype(str).str.cat(dff['SNR'].astype(str), sep='_')

sns.catplot(data=dff, x='transfo', y='metric_dice_loss_GM', kind='box', col='model', hue='gm_snr')
sns.catplot(data=dff, x='transfo', y='metric_dice_loss_CSF', kind='box', col='model', hue='gm_snr')
sns.catplot(data=dff, x='transfo', y='metric_l1_loss_GM_mask_PV_GM', kind='box', col='model', hue='gm_snr')
dd = df2.loc[(df2.GM==0.6) & (df2.SNR==0.01) & (df2.transfo=='bias') & (df2.metric_dice_loss_GM>0.1),:]
dd = df1.loc[(df1.GM==0.6) & (df1.SNR==0.01) & (df1.transfo=='motion'),:]

for k in df.keys():
#    if k.rfind('ratio')>0:
#    if k.startswith('occupied_volume_'): # in k:
    if k.startswith('metric_'):  # in k:

        print(k)
        #sns.catplot(data=df, x='suj_name', y=k, kind='box', col='model_name')
        sns.catplot(data=df1, x='transfo', y=k, kind='box', col='SNR', hue='GM')
        plt.show()
metric_mean_dice_loss
metric_dice_loss_CSF

#read transform param and metric form train.csv
import glob, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from read_csv_results import ModelCSVResults
ft = glob.glob('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion_New/train_random_synth/result_regre*/Tra*csv')
ft.sort(key=os.path.getmtime)

mres = ModelCSVResults(ft[29],  out_tmp="/tmp/rrr")
evalfunc = lambda x: eval(x) if not (pd.isna(x)) else None

mres.normalize_dict_to_df('T_RandomMotionFromTimeCourse_metrics_t1', eval_func=evalfunc)
mres.normalize_dict_to_df('T_RandomAffine', eval_func=evalfunc)
mres.normalize_dict_to_df('T_RandomLabelsToImage', eval_func=evalfunc)
mres.normalize_dict_to_df('T_RandomNoise', eval_func=evalfunc)

mres.scatter('loss','T_RandomMotionFromTimeCourse_metrics_t1_L1_map')
mres.scatter('T_RandomMotionFromTimeCourse_metrics_t1_L1_map','T_RandomMotionFromTimeCourse_metrics_t1_SSIM_ssim_SSIM')
mres.scatter('T_RandomMotionFromTimeCourse_metrics_t1_SSIM_ssim_SSIM','T_RandomMotionFromTimeCourse_metrics_t1_NCC')

from torch import tensor
df['scale'] = df['T_RandomAffine_apply_scales'].apply(lambda x: eval(x).prod().numpy() if not (pd.isna(x)) else None)


df['mean_mean'] = df['T_RandomLabelsToImage_random_parameters_images_dict'].apply(lambda x: np.mean(x['mean']))

df = mres.df_data
plt.scatter(df['loss'], df['T_RandomNoise_std'])
plt.scatter(df['loss'], df['scale'])
plt.scatter(df['loss'], df['mean_mean'])
plt.scatter(df['T_RandomMotionFromTimeCourse_metrics_t1_L1_map'], df['T_RandomMotionFromTimeCourse_metrics_t1_SSIM_ssim_SSIM'])
plt.scatter(df['loss'], df['T_RandomMotionFromTimeCourse_metrics_t1_SSIM_ssim_SSIM'])
plt.scatter(df['T_RandomMotionFromTimeCourse_metrics_t1_SSIM_structure_SSIM'], df['T_RandomMotionFromTimeCourse_metrics_t1_SSIM_ssim_SSIM'])
plt.scatter(df['T_RandomMotionFromTimeCourse_metrics_t1_SSIM_contrast_SSIM'], df['T_RandomMotionFromTimeCourse_metrics_t1_SSIM_ssim_SSIM'])
plt.scatter(df['T_RandomMotionFromTimeCourse_metrics_t1_SSIM_contrast_SSIM'],df['T_RandomMotionFromTimeCourse_metrics_t1_NCC'])
df.shape
df['error'] = df['l']
pd.isna(df['T_RandomElasticDeformation']).sum()
pd.isna(df['T_RandomMotionFromTimeCourse_metrics_t1']).sum()


res = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion_New/train_random_synth/eval/data/S*/*/*csv'
file = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion_New/train_random_synth/eval/res_eval1.csv'
aggregate_all_csv(res, file, parse_names=['Suj','model'], fragment_position=-3)

df = pd.read_csv('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion_New/train_random_synth/eval/res_eval1.csv')
#sns.relplot(data=df, x='pred_ssim_SSIM_brain', y = 'tar_ssim_SSIM_brain', col='Suj', hue='model',kind='scatter')
for s in df.Suj.unique():
    for m in df.model.unique():
        dfsub = df[df.Suj==s]
        dfsub = dfsub[dfsub.model==m ]
        plt.figure()
        plt.title(s+' mod ' + m)
        plt.scatter(dfsub.pred_ssim_SSIM_brain,dfsub.tar_ssim_SSIM_brain)
        plt.xlabel('prediction'); plt.ylabel('target ssim_brain')
        plt.plot([0, 1], [0,1])

dfsub = df[df.Suj=='hcp_synth']; dfsub = dfsub[dfsub.model=='noise_sim']

mres = ModelCSVResults(df_data=dfsub,  out_tmp="/tmp/rrr")
mres.scatter('pred_ssim_SSIM_brain', 'tar_ssim_SSIM_brain',port_number=8086)
#keys_unpack = ['T_RandomLabelsToImage','T_RandomMotionFromTimeCourse_metrics_t1','T_RandomAffine', 'T_RandomMotionFromTimeCourse']
keys_unpack = ['T_Affine','T_ElasticDeformation', 'T_Noise','T_RandomMotionFromTimeCourse', 'T_BiasField','T_LabelsToImage']
suffix = ['Taff', 'Tela','Tnoi', 'Tmot', 'Tbias', 'Tlab']
df1 = mres.normalize_dict_to_df(keys_unpack, suffix=suffix);

def xxx(s):
    return s['t1'] if s is not None else 0
df1.Tnoi_std = df1.Tnoi_std.apply(lambda  x: xxx(x))

sns.scatterplot(data=df1,x ='pred_ssim_SSIM_brain',y = 'tar_ssim_SSIM_brain', hue='Tnoi_std', legend="full")
df1.loc[['pred_ssim_SSIM_brain','tar_ssim_SSIM_brain','Tnoi_std']].describe()


#test load history 01 2021
from read_gz_results import GZReader
fres = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_1mm_all_tissue/dataS_GM6_T1_SNR_100_model_dp_mod3_Aug_ep106/704238/eval.gz'
gz = GZReader(fres)

df = gz.df_data
vol = gz.get_volume_torchio(0)
df = torch.load(fres)
trfms = gz.get_transformations(0)
sel_key=[]
for k in df.keys():
    if 'l1' in k:
        print(k)
        sel_key.append(k)
sel_key.append("metric_far_isolated_GM_points_in_CSF"); sel_key.append("metric_far_isolated_GM_points_in_WM")
sns.pairplot(df[sel_key], kind="scatter", corner=True)
