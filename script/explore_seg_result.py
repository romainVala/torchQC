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
import glob, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from read_csv_results import ModelCSVResults

from segmentation.eval_results.compare_results import plot_value_vs_GM_level, aggregate_csv_files,aggregate_all_csv, plot_metric_against_GM_level
import glob
from segmentation.eval_results.learning_curves import  report_learning_curves, read_csv_in_data_frame, report_df_col
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
res = gdir('/home/romain.valabregue/datal/PVsynth/jzay/training/RES1mm_prob','pve_synth_mod3_P128$')
res = gdir('/home/romain.valabregue/datal/PVsynth/jzay/training/RES1mm_prob','aniso')
res = gdir(res,'results_cluster')
res = ['/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/jzay/training/RES1mm_prob/pve_synth_mod3_P128_aniso_LogLkd_reg_multi/result',
       '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/jzay/training/RES1mm_prob/pve_synth_mod3_P128_aniso_LogLkd_reg_unis_lam1/results_cluster',
       '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/jzay/training/RES1mm_prob/pve_synth_mod3_P128_aniso_LogLkd_classif/results_cluster',
       ]
res = ['/home/romain.valabregue/datal/PVsynth/jzay/training/RES1mm_prob/pve_synth_mod3_P128/results_cluster/']
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
res = '/home/romain.valabregue/datal/PVsynth/eval_cnn/ext_tool/eval_synth_seg/*/*csv'
file1 = '/home/romain.valabregue/datal/PVsynth/eval_cnn/ext_tool/eval_synth_seg/all.csv'

res = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_1mm_prob/*RES14*/*/*csv'
file ='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_1mm_prob/res14mm_all.csv'

res = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_1mm_prob/*RES14*/*/*csv'
file ='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_1mm_prob/res14mm_all_new.csv'

res = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_prob_tissue/RES14mm_tissu_16_14/*/*/*csv'
file ='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_prob_tissue/res14mm_all.csv'
res = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_prob_tissue/RES1mm_tissu_17_15/*/*/*csv'
file ='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_prob_tissue/res1mm_all.csv'
res = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_prob_tissue/RES1mm_tissu_16_14_augment/*/*/*csv'
file ='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_prob_tissue/res1mm_augment_all.csv'
file_list = [file1, file]

res = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_prob_tissue/RES1mm_tissu_17_15_mask_reduce/*/*/*csv'
res = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_prob_tissue/RES14mm_tissu_16_14_mask_reduce/*/*/*csv'
res = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_prob_tissue/RES28mm_tissu_16_14_mask_reduce/*/*/*csv'

aggregate_csv_files(res, file, fragment_position=-3)
aggregate_csv_files(res, file, fragment_position=-3, name_type=0) #for HCP
aggregate_csv_files(res, file1, fragment_position=-2, name_type=2)
aggregate_csv_files(res, file, fragment_position=-3, name_type=2)# for RES_prob_tissue

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

metrics = ['metric_dice_loss_GM', 'metric_dice_loss_GM_mask_GM_all','metric_bin_dice_loss_GM',
           'metric_far_isolated_GM_points_in_CSF',
           'metric_l1_loss_GM','metric_far_isolated_GM_points_in_WM',
           'metric_l1_loss_GM_mask_GM_PV','metric_l1_loss_GM_mask_GM_all','metric_l1_loss_GM_mask_GM_noPV']
metrics = ['metric_dice_loss_GM','metric_bin_dice_loss_GM',
           'metric_l1_loss_GM','metric_l1_loss_GM_mask_GM_all']

#metrics += ['metric_l1_loss_on_band_GM','metric_l1_loss_on_band_GM_far']

filter = dict(col='model', str='mRes')
filter = dict(col='model', str='M40')
filter = dict(col='model', str='(bin_dice)|(bin_syn)|(pve_synth)')
filter = dict(col='model', str='(pve_mResDp_)|(pve_synth)')
filter = dict(col='model', str='(synDp)|(pve_synth)')
filter = dict(col='model', str='(pve.*128$)|(low$)|(3$)|(mot$)' )
filter=None
filter = dict(col='model', str='RES14' ) #dict(col='model', str='(rZ3)|(128)' )
filter = [dict(col='model', str='ep90'), dict(col='mode', str='t1')]
plot_metric_against_GM_level(df, metrics=metrics, filter=filter, remove_max=False, add_strip=False,
                             save_fig='/home/romain.valabregue/datal/PVsynth/figure/new5/tissue_1mm_modProb_aug_bias',
                             kind='box', enlarge=True, showfliers=False,)


file1 ='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_prob_tissue/res1mm_all_mreduce.csv'
file2 ='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_prob_tissue/res14mm_all_mreduce.csv'
file3 ='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_prob_tissue/res28mm_all_mreduce.csv'

df1 = pd.read_csv(file1);df2 = pd.read_csv(file2);df3 = pd.read_csv(file3)

df1.model.replace('synthseg_model_RES1mm_pveP128','pve',inplace=True, regex=True);
df1.model.replace('synthseg_model_RES1mm_binP128','bin',inplace=True, regex=True);
df2.model.replace('synthseg_model_RES14_ep90_','',inplace=True, regex=True);
df3.model.replace('synthseg_model_RES14_ep90_','',inplace=True, regex=True);
df2['mode']='res14mm'; df3['mode']='res28mm';df1['mode']='res1mm'

df = pd.concat([df1,df2,df3],axis=0)
col = 'GM'; hue = 'model_and_SNR'; kind='boxen'
df.sort_values(['model', 'SNR'], axis=0, inplace=True)
df['model_and_SNR'] = df['model'].str.cat(df['SNR'].astype(str), sep='_')
palette = _get_color_palette(len(df['model_and_SNR'].unique()), len(df['SNR'].unique()))

for metric in metrics:
    fig = sns.catplot(x=col, y=metric, hue=hue, kind='boxen', col='mode', data=df, palette=palette, col_wrap=3)
    fig.savefig('seg_pve_bin_res' + metric + '.png')

#plot augmentation
hue_order=[ 'binP128', 'pveP128', 'pveP128_Aff', 'pveP128_Aff_low','pveP128_Aff_low_s']#['binP128', 'pveP128', 'pveP128_Aff', 'pveP128_Aff_low']
for metric in metrics:
    fig = sns.catplot(data=dfa, x='transfo', y=metric, kind='boxen', hue='model', hue_order=hue_order)
    fig.savefig('seg1mm_cmp_orig' + metric + '.png')



df = pd.read_csv(file, index_col=0)

# concat reference
df1 = pd.read_csv('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_prob_tissue/res1mm_all_mreduce.csv')
df1 = df1[(df1.model=='synthseg_model_RES1mm_pveP128') & (df1.GM!=0.3)  & (df1.SNR!=0.05)  ]
df1['model'] = 'ref_pve128';  dfa = pd.concat([df,df1])

rn = df['T_RandomLabelsToImage']
noisdic = [json.loads(r)['seed'] for r in rn]

df.model.replace('synthseg_model_RES1mm_','',inplace=True, regex=True); df.model_name = df.model
df.model_name.replace('bin_synth_mod3_results_cluster_model_ep90_it807_loss11.0000.pth.tar', 'bin', inplace=True)
df.model_name.replace('pve_synth_mod3_P128_results_cluster_model_ep90_it1008_loss11.0000.pth.tar', 'pve', inplace=True)
df.model_name.replace('pve_synth_mod3_results_cluster_model_ep90_it807_loss11.0000.pth.tar', 'pveP128', inplace=True)
df.model_name.replace('pve_synth_mod3_P128_rZ3_results_cluster_model_ep90_it1008_loss11.0000.pth.tar', 'pverZ3', inplace=True)
df.model_name.replace('bin_synth_mod3_P128_results_cluster_model_ep90_it1008_loss11.0000.pth.tar', 'binP128', inplace=True)


def split_model_name(s):
    s_list = s.split('_')
    return str.join('_',s_list[0:2])
def split_model_epoch(s):
    s_list = s.split('_')
    return int(s_list[-1][2:])
def gess_transfo(s):
    return 'affine' if 'Affine' in s else 'bias' if 'BiasFiel' in s else 'motion' if 'Motion' in s else None

df['model_name'] = df['model'].apply(lambda s: split_model_name(s))
df['epoch'] = df['model'].apply(lambda s: split_model_epoch(s))
df['transfo'] = df['transfo_order'].apply(lambda s: gess_transfo(s))

from pathlib import PosixPath
if not isinstance(df['image_filename'][0], str):  # np.isnan(df['image_filename'][0]):
    ffarg = [eval(fff)[0] for fff in df['label_filename'].values]
    ff = [eval(fff)[0].parent.parent.parent.name for fff in ffarg] #df['label_filename'].values]
    #ff = [fff.parent.parent.parent.name for fff in ffarg] #df['label_filename'].values]
else:
    ff = [eval(fff)[0].parent.name for fff in df['image_filename'].values]
df['suj_name'] = ff


#study motion severity
dfs = df[(df.transfo=='motion')&(df.model=='pveP128_Aff_low') ]
dfsg = dfs.groupby(['sample_time','suj_name']).mean()
dfsg.metric_dice_loss_GM.sort_values()

#separate different affine
mres = ModelCSVResults(df_data=df,  out_tmp="/tmp/rrr")
mres.normalize_dict_to_df('T_Affine') #, eval_func=evalfunc)
df1 = mres.normalize_dict_to_df('T_Affine', suffix='aff')
#df1.aff_degrees = df1.aff_degrees.apply(lambda x: np.array(x))
rot1 = np.vstack( df1[df1.transfo == 'affine' ].aff_degrees.values )
ind_high = rot1[:,1]>19; df.loc[ind_high,'transfo']='aff_high'

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

#read transform param and metric form train.csv
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
dfsub = df[df.transfo=='motion']; dfsub.drop_duplicates(['suj_name', 'model'], inplace=True)
df1 = mres.normalize_dict_to_df('T_MotionFromTimeCourse', suffix='mot');


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


#get sigma and BCE from log
#cat  log |grep BCE | awk '{print $13}'  >log_sigma
a=np.loadtxt('log_sigma')
from segmentation.eval_results.learning_curves import smooth
ass = smooth(a,20)
plt.figure();plt.plot(ass)
od = '/home/romain.valabregue/datal/PVsynth/training/RES_1mm_tissue_prob_loglkd/'
traindir =
res = ['pve_synth_mod3_P128_aniso_LogLkd_classif' +'/result/',
       'pve_synth_mod3_P128_aniso_LogLkd_classif_lr5_logsigmoid' +'/result/',
       'pve_synth_mod3_P128_aniso_LogLkd_classif_onlyGM' +'/result/',
       'pve_synth_mod3_P128_aniso_LogLkd_classif_onlyGM_logsigmoid' + '/result/'        ]
df=[]; col_list = ['loss', 'loss_sigma_mean', 'loss_kll_mean' ]
for rrr in res:
    print(f'reporting {rrr}')
    onedf = read_csv_in_data_frame(od + rrr)
    report_df_col(onedf, col_list, info_title=rrr)
    df.append(onedf)

mres = ModelCSVResults(df_data=onedf,  out_tmp="/tmp/rrr")
mres.normalize_dict_to_df('T_Affine') #, eval_func=evalfunc)

df1 = mres.normalize_dict_to_df('T_Affine', suffix='aff')
#df1.aff_degrees = df1.aff_degrees.apply(lambda x: np.array(x))
df1['aff_degrees'] = df1['aff_degrees'].replace(np.nan, [0,0,0])
df1['isaff']=1;  df1.loc[df1.aff_degrees.isnull(),'isaff']=0
df1.loc[df1.aff_degrees.isnull(), 'aff_degrees'] = df1.loc[df1.aff_degrees.isnull(), 'aff_degrees'].apply(
    lambda x: [0, 0, 0])
rot1 = np.vstack( df1.aff_degrees.values )
plt.scatter(df1.loss_sigma_mean,np.linalg.norm(rot1,axis=1))

dfm=mres.normalize_dict_to_df('T_MotionFromTimeCourse',suffix='m')
dfm['ismot']=1; dfm.loc[dfm.m_nufft_type.isnull(),'ismot']=0
dfm=mres.normalize_dict_to_df('transforms_metrics',suffix='tm')


def get_target_from_hist_resample(ligne, col_name):
    cell_content = ligne[col_name]
    # if np.isnan(cell_content):
    #    return 1
    if isinstance(cell_content, str):
        i1 = cell_content.find('target')
        i2 = cell_content[i1:].find(':') + i1
        i3 = cell_content[i2:].find('[') + i2
        i4 = cell_content[i3:].find(']') + i3
        target = np.array(eval(cell_content[i3:i4 + 1]))
        return target.prod()
    else:
        return 1

from pathlib import PosixPath; from utils_file import get_parent_path
def get_sujname_from_label_filename(ligne, col_name):
    cell_content = ligne[col_name]
    array_path = eval(cell_content)
    return get_parent_path(str(array_path[0]),level=4)[1]

df1['sujname'] = df1.apply(lambda s: get_sujname_from_label_filename(s, 'label_filename'), axis=1)
df1['resolution'] = df1.apply(lambda s: get_target_from_hist_resample(s, 'T_Resample'), axis=1)
