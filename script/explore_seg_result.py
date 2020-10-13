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

from segmentation.eval_results.compare_results import plot_value_vs_GM_level, aggregate_csv_files, plot_metric_against_GM_level
import glob
from segmentation.eval_results.learning_curves import  report_learning_curves
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt

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


 #ration volume label 1mm 1.4mm 5.177/5.165





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


#explore PV proportion
fpv1 = gfile('/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/513130/T1w/ROI_PVE_1mm','^GM')
fpv14 = gfile('/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/513130/T1w/ROI_PVE_14mm','^GM')
seuils = [ [0, 1], [0.01, 0.99], [0.05, 0.95], [0.1, 0.9] ]
seuil = [0, 1]
for seuil in seuils:
    print(seuil)
    for ff, nn, res in zip( (fpv1 + fpv14), ['1mm', '1.4mm'], [1,1.4] ) :
        img = nb.load(ff)
        data = img.get_fdata(dtype=np.float32)
        fig = plt.figure(nn)
        dd = data[ (data>seuil[0]) * (data<seuil[1])]
        nGM = np.sum(data>=seuil[1])
        nPV = len(dd)
        volGM = np.sum(data>=0.5)
        volPV = np.sum(data)
        print('{} : pure GM {} PV {}  PV/GM {}  Vol GM vox {}  mm {} pvmm{} '.format(nn, nGM, nPV, nPV/nGM, volGM,
                                                                              volGM*res*res*res, volPV*res*res*res ))
        hh = plt.hist(dd, bins=500)


#concat eval.csv
res = '/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_14mm_tissue/dataS*/*/eval.csv'
file = '/home/romain.valabregue/datal/PVsynth/eval_cnn/res14mm_tissue_all.csv'
res = '/home/romain.valabregue/datal/PVsynth/jzay/eval/eval_cnn/RES_14mm/data_G*/*/eval.csv'
file = '/home/romain.valabregue/datal/PVsynth/jzay/eval/eval_cnn/res14mm_all.csv'
res = '/home/romain.valabregue/datal/PVsynth/jzay/eval/eval_cnn/RES_1mm_tissue/data_S*/*/eval.csv'
res = '/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_1mm_tissue/dataS*/*/eval.csv'
file = '/home/romain.valabregue/datal/PVsynth/eval_cnn/res1mm_tissue_all.csv'

aggregate_csv_files(res, file, fragment_position=-3)

file = '/home/romain.valabregue/datal/PVsynth/jzay/eval/eval_cnn/res1mm_tissue_all.csv'
file = '/home/romain.valabregue/datal/PVsynth/jzay/eval/eval_cnn/res14mm_all.csv'
metrics = ['metric_dice_loss_GM', 'metric_bbin_dice_loss_GM', 'metric_bin_dice_loss_GM',
           'metric_l1_loss_GM', 'metric_l1_loss_GM_mask_GM', 'metric_l1_loss_GM_mask_NO_GM',
           'metric_bin_volume_ratio_GM','metric_volume_ratio_GM']

#metrics += ['metric_l1_loss_on_band_GM','metric_l1_loss_on_band_GM_far']

filter = dict(col='model', str='M90')

plot_metric_against_GM_level(file, metrics=metrics, filter=filter, kind='boxen', enlarge=True,
                             save_fig='/home/romain.valabregue/datal/PVsynth/figure/new2/tissue_1mm_M90')

import pandas as pd
import json
df = pd.read_csv(file, index_col=0)
rn = df['T_RandomLabelsToImage']
noisdic = [json.loads(r)['seed'] for r in rn]

