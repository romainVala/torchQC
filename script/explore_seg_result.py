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

metric = ['dice_loss', 'bin_dice_loss',
          'dice_loss_GM_mask',  'dice_loss_GM_mask_1',  'dice_loss_GM_mask_NO',  'dice_loss_GM_mask_PV',
          'l1_loss', 'l1_loss_GM_mask', 'l1_loss_GM_mask_1',  'l1_loss_GM_mask_NO',  'l1_loss_GM_mask_PV',
          'far_isolated_GM_points_in_CSF', 'far_isolated_GM_points_in_WM',  'average_hausdorff_distance',
          'volume_ratio',
#          'predicted_volume_in_mm3', 'label_volume_in_mm3',
          ]
resfig = '/home/romain.valabregue/datal/PVsynth/figure/res1mm_on_bin/'
resolution = '1mm'
for m in metric:
    figname = resfig + m + '_' + resolution
    label = 'GM'
    if 'far_isol' in m:
        ylim = (0, 100000)
        label =''
    elif 'hausdorff' in m:
        ylim = None
    elif 'dice' in m:
        ylim = (0, 0.02) if 'mask' in m else (0, 0.15) #
        ylim = (0, 0.15)
    elif 'l1' in m:
        ylim = (0, 0.005) if 'mask' in m else (0, 0.03)
        ylim = (0, 0.025)
    elif 'ratio' in m:
        ylim = (0.9, 1.1)

    try:
        plot_value_vs_GM_level(results_dirs, m, ylim=ylim, save_fig=figname, label=label)
    except Exception as e:
        print(e)


m='dice_loss_GM_mask_NO'
m='volume_ratio'
plot_value_vs_GM_level(results_dirs,m, ylim=(0.98, 1.02),
                       save_fig=resfig + m + '_' + resolution+'_zoom')



plot_value_vs_GM_level(results_dirs, 'l1_loss_GM_mask', None,
                       save_fig='/home/romain.valabregue/datal/PVsynth/figure/test')

 #ration volume label 1mm 1.4mm 5.177/5.165





#Explore training curve
res='/home/fabien.girka/data/segmentation_tasks/RES_1.4mm/bin_synth_data_64_common_noise_no_gamma/results_cluster'
res = gdir('/home/fabien.girka/data/segmentation_tasks/RES_1mm/','data_64_common_noise_no_gamma')
res = gdir('/home/romain.valabregue/datal/PVsynth/training','pve')
res = gdir('/home/romain.valabregue/datal/PVsynth/jzay/training/RES28mm','data')
res = gdir('/home/romain.valabregue/datal/PVsynth/jzay/training/RES1mm','data')

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
res = '/home/romain.valabregue/datal/PVsynth/jzay/eval/eval_cnn/RES_14mm/data_G*/*/eval.csv'
file = '/home/romain.valabregue/datal/PVsynth/jzay/eval/eval_cnn/res14mm_all.csv'
aggregate_csv_files(res, file, fragment_position=-3)

file = '/home/romain.valabregue/datal/PVsynth/jzay/eval/eval_cnn/res1mm_all.csv'
file = '/home/romain.valabregue/datal/PVsynth/jzay/eval/eval_cnn/res14mm_all.csv'
metrics = ['metric_dice_loss_GM', 'metric_bbin_dice_loss_GM', 'metric_bin_dice_loss_GM',
           'metric_l1_loss_GM', 'metric_l1_loss_GM_mask_GM', 'metric_l1_loss_GM_mask_NO_GM',
           'metric_bin_volume_ratio_GM','metric_volume_ratio_GM']


plot_metric_against_GM_level(file, metrics=metrics, save_fig='/home/romain.valabregue/datal/PVsynth/figure/new/ttt')

