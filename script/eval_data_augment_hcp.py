
import torch,numpy as np,  torchio as tio
from utils_metrics import get_tio_data_loader, predic_segmentation, load_model
from timeit import default_timer as timer
import json, os, seaborn as sns

import pandas as pd
from nibabel.viewers import OrthoSlicer3D as ov


sns.set_style("darkgrid")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


labels_name = np.array(["bg","CSF","GM","WM","skin","vent","cereb","deepGM","bstem","hippo",])
selected_index = [1,2,3,5,6,7,8,9]
selected_label = torch.zeros(labels_name.shape); selected_label[selected_index]=1; selected_label = selected_label>0
labels_name = labels_name[selected_index]
labels_name=['GM']
selected_label = [1]

device='cuda'
nb_test_transform = 0

fsuj_csv = pd.read_csv('/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/HCPdata/file_hcp_retest.csv')

t1_column_name = "vol_T1"; label_column_name = "label_name"; sujname_column_name = "sujname"

resname = 'retest_HCP1mm_T2'
result_dir = '/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_prob_tissue/' + resname
save_data = 1
#result_dir = None

model_dir = '/home/romain.valabregue/datal/PVsynth/jzay/training/RES1mm_prob/'
models, model_name = [], []
models.append(model_dir +'pve_synth_mod3_P128_SN_clean/results_cluster/model_ep120_it1008_loss11.0000.pth.tar'); model_name.append('SNclean_ep120')
models.append(model_dir +'pve_synth_mod3_P128_mida_motion/results_cluster/model_ep40_it1000_loss11.0000.pth.tar'); model_name.append('mida_mot_ep40')
#models.append(model_dir +''); model_name.append('')
#models.append(model_dir +''); model_name.append('')
#
for mm in models:
    if not os.path.exists(mm):
        print(f'WARNING model {mm} does not exist ')

thot = tio.OneHot()
treshape = tio.EnsureShapeMultiple(2 ** 4)
tscale = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 99))

test_transfo = tio.RandomAffine(scales=0.1, degrees=180, translation=0, default_pad_label=0)
test_transfor_name = 'rdAff'
test_transfo = tio.RandomAnisotropy(axes=[0,1,2], downsampling=[3, 6])
test_transfor_name = 'rdAniso'
test_transfo = tio.RandomAffine(scales=0.1, degrees=10, translation=0, default_pad_label=0)
test_transfor_name = 'Aff10'
test_transfo = tio.RandomFlip(axes=[0, 1, 2], flip_probability=1)
test_transfor_name = 'Flip'

if nb_test_transform:
    tpreproc = tio.Compose([treshape, tscale, test_transfo])
else:
    tpreproc = tio.Compose([treshape, tscale])
    test_transfor_name = 'none'


tio_data = get_tio_data_loader(fsuj_csv, tpreproc, replicate=nb_test_transform ,
                               t1_column_name=t1_column_name, label_column_name=label_column_name, sujname_column_name=sujname_column_name)

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

df = pd.DataFrame()

for nb_model, model_path in enumerate(models):
    model = load_model(model_path, device)
    for suj in tio_data:
        suj_name = suj["name"][0]
        print(f'Suj {suj_name}')

        res_dict = {'sujname': suj_name, 'model_name':  model_name[nb_model], 'test_transfo': test_transfor_name, 'model_path': model_path}

        df = predic_segmentation(suj, model, df, res_dict, device, labels_name, selected_label=selected_label,
                                 out_dir=result_dir, save_data=save_data)

#df.to_csv(f'res_all_metric_hcp80_2model_mot.csv')

df.to_csv(f'res_{resname}.csv')





test=False
if test:
    from utils_file import get_parent_path, gfile, gdir
    ress = gfile('/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_prob_tissue/retest_HCP1mm_T2','.*')
    ress = gfile('/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/RES_prob_tissue/retest_HCP1mm','.*')
    df_file = gfile(gdir(ress,'.*','.*'),'metric')
    df = pd.concat([pd.read_csv(ff) for ff in df_file])
    #df = df.drop_duplicates()à
    dfT2.model_name = dfT2.model_name.apply(lambda x: 't2' + x)
    dfsub = df[df.sujname.apply(lambda x: isinstance(x, int))]
    dfsub = df[( df.sujname.apply(lambda x: isinstance(x, str)) & df.model_name.apply(lambda  x : 'SN' in x) )]


    ymet = []
    for k in df.keys():
        if  k.startswith('dice_') : # miss fdr   'dice' softdice
            ymet.append(k)
    ymet = ['dice_GM', 'fdr_GM', 'miss_GM']

    #dfmm = df.melt(id_vars=['sujname', 'model_name', 'test_transfo' , 'age'], value_vars=ymet, var_name='metric', value_name='y')
    #dfmm = df.melt(id_vars=['sujname', 'model_name', 'test_transfo', 'input_data' ], value_vars=ymet, var_name='metric', value_name='y')
    dfmm = df.melt(id_vars=['sujname', 'model_name', 'test_transfo' ], value_vars=ymet, var_name='metric', value_name='y')
    dfmm['one']=1
    ll = list(dfmm.sujname.unique()); ll.sort()
    fig = sns.catplot(data=dfmm, y='y', col='metric', col_wrap=4, hue='model_name', x='sujname',kind='strip',order=ll)
    fig = sns.catplot(data=dfmm, y='y', col='metric',col_wrap=4 , hue='model_name', x='sujname',kind='strip')
    fig = sns.catplot(data=dfmm, y='y', col='metric',col_wrap=4 , hue='model_name', x='one',kind='boxen')
    fig = sns.relplot(data=dfmm, y='y', x='sujname', hue='model_name', col='metric', col_wrap=3)

    from segmentation.metrics.utils import MetricOverlay
    from segmentation.losses.dice_loss import Dice
    dice_instance = Dice()
    metric_dice = dice_instance.all_dice_loss

    met_overlay = MetricOverlay(metric_dice, channels=[2], band_width=5)  # , channels=[2])
    border_mask, far_mask = met_overlay.get_border_and_far_mask(gm)

    from scipy.spatial.distance import directed_hausdorff #only 2D
    import seg_metrics.seg_metrics as sg  #ok gives same hausdorf as monai
    metrics = sg.write_metrics(labels=[1], gdth_img=gm[0,0,...].numpy(), pred_img=prediction_gm_bin[0,0,:].numpy(),
                               metrics=['dice', 'hd'], csv_file='/home/romain.valabregue/tmp/res.csv' )
    directed_hausdorff(prediction_gm_bin[0,0,:], gm[0,0,:])

    from segmentation.metrics.distance_metrics import DistanceMetric
    dist_instance = DistanceMetric()
    metric_dist1 = dist_instance.average_hausdorff_distance # much smaller values
    metric_dist2 = dist_instance.surface_distances #ok same as monai but much slower
    metric_dist1(prediction_gm_bin, gm)
    metric_dist1(prediction_gm_bin[0,0,:], gm[0,0,:])
    metric_dist2(prediction_bin, target)

    from timeit import default_timer as timer
    from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance  # warning cpu metrics

    start = timer()
    res = compute_hausdorff_distance(prediction_bin, target, percentile=95, include_background=True)
    print(f'done in {timer()-start}')

    # "miss rate",  == False Negative rate == 1 - sensitivity (TPR)  = FN/P = FN / (FN+TP)    #P nb posit -> % manquant
    # "false discovery rate = FP / (FP+TP) == 1 - precision
    # fall-out or false positive rate = FP/N  #moins pertinant car relatif au bcground size

    #eval hcp T1 T2 with coreg

    t1 = tio.ScalarImage('~/datal/PVsynth/eval_cnn/baby/eval_T1_model_fetaBgT2_hcp_ep1/sub-CC00293AN14_ses-97401/data.nii.gz')
    t2 = tio.ScalarImage('~/datal/PVsynth/eval_cnn/baby/eval_T2_model_hcpT2_elanext_5suj_ep1/sub-CC00293AN14_ses-97401/data.nii.gz')
    lab = tio.LabelMap('~/datal/PVsynth/eval_cnn/baby/eval_T1_model_fetaBgT2_hcp_ep1/sub-CC00293AN14_ses-97401/label.nii.gz')
    pred = tio.LabelMap('~/datal/PVsynth/eval_cnn/baby/eval_T1_model_fetaBgT2_hcp_ep1/sub-CC00293AN14_ses-97401/prediction.nii.gz')
    suj = tio.Subject(dict(t1=t1, t2=t2, lab=lab, pred=pred))
    #dice = metric_dice(suj.lab.data.unsqueeze(0), suj.pred.data.unsqueeze(0))

    suj.add_image(tio.ScalarImage(tensor=suj.pred.data[2].unsqueeze(0), affine=suj.pred.affine), 'predGM')
    suj.add_image(tio.ScalarImage(tensor=suj.lab.data[2].unsqueeze(0), affine=suj.pred.affine), 'labGM')

    tcoreg = tio.Coregister(target='predGM', default_parameter_map='affine', estimation_mapping={'labGM': ['labGM', 'lab']})
    tcoreg = tio.Coregister(target='predGM', default_parameter_map='affine', estimation_mapping={'labGM': ['lab']})
    sujcoreg = tcoreg(suj)

    dice = metric_dice(suj.pred.data.unsqueeze(0), sujcoreg.lab.data.unsqueeze(0))

    #cool dice loss decrease from 0.136 to 0.801

