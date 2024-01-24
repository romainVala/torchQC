
import torch,numpy as np,  torchio as tio
from utils_metrics import get_tio_data_loader, predic_segmentation, load_model
from utils_file import gfile
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

scale_to_volume = 0 # 350000

device='cuda'
nb_test_transform = 0

#fsuj_csv = pd.read_csv('/data/romain/baby/suj_hcp_feta_T2_5test_suj.csv')
#fsuj_csv = pd.read_csv('/data/romain/baby/suj_hcp_feta_T2_local.csv')
#fsuj_csv = pd.read_csv('/data/romain/baby/suj_hcp_next80-160_T2_lustre.csv')
#fsuj_csv = pd.read_csv('/data/romain/baby/suj_hcp_feta_T2_local_10young_suj.csv')
fsuj_csv = pd.read_csv('/data/romain/baby/suj_hcp_10oldest_lustre.csv')
fsuj_csv = pd.read_csv('/data/romain/baby/suj_hcp_76_T1T2_local.csv')
t1_column_name = "vol_T2"; label_column_name = "label_name"; sujname_column_name = "sujname"
t1_column_name = "vol_T2_sc"; label_column_name = "label_name_sc"; sujname_column_name = "sujname"


#fsuj_csv = pd.read_csv('/data/romain/baby/suj_hcp_30oldest_lustre.csv')
#fsuj_csv = pd.read_csv('/data/romain/baby/marseille/file_5suj_GT.csv')
#fsuj_csv = pd.read_csv('/data/romain/baby/marseille/file_12suj.csv')
#t1_column_name = "srr_tpm_masked"; label_column_name = "ground_truth"; sujname_column_name = "suj"

resname = 'suj80'; #'suj_old'#'hcp10' #mar_12suj_tpm_masked_hast'#'res_Flip' #'hcp_next' #'res_Aff_suj80' #'suj_mar_from_template_tru_masked'
result_dir = '/data/romain/PVsynth/eval_cnn/baby/DATA_augm/' + resname
result_dir = '/data/romain/PVsynth/eval_cnn/baby/test_scale/' + resname

save_data = 2
#result_dir = None

model_dir = '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/jzay/training/baby/'
model_dir = '/data/romain/PVsynth/jzay/training/baby/'
models, model_name = [], []
#models.append(model_dir + 'bin_dseg9_10young/results_clusterBigAff/model_ep19_it960_loss11.0000.pth.tar');        model_name.append('10suj_BigAff_ep19')
#models.append(model_dir + 'bin_dseg9_5suj/results_cluster_affBig/model_ep54_it1000_loss11.0000.pth.tar');        model_name.append('5suj_BigAff_ep59')

#models.append(model_dir + 'bin_dseg9_5suj_hcp_T2/results_cluster_nextela/model_ep1_it640_loss11.0000.pth.tar');                     model_name.append('hcpT2_elanext_5suj_ep1')
#models.append(model_dir + 'bin_dseg9_5suj_hcp_T2/results_cluster_nextela_fromAffBig/model_ep1_it640_loss11.0000.pth.tar');          model_name.append('hcpT2_elanext_5suj_BigAff_ep1')

#models.append(model_dir + 'bin_dseg9_5suj_hcp_T2/results_cluster_nextela_fromMotion/model_ep1_it640_loss11.0000.pth.tar');          model_name.append('hcpT2_elanext_5suj_Mote30BigAff_ep1')

#models.append(model_dir + 'bin_dseg9_feta_hcp_T2/results_cluster/model_ep1_it480_loss11.0000.pth.tar');                            model_name.append('hcpT2_ep1')

#models.append(model_dir +'bin_dseg9_feta_hcp_T2/results_cluster_ep90/model_ep1_it1280_loss11.0000.pth.tar');                        model_name.append('hcpT2_ep90')
models.append(model_dir + 'bin_dseg9_5suj_midaMotion_fromdir/results_cluster_fromBigAff/model_ep30_it1000_loss11.0000.pth.tar');    model_name.append('5suj_motBigAff_ep30')
#models.append(model_dir +'bin_dseg9_feta_bg_midaMotion_fromdir/results_cluster_tio_mot1strong/model_ep18_it1020_loss11.0000.pth.tar'); model_name.append('BGmida_mot1stran_ep18')
#models.append(model_dir +'bin_dseg9_feta_bg_midaMotion_fromdir/results_cluster_tio_mot_BG0/model_ep19_it1020_loss11.0000.pth.tar'); model_name.append('BGmida_motBG0_ep19')
#models.append(model_dir +'bin_dseg9_feta_bg_midaMotion_fromdir/results_cluster_tio_5suj/model_ep10_it1000_loss11.0000.pth.tar');    model_name.append('BGmida_mot5suj_ep10')
#models.append(model_dir +''); model_name.append('')
#models.append(model_dir +''); model_name.append('')
#
models = gfile('/data/romain/PVsynth/training/bin_dseg9_15suj/results_le70_mot_from_iWM_thin_SC_ep30', 'model_ep[468]_')+gfile('/data/romain/PVsynth/training/bin_dseg9_15suj/results_le70_mot_from_iWM_thin_SC_ep30', 'model_ep[123]0_')
model_name = ['mot_from_iWM_thin_SC_ep3' + mm[90:95] for mm in models]
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
    tpreproc = tio.Compose([treshape, tscale, thot, test_transfo])
else:
    tpreproc = tio.Compose([treshape, tscale, thot])
    test_transfor_name = 'none'


tio_data = get_tio_data_loader(fsuj_csv, tpreproc, replicate=nb_test_transform ,get_dataset=False,
                               t1_column_name=t1_column_name, label_column_name=label_column_name, sujname_column_name=sujname_column_name)

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

df = pd.DataFrame()

only_first_suj = 20

for nb_model, model_path in enumerate(models):
    model = load_model(model_path, device)
    sujn = 0;
    for suj in tio_data:
        if sujn>only_first_suj:
            break
        suj_name = suj.name if isinstance(suj,tio.Subject) else suj["name"][0]
        print(f'Suj {suj_name}')

        if scale_to_volume :
            head_volume = ((suj.label.data[1:,...]>0).sum() * 0.5**3 ).numpy()

            scale_factor = (scale_to_volume/head_volume)**(1/3)
            trescale = tio.Affine(scales=scale_factor,degrees=0, translation=0)
            suj = trescale(suj)

        res_dict = {'sujname': suj_name, 'model_name':  model_name[nb_model], 'test_transfo': test_transfor_name, 'model_path': model_path}

        df = predic_segmentation(suj, model, df, res_dict, device, labels_name, selected_label=selected_label,
                                 out_dir=result_dir, save_data=save_data)
        sujn+=1
#df.to_csv(f'res_all_metric_hcp80_2model_mot.csv')

df.to_csv(f'res_{resname}.csv')





test=False
if test:
    from utils_file import get_parent_path, gfile, gdir
    ress = gfile('/data/romain/PVsynth/eval_cnn/baby/DATA_augm/res_Aff_suj80','.*')
    ress = gfile('/data/romain/PVsynth/eval_cnn/baby/DATA_augm/mar_12suj_tpm_masked_hast','.*')
    df_file = gfile(gdir(ress,'.*','.*'),'metric')
    df = pd.concat([pd.read_csv(ff) for ff in df_file])
    df = df.drop_duplicates()

    res_rd = '/data/romain/PVsynth/eval_cnn/baby/DATA_augm/'
    #res_file = res_rd + 'res_all_metric_30oldest.csv';     #df2 = pd.read_csv(res_file); df2['age'] = 'old'
    res_file = res_rd + 'res_all_metric_hcp80.csv';     df1 = pd.read_csv(res_file);       df1['age'] = 'young'
    res_file = res_rd + 'res_all_metric_5sujest_Aff.csv';    df2 = pd.read_csv(res_file);  df2['age'] = 'young'
    res_file = res_rd + 'res_all_metric__hcp_feta_T2_5test_suj_RdmAniso.csv';    df3 = pd.read_csv(res_file);  df3['age'] = 'young'

    df1 = pd.read_csv(res_rd + 'res_mars_haste_template.csv'); df1['input_data'] = 'hast'; df1.sujname = df1.sujname.apply(lambda x: x[5:]+'H')
    df2 = pd.read_csv(res_rd + 'res_mars_haste_template_masked.csv'); df2['input_data'] = 'hast_mask'; df2.sujname = df2.sujname.apply(lambda x: x[5:]+'HM')
    df1 = pd.read_csv(res_rd + 'res_suj_mar_from_template_tru_masked.csv'); df1['input_data'] = 'true_mask'; df1.sujname = df1.sujname.apply(lambda x: x[5:]+'TM')
    df = pd.concat([df1,df2])
    df = df[((df.model_name == 'hcpT2_elanext_5suj_ep1') | (df.model_name == 'hcpT2_elanext_5suj_BigAff_ep1'))]

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

