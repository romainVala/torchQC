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

from utils_file import get_parent_path, gfile, gdir
import glob, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import glob
from segmentation.eval_results.learning_curves import  report_learning_curves, read_csv_in_data_frame, report_df_col
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt, pandas as pd
#manual ploting
import seaborn as sns
import pandas as pd
import json
import torch
import torchio as tio

sns.set_style("darkgrid")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def mod_name(x):
    str_split = x.split('_')
    res = f'{str_split[1]}_{str_split[3][4:]}_{str_split[4]}'
    return res
def ep_name(x):
    str_split = x.split('_')
    res =  str_split[-1]
    return res

def read_feta_eval(ress):
    df_all = []
    for res in ress:
        resname = get_parent_path(res)[1]
        df, suj_num = [],  0
        for i in range(len(df_feta)):
            dfsl = pd.DataFrame(df_feta.iloc[i, :]).T
            dfsl.index = [0]
            one_suj = gdir(res, f'{dfsl.participant_id[0]}')
            if len(one_suj) == 1:
                # print(f'found {get_parent_path(one_suj)[1]} for {df_feta.suj[i]} and {df_feta.session_id[i]}')
                fcsv = gfile(one_suj, 'csv')
                dfone = pd.read_csv(fcsv[0], index_col=0)
                dfone['suj_num'] = suj_num
                suj_num += 1
                for k in dfsl.keys():
                    dfone[k] = dfsl[k]
                # dd = pd.concat([dfone , dfsl],ignore_index=True)  #ignore_index do not work
                df.append(dfone)
            else:
                if len(one_suj) > 0:
                    print(f'double for {i}')
                else:
                    print(f'skipin {i} {dfsl.participant_id[0]}')
        df = pd.concat(df)
        df['model'] = resname
        print(f'model {resname} shape is {df.shape}')
        df_all.append(df)
    dfm = pd.concat(df_all)
    dfm["model_name"], dfm["ep_name"] = dfm.model.apply(lambda x: mod_name(x)), dfm.model.apply(lambda x: ep_name(x))
    dfm["scan_age"] = dfm['Gestational age']

    return dfm

def read_hcp_eval(ress):
    df_all = []
    for res in ress:
        resname = get_parent_path(res)[1]
        suj, ind_skip, df, suj_num = [], [], [], 0
        for i in range(len(df_hcp)):
            dfsl = pd.DataFrame(df_hcp.iloc[i, :]).T
            dfsl.index = [0]
            sesstr = f'{dfsl.session_id[0]}'[:4]
            one_suj = gdir(res, f'{dfsl.suj[0]}.*{sesstr}')
            if len(one_suj) == 1:
                suj.append(one_suj[0])
                # print(f'found {get_parent_path(one_suj)[1]} for {df_hcp.suj[i]} and {df_hcp.session_id[i]}')
                fcsv = gfile(one_suj, 'csv')
                dfone = pd.read_csv(fcsv[0], index_col=0)
                dfone['suj_num'] = suj_num
                finput = gfile(one_suj,'data.nii.gz')
                fpred = gfile(one_suj,'prediction.nii.gz')
                flabel = gfile(one_suj,'label.nii.gz')
                dfone['finput'],dfone['fpred'],dfone['flabel'] = finput, fpred, flabel
                suj_num += 1
                for k in dfsl.keys():
                    dfone[k] = dfsl[k]
                # dd = pd.concat([dfone , dfsl],ignore_index=True)  #ignore_index do not work
                df.append(dfone)
            else:
                if len(one_suj) > 0:
                    print(f'double for {i}')
                else:
                    print(f'skipin {i} {dfsl.suj[0]}')
                    ind_skip.append(i)
        df = pd.concat(df)
        df['model'] = resname
        print(f'model {resname} shape is {df.shape}')
        df_all.append(df)
    dfm = pd.concat(df_all)
    dfm["model_name"], dfm["ep_name"] = dfm.model.apply(lambda x: mod_name(x)), dfm.model.apply(lambda x: ep_name(x))
    return dfm

rootdir = '/data/romain/baby/'
rootdir = '/network/lustre/iss02/opendata/data/baby/devHCP/'
df_hcp_all = pd.read_csv(rootdir+'rel3_dhcp_anat_pipeline/all_seesion_info_order.csv', index_col=0)
df_hcp_all["sujnum"] = range(len(df_hcp_all))
df_hcp = df_hcp_all[:80]
df_feta = pd.read_csv('/data/romain/baby/feta_2.1/participants.tsv', sep='\t')

dfp = df_feta[df_feta.Pathology=='Pathological']; dfn = df_feta[df_feta.Pathology=='Neurotypical']

#hcp stats
df1 = df_hcp;
y =  "head_circumference_scan"; x = "scan_age"; #scan_number";
hue ="radiology_score" ;col = 'sedation';
cmap = sns.color_palette(n_colors=len(df1[hue].unique())) ; #"coolwarm",
fig = sns.relplot(data=df1,y=y,x=x, hue = hue, palette=cmap) #, kind="line")

ress=gdir('/data/romain/PVsynth/eval_cnn/baby','_T2_.*'); ress = ress[1:-1]

ress=gdir('/data/romain/PVsynth/eval_cnn/baby','eval_T2.*fetaBgT2.*_hcp_');ress = ress[:-1]
ress=gdir('/data/romain/PVsynth/eval_cnn/baby/feta','fetaBgT2.*ep[123]')
ress=[#'/data/romain/PVsynth/eval_cnn/baby/bad/eval_T1_model_fetaBgShape_ep120' ,
      '/data/romain/PVsynth/eval_cnn/baby/bad/eval_T2_model_fetaBgT2_wmsh_ep50',
      #'/data/romain/PVsynth/eval_cnn/baby/bad/eval_T1_model_fetaBgT1_wmsh_ep60',
      #'/data/romain/PVsynth/eval_cnn/baby/bad/eval_T1_model_fetaBgT2_wmsh_ep50',
      #'/data/romain/PVsynth/eval_cnn/baby/bad/eval_T1_model_fetaBgT1T2_wmcl_ep40',
      #'/data/romain/PVsynth/eval_cnn/baby/bad/eval_T1_model_fetaBgT1T2_wmsh_ep40',
    '/data/romain/PVsynth/eval_cnn/baby/eval_T2_model_fetaBgT2_hcp_ep1',
    #'/data/romain/PVsynth/eval_cnn/baby/eval_T2_model_fetallhcp_ep01',
    #'/data/romain/PVsynth/eval_cnn/baby/eval_T2_model_fetaBgT2_hcpNex_ep10',
    ]
#ress.append('/data/romain/PVsynth/eval_cnn/baby/eval_T2_model_fetaBgT2mP192ep30_hcpNex_ep1')
ress = []
ress.append('/data/romain/PVsynth/eval_cnn/baby/eval_T2_model_fetaBgT2_hcp_ep1',)

ress.append('/data/romain/PVsynth/eval_cnn/baby/eval_T2_model_fetaBgT2_hcpNex_ep1')
ress.append('/data/romain/PVsynth/eval_cnn/baby/eval_T2_model_fetallhcpElas_ep01')
ress.append('/data/romain/PVsynth/eval_cnn/baby/eval_T2_model_fetaBgT2ep90_hcp_ep1')
ress.append('/data/romain/PVsynth/eval_cnn/baby/eval_T2_model_fetaBgmidaWmclean_ep20')
ress.append('/data/romain/PVsynth/eval_cnn/baby/eval_T2_model_fetaBgmidaWmsh_ep40')
ress.append('/data/romain/PVsynth/eval_cnn/baby/eval_T2_model_fetaBgmidaMot_ep17')
ress.append('/data/romain/PVsynth/eval_cnn/baby/eval_T1_model_fetaBgT2_hcp_ep1')
ress.append('/data/romain/PVsynth/eval_cnn/baby/feta/eval_feta_model_fetallhcp_ep01')
rd = '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/baby/'
#rd = '/data/romain/PVsynth/eval_cnn/baby//'
ress=[];

ress.append(rd+'eval_T2_model_fetaBgT2_hcp_ep1')
ress.append(rd+'eval_T2_model_fetaBgmidaMotScale_ep8')
ress.append(rd+'eval_T2_model_fetaBgmidaMot_ep17')
ress.append(rd+'eval_T2_model_midaMot1strong_ep18')
ress.append(rd+'eval_T2_model_midaMot5suj_ep10')
ress.append(rd+'eval_T2_model_midaMot5suj_ep30')
ress.append(rd+'')
ress.append(rd+'eval_T2old_model_fetaBgT2_hcp_ep1')
#ress.append(rd+'eval_T2old_model_fetallhcp_ep01')
ress.append(rd+'eval_T2old_model_midaMot5suj_ep30')
ress.append(rd+'eval_T2_model_affBig_5suj_ep54')
ress.append(rd+'eval_T2_model_hcpT2_elanext_5suj_ep1')
ress.append(rd+'eval_T2_model_hcpT2_elanext_5suj_BigAff_ep1')
ress.append(rd+'eval_T2_model_hcpT2_elanext_5suj_ep30')
ress.append(rd+"eval_T2_model_5suj_mot_ep26")
ress.append(rd+"eval_T2_model_5suj_motBigAff_ep30")

ymet=[]
for k in dfm.keys():
    if "metric_" in k:
        ymet.append(k)
ymet = ymet[:3]

dfm = read_feta_eval(ress);
dfmm = dfm.melt(id_vars=['scan_age', 'model_name','ep_name', 'sujnum','model','Pathology','Gestational age'], value_vars=ymet, var_name='metric', value_name='y')

dfmh = read_hcp_eval(ress)
dfmm = dfmh.melt(id_vars=['scan_age', 'model_name','ep_name', 'sujnum','model','radiology_score'], value_vars=ymet, var_name='metric', value_name='y')
#fig = sns.catplot(data=dfmm, y='y', x='model_name', hue='ep_name', col='metric', kind='boxen',col_wrap=3)
fig = sns.relplot(data=dfmm, y='y', x='sujnum', hue='model', col='metric',col_wrap=3)

fig = sns.relplot(data=dfmm, y='y', x='Gestational age', hue='Pathology', col='metric',col_wrap=3)

fig = sns.relplot(data=dfm, x='scan_age', y='metric_dice_loss_GM', hue="model")
fig = sns.catplot(data=dfm, y='metric_dice_loss_GM', x='model_name', hue='ep_name', kind='boxen',)


yk = 'metric_mean_dice_loss'
sns.relplot(data=dfm, y=yk, x='sujnum', hue='model')
mrview_from_df(dfmh, 'suj_num','5')
def mrview_from_df(df, col_name, condition):
    dfsub = df[df[col_name]==condition]
    for dfser in dfsub.iterrows():
        dfser = dfser[1]
        #print(f'{dfser.metric_dice_loss_GM:.2} GM dicm from {dfser.model} Predction {dfser.fpred} ')
        print(f'{dfser.metric_dice_loss_GM:.2} GM dicm from {dfser.model} ')

    return mrview_overlay(list(dfsub.finput.values), [dfsub.flabel.values[0]] + list(dfsub.fpred.values) )

def mrview_overlay(bg_img, overlay_list):
    if not isinstance(bg_img, list):
        bg_img = [bg_img]
    if not isinstance(overlay_list, list):
        overlay_list = [overlay_list]
    col_overlay = [ '0,1,0', '1,0,0', '0,0,1', '1,1,0', '0,1,1', '1,0,1', '1,0.5,0', '0.5,1,0']
    mrviewopt = [
        f'-overlay.opacity 0.6 -overlay.colour {col_overlay[k]} -overlay.intensity 0,1   -overlay.threshold_min 0.5  -overlay.interpolation 0 -mode 2'
        for k in range(len(overlay_list)) ]

    cmd = 'vglrun mrview '
    for img in bg_img:
        cmd += (f' {img} ')
    for nb_over, img in enumerate(overlay_list):
        cmd += (f' -overlay.load {img} {mrviewopt[nb_over]} ')
    print(f'{cmd} &')
    return cmd


def get_anat_path(df,rd='/data/romain/baby/rel3_dhcp_anat_pipeline/', dataset='devhcp'):
    if dataset=="devhcp":
        return   os.path.join(rd,df.suj,f'ses-{df.session_id}','anat')
    elif (dataset=="feta2"):
        return  os.path.join(rd,df.suj, 'anat')
def get_suj_id(df, dataset='devhcp'):
    if dataset=="devhcp":
        return f'{df.suj}_ses-{df.session_id}'
    elif (dataset=="feta2"):
        return f'{df.suj}'

def get_file_path(df,dir_file, file_reg,change_root=False):
    file = glob.glob(f'{df[dir_file]}/{file_reg}')
    if change_root:
        #ff = file[0].replace('/data/romain/baby/', '/gpfswork/rech/ezy/urd29wg/data/devHCP/')
        ff = file[0].replace('/data/romain/baby/', '/gpfswork/rech/ezy/urd29wg/data/devHCP/')
    else:
        ff = file[0]
    return ff
def load_torchio_data(df,colname):
    return tio.Subject({"t1":tio.ScalarImage(df[colname])})
def get_volume(df,colname):
    data = df['sujtio'].t1.data
    voxel_size = np.array( df['sujtio'].t1.spacing).prod()
    return (data>0).sum().numpy()  * voxel_size


#dhcp to _csv data json sel
df = df_hcp_all.copy()
#df['anat_path'] = df.apply(lambda r: get_anat_path(r, rd="/data/romain/baby/devHCP/rel3_dhcp_anat_pipeline_sub_small/"), axis=1 )
df['anat_path'] = df.apply(lambda r: get_anat_path(r, rd="/network/lustre/iss02/opendata/data/baby/devHCP/rel3_dhcp_anat_pipeline/"), axis=1 )
df["label_name"] = df.apply(lambda r: get_file_path(r,"anat_path","s*drawem9_dseg*gz"), axis=1 )
df["sujtio"] = df.apply(lambda r: load_torchio_data(r,"label_name"), axis=1 )
df["volume"] = df.apply(lambda r: get_volume(r,"sujtio"), axis=1 )
fig = sns.relplot(data=df,y="volume",x=range(80))

df = df_feta.copy()
df["suj"] = df.participant_id
df["sujname"] = df.participant_id
df['anat_path'] = df.apply(lambda r: get_anat_path(r, rd="/data/romain/baby/feta_2.1/", dataset='feta2'), axis=1 )
df["vol_name"] = df.apply(lambda r: get_file_path(r,"anat_path","*T2*gz"), axis=1 )
df["label_name"] = df.apply(lambda r: get_file_path(r,"anat_path","*dseg*gz"), axis=1 )
df["sujtio"] = df.apply(lambda r: load_torchio_data(r,"label_name"), axis=1 )
df["volume"] = df.apply(lambda r: get_volume(r,"sujtio"), axis=1 )

dfs = df.sort_values(by='Gestational age')
fig = sns.relplot(data=dfs,y="volume",x=range(80), hue='Pathology')
fig.axes[0][0].invert_yaxis()
gb = dfs.groupby(['Pathology'])
gb.agg({'volume' : np.max})


dfp = df[df.Pathology=='Pathological']; dfn = df[df.Pathology=='Neurotypical']
dfp.iloc[:15].to_csv('/tmp/suj_feta_15x2_jzay.csv',index=False)
dfp.iloc[:15].to_csv('/tmp/suj_feta_15x2_jzay.csv',index=False)
df.to_csv('/tmp/s.csv', index=False)

df_file=pd.DataFrame()
df_file["sujname"] = df.apply(lambda r: get_suj_id(r, dataset='feta2'), axis=1 )
df_file["sujname"] = df.apply(lambda r: get_suj_id(r), axis=1 )

df_file["vol_name"] = df.apply(lambda r: get_file_path(r,"anat_path","*T2*gz"), axis=1 )
df_file["label_name"] = df.apply(lambda r: get_file_path(r,"anat_path","s.*drawem9_.*gz"), axis=1 )

df_file.iloc[80:160,:].to_csv('/tmp/suj_hcp_next80-160_T2_jzay.csv',index=False)
#df.iloc[[1,3,5],[1,3]]

#some display
cmd = 'mrview';
for pi in df_feta[df_feta.Pathology == 'Pathological'].sort_values(by=['Gestational age']).participant_id:
    cmd.join(f' {pi}/anat/*T2*gz ')
cmd
df_hcp.groupby(['radiology_score']).count()
cmd = 'mrview';
for ind, pi in df_hcp[df_hcp.radiology_score == 1].sort_values(by=['scan_age']).iterrows():
    #print(pi)
    cmd += f' {pi.suj}/ses-{pi.session_id}/anat/*T2*gz '
#redoo transfrom
np.random.seed(12);torch.manual_seed(12)
for i in range(5):
    suj = viz_set[0]
    suj.t1.save(f'data_rdLabel_{i}.nii')
np.random.seed(12);torch.manual_seed(12)
suj0 = viz_set[0]
#affine
ta = tio.RandomAffine(scales=0.1, degrees=20, translation=10)
for i in range(5):
    suj = ta(suj0)
    suj.t1.save(f'data_rdAff_{i}.nii')
#elas
ta = tio.RandomElasticDeformation()
for i in range(5):
    suj = ta(suj0)
    suj.t1.save(f'data_rdElas_{i}.nii')
#aniso
ta = tio.RandomAnisotropy(axes=[0,1,2], downsampling=[3, 6])
for i in range(5):
    suj = ta(suj0)
    suj.t1.save(f'data_rdAniso_{i}.nii')
#Bias
ta = tio.RandomBiasField()
ta = tio.Compose([tio.RandomBiasField(), tio.RescaleIntensity(out_min_max=[0, 1], percentiles=[1, 99])])
for i in range(5):
    suj = ta(suj0)
    suj.t1.save(f'data_rdBias_{i}.nii')
    print(suj.history[-1])
#Noise
ta = tio.RandomNoise(mean=0, std=[0.01, 0.1])
for i in range(5):
    suj = ta(suj0)
    suj.t1.save(f'data_rdNoise_{i}.nii')
    print(suj.history[-1])


#local dice test
rd = '/data/romain/PVsynth/eval_cnn/baby/bad/eval_T2_model_fetaBgT2_wmsh_ep50/sub-CC00389XX19_ses-11910/'
from segmentation.losses.dice_loss import Dice
from segmentation.metrics.utils import MetricOverlay

dice_instance = Dice()
metric_dice = dice_instance.mean_dice_loss
met_overlay = MetricOverlay(metric_dice, channels=[2])

itar = tio.ScalarImage(rd+'label.nii.gz')
ipred = tio.ScalarImage(rd+'prediction.nii.gz')
idata = tio.ScalarImage(rd+'data.nii.gz')
suj = tio.Subject({'target':itar, "prediction":ipred, 'data':idata})

res = met_overlay(suj.prediction.data.unsqueeze(0), suj.target.data.unsqueeze(0))

sampler = tio.GridSampler(subject=suj, patch_size=12)#48)
grid_ag = tio.data.GridAggregator(sampler)

res_p=[]
for i, patch in enumerate(sampler(suj)):
    #patch.data.save(f'patch_{i}.nii.gz')
    res_patch = met_overlay(patch.prediction.data.unsqueeze(0), patch.target.data.unsqueeze(0))
    patch_tensor_dim = patch.data.data.unsqueeze(0).shape
    patch_tensor = torch.ones(patch_tensor_dim)*res_patch
    res_p.append(res_patch)
    grid_ag.add_batch(patch_tensor,patch.location.unsqueeze(0))

suj.data.data = grid_ag.get_output_tensor()
res_p = np.array(res_p)
plt.plot(res_p,'x')

loss = torch.nn.MSELoss(reduction='none')
mse=loss(suj.prediction.data[2],suj.target.data[2])
mse_inter = (mse>0.5) * suj.target.data[2]
mse_ext = (mse>0.5) * (suj.target.data[2]==0)
mse_ext.sum()/suj.target.data[2].sum() * 100
mse_inter.sum()/suj.target.data[2].sum() * 100

suj.data.data[0] = mse_inter


#test coregister
img_tpl = tio.ScalarImage('/data/romain/template/MIDA_v1.0/MIDA_v1_voxels/mask/crop/mask_brain.nii')
img_tar = tio.ScalarImage('/data/romain/baby/devHCP/rel3_dhcp_anat_pipeline_sub_small/sub-CC00867XX18/ses-37111/anat/mask.nii')

suj = tio.Subject({'target':img_tar, 'mask_tpl':img_tpl})
tcoreg = tio.Coregister(target='target', estimation_mapping='mask_tpl',default_parameter_map='affine' )
sujt = tcoreg(suj)

from util_affine import ElastixRegister
import SimpleITK as sitk

out_aff, elastix_trf = ElastixRegister(suj.mask_tpl, suj.target, type='affine')
sitk.WriteImage(elastix_trf.GetResultImage(),'res.nii')

import torchio as tio
import numpy as np

colin = tio.datasets.Colin27()
colin.load()
del colin['brain'], colin['head']
tres=tio.Resize(target_shape=[64, 65, 64]) #64,64,64])181, 217, 181
colin = tres(colin)

fpars_ones = np.ones((6, 200))

trsfm_z = tio.RandomMotionFromTimeCourse(oversampling_pct=0.0, displacement_shift_strategy="center_zero")
trsfm_z = tio.RandomMotionFromTimeCourse(oversampling_pct=0.0, displacement_shift_strategy="center_zero")

trsfm_z.simulate_displacement = False
trsfm_z.euler_motion_params = fpars_ones

trsfmed_z = trsfm_z(colin)
hmot = trsfmed_z.history[1]
print(hmot.euler_motion_params['t1'])

def display_res(dir_pred, bg_files):

    cmd = []
    for nb_pred, one_dir_pred in enumerate(dir_pred):
        # one_dir_pred = dir_pred[0]
        all_file = gfile(one_dir_pred, 'nii')
        print(f'working on {one_dir_pred}')
        for ii, one_pred in enumerate(all_file):
            gt_file, sr_file = gt_files[ii], sr_files[ii]
            if nb_pred == 0:
                cmd.append( [one_pred])
            else:
                cmd[ii].append(one_pred)
    mrview_cmd=[ mrview_overlay(bg_files[kk], cmd[kk]) for kk in range(len(cmd))]
    return mrview_cmd


#ground truth marseille
suj_gt = ['sub-0307_ses-0369', 'sub-0427_ses-0517', 'sub-0457_ses-0549', 'sub-0483_ses-0589', 'sub-0567_ses-0681','sub-0665_ses-0791'] #'sub-0179_ses-0212',
dfdata = pd.read_csv('/data/romain/baby/marseille/file_6suj_GT.csv')
dir_pred = glob.glob('/data/romain/baby/marseille/fetal/derivatives/segmentation/prediction_template_space/*')
dir_pred = glob.glob('/data/romain/baby/marseille/fetal/derivatives/segmentation/prediction_template_masked//*')
 dir_pred = [dir_pred[i] for i in (1,2,3,6,7)]

pred_name = get_parent_path(dir_pred)[1]
gt_files, sr_files = dfdata.ground_truth.values, dfdata.srr_tpm
labels = ["bg","CSF","GM","WM","skin","vent","cereb","deepGM","bstem","hippo",]

display_res(dir_pred, sr_files)

from segmentation.losses.dice_loss import Dice
from segmentation.metrics.utils import MetricOverlay
import torchio as tio

dice_instance = Dice()
metric_dice = dice_instance.all_dice_loss
met_overlay = MetricOverlay(metric_dice) #, channels=[2])
labels = ["bg","CSF","GM","WM","skin","vent","cereb","deepGM","bstem","hippo",]
col_overlay =['1,0,0','0,1,0','0,0,1','1,1,0','0,1,1']
mrviewopt = [f'-overlay.opacity 0.6 -overlay.colour {col_overlay[k]} -overlay.intensity 0,1   -overlay.threshold_min 0.5  -overlay.interpolation 0 -mode 2'
             for k in range(len(dir_pred))

df, cmd = pd.DataFrame(), []
for nb_pred, one_dir_pred in enumerate(dir_pred):
    #one_dir_pred = dir_pred[0]
    all_file = gfile(one_dir_pred,'nii')
    print(f'working on {one_dir_pred}')
    for ii,  one_pred in enumerate(all_file):
        gt_file, sr_file = gt_files[ii], sr_files[ii]
        if nb_pred==0:
            cmd.append( f'vglrun mrview {sr_file} ')
        #print(f'vglrun mrview {gt_file} -overlay.load {gt_file} {mrviewopt}&')
        cmd[ii] = f'{cmd[ii]} -overlay.load {one_pred} {mrviewopt[nb_pred]} '

    print(f'{cmd} &')

        [pp, sujname] = get_parent_path(one_pred, remove_ext=True)
        model = get_parent_path(pp,level=-2)[1]

        suj = tio.Subject({'target':tio.LabelMap(gt_file), "prediction":tio.ScalarImage(one_pred)})
        thot = tio.OneHot(include='target')
        suj = thot(suj)

        res = met_overlay(suj.prediction.data.unsqueeze(0), suj.target.data.unsqueeze(0))
        res = [float(rr.numpy()) for rr in res]
        resdic = {k:res for k,res in zip(labels,res)}
        resdic['suj']  = sujname; resdic['model'] = model; resdic['sujnum'] = ii
        dfone = pd.DataFrame(resdic,index={0})
        df = pd.concat([df, dfone])

plt.interactive(True)
sns.relplot(data=df,x='suj',y='GM',hue='model')

