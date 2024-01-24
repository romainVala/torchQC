
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


labels_name = np.array(["bg","CSF","GM","WM","skin","vent","cereb","deepGM","bstem","hippo",])
selected_index = [1,2,3,5,6,7,8,9]
selected_label = torch.zeros(labels_name.shape); selected_label[selected_index]=1; selected_label = selected_label>0
labels_name = labels_name[selected_index]

scale_to_volume = 0 # 350000
device='cuda'

fsuj_csv = pd.read_csv('/data/romain/baby/suj_hcp_PV_704.csv')
t1_column_name = "vol"; label_column_name = "label"; sujname_column_name = "sujname"

resname = '704_sujPV'; #'suj_old'#'hcp10' #mar_12suj_tpm_masked_hast'#'res_Flip' #'hcp_next' #'res_Aff_suj80' #'suj_mar_from_template_tru_masked'
result_dir = '/data/romain/PVsynth/eval_cnn/baby/Article/pve/' + resname
save_data = 2

thot = tio.OneHot()

tpreproc = tio.Compose([thot])
test_transfor_name = 'none'

fv1 = fsuj_csv.vol_label
fv2 = fsuj_csv.binPV


if not os.path.exists(result_dir):
    os.mkdir(result_dir)



#again
df_label = pd.read_csv('/network/lustre/iss02/opendata/data/template/MIDA_v1.0/MIDA_v1_voxels/new_label_v3.csv')
labels_name = df_label.Name.values[1:13] ; sel_label = df_label.targ.values[1:13] ;
#labels_name = df_label.Name.values[np.r_[1:4,6:13]] ; sel_label = df_label.targ.values[np.r_[1:4,6:13]] ;

suj =gdir('/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/session1/',['\d.*','T1w'])
f1 = gfile(suj,'remap_aparc.aseg.nii.gz')
f2 = gfile(gdir(suj,'ROI$'), 'bin_PV')
sujname = get_parent_path(f1,3)[1]

df = compute_metric_from_list(f1, f2, sujname, labels_name, sel_label)

df.to_csv('test_retest_aseg_versus_PV.csv')



#adult hcp different predictions
dt1= gdir('/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/session1',['\d.*','T1w','T1_1mm'])
droi= gdir('/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/session1',['\d.*','T1w','ROI_PVE_1mm'])
droi= gdir('/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/session1',['\d.*','T1w','ROI$'])
dfs =  gdir('/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/session1',['\d.*','T1w','pred_fs','[1234567890]'])
dfree =  gdir('/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/session1',['\d.*','T1w','freesurfer','suj','mri'])
ft1 = gfile(dt1,'^T1w_1mm.nii$');
flab = gfile(droi, '^bin_PV'); flab2 = gfile(dfs, '^rstd_remap_ap') # flab2 = gfile(dfree,'^remap') #pour le reslice
flab = gfile(dfs, '^remap_ap'); flab2 = gfile(dfree,'^remap')

flab2 = gfile(dfs, '^rstd_remap_ap')

fout = addprefixtofilenames(flab2,'rstd_')
for fin, fref, fo in zip (flab2,flab, fout):
    tmap = tio.Resample(target=fref)
    il = tio.LabelMap(fin)
    ilt = tmap(il)
    ilt.save(fo)

sujname = get_parent_path(droi,3)[1]
sujnameID = [s[:6] for s in sujname]
sujname = [ f'ahcp_S{i}_retest2_{s}_T1_1mm' for i,s in enumerate(sujname)]
labels_name = np.array(["bg","GM","WM","CSF","vent","cereb","thal","Pal","Put","Cau","amyg","accuben"])
selected_index = [1,2,4,5,6,7,8,9,10,11]
selected_label = torch.zeros(labels_name.shape); selected_label[selected_index]=1; selected_label = selected_label>0
labels_name = labels_name[selected_index]

df = compute_metric_from_list(flab,flab2,sujname, labels_name, selected_label, distance_metric=True, volume_metric=True)

df['sujnameID'] = sujnameID
df.to_csv('/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/res/res_dice_freesurfer_predFS.csv')

ymet=[]
for k in df.keys():
    if 'dice' in k:
        ymet.append(k)

dfmm = df.melt(id_vars=['sujname', ], value_vars=ymet, var_name='metric', value_name='y')
dfmm['one']=1
fig = sns.catplot(data=dfmm, y='y', col='metric', col_wrap=4, x='sujname',kind='strip')

df1 = pd.read_csv('/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/res/res_dist_binPV_versus_freesurfer.csv')
df2 = pd.read_csv('/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/res/res_dist_predFS_versus_freesurfer.csv')
df1['comp'] = "binPV_freesurfer"; df2['comp'] = "predFS_freesurfer"
df = pd.concat([df1,df2])
dfmm = df.melt(id_vars=['sujname','comp' ], value_vars=ymet, var_name='metric', value_name='y')
fig = sns.catplot(data=dfmm,x='metric',y='y', hue='comp',kind="boxen")


#again adult HCP on T1 1mm
dt1= gdir('/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/session1',['\d.*','T1w','T1_1mm'])
#on training dt1= gdir('/network/lustre/iss02/opendata/data/HCP/training39',['\d.*','T1w',])
dfs =  gdir(dt1,['pred_fs','[1234567890]']);dfree =  gdir(dt1,['freesurfer','suj','mri']); dSynthSeg = gdir(dt1,['SynthSeg'])
dpred = gdir(dt1,'pred_myS')
ft1 = gfile(dt1,'^rT1'); ft1 = gfile(dt1,'^T1w_acpc_dc_restore.nii.gz')

#remap freesurfer / fastsurfer
fapar = gfile(dfree,'aparc.aseg'); fapar = gfile(dfs,'^aparc')
tmap = get_fastsurfer_remap(fapar[0],fcsv ='/network/lustre/iss02/opendata/data/template/free_remap.csv')
remap_filelist(fapar,tmap, fref = ft1)

#remap Synthseg
fapar = gfile(dSynthSeg,'rT1w_1mm_synthseg')
tmap = get_fastsurfer_remap(fapar[0],fcsv ='/network/lustre/iss02/opendata/data/template/free_remap.csv')
remap_filelist(fapar,tmap)

#remap old model to new
tmap = tio.RemapLabels(remapping= {0:2, 1:1, 2:3, 3:12, 4:2, 5:11, 6:9, 7:7, 8:8, 9:6, 10:5, 11:19, 12:17, 13:0} )
#["WM", "GM", "CSF", "both_SN","both_red","both_R_Accu", "both_R_Caud", "both_R_Pall", "both_R_Puta", "both_R_Thal",  "cereb_gm", "skin","skull", "background"],
#["BG","GM","WM","CSF","CSFv","cerGM","thal","Pal","Put","Cau","amyg","accuben","SN", "dura","eyes","nerve","vessel","skull","skull_diploe","head"],
f1 = gfile(dpred,'YEB'); f1 = gfile(dpred,'SN_clean')
remap_filelist(f1,tmap)

dfall = pd.DataFrame()
ftar = gfile(dfree,'^rema'); f1 = gfile(dfs,'^rema')
df = compute_metric_from_list(f1,ftar,sujname, labels_name, selected_label, distance_metric=False, volume_metric=True)
df['sujnameID'] = sujnameID; df['comp'] = "predFS_freesurfer"
dfall = pd.concat([dfall,df])

ftar = gfile(dfree,'^rema'); f1 = gfile(dpred,'^rema.*SN')
df = compute_metric_from_list(f1,ftar,sujname, labels_name, selected_label, distance_metric=False, volume_metric=True)
df['sujnameID'] = sujnameID; df['comp'] = "SN_clean_freesurfer"
dfall = pd.concat([dfall,df])

ftar = gfile(dfree,'^rema'); f1 = gfile(dpred,'^rema.*YEB')
df = compute_metric_from_list(f1,ftar,sujname, labels_name, selected_label, distance_metric=False, volume_metric=True)
df['sujnameID'] = sujnameID; df['comp'] = "bYEB_erod_freesurfer"
dfall = pd.concat([dfall,df])

ftar = gfile(dfree,'^rema'); f1 = gfile(dSynthSeg,'^rema')
df = compute_metric_from_list(f1,ftar,sujname, labels_name, selected_label, distance_metric=False, volume_metric=True)
df['sujnameID'] = sujnameID; df['comp'] = "SynthSeg_freesurfer"
dfall = pd.concat([dfall,df])
dfall.to_csv('/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/res/res_1mm_mypred.csv')