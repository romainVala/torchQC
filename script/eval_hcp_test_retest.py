
import torch,numpy as np,  torchio as tio
from utils_metrics import compute_metric_from_list #get_tio_data_loader, predic_segmentation, load_model, computes_all_metric
from timeit import default_timer as timer
import json, os, seaborn as sns, shutil
from utils_file import gfile, gdir, get_parent_path, addprefixtofilenames, r_move_file,,delete_file_list
import pandas as pd
from nibabel.viewers import OrthoSlicer3D as ov
from utils_labels import remap_filelist, get_fastsurfer_remap, get_remapping, create_mask
import matplotlib.pyplot as plt
from script.create_jobs import create_jobs

plt.interactive(True)
sns.set_style("darkgrid")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


#adult hcp different predictions
suj = gdir('/network/iss/opendata/data/HCP/raw_data/test_retest/session1',['\d.*','T1w'])
#on training suj= gdir('/network/iss/opendata/data/HCP/training39',['\d.*','T1w',])
droi1= gdir(suj,'ROI_PVE_1mm')
droi= gdir(suj,'ROI$')
#suj = gdir(suj,'T1_1mm')
dfs =  gdir(suj,['pred_fs','[1234567890]']);dfree =  gdir(suj,['freesurfer_crop','suj','mri'])
dAssN = gdir(suj,'AssemblyNet');dpred = gdir(suj,'pred_myS');dSynthS = gdir(suj,'SynthSeg2')
dSynthSeg = gdir(suj,['SynthSeg'])

dfsl_bin =  gdir(suj, 'first')

#ft1 = gfile(suj,'^T1w_1mm.nii$'); # ft1 = gfile(suj,'^rT1');
ft1 = gfile(suj,'^T1w_acpc_dc_restore.nii.gz')

#remap
fapar = gfile(dfree,'^aparc.aseg')
tmap = get_fastsurfer_remap(fapar[0],fcsv ='/network/iss/opendata/data/template/remap/free_remapV2.csv', index_col_remap=4)
remap_filelist(fapar,tmap,fref=ft1, prefix='remapHyp_',reslice_with_mrgrid=True)

#make even T1 (for freesurfer)
cmd = [f'mrgrid {f1}  crop -axis 1 0,1 {f2}' for f1,f2 in zip(fanat,fo)]

#lobes remap
fass = gfile(dAssN,'^native_structures_T1w_acpc_dc_restore.nii.gz')
tmap = get_remapping('assn',tmap_index_col=[0,3])
remap_filelist(fass,tmap, prefix='remapLobes_')
remap_filelist(fass, tmap, prefix='Dillremap_', fref=fass, skip=True, reslice_4D=True, blur4D=6,save=True, reduce_BG=0.1 , reslice_with_mrgrid=False)
f = gfile(dAssN, '^Dill')
diclab = get_remapping('assn',lab_name=['names_lobes','value_lobes']);diclab.pop('BG')
create_mask(f,diclab)
df = pd.read_csv('/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/csv_prediction/HCP_test_retest_07mm_suj82_vol_T1_07_free_Ass_siam.csv')
for k,v in diclab.items():
    df[f'mask_{k}'] = gfile(dAssN,f'm_{k}')
    print(k)
df.to_c

sujname = get_parent_path(droi,3)[1]
sujnameID = [s[:6] for s in sujname]
sujname = [ f'ahcp_S{i}_retest2_{s}_T1_1mm' for i,s in enumerate(sujname)]
labels_name = np.array(["bg","GM","WM","CSF","vent","cereb","thal","Pal","Put","Cau","amyg","accuben"])
selected_index = [1,2,4,5,6,7,8,9,10,11]
selected_label = torch.zeros(labels_name.shape); selected_label[selected_index]=1; selected_label = selected_label>0
labels_name = labels_name[selected_index]
concat_label_list = [[0,0,0,0,0,0,0,1,0,1]]

#df = compute_metric_from_list(flab,flab2,sujname, labels_name, selected_label, distance_metric=True, volume_metric=True)

fcsv=gfile('/network/iss/opendata/data/HCP/raw_data/test_retest/res/compare/res1mm/again','csv')
fcsv=gfile('/network/iss/opendata/data/template/manual_seg/MICCAI_2012/metriques/all','csv')
df = pd.concat([pd.read_csv(ff) for ff in fcsv])
def get_pred_GT(ligne,pred=True, col_name='comp'):
    cell_content = ligne[col_name]    #array_path = eval(cell_content)
    res = cell_content.split('_')[0] if pred else cell_content.split('_')[1]
    return res

def filter_comp_mat(dfa, x_sel,y_sel, half=False):
    dfs = pd.DataFrame()
    nbx, nby = len(x_sel), len(y_sel)
    if half:
        for xx in range(nbx - 1):
            for yy in range(ii + 1, nby):
                print(f'{x_sel[xx]} {y_sel[yy]} ')
                dfs = pd.concat([dfs, dfa[(dfa.PRED == x_sel[xx]) & (dfa.GT == y_sel[yy])]])
    else:
        for xx in range(nbx):
            for yy in range(nby):
                print(f'{x_sel[xx]} {y_sel[yy]} ')
                dfs = pd.concat([dfs, dfa[(dfa.PRED == x_sel[xx]) & (dfa.GT == y_sel[yy])]])

    return dfs

def replace_name_from_dict(dfin, col_name, dict_translate):
    for k,v in dict_translate.items():
        dfin[col_name] = dfin[col_name].str.replace(k,v)
    return dfin


df.comp = df.comp.str.replace('PICSL_BC_3','#1');df.comp = df.comp.str.replace('NonLocalSTAPLE_2','#2');df.comp = df.comp.str.replace('MALP_EM_3','#3')

df1 = df.copy()
for k in df1.keys():
    if 'vol_pred_ration' in k:
        print(k)
        if sum(df1['vol_pred_rationthal'].isin([0])) > 0:
            print(f'zero in {k}, ')
        df1[k] = 1/df1[k]

df1['PRED'] = df.apply(lambda s: get_pred_GT(s,pred=False), axis=1)
df1['GT'] = df.apply(lambda s: get_pred_GT(s), axis=1)
df['GT'] = df.apply(lambda s: get_pred_GT(s,pred=False), axis=1)
df['PRED'] = df.apply(lambda s: get_pred_GT(s), axis=1)
dfa=pd.concat([df,df1])

yorder = ['freeS']#[ 'SN', 'Syeb', 'SynS','fsl', 'AN', 'fs', 'freeS']
xorder =  [ 'SN', 'fslB', 'SNfs', 'SynS', 'freeS', 'fs', 'AN' ] #xorder =  [ 'SN', 'Syeb', 'SynS','fsl', 'AN', 'fs']
xorder = ['SN', 'Syeb', 'SynS', 'fs', 'fsl', 'AN', 'freeS']
xorder = [ 'SN', 'FSL', 'SynthS', 'FreeS', 'FS', 'AN',  '#1', '#2', '#3',  'GT01', 'GT',]
xorder = [ 'SN', 'FSL', 'SNfs', 'SynthS', 'FreeS', 'FS', 'AN',  '#1', 'GT',]

xorder_dict = {'^FS$':'FastS', 'SNfs':'SynthFree', 'SN':'SynthFSL', 'SynthS':'SynthSeg','AN':'AssN', 'GT':'GT manu',}
xorder = [ 'SynthFSL', 'FSL', 'SynthFree', 'SynthSeg', 'FreeS', 'FastS', 'AssN',  '#1', 'GT manu',]
dfa = replace_name_from_dict(dfa,'PRED',xorder_dict); dfa = replace_name_from_dict(dfa,'GT',xorder_dict)
xorder_dict = {'^fs$':'FastS', 'SNfs':'SynthFree', 'SN':'SynthFSL', 'SynS':'SynthSeg','AN':'AssN', 'fslB':'FSL', 'freeS':'FreeS'}
xorder = [ 'SynthFSL', 'FSL', 'SynthFree', 'SynthSeg', 'FreeS', 'FastS', 'AssN']

#dfsub = filter_comp_mat(dfa,xorder,yorder)
ymet=[]
for k in df.keys():
    #if 'Sdis_' in k:
    if 'dice' in k:
            ymet.append(k)
#ymet.pop(1);ymet.pop(1);ymet.pop(1);#ymet.pop(-2)
corder= [ymet[0], ymet[6],ymet[7],ymet[4],ymet[5],ymet[9],]#['dice_GM', 'dice_Put', 'dice_Cau','dice_thal','dice_Pal','dice_accuben']
corder = [ymet[0], ymet[7], ymet[8], ymet[5], ymet[6], ymet[10], ]  # ['dice_GM', 'dice_Put', 'dice_Cau','dice_thal','dice_Pal','dice_accuben']

metric_name = ymet[0].split('_')[0]
ctitle = ['GM','Putamen','Caudate','Thalamus','Palidum','Accuben',]
GT=[ 'freeS', 'fslB', 'AN' ];
GT=[ 'FreeS', 'FSL', 'AN', 'GT' ];
figs=[]
for mod in GT:
    yorder = [mod]  # [ 'SN', 'Syeb', 'SynS','fsl', 'AN', 'fs', 'freeS']
    dfs = filter_comp_mat(dfa, xorder, yorder)

    dfmm = dfs.melt(id_vars=['sujname', 'comp','PRED','GT'], value_vars=ymet, var_name='metric', value_name='y');dfmm['one']=1
    #dfmm.replace([np.inf, -np.inf], 0, inplace=True) #dfmm[dfmm.y.isin([np.inf, -np.inf])]
    #fig = sns.catplot(data=dfmm, y='y', x='PRED', col='metric', hue='GT',order=xorder, hue_order=yorder, col_wrap=3, kind='boxen', col_order=corder)
    figs.append(sns.catplot(data=dfmm, y='y', x='one', col='metric', hue='PRED', hue_order=xorder, col_wrap=3, kind='boxen', col_order=corder))
    plt.ylim([0.6,1.4]) #plt.ylim([0.7,1])
for fig,mod in zip(figs,GT):
    for ii, ax in enumerate(fig.axes):
        ax.set_title(ctitle[ii], fontdict=dict(fontsize='x-large'))

        if (ii==0) | (ii==3):
            #ax.set_ylabel('Dice',bbox=dict(boxstyle="round", fc=(0.8, 0.8, 0.8),ec=(0,0,0)),fontsize='x-large')
            ax.set_ylabel(f'{metric_name}',fontsize='x-large') #'Dice'   'Average Surface dist' 'Volume Ratio'
            yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize='x-large' )
        if ii>2:
            ax.set_xticklabels('', rotation=0,fontsize='x-large' );ax.set_xlabel(' '.join(xorder),fontsize='x-large')
        sns.move_legend(fig,'right',fontsize='large',frameon=True, shadow=True, title=f'GT={mod} ',title_fontsize='x-large')
        fig.savefig(f'{dout}/res_{metric_name}_GT_{mod}.png')


morder = [ 'SN', 'Syeb', 'SynS','fsl', 'AN', 'fs', 'freeS']

xorder = morder#['SN']
fig = sns.catplot(data=dfmm, y='y', x='PRED', col='metric', hue='GT',order=xorder, hue_order=yorder, col_wrap=2, kind='boxen')

#all GT in coll, only one region
ymetOne,ymetName = ['dice_Put'], 'Dice'; #ymetOne = ['vol_pred_rationPut'];
corderOne = ['GT manu','FreeS', 'AssN']; ctitleOne = ['GT manual', 'GT Freesurfer', 'GT AssemblyNet', ]
corderOne = ['FSL', 'FreeS', 'AssN']; ctitleOne = ['GT FSL', 'GT Freesurfer', 'GT AssemblyNet', ]
dfmm = dfa.melt(id_vars=['sujname', 'comp', 'PRED', 'GT'], value_vars=ymetOne, var_name='metric', value_name='y');
dfmm['one'] = 1
fig = sns.catplot(data=dfmm, y='y', x='one', col='GT', hue='PRED', hue_order=xorder, col_wrap=3, kind='boxen',col_order=corderOne)
plt.ylim([0.8,0.96]) ; #plt.ylim([0.7,1.3])
for ii, ax in enumerate(fig.axes):
    ax.set_title(ctitleOne[ii], fontdict=dict(fontsize='xx-large'))
    if (ii==0) | (ii==3):
        ax.set_ylabel(f'{ymetName}',fontsize='xx-large')
        yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize='x-large' )
    ax.set_xticklabels('', rotation=0);ax.set_xlabel(' ')
sns.move_legend(fig,'right',fontsize='xx-large',frameon=True, shadow=True, title=f'model ',title_fontsize='xx-large')
fig.savefig(f'{dout}/res_GTs_{ymetOne[0]}.png')

df1 = pd.read_csv('/network/iss/opendata/data/HCP/raw_data/test_retest/res/res_dist_binPV_versus_freesurfer.csv')
df2 = pd.read_csv('/network/iss/opendata/data/HCP/raw_data/test_retest/res/res_dist_predFS_versus_freesurfer.csv')
df1['comp'] = "binPV_freesurfer"; df2['comp'] = "predFS_freesurfer"
df = pd.concat([df1,df2])

#by subject
dfmm = dfa.melt(id_vars=['sujname', 'comp', 'PRED', 'GT'], value_vars=ymetOne, var_name='metric', value_name='y');
dfmm['one'] = 1
fig = sns.relplot(data=dfmm, y='y', x='sujname', col='GT', hue='PRED', hue_order=xorder, col_wrap=1)

import plotly.express as px
import plotly.graph_objs as go

f = go.FigureWidget(); f.layout.hovermode = 'closest'; #f.layout.hoverdistance = -1 #ensures no "gaps" for selecting sparse data
default_linewidth = 2; highlighted_linewidth_delta = 2
allsuj = dfmm.sujname.unique()
for i in range(len(allsuj)):
    y = dfmm[dfmm.sujname==allsuj[i]]
    trace = go.Scatter(y=y, mode='lines', line={ 'width': 3 })
    f.add_trace(trace)

fig = px.line(dfmm, x='PRED', y='y', color='sujname', labels={v:v for v in dfmm.PRED.unique()})

fig.update_traces( line_width= 5)

fig.on_click(update_trace)
# our custom event handler
def update_trace(trace, points, selector):
    # this list stores the points which were clicked on
    # in all but one trace they are empty
    if len(points.point_inds) == 0:
        return
    for i,_ in enumerate(f.data):
        f.data[i]['line']['width'] = 10  # default_linewidth + highlighted_linewidth_delta * (i == points.trace_index)


# we need to add the on_click event to each trace separately
for i in range( len(f.data) ):
    f.data[i].on_click(update_trace)

fig.update_layout(dragmode='drawline', newshape={'line': { 'width': 10}})
fig.show()
fig.show(config={'modeBarButtonsToAdd':['drawline',
                                        'drawopenpath',
                                        'drawclosedpath',
                                        'drawcircle',
                                        'drawrect',
                                        'eraseshape'
                                       ]})

plt.ylim([0.8,0.96]) ; #plt.ylim([0.7,1.3])


#again adult HCP on T1 1mm

#remap freesurfer / fastsurfer
fapar = gfile(dfree,'^aparc.aseg'); fapar = gfile(dfs,'^aparc')
tmap = get_fastsurfer_remap(fapar[0],fcsv ='/network/iss/opendata/data/template/remap/free_remap.csv')

#remap Synthseg
fapar = gfile(dSynthS,'rT1w_1mm_synthseg')
tmap = get_fastsurfer_remap(fapar[0],fcsv ='/network/iss/opendata/data/template/remap/free_remap.csv')
remap_filelist(fapar,tmap)

#remap old model to new
tmap = tio.RemapLabels(remapping= {0:2, 1:1, 2:3, 3:12, 4:2, 5:11, 6:9, 7:7, 8:8, 9:6, 10:5, 11:19, 12:17, 13:0} )
#["WM", "GM", "CSF", "both_SN","both_red","both_R_Accu", "both_R_Caud", "both_R_Pall", "both_R_Puta", "both_R_Thal",  "cereb_gm", "skin","skull", "background"],
#["BG","GM","WM","CSF","CSFv","cerGM","thal","Pal","Put","Cau","amyg","accuben","SN", "dura","eyes","nerve","vessel","skull","skull_diploe","head"],
f1 = gfile(dpred,'YEB'); f1 = gfile(dpred,'SN_clean')
remap_filelist(f1,tmap)

#remap fsl first output
dfsl = gdir(suj,['BADcat12'])
ffirst = gfile(dfsl,'^first.*firstseg.nii.gz')
ffirst = gfile(dfsl_bin,'first.*firstseg.nii.gz')
tmap = get_fastsurfer_remap(ffirst[0],fcsv ='/network/iss/opendata/data/template/remap/remap_fsl_first.csv')
remap_filelist(ffirst,tmap, fref=ft1, prefix='r1mm_remap_')


dfall = pd.DataFrame()
ftar = gfile(dfree,'^rema'); f1 = gfile(dfs,'^rema')
df = compute_metric_from_list(f1,ftar,sujname, labels_name, selected_label, distance_metric=False, volume_metric=True);
df['sujnameID'] = sujnameID; df['comp'] = "predFS_freesurfer"
dfall = pd.concat([dfall,df])

dfall.to_csv('/network/iss/opendata/data/HCP/raw_data/test_retest/res/res_1mm_mypred2.csv',index=False)
dfall = pd.read_csv('/network/iss/opendata/data/HCP/raw_data/test_retest/res/res_1mm_mypred.csv')

f1 = gfile(dpred,'^rema.*SN');  df['comp'] = "SN_clean_freesurfer"
f1 = gfile(dpred,'^rema.*YEB');  df['comp'] = "bYEB_erod_freesurfer"
f1 = gfile(dSynthSeg,'^rema'); df['comp'] = "SynthSeg_freesurfer"


#FSL freesurfer
f1 = gfile(dfsl,'^r1mm_remap_'); ftar = gfile(dfree,'^rema'); df['comp'] = "first_freesurfer"

#fsl bin roi
tmap = tio.RemapLabels(remapping={1:1}); remap_filelist(ftar,tmap, fref=ft1, prefix='rrT1mm_')
ftar = gfile(droi1,'^rrT1mm_bin_PV'); f1 = gfile(dfsl,'^r1mm_remap_');df['comp'] = "first_binPV_1mm";

ftar = gfile(dfree,'^rema');  f1 = gfile(droi1,'^rrT1mm_bin_PV');df['comp'] = "binPV_freesurfer"
#make binPV in trainin39
fassemb = gfile(dAssN,'^native_structure.*nii.gz')
tmap = get_fastsurfer_remap(fassemb[0],fcsv ='/network/iss/opendata/data/template/remap/remap_vol2Brain_label.csv')
remap_filelist(fassemb,tmap,prefix='remapHyp_')

# again with all at once
# 0.7 mmdict_files = {'freeS': gfile(dfree, 'remap'),  'AN': gfile(dAssN, '^remap'),'fsl': gfile(droi, 'bin_PV'), 'fs': gfile(dfs, '^rstd_remap_ap')}
dict_files = {'freeS': gfile(dfree, 'remap'),  'AN': gfile(dAssN, '^remap'),
              'fsl': gfile(droi1, '^rrT1mm_bin_PV'), 'fs': gfile(dfs, '^remap_ap'),
              'Syeb':  gfile(dpred,'^rema.*YEB'), 'SN': gfile(dpred,'^rema.*SN')}
dict_files['fsl'] = gfile(suj,'^nr_bin_PV');dict_files['SynS'] = gfile(dSynthS,'^remap')
dict_files['fslB'] = gfile(dfsl_bin,'^remap')
dict_files['SNfs'] = gfile(dpred,'hcp39GTfs')

for k,v in dict_files.items():
    print(f'{k} {len(v)} {v[0]}')


cmd_init = '\n'.join(['python -c "','from utils_metrics import compute_metric_from_list ','from torch import tensor as tensor'])
dout = '/network/iss/opendata/data/HCP/raw_data/test_retest/res/compare/res1mm/again'
dout = '/network/iss/opendata/data/template/manual_seg/MICCAI_2012/metriques/all'

pred_list = list(dict_files.keys())
nbc = len(pred_list); jobs=[]
for ii in range(nbc-1):
    for jj in range(ii+1,nbc):
        comp_name = f'{pred_list[ii]}_{pred_list[jj]}'
        fout = f'{dout}/compDist_{pred_list[jj]}_{pred_list[ii]}.csv'
        if os.path.isfile(fout):
            print(f'skiping  {comp_name}')
        else:
            print(f' comparing {pred_list[jj]} with {pred_list[ii]}')
            cmd = '\n'.join([cmd_init, f'f1 = {dict_files[pred_list[jj]]}',f'f2 = {dict_files[pred_list[ii]]}',
                             f'sujname = {sujname}', f'labels_name = {list(labels_name)}',
                             f'selected_label = {selected_label}',f'concat_label_list = {concat_label_list}',
                             'df = compute_metric_from_list(f1,f2,sujname, labels_name, selected_label, concat_label_list=concat_label_list, distance_metric=True, volume_metric=True, confu_metric=True)',
                             f"df['comp']='{pred_list[jj]}_{pred_list[ii]}'",f"df.to_csv('{fout}')",
                             '"'
                             ])
            jobs.append(cmd)

job_params = dict()
job_params[
    'output_directory'] = dout + '/job'
job_params['jobs'] = jobs
job_params['job_name'] = 'metrics'
job_params['cluster_queue'] = 'bigmem,normal'
job_params['cpus_per_task'] = 8
job_params['mem'] = 32000
job_params['walltime'] = '02:00:00'
job_params['job_pack'] = 1

create_jobs(job_params)


#eval manual segmentation
#remap
suj = ['/network/iss/opendata/data/template/manual_seg/oasis_1103_3/'] #['/data/romain/template/manual_seg/oasis_1103_3/']
dfree = gdir(suj, ['freesurfer', 'suj', 'mri'])
ft1 = gfile(suj,'^1.*nii.gz')
fgt = gfile('/network/iss/opendata/data/template/manual_seg/MICCAI_2012/MICCAI-2012-Multi-Atlas-Challenge-Data/testing-labels','^1.*gz')
fapar = gfile(dfree,'^aparc.aseg')
tmap = get_fastsurfer_remap(fapar[0],fcsv ='/network/iss/opendata/data/template/remap/free_remapV2.csv', index_col_remap=4)
remap_filelist(fapar,tmap,fref=ft1, prefix='remapHyp_',reslice_with_mrgrid=True)

fapar = gfile(gdir(suj,['fastS','suj']),'^aparc')
tmap = get_fastsurfer_remap(fapar[0],fcsv ='/network/iss/opendata/data/template/remap/free_remapV2.csv')
remap_filelist(fapar,tmap,fref=ft1)

tmap = get_fastsurfer_remap(fgt[0],fcsv ='/network/iss/opendata/data//template/manual_seg/remap_label_brainCOLOR_neuromorphometrics.csv', index_col_remap=2)
remap_filelist(fgt,tmap, prefix='remapHyp_')

fvol2b = gfile(gdir(suj,'vol2Brain'),'native_structure.*nii.gz')
tmap = get_fastsurfer_remap(fvol2b[0],fcsv ='/data/romain/template/manual_seg/remap_vol2Brain_label.csv')
remap_filelist(fvol2b,tmap)
fassemb = gfile(gdir(suj,'AssemblyNet'),'native_structure.*nii.gz')
tmap = get_fastsurfer_remap(fassemb[0],fcsv ='/data/romain/template/manual_seg/remap_vol2Brain_label.csv')
remap_filelist(fassemb,tmap)

ffsl = gfile(gdir(suj,'first'),'_all_fast_firstseg')
tmap = get_fastsurfer_remap(ffsl[0],fcsv ='/data/romain/template/remap/remap_fsl_first.csv')
remap_filelist(ffsl,tmap)

ffsl = gfile(gdir(suj,'SynthSeg'),'_synthseg.nii')
tmap = get_fastsurfer_remap(ffsl[0],fcsv ='/network/iss/opendata/data/template/remap/free_remap.csv')
remap_filelist(ffsl,tmap)

#remap old model to new
f1 = gfile(gdir(suj,'mySynth'),'^bin.*SN_clea')
tmap = tio.RemapLabels(remapping= {0:2, 1:1, 2:3, 3:12, 4:2, 5:11, 6:9, 7:7, 8:8, 9:6, 10:5, 11:19, 12:17, 13:0} )
#["WM", "GM", "CSF", "both_SN","both_red","both_R_Accu", "both_R_Caud", "both_R_Pall", "both_R_Puta", "both_R_Thal",  "cereb_gm", "skin","skull", "background"],
#["BG","GM","WM","CSF","CSFv","cerGM","thal","Pal","Put","Cau","amyg","accuben","SN", "dura","eyes","nerve","vessel","skull","skull_diploe","head"],
f1 = gfile(dpred,'YEB'); f1 = gfile(dpred,'SN_clean')
remap_filelist(f1,tmap)

#compute dice
sujs = gdir(suj,'.*')

fpred=gfile(sujs,'^remap')
flab = gfile(get_parent_path(fpred,2)[0],'^remap')
sujname = ['ANet', 'SynS', 'fastS', 'freeS', 'fsl', 'SN', 'vol2B']
df = compute_metric_from_list(fpred,flab,sujname, labels_name, selected_label, distance_metric=False, volume_metric=True)
df.to_csv('/data/romain/template/manual_seg/oasis_1103_3/comp_dice.csv',index=False)

dfmm = df.melt(id_vars=['sujname', ], value_vars=ymet, var_name='metric', value_name='y')
xorder =['SN',  'vol2B', 'SynS', 'fastS', 'fsl', 'ANet' , 'freeS']
fig = sns.catplot(data=dfmm, y='y', col='metric', order=xorder, col_order=corder, col_wrap=3, x='sujname',kind='strip',s=10)



#manual seg oasis
#adult hcp different predictions
suj = gdir('/network/iss/opendata/data/template/manual_seg/MICCAI_2012/nii',['.*'])
dfs =  gdir(suj,['pred_fs','[1234567890]']);dfree =  gdir(suj,['freesurfer','suj','mri'])
dAssN = gdir(suj,'AssemblyNet');dpred = gdir(suj,'pred_myS');dSynthS = gdir(suj,'SynthSeg'); dfsl = gdir(suj,'fsl_first_rstd')

ffsl = gfile(dfsl,'_all_fast_firstseg')
tmap = get_fastsurfer_remap(ffsl[0],fcsv ='/data/romain/template/remap/remap_fsl_first.csv')
remap_filelist(ffsl,tmap,fref=fgt)

fpred = gfile(dpred,'_synthseg.nii')
tmap = get_fastsurfer_remap(fpred[0],fcsv ='/network/iss/opendata/data/template/remap/free_remap.csv')
remap_filelist(fpred,tmap)

fapar = gfile(dfree,'^aparc.aseg')
tmap = get_fastsurfer_remap(fapar[0],fcsv ='/network/iss/opendata/data/template/remap/free_remapV2.csv')
tmap = get_fastsurfer_remap(fapar[0],fcsv ='/network/iss/opendata/data/template/remap/free_remapV2.csv', index_col_remap=4)
remap_filelist(fapar,tmap,fref=ft1, prefix='remapHyp_',) #remap_filelist(fapar,tmap,fref=fgt)

fpred = gfile(dfs,'^aparc')
tmap = get_fastsurfer_remap(fpred[0],fcsv ='/network/iss/opendata/data/template/remap/free_remapV2.csv')
remap_filelist(fpred,tmap,fref=fgt)

fpred = gfile(dAssN,'^native_structure.*nii.gz')
tmap = get_fastsurfer_remap(fpred[0],fcsv ='/network/iss/opendata/data/template/remap/remap_vol2Brain_label.csv')
remap_filelist(fpred,tmap,prefix='remapHyp_')


#remap old model to new
f1 = gfile(gdir(suj,'mySynth'),'^bin.*SN_clea')
tmap = tio.RemapLabels(remapping= {0:2, 1:1, 2:3, 3:12, 4:2, 5:11, 6:9, 7:7, 8:8, 9:6, 10:5, 11:19, 12:17, 13:0} )
#["WM", "GM", "CSF", "both_SN","both_red","both_R_Accu", "both_R_Caud", "both_R_Pall", "both_R_Puta", "both_R_Thal",  "cereb_gm", "skin","skull", "background"],
#["BG","GM","WM","CSF","CSFv","cerGM","thal","Pal","Put","Cau","amyg","accuben","SN", "dura","eyes","nerve","vessel","skull","skull_diploe","head"],
f1 = gfile(dpred,'YEB'); f1 = gfile(dpred,'SN_clean')
remap_filelist(f1,tmap, fref=fgt)  #because image were not in standard space (todo should change the predict)
f1 = gfile(dpred, 'hcp39GTfs'); tmap = tio.RemapLabels(remapping= {0:0} )


labels_name = np.array(["bg","GM","WM","CSF","vent","cereb","thal","Pal","Put","Cau","amyg","accuben"])
selected_index = [1,2,3,4,5,6,7,8,9,10,11]
selected_label = torch.zeros(labels_name.shape); selected_label[selected_index]=1; selected_label = selected_label>0
labels_name = labels_name[selected_index]

dout = '/network/iss/opendata/data/template/manual_seg/MICCAI_2012/metriques/' #'/data/romain/template/manual_seg/MICCAI_2012/metriques/'
dres = gdir('/data/romain/template/manual_seg/MICCAI_2012/processed','.*')
dres = gdir('/network/iss/opendata/data/template/manual_seg/MICCAI_2012/processed','(PICSL_BC_3|NonLocalSTAPLE_2|MALP_EM_3)')
fres = gfile(dres,'.*gz')
tmap = get_fastsurfer_remap(fapar[0],fcsv ='/data/romain/template/manual_seg/MICCAI_2012/MICCAI_MultiAtlasChallenge2012_corrected/labels_name.csv')
remap_filelist(fres,tmap)
resname = get_parent_path(dres)[1]
fgt = gfile('/network/iss/opendata/data/template/manual_seg/MICCAI_2012/MICCAI-2012-Multi-Atlas-Challenge-Data/testing-labels','^remapHyp')
fgt01 = gfile('/network/iss/opendata/data/template/manual_seg/MICCAI_2012/MICCAI-2012-Multi-Atlas-Challenge-Data/testing-labels','^s01_remap')
sujname = [ss[6:10] for ss in get_parent_path(fgt)[1] ]
dict_files ={}
for one_res, rname in zip(dres, resname):
    fpred = gfile(one_res,'^remap')
    dict_files[rname] = fpred
#dict_files ={} ;
dict_files['FreeS']=gfile(dfree,'^remap'); dict_files['SynthS'] = gfile(dSynthS,'^remap');
dict_files['SN'] = gfile(dpred,'^remap.*SN'); dict_files['Syeb'] = gfile(dpred,'^remap.*YEB')
dict_files['AN'] = gfile(dAssN,'^remap'); dict_files['FS'] = gfile(dfs,'^remap');
dict_files['FSL'] = gfile(dfsl,'^remap')
#pour la matrice complet ...
dict_files['GT'] = fgt;dict_files['GT01'] = fgt01
dict_files['SNfs'] = gfile(dpred,'hcp39GTfs')

for k,v in dict_files.items():
    print(f'{k} {len(v)} {v[0]}')

#compute metrics
concat_label_list = [[0,0,0,0,0,0,0,0,1,0,1]]

for rname, fpred  in dict_files.items():
    df = compute_metric_from_list(fpred,fgt,sujname, labels_name, selected_label, concat_label_list=concat_label_list, distance_metric=False, volume_metric=True)
    df['comp'] = f'{rname}_GT'; df.to_csv(dout + f'/{rname}_gt.csv', index=False)
    print(f'working on GT01 {rname} done')

    df = compute_metric_from_list(fpred,fgt01,sujname, labels_name, selected_label, concat_label_list=concat_label_list, distance_metric=False, volume_metric=True)
    df['comp'] = f'{rname}_GT01';    df.to_csv(dout + f'/{rname}_gt01.csv', index=False)
    print(f'working on {rname} done')

fcsv= gfile(dout,'gt.csv')
df = pd.concat([pd.read_csv(ff) for ff in fcsv])

dic_res = {dd:f'r{kk+1:02}'  for kk,dd in enumerate(df.comp.unique())}
for kk,vv in dic_res.items():
    df.comp = df.comp.str.replace(kk,vv,regex=False)
#ou
df['GT'] = df.comp.str.contains('GT01')
df.comp = df.comp.str.replace('GT01','GT')
df.comp = df.comp.str.replace('_GT','')
df.comp = df.comp.str.replace('PICSL_BC_3','#1');df.comp = df.comp.str.replace('NonLocalSTAPLE_2','#2');df.comp = df.comp.str.replace('MALP_EM_3','#3')

corder = [ymet[0], ymet[7], ymet[8], ymet[5], ymet[6], ymet[11], ]  # ['dice_GM', 'dice_Put', 'dice_Cau','dice_thal','dice_Pal','dice_accuben']

dfsub = df[~ ((df.sujname==1119) | (df.sujname==1128))]

dfmm = dfsub.melt(id_vars=['sujname', 'comp' ], value_vars=ymet, var_name='metric', value_name='y');dfmm['one']=1

figs = sns.catplot(data=dfmm, y='y', x='comp', col='metric', col_wrap=3, kind='boxen', col_order=corder)
figs = sns.catplot(data=dfmm, y='y', x='sujname', col='metric', col_wrap=3, kind='boxen', col_order=corder)


dfsub = df[~ ((df.sujname==1119) | (df.sujname==1128))]

dfmm = dfsub.melt(id_vars=['sujname', 'comp', 'GT' ], value_vars=ymet, var_name='metric', value_name='y');dfmm['one']=1
figs = sns.catplot(data=dfmm, y='y', x='comp', col='metric', col_wrap=3, kind='boxen', col_order=corder, hue='GT')

#4D smooth of manual GT
ts = tio.Blur(std=0.5)
thot = tio.OneHot(); thoti = tio.OneHot(invert_transform=True)
fout = addprefixtofilenames(fgt,'s01_')

for ffin, ffout in zip(fgt,fout):
    il = tio.LabelMap(ffin)
    ilr = thot(il)
    ilr['data'] = ilr.data.float()
    for k in range(ilr.data.shape[0]):
        ilk = tio.ScalarImage(tensor=ilr.data[k].unsqueeze(0), affine=ilr.affine)
        iltk = ts(ilk)
        ilr.data[k] = iltk['data'][0]
    ila = thoti(ilr)
    ila['data'] = ila.data.to(torch.uint8)
    ila.save(ffout)
    print(f'save {ffout}')



### write validation csv
def generate_job_model_file_and_name(fin_list, model_file, model_name, jobdir, skip_if_exist=True, option=''):
    from script.create_jobs import create_jobs

    cmd_ini = 'python /network/iss/cenir/software/irm/toolbox_python/romain/torchQC/segmentation/predict.py '
    fout_list = addprefixtofilenames(fin_list, 'pred_')
    jobs = []
    for fin, fout in zip(fin_list, fout_list):
        for mod, mod_name in zip(model_file, model_name):
            if not os.path.isfile(mod):
                raise(f'Error model {mod} BAD file')
            ffout = f'{fout}_{mod_name}'
            if skip_if_exist:
                real_fout = addprefixtofilenames(ffout, 'bin_')[0]
                if os.path.isfile(real_fout + '.nii.gz'):
                    print(f'skiping out exist {real_fout}')
                    continue
            jobs.append(f'{cmd_ini} -v {fin} -m {mod} -f {ffout} {option}')

    job_params = dict()
    job_params[
        'output_directory'] = jobdir
    job_params['jobs'] = jobs
    job_params['job_name'] = 'predict'
    job_params['cluster_queue'] = 'bigmem,normal'
    job_params['cpus_per_task'] = 8
    job_params['mem'] = 32000
    job_params['walltime'] = '02:00:00'
    job_params['job_pack'] = 1

    create_jobs(job_params)

def generate_job_from_model_link(fin_list, model_file, jobdir, skip_if_exist=True, option=''):
    from script.create_jobs import create_jobs

    cmd_ini = 'python /network/iss/cenir/software/irm/toolbox_python/romain/torchQC/segmentation/predict.py '
    fout_list = addprefixtofilenames(fin_list, 'pred_')
    jobs = []
    for fin, fout in zip(fin_list, fout_list):
        for mod in model_file:
            mod_name = get_parent_path(mod)[1][:-8]

            if not os.path.isfile(mod):
                raise(f'Error model {mod} BAD file')
            ffout = f'{fout}_{mod_name}'
            if skip_if_exist:
                real_fout = addprefixtofilenames(ffout, 'bin_')[0]
                if os.path.isfile(real_fout + '.nii.gz'):
                    print(f'skiping out exist {real_fout}')
                    continue
            jobs.append(f'{cmd_ini} -v {fin} -m {mod} -f {ffout} {option}')

    job_params = dict()
    job_params[
        'output_directory'] = jobdir
    job_params['jobs'] = jobs
    job_params['job_name'] = 'predict'
    job_params['cluster_queue'] = 'bigmem,normal'
    job_params['cpus_per_task'] = 8
    job_params['mem'] = 32000
    job_params['walltime'] = '02:00:00'
    job_params['job_pack'] = 1

    create_jobs(job_params)

def copy_label_data(dfval, dout):
    import os
    for ii, dfser in dfval.iterrows():
        print(f'ligne {ii} {dfser.sujname}')
        sujdir = dout + dfser.sujname + '/'
        if not os.path.isdir(sujdir):
            os.mkdir(sujdir)
        for vol_key in dfval.keys():
            if vol_key == 'sujname':
                continue
            fin = dfser[vol_key]
            ext = 'nii.gz' if fin[-2:] == 'gz' else 'nii'
            fout = sujdir + f'{vol_key}.{ext}'
            os.symlink(fin, fout)

def copy_validation_set(df, outdir):
    vol_key = []
    dic_list = []
    for kk in df.keys():
        if ('vol' in kk) or ('lab' in kk):
            vol_key.append(kk)
    for ii, dfser in df.iterrows():
        dic_out={}; dic_out['sujname'] = dfser["sujname"]
        for volname in vol_key:
            fin = dfser[volname];
            fout = f'{outdir}/{dfser["sujname"]}_{volname}_{get_parent_path(fin)[1]}'
            print(f'cp {fin} {fout}')
            shutil.copyfile(fin,fout)
            dic_out[volname] = fout
        dic_list.append(dic_out)
    dfout = pd.DataFrame(dic_list)
    dfout.to_csv(f'{outdir}/validataion_set.csv')

def copy_link_from_model_file(model_list,model_name,outdir):
    for fin, fname, od in zip(model_list, model_name, outdir):
        fout = f'{outdir}/{fname}.pth.tar'
        if os.path.isfile(fout):
            print(f'skip Exist {fout}')
        else:
            os.symlink(fin, fout)

outdir = '/data/romain/template/validation_set/HCP'
copy_validation_set(dfv,outdir)

dfv = pd.read_csv('/network/iss/opendata/data/template/validataion_set/HCP_test_retest_07mm_suj82.csv')

# training set 16
sujname = get_parent_path(suj,2)[1];  sujname = [ f'hcp_S{i+1:02}_{s}' for i,s in enumerate(sujname)]
flab_rib = gfile(suj,'^remapGM_ribbo'); flab_mid = gfile(droi,'r07_bin.*')
df['lab_rib'] = flab_rib;df['lab_mid'] = flab_mid;
df.to_csv('/network/iss/opendata/data/template/validataion_set/HCP_trainset_07mm_suj16.csv', index=False)

# validataion csv HCP
ft1 = gfile(suj,'^T1w_acpc_dc_restore.nii.gz');ft2 = gfile(suj,'^T2w_acpc_dc_restore.nii.gz')
flab_Ass = gfile(dAssN,'remapHy');flab_free = gfile(dfree,'remapHy')

sujname = get_parent_path(droi,3)[1];sujname = [ss if 'S2' in ss else ss+'_S1' for ss in sujname]; sujname = [ f'hcp_S{i+1:02}_{s}' for i,s in enumerate(sujname)]

print(f'fT1 {len(ft1)} first {ft1[0]}');print(f'fT2 {len(ft2)} first {ft2[0]}');print(f'flab_Ass {len(flab_Ass)} first {flab_Ass[0]}');print(f'flab_free {len(flab_free)} first {flab_free[0]}')
df = pd.DataFrame(); df['sujname'] = sujname
df['vol_T1_07'] = ft1;df['vol_T2_07'] = ft2;df['lab_Free'] = flab_free;df['lab_Assn'] = flab_Ass
df.to_csv('/network/iss/opendata/data/template/validataion_set/HCP_test_retest_07mm_suj82.csv', index=False)

# validataion csv MICCAI
ft1 = gfile(suj,'^1.*nii.gz')
flab_Ass = gfile(dAssN,'remapHy');flab_free = gfile(dfree,'remapHy')
sujname = [ss[6:10] for ss in get_parent_path(fgt)[1] ]; sujname = [f'mic_S{i+1:02}_{s}' for i,s in enumerate(sujname)]
df = pd.DataFrame()
df['sujname'] = sujname
df['vol_T1'] = ft1; df['lab_gt'] = fgt; df['lab_Free'] = flab_free;df['lab_Assn'] = flab_Ass
df.to_csv('/network/iss/opendata/data/template/validataion_set/MICCAI_testset_suj20.csv', index=False)

#DBB
dfDBB = pd.read_csv('/network/iss/opendata/data/template/manual_seg/DBB/testset/my_qc.csv')
dfs=dfDBB.sort_values('selected');dfs = dfs[dfs.selected>0] #18
suj = [ gdir('/network/iss/opendata/data/template/manual_seg/DBB/testset',sub)[0] for sub in dfs.sujname.values]
sujt = gdir(suj,'anat'); sujl = gdir(suj,'parcel');sujm = gdir(suj,'mask');
df = pd.DataFrame()
df['sujname'] = [f'S{kk:02}_{sn}' for kk,sn in enumerate(dfs.sujname.values)]
df['vol_T1'] = gfile(sujt,'gz'); df['lab_GT'] = gfile(sujl,'gz');
df.to_csv('/network/iss/opendata/data/template/validataion_set/DBB_sel18.csv', index=False)
create_nnunet_testset_from_csv(['/network/iss/opendata/data/template/validataion_set/DBB_sel18.csv'],
                               '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set')

# ULTRABRAIN
suj = gdir('/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/UTE/','ULTRABRAIN')
dmid = gdir(suj,'mida_v5'); suj = get_parent_path(dmid)[0] #to get training set
suj = suj[:3]+suj[4:5] #to get testset
#suj = suj[-4:]; #suj = get_parent_path(dAssN)[0];
dfreeS = gdir(suj, ['freesurfer','suj$','surf']); dAssN = gdir(suj,'AssemblyNet')
dfree = gdir(suj, ['freesurfer','suj$','mri'])
sujname = [f'ULTRA_{s[15:18]}' for s in get_parent_path(suj)[1]]
dcoreg = gdir(suj,'coreg_head_skull')

fute, funi, finv2, fflair, fwmn, fct = gfile(suj,'^UTE'),gfile(suj,'^rUNI'), gfile(suj,'^rINV2'), gfile(suj,'^rFLAI'), gfile(suj,'^rWMn'), gfile(suj,'^rCT')
finv1 = gfile(suj,'^rINV1')
fo = [f'/vol_inv1/{nn}_{ss}' for nn,ss in zip(sujname,get_parent_path(finv1)[1])]

fute, funi, finv2, fflair, fct = (gfile(dcoreg,'^rHSonCT.*UTE'),gfile(dcoreg,'^rHSonCT.*UNI'),
                                        gfile(dcoreg,'^rHSonCT.*INV2'), gfile(dcoreg,'^rHSonCT.*FLAI'),
                                        gfile(suj,'^rCT'))
fapar = gfile(dfree,'^aparc.aseg')
tmap = get_fastsurfer_remap(fapar[0],fcsv ='/network/iss/opendata/data/template/remap/free_remapV2.csv', index_col_remap=4)


fall = fute+ funi+ finv2+ fflair+ fwmn + fct
fmask = gfile(gdir(get_parent_path(fall)[0],'spm'),'mask_brain_erode_dila')
fout = addprefixtofilenames(fall,'crop_')
fcmd = open('cmd2.bash', 'w')
for fi,fm, fo in zip(fall,fmask,fout):
    cmd = f'mrgrid {fi} crop -mask {fm} -uniform -30 -crop_unbound - | mrgrid -force - crop -axis 2 30,0 {fo}\n'
    fcmd.write(cmd); print(cmd)#outvalue = subprocess.run(cmd.split(' '))
fcmd.close()
fin = gfile(suj,'^crop_'); fout = addprefixtofilenames(fin,'r07')
fcmd = open('cmd2.bash', 'w')
for fi,fo in zip(fin,fout):
    if not os.path.isfile(fo):
        cmd = f'echo {fo}\n mrgrid -force {fi} regrid -voxel 0.7 {fo}\n'
        fcmd.write(cmd); #    outvalue = subprocess.run(cmd.split(' '))
fcmd.close()

dout='/network/iss/opendata/data/template/validataion_set/'
df = pd.DataFrame()
df['sujname'] = sujname
df['vol_ute'], df['vol_uni'], df['vol_inv2'], df['vol_flair'], df['vol_wmn'], df['vol_ct'] = fute, funi, finv2, fflair, fwmn, fct
df['vol_ute'], df['vol_uni'], df['vol_inv2'], df['vol_flair'], df['vol_ct'] = fute, funi, finv2, fflair, fct
df['vol_ute'], df['vol_uni'], df['vol_inv2'], df['vol_flair'], df['vol_wmn'] = gfile(suj,'^crop_UTE'),gfile(suj,'^crop_rUNI'), gfile(suj,'^crop_rINV2'), gfile(suj,'^crop_rFLAI'), gfile(suj,'^crop_rWMn')
df.to_csv(dout + 'ULTRA_trainset_suj5x5.csv', index=False)
df['vol_ute'], df['vol_uni'], df['vol_inv2'], df['vol_wmn'] , df['vol_ct'] = gfile(suj,'^r07crop_UTE'),gfile(suj,'^r07crop_rUNI'), gfile(suj,'^r07crop_rINV2'), gfile(suj,'^r07crop_rFLAI'), gfile(suj,'^r07crop_rWMn'), gfile(suj,'^r07crop_rCT')
df['lab_mid'] =  gfile(dmid,'^rUTE_binmr')
df['mask_top'] = gfile(gdir(suj, 'slicer'), 'mask_top')

#to add
flabel = gfile(gdir(dir_suj, 'mida_v5'), '^crop_rUTE_binmrt_r025_bin_PV_head_mida_Aseg_cereb')
# flabel = gfile(gdir(dir_suj, 'mida_v5'), '^rUTE_binmrt_r025_bin_PV_head_mida_Aseg_cereb')
df['lab_Assn'] = flabel
df.to_csv(dout + '/ULTRA_trainset_suj5x5_r07.csv', index=False)
df.to_csv(dout + '/ULTRA_all.csv', index=False)

#run predict
dir_model='/data/romain/template/validation_set/model'
dout = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/new_eval/HCP/'
dout = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/new_eval/MIC/'
dout = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/new_eval/ULTRAr07/'
dfval = pd.read_csv('/network/iss/opendata/data/template/validataion_set/HCP_test_retest_07mm_suj82.csv')
dfval = pd.read_csv('/network/iss/opendata/data/template/validataion_set/MICCAI_testset_suj20.csv')
dfval = dfval[:10]
copy_label_data(dfval, dout)

suj_eval = gfile(gdir(dout,'.*'),'^vol_')

dout = '/data/romain/template/validation_set/'
suj_eval = gfile('/data/romain/template/validation_set/HCP', 'vol_T1')
suj_eval = gfile('/data/romain/template/validation_set/ULTRA_07', 'vol_')

model_file = gfile(dir_model,'.*tar')
jobdir = dout + 'jobs/'
#generate_job_model_file_and_name(suj_eval, model_file, model_name, jobdir) #, option=' -d cpu ')
generate_job_from_model_link(suj_eval, model_file,  jobdir)
generate_job_from_model_link(suj_eval, model_file,  jobdir, option=' --VoxelSize 1.5 ')
generate_job_from_model_link(suj_eval, model_file,  jobdir, option=' -mj /network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/jzay/training/RES_mida/Uhcp4_skul_v5/skv5.1/noSpMean_nnUnet/res_bs2_P192_gpu1cpu40/model.json ')

##########  ultracortex import
fGT=gfile('/network/iss/opendata/data/template/manual_seg/ultracortex/ds005216/derivatives/manual_segmentation','seg')
ffre=gfile('/network/iss/opendata/data/template/manual_seg/ultracortex/ds005216/derivatives/freesurfer_segmentation','seg')
sujn = [ss[:6] for ss in get_parent_path(fGT)[1]];sujn[3] = 'sub-3'; sujn[-1] = 'sub-9'
suj = [ gdir('/network/iss/opendata/data/template/manual_seg/ultracortex/ds005216',ss)[0] for ss in sujn]
suj = gdir(suj,['ses-1','ana'])
fT1 = gfile(suj,'.*nii')
dout='/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/ultracortex/vol_T1std/freesurfer_seg/'

fana_brain = [ gfile('/network/iss/opendata/data/template/manual_seg/ultracortex/ds005216/derivatives/skullstrips/',ss+'_')[0] for ss in sujn]
fo = gfile('/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/ultracortex/vol_T1std','.*gz')
fo = ['/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/ultracortex/vol_T1std/gouhfi_brain_masked/conform/'+ss for ss in get_parent_path(fo)[1]]
to = tio.ToOrientation('LIA')
for f1,f2 in zip(fana_brain,fo):
    it = to(tio.ScalarImage(f1))
    it.save(f2)


fos = [dout + ff for ff in  get_parent_path(fT1)[1] ]
tc = tio.ToCanonical()
for fi,fo in zip(fT1,fos):
    il = tc(tio.ScalarImage(fi))
    il.save(fo)

fl = gfile(dout,'.*nii')
tmap = tio.RemapLabels({0:0, 2:0, 41:0, 43:0, 44:0, 45:0, 42:1,  3:1})
remap_filelist(fl, tmap, prefix='remapGM_', skip=False)
fT1 = gfile(dout,'gz')
ffr = gfile(gdir(dout,'free'),'^remap.*gz')
fgt = gfile(gdir(dout,'label'),'^remap.*gz')
suj =[ s[:-17] for s in get_parent_path(fT1)[1] ]
df = pd.DataFrame()






dfall = df.copy()
dfs1 = dfall[dfall.dataset_name=='MICCAI_testset_suj20_vol_T1']
dfs2 = dfall[dfall.dataset_name=='HCP_trainset_07mm_suj16_vol_T1_07']

df['one'] = 1
sns.catplot(data=df, y='dice_GM', x='label', col='dataset_name', hue='model_name', col_wrap=3, kind='boxen') #hue_order=xorder                        col_order=corder)
dfs =df[( df.dataset_name=='HCP_test_retest_07mm_suj82_vol_T1_07') & (df.label=='Assn') & (df.hausdorff_dist_Put>30)]
dfs1 =df[( df.dataset_name.str.startswith('UL') ) ]
dfs2 =df[( df.dataset_name.str.startswith('HCP') ) | ( df.dataset_name.str.startswith('MIC') ) ]


morder = ['nnUnet_lowres', 'nnUnet_NoDA', 'nnUnet_NoDeep', 'nnUnet', 'nnUnet_XL',
         'e3unet', 'unet_hcp16','unet_Uhcp4',]
dsorder = ['HCP_test_retest_07mm_suj82_vol_T1_07','HCP_trainset_07mm_suj16_vol_T1_07','MICCAI_testset_suj20_vol_T1']
df.model_name.str.replace('3d_fullres_nnUNetTrainerNoDA_nnUNetPlans','nnUnet_NoDA',inplace=True)

dfs = df[df.dataset_name!= 'wmn']
dfs = dfs[dfs.model_name=='3d_fullres_nnUNetTrainer_nnUNetResEncUNetXLPlans_DS702']
dfs = df[(df.dataset_name=='dHcp_old075_volT2_GT') & (df.label_column=='lab_binPV')]
dfmm = df.melt(id_vars=['subject_id', 'model_name', 'dataset_name', 'label'], value_vars=ymet, var_name='from', value_name='dice');
ymet=['dice_Put','dice_Cau-acc','dice_Pal','dice_thal','dice_hypp','dice_amyg','dice_cerGM','dice_CSFv']
sns.catplot(data=dfmm, y='dice', x='label', col='from', hue='model_name', kind='boxen',hue_order=morder,col_wrap=4)

cc = sns.color_palette()

dfs22 = df[df.dataset_name=='HCP_test_retest_07mm_suj82_vol_T1_07']
dfs22 = df[df.dataset_name=='MICCAI_testset_suj20_vol_T1']
dfs22 = df[(df.dataset_name=='ULTRA_testset_r06_vol_uni')|(df.dataset_name=='ULTRA_trainset_suj5x5_vol_uni')]
dfmm = dfs.melt(id_vars=['sujname', 'model_name', 'dataset_name', 'label'], value_vars=ymet, var_name='from', value_name='dice');
sns.catplot(data=dfmm, y='dice', x='dataset_name',kind='boxen', col='from',col_wrap=1)
sns.catplot(data=dfmm, y='dice', x='dataset_name',kind='boxen', col='from', hue='model_name')
sns.catplot(data=dfmm, y='y', x='label', col='dice', hue='model_name', col_wrap=4, kind='boxen',hue_order=morder)


morder=['pred_DS704_3nnResXL_res', 'pred_DS706_5nnResXL_res','pred_DS708_5nnResXL_res','pred_DS709_CascadeNoDA_res',
       'pred_DS710_5ResEncXL_res','pred_DS718_5nnResXXLres','pred_DS712_5ResXLres']

ymet=[]
for k in df.keys():
    if 'hausd' in k:
        ymet.append(k)


dfa = pd.concat([dfs,dfsr])
dfa.model_name.replace('synth_fuzzy_resunet_without_back_original','Ines_orig',inplace=True)
dfa.model_name.replace('synth_fuzzy_resunet_without_back_bce_dice_original','Ines_bce_dice',inplace=True)
dfa.model_name.replace('synth_fuzzy_resunet_without_back_tversky_original','Ines_tversky',inplace=True)
morder = ['nnUnet_lowres', 'nnUnet_NoDA', 'nnUnet_NoDeep', 'nnUnet', 'nnUnet_XL',
         'e3unet', 'unet_hcp16','unet_Uhcp4','Ines_orig','Ines_bce_dice','Ines_tversky']

for ff in fcsv:
    df = pd.read_csv(ff)
    dorig = get_parent_path(df.vol_path_orig)[0]
    dmask = gfile(gdir(dorig,'slicer'),'mask_top')
    if len(dmask)!=9:
        qsdf
    df['mask_top'] = dmask
    df.to_csv(ff,index=False)


