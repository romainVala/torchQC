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
from utils_metrics import display_res, display_res2, mrview_from_df, mrview_overlay
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

sns.set_style("darkgrid"); plt.interactive(True)
pd.set_option('display.max_rows', 500);pd.set_option('display.max_columns', 500);pd.set_option('display.width', 1000)

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
def read_hcp_eval(ress, csv_file_name='eval.csv'):  #warning no libreoffice open on to avoid including .~lock.metrics.csv
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
                fcsv = gfile(one_suj, csv_file_name)
                dfone = pd.read_csv(fcsv[0], index_col=0)
                #print(dfone.keys())
                dfone['suj_num'] = suj_num
                if  not ( ('fpred'  in dfone)): #else label and pred are already in metrics files ('pred'  in dfone) |
                    finput = gfile(one_suj,'^data.nii.gz')
                    fpred = gfile(one_suj,'bin_prediction.nii.gz')
                    flabel = gfile(one_suj,'bin_label.nii.gz')
                    if dfone.shape[0]==2:
                        finput, fpred, flabel = [finput[0], finput[0]], [fpred[0], fpred[0]], [flabel[0], flabel[0]]
                        dfsl = pd.concat([dfsl]*2, ignore_index=True)
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
        print(df.keys())
        df['model'] = resname
        print(f'model {resname} shape is {df.shape}')
        df_all.append(df)
    dfm = pd.concat(df_all)
    dfm["model_name_short"], dfm["ep_name"] = dfm.model.apply(lambda x: mod_name(x)), dfm.model.apply(lambda x: ep_name(x))
    return dfm
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
def compute_mean(df, met):
    for ii, col in enumerate(met):
        if ii==0:
            dfmean = df[col]
        else:
            dfmean = dfmean + df[col]
    return dfmean/len(met)

def filter_df_local(df, quantile = 0.3):
    #keep the 1-quantile part of local patch
    vox_of_interest_in_patches = df.nb_pts_label + df.nb_pts_pred
    keep_all_index = vox_of_interest_in_patches <0
    for models in df.model_name.unique():
        for sujname in df.sujname.unique():
            index_suj_mod = (df.model_name==models) & (df.sujname==sujname)

            threshol = np.quantile(vox_of_interest_in_patches[index_suj_mod].values, quantile)
            keep_index = (vox_of_interest_in_patches>threshol) & index_suj_mod
            keep_all_index = keep_index | keep_all_index

    return(df[keep_all_index])
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_
def q95m(x):
    th_val =  np.percentile(x, 98) #x.quantile(0.95)
    xsel = x[x>th_val]
    return xsel.mean()

def nb_outlier(x):
    #x_values = x.values
    zscore = np.abs( (x - np.mean(x)) / np.std(x) )
    nb_out = np.sum(zscore>3) / len(zscore) * 100
    return nb_out



rootdir = '/data/romain/baby/'
#rootdir = '/network/lustre/iss02/opendata/data/baby/devHCP/'
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

if False:
    ress=gdir('/data/romain/PVsynth/eval_cnn/baby/Article/','eval_T2');
    ress=gdir('/data/romain/PVsynth/eval_cnn/baby/Article/','eval_.*(mot|next)');

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
for k in dfmh.keys():
    if k.startswith("metric_d"):
        ymet.append(k)

dfmh = read_hcp_eval(ress, csv_file_name='metrics_patch_16_grid.csv') #'metrics_patch_32_128.csv')
dfmh['sujn']=dfmh.sujnum;dfmh['mod']=dfmh.model;
dfmh['snr'] = dfmh.Smean_GM.values / dfmh.Sstd_GM.values

ymet_str=[]; ymet_int=[]
for k in dfmh.keys():
    if (isinstance(dfmh[k].values[0], str) ):
        ymet_str.append(k)
    else:
        ymet_int.append(k)
average_dic={k:'first' for k in ymet_str}; average_dic.update({ k:'max' for k in ymet_int })
average_dic={k:'first' for k in ymet_str}; average_dic.update({ k:'mean' for k in ymet_int })
average_dic={k:'first' for k in ymet_str}; average_dic.update({ k:q95m for k in ymet_int })
average_dic['sedation']='first'  #boolean type ... todo
average_dic['sujnum']='first'; average_dic['sujn']='first'

dfmh_filt=dfmh.copy(); #dfmh_filt = filter_df_local(dfmh)
df = dfmh_filt.groupby(['sujn','mod'], as_index=False).agg(average_dic)
sujn_outlier =[18,27,31] #meand dice > 0.1 for dfs=df[df.model=='eval_T2_model_hcpT2_on5suj_ep16']
df = df.drop(df[(df.sujnum==18)|(df.sujnum==27)|(df.sujnum==31)].index)
dfmh.index=range(len(dfmh))
df = df.drop(df[(df.sujnum == 18) | (df.sujnum == 27) | (df.sujnum == 31)| (df.sujnum == 3)| (df.sujnum == 72)].index)

df = dfmh.groupby(['sujn','mod']).max()
idx = dfmh.groupby(['sujn','mod'])['Sdis_GM'].max()
idx = df.groupby(['Mt'])['count'].transform(max) == df['count']

dfmh = read_hcp_eval(ress)
dfmh = read_hcp_eval(ress, csv_file_name='metrics.csv')
ymet = ['metric_dice_GM', 'metric_dice_WM', 'metric_dice_brstem', 'metric_dice_cereb', 'metric_dice_deepgm']
ymet = ['dice_GM', 'dice_WM', 'dice_bstem', 'dice_cereb', 'dice_deepGM', 'dice_hippo']
ymet = ['haus_GM', 'haus_WM']

dfmh['average'] = dfmh.apply(lambda x:  compute_mean(x, ymet), axis=1)
ymet = ['average'] + ymet

dfm = read_feta_eval(ress);
dfmm = dfm.melt(id_vars=['scan_age', 'model_name','ep_name', 'sujnum','model','Pathology','Gestational age'], value_vars=ymet, var_name='metric', value_name='y')

dfmm = dfmh.melt(id_vars=['scan_age', 'model_name','ep_name', 'sujnum','model','radiology_score'], value_vars=ymet, var_name='metric', value_name='y')
#fig = sns.catplot(data=dfmm, y='y', x='model_name', hue='ep_name', col='metric', kind='boxen',col_wrap=3)
fig = sns.relplot(data=dfmm, y='y', x='sujnum', hue='model', col='metric') #,col_wrap=3)
dfmm['one']=1
fig = sns.catplot(data=dfmm, y='y', x='one',hue='model', col='metric', kind='boxen') #,col_wrap=3)


fig = sns.relplot(data=dfm, x='scan_age', y='metric_dice_loss_GM', hue="model")
fig = sns.catplot(data=dfm, y='metric_dice_loss_GM', x='model_name', hue='ep_name', kind='boxen',)

dfs1 = dfmh[(dfmh.model_name=='eval_T1_model_fetaBgT2_hcp_ep1')]
dfs2 = dfmh[(dfmh.model_name=='coreg_eval_T1_model_fetaBgT2_hcp_ep1')]

#figure 2 :  T1 versus T2
df = dfmh.copy()
df['eval_on']='T2'
df.loc[df.model.str.startswith('eval_T1'),'eval_on']='T1'
df.model_name = df.model;
df.model = df.model.str.replace('eval_T1_coreg_model_5suj_motBigAff_ep30','SynthMot');df.model = df.model.str.replace('eval_T2_model_5suj_motBigAff_ep30','SynthMot');df.model = df.model.str.replace('eval_T1_coreg_model_hcpT2_elanext_5suj_ep1','SynthT2');df.model = df.model.str.replace('eval_T2_model_hcpT2_elanext_5suj_ep1','SynthT2')
df.model = df.model.str.replace('eval_T2_model_5suj_ep110','Synth'); df.model = df.model.str.replace('eval_T1_coreg_model_5suj_ep110','Synth'); df.model = df.model.str.replace('eval_T1_coreg_model_hcpT2_on5suj_ep16','SynthT2');df.model = df.model.str.replace('eval_T2_model_hcpT2_on5suj_ep16','SynthT2');
df.model = df.model.str.replace('eval_T1_coreg_model_hcpT2_on5sujAug_ep16','SynthT2');df.model = df.model.str.replace('eval_T2_model_hcpT2_on5sujAug_ep16','SynthT2');
df.model = df.model.str.replace('eval_T1_coreg_model_hcp5sujOnT2_ep64','DataT2');df.model = df.model.str.replace('eval_T2_model_hcp5sujOnT2_ep64','DataT2');
dfmm = df.melt(id_vars=['scan_age', 'model_name', 'sujnum','model','eval_on'], value_vars=ymet, var_name='label', value_name='dice')
#dfmm.label = dfmm.label.str.replace('dice_','')
fig = sns.catplot(data=dfmm, y='dice', x='eval_on',hue='model', col='label', kind='boxen',col_wrap=4) #7, order=['T2','T1'], hue_order=['SynthT2','SynthMot'])

fig = sns.relplot(data=dfmm, y='dice', x='sujnum', hue='model', col='label',col_wrap=2)

#fig1 article MIDL
col_order =  [ 'dice_GM', 'dice_WM', 'dice_CSF', 'dice_cereb', 'dice_deepGM', 'dice_bstem', 'dice_hippo'] #,'dice_vent'
col_order =  [ 'Sdis_GM', 'Sdis_WM', 'Sdis_CSF', 'Sdis_cereb', 'Sdis_deepGM', 'Sdis_bstem', 'Sdis_hippo'] #,'Sdis_vent'
#col_order =  [ 'Vol_GM', 'Vol_WM', 'Vol_CSF','Vol_vent', 'Vol_cereb', 'Vol_deepGM', 'Vol_BrStem', 'Vol_HipAmy']
col=sns.color_palette('muted'); #col = [col[0], col[1], col[3]]

fig = sns.catplot(data=dfmm, y='dice', x='eval_on',hue='model', col='label', kind='boxen',col_wrap=4, order=['T2','T1'],
                  hue_order=['Synth','SynthMot','DataT2'], col_order=col_order, palette=col)
ctitle = [ 'GM', 'WM',  'CSF', 'Cereb', 'DeepGM', 'Bstem', 'Hippo_Amyg'] # 'Ventricules'
plt.ylim([0, 0.2])

for ii, ax in enumerate(fig.axes):
    if (ii==0) | (ii==4):
        #ax.set_ylabel('Dice',bbox=dict(boxstyle="round", fc=(0.8, 0.8, 0.8),ec=(0,0,0)),fontsize='x-large')
        ax.set_ylabel('Dice',fontsize='xx-large') #'Dice'   'Average Surface dist'
        #ax.set_ylabel('Volume Ratio',fontsize='xx-large') #ax.set_ylabel('Average Surface dist',fontsize='xx-large')
        yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize='xx-large' )
    if ii>2:
        ax.set_xticklabels(['T2', 'T1'], rotation=0,fontsize='xx-large' );ax.set_xlabel('')
    #ax.set_xlabel('',fontsize='x-large', va = 'bottom')# #ax.xaxis.set_label_coords(1.05, -0.025) #ax.tick_params(labelbottom=True)
    #ax.set_xlabel('Subject number',fontsize='x-large')# #ax.xaxis.set_label_coords(1.05, -0.025) #ax.tick_params(labelbottom=True)
    ax.set_title(ctitle[ii], fontdict=dict(fontsize='xx-large'))

#plt.subplots_adjust(bottom=0.1, left=0.06, hspace=0.2)
sns.move_legend(fig,"lower right",bbox_to_anchor=(.58, 0.4),fontsize='xx-large',frameon=True, shadow=True, title=None)

#figure 1
fig = sns.relplot(data=dfmm, y='dice', x='sujnum', hue='model', col='label',col_wrap=5, hue_order=['SynthT2','SynthMot','Synth'])

#figure volue ration
yvol=[]
for k in dfmh.keys():
    if k.startswith("occupied_volume"):
        yvol.append(k)
yvolp=[]
for k in dfmh.keys():
    if "predicted_occupied_volume" in k:
        yvolp.append(k)
ytissue = [ss[16:] for ss in yvol]
yvolr=[]
for (y1,y2) in zip(yvol, yvolp):
    new_keys = f'Vol_{y1[16:]}'
    dfmh[new_keys] = dfmh[y2]/dfmh[y1]
    yvolr.append(new_keys)
yvolr = yvolr[1:4] + yvolr[5:] ; ymet=yvolr

model_sel='SynthMot'
for the_tissue in ytissue:
    dfv = pd.DataFrame()
    vv1 = df[(df.model==model_sel)&(df.eval_on=='T1')][f'predicted_occupied_volume_{the_tissue}']
    vv2 = df[(df.model==model_sel)&(df.eval_on=='T2')][f'predicted_occupied_volume_{the_tissue}']
    vv3 = df[(df.model==model_sel)&(df.eval_on=='T2')][f'occupied_volume_{the_tissue}']
    dfv['T1/lab']=vv1/vv3; dfv['T2/lab']=vv2/vv3;
    dfv['T1']=vv1; dfv['T2']=vv2; dfv['lab']=vv3
    dfv['T1/T2'] = vv1/vv2
    dfv['sujnum']=range(len(vv1))
    dfmv =  dfv.melt( value_vars=['T1/lab','T2/lab','T1/T2'], var_name='vol',id_vars=['sujnum'])
    #sns.relplot(data=dfmv, x='sujnum', y='value', hue='vol')
    sns.catplot(data=dfmv, x='vol', y='value', kind='boxen');  plt.ylabel(the_tissue)
    plt.figure(); plt.scatter(dfv.T1, dfv.T2);  plt.plot([dfv.T1.min(), dfv.T1.max()],[dfv.T1.min(), dfv.T1.max()])
    plt.ylabel('T2'); plt.xlabel('T1'); plt.title(the_tissue)
    plt.figure(); plt.scatter(dfv.lab, dfv.T2);  plt.plot([dfv.T1.min(), dfv.T1.max()],[dfv.T1.min(), dfv.T1.max()])
    plt.ylabel('T2'); plt.xlabel('lab'); plt.title(the_tissue)

#repport hue model_per_age
# df['suj_age'] = 0 #NaN ar not ploted
df.loc[df.sujnum<20,'suj_age'] = 1
df.loc[df.sujnum>60,'suj_age'] = 2
#df = df[df.suj_age>0]
df.loc[df.eval_on=='T1','model'] = 'eT1_' + df.loc[df.eval_on=='T1','model'] ;
df.loc[df.eval_on=='T2','model'] = 'eT2_' + df.loc[df.eval_on=='T2','model']
#df.loc[df.suj_age==1, 'model'] = 'young_' + df.loc[df.suj_age==1, 'model']
#df.loc[df.suj_age==2, 'model'] = 'old_' + df.loc[df.suj_age==2, 'model']
df.loc[df.label=='T2', 'model'] = 'lT2_' + df.loc[df.label=='T2', 'model']
df.loc[df.label=='T2_hcpT2', 'model'] = 'lT2HCP_' + df.loc[df.label=='T2_hcpT2', 'model']

df.model_name = df.model;
dfmm = df.melt(id_vars=['scan_age', 'model_name', 'sujnum','model','eval_on','suj_age'], value_vars=ymet, var_name='label', value_name='dice')
#dfmm.label = dfmm.label.str.replace('dice_','')
fig = sns.catplot(data=dfmm, y='dice', x='suj_age',hue='model', col='label', kind='boxen',col_wrap=4)

fig = sns.boxenplot(data=df, y='fdr_GM', x='suj_age',hue='model', k_depth='proportion')

ress = ['/data/romain/PVsynth/eval_cnn/baby/Article/eval_T1_coreg_model_hcpT2_elanext_5suj_ep1', '/data/romain/PVsynth/eval_cnn/baby/Article/eval_T1_coreg_model_5suj_motBigAff_ep30/']
dfmh = read_hcp_eval(ress, csv_file_name='metrics.csv'); dfmh['gt'] = 'label'
dfmh2 = read_hcp_eval(ress, csv_file_name='metrics_labelT2.csv'); dfmh2['gt'] = 'predT2'
dfmh = pd.concat([dfmh, dfmh2])

# figure T2 versus T1 diff
dfs1 = dfmh[ dfmh.model.str.startswith('eval_T1')]
dfs2 = dfmh[ dfmh.model.str.startswith('eval_T2')]
for kk in ymet:
    dfs1[kk] = dfs1[kk] - dfs2[kk]

#write patch of max Sdis_GM
models = df.model_name.unique(); nbpath=30
sujnum=4; sel_ascendig=False; sel_metric='Sdis_GM'; sel_metric='dice_GM'; sel_metric='fdr_GM'
#sel_ascendig=True; sel_metric='snr'
for model in models:
    #print(f'WORKING on {model}')
    dfsub = df[(df.sujnum==sujnum)&(df.model_name==model)]
    #dfh_sub = dfmh[(dfmh.sujnum==sujnum)&(dfmh.model_name==model)]
    dfh_sub = dfmh_filt[(dfmh_filt.sujnum==sujnum)&(dfmh_filt.model_name==model)].copy()
    dfh_sub['snr'] = dfh_sub.Smean_GM/dfh_sub.Sstd_GM
    dfh_sub['snr'] = dfh_sub.Sstd_GM / dfh_sub.Smean_GM
    dfsort = dfh_sub.sort_values(sel_metric, ascending=sel_ascendig)
    print(f'{sel_metric} {dfsub[sel_metric].mean()} model {model} suj {sujnum} ')

    plt.figure(); plt.scatter(dfh_sub[sel_metric], dfh_sub.fdr_GM);  plt.xlabel(sel_metric); plt.title(model);plt.xlim([0,1])



    limg = tio.ScalarImage(dfsub.fpred.values[0]P) #fucking csv formating ...
    patch_select = torch.zeros([nbpath+1] + list(limg.data.shape[1:]))
    for nbp in range(nbpath):
        location = dfsort['location'].values[nbp] #dfh_sub[dfh_sub[sel_metric] == dfsub[sel_metric].values[0]]['location'].values[0]
        ll = location.split(' ');
        ll = location.split('[')[-1].split(']')[0].split(' ')
        loclist = [];
        for elem in ll:
            if len(elem)>0:
                loclist.append(int(elem))
        location = np.array(loclist)

        patch_select[nbp+1,location[0]:location[3],location[1]:location[4],location[2]:location[5]] = dfsort[sel_metric].values[nbp];

    patch_select_all = torch.zeros([1] + list(limg.data.shape[1:]))
    for nbp in range(nbpath):
        patch_one = torch.zeros([1] + list(limg.data.shape[1:]))
        location = dfsort['location'].values[nbp] #dfh_sub[dfh_sub[sel_metric] == dfsub[sel_metric].values[0]]['location'].values[0]
        ll = location.split(' ');
        ll = location.split('[')[-1].split(']')[0].split(' ')
        loclist = [];
        for elem in ll:
            if len(elem)>0:
                loclist.append(int(elem))
        location = np.array(loclist)

        patch_one[0,location[0]:location[3],location[1]:location[4],location[2]:location[5]] = 1;
        patch_select_all+= patch_one

    patch_select[0] = patch_select_all
    limg.data=patch_select
    limg.save(os.path.dirname(limg.path) + f'/patch_{sel_metric}.nii.gz')


#figure on all data T2 bins by age
dfsub = df[df.model=='eval_T2all_model_5suj_motBigAff_ep30']
all_ages = dfsub.scan_age.values
age_bins=[]
for ii in np.arange(20, 709,20):
    print(ii)
    age_bins.append(all_ages[ii])
age_bins = [0] + age_bins +[55]
for ii in range(len(age_bins)-1):
    df.loc[( (df.scan_age > age_bins[ii]) & (df.scan_age < age_bins[ii+1]) ) , 'suj_age'] = ii

yk = 'metric_mean_dice_loss'
sns.relplot(data=dfm, y=yk, x='sujnum', hue='model')
mrview_from_df(dfmh, 'suj_num','5')

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


dana = df.anat_path.values
fT1 = gfile(dana,'^s.*[1234567890]_T1w.nii.gz',opts={"items":1})
dana = get_parent_path(fT1)[0]
fT2 = gfile(dana,'^s.*[1234567890]_T2w.nii.gz',opts={"items":1})
flab = gfile(dana,'^s.*-drawem9_dseg.nii.gz')
sess = get_parent_path(dana,level=2)[1]
suj =  get_parent_path(dana,level=3)[1]
sujid = [f'{su}_{se}' for su,se in zip(suj,sess)]
df_file=pd.DataFrame()
df_file["sujname"] = sujid
df_file["label_name"] = flab
df_file["vol_T1"] = fT1
df_file["vol_T2"] = fT2


#attention ne marche pas car plusieur volume
#df_file["sujname"] = df.apply(lambda r: get_suj_id(r, dataset='feta2'), axis=1 )
df_file["sujname"] = df.apply(lambda r: get_suj_id(r), axis=1 )
df_file["vol_name"] = df.apply(lambda r: get_file_path(r,"anat_path","*T2*gz"), axis=1 )
df_file["vol_T1"] = df.apply(lambda r: get_file_path(r,"anat_path","*T1*gz"), axis=1 )
df_file["label_name"] = df.apply(lambda r: get_file_path(r,"anat_path","s*drawem9_*gz"), axis=1 )

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

#explore on suj
def get_slice_dice(suj, nbs, orient):
    if orient=="sag":
        dl = suj.lab.data[:, [nbs], :,: ].unsqueeze(0) ; dp = suj.pred.data[:, [nbs], :,: ].unsqueeze(0)
        di = suj.din.data[:, [nbs], :,: ].unsqueeze(0)
    elif orient=="ax":
        dl = suj.lab.data[..., [nbs]].unsqueeze(0) ; dp = suj.pred.data[..., [nbs]].unsqueeze(0) #axial
        di = suj.din.data[..., [nbs]].unsqueeze(0)
    return met_overlay(dp, dl), dl, dp, di

thot = tio.OneHot()
dfsub = df[df['sujnum']==1]
dfsub = df[df['sujnum']==6]
nbs, orient = 145, 'sag'
nbs, orient = 179, 'ax'

fig = plt.figure('1')
legend_str=[]
for dfser in dfsub.iterrows():
    dfser = dfser[1]
    suj = tio.Subject({'lab':tio.LabelMap(dfser.flabel), "pred":tio.LabelMap(dfser.fpred) , "din":tio.ScalarImage(dfser.finput) })
    suj = thot(suj)
    res = met_overlay(suj.pred.data.unsqueeze(0), suj.lab.data.unsqueeze(0))
    slice_dice = []
    print(f'{dfser.metric_dice_GM:.3} GM dicm from {dfser.model} ')
    legend_str.append(dfser.model)
    dice_slice, dl, dp, di = get_slice_dice(suj, nbs, orient)
    print(f'Dice slice {dice_slice}')
    #plt.figure();    plt.imshow(dp[0, 2, :, :, 0])  # axial
    #plt.figure();    plt.imshow(dp[0, 2, 0, ...])  # sag

    gmi =  dp[:,[2],:]>0;    gmi = di[gmi].flatten().numpy()
    gmil = dl[:, [2], :] > 0;    gmil = di[gmil].flatten().numpy()
    bins = np.linspace(0, 1, 30)
    fig = plt.figure(); fig.set_size_inches(10, 7); plt.hist([gmi, gmil], bins);
    plt.title(f'{dfser.model} on {dfser.eval_on}',fontsize='xx-large')
    ax = plt.gcf().get_axes()[0]
    plt.ylabel('nb voxel', fontsize='xx-large'); plt.xlabel('Signal Intensity ', fontsize='xx-large')
    yy = ax.get_yticklabels();    ax.set_yticklabels(yy, fontsize='xx-large')
    yy = ax.get_xticklabels();    ax.set_xticklabels(yy, fontsize='xx-large')

    if dfser.eval_on=='T2':
        plt.legend(['prediction', 'label'], fontsize='xx-large', loc='upper right')
    else:
        plt.legend(['prediction', 'label'], fontsize='xx-large', loc='upper left')
    plt.tight_layout()

    print(f'mean WM {di[dl[:,[3],:]>0].mean()} mean CSF {di[dl[:,[1],:]>0].mean()} mean GM {di[dl[:,[2],:]>0].mean()}')

    plt.figure(); plt.hist(gmi, bins=100)
    plt.figure(); plt.imshow(dp[0,2,0,:,:],origin='lower') #sagital

#    for nbs in range(0, suj.lab.data.shape[3]): # [3] for axial zslice [1] for sag
    for nbs in range(0, suj.lab.data.shape[1]): # [3] for axial zslice [1] for sag
        #dl = suj.lab.data[...,[nbs]].unsqueeze(0);        dp = suj.pred.data[...,[nbs]].unsqueeze(0)
        dl = suj.lab.data[:, [nbs], :, :].unsqueeze(0);        dp = suj.pred.data[:, [nbs], :, :].unsqueeze(0)

        slice_dice.append(met_overlay(dp, dl))
    fig = plt.figure('1'); plt.plot(slice_dice)
plt.legend(legend_str)

suj = tio.Subject({'predt2':tio.LabelMap('/data/romain/PVsynth/eval_cnn/baby/Article/eval_T2_model_5suj_motBigAff_ep30/S001_sub-CC00834XX18_ses-21210/bin_prediction.nii.gz'),
                   "predt1":tio.LabelMap('/data/romain/PVsynth/eval_cnn/baby/Article/eval_T1_coreg_model_5suj_motBigAff_ep30/S001_sub-CC00834XX18_ses-21210/bin_prediction.nii.gz')})
suj = tio.Subject({'predt2':tio.LabelMap('/data/romain/PVsynth/eval_cnn/baby/Article/eval_T2_model_5suj_motBigAff_ep30/S005_sub-CC00634AN16_ses-184100/bin_prediction.nii.gz'),
                   "predt1":tio.LabelMap('/data/romain/PVsynth/eval_cnn/baby/Article/eval_T1_coreg_model_5suj_motBigAff_ep30/S005_sub-CC00634AN16_ses-184100/bin_prediction.nii.gz')})
suj = thot(suj)
#dl = suj.predt1.data[:, [nbs], :,: ].unsqueeze(0) ; dp = suj.predt2.data[:, [nbs], :,: ].unsqueeze(0) #sag
dl = suj.predt1.data[..., [nbs]].unsqueeze(0) ; dp = suj.predt2.data[..., [nbs]].unsqueeze(0) #ax
print(f'Dice slice {met_overlay(dp, dl)}')

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

#ground truth marseille
suj_gt = ['sub-0307_ses-0369', 'sub-0427_ses-0517', 'sub-0457_ses-0549', 'sub-0483_ses-0589', 'sub-0567_ses-0681','sub-0665_ses-0791'] #'sub-0179_ses-0212',
dfdata = pd.read_csv('/data/romain/baby/marseille/file_6suj_GT.csv')
dir_pred = glob.glob('/data/romain/baby/marseille/fetal/derivatives/segmentation/prediction_template_space/*')
dir_pred = glob.glob('/data/romain/baby/marseille/fetal/derivatives/segmentation/prediction_template_masked//*')
dir_pred = [dir_pred[i] for i in (1,2,3,4,5,7)]

pred_name = get_parent_path(dir_pred)[1]
gt_files, sr_files = dfdata.ground_truth.values, dfdata.srr_tpm
#gt_files = gfile('/data/romain/baby/marseille/fetal/derivatives/segmentation/prediction_template_masked/5suj_hcp_T2_nextela_model_fromBigAff_ep1','.nii')
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

df, cmd = pd.DataFrame(), []
for nb_pred, one_dir_pred in enumerate(dir_pred):
    #one_dir_pred = dir_pred[0]
    all_file = gfile(one_dir_pred,'nii')
    print(f'working on {one_dir_pred}')
    for ii,  one_pred in enumerate(all_file):
        gt_file, sr_file = gt_files[ii], sr_files[ii]

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
g = sns.relplot(data=df,x='suj',y='GM',hue='model')
sns.move_legend(g,'upper right')

 display_res(dir_pred,bg_files=sr_files, gt_files=gt_files)