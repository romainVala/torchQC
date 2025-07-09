
import torch,numpy as np,  torchio as tio
from utils_metrics import compute_metric_from_list #get_tio_data_loader, predic_segmentation, load_model, computes_all_metric
from timeit import default_timer as timer
import json, os, seaborn as sns, shutil
from utils_file import gfile, gdir, get_parent_path, addprefixtofilenames, r_move_file
import pandas as pd
from nibabel.viewers import OrthoSlicer3D as ov
from utils_labels import remap_filelist, get_fastsurfer_remap
import matplotlib.pyplot as plt
from script.create_jobs import create_jobs

plt.interactive(True)
sns.set_style("darkgrid")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def get_min_max(df,column, condition ):
    dfs = df[condition]
    if len(dfs)==len(dfs.subject_id.unique()):
        print('subset as big as subject_id')
    else:
        print(f'may be to big dfs {len(dfs)} but only {len(dfs.subject_id)} sujid')
    dfs = dfs.sort_values(by=column)

    print(f'max is {dfs[column].values[-1]} for suj {dfs.subject_id.values[-1]}')
    print(f'max is {dfs[column].values[-2]} for suj {dfs.subject_id.values[-2]}')
    print(f'max is {dfs[column].values[-3]} for suj {dfs.subject_id.values[-3]}')
    print(f'min is {dfs[column].values[0]} for suj {dfs.subject_id.values[0]}')

def get_confu_suj_average(df, lab_col, labels):
    suj_avg = {}
    li = lab_col
    for lj in labels:
        if li == lj:
            continue
        if (li == 'head') & (lj == 'BG'):
            continue
        confu_name = f'confusion_GT_{li}_P_{lj}'
        subject_average = df[confu_name].mean()
        val_max = df[confu_name].max()
        if subject_average == 0:
            print(f'Skiping {lj} never predicted for {li}')
        elif val_max<0.1:
            print(f'Skiping {lj} max is {val_max} for {li}')
        else:
            suj_avg[confu_name] = subject_average

    suj_avg = dict(sorted(suj_avg.items(), key=lambda item: item[1], reverse=True))
    short_name = [ll[len(f'confusion_GT_{lab_col}_P_'):] for ll in suj_avg.keys()]
    return suj_avg, short_name
def norm_conf(df, lab):
    dfo = df.copy()
    for li in lab:
        ysum = 0
        for lj in lab:
            if li == lj:
                continue
            if (li == 'head') & (lj == 'BG'):
                continue
            ysum += df[f'confusion_GT_{li}_P_{lj}']

        for lj in lab:
            dfo[f'confusion_GT_{li}_P_{lj}'] /= ysum
    return dfo
def get_confu_all_lab(df,lab,onelab):
    all_key, all_key_short = [], []
    for ll in lab:
        if ll == onelab:
            continue
        all_key.append(f'confusion_GT_{ll}_P_{onelab}')
        all_key.append(f'confusion_GT_{onelab}_P_{ll}')
        all_key_short.append(f'G_{ll}')
        all_key_short.append(f'P_{ll}')
    return all_key,all_key_short
def get_confu_all_lab_meanByDS(df,lab,onelab):
    dsname = list(df.model_name.unique())
    all_res = {}
    for dsn in dsname:
        dfs = df[df.model_name==dsn]
        dic_res = {}
        for ll in lab:
            if ll==onelab:
                continue
            dic_res[f'C_{onelab}_P_{ll}'] =  dfs[f'confusion_GT_{onelab}_P_{ll}'].mean()
            dic_res[f'C_{ll}_P_{onelab}'] =  dfs[f'confusion_GT_{ll}_P_{onelab}'].mean()

        all_res[dsn] = dict(sorted(dic_res.items(), key=lambda item: item[1], reverse=True))
    return all_res
def sel_df_model(dfo,value_list, col_name='model_name'):
    sel = df.model_name == 'qsdf'
    for mm in value_list:
        sel = sel | (df[col_name] == mm)
    return dfo[sel]
def change_df_model_name_withDS(df, dsname='HCP_test_retest_07mm_suj82_vol_T2_07_free_Ass_siam',ds_newname='T2'):
    sel_ds = df['dataset_name'] == dsname
    for mm in df[sel_ds].model_name.unique() :
        sel_mod = (sel_ds) & (df['model_name']==mm)
        df.loc[sel_mod,'model_name'] = f"{mm}_{ds_newname}"
        print(f'changin {sel_mod.sum()}')
    return df

def get_data(name='hcp'):
    dunt = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/'
    match name:
        case 'hcp':
            fcsv = gfile(dunt + 'csv_validationHCP_test/results/','csv$')
        case 'hcpreg':
            fcsv = gfile(dunt + 'csv_validationHCP_test/results_region/','csv$')
            name='hcp'
        case 'hcp_confu':
            fcsv = gfile(dunt + 'csv_validationHCP_test/results_confu/','csv$')
        case 'ultra':
            fcsv = gfile(dunt + 'csv_validationULTRA/results/','csv$')
        case 'dbb':
            fcsv = gfile(dunt + 'csv_validationDBB/results/','csv$')
        case 'miccai':
            fcsv = gfile(dunt + 'csv_validationHCP_MICCAI/results/','csv$')
        case 'ultracortex':
            fcsv = gfile(dunt + 'csv_validation_ultracortex/results/','GTrib.*csv$')
        case 'dhcp':
            fcsv = gfile(dunt + 'csv_validationdHCP/results/','csv$')

    df_list=[]
    for ff in fcsv:
        df = pd.read_csv(ff)
        fname = get_parent_path(ff)[1]
        ii = fname.find('_lab_')
        label_name = f'{fname[ii+5:-4]}'
        df['label'] =label_name

        ii = fname.find('_mask_')
        if ii>0:
            ind_underscore = fname[ii + 6:].find('_')
            label_name = f'{fname[(ii + 6):(ii+6+ind_underscore)]}'
            df['region'] = label_name
        df_list.append(df)
    df = pd.concat(df_list)

    match name:
        case 'hcp':
            df['sujnum'] = [int(ss[25:27]) for ss in df.subject_id]  # HCP
            morder = ['FastSurfer', 'SynthSegGouhfi', 'SynthSeg', 'pred_DS715_NODA3ResXLres', 'pred_DS708_5nnResXL_res',
                      'pred_DS712_NODA3ResXLres', 'pred_DS714_5ResXLres', 'pred_DS713_3ResXLres']
            mordernn = ['FastSurfer', 'Gouhfi', 'SynthSeg', 'SIAM', 'SynthSkull', 'SynthVasc', 'SynthMIDA3',
                        'SynthMIDA']

        case 'hcp_confu':
            fcsv = gfile(dunt + 'csv_validationHCP_test/results_confu/','csv$')
            df['sujnum'] = [int(ss[25:27]) for ss in df.subject_id]  # HCP
            morder = ['FastSurfer', 'SynthSegGouhfi', 'SynthSeg', 'pred_DS715_NODA3ResXLres', 'pred_DS708_5nnResXL_res',
                      'pred_DS712_NODA3ResXLres', 'pred_DS714_5ResXLres', 'pred_DS713_3ResXLres']
            mordernn = ['FastSurfer', 'Gouhfi', 'SynthSeg', 'SIAM', 'SynthSkull', 'SynthVasc', 'SynthMIDA3',
                        'SynthMIDA']

        case 'ultra':
            df.index = range(len(df))
            sujnum = []
            for ii, rr in df.iterrows():
                ss = rr['subject_id']
                ii = ss.find('ULTRA_')
                # print(f'find {ii} for {ss}')
                sujnum.append(int(ss[ii + 6:ii + 9]))
            df['sujnum'] = sujnum
            recount_suj=False
            if recount_suj:
                df['sujnum'] = [int(ss[-3:]) for ss in df.subject_id]  # ULTRA
                ssnn = np.sort(df.sujnum.unique())
                for i, j in zip(ssnn, range(len(ssnn))):
                    df.loc[df.sujnum == i, 'sujnum'] = j + 1
            # ULTRA_all
            morder = ['pred_DS715_NODA3ResXLres', 'pred_DS708_5nnResXL_res', 'pred_DS712_NODA3ResXLres',
                      'pred_DS714_5ResXLres', 'pred_DS713_3ResXLres']
            morder = ['pred_DS715_NODA3ResXLres', ['pred_DS708_5nnResXL_res','pred_DS708_3d_fullres_nnUNetTrainer_nnUNetResEncUNetXLPlans'],
                      ['pred_DS712_5ResXLres','pred_DS712_NODA3ResXLres'], 'pred_DS714_5ResXLres', 'pred_DS713_3ResXLres']
            mordernn = ['SIAM', 'SynthSkull', 'SynthVasc', 'SynthMIDA3', 'SynthMIDA']
            # ultra morder = ['pred_DS715_NODA3ResXLres','pred_DS708_3d_fullres_nnUNetTrainer_nnUNetResEncUNetXLPlans','pred_DS712_5ResXLres','pred_DS714_5ResXLres','pred_DS713_3ResXLres']
            # mordernn = ['SIAM', 'SynthSkull','SynthVasc', 'SynthMIDA3','SynthMIDA']

        case 'dbb':
            fcsv = gfile(dunt + 'csv_validationDBB/results/','csv$')
        case 'miccai':
            df['sujnum'] = [int(ss[12:14]) for ss in df.subject_id]  # miccai
        case 'ultracortex':
            fcsv = gfile(dunt + 'csv_validation_ultracortex/results/','GTrib.*csv$')
        case 'dhcp':
            # dHCP
            morder = ['SynthSegGouhfi', 'SynthSeg', 'pred_DS715_NODA3ResXLres', 'pred_DS708_5nnResXL_res',
                      'pred_DS712_5ResXLres', 'pred_DS714_5ResXLres', 'pred_DS713_3ResXLres', 'wmEM_mot_ep240']
            mordernn = ['Gouhfi', 'SynthSeg', 'SIAM', 'SynthSkull', 'SynthVasc', 'SynthMIDA3', 'SynthMIDA', 'SynthBaby']

    for m1, m2 in zip(morder, mordernn):
        if isinstance(m1,list):
            for mm in m1:
                df.model_name.replace(mm, m2, inplace=True)
        else:
            df.model_name.replace(m1, m2, inplace=True)

    return df,morder,mordernn

df,morder,mordernn = get_data('hcp_confu')

ymet = []
for k in df.keys():
    # if 'Sdis_' in k:
    if 'dice' in k:
        ymet.append(k)
# for dice ymet.pop(-1);ymet.pop(0);ymet.pop(1);ymet.pop(1)

#morder = ['FastSurfer', 'SynthSegGouhfi','SynthSeg','pred_DS715_3ResXLres','pred_DS708_5nnResXL_res','pred_DS712_5ResXLres','pred_DS714_5ResXLres','pred_DS713_3ResXLres']
#morder = ['FastSurfer', 'SynthSegGouhfi','SynthSegGouhfiBM','SynthSeg','pred_DS715_3ResXLres','pred_DS708_5nnResXL_res','pred_DS712_5ResXLres','pred_DS714_5ResXLres','pred_DS713_3ResXLres']
#mordernn = ['FastSurfer','Gouhfi_BM','Gouhfi','SynthSeg', 'SIAM', 'SynthSkull','SynthVasc', 'SynthMIDA3','SynthMIDA']
mordernn2= ['FastSurfer', 'SIAM', 'SynthSkull','SynthVasc', 'SynthMIDA3','SynthMIDA']
#mordernn3= ['FastSurfer','Gouhfi','SynthSeg','SynthSegI', 'SIAM',]




#DBB group CSFv
df['group'] = 'bV' ; df.loc[df.index<12.5,'group'] = 'sV';  # gros mais pas assez df.loc[df.index==7,'group'] = 'Vs';
df = df.drop(15)#too extrem ... almost no brain
##ultracortex group
df['group'] = 'uni'; df.loc[(df.index == 2) | (df.index == 5) | (df.index == 7) ,'group'] = 'sV'

volu = [df[k].mean() for k in ymetv]
tname = [f'{ss[14:]} V={y/min(volu):0.1f}' for y,ss in sorted(zip(volu, ymetv),reverse=True)]
ymets = [ss for y,ss in sorted(zip(volu, ymet),reverse=True) ]


sns.catplot(data=df, y='dice_GM', x='label', col='dataset_name', hue='model_name', col_wrap=3, kind='boxen') #hue_order=xorder                        col_order=corder)

dfmm = df.melt(id_vars=['sujnum', 'model_name', 'dataset_name', 'label'], value_vars=ymet, var_name='from', value_name='dice');
sns.catplot(data=dfmm, y='dice', x='dataset_name',kind='boxen', col='from',col_wrap=1)
fig = sns.catplot(data=dfmm, y='dice', x='label', col='from', hue='model_name', kind='boxen',hue_order=mordernn[:4],col_wrap=3, palette=cc)


c1 = sns.color_palette()
c2 = sns.color_palette("Paired")
cc = [c1[0],c1[1],c1[3], c1[2], c2[2], c2[11], c1[4], c1[6],]
cc2 = [c1[0], c1[2], c2[2], c2[11], c1[4], c1[6],]
morder2 = ['FastSurfer', 'Gouhfi', 'Gouhfi_T2', 'SynthSeg', 'SynthSeg_T2', 'SIAM','SIAM_T2','SynthSkull','SynthSkull_T2', 'SynthVasc','SynthVasc_T2',
        'SynthMIDA3','SynthMIDA3_T2', 'SynthMIDA','SynthMIDA_T2']
cc = [c1[0],c1[1],c1[1],c1[3],c1[3], c1[2],c1[2], c2[2],c2[2], c2[11],c2[11], c1[4],c1[4], c1[6],c1[6]]

cg = (0.75, 0.35, 0.025)
cc = [c1[0],cg,c1[1],c1[3], c1[2], c2[2], c2[11], c1[4], c1[6],]

# GM
dfs = df[(df.model_name=='SIAM') & (df.label=='Free')]
#fig volume
dfs = df[(df.label=='Free')]# | (df.label=='Assn')]
#dfss = dfs[(dfs.label=='Assn')&(dfs.dataset_name=='HCP_test_retest_07mm_suj82_vol_T1_07')]#&(dfs.model_name=='SIAM')]
dfs = dfs.sort_values(by='volume_target_CSFv')

fig=plt.figure();plt.plot(range(82),dfss.volume_target_CSFv*0.7**3/1000);plt.ylabel('total Ventricle volume in cm^3');plt.xlabel('Subjects')
fig=sns.relplot(data=dfs, y='volume_ratio_CSFv', x='subject_id', col='label', hue='model_name', kind='line',hue_order=mordernn[:4], palette=cc)
ax = fig.axes[0][0];plt.xlim([0, 82 ]); ax.set_xticklabels('', rotation=0,fontsize='x-large' )
sns.move_legend(fig,'right',bbox_to_anchor=(.8, .8),fontsize='large',frameon=True, shadow=True, title=f'Model',title_fontsize='large')
ax.set_xticklabels('', rotation=0,fontsize='x-large' );ax.set_xlabel('Subject order by increasing Ventricle Volume',fontsize='x-large')
ax.set_ylabel(f' (Vol predict) / (Vol GT Free) ',fontsize='x-large');yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize='large' )
ax.set_title('Ventricle', fontsize='x-large')


#Figure Ultracortex
ax = fig.axes[0][0]
ax.set_title('GM', fontsize='x-large')
ax.set_ylabel(f'Dice',fontsize='x-large');ax.set_xlabel('',fontsize='x-large'); ax.set_xticklabels('')
yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize='large' )
sns.move_legend(fig,'right',bbox_to_anchor=(.9, .2),fontsize='large',frameon=True, shadow=True, title=f'Model',title_fontsize='large')

fig.savefig('GM_ultra_cortex.png')

#Figure HCP
ymet =['dice_GM','dice_cerGM','dice_CSFv', 'dice_thal', 'dice_Put', 'dice_hypp', 'dice_Cau-acc', 'dice_Pal', 'dice_amyg']
ctitle = ['GM V=203', 'Cerebellum GM V=41','Ventricle V=6.4', 'thalamus V=6.0','Putamen V=3.6','Hippocampus V=2.9','Caudate-Accubens V=2.8','Palidum V=1.3','Amygdala V=1']#HCP
ctitle = ['GM V=223', 'Cerebellum GM V=45','Ventricle V=22', 'thalamus V=6.2','Putamen V=3.7','Hippocampus V=3.1','Caudate-Accubens V=3','Palidum V=1.3','Amygdala V=1']#MICCAI

#df = pd.concat([pd.read_csv(ff) for ff in fcsv])
fig = sns.catplot(data=dfmm, y='dice', x='label', col='from', hue='model_name', kind='boxen',hue_order=mordernn[:4],col_wrap=3, palette=cc)

for ii, ax in enumerate(fig.axes):
    ax.set_title(ctitle[ii], fontdict=dict(fontsize='x-large'))
    if (ii==0) | (ii==3)| (ii==6):
        ax.set_ylabel(f'Dice',fontsize='x-large') #'Dice'   'Average Surface dist' 'Volume Ratio'
        yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize='x-large' )
    if ii>5:
        ax.set_xticklabels(['Manual GT','Ass GT','Free GT'], rotation=0,fontsize='x-large' );ax.set_xlabel(' ')

sns.move_legend(fig,'right',bbox_to_anchor=(.9, .3),fontsize='large',frameon=True, shadow=True, title=f'model ',title_fontsize='x-large')
fig.savefig('dice_all_4mod.png')


#DBB fig
ctitle = ['GM', 'Deep Nucleus']
for ii, ax in enumerate(fig.axes[0]):
    ax.set_title(ctitle[ii], fontdict=dict(fontsize='x-large'))
    if (ii==0) | (ii==3)| (ii==6):
        ax.set_ylabel(f'Dice',fontsize='x-large') #'Dice'   'Average Surface dist' 'Volume Ratio'
        yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize='x-large' )
    if ii>=0:
        ax.set_xticklabels(['N=14', 'large Ventricle (N=4)'], rotation=0,fontsize='x-large' );ax.set_xlabel(' ')


for yy in ymet:
    fig = sns.catplot(data=df, y=yy, x='label', col='from', hue='model_name', kind='boxen',
                      hue_order=mordernn[:4], col_wrap=3, palette=cc)


#confusion mat


mordernn2 = ['FastSurfer', 'Gouhfi', 'SIAM', 'SynthSkull', 'SynthVasc', 'SynthMIDA3', 'SynthMIDA']
#mordernn2 = ['FastSurfer', 'Gouhfi', 'SIAM']
ymet=[]
for k in df.keys():
    #if 'Sdis_' in k:
    if 'dice' in k:
            ymet.append(k)
ymet.pop(-1)
labels =[ss[5:]  for ss in ymet] #BG to head
labels2 = ['GM','CSFv','CSF','cerGM','head']
dfo, dfonorm = sel_df_model( df, mordernn2 ), sel_df_model( norm_conf(df,labels), mordernn2 )
dfo = change_df_model_name_withDS(dfo,dsname='HCP_test_retest_07mm_suj82_vol_T2_07_SIAM',ds_newname='T2')
mordernn2 = list(dfo.model_name.unique())

for li in labels2:
    ycmet, yynn = get_confu_suj_average(dfonorm,li, labels); ycmet = list(ycmet.keys())
    dfmm = dfo.melt(id_vars=['sujnum', 'model_name', 'dataset_name', 'label'], value_vars=ycmet, var_name='from',value_name='c');
    for m1,m2 in zip(ycmet, yynn):
        dfmm['from'].replace(m1,m2,inplace=True)

    fig = sns.catplot(data=dfmm, y='c',x='from', hue='model_name',kind='boxen', hue_order=mordernn2)
    #fig = sns.catplot(data=dfmm, y='c',x='from',col='dataset_name', hue='model_name',kind='boxen',hue_order=mordernn2, col_wrap=1, height=5, aspect=2)
    plt.title(f'Confu {li}')

lab_sel = ['BG', 'WM', 'CSF', 'head']
lab_sel = ['CSFv', 'cerGM','thal','Pal', 'Put', 'Cau-acc', 'amyg', 'hypp']
c1 = sns.color_palette()
c2 = sns.color_palette("Paired")
morder2 = ['FastSurfer', 'Gouhfi', 'Gouhfi_T2','SIAM_T2','SynthSkull','SynthSkull_T2', 'SynthVasc','SynthVasc_T2',
        'SynthMIDA3','SynthMIDA3_T2', 'SynthMIDA','SynthMIDA_T2']
cc = [c1[0],c1[1],c1[1],c1[2], c2[2],c2[2], c2[11],c2[11], c1[4],c1[4], c1[6],c1[6]]

ycmet, ycmet_short = get_confu_all_lab(df,lab_sel,'GM')
dfmm = dfo.melt(id_vars=['sujnum', 'model_name', 'dataset_name', 'label'], value_vars=ycmet, var_name='from',value_name='c');
for m1,m2 in zip(ycmet, ycmet_short):
    dfmm['from'].replace(m1,m2,inplace=True)
fig = sns.catplot(data=dfmm, y='c',x='from', hue='model_name',kind='boxen', hue_order=morder2, palette=cc)

#checked with mrtrix S01 HCP
#confusion_GT_CSF_P_GM 53191   siamT1==3 & siamT2==1
# confusion_GT_GM_P_CSF 9802   siamT1==1 & siamT2==3