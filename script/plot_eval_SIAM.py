import torch,numpy as np,  torchio as tio
from utils_metrics import compute_metric_from_list #get_tio_data_loader, predic_segmentation, load_model, computes_all_metric
from timeit import default_timer as timer
import json, os, seaborn as sns, shutil
import scipy, statsmodels
from utils_file import gfile, gdir, get_parent_path, addprefixtofilenames, r_move_file
from utils_metrics import mrview_overlay
import pandas as pd
from nibabel.viewers import OrthoSlicer3D as ov
from utils_labels import get_remapping, remap_filelist, get_fastsurfer_remap
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from script.create_jobs import create_jobs
from scipy.stats import ttest_ind,wilcoxon

plt.interactive(True)
sns.set_style("darkgrid"); sns.set_context('poster')
pd.set_option('display.max_rows', 500);pd.set_option('display.max_columns', 500);pd.set_option('display.width', 1000)

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
def get_met(df,regex=None,regstart=None, exclude=None):
    ymet=[]

    if isinstance(regex,list):
        for rr in regex:
            ymet += get_met(df,regex=rr,exclude=exclude)
        return ymet
    if isinstance(regstart,list):
        for rr in regstart:
            ymet += get_met(df,regstart=rr,exclude=exclude)
        return ymet

    if regstart is not None:
        for k in df.keys():
            if k.startswith(regstart):
                ymet.append(k)
    else:
        for k in df.keys():
            if regex in k:
                ymet.append(k)
    if exclude is not None:
        if not isinstance(exclude,list):
            exclude=[exclude]
        ind_exclude=[]
        for ii,yy in enumerate(ymet):
            ex_y = False
            for xxx in exclude:
                if xxx in yy:
                    ex_y=True
                    break
            if ex_y:
                ind_exclude.append(ii)
        ind_exclude.reverse()
        for ii in ind_exclude:
            ymet.pop(ii)

    return ymet
def sum_vol(df,ymet,outname='volume_brain'):
    df[outname] = df[ymet[0]]
    for yy in ymet[1:]:
        df[outname] += df[yy]
def scale_vol_to_cm(df):
    # only take resolution from the firs
    il = tio.LabelMap(df.pred_path.values[1])
    scale_factor = 1000/(np.array(il.spacing).prod() ) #1/voxel_size*1e-3

    scale_vol(df,get_met(df,regstart=['volume_input','volume_target']), scale=scale_factor);

def scale_vol(df,ymet,scale=1000):
    for yy in ymet:
        df[yy] /= scale
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
def get_confu_all_lab(df,lab,onelab,type='confusion'):
    all_key, all_key_short, all_lab = [], [], []
    if type=="confusion":
        if len(lab)==0:
            k_diag = f'confusion_GT_{onelab}_P_{onelab}'
            for kk in df.keys():
                if 'confusion' in kk:
                    if f'_{onelab}' in kk:
                        if k_diag==kk:
                            continue
                        all_key.append(kk)
                    lll = f'confusion_GT_{onelab}_P_'
                    if kk.startswith(lll):
                        all_lab.append(kk[len(lll):])
            return all_key, all_lab, k_diag
        else:
            for ll in lab:
                if ll == onelab:
                    continue
                all_key.append(f'confusion_GT_{ll}_P_{onelab}')
                all_key.append(f'confusion_GT_{onelab}_P_{ll}')
                all_key_short.append(f'GT_{ll}')
                all_key_short.append(f'Pred_{ll}')
    elif type=="conf_sum":
        for ll in lab:
            if ll == onelab:
                continue
            all_key.append(f'{ll}_err_{onelab}')
            all_key_short.append(f'{ll}_ass_{onelab}')

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
def change_df_model_name_withDS(df, dsname=None,ds_newname='T2', ds_col="dataset_name"):
    if dsname==None:
        for dsn in df[ds_col].unique():
            if ds_newname in dsn:
                dsname = dsn
                break

    sel_ds = df[ds_col] == dsname
    for mm in df[sel_ds].model_name.unique() :
        sel_mod = (sel_ds) & (df['model_name']==mm)
        df.loc[sel_mod,'model_name'] = f"{mm}_{ds_newname}"
        print(f'changin {sel_mod.sum()}')
    return df
def group_by_region_df(df):
    # sum all region
    yconf, yother = [], []
    for k in df.keys():
        if "confusion" in k:
            yconf.append(k)
        else:
            yother.append(k)
    aggdic = {kk: 'sum' for kk in yconf}
    aggdic.update({kk: 'first' for kk in yother})
    # dfg = dfo.groupby(['model_name','label', 'dataset_name','subject_id'], as_index=False).sum(numeric_only = True)
    dfg = df.groupby(['model_name', 'label', 'dataset_name', 'subject_id'], as_index=False).agg(aggdic)
    return dfg
def get_confusion_matrix_one_row(dfser,labels):
    mat_conf = np.zeros([len(labels),len(labels)])
    for ii,i in enumerate(labels):
        for jj,j in enumerate(labels):
            mat_conf[ii,jj] = dfser[f"confusion_GT_{i}_P_{j}"]
    return mat_conf
def get_sum_confusion(dfser, label_idx, sum_axis=0):
    mat_conf = dfser['mat_conf']
    sum_all = np.nansum(mat_conf, axis=sum_axis)
    return sum_all[label_idx]
def get_metric_from_confusion(df,labels=None):
    if labels is None:
        kconf,labels,kdiag = get_confu_all_lab(df,'','GM')
        labels.append('GM')

    df['mat_conf'] = df.apply(lambda x:  get_confusion_matrix_one_row(x, labels), axis=1)
    for ii,i in enumerate(labels):
        df[f"volume_target_{i}"] = df.apply(lambda x: get_sum_confusion(x,ii,1), axis=1)
        df[f"volume_input_{i}"] = df.apply(lambda x: get_sum_confusion(x,ii,0), axis=1)
        df[f"volume_ratio_{i}"] = df[f"volume_input_{i}"] / df[f"volume_target_{i}"]
    return df
def norm_confusion(df,one_label, do_norm=None):
    df = df.copy()
    df = df.fillna(0)
    kconf,kmet,kdiag = get_confu_all_lab(df,'',one_label)
    df[f'sum_error_{one_label}'] = 0
    for kk in kconf:
        df[f'sum_error_{one_label}'] += df[kk]

    vol_tot = df[f'sum_error_{one_label}'] + 2*df[kdiag]
    df[f'conf_dice'] = 2*df[kdiag] / vol_tot
    df[f'vol_tot'] = vol_tot
    if do_norm=='volume_first': #mauvaise comprehen il faut diviser par 2 fois volGM pour avoir 1-Dice !
        for kk in kconf:
            df[kk] /= (vol_tot/2) / 100  #ARG cree des nan si vol_tot == 0
    if do_norm=='volume':
        for kk in kconf:
            df[kk] /= (vol_tot) / 100  #ARG cree des nan si vol_tot == 0
    if do_norm=='errors':
        for kk in kconf:
            df[kk] /= df[f'sum_error_{one_label}'] / 100  #ARG cree des nan si vol_tot == 0
    return df
def corect_confusion_head_bg(df):
    dfc = df.copy()
    dfc = dfc.fillna(0)
    dfc['confusion_GT_GM_P_CSF'] = dfc['confusion_GT_GM_P_CSF'] + dfc['confusion_GT_GM_P_BG'] + dfc['confusion_GT_GM_P_head']
    dfc['confusion_GT_CSF_P_GM'] = dfc['confusion_GT_CSF_P_GM'] + dfc['confusion_GT_BG_P_GM'] + dfc['confusion_GT_head_P_GM']
    dfc['confusion_GT_GM_P_BG'] = dfc['confusion_GT_GM_P_head'] = dfc['confusion_GT_BG_P_GM'] = dfc['confusion_GT_head_P_GM'] = 0
    return dfc
def get_confusion_ratio(df,lab, onelab):
    for ll in lab:
        df[f'Cratio_{ll}'] = df[f'confusion_GT_{ll}_P_{onelab}']  / (df[f'confusion_GT_{ll}_P_{onelab}'] +  df[f'confusion_GT_{onelab}_P_{ll}']) *100
        df[f'{ll}_err_{onelab}'] = df[f'confusion_GT_{onelab}_P_{ll}'] + df[f'confusion_GT_{ll}_P_{onelab}']
        df[f'{ll}_ass_{onelab}'] = (df[f'confusion_GT_{ll}_P_{onelab}'] - df[f'confusion_GT_{onelab}_P_{ll}']) / df[f'{ll}_err_{onelab}'] * 100

    return df
def norm_confusion_from(df,norm_colname ,sel_lab=None):
    if sel_lab is None:
        sel_lab=[]
        for k in df.keys():
            if 'confusion' in k:
                sel_lab.append(k)
    dfo = df.copy()
    for lab in sel_lab:
        dfo[lab] = dfo[lab]/dfo[norm_colname] * 100
    return dfo
def select_df(df, sel_factor):
    for k, v in sel_factor.items():
        #print(v)
        #print(not isinstance(v,list))
        if not isinstance(v,list):
            v = [v]
        sel_value_list=[]
        for vv in v:
            for sel_key in df[k].unique():
                if vv in sel_key:
                    sel_value_list.append(vv)
                    break
        for ii,sel_key in enumerate(sel_value_list):
            if ii==0:
                ind_sel = (df[k] == sel_key)
            else:
                ind_sel = ind_sel | (df[k] == sel_key)
            print(f'select {k}=={sel_key}           nb ligne {(df[k] == sel_key).sum()}')
        df = df[ind_sel]
    print(f"final shape is {df.shape}")
    return df
def select_df_ask(df):
    factor_col = ['model_name','label_column','input_type','region','group']
    all_keys=df.keys()
    factor_sel={}
    for factor in factor_col:
        if factor in all_keys:
            choice_list = df[factor].unique()
            if len(choice_list)==1:
                print(f"one dim factor {choice_list} ")
                continue

            print(f'choose {factor}')
            for kk,cc in enumerate(choice_list):
                print(f'{kk+1}  : {cc}')
            user_choice = input()
            if user_choice:
                user_choice = int(user_choice)-1
                factor_sel.update({factor:choice_list[user_choice]})
    #print(f'selected factor {factor_sel} ')
    return select_df(df,factor_sel), factor_sel
def compare_pred_mrview(df, sort_key=None, sel_factor=None, ascending=False,nb_ex=5, select_label=1,mrview_cmd=True):
    def get_col_list_value(df,key_list):
        str_out =''
        for kk in key_list:
            str_out += f"{df[kk]}   "
        return str_out
    if sel_factor:
        df = select_df(df,sel_factor)
    if sort_key:
        if not isinstance(sort_key, list):
            sort_key=[sort_key]
        df = df.sort_values(by=sort_key[0], ascending=ascending)

    #print mrview cmd
    nbrow=0
    for ii,dfser in df.iterrows():
        if nbrow>nb_ex:
            break
        fGT = dfser['target_path']
        fpred = dfser['pred_path']
        sujid = dfser['subject_id']
        fT1 = gfile(get_parent_path(fpred,2)[0],sujid)
        if len(fT1)==0: #look up as for instance for FastSurfer
            fT1 = gfile(get_parent_path(fpred,4)[0],sujid)
        if os.path.isdir(fT1[0]):
            fT1 = gfile(get_parent_path(fpred, 3)[0], sujid)

        if len(fT1)==0:
            fT1 = gfile(get_parent_path(fpred,4)[0],sujid)
            print(f'T1 is {fT1}')

        if mrview_cmd:
            print(f'{sujid} : {get_col_list_value(dfser,sort_key)}')
            mrview_overlay(fT1, [fGT,fpred], bin_overlay_class=select_label)
        nbrow +=1
    # just print value for clarity
    nbrow=0
    print(f'sorting keys is {"   ".join(sort_key)}')
    for ii,dfser in df.iterrows():
        if nbrow>nb_ex:
            break
        sujid = dfser['subject_id']

        print(f'{sujid} : {get_col_list_value(dfser,sort_key)}')
        nbrow +=1
def make_the_table_gpt(df, yval='dice_GM', group_col=['model_name', 'dataset_name'],group_col_order=None,
                       alpha = 0.05, use_min=False, bonferroni=True,test_stat='wilcoxon', float_str_precision=1,
                       print_table=True):

    summary = (df.groupby(group_col)[yval].agg(['mean', 'std']))
    # Formater en "mean ± std"
    summary['mean_std'] = summary.apply(
        lambda row: f"{row['mean']:.2f} ± {row['std']:.3f}", axis=1
    )
    if group_col_order is not None:
        summary = summary.reindex(pd.MultiIndex.from_product(group_col_order, names=group_col))

    mean = summary['mean'].unstack()
    std = summary['std'].unstack()
    if group_col_order is not None:
        mean = mean.reindex(index=group_col_order[0], columns=group_col_order[1])
    if use_min:
        best_idx = mean.idxmin()
    else:
        best_idx = mean.idxmax()

    # stocker les p-values par dataset
    pvals = {}
    for col in mean.columns:
        best_model = best_idx[col]
        pvals[col] = {}
        best_values = df[
            (df[group_col[0]] == best_model) &
            (df[group_col[1]] == col)
        ][yval]

        for model in mean.index:
            values = df[
                (df[group_col[0]] == model) &
                (df[group_col[1]] == col)
            ][yval]

            if len(values) > 1 and len(best_values) > 1:
                if test_stat=='wilcoxon':
                    if use_min:
                        stat, ppp = wilcoxon(best_values, values, alternative='less')
                    else:
                        stat, ppp = wilcoxon(best_values, values, alternative='greater')
                    if np.isnan(ppp):
                        print(f'isNAN for {model} and {col}')
                        ppp=1
                else:
                    stat, ppp = ttest_ind(best_values, values, equal_var=False)
            else:
                print('APPPPPPAAAA')
                ppp = 1.0  # fallback si pas assez de données
            pvals[col][model] = float(ppp)

    if bonferroni:
        num_comparisons = len(mean.index)-1
        alpha_thr = alpha / num_comparisons
        print(f'bonferooni new alpha  {alpha_thr} nb {num_comparisons}')
    else:
        alpha_thr = alpha
    if print_table:
        latex_table = pd.DataFrame(index=mean.index, columns=mean.columns)
        for col in mean.columns:
            for row in mean.index:
                m = mean.loc[row, col];
                s = std.loc[row, col]
                if pd.isna(m):
                    latex_table.loc[row, col] = "--"
                    continue
                if float_str_precision==1:
                    value = f"{m:.1f} $\\pm$ {s:.1f}"
                    value = f"{m:.1f}  ({s:.1f})"
                else:
                    value = f"{m:.2f} $\\pm$ {s:.3f}"
                # Mettre en gras si meilleur
                if  pvals[col][row] > alpha_thr: #row == best_idx[col]:
                    value = f"\\textbf{{{value}}}"
                latex_table.loc[row, col] = value

        # Export LaTeX
        latex_str = latex_table.to_latex(escape=False)
        print(latex_str)
    else:
        return summary,mean,std, pvals
def get_data(name='hcp',model_flat=None,model_flat_column='dataset_name',add_GT_as_model=False,
             do_scale_vol=True, scale_dice=True):
    dunt = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/'


    match name:
        case 'hcp':
            fcsv = gfile(dunt + 'csv_validationHCP_test/results/','csv$')
        case 'hcp_retest':
            fcsv = gfile(dunt + 'csv_validationHCP_test/results_retest/','csv$')
        case 'hcp_mot':
            fcsv = gfile(dunt + 'csv_validationHCP_test/results_mask_mot_confu/','csv$')
        case 'hcp_reg':
            fcsv = gfile(dunt + 'csv_validationHCP_test/results_region/','csv$')
            name='hcp'
        case 'hcp_confu':
            fcsv = gfile(dunt + 'csv_validationHCP_test/results_confu/','csv$')
        case 'hcp_confu_reg':
            fcsv = gfile(dunt + 'csv_validationHCP_test/results_confu_region/','csv$')
        case 'hcp_confu_dill':
            fcsv = gfile(dunt + 'csv_validationHCP_test/results_dill/','csv$')
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
        case 'dhcp_retest':
            fcsv = gfile(dunt + 'csv_validationdHCP/results_T1T2/','csv$')
        case 'bobs' :
            fcsv = gfile(dunt + 'BOBS/eval/results','csv$')
        case 'vasc' :
            fcsv = gfile('/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/nnunet_pred/Vascular/eval/results_GMsmall','csv$')
        case 'vascGM' :
            fcsv = gfile('/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/nnunet_pred/Vascular/eval/results','csv$')
        case 'GMatro':
            fcsv = gfile('/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/Synth_Atro_Ctx/eval/results','.*csv')
        case 'MiBo':
            fcsv = gfile('/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/MindBoggle101/eval/results','.*csv')

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

    c1 = sns.color_palette()
    c2 = sns.color_palette("Paired")
#add models name convertion by dataset
    morder = None
    match name:
        case 'vasc' | 'vascGM':
            morder = ['pred_DS715_NODA3ResXLres', 'pred_DS716_NODAres','pred_DS708_5nnResXL_res',
                      'pred_DS712_5ResXLres', 'pred_DS714_5ResXLres', 'pred_DS713_3ResXLres']
            mordernn = ['SIAM', 'SiamL', 'SynthSkull', 'SynthVasc', 'SynthMIDA3','SynthMIDA']
            cc = [ c1[2], c2[2], c2[6], c2[11], c1[4], c1[6]]

        case 'bobs':
            morder = ['bibsnet', 'SynthSegGouhfi', 'pred_DS715_NODA3ResXLres', 'pred_DS716_NODAres',
                      'pred_DS708_5nnResXL_res',
                      'pred_DS712_NODA3ResXLres', 'pred_DS714_5ResXLres', 'pred_DS713_3ResXLres']
            mordernn = ['BibsN', 'GOUHFI', 'SIAM', 'SiamL', 'SynthSkull', 'SynthVasc', 'SynthMIDA3',
                        'SynthMIDA']
            cc = [c1[0], c1[1], c1[2], c2[2], c2[6], c2[11], c1[4], c1[6] ]

        case 'hcp'| 'hcp_reg' | 'hcp_retest' :
            df['sujnum'] = [int(ss[25:27]) for ss in df.subject_id]  # HCP
            morderNN = ['FastSurfer', 'SynthSegGouhfi', 'SynthSeg','SuperSynth_invivo', 'pred_DS715_NODA3ResXLres', 'pred_DS716_NODAres','pred_DS708_5nnResXL_res',
                      'pred_DS712_NODA3ResXLres', 'pred_DS714_5ResXLres', 'pred_DS713_3ResXLres']
            mordernnNN = ['FastSurfer', 'GOUHFI', 'SynthSeg', 'SynthSegSuper', 'SIAM','SiamL', 'SynthSkull', 'SynthVasc', 'SynthMIDA3',
                        'SynthMIDA']
            morder = ['FreeSurfer','AssN','FastSurfer', 'SynthSeg','SuperSynth_invivo', 'gouhfi',  'pred_DS715_NODA3ResXLres','pred_DS708_5nnResXL_res',
                      'pred_DS712_NODA3ResXLres', 'pred_DS714_5ResXLres', 'pred_DS713_3ResXLres']#'pred_DS108_LcsfP_Anores',
            mordernn = ['FreeSurfer','AssN','FastSurfer', 'SynthSeg', 'SuperSynth', 'GOUHFI', 'SIAM', 'SynthSkull', 'SynthVasc', 'SynthMIDA3',
                        'SynthMIDA'] #'SIAMano',
            cc = [(0.17,0.65,1), (0.24,0.99,0.24),c1[0], c2[4], c1[3], c1[1], c1[2], c2[6], c2[11], c1[4], c1[6], ] #, c2[2]

        case 'hcp_mot' :
            df['sujnum'] = [int(ss[25:27]) for ss in df.subject_id]  # HCP
            morder = ['FastSurfer', 'pred_DS715_NODA3ResXLres']
            mordernn = ['FastSurfer' 'SIAM']
            cc = [c1[0], c1[2], ]

        case 'hcp_confu'|'hcp_confu_reg' | 'hcp_confu_dill':
            df['sujnum'] = [int(ss[25:27]) for ss in df.subject_id]  # HCP
            morder = ['FastSurfer', 'SynthSegGouhfi', 'SynthSeg', 'pred_DS715_NODA3ResXLres', 'pred_DS716_NODAres', 'pred_DS708_5nnResXL_res',
                      'pred_DS712_NODA3ResXLres', 'pred_DS714_5ResXLres', 'pred_DS713_3ResXLres']
            mordernn = ['FastSurfer', 'GOUHFI', 'SynthSeg', 'SIAM', 'SiamL','SynthSkull', 'SynthVasc', 'SynthMIDA3',
                        'SynthMIDA']
            cc = [c1[0], c1[1], c1[3], c1[2], c2[2], c2[6], c2[11], c1[4], c1[6], ]

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
            morder = [ 'SuperSynth','pred_DS715_NODA3ResXLres', 'pred_DS716_NODAres', ['pred_DS708_5nnResXL_res','pred_DS708_3d_fullres_nnUNetTrainer_nnUNetResEncUNetXLPlans'],
                      ['pred_DS712_5ResXLres','pred_DS712_NODA3ResXLres'], 'pred_DS714_5ResXLres', 'pred_DS713_3ResXLres']
            mordernn = ['SuperSynth','SIAM', 'SiamL', 'SynthSkull', 'SynthVasc', 'SynthMIDA3', 'SynthMIDA']
            # ultra morder = ['pred_DS715_NODA3ResXLres','pred_DS708_3d_fullres_nnUNetTrainer_nnUNetResEncUNetXLPlans','pred_DS712_5ResXLres','pred_DS714_5ResXLres','pred_DS713_3ResXLres']
            # mordernn = ['SIAM', 'SynthSkull','SynthVasc', 'SynthMIDA3','SynthMIDA']
            cc = [c1[3], c1[2], c2[2], c2[6], c2[11], c1[4], c1[6], ]

            df['group'] = 'test'
            df.loc[(df.sujnum==6)|(df.sujnum==10)|(df.sujnum==13),'group']  = 'train'

        case 'dbb':
            morder = ['FastSurfer', 'SynthSegGouhfi', 'SynthSeg', 'SynthSegSuper', 'pred_DS715_3ResXLres', 'pred_DS716_NODAres', 'pred_DS708_5nnResXL_res',
                      'pred_DS712_5ResXLres', 'pred_DS714_5ResXLres', 'pred_DS713_3ResXLres']
            mordernn = ['FastSurfer', 'GOUHFI', 'SynthSeg', 'SynthSegSuper', 'SIAM', 'SiamL','SynthSkull', 'SynthVasc', 'SynthMIDA3',
                        'SynthMIDA']
            cc = [c1[0], c1[1], c1[3],  c2[4], c1[2], c2[2], c2[6], c2[11], c1[4], c1[6], ]
            # DBB group CSFv
            df['group'] = 'bV';
            df.loc[df.index < 12.5, 'group'] = 'sV';  # gros mais pas assez df.loc[df.index==7,'group'] = 'Vs';
            df = df.drop(15)  # too extrem ... almost no brain

        case 'miccai':
            df['sujnum'] = [int(ss[12:14]) for ss in df.subject_id]  # miccai
            #morder = ['FastSurfer', 'gouhfi', 'SynthSeg', 'SynthSegSuper','pred_DS715_NODA3ResXLres', 'pred_DS716_NODAres','pred_DS708_5nnResXL_res',
            #          'pred_DS712_NODA3ResXLres', 'pred_DS714_5ResXLres', 'pred_DS713_3ResXLres']
            #mordernn = ['FastSurfer', 'GOUHFI', 'SynthSeg', 'SuperSynth', 'SIAM','SIAMM', 'SynthSkull', 'SynthVasc', 'SynthMIDA3',
            #            'SynthMIDA']
            morder = ['FastSurfer', 'SynthSeg','SuperSynth','gouhfi','pred_DS715_NODA3ResXLres','pred_DS708_5nnResXL_res',
                      'pred_DS712_NODA3ResXLres', 'pred_DS714_5ResXLres', 'pred_DS713_3ResXLres'
                       ]
            mordernn =['FastSurfer', 'SynthSeg', 'SuperSynth', 'GOUHFI', 'SIAM','SynthSkull',
                       'SynthVasc', 'SynthMIDA3','SynthMIDA']

            cc = [  c1[0], c1[1], c2[4], c1[3], c1[2], c2[2], c2[6], c2[11], c1[4], c1[6], ]

        case 'ultracortex':
            morder = ['FastSurfer', 'SynthSegGouhfi', 'SynthSegGouhfiBM', 'SynthSeg','pred_DS715_3ResXLres', 'pred_DS716_NODArfake075',
                      'pred_DS708_5nnResXL_res', 'pred_DS712_5ResXLres','pred_DS714_5ResXLres', 'pred_DS713_3ResXLres']
            mordernn = ['FastSurfer', 'GOUHFI', 'GOUHFI_BM', 'SynthSeg', 'SIAM','SiamL', 'SynthSkull', 'SynthVasc', 'SynthMIDA3',
                        'SynthMIDA']
            cc = [c1[4], c1[6],(0.78,0.5,0), c1[3], c1[2], c2[2], c2[6], c2[11], c1[4], c1[6], ]

            ##ultracortex group
            df['group'] = 'uni';
            df.loc[(df.index == 2) | (df.index == 5) | (df.index == 7), 'group'] = 'sV'

        case 'dhcp' | 'dhcp_retest':
            # dHCP
            morder = ['SynthSegGouhfi', 'SynthSeg','SuperSynth', 'pred_DS715_NODA3ResXLres','pred_DS716_NODAres', 'pred_DS708_5nnResXL_res',
                      'pred_DS712_5ResXLres', 'pred_DS714_5ResXLres', 'pred_DS713_3ResXLres', 'wmEM_mot_ep240']
            mordernn = ['GOUHFI', 'SynthSeg','SuperSynth', 'SIAM','SiamL', 'SynthSkull', 'SynthVasc', 'SynthMIDA3', 'SynthMIDA', 'SynthBaby']
            cc = [c1[1], c1[3], c2[4], c1[2],c2[2], c2[6], c2[11], c1[4], c1[6],(0.78,0.5,0), ]
            morder = ['SynthSegGouhfi', 'SynthSeg','SuperSynth','pred_DS716_NODAres',
                    'pred_DS713_3ResXLres', 'wmEM_mot_ep240', 'pve_wmEM_mot']
            mordernn = ['GOUHFI', 'SynthSeg','SuperSynth', 'SIAM', 'SynthMIDA', 'SynthBaby','SynthBabyPVE',]
            cc = [c1[1], c1[3], c2[4], c1[2],c2[2], c1[4], c1[6]]

        case 'MiBo':
            mordernn = ['FastSurfer', 'GOUHFI', 'SynthSeg','SuperSynth', 'SIAM','SIAMano','SynthMIDA']
            morder = ['FastSurfer', 'SynthSegGouhfi', 'SynthSeg','SynthSegSuper','pred_DS716_NODAres',
                      'pred_DS108_LcsfP_Anores','pred_DS713_3ResXLres' ]
            morder = ['FastSurfer', 'SynthSeg','SuperSynth','gouhfi','pred_DS715_NODA3ResXLres','pred_DS708_5nnResXL_res',
                      'pred_DS712_NODA3ResXLres', 'pred_DS714_5ResXLres', 'pred_DS713_3ResXLres'
                       ]
            mordernn =['FastSurfer', 'SynthSeg', 'SuperSynth', 'GOUHFI', 'SIAM','SynthSkull',
                       'SynthVasc', 'SynthMIDA3','SynthMIDA']

            cc = [c1[0], c2[4], c1[3], c1[1], c1[2], c2[6], c2[11], c1[4], c1[6],  ]
        case 'GMatro':
            mordernn = ['DlDirectCT', 'FastSurfer', 'GOUHFI', 'SynthSeg', 'SynthSegSuper', 'SiamL', 'SIAMano', 'SynthMIDA']
            morder = ['DlDirectCT','FastSurfer', 'SynthSegGouhfi', 'Synthseg', 'SynthSegSuper', 'pred_DS716_NODAres',
                      'pred_DS108_LcsfP_Anores', 'pred_DS713_3ResXLres']
            cc = [c2[0],c1[0], c1[1], c1[3], c2[4], c1[2], c1[6], c2[2], ]

    # add models for all DS
    all_model = df.model_name.unique()
    if morder is None: #new case all model
        morder = df.model_name.unique()
        mordernn = morder; cc = c1
        print(morder)
    else :
        if 'SuperSynth_invivoNNNNNN' in all_model:
            morder += ['SuperSynth_invivo']
            mordernn += ['SuperS']
            cc += [(0.42,0.08,0.08)]
        if ('pred_DS108_LcsfP_AnoresNNNNNN' in all_model) & ('pred_DS108_LcsfP_Anores' not in morder):
            morder += ['pred_DS108_LcsfP_Anores']
            mordernn += ['SIAMano']
            cc += [c1[4]]
        if 'pred_DS107_MixLow_csfPush_ms_tumorresNNNNNN' in all_model:
            morder += ['pred_DS107_MixLow_csfPush_ms_tumorres']
            mordernn += ['S7']
            cc += [c1[6]]
        if 'pred_DS610_nnAug3DSres' in all_model:
            morder += ['pred_DS610_nnAug3DSres']
            mordernn += ['nnU-Net-DA']
            cc += [ c2[2]]
        if 'pred_DS715_brainmasked' in all_model:
            morder += ['pred_DS715_brainmasked']
            mordernn += ['siamBM']
            cc += [c1[4]]


#filter model not in df (grrr on defait ce qui precedent ...)
    if name=="ultra":
        skip=2
    else:
        newcc, newmo, newmon = [], [], []
        for ii,mm in enumerate(morder):
            if mm in all_model:
                newcc.append(cc[ii])
                newmo.append(morder[ii])
                newmon.append(mordernn[ii])
        cc, morder, mordernn = newcc, newmo, newmon

    for m1, m2 in zip(morder, mordernn):
        if isinstance(m1,list):
            for mm in m1:
                #df.model_name.replace(mm, m2, inplace=True)
                df['model_name'] = df['model_name'].replace(mm, m2)
        else:
            #df.model_name.replace(m1, m2, inplace=True)
            df['model_name'] = df['model_name'].replace(m1, m2)
    if model_flat:
        df = change_df_model_name_withDS(df,ds_newname=model_flat,ds_col=model_flat_column)
        all_model_flat = df.model_name.unique()
        newmodel, newcc = [],[]
        for nm,ncc in zip(mordernn,cc):
            newmodel.append(nm); newcc.append(ncc)
            nnmm = f'{nm}_{model_flat}' #nm+'_T2'
            if nnmm in all_model_flat:
                newmodel.append(nnmm); newcc.append(ncc)
        mordernn = newmodel; cc = newcc
    #check missing model

    #extra change
    #bobs DS get age and sex
    if name=='bobs':
        dfsuj = pd.read_csv('/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/BOBS/sessions.tsv',sep='\t')
        dfsuj = dfsuj.sort_values('age')
        dfsuj = dfsuj.reset_index(drop=True) #so that ii next line match
        dfsuj['subject_id'] = [f'S{ii:02}_' + ss['participant_id'][4:]+'_'+ss['session_id'] for ii,ss in dfsuj.iterrows() ]

        df['age'], df['sex'] = 0,'N'
        for ii,dfser in dfsuj.iterrows():
            sid = dfser['subject_id']
            df.loc[df.subject_id==sid,'age'] = dfser['age']
            df.loc[df.subject_id==sid,'sex'] = dfser['sex']
        #add brain volume
        dfsuj = pd.read_csv('/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/BOBS/bobs_T1.csv')
        for ii,dfser in dfsuj.iterrows():
            sid = dfser['sujname']
            df.loc[df.subject_id==sid,'brain_vol'] = dfser['brain_vol']
        #replace label siam, I don't know why it has the model name in it
        df.loc[~(df.label=='gt'),'label'] = 'siam'
    #confu diff rename col
    if name=='hcp_confu_dill':
        ymet=[]
        for k in df.keys():
            if 'GT' in k:
                    ymet.append(k)
        ymetn = [f'confusion{ss[9:]}' for ss in ymet]
        dic_renam = {k1:k2 for k1,k2 in zip(ymet,ymetn)}
        df = df.rename(columns=dic_renam)
    #group
    if name=='MiBo':
        all_group=[]
        for ii,dfs in df.iterrows():
            if 'NKI-RS' in dfs.subject_id:
                all_group.append('NKI-RS')
            if 'NKI-TRT' in dfs.subject_id:
                all_group.append('NKI-TRT')
            if 'OASIS' in dfs.subject_id:
                all_group.append('OASIS')
        df['group'] = all_group
    #GT as model
    if add_GT_as_model:
        gt_names = df.label.unique()
        input_type_one = df.input_type.unique()[0]
        ymet_in, ymet_tar = get_met(df, regstart=['volume_input']), get_met(df, regstart=['volume_target'])
        for gtname in gt_names:
            dfss = select_df(df,{'label': gtname,'input_type':input_type_one,'model_name':morder[0]})
            dfss.model_name = gtname
            for y1,y2 in zip(ymet_in, ymet_tar):
                dfss[y1] = dfss[y2]
            df = pd.concat([df, dfss], ignore_index=True)
            mordernn = [gtname] + mordernn

    #Add brain volumes
    if do_scale_vol:
        scale_vol_to_cm(df);    # from pixel to cm3
    ymet = get_met(df, regstart=['volume_input'], exclude=['head', 'BG', 'CSF','brain'])
    if len(ymet)>0:
        sum_vol(df, ymet, outname='volume_input_brain')
    ymet = get_met(df, regstart=['volume_target'], exclude=['head', 'BG', 'CSF','brain'])
    if len(ymet) > 0:
        sum_vol(df, ymet, outname='volume_target_brain')

    ymet = get_met(df, regstart=['dice'], exclude=['head', 'BG', 'WM', 'mean'])

    if scale_dice:
        scale_vol(df,ymet,0.01)
    return df,morder,mordernn,cc, ymet

df_list = [];
morderr=['FastSurfer', 'GOUHFI', 'SIAM','SiamL', 'SynthMIDA']
morderr=['FastSurfer','GOUHFI', 'SynthSeg', 'SynthSegSuper', 'SiamL', 'SynthMIDA']
df,morder,mordernn,cc,ymet = get_data('miccai');df_list.append(select_df(df,{'model_name':morderr, 'label':['gt']}))
df,morder,mordernn,cc,ymet = get_data('ultracortex');df_list.append(select_df(df,{'model_name':morderr}))
df,morder,mordernn,cc,ymet = get_data('dbb');df_list.append(select_df(df,{'model_name':morderr,'group':'sV'}))
df,morder,mordernn,cc,ymet = get_data('MiBo');df_list.append(select_df(df,{'model_name':morderr}));
cc2 = cc; cc2.pop(-1)
df,morder,mordernn,cc,ymet = get_data('GMatro');
dfsno = df[df.atrophi==0];  dfsno.dataset_name = dfsno.dataset_name.replace("SynthAtro","SynthNoAtrophy")
df_list.append(select_df(dfsno,{'model_name':morderr}))
#df_list.append(select_df(df,{'model_name':morderr})) #mean over all atrophy
df,morder,mordernn,cc,ymet = get_data('hcp')
dfs1 = select_df(df,{'model_name':morderr,'input_type': 'vol_T1_07', 'label':['Free']})
#dfs2 = select_df(df,{'model_name':morderr,'input_type': 'vol_T1_07', 'label':['Assn']})
dfs1.dataset_name = dfs1.dataset_name.replace("HCP_test_retest_07mm_suj82_vol_T1_07_free_Ass_siam","HCP Free GT")
#dfs2.dataset_name = dfs2.dataset_name.replace("HCP_test_retest_07mm_suj82_vol_T1_07_free_Ass_siam","HCP AssN GT")
df_list.append(dfs1);#df_list.append(dfs2)
#appen dhcp   HCP_test_retest_07mm_suj82_vol_T1_07_free_Ass_siam_nomask
dfd,morder,mmm,cc,ymet = get_data('dhcp')
dfd=select_df(dfd,{'dataset_name':'dHcp_old075_volT2_GT','label_column':'lab_free'})
#dfd = dfd[~ (dfd.model_name=='FastSurfer')]
df_list.append(dfd )  #cf ('dhcp')

dfa = pd.concat(df_list)
morderr = morderr[:1] + morderr[2:4] + morderr[1:2] + morderr[4:6]; cc2 = cc2[:1] + cc2[2:4] + cc2[1:2] + cc2[4:6]; cc2 = cc2[:1] + cc2[2:3] + cc2[1:2] + cc2[3:] #inver syntseg
if True:
    dfa.model_name = dfa.model_name.replace('SynthSegSuper', 'SuperSynth'); dfa.model_name = dfa.model_name.replace('SiamL', 'SIAM')
    morderr[2] = 'SuperSynth';    morderr[4] = 'SIAM'
    dfa['Dice GM'] = dfa['dice_GM']
    DSname = ['MICCAI', 'Ultracortex', 'DBB', 'Mindboggle','SynthNoAtrophy','HCP','dHCP'];
    DSold = dfa.dataset_name.unique()
    for n1,n2 in zip(DSold, DSname):
        print(f"replace {n1} by {n2}")
        dfa.dataset_name = dfa.dataset_name.replace(n1,n2)
    xorder = ['DBB','MICCAI', 'Mindboggle', 'SynthNoAtrophy', 'Ultracortex', 'HCP','dHCP' ]

    fig = sns.catplot(data=dfa, y='volume_ratio_GM',x='dataset_name', hue='model_name', kind='boxen', palette=cc2, hue_order=morderr, order=xorder, height=9, aspect=2.5,)
    plt.ylim([0.8 ,1.2]); ylab='Volume ratio GM'; xlab=''; ax = fig.axes[0][0]
    xlim = plt.xlim()
    plt.plot(xlim,(1,1),linestyle='dashed', color='k', linewidth=2);plt.xlim(xlim)

    fig = sns.catplot(data=dfa, y='Dice GM',x='dataset_name', hue='model_name', kind='boxen', palette=cc2, hue_order=morderr, order=xorder, height=9, aspect=2.5,)
    ylab='Dice GM'; xlab=''; ax = fig.axes[0][0]
    ax.set_ylim(75, 99); ax.set_yticks(np.arange(75, 97.6, 2.5))
#    ax.set_ylabel(ylab, fontsize='x-large');    yyy = ax.get_yticklabels(); ax.set_yticklabels(yyy, fontsize='large')
#    ax.set_xlabel(xlab, fontsize='x-large'); yyy = ax.get_xticklabels();ax.set_xticklabels(yyy, fontsize='large')
    ax.set_ylabel(ylab);    yyy = ax.get_yticklabels(); ax.set_yticklabels(yyy)
    ax.set_xlabel(xlab); yyy = ax.get_xticklabels();ax.set_xticklabels(yyy)
    sns.move_legend(fig,'right',bbox_to_anchor=(0.99, .25),fontsize='x-small',frameon=True, shadow=True, title=f'Model',title_fontsize='small')

#avec un tableau
# Calcul mean et std
sel_col, yval, group_col_order = ['model_name', 'dataset_name'], 'Dice GM',  [morderr, xorder]
#sel_col, yval, group_col_order = ['model_name', 'from'], 'dice', [morderr,ymet]  #subcorti avec ymet et dfmm
#sel_col, yval  = ['model_name', 'input_type'], 'dice_skull'

make_the_table_gpt(dfa,yval=yval, group_col=sel_col,group_col_order=group_col_order)
#save to csv dfs = select_df(dfa,{sel_col[0] : group_col_order[0][:-1],sel_col[1] : group_col_order[1] })

sel_col, yval, group_col_order = ['model_name', 'dataset_name'], 'Atrophy relative error', [wanted_order, ['SynthAtro']]#[mordernn[:-1],ymet]  #subcorti avec ymet et dfmm
make_the_table_gpt(dfsa,yval=yval, group_col=sel_col,group_col_order=group_col_order, use_min=True)

df,morder,mordernn,cc,ymet = get_data('bobs')
df.model_name=df.model_name.replace('BibsN','FastSurfer')
df_list.append( select_df(df,{'model_name':morderr,'label':'gt','input_type':'T1'}) )


df,morder,mordernn,cc,ymet = get_data('hcp_confu_dill') #df,morder,mordernn = get_data('hcp_confu_reg')

dic_few = get_remapping('712', lab_name=['Map_to_label_GT_Head_Name','synth'])
labels = list(dic_few.keys())
dfg = get_metric_from_confusion(dfg, labels)


# for dice ymet.pop(-1);ymet.pop(0);ymet.pop(1);ymet.pop(1)
dfmm = df.melt(id_vars=[ 'model_name', 'dataset_name', 'label'], value_vars=ymet, var_name='from', value_name='dice');
fig = sns.catplot(data=dfmm, y='dice', x='label', col='from', hue='model_name', kind='boxen',hue_order=mordernn,col_wrap=3, palette=cc)

sns.catplot(data=df, y='dice_GM', x='label', col='dataset_name', hue='model_name', col_wrap=3, kind='boxen') #hue_order=xorder                        col_order=corder)
sns.catplot(data=dfmm, y='dice', x='dataset_name',kind='boxen', col='from',col_wrap=1)


sel_factor = {'model_name':'SynthSkull_T2','dataset_name':'T1','label':'siam'}
dfss,sel=select_df_ask(df)
compare_pred_mrview(df,sort_key='dice_GM',sel_factor=sel_factor)
compare_pred_mrview(dfss,'dice_skull',ascending=True,select_label=1,nb_ex=5)

############################################# SynthAtro ###############################
if True:
    df,morder,mordernn,cc,ymet = get_data('GMatro');
    morderr = mordernn[1:2] + mordernn[3:5] + mordernn[2:3] + mordernn[5:6]+ mordernn[7:8];
    cc2 = cc[1:2] + cc[3:5] + cc[2:3] + cc[5:6]+ cc[7:8];cc2 = cc2[:1] + cc2[2:3] + cc2[1:2] + cc2[3:] #inver syntseg
    df.model_name = df.model_name.replace('SynthSegSuper', 'SuperSynth'); df.model_name = df.model_name.replace('SiamL', 'SIAM');df.model_name = df.model_name.replace('Gouhfi','GOUHFI');
    morderr[2] = 'SuperSynth';    morderr[4] = 'SIAM'; morderr[3] = 'GOUHFI'
    morderr = ['freesurfer'] + morderr;  cc2 = [(0.17, 0.65, 1)] + cc2
    #add last instead of SynthMIDA
    morderr[-1] = mordernn[-1]


    #fig=sns.catplot(data=df, y='dice_GM',x='atrophi', hue='model_name', kind='boxen', palette=cc2, hue_order=morderr)
    vt_no, vi_no=[],[]; verror = []; verrorABS = []
    for ii,dfl in df.iterrows():
        sujn, modn = dfl.sujname_short, dfl.model_name
        dfs = df[(df.atrophi==0)&(df.sujname_short==sujn)&(df.model_name==modn)]
        vt_no.append((1- dfl.volume_target_GM/dfs.volume_target_GM.values[0])*100)
        vi_no.append((1- dfl.volume_input_GM/dfs.volume_input_GM.values[0] )*100)
        verrorABS.append(np.abs(vt_no[-1] - vi_no[-1]) )
        ee = verrorABS[-1]/vt_no[-1] *100 if vt_no[-1]>0 else 0
        verror.append(ee)
        print(len(dfs))
    df['Predicted Atrophy'] = vi_no;df['Induced Atrophy'] = vt_no; df['Atrophy relative error'] = verror;df['Atrophy Error'] = verrorABS;
    df.loc[df.atrophi == 0, 'dataset_name'] = 'Synth No Atrophy'
    dfsa = df[df.atrophi>0]

    fig=sns.catplot(data=dfsa, y='Atrophy relative error',x='dataset_name', hue='model_name', kind='boxen', palette=cc2, hue_order=morderr)
    fig=sns.catplot(data=df, y='dice_GM',x='dataset_name', hue='model_name', kind='boxen', palette=cc2, hue_order=morderr)
    sns.move_legend(fig,'right',bbox_to_anchor=(0.55, .35),fontsize='xx-small',frameon=True, shadow=True, title=f'Model',title_fontsize='x-small')

    col_wrap=3;
    xlab,ylab='REF Volume', 'Predicted Volume';lims=[300,700]; y='volume_input_GM'; x='volume_target_GM'; xrange=[300,701,100]
    xlab,ylab='Induced Atrophy %', 'Predicted Athrophy %';lims=[0,40]; y='Predicted Atrophy'; x='Induced Atrophy'
    xlab,ylab='Induced Atrophy (mm)', 'Atrophy relative error %';lims=[0,40]; y='Atrophy relative error'; x='atrophi'
    fig=sns.relplot(data=df, x=x,y=y, col='model_name',
                    col_wrap=col_wrap, hue='model_name', col_order=morderr[:-1], hue_order=morderr, palette=cc2)
    fig=sns.catplot(data=df, x=x,y=y, hue='model_name', col_order=morderr, kind='boxen', hue_order=morderr, palette=cc2)
    #ax = fig.axes[0][0];ax.set_yticks(np.arange(0, 101, 10)); ax.set_ylabel(ylab); ax.set_xlabel(xlab);

   wanted_order = ['freesurfer', 'FastSurfer',  'SynthSeg', 'SuperSynth', 'GOUHFI', 'SIAM']
    sel_col, yval, group_col_order = ['model_name', 'dataset_name'], 'Atrophy relative error', [wanted_order, ['SynthAtro']]#[mordernn[:-1],ymet]  #subcorti avec ymet et dfmm
    make_the_table_gpt(dfsa,yval=yval, group_col=sel_col,group_col_order=group_col_order, use_min=True,alpha=0.01)

    sel_col, yval, group_col_order = ['model_name', 'atrophi'], 'Atrophy relative error', [wanted_order,df.atrophi.unique()]  #subcorti avec ymet et dfmm
    make_the_table_gpt(df,yval=yval, group_col=sel_col,group_col_order=group_col_order, use_min=True, alpha=0.01)

############################################# hcp_retest ###############################
if True:
    df,morder,mordernn,cc,ymett = get_data('hcp_retest')
    #mordernn = mordernn[:2] + mordernn[3:5] + mordernn[2:3] + mordernn[6:8]; cc2 = cc;
    #cc2 = cc2[:2] + cc2[3:5] + cc2[2:3] + cc2[6:8]; cc2 = cc2[:2] + cc2[3:4] + cc2[2:3] + cc2[4:] #inver syntseg
    #c1 = sns.color_palette();cc2[-1] = cc2[-2]; cc2[-2] = c1[2]
    new_name1, new_name2 = 'T1/T1' , 'T2/T1'
    df.input_type = df.input_type.replace('vol_T2_07',new_name2); df.input_type = df.input_type.replace('vol_T1_07_realign',new_name1)
    #df.model_name = df.model_name.replace('SynthSegSuper','SuperSynth'); df.model_name = df.model_name.replace('pred_DS716_NODAres','SIAM');df.model_name = df.model_name.replace('Gouhfi','GOUHFI');
    #mordernn[3]='SuperSynth'; mordernn[5]='SIAM'; mordernn[4] = 'GOUHFI'

    fig=sns.catplot(data=df, y='dice_GM',x='input_type', hue='model_name', kind='boxen', palette=cc,
                    hue_order=mordernn, order=[new_name1, new_name2], height=7, aspect=1,)


    dfs1=select_df(df,{'input_type':'T1/T1_repeat'})
    dfs1=select_df(df,{'input_type':'T2/T1'})
    sel_col, yval, group_col_order = ['model_name', 'from'], 'dice', [mordernn[:-1],ymet]  #subcorti avec ymet et dfmm
    make_the_table_gpt(dfmm,yval=yval, group_col=sel_col,group_col_order=group_col_order, alpha=0.01)

    #pour toutes les regions:
    ymet =[ 'dice_Put', 'dice_Pal', 'dice_Cau-acc', 'dice_thal','dice_cerGM','dice_CSFv', 'dice_hypp', 'dice_amyg',
            'dice_GM', 'dice_dura','dice_vessel','dice_skull'] #'dice_GM',
    ctitle = [ 'Putamen','Palidum','Caudate-Accubens','Thalamus', 'Cerebellum','Ventricle','Hippocampus','Amygdala']#MICCAI 'GM',
    ymet = ['dice_skull','dice_CSF','dice_vascular', 'dice_Dura']
    ctitle = [ 'Skull', 'CSF', 'Vessels', 'Dura mater']
    #mordernn =  mordernn[3:6:2]; cc = cc[3:6:2]
    dfmm = df.melt(id_vars=[ 'model_name', 'dataset_name','input_type', 'label'], value_vars=ymet, var_name='from', value_name='dice');
    fig=sns.catplot(data=dfmm, y='dice',x='input_type', hue='model_name', kind='boxen', palette=cc, col='from', col_wrap=4,
                    hue_order=mordernn, order=[new_name1, new_name2], height=4, aspect=0.5) # height=3.2, aspect=1.8)
    ylim = [85, 99]; yrange = np.arange(85, 100, 2.5);sizefont = 'medium' #'x-large'
    ylim = [72.5, 99]; yrange = np.arange(72.5, 100, 2.5);sizefont = 'medium' #'x-large'
    for ii, ax in enumerate(fig.axes):
        ax.set_xlabel('')
        ax.set_title(ctitle[ii], fontdict=dict(fontsize=sizefont))
        if (ii==0) | (ii==4):
            ax.set_ylabel(f'Dice %',fontsize=sizefont) #'Dice'   'Average Surface dist' 'Volume Ratio'
            ax.set_ylim(ylim);
            ax.set_yticks(yrange)
            ticks = ax.get_yticks()  # positions réelles (pas les Text)
            labels = [f"{t:.0f}" if t % 1 == 0 else "" for t in ticks]
            ax.set_yticks(ticks, labels, fontsize=sizefont);
            ax.grid(True, axis='y')

    plt.subplots_adjust(hspace=0.15, bottom=0.12, top=0.9,wspace=0.2,right=0.9,left=0.1)
    sns.move_legend(fig,'right',bbox_to_anchor=(0.65, .75),fontsize='x-small',frameon=True, shadow=True, title='',title_fontsize='xx-small')

    sns.move_legend(fig,'right',bbox_to_anchor=(1, .3),frameon=True, shadow=True, title=f'')#,title_fontsize='x-large',fontsize='x-large'
    plt.ylim(0.78,0.98);
    sns.move_legend(fig,'right',bbox_to_anchor=(1, .3),frameon=True, shadow=True, title_fontsize='small',fontsize='x-small')


    #avec un tableau
    # Calcul mean et std
    summary = ( df.groupby(['model_name', 'input_type'])['dice_GM'].agg(['mean', 'std']) )

    #summary = summary.reindex(    pd.MultiIndex.from_product([morderr, DSname],names=['model_name', 'input_type']))
    mean = summary['mean'].unstack()
    std = summary['std'].unstack()
    mean = mean.reindex(index=mordernn[:-1])
    best_idx = mean.idxmax()
    latex_table = pd.DataFrame(index=mean.index, columns=mean.columns)
    for col in mean.columns:
        for row in mean.index:
            m = mean.loc[row, col];        s = std.loc[row, col]
            if pd.isna(m):
                latex_table.loc[row, col] = "--"
                continue
            value = f"{m:.2f} $\\pm$ {s:.3f}"
            # Mettre en gras si meilleur
            if row == best_idx[col]:
                value = f"\\textbf{{{value}}}"
            latex_table.loc[row, col] = value

    # Export LaTeX
    latex_str = latex_table.to_latex(escape=False)
    print(latex_str)



    dfd,morderd,mordernnd,ccd,ymetd = get_data('dhcp_retest')
    dfd.input_type = dfd.input_type.replace('vol_T2', 'dHCP: T2/T1')
    df = pd.concat([df,dfd])
    df['Dice GM'] = df['dice_GM']; ylab = 'Dice GM';xlab = '';
    df['Volume ratio GM'] = df['volume_ratio_GM']; ylab = 'Volume ratio GM';xlab = '';

    fig=sns.catplot(data=df, y=ylab,x='input_type', hue='model_name', kind='boxen', palette=cc2, hue_order=mordernn,
                    order=['HCP: T1/T1_repeat','HCP: T2/T1','dHCP: T2/T1'], height=9, aspect=1,)
    ax = fig.axes[0][0]
    ax.set_ylim(0.75, 0.99); ax.set_yticks(np.arange(0.75, 0.976, 0.025)); ax.set_xlabel('')
    plt.ylim([0.8 ,1.2]); ylab='Volume ratio GM'; xlab=''; ax = fig.axes[0][0]


############################################# dHCP_old ###############################
if True:
    df,morder,mordernn,cc,ymet = get_data('dhcp')
    df=select_df(df,{'dataset_name':'dHcp_old075_volT2_GT'})
    mordernn =  mordernn[1:3] + mordernn[:1] + mordernn[3:]; cc2 = cc;
    cc2 =  cc2[1:3] + cc2[:1] + cc2[3:]; cc2 = cc2[1:2] + cc2[:1] + cc2[2:] #inver syntseg
    df['Dice GM'] = df['dice_GM']; ylab = 'Dice GM';xlab = '';
    df['Volume ratio GM'] = df['volume_ratio_GM']; ylab = 'Volume ratio GM';xlab = '';
    df.label_column = df.label_column.replace('lab_binPV','Surf GT');df.label_column = df.label_column.replace('lab_free','DrawEM GT');
    df.model_name = df.model_name.replace('SynthBaby', 'SynthBabyDrawEM');
    df.model_name = df.model_name.replace('SynthBabyPVE', 'SynthBabySurf');
    mordernn[-2] = 'SynthBabyDrawEM';    mordernn[-1] = 'SynthBabySurf'


    fig=sns.catplot(data=df, y=ylab,x='label_column', hue='model_name', kind='boxen', palette=cc2, hue_order=mordernn, height=9, aspect=1,)
    ax = fig.axes[0][0]; ax.set_ylim(0.75, 0.99); ax.set_yticks(np.arange(0.75, 0.976, 0.025))
    sns.move_legend(fig,'right',bbox_to_anchor=(1, .25),frameon=True, shadow=True, title_fontsize='x-small',fontsize='xx-small')
    ax = fig.axes[0][0];xl = ax.get_xlim(); ax.plot(xl,[1,1],'--k'); ax.set_xlim(xl);
    sns.move_legend(fig,'right',bbox_to_anchor=(1, .78),frameon=True, shadow=True, title_fontsize='x-small',fontsize='xx-small')

    dfpred = pd.read_csv('/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/csv_validationdHCP/all_metric_previous_baby.csv')
    dfs = dfpred[(dfpred.model_name=='DataT2')|(dfpred.model_name=='DataT2_surf') ]
    dfs = dfs[dfs.eval_on=='T2']
    dfs = dfs[dfs.sujnum>683]
    dfs = dfs[(dfs.GroundTruth=='drawEM') | (dfs.GroundTruth=='surf')]
    dfs['label_column'] = dfs['GroundTruth']
    dfs['label_column'] = dfs['label_column'].replace('surf','Surf GT')
    dfs['label_column'] = dfs['label_column'].replace('drawEM','DrawEM GT')
    dfs['model_name'] = dfs['model_name'].replace('DataT2','DataT2_DrawEM')
    dfs['model_name'] = dfs['model_name'].replace('DataT2_surf','DataT2_Surf')
    mordernn += ['DataT2_Surf', 'DataT2_DrawEM']
    dfs['volume_ratio_GM'] = dfs['predicted_occupied_volume_GM'] / dfs['occupied_volume_GM']
    c1 = sns.color_palette(); cc2 += [(0.12/0.7,0.46/0.7,0.70/0.7), (0.12/1.2,0.46/1.2,0.70/1.2)]
    df = pd.concat([df,dfs])
    #pour le T1/T2
    ax = fig.axes[0][0]; ax.set_xlabel(xlab);  # yyy = ax.get_xticklabels(); ax.set_xticklabels(yyy, fontsize='large')
    ax.set_xticklabels([plt.Text(0,0,'T1 / T2')])
    xl = ax.get_xlim(); ax.plot(xl,[1,1],'--k'); ax.set_xlim(xl); ax.set_ylim([0.8,1.2])

    #pour le concat to all
    dfd,morder,mordernn,cc,ymet = get_data('dhcp')
    dfd=select_df(dfd,{'dataset_name':'dHcp_old075_volT2_GT','label_column':'lab_free'})
    dfd = dfd[~ (dfd.model_name=='FastSurfer')]
    dfs.model_name = dfs.model_name.replace('DataT2_DrawEM','FastSurfer')
    dfs['dataset_name'] = 'dHcp_old075_volT2_GT'
    dfd = pd.concat([dfd,dfs])


############################################# BOBS ###############################
if True:
    df,morder,mordernn,cc,ymet = get_data('bobs',model_flat=('T2'))
    df = select_df(df,{'label':'gt'})
    df = df[~ (df.subject_id=='S66_372377_ses-8mo')]
    df =df[~((df.model_name=='SynthVasc') | (df.model_name=='SynthMIDA3')| (df.model_name=='SynthSkull'))]
    df =df[~((df.model_name=='SynthVasc_T2') | (df.model_name=='SynthMIDA3_T2')| (df.model_name=='SynthSkull_T2'))]
    mordernn.pop(-3);mordernn.pop(-3);mordernn.pop(-3); cc.pop(-3); cc.pop(-3); cc.pop(-3)

    fig = sns.catplot(df, y='dice_GM', x='label',kind='boxen',hue='model_name',hue_order=mordernn, palette=cc)

    df,morder,mordernn,cc,ymet = get_data('bobs')
    #ymet.pop(1);ymet.pop(-1)
    dfss = select_df(df,{'input_type': 'T1','model_name':'SIAM'})
    dfss.model_name='GT'; dfss.volume_input_GM = dfss.volume_target_GM
    df = pd.concat([df, dfss], ignore_index=True)
    mordernn = ['GT']+mordernn


    df =df[~((df.model_name=='SynthVasc') | (df.model_name=='SynthMIDA3')| (df.model_name=='SynthSkull'))]
    mordernn.pop(-2);mordernn.pop(-2);mordernn.pop(-2); cc.pop(-2); cc.pop(-2); cc.pop(-2)
    df['GM_brain_ratioGT'] = df['volume_input_GM']/df['brain_vol']*100
    df['GM_brain_ratioGT'] = df['volume_input_GM']/df['volume_target_brain']*100
    df['GM_brain_ratio'] = df['volume_input_GM']/df['volume_input_brain']*100
    #dfs = select_df(df,{'input_type': 'T1','label':'gt'})
    dfs = select_df(df,{'label':'gt'})
    dfs['model'] = dfs.model_name

    col_wrap=3; sharey=False#True
    yy='volume_input_GM'; ch = 'sex'
    yy='GM_brain_ratio'; ch="input_type"
    g = sns.FacetGrid(dfs,col='model',col_order=mordernn, hue=ch ,
                      col_wrap=col_wrap,sharey=sharey, legend_out=True, despine=True,height=4, aspect= 1.33)
    g = g.map_dataframe(sns.regplot, x='age', y=yy, lowess=True)
    g.add_legend()
    plt.ylim([150,600]);plt.yticks([200,300,400,500,600])

    ylab,xlab, legend_title = 'GM vol (in cm3)', 'age', 'sex'
    ylab,xlab, legend_title = 'GM vol/Brain vol', 'age', 'Input '
    ctitle = mordernn
    for ii, ax in enumerate(g.axes):
        ax.set_title(ctitle[ii], fontdict=dict(fontsize='x-large'))
        if (ii % col_wrap)==0 :
            ax.set_ylabel(ylab,fontsize='x-large'); yyy = ax.get_yticklabels(); ax.set_yticklabels(yyy, fontsize='large')
        else:
            yyy = ax.get_yticklabels(); ax.set_yticklabels(yyy, fontsize='large')
        #else:
        #    ax.set_ylabel(''); yyy = ax.get_yticklabels(); ax.set_yticklabels(yyy, fontsize='large')
        if ii>(len(g.axes)-col_wrap-1):
            ax.set_xlabel(xlab,fontsize='x-large'); yyy = ax.get_xticklabels(); ax.set_xticklabels(yyy, fontsize='large')

    sns.move_legend(g,'right',bbox_to_anchor=(1, .5),fontsize='large',frameon=True, shadow=True, title=legend_title,title_fontsize='x-large')


    def annotate(data, **kwargs):
        #dfs1 = data[data.sex=='Female']
        print(f'data unique is {data.sex.unique()}')
        r, p = scipy.stats.pearsonr(data['age'], data['volume_input_GM'])
        #res = scipy.stats.linregress(data['age'], data['volume_input_GM'])  #same
        ax = plt.gca()
        if "Female" in data.sex.unique():
            tt,ypos = 'Female', .8
        else:
            tt,ypos = 'Male', .9
        #print(kwargs)
        ax.text(.05, ypos, f'{tt} r={r:.2f}',transform=ax.transAxes,color=kwargs['color'])
        #R={res[2]:.2g} usefulle .2g for small number (to get 0.00043)
    def annotate2(data, **kwargs):
        res = scipy.stats.linregress(data['age'], data['GM_brain_ratio'])  #same
        ax = plt.gca()
        ypos = .9
        ax.text(.05, ypos, f'slope {res[0]:.2f} \nR={res[2]:.2f}',transform=ax.transAxes,color=kwargs['color'])
    fig = sns.lmplot(x="age", y="GM_brain_ratio", data=dfs,col='model',col_wrap=3, col_order=mordernn, robust=True)
    fig.map_dataframe(annotate2)

    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    from matplotlib.ticker import ScalarFormatter; formatter = ScalarFormatter(useMathText=True); formatter.set_scientific(True); formatter.set_powerlimits((0,0))  # force la notation scientifique
    ax.yaxis.set_major_formatter(formatter)

############################################# ultracortex ###############################
if True:
    df,morder,mordernn,cc,ymet = get_data('ultracortex')
    fig = sns.catplot(df, y='dice_GM', x='label',kind='boxen',hue='model_name',hue_order=mordernn, palette=cc)
    ax = fig.axes[0][0]
    ax.set_title('GM', fontsize='x-large')
    ax.set_ylabel(f'Dice',fontsize='x-large');ax.set_xlabel('',fontsize='x-large'); ax.set_xticklabels('')
    yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize='large' )
    sns.move_legend(fig,'right',bbox_to_anchor=(1, .4),fontsize='large',frameon=True, shadow=True, title=f'Model',title_fontsize='large')

    fig.savefig('GM_ultra_cortex.png')
    sel_col, yval= ['model_name', 'dataset_name'], 'Dice GM',

    sel_col, yval, group_col_order   = ['model_name', 'input_type'], 'dice_skull',  [morderr, xorder]
    make_the_table_gpt(df,yval=yval, group_col=sel_col,group_col_order=group_col_order)
    #save to csv dfs = select_df(dfa,{sel_col[0] : group_col_order[0][:-1],sel_col[1] : group_col_order[1] })

############################################# mindboggle ###############################
if True:
    def swap_cereb_assn_ceres(df,mordernn,test_print=False):
        for mm in mordernn:
            mask_line1 = (df.model_name==mm) & (df.label_column=='lab_ceres')
            mask_line2 = (df.model_name == mm) & (df.label_column == 'lab_Assn')
            dfs1 = df[mask_line1]
            dfs2 = df[mask_line2]
            yval1, yval2 = dfs1.dice_cerGM, dfs2.dice_cerGM
            if test_print:
                print(f"M {mm} shape {np.mean(yval1)} {np.mean(yval2)}")
            else:
                df.loc[mask_line2,'dice_cerGM'] = dfs1.dice_cerGM
        return df


    init_DS=True;         augment_model = False
    if init_DS:
        df_list = [];
        df,morder,mordernn,cc,ymet = get_data('miccai');  #df = select_df(df,{'label':['Assn']})
        df = swap_cereb_assn_ceres(df,mordernn);
        df.dataset_name = df.dataset_name.str.replace('MICCAIstd_testset_suj20_vol_T1','MICCAI')
        df.label = df.label.str.replace('gt','Manu')
        df_list.append(df)

        df,morder,mordernn,cc,ymet = get_data('MiBo');  #df = select_df(df,{'label':['Assn']})
        df = swap_cereb_assn_ceres(df,mordernn);
        df = select_df(df,{'group':['NKI-TRT','NKI-RS']})
        df.dataset_name = df.dataset_name.str.replace('MiBo','Mindboggle')
        df_list.append(df)

        df1, morder, mordernn, cc, ymet = get_data('hcp')
        df1 = swap_cereb_assn_ceres(df1,mordernn);
        df1 = select_df(df1, { 'input_type': 'vol_T1_07', 'label': ['Assn','Free','siam']})
        df1['dataset_name']='HCP'

        df,morder,mordernn,cc,ymett = get_data('hcp_retest')
        new_name1, new_name2 = 'T1/T1' , 'T2/T1'
        df.input_type = df.input_type.replace('vol_T2_07',new_name2); df.input_type = df.input_type.replace('vol_T1_07_realign',new_name1)
        df['dataset_name']='HCP test retest'
        df.label = df.input_type
        df_list.append( pd.concat([df1,df]) )

        morderr = mordernn; cc2=cc #morderr = mordernn[:1] + mordernn[2:4] + mordernn[1:2] + mordernn[4:]; cc2 = cc[:1] + cc[2:4] + cc[1:2] + cc[4:]; cc2 = cc2[:1] + cc2[2:3] + cc2[1:2] + cc2[3:] #inver syntseg
        df = pd.concat(df_list)
        orderDS = [ 'MICCAI', 'Mindboggle' ,'HCP', 'HCP test retest']
        xorder = ['Manu', 'Assn', 'Free', 'T1/T1' , 'T2/T1']
        if augment_model:
            morderr.pop(0);morderr.pop(0);morderr.pop(1);morderr.pop(1);
            cc2.pop(0);cc2.pop(0);cc2.pop(1);cc2.pop(1);
        else:
            morderr = morderr[:7]; cc2 = cc2[:7]

#one shot ! yeah 2026_08_08
    def my_boxen(data, x=None, y=None, hue=None, order=None, hue_order=None,palette=None, **kws):
        # ne garde que les catégories réellement présentes, dans l'ordre voulu
        present = data[x].unique()
        o = [c for c in order if c in present]
        presentHue = data[hue].unique()
        ho = [c for c in hue_order if c in presentHue]
        pal = [pal for pal,c in zip(palette,hue_order) if c in presentHue]
        sns.boxenplot(data=data, x=x, y=y, hue=hue, hue_order=ho, order=o, palette=pal, legend=False, **kws)

    ymets = [['dice_GM', 'dice_Put', 'dice_Pal', ],[  'dice_Cau-acc', 'dice_thal','dice_cerGM',],[ 'dice_CSFv','dice_hypp', 'dice_amyg'] ]
    #ymets = [['hausdorff_avg_GM', 'hausdorff_avg_Put', 'hausdorff_avg_Pal', ],[  'hausdorff_avg_Cau-acc', 'hausdorff_avg_thal','hausdorff_avg_cerGM',],[ 'hausdorff_avg_CSFv','hausdorff_avg_hypp', 'hausdorff_avg_amyg'] ]
    ctitles = [['GM', 'Putamen', 'Palidum'],[ 'Caudate-\nAccubens' ,'Thalamus','Cerebellum'],[  'Ventricle','Hippocampus','Amygdala'  ]]
    #ymet = ymets[0] ; ctitle = ctitles[0]
    ylim = None #
    ylim = [77.5, 99]; #[75, 99];
    yrange = np.arange(77.5, 100, 2.5) #    xorder = ['Assn','Free']#xorder = ['gt','Assn','Free']
    add_legend=False
    sizefont = 'medium'  # 'x-large'
    rdout = "/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/figure/other_3label_3DS/"

    for ymet,ctitle in zip(ymets,ctitles):
        dfmm = df.melt(id_vars=[ 'model_name', 'dataset_name','input_type', 'label'], value_vars=ymet, var_name='from', value_name='dice');

        g = sns.FacetGrid(dfmm, col="dataset_name", row="from", margin_titles=True,
                          sharex=False,height=3, aspect= 1.8,
                          gridspec_kws={"width_ratios": [3, 2, 2, 2]} )

        g.map_dataframe(my_boxen,x='label',y='dice', hue='model_name',
                        hue_order=morderr, palette=cc2, order=xorder)
        plt.subplots_adjust(hspace=0.02, wspace=0.02, bottom=0.06, top=0.95,right=0.99,left=0.1)
        for li, axline in enumerate(g.axes):
            for ci, ax in enumerate(axline):
                if li==0:
                    ax.set_title(orderDS[ci], fontdict=dict(fontsize=sizefont))
                if ylim is not None:
                    ax.set_ylim(ylim);                ax.set_yticks(yrange)
                # ax.set_ylim(0.75, 0.99);    ax.set_yticks(np.arange(0.75, 0.976, 0.025))
                if ci==0:
                    yleg = f"{ctitle[li]}\nDice %"
                    if ctitle[li] in ['Putamen', 'Palidum', 'Caudate-\nAccubens' ,'Thalamus']:
                        ax.set_ylabel(yleg, fontsize=sizefont, color='red')
                    elif ctitle[li] in ['Cerebellum']:
                        ax.set_ylabel(yleg, fontsize=sizefont)
                    else:
                        ax.set_ylabel(yleg, fontsize=sizefont, color='blue')

                    # yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize=sizefont )
                    if ylim is not None:
                        ticks = ax.get_yticks()  # positions réelles (pas les Text)
                        labels = [f"{t:.0f}" if t % 1 == 0 else "" for t in ticks]
                        ax.set_yticks(ticks, labels, fontsize=sizefont);
                    ax.grid(True, axis='y')
                if li == 2:
                    #ax.set_xticklabels(xorder, rotation=0, fontsize="small");
                    ax.set_xlabel('')
                    for label in ax.get_xticklabels():
                        if label.get_text() == 'Assn':
                            label.set_color('red')
                        if label.get_text() == 'Free':
                            label.set_color('blue')
                if ci==3:
                    ax.texts[0].remove()


        # handles colorés avec ta palette
        if add_legend:
            handles = [mpatches.Patch(color=cc2[morderr.index(m)], label=m) for m in morderr]
            g.fig.legend(handles, morderr,title='',bbox_to_anchor=(0.85, .75),fontsize='xx-small',frameon=True, shadow=True,title_fontsize='xx-small')
            #sns.move_legend(g,'right',bbox_to_anchor=(0.85, .75),fontsize='xx-small',frameon=True, shadow=True, title='',title_fontsize='xx-small')

        sufix = 'Aug_' if augment_model else 'Main_'
        if add_legend:
            sufix += "Leg_"

        g.savefig(f'{rdout}/Dice3DS{sufix}{ymet[0]}_label.eps')#_{dn}
        g.savefig(f'{rdout}/Dice3DS{sufix}{ymet[0]}_label.png')
        plt.close()




    ymet =['dice_GM','dice_cerGM','dice_CSFv', 'dice_Put', 'dice_Pal', 'dice_Cau-acc', 'dice_thal', 'dice_hypp', 'dice_amyg']
    ymet =[ 'dice_Put', 'dice_Pal', 'dice_Cau-acc', 'dice_thal', 'dice_cerGM','dice_CSFv','dice_hypp', 'dice_amyg']
    ctitle = ['Putamen', 'Palidum', 'Caudate-Accubens', 'Thalamus', 'Cerebellum', 'Ventricle', 'Hippocampus','Amygdala']
    dfmm = df.melt(id_vars=[ 'model_name', 'dataset_name','input_type', 'label'], value_vars=ymet, var_name='from', value_name='dice');
    fig = sns.catplot(data=dfmm, y='dice', x='dataset_name', col='from', hue='model_name',
                      kind='boxen',hue_order=morderr[:9],col_wrap=4, palette=cc2[:9],
                      order=orderDS)


    ymet =[ 'dice_Put', 'dice_Pal', 'dice_Cau-acc', 'dice_thal',] ; ctitle = ['Putamen', 'Palidum', 'Caudate-Accubens', 'Thalamus', ]
    ymet =[  'dice_thal','dice_cerGM','dice_CSFv',] ; ctitle = [ 'Thalamus','Cerebellum', 'Ventricle', ]
    ymet =[ 'dice_hypp', 'dice_amyg','dice_GM'] ; ctitle = [ 'Hippocampus','Amygdala','GM' ]
    #ymet = ['dice_cerGM','dice_CSFv','dice_hypp', 'dice_amyg']; ctitle = ['Cerebellum', 'Ventricle', 'Hippocampus','Amygdala']
    for dn,df in zip(orderDS,df_list):
        xorder = ['Manu', 'Assn', 'Free'] if dn=='MICCAI' else ['Assn','Free']
        dfmm = df.melt(id_vars=[ 'model_name', 'dataset_name','input_type', 'label'], value_vars=ymet, var_name='from', value_name='dice');

        fig = sns.catplot(data=dfmm, y='dice', x='label', col='from', hue='model_name',
                          kind='boxen',hue_order=morderr[:5],col_wrap=1, palette=cc2[:5],order=xorder)
        for ii, ax in enumerate(fig.axes):
            #ax.set_title(ctitle[ii], fontdict=dict(fontsize=sizefont))
            ax.set_title('', fontdict=dict(fontsize=sizefont))
            ax.set_ylim(ylim); ax.set_yticks(yrange)
            #ax.set_ylim(0.75, 0.99);    ax.set_yticks(np.arange(0.75, 0.976, 0.025))
            #if (ii==0) | (ii==4):
            if ii > -1:  # ii>3:
                #ax.set_ylabel(f'Dice',fontsize=sizefont) #'Dice'   'Average Surface dist' 'Volume Ratio'
                ax.set_ylabel(ctitle[ii],fontsize=sizefont)
                #yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize=sizefont )
                ticks = ax.get_yticks()                     # positions réelles (pas les Text)
                labels = [f"{t:.0f}" if t % 1 == 0 else "" for t in ticks]
                ax.set_yticks(ticks, labels, fontsize=sizefont) ;ax.grid(True, axis='y')
            if (ii==2):
            #if ii > -1:  # ii>3:
                ax.set_xticklabels(xorder, rotation=0,fontsize="small" );ax.set_xlabel('')

        sns.move_legend(fig,'right',bbox_to_anchor=(1, .5),fontsize='x-small',frameon=False, shadow=True,
                        title=f'{dn}',title_fontsize=sizefont)
        plt.subplots_adjust(hspace=0.05, bottom=0.05, top=0.99)
        if dn=='MICCAI':
            plt.subplots_adjust(left=0.2)
        fig.savefig(f'{rdout}/V3_{ymet[0]}_{dn}_label.png')
        fig.savefig(f'{rdout}/V3_{ymet[0]}_{dn}_label.eps')

    rdout = "/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/figure/other_3label_3DS/"


    sel_col, yval, group_col_order = ['model_name', 'from'], 'dice', [morderr,ymet]  #subcorti avec ymet et dfmm
    dfs1 = select_df(dfmm,{'dataset_name':'MICCAIstd_testset_suj20_vol_T1'})
    dfs1 = select_df(dfmm,{'dataset_name':'MiBo'})
    make_the_table_gpt(dfs1,yval=yval, group_col=sel_col,group_col_order=[morderr[:-1],ymet], alpha=0.01)
    #save to csv dfs = select_df(df,{sel_col[0] : group_col_order[0][:-1],sel_col[1] : group_col_order[1] })
    dfs = select_df(df, {sel_col[0]: group_col_order[0][:-1], sel_col[1]: group_col_order[1]})



############################################# HCP ###############################
if True:
    df,morder,mordernn,cc,ymet = get_data('hcp')

    morderr = mordernn[:1] + mordernn[2:4] + mordernn[1:2] + mordernn[5:]; cc2 = cc[:1] + cc[2:4] + cc[1:2] + cc[5:]; cc2 = cc2[:1] + cc2[2:3] + cc2[1:2] + cc2[3:] #inver syntseg
    df.model_name = df.model_name.replace('SynthSegSuper', 'SuperSynth'); df.model_name = df.model_name.replace('SiamL', 'SIAM')
    morderr[2] = 'SuperSynth';    morderr[4] = 'SIAM'
    dfs = select_df(df,{'input_type': 'vol_T1_07', 'label':['Free','Assn']})
    #for all siams
    morderr = morderr[:1] + morderr[3:4] + ['blanc'] + morderr[4:9]  ; cc2 = cc2[:1] + cc2[3:4]+[(1,1,1)] +cc2[4:9]

    ymet =['dice_GM','dice_cerGM','dice_CSFv', 'dice_thal', 'dice_Put', 'dice_hypp', 'dice_Cau-acc', 'dice_Pal', 'dice_amyg']
    ymet =['dice_GM','dice_cerGM','dice_CSFv', 'dice_Put', 'dice_Pal', 'dice_Cau-acc', 'dice_thal', 'dice_hypp', 'dice_amyg']
    ymett =[ss.replace('dice','volume_ratio') for ss in ymet]
    ctitle = ['GM (V=203)', 'Cerebellum GM (V=41)','Ventricle (V=6.4)', 'thalamus (V=6.0)','Putamen (V=3.6)','Hippocampus (V=2.9)','Caudate-Accubens (V=2.8)','Palidum (V=1.3)','Amygdala (V=1)']#HCP
    ctitle = ['GM (V=223)', 'Cerebellum GM (V=45)','Ventricle (V=22)', 'thalamus (V=6.2)','Putamen (V=3.7)','Hippocampus (V=3.1)','Caudate-Accubens (V=3)','Palidum (V=1.3)','Amygdala (V=1)']#MICCAI
    ctitle = ['GM', 'Cerebellum','Ventricle', 'thalamus','Putamen','Hippocampus','Caudate-Accubens','Palidum','Amygdala']#MICCAI
    ctitle = ['GM', 'Cerebellum','Ventricle', 'Putamen','Palidum','Caudate-Accubens','thalamus','Hippocampus','Amygdala']#MICCAI

    dfmm = dfs.melt(id_vars=['sujnum', 'model_name', 'dataset_name', 'label'], value_vars=ymet, var_name='from', value_name='dice');
    #facet_kws=dict(sharex=False, sharey=False))

    fig = sns.catplot(data=dfmm, y='dice', x='label', col='from', hue='model_name', kind='boxen',hue_order=morderr[:5],col_wrap=3, palette=cc2[:5])

    sizefont = 'medium' #'x-large'
    ylim = [0.85, 0.99]; yrange = np.arange(0.85, 1, 0.025)
    for ii, ax in enumerate(fig.axes):
        ax.set_title(ctitle[ii], fontdict=dict(fontsize=sizefont))
        ax.set_ylim(ylim); ax.set_yticks(yrange)
        #ax.set_ylim(0.75, 0.99);    ax.set_yticks(np.arange(0.75, 0.976, 0.025))
        if (ii==0) | (ii==3)| (ii==6):
            ax.set_ylabel(f'Dice',fontsize=sizefont) #'Dice'   'Average Surface dist' 'Volume Ratio'
            yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize=sizefont )
        if ii>5:
            #ax.set_xticklabels(['AssN GT','Free GT'], rotation=0,fontsize=sizefont );ax.set_xlabel(' ')
            ax.set_xticklabels(['Manual GT','AssN GT','Free GT'], rotation=0,fontsize=sizefont );ax.set_xlabel(' ')

    sns.move_legend(fig,'right',bbox_to_anchor=(0.76, .28),fontsize='x-small',frameon=True, shadow=True, title='',title_fontsize='xx-small')


    df['GM_brain_ratio'] = df['volume_input_GM']/df['volume_input_brain']*100
    df['GM_brain_ratioGT'] = df['volume_input_GM']/df['volume_target_brain']*100
    selk=[ 'FastSurfer', 'GOUHFI', 'SIAM','SiamL','SynthMIDA3','SynthMIDA']
    selk=[ 'FastSurfer', 'GOUHFI', 'SIAM','SiamL', 'SynthMIDA','GOUHFI_T2','SIAM_T2','SiamL_T2','SynthMIDA_T2']
    selk=[ 'BibsN', 'GOUHFI', 'SIAM', 'SiamL', 'SynthMIDA']
    selk=[ 'GT','BibsN', 'GOUHFI', 'SIAM', 'SiamL', 'SynthMIDA', 'GOUHFI_T2', 'SIAM_T2', 'SiamL_T2', 'SynthMIDA_T2']
    dfs=select_df(df,{'model_name':selk,'input_type':'vol_T1_07','label':'Free'} )
    dfs=select_df(df,{'label':'Free'} )
    dfp = dfs.pivot(columns='model_name', values='GM_brain_ratio',index='sujnum')
    def corrfunc(x, y, **kws):
        r = np.corrcoef(x, y)[0, 1]
        ax = plt.gca()
        ax.annotate(f"r = {r:.2f}", xy=(0.1, 0.9), xycoords=ax.transAxes, fontsize=10)
    def identity_line(x, y, **kws):
        ax = plt.gca()
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]
        lims = ax.get_xlim()
        ax.plot(lims, lims, '--', color='gray', linewidth=1)
        #ax.set_xlim(lims)
        #ax.set_ylim(lims)
    g = sns.pairplot(dfp, kind="reg", diag_kind="kde", corner=True)
    g.map_lower(corrfunc); g.map_lower(identity_line)
    corr = dfp.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=0.8, vmax=1, square=True)
    ax=plt.gca()
    ax.set_ylabel('');ax.set_xlabel('');yyy = ax.get_xticklabels(); ax.set_xticklabels(yyy, fontsize='large'); yyy = ax.get_yticklabels(); ax.set_yticklabels(yyy, fontsize='large');

    #CSFv volume
    volu = [df[k].mean() for k in ymetv]
    tname = [f'{ss[14:]} V={y/min(volu):0.1f}' for y,ss in sorted(zip(volu, ymetv),reverse=True)]
    ymets = [ss for y,ss in sorted(zip(volu, ymet),reverse=True) ]

    dfs = df[(df.label=='Free') ]; dfs = select_df(dfs,{'input_type': 'T1'})
    dfs = dfs.sort_values(by='volume_target_CSFv')

    mm = [ mordernn[0]] + [mordernn[3]] + mordernn[5:]
    ccc = [ cc[0]] + [cc[3]] + cc[5:]
    fig=plt.figure();plt.plot(range(82),dfss.volume_target_CSFv*0.7**3/1000,marker='x');plt.ylabel('total Ventricle volume in cm^3',fontsize='x-large');plt.xlabel('Subject order by increasing Ventricle Volume',fontsize='x-large')
    fig=sns.relplot(data=dfs, y='volume_ratio_CSFv', x='subject_id', col='label', hue='model_name', kind='line',hue_order=mordernn[:4], palette=cc)
    ax = fig.axes[0][0];plt.xlim([0, 82 ]); ax.set_xticklabels('', rotation=0,fontsize='x-large' )
    sns.move_legend(fig,'right',bbox_to_anchor=(.8, .8),fontsize='large',frameon=True, shadow=True, title=f'Model',title_fontsize='large')
    ax.set_xticklabels('', rotation=0,fontsize='x-large' );ax.set_xlabel('Subject order by increasing Ventricle Volume',fontsize='x-large')
    ax.set_ylabel(f' (Vol predict) / (Vol GT Free) ',fontsize='x-large');yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize='large' )
    ax.set_title('Ventricle', fontsize='x-large')

    #HCP region
    df,morder,mordernn,cc,ymet = get_data('hcp_reg',model_flat=('T2'))

    fig = sns.catplot(data=df, y='hausdorff_GM',x='label', hue='model_name',kind='boxen', hue_order=mordernn,col=df[['dataset_name','region']].apply(tuple,axis=1), col_wrap=4)
    #pour des sclae en y different !!!  facet_kws=dict(sharex=False, sharey=False))

############################################# MICCAI ###############################
if True:
    df,morder,mordernn,cc,ymet = get_data('miccai')
    #df,morder,mordernn,cc,ymet = get_data('miccai',add_GT_as_model=True)

    morderr = mordernn[:1] + mordernn[2:4] + mordernn[1:2] + mordernn[5:6]; cc2 = cc[:1] + cc[2:4] + cc[1:2] + cc[4:5]; cc2 = cc2[:1] + cc2[2:3] + cc2[1:2] + cc2[3:] #inver syntseg
    df.model_name = df.model_name.replace('SynthSegSuper', 'SuperSynth'); df.model_name = df.model_name.replace('SiamL', 'SIAM')
    morderr[2] = 'SuperSynth';    morderr[4] = 'SIAM'
    ymet =['dice_GM','dice_cerGM','dice_CSFv', 'dice_Put', 'dice_Pal', 'dice_Cau-acc', 'dice_thal', 'dice_hypp', 'dice_amyg']
    dfmm = df.melt(id_vars=['sujnum', 'model_name', 'dataset_name', 'label'], value_vars=ymet, var_name='from', value_name='dice');
    xorder=[ 'gt','Assn', 'Free']
    fig = sns.catplot(data=dfmm, y='dice', x='label', col='from', hue='model_name', kind='boxen',
                       hue_order=morderr[:5],col_wrap=3, palette=cc2[:5], order=xorder)

    morder2 = mordernn[2:6]+mordernn[:2] ; cc2 = cc[2:6]+cc[:2]; morder2 = mordernn[:4] ; cc2 = cc[:4]; morder2 = mordernn[3:] ; cc2 = cc[3:]
    xorder=[ 'gt','Assn', 'Free']
    g = sns.FacetGrid(dfmm,col='from', col_wrap=3,sharey=False, legend_out=True, despine=True,height=4, aspect= 1.33)
    g = g.map_dataframe(sns.boxenplot, x='label', y='dice', hue='model_name',hue_order=morder2, palette=cc2,order=xorder)
    g.add_legend()
    sizefont = 'medium' #'x-large'
    for ii, ax in enumerate(fig.axes):
        ax.set_title(ctitle[ii], fontdict=dict(fontsize=sizefont))
        if (ii==0) | (ii==3)| (ii==6):
            ax.set_ylabel(f'Dice',fontsize=sizefont)
            yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize=sizefont )
        #yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize=sizefont )
        ax.set_xticklabels(['Manual GT','AssN GT','Free GT'], rotation=0,fontsize=sizefont );ax.set_xlabel(' ')
    #manual left bot righ top ws hs = 0.076 0.04 1 0.97 0.02 0.098
    sns.move_legend(g,'right',bbox_to_anchor=(1, .66),fontsize='large',frameon=True, shadow=True, title=f'model ',title_fontsize='x-large')


############################################# DBB ###############################
if True:
    df,morder,mordernn,cc,ymet = get_data('dbb')
    morderr = mordernn[:1] + mordernn[2:4] + mordernn[1:2] +  mordernn[5:6]; cc2 = cc[:1] + cc[2:4] + cc[1:2] + cc[4:5];cc2 = cc2[:1] + cc2[2:3] + cc2[1:2] + cc2[3:]  # inver syntseg
    morderr[2] = 'SuperSynth';    morderr[4] = 'SIAM'
    df.model_name = df.model_name.replace('SynthSegSuper', 'SuperSynth');
    df.model_name = df.model_name.replace('SIAM', 'SIAMmm'); df.model_name = df.model_name.replace('SiamL', 'SIAM')

    ymet=['dice_GM','dice_dGM',]
    dfmm = df.melt(id_vars=[ 'model_name', 'dataset_name', 'label','group'], value_vars=ymet, var_name='from', value_name='dice');
    fig = sns.catplot(dfmm, y='dice', x='group',kind='boxen',hue='model_name',hue_order=morderr, palette=cc2, col='from', col_wrap=2)
    ctitle = ['GM', 'Deep Nucleus']
    sizefont = 'small'
    for ii, ax in enumerate(fig.axes):
        ax.set_yticks(np.arange(0.5, 1, 0.1))
        ax.set_title(ctitle[ii], fontdict=dict(fontsize='medium'))
        if (ii==0) | (ii==3)| (ii==6):
            ax.set_ylabel(f'Dice',fontsize=sizefont) #'Dice'   'Average Surface dist' 'Volume Ratio'
            yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize=sizefont )
        if ii>=0:
            ax.set_xticklabels(['N=14', 'XXL Ventricle (N=4)'], rotation=0,fontsize=sizefont );ax.set_xlabel(' ')

    for yy in ymet:
        sns.catplot(df, y=yy, x='group',kind='boxen',hue='model_name',hue_order=mordernn, palette=cc)

############################################# ULTRA ###############################
if True:
    df,morder,mordernn,cc,ymet = get_data('ultra',model_flat='train',model_flat_column='group')

    df,morder,mordernn,cc,ymet = get_data('ultra')
    #morderr = mordernn[:2] + mordernn[3:7]; cc2 = cc[:2] + cc[3:7]
    df.model_name = df.model_name.replace('SIAM', 'SIAMmmm'); df.model_name = df.model_name.replace('SiamL', 'SIAM')
    morderr = mordernn[:2]; cc2 = cc[:2]
    df.input_type = df.input_type.replace('vol_ct','CT'); df.input_type = df.input_type.replace('vol_ute','UTE'); df.input_type = df.input_type.replace('vol_flair','FLAIR');df.input_type = df.input_type.replace('vol_uni','UNI')
    xorder= ['CT', 'UTE', 'FLAIR', 'UNI']
    fig = sns.catplot(data=df, y='dice_skull', x='input_type', hue='model_name', kind='boxen',
                      hue_order=morderr , palette=cc2, order=xorder)
    ax = fig.axes[0][0]
    ax.set_ylim([0.7 ,1]); ax.set_ylabel(f'Dice Skull'); ax.set_xlabel('');
    ymet = [ 'dice_skull','dice_CSFv', 'dice_GM']#, 'dice_Pal', 'dice_Put']
    #ymet = [ 'volume_ratio_skull', 'volume_ratio_GM']
    dfmm = df.melt(id_vars=[ 'model_name', 'dataset_name','input_type', 'label','group'], value_vars=ymet, var_name='from', value_name='dice');
    #sns.catplot(dfmm, y='dice', x='input_type',kind='boxen',hue='model_name',hue_order=mordernn, palette=cc, col='from', col_wrap=2)
    dfmms = dfs.melt(id_vars=[ 'model_name', 'dataset_name','input_type', 'label','group'], value_vars=ymet, var_name='from', value_name='dice');

    xorder= ['vol_ct', 'vol_ute', 'vol_flair', 'vol_inv1', 'vol_inv2', 'vol_uni']
    g = sns.FacetGrid(dfmm,col='from', col_wrap=2,sharey=False, legend_out=True,
                      despine=True,height=6, aspect= 1.33)
    g = g.map_dataframe(sns.boxenplot, x='input_type', y='dice', hue='model_name',hue_order=mordernn, palette=cc, order=xorder)
    # Ajouter les croix rouges
    for ax, var in zip(g.axes.flatten(), g.col_names):
        # On filtre le sous-DataFrame pour cette colonne ("from")
        subset = dfmms[dfmms['from'] == var]
        sns.stripplot(
            data=subset,x='input_type', y='dice',hue='model_name', hue_order=mordernn,
            order=xorder,marker='x', color='red', s=5, dodge=True, linewidth=2,
            jitter=0.05, ax=ax
        )
    g.add_legend()
    ctitle = ['Skull', 'Ventricle','GM']#  , 'Palidum', 'Putamen']
    xlabel=['CT', 'UTE', 'FLAIR', 'INV1', 'INV2', 'UNI']
    for ii, ax in enumerate(g.axes):
        ax.set_title(ctitle[ii], fontdict=dict(fontsize='x-large'))
        if ii==0:
            ax.set_ylim([0.75, 0.96]) #skull
            #ax.set_ylim([0.7, 1.1]) #Vol ratio
        elif ii==1:
            ax.set_ylim([0.7, 0.95])
            #  ax.set_ylim([0.9, 1.5])
        else:
            ax.set_ylim([0.6, 0.95])
            #  ax.set_ylim([0.9, 1.5])

        if ii>-1:#(ii==0) | (ii==2):
            ax.set_ylabel(f'Dice',fontsize='x-large')
            #ax.set_ylabel(f'Vol predict / Vol GT',fontsize='x-large')
            yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize='large' )

        ax.set_xticklabels(xlabel, rotation=0,fontsize='x-large' );ax.set_xlabel(' ')

    ax.set_ylabel(f' GM : (Vol predict) / (Vol GT) ',fontsize='x-large');yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize='large' )
    ax.set_title('GM volume ratio', fontdict=dict(fontsize='x-large'))

    dfs,sel=select_df_ask(df)
    g = sns.FacetGrid(dfs,col='input_type', col_wrap=2,sharey=True, legend_out=True,
                      hue='model_name',hue_order=mordernn, palette=cc,
                      despine=True,height=6, aspect= 1.33 )
    g = g.map_dataframe(sns.lineplot, x='sujnum', y='dice_skull');g.add_legend()


    sns.catplot(data=df,x='model_name', y='dice_skull', order=mordernn, palette=cc, col='input_type',
                col_wrap=2,kind='strip')


############################################# confusion mat HCP #############################
df,morder,mordernn,cc,ymet = get_data('hcp_confu_reg',model_flat=('T2')) #df,morder,mordernn = get_data('hcp_confu_reg')
dfg = group_by_region_df(df)
dfg = get_metric_from_confusion(dfg)
kconf,kmet,kdiag = get_confu_all_lab(df,'','GM')
kmet=['CSFv']#['CSF','WM'] #['BG','CSF','head']
ycmet, ycmet_short = get_confu_all_lab(df,kmet,'GM')
ycmet, ycmetA = get_confu_all_lab(df, kmet, 'GM','conf_sum')
#group
#dfgn = norm_confusion(dfg,'GM',do_norm='errors')
dfgn = norm_confusion(dfg,'GM',do_norm='volume')
dfg, dfgn = corect_confusion_head_bg(dfg), corect_confusion_head_bg(dfgn)
dfgn = get_confusion_ratio(dfgn, kmet, 'GM')
 #region
dfn = norm_confusion(df,'GM',do_norm='volume');
dfn = corect_confusion_head_bg(dfn);
dfn = get_confusion_ratio(dfn, kmet, 'GM')
fig = sns.catplot(data=dfmm, y='c',x='from', hue='model_name',kind='boxen',col='region', hue_order=morder2, palette=cc)
mordernnn = ['FastSurfer','GOUHFI','GOUHFI_T2', 'SiamL','SiamL_T2']
dfs = select_df(dfn,{'label': 'siam','model_name':mordernnn} )
dfs['region'] = dfs['region'].replace('pariental','parietal')
fig = sns.catplot(data=dfmm,x='model_name', y='c', hue='region',kind='boxen',hue_order=['frontal','temporal','parietal','occipital'], order=mordernnn,col='from')
ctitle = ['Confusion CSF','Confusion WM']; ylabel = f'1 - Dice' ; ylabel='(FP - FN)/(FP + FN) * 100'
for ii, ax in enumerate(fig.axes[0]):
    ax.set_title(ctitle[ii], fontdict=dict(fontsize='x-large'))
    if ii==0:
        ax.set_ylabel(ylabel, fontsize='x-large'); yy = ax.get_yticklabels();
        ax.set_yticklabels(yy, fontsize='large')
    yy = ax.get_xticklabels();
    ax.set_xticklabels(yy, rotation=0, fontsize='x-large');
    ax.set_xlabel('', fontsize='x-large')


dfmm = dfgn.melt(id_vars=['sujnum', 'model_name', 'input_type','dataset_name', 'label','region'], value_vars=ycmet, var_name='from',value_name='c');
for m1,m2 in zip(ycmet, ycmet_short):
    dfmm['from'] = dfmm['from'].replace(m1,m2)
fig = sns.catplot(data=dfmm, y='c',x='from', hue='model_name',kind='boxen',col='label',col_wrap=1, hue_order=mordernn, palette=cc)
#kind='strip',dodge=True,,jitter=0.25

g = sns.FacetGrid(dfmm,col='label', col_wrap=1,sharey=False, legend_out=True, despine=True,height=4, aspect= 1.33)
g = g.map_dataframe(sns.boxenplot, x='from', y='c', hue='model_name',hue_order=mordernn, palette=cc)
g.add_legend()

ctitle = ['AssN GT','Free GT', 'SIAM GT']
xlabel= ['GT CSF', 'Pred CSF', 'GT WM', 'Pred WM'];
xlabel= ['Confusion CSF','Confusion WM']; ylabel = '1 - Dice'; ylabel='(FP - FN)/(FP + FN) * 100'
for ii, ax in enumerate(fig.axes):
    ax.set_title(ctitle[ii], fontdict=dict(fontsize='x-large'))
    if ii==3:
        ax.set_xticklabels(xlabel, rotation=0, fontsize='x-large');
        ax.set_xlabel(' ')

    ax.set_ylabel(ylabel,fontsize='x-large')
    yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize='large' )
    ax.set_xticklabels(xlabel, rotation=0,fontsize='x-large' );ax.set_xlabel(' ')

#ventricle
df,morder,mordernn,cc,ymet = get_data('hcp_confu_reg')
df = select_df(df,{'input_type': 'vol_T1_07', 'label':['Assn'],'model_name':['FastSurfer','GOUHFI', 'SynthSeg','SIAM','lab_Free']})
morder2 = mordernn[:4] ; cc2 = cc[:4];
morder2 = ['lab_Free'] + morder2; cc2=[(0.1,0.3,0.9)] + cc2



ymet=[]
for k in df.keys():
    #if 'Sdis_' in k:
    if 'dice' in k:
            ymet.append(k)
ymet.pop(-1)
labels =[ss[5:]  for ss in ymet] #BG to head
labels2 = ['GM','CSFv','CSF','cerGM','head']

for li in labels2:
    ycmet, yynn = get_confu_suj_average(dfonorm,li, labels); ycmet = list(ycmet.keys())
    dfmm = dfo.melt(id_vars=['sujnum', 'model_name', 'dataset_name', 'label'], value_vars=ycmet, var_name='from',value_name='c');
    for m1,m2 in zip(ycmet, yynn):
        dfmm['from'] = dfmm['from'].replace(m1,m2)

    fig = sns.catplot(data=dfmm, y='c',x='from', hue='model_name',kind='boxen', hue_order=mordernn2)
    #fig = sns.catplot(data=dfmm, y='c',x='from',col='dataset_name', hue='model_name',kind='boxen',hue_order=mordernn2, col_wrap=1, height=5, aspect=2)
    plt.title(f'Confu {li}')

morder2 = mordernn[:7]; cc2 = cc[:7]  ; morder2 = mordernn[5:]; cc2 = cc[5:]  ;
morder2 = mordernn[:3] + mordernn[7:9] + mordernn[13:] ; cc2 = cc[:3] + cc[7:9] + cc[13:] #remove SIAM SynthSeg vasc skull
morder2 = mordernn[:3] + mordernn[5:9] + mordernn[15:] ; cc2 = cc[:3] + cc[5:9] + cc[15:] #remove SIAM SynthSeg vasc skull

lab_sel = [ 'GM', 'WM',  'Pal', 'Cau-acc',]
ycmet, ycmet_short = get_confu_all_lab(df,lab_sel,'Put')
lab_sel = ['BG',  'CSF', 'head', 'WM',]
lab_sel = ['CSFv', 'cerGM','thal','Pal', 'Put', 'Cau-acc', 'amyg', 'hypp']

ycmet, ycmet_short = get_confu_all_lab(df,lab_sel,'GM')
dfs = dfg[dfg.label=='Free']
dfmm = dfs.melt(id_vars=['sujnum', 'model_name', 'input_type','dataset_name', 'label','region'], value_vars=ycmet, var_name='from',value_name='c');
for m1,m2 in zip(ycmet, ycmet_short):
    dfmm['from'] = dfmm['from'].replace(m1,m2)
fig = sns.catplot(data=dfmm, y='c',x='from', hue='model_name',kind='boxen',col='region', hue_order=morder2, palette=cc)

g = sns.FacetGrid(dfmm,row='label',sharey=False, legend_out=True, despine=True,height=4, aspect= 4.33)
g = g.map_dataframe(sns.boxenplot, x='from', y='c', hue='model_name',hue_order=morder2, palette=cc2)
g.add_legend()
ctitle = ['AssN GT', 'Free GT', 'SIAM GT']
for ii, ax in enumerate(g.axes):
    ax = ax[0]
    ax.set_title(ctitle[ii], fontdict=dict(fontsize='x-large'))
    ax.set_ylabel(f'nb voxels',fontsize='x-large') #'Dice'   'Average Surface dist' 'Volume Ratio'
    yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize='large' )
    if ii==2:
        xx = ax.get_xticklabels();
        ax.set_xticklabels(xx, rotation=0,fontsize='x-large' );ax.set_xlabel(' ')


#checked with mrtrix S01 HCP
#confusion_GT_CSF_P_GM 53191   siamT1==3 & siamT2==1
# confusion_GT_GM_P_CSF 9802   siamT1==1 & siamT2==3
dfss['rrr'] = dfss.confusion_GT_WM_P_GM - dfss.confusion_GT_GM_P_WM
#SIAM More pred GM in WM   S75:48787  S24 : 35431.0 S57: 35276.0
#les     -24125.0 S21  -18013.0 S29
#siam confusion_GT_BG_P_GM,  S40:162732  S72 150165

#check volumes on synth training DS
dsyn='/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/MidaS1_all/synth_bin'
dsyn='/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/Mida_all/synth_bin'
dsyn='/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/Vascular4_mida/synth_bin'
flab = gfile(dsyn,'Lab_gen.*AffRem.*gz')
fcsv = gfile(dsyn,'Sim_gen.*AffRem.*csv')
flab = gfile(dsyn,'Lab.*Suj_02.*AffRem.*gz')
fcsv = gfile(dsyn,'Sim_gen.*Suj_02.*AffRem.*csv')
flab,fcsv=[],[]
for f1,f2 in zip(gfile(dsyn,'^Lab.*gz'),gfile(dsyn,'^Sim.*csv')):
    if 'GM' in f1:
        continue
    flab.append(f1);fcsv.append(f2);

from utils_labels import get_label_set_map
mm = get_label_set_map(1) #
gmlab = mm['name_map']['GM']
fin = mm['files'][2]

il = tio.LabelMap(fin)
dfnew = pd.DataFrame();
vol = (il.data==gmlab).sum().numpy() * 0.25**3 / 0.75**3 #from 0.25 mm to 0.75
dfnew['volgm'] = [vol];
dfnew['scale'] = [1];dfnew['lab'] = [fin]
df_list = [dfnew]
for ii,(fc,fi) in enumerate(zip(fcsv,flab)):
    dfaff = pd.read_csv(fc)
    aff_dic = eval(dfaff.iloc[0,1].replace('true','True'))
    scale = np.prod(aff_dic['scales'])
    il = tio.LabelMap(fi)

    dfnew = pd.DataFrame()
    dfnew['scale'] = [scale]
    dfnew['volgm'] = [(il.data==gmlab).sum().numpy()]
    dfnew['lab'] = [fi]

    df_list.append(dfnew)

    if ii>100 :
        break
df1 = pd.concat(df_list)
df1['volscale_vas_nodil'] = df1.volgm/df1.scale

dfmm = df3.melt(id_vars=['lab'], value_vars=['volscaleM1','volscale', 'volscaleV4',], var_name='from',value_name='v');
