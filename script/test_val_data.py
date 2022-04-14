from utils import print_accuracy, print_accuracy_df, print_accuracy_all
from utils_file import gfile, gdir, get_parent_path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

prefix = "/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/"

# CATI label
# rlabel  = pd.read_csv(prefix+'/CATI_datasets/all_cati_mriqc_pred.csv')
rlabel = pd.read_csv(prefix + '/CATI_datasets/all_cati.csv')
rlabell = pd.read_csv(prefix + '/CATI_datasets/cenir_cati_QC_et_lession.csv', index_col=0)
# construc the sujid
rid = []
for ff in rlabel.cenir_QC_path.values:
    dd = ff.split('/')
    if dd[-1] is '': dd.pop()
    nn = len(dd)
    rid.append(dd[nn - 3] + '+' + dd[nn - 2] + '+' + dd[nn - 1])

rlabel.index = rid
rlabel = rlabel.sort_index()  # alphabetic order
labelsujid = rlabel.index

rr = rlabell.reindex(rlabel.index).loc[:, ['lesion_PV', 'lesion_WM']]
rlabel = pd.concat([rlabel, rr], axis=1, sort=True)

# reorder the label as res[0]
# rlabel = rlabel.loc[sujid]
# ytrue = rlabel.QCOK.values
ytrue = rlabel.globalQualitative.values
print_accuracy_df(rlabel, ytrue)

# prediction mriqc
rd = '/network/lustre/dtlake01/opendata/data/ABIDE/mriqc_data/retrain'
resfile = gfile(rd, 'data_CATI.*csv$')
resname = get_parent_path(resfile, 1)[1]
res = [pd.read_csv(f) for f in resfile]

for ii, rr in enumerate(res):
    sujid = []
    for ff in rr.subject_id:
        dd = ff.split('/')
        if dd[-1] is '': dd.pop()
        nn = len(dd)
        sujid.append(dd[nn - 3] + '+' + dd[nn - 2] + '+' + dd[nn - 1])
    rr.index = sujid
    res[ii] = rr.loc[labelsujid]  # rr.loc[sujid[::-1]]

print_accuracy(res, resname, ytrue, prediction_name='prob_y', inverse_prediction=False)
print_accuracy_all(res[0:1], resname[0:1], ytrue, prediction_name='prob_y', inverse_prediction=False)
# CAT12
rescat = pd.read_csv('/home/romain.valabregue/datal/QCcnn/CATI_datasets/res_cat12_suj18999.csv')
rescat.index = [sss.replace(';', '+') for sss in rescat.sujid]  # .values.replace(";","+")
rescat = rescat.loc[labelsujid]
print_accuracy_df(rescat, ytrue)
print_accuracy([rescat], ['IQR'], ytrue, prediction_name='IQR', inverse_prediction=False)


# READ RESULT cnn
prefix = "/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/"
resdir = "/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/predict_torch/"
rd = gdir(resdir, '^cati')
resfile = gfile(rd, 'all.csv')
resname = get_parent_path(resfile, 2)[1]
res = [pd.read_csv(f) for f in resfile]

for ii in range(len(resname)):
    print(ii)
    ssujid = []
    for ff in res[ii].fin:
        dd = ff.split('/')
        if dd[-1] is '': dd.pop()
        nn = len(dd)
        ssujid.append(dd[nn - 5] + '+' + dd[nn - 4] + '+' + dd[nn - 3])
    res[ii].index = ssujid
    if ii == 0: sujid = ssujid
    aa = set(ssujid).difference(set(labelsujid));
    print(aa)
    aa = set(labelsujid).difference(set(ssujid));
    print(aa)
    res[ii] = res[ii].loc[labelsujid]

print_accuracy(res, resname, ytrue, prediction_name='ymean', inverse_prediction=True,note_thr=3)
print_accuracy([rescat], "cat12", ytrue, prediction_name='IQR', inverse_prediction=True)


def mybin(x):
    if x < 3:
        return 0
    else:
        return 1


# reoder other res as res[0
for ii, rr in enumerate(res):
    rr.index = sujid
    rr = rr.loc[sujid]
    res[ii] = rr
    res.ymean = 1 - res.ymean  # label 1 is artefacted mriqc convention

print_accuracy(res, resname, ytrue, prediction_name='prob_y', inverse_prediction=False)

# ds30
rlabel = pd.read_csv('/network/lustre/dtlake01/opendata/data/ds000030/y_ds030.csv')
rlabel = pd.read_csv('/network/lustre/dtlake01/opendata/data/ds000030/y_ds030_noghost.csv')
rlabel.index = pd.Series(['sub-' + str(rrr) for rrr in rlabel.subject_id])
sujid = rlabel.index

resx = pd.read_csv('/network/lustre/dtlake01/opendata/data/ds000030/x_ds030.csv')
resx.index = pd.Series(['sub-' + str(rrr) for rrr in resx.subject_id])
resx = resx.loc[sujid]
print_accuracy_df(resx, rlabel.rater_1, note_thr=0)

resfile = '/home/romain.valabregue/datal/QCcnn/predict_torch/ds30_torcho2_msbrain098_equal_BN05_b4_BCEWithLogitsLoss_SDG/all.csv'
res, resname = pd.read_csv(resfile), get_parent_path([resfile], 2)[1]
resfile = '/network/lustre/dtlake01/opendata/data/ds000030/cat12_res.csv'
rescat, resnamecat = pd.read_csv(resfile), get_parent_path([resfile], 2)[1]

res.index = get_parent_path(res.fin, 3)[1]
rescat.index = rescat.sujid

aa = set(rlabel.index).difference(set(res.index));
print(aa)
bb = set(res.index).difference(set(rlabel.index));
print(bb)
res, rescat = res.drop(bb), rescat.drop(bb)

res = res.loc[sujid]
rescat = rescat.loc[sujid]

ytrue = rlabel.rater_1.values

print_accuracy_df(rescat, rlabel.rater_1, note_thr=0)

print_accuracy([res], resname, ytrue, prediction_name='ymean', inverse_prediction=True)
print_accuracy([rescat], resnamecat, ytrue, prediction_name='IQR', inverse_prediction=True)

# ABIDE
rlabel = pd.read_csv('/network/lustre/dtlake01/opendata/data/ABIDE/mriqc_data/y_abide.csv')
rlabel.index = pd.Series(['sub-' + str(rrr) for rrr in rlabel.subject_id])
sujid = rlabel.index

resfile = '/home/romain.valabregue/datal/QCcnn/predict_torch/abide_torcho2_msbrain098_equal_BN05_b4_BCEWithLogitsLoss_SDG/all.csv'
res, resname = pd.read_csv(resfile), get_parent_path([resfile], 2)[1]
resfile = '/network/lustre/dtlake01/opendata/data/ABIDE/res_cat12.csv'
rescat, resnamecat = pd.read_csv(resfile), get_parent_path([resfile], 2)[1]

res.index = get_parent_path(res.fin, 3)[1]
rescat.index = rescat.sujid

aa = set(rlabel.index).difference(set(res.index));
print(aa)
bb = set(res.index).difference(set(rlabel.index));
print(bb)
rlabel = rlabel.drop(aa)

res = res.loc[rlabel.index]
rescat = rescat.loc[rlabel.index]

ytrue = rlabel.rater_3.values

print_accuracy([res], resname, ytrue, prediction_name='ymean', inverse_prediction=True)
print_accuracy([rescat], resnamecat, ytrue, prediction_name='IQR', inverse_prediction=True)

# x mriqc
resx = pd.read_csv('/home/romain.valabregue/datal/QCcnn/res/res_mriqc_18999.csv')
ss = resx.sujid
rid = []
for ff in resx.sujid:
    dd = ff.split('/')
    if dd[-1] is '': dd.pop()
    nn = len(dd)
    rid.append(dd[nn - 3] + '+' + dd[nn - 2] + '+' + dd[nn - 1])
resx = resx.drop('sujid', axis=1)
resx.index = rid
resx = resx.loc[rlabel.index]

print_accuracy_df(resx, ytrue)

# mric
r = pd.read_csv('/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Results/mriqc/results_cv/train_abide/filtered_res.csv')
rr = r.loc[:, r.columns.str.contains('(mean|std)')]
rr = rr.loc[:, rr.columns.str.contains('roc_auc')]
# ii=rr.loc[:,'mean_abide_roc_auc'].idxmax()
ii = rr.loc[:, 'mean_test_roc_auc'].idxmax()
rr.loc[ii, :]

# sequence param
rparam = pd.read_csv('/home/romain.valabregue/datal/dicom/res/res_param_18999_one_col_less.csv')
rparam.index = [sss.replace('/', '+') for sss in get_parent_path(rparam.Var1)[0]]  # .values.replace(";","+")
bb = set(sujid).difference(set(rparam.index));
print(bb)
rparam = rparam.loc[sujid]
# pd.concat([df3, df4], axis='col')

F1 = rparam.ManufacturerModelName.astype('category')
voxprod = rparam.PixelSpacing_1 * rparam.PixelSpacing_2 * rparam.SliceThickness

# c('ManufacturerModelName','CoilString','AccelFactPE','PATMode','voxprod')


# write test set in csv file
rtest = pd.DataFrame()
rtest['fin'] = fin
rtest['fmask'] = fmask
rtest['faff'] = faff
ffref = []
ffref = [fref for i in range(len(fin))]
rtest['fref'] = ffref
rtest['sujid'] = sujid
rtest.index = sujid
rtest = rtest.loc[sujid]

rlabel = rlabel.loc[sujid]

ytrue = rlabel.globalQualitative.copy()
ytrue[rlabel.globalQualitative < 2] = 0;
ytrue[rlabel.globalQualitative > 1] = 1;

rtest['isOK'] = ytrue.astype(int)
rtest.to_csv('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/CATI_datasets/cati_test_cat12_ms.csv', index=False)

ytrue = np.concatenate([np.ones((200, 1)), np.zeros((10000, 1))], axis=0)
rtest.to_csv('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/val_cat12_ms.csv', index=False)

rr = pd.read_csv('/home/romain.valabregue/datal/QC/CATI/cenir_cati_QC_no_doublon.csv')
y1 = pd.read_csv('/home/romain.valabregue/datal/QCcnn/CATI_datasets/CENIR_Baltazar_WMH.csv')
y1.index = y1.subject

sujcat = rlabel.sujcat
sujcatall = rr.sujcat
rr.index = sujcatall

# 132 do not match   255 match 1   70 match 2

y2 = pd.read_csv('/home/romain.valabregue/datal/QCcnn/CATI_datasets/CENIR_MEMENTO_WMH.csv')

sid = np.array(['%.4d%s' % (a, b) for a, b in zip(y2.participant, y2.code)])
y2.index = sid
y2 = y2.drop(sid[y2.centre.values == 20])

y2 = y2.rename(columns={"ParaV": "Fazekas Parventricular grade", "SB Profonde": "Fazekas WM grade"})
y2['subject'] = y2.index
y2 = y2[['subject', 'Fazekas Parventricular grade', 'Fazekas WM grade']]
y1 = y1.append(y2)

mko = []
for s1 in y1.subject:
    match = [s for s in sujcat if s1 in s]
    if len(match) == 0:
        match2 = [s for s in sujcatall if s1 in s]
        if len(match2) == 0:
            print('NOT FOUND {}'.format(s1))
        else:
            rrr = rr.loc[match2]
            print('found {} {} for {} suj {} QC {}'.format(len(match2), match2, s1, rrr.suj.values, rrr.globalQualitative.values))
        mko.append(s1)

mok = []
yy = pd.DataFrame()
for s1, row in y1.iterrows():
    match = [s for s in sujcat if s1 in s]
    match = [];
    lid = []
    for ind, s in enumerate(sujcat):
        if s1 in s:
            match.append(s)
            lid.append(ind)

    if len(match) == 0:
        continue

    for kk, mm in enumerate(match):
        rrrindex = [rlabel.index[lid[kk]]]  # rlabel index for the match
        aaa = pd.DataFrame({'sujid_cat': mm, 'sujcat': row['subject'], 'lesion_PV': row['Fazekas Parventricular grade'], 'lesion_WM': row['Fazekas WM grade']}, index=rrrindex)
        yy = yy.append(aaa)

rrr = pd.concat([rlabel, yy.reindex(rlabel.index)], sort=True, axis=1)

aaa = rrr.iloc[~ np.isnan(rrr.lesion_PV.values)]


# concatenate res on all CATI
resdir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QC/CATI/res/'
rlabel = pd.read_csv(resdir+'CATI_allQC_center.csv')

rmriqc = pd.read_csv(resdir+'res_mriqc_9118_singu15.csv')
rmriqc = rmriqc.drop(['bids_meta','provenance'],axis=1)
rmriqc.index  = rmriqc.sujid.str.replace('/','+')

# construc the sujid
rid = []
for ff in rlabel.serie_path_proc:
    dd = ff.split('/')
    if dd[-1] is '': dd.pop()
    nn = len(dd)
    rid.append(dd[nn - 5] + '+' + dd[nn - 4] + '+' + dd[nn - 3] + '+' + dd[nn - 2] + '+' + dd[nn - 1])
rlabel.index = rid

rlabel = rlabel.loc[rmriqc.index.values]
rlabel.to_csv(resdir + 'CATI_allQC_center_sorted.csv')


res = pd.read_csv('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QC/CATI/CATI_allQC_center.csv')
ii=res['scanner_name'].isna()
sn = res['scanner_name']
sn.value_counts()
sn.value_counts().sum()

rall = pd.read_csv('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QC/CATI/res/CATI_allQC_center_sorted.csv')
res_cenir = pd.read_csv('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QC/CATI/res/all_cati.csv')

sujid_cen = res_cenir['sujcat'] + res_cenir['proto_cata'] + res_cenir['center'] # + rcen['visit']
sujid_all = rlabel['subject_Name_QualiCATI'] + rlabel['protocol'] + rlabel['center'] # + rcen['visit']

rlabel.index=sujid_all
res_cenir.index = sujid_cen

not_in_all = set(sujid_cen).difference(set(sujid_all))

res_cenir = res_cenir.drop(index=not_in_all)
sujid_cen = res_cenir['sujcat'] + res_cenir['proto_cata'] + res_cenir['center'] # + rcen['visit']

rlabel_not_cenir = rlabel.drop(index=sujid_cen)

#take again same index as mriqc results
rid = []
for ff in rlabel_not_cenir.serie_path_proc:
    dd = ff.split('/')
    if dd[-1] is '': dd.pop()
    nn = len(dd)
    rid.append(dd[nn - 5] + '+' + dd[nn - 4] + '+' + dd[nn - 3] + '+' + dd[nn - 2] + '+' + dd[nn - 1])
rlabel_not_cenir.index = rid

index_cenir = set(rmriqc.index).difference(set(rlabel_not_cenir.index))
rmriqc_not_cenir = rmriqc.drop(index=index_cenir)

#now exclude center un peu laborieu mais ca marche !
v = rlabel_not_cenir['scanner_name']
vv = v.value_counts()
scanner_remove = vv.index[v.value_counts().lt(10)]

index_remove = [False if  (aa in scanner_remove) else True for aa in v]
rr = rlabel_not_cenir.drop(labels=index_remove,axis=1)


rlabel_not_cenir = rlabel_not_cenir.iloc[index_remove,:]
rmriqc_not_cenir = rmriqc_not_cenir.iloc[index_remove,:]

rlabel_not_cenir.to_csv(resdir + 'CATI_allQC_center_sorted_not_cenir.csv')
rmriqc_not_cenir.to_csv(resdir + 'res_mriqc_9118_singu15_not_cenir.csv')


#2020/12 redo csv cati QC4 cenir
df = pd.read_csv('/home/romain.valabregue/datal/QC/CATI/CATI_allQC_center.csv')
dfQC4 = df[df.globalQualitative==4]
aa=dfQC4.mvtWrinkles + dfQC4.mvtBlur + dfQC4.mvtDuplication + dfQC4.artefactCortex + dfQC4.artefactHippo +dfQC4.artefactOther+dfQC4.mvtGhost + dfQC4.backgroundGhost
df100 = dfQC4[aa==0]

df100= pd.DataFrame()
for i in range(0,4):
    dfsel = df[df.globalQualitative== i]
    if len(dfsel)>100:
        i = np.random.randint(0, len(dfsel), 100);
        dfsel = dfsel.iloc[i, :]
    df100 = df100.append(dfsel)

df100.index=range(0,len(df100))

for ii,ff in enumerate(df100.serie_path_proc):
    if ff.find('/network/lustre/dtlake01/opendata/data/ds000030/rrr/')<0:
        print(ff)
        df100.at[ii, 'serie_path_conv']
    new_path=ff[:52] + 'nii/' + ff[52:]
    df100.at[ii,'serie_path_conv']=new_path
    if not os.path.exists(new_path):
        print(new_path);ppp
    dc = gdir(new_path,'cat')
    fms = gfile(dc, '^ms.*nii')
    fs = gfile(dc,'^s.*nii')
    fmask = gfile(dc,'^mask_brain')
    if not (len(fms) == 1) or not (len(fs)==1) or not (len(fmask)==1):
        print('arggg');kkqsdf
    df100.at[ii,'serie_path'] = fms[0]
    df100.at[ii,'dataPathConv'] = fs[0]
    df100.at[ii, 'dataPath_orig'] = fmask[0]

df100 = df100.rename({'serie_path_conv':'serie_path_proc_new', 'serie_path':'volume_ms',
                      'dataPathConv': 'volume_T1', 'dataPath_orig': 'volume_mask_brain'}, axis='columns')
df100 = df100.drop('serie_path_proc',axis='columns')
df100 = df100.rename({'serie_path_proc_new':'serie_path_proc'}, axis='columns')


df100.to_csv('/home/romain.valabregue/datal/QC/CATI/CATI_QCall_100suj.csv')
df100.to_csv('/home/romain.valabregue/datal/QC/CATI/CATI_QC4_100suj.csv')

#csv HCP validation
suj = gdir('/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii','^7')
suj = gdir(suj,['T1w','T1_1mm'])
sujname  = get_parent_path(suj,level=3)[1]
vol_T1 = gfile(suj,'^T1w_1mm.nii.gz')
vol_T1ms = gfile(suj,'^mT')
vol_brain = gfile(suj,'^brain_T1w_1mm.nii.gz')
df = pd.DataFrame({"volume_T1": vol_T1, "volume_mask_brain":vol_brain,"volume_ms":vol_T1ms,  "suj_name":sujname})
df.to_csv('/home/romain.valabregue/datal/QCcnn/CATI_datasets/HCP_val44_suj7.csv')


#again to add affine mni coregistration file
#df100 = pd.read_csv('/home/romain.valabregue/datal/QC/CATI/CATI_QCall_100suj.csv')
df100 = pd.read_csv('/home/romain.valabregue/datal/QCcnn/CATI_datasets/CATI_QCall_100suj.csv')
fT1 = df100.volume_T1
dT1 = [os.path.dirname(ff) for ff in fT1  ]
faff = gfile(dT1,'^aff.*txt')
df100['affine_mni'] = faff
df100.to_csv('/home/romain.valabregue/datal/QCcnn/CATI_datasets/CATI_QCall_100suj.csv')

df = pd.read_csv('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/CATI_datasets/CATI_QCall_100suj.csv')

ff = gfile(df.serie_path_proc.values+'/cat12/','^p1')
df["vol_p1"] = ff
ff = gfile(df.serie_path_proc.values+'/cat12/','^p2')
df["vol_p2"] = ff
ff = gfile(df.serie_path_proc.values+'/cat12/','^p3')
df["vol_p3"] = ff
ff = gfile(df.serie_path_proc.values+'/cat12/','^p4')
df["vol_p4"] = ff
ff = gfile(df.serie_path_proc.values+'/cat12/','^p5')
df["vol_p5"] = ff
df['sujname'] = df.concat + '_' + df.subject_Name_QualiCATI


dfs = df[df.globalQualitative==0]
dfs.to_csv('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/CATI_datasets/CATI_QCall_QC0.csv')
dfs = df[df.globalQualitative==3]
dfs.to_csv('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/CATI_datasets/CATI_QCall_QC3.csv')
