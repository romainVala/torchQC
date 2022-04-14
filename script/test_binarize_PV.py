### arg eval PV
import torchio, torch, glob
from pathlib import Path, PosixPath
import nibabel as nb
import pandas as pd, seaborn as sns, numpy as np
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D as ov
import glob
import sys

dr = '/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/727553/T1w/ROI_PVE_1mm/'
dr = '/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/165941/T1w/ROI_PVE_1mm/' #min Acc PV
dr = '/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/256540/T1w/ROI_PVE_1mm/' #max Acc PV

dres = glob.glob('/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/7*/T1w/ROI_PVE*')
df, df_seuil = pd.DataFrame(),  pd.DataFrame()
seuil = 0.1 #0.001
for iii, dr in enumerate(dres):
    subject = Path(dr).parent.parent.name
    resolution = Path(dr).name
    print("Suj {} {} {:.2f}".format(subject,resolution, iii/len(dres)))
    dr += '/'
    label_list = ['GM', 'WM', 'CSF',  'L_Accu', 'L_Caud', 'L_Pall', 'L_Thal', 'L_Amyg', 'L_Hipp', 'L_Puta',
                  'R_Amyg', 'R_Hipp', 'R_Puta',  'R_Accu', 'R_Caud', 'R_Pall', 'R_Thal', 'BrStem', 'cereb_GM',
                 'cereb_WM',  'skull', 'skin', 'background']
    suj = [torchio.Subject (label=torchio.Image(type = torchio.LABEL, path=[dr + ll + '.nii.gz' for ll in label_list]))]
    PV = suj[0].label.data
    #dd = torchio.SubjectsDataset(suj);     ss=dd[0];     PV = ss['label']['data'] #nb.load(ff).get_fdata()  #sample0['label']['data']

    pseudo_PV = np.minimum(PV.numpy(), (1 - PV.numpy()))

    tbin = PV > seuil
    #PV[~tbin] = 0  #no need to trunck the PV values
    res = 1.4 if '14mm' in resolution else 2.8 if '28mm' in resolution else 0.7 if '07mm' in resolution else 1
    voxel_volume = res * res * res

    dd = dict(subject=subject, resolution=resolution)
    # get global volume
    for ii, ll in enumerate(label_list):
        dd[ll + '_vol'] = torch.sum(PV[ii]).numpy() * voxel_volume / 1000

    for label_index in range(0,10):
        #print('label {}'.format(label_list[label_index]) )
        data = PV[label_index].numpy()
        vol_label = np.sum(data)
        volPureGM = np.sum((data >= (1-seuil) ))
        dd['PV_in_{}_Tot2'.format(label_list[label_index])]  = (100 - volPureGM / vol_label * 100)

        dd['PV_in_{}_Tot3'.format(label_list[label_index])]  = ( np.sum(pseudo_PV[label_index]) )/ vol_label * 100

        #volgm = dd[label_list[label_index] + '_vol'] / voxel_volume * 1000 #torch.sum(PV[label_index])
        #volgm_vox = torch.sum(tbin[label_index]).float()
        #volume partiel avec la GM
        vol_nb_pv=torch.zeros_like(tbin[0]).float()
        #print(f'Patrial volume of {label_list[label_index]} vol is {volgm}')
        for ii, vv in enumerate(tbin) :
            if ii==label_index:
                continue
            #vol_vox = (vv * tbin[label_index]).sum() / volgm_vox *100
            # to get which label has pv with gm
            vol_gm = (vv * tbin[label_index] * PV[label_index]).sum() / vol_label * 100
            #if vol_gm>0.01:
                #print("{:.2f} % of PV voxel \t {:.2f} % vol with index {}".format(vol_vox, vol_gm, label_list[ii]))
            vol_nb_pv += (vv * tbin[label_index]).float()


        for ii in range(1,3): #4
            #vol_vox = (vol_nb_pv==ii).sum() / volgm_vox *100
            vol_mm = ((vol_nb_pv==ii).float() * PV[label_index]).sum() / vol_label *100
            vol_mm_pseudo = ((vol_nb_pv==ii).float() * pseudo_PV[label_index]).sum() / vol_label *100
            #print("{:.2f} % \t of PV voxel \t{:.2f} % of PV volume with {} tissues ".format(vol_vox, vol_mm, ii+1))
            #dd['vol_vox_T{}'.format(ii+1)] = vol_vox.numpy()
            dd['PV_in_{}_T{}'.format(label_list[label_index], ii+1)] = vol_mm.numpy()
            dd['pseudoPV_in_{}_T{}'.format(label_list[label_index], ii+1)] = vol_mm_pseudo.numpy()

        #vol_vox_tot_true = (vol_nb_pv > 0).sum() / volgm_vox *100
        vol_mm_tot_true = ((vol_nb_pv > 0).float() * PV[label_index]).sum() / vol_label *100
        #print("{:.2f} % \t of PV voxel \t{:.2f} % of PV volume in total ".format(vol_vox_tot_true, vol_mm_tot_true))
        #dd['vol_vox_tot']  = vol_vox_tot_true.numpy()
        dd['PV_in_{}_Tot'.format(label_list[label_index])] = vol_mm_tot_true.numpy()


    df = df.append(dd, ignore_index=True)

df.to_csv('/home/romain.valabregue/datal/PVsynth/figure/csv/HCP_suj_all_vol_PV_new2.csv')

    #volume thershold dependence
    seuil  = [0.001, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.999]
    dict_seuil = dict(subject=subject, resolution=resolution)
    for ss in seuil:
        dict_seuil['seuil'] = ss
        dict_seuil['vol_gm'] = (torch.sum(PV[0] >= ss) / volgm * 100).numpy()
        df_seuil = df_seuil.append(dict_seuil, ignore_index=True)
        if ss==0.5: print(dict_seuil['vol_gm'])
    #volume as argmax
    val, indices = torch.max(PV, dim=0)
    tensor = torch.nn.functional.one_hot(indices)
    tensor = tensor.permute( 3, 0, 1, 2).float()

    dict_seuil['seuil'] = 0.51
    dict_seuil['vol_gm'] = (torch.sum(tensor[0]) / volgm * 100).numpy()
    df_seuil = df_seuil.append(dict_seuil, ignore_index=True)
    print(dict_seuil['vol_gm'])

df.to_csv('/home/romain.valabregue/datal/PVsynth/figure/csv/HCP_suj_all_vol_PV_new.csv')
df_seuil.to_csv('/home/romain.valabregue/datal/PVsynth/figure/csv/HCP_suj_all_vol_threshold.csv')

sys.exit()

df = pd.read_csv('/home/romain.valabregue/datal/PVsynth/figure/csv/HCP_suj7_vol_PV.csv')
df = pd.read_csv('/home/romain.valabregue/datal/PVsynth/figure/csv/HCP_suj_all_vol_PV.csv')
df = pd.read_csv('/home/romain.valabregue/datal/PVsynth/figure/csv/HCP_suj_all_vol_PV_new.csv')
df = pd.read_csv('/data/romain/PVsynth/figure_SegPV/csv/HCP_suj_all_vol_PV_new.csv')

df_seuil = pd.read_csv('/home/romain.valabregue/datal/PVsynth/figure/csv/HCP_suj7_vol_threshold.csv')
df_seuil = pd.read_csv('/home/romain.valabregue/datal/PVsynth/figure/csv/HCP_suj_all_vol_threshold.csv')

def to_num(s):
    s = eval(s)
    return s.numpy()
def get_epoch(s):
    s_list = s.split('_')
    res = int(s_list[-1][:-2])
    if res != 1:
        res/=10
    return res

df['res'] = df['resolution'].apply(lambda s: get_epoch(s))

for k in df.keys():
    if '_vol' in k:
        print(k)
        df[k] = df[k].apply(lambda s: to_num(s))
        df[k] = df[k] * df['res']* df['res']* df['res']

dfsub = df[df.subject.astype(str).str.startswith('7')] #select suj starting
dfsub_PV_rel = dfsub.copy()


PV_keys=[]
for k in df.keys():
    if 'PV' in k:
        PV_keys.append(k)
        print(k)

#Pandas power ! means by group
dfsub_PV_rel[PV_keys] = dfsub_PV_rel[PV_keys].div(dfsub_PV_rel[PV_keys].groupby(dfsub_PV_rel.res).transform('mean'))
label_list = ['GM', 'WM', 'CSF', 'L_Accu',  'L_Amyg', 'L_Caud', 'L_Hipp', 'L_Pall', 'L_Puta', 'L_Thal']
hue_order = ['ROI_PVE_1mm', 'ROI_PVE_14mm', 'ROI_PVE_07mm']
for lab in label_list:
    col = f'PV_in_{lab}_Tot'
    sns.catplot(data=dfsub_PV_rel, x='subject', y=col, hue='resolution', hue_order=hue_order); plt.grid()

df1 = df[df['resolution']=='ROI_PVE_1mm']
dd= df1.sort_values(by='GM_vol')

sns.catplot(data=df, x='resolution', y='PV_in_GM_T2')

sns.catplot(data=dfsub, x='subject', y='GM_vol', hue='resolution') #pourquoi CSF vol a 1mm est different des autres ???
sns.catplot(data=df, x='subject', y='WM_vol', hue='resolution')
df['GM_ratio'] = df['GM_vol'] / (df['WM_vol'] + df['CSF_vol'])
df['GM_ratio'] = df['GM_vol'] / df['WM_vol']
sns.catplot(data=df, x='subject', y='GM_ratio', hue='resolution')
sns.catplot(data=df_seuil, x='seuil', y='vol_gm', hue='subject', col='resolution')
sns.catplot(data=df_seuil, x='seuil', y='vol_gm', col='resolution', col_order=hue_order)
sns.catplot(data=df_seuil, x='seuil', y='vol_gm', hue='resolution', hue_order=hue_order,kind='strip', dodge=True)
sns.catplot(data=df, x='resolution', y='GM_vol')
for k in df.keys():
    if '_vol' in k:
        sns.catplot(data=df, x='subject', y=k, hue='resolution')

#volume relative to 0.7 mm res, arg missing resolution for some subject
df['GM_vol_ref'] = 0
suj_num = np.unique(df['subject'].values)
for ns in suj_num:
    ind_suj = df['subject'] == ns
    ind_res = 'ROI_PVE_07mm' == df[ind_suj]['resolution'].values
    if not any(ind_res):
        print('suj {} has missing 07mm'.format(ns))
    else:
        #df.loc[ind_suj,'GM_vol_ref'] = df[ind_suj][ind_res]['GM_vol'] #.values #
        df.loc[ind_suj,'GM_vol_ref'] = np.tile(df[ind_suj][ind_res]['GM_vol'].values, len(ind_res))

df['GM_vol_rel'] = df['GM_vol'] / df['GM_vol_ref']*100
res = df['resolution'].unique()

g= sns.catplot(data=df_seuil[df_seuil['seuil']==0.5], x='resolution', y='vol_gm', order = res[[2,0,1,3]])
plt.grid(); ax = g.axes.flat[0]; ax.set_xticklabels(['0.7 mm', '1 mm', '1.4 mm', '2.8 mm'])
plt.ylabel('GM volume relatif to the PV estimate ')
plt.title('GM volume estimate from Binary mask ', y=0.9, backgroundcolor='w')

g= sns.catplot(data=df, x='resolution', y='GM_vol_rel', order = res[[2,0,1,3]])
plt.grid(); plt.ylim([99.6, 101])
ax = g.axes.flat[0]; ax.set_xticklabels(['0.7 mm', '1 mm', '1.4 mm', '2.8 mm'])
plt.ylabel('GM volume relatif to 07mm ')
plt.title('GM volume estimate from PV map', y=0.9, backgroundcolor='w')

#plot abs volumes over columns
sns.catplot(data=df, x='resolution', y='GM_vol')
sel_keys =[ 'L_Accu_vol', 'L_Amyg_vol', 'L_Caud_vol', 'L_Hipp_vol',  'L_Pall_vol', 'L_Puta_vol',]
sel_keys =[ 'GM_vol', 'CSF_vol',  'WM_vol',  'L_Accu_vol', 'L_Amyg_vol', 'L_Caud_vol', 'L_Hipp_vol', 'L_Pall_vol',
            'L_Puta_vol', 'L_Thal_vol', 'BrStem_vol','cereb_GM_vol', 'cereb_WM_vol',  'skin_vol', 'skull_vol','background_vol']
sel_val = ['subject', 'resolution']

dfrel = df
for k in sel_keys:
    dfrel[k] = dfrel[k] / dfrel[k].mean()

dfm = dfrel.melt(id_vars=sel_val, value_vars=sel_keys, var_name='structure', value_name='Volume')
g=sns.catplot(data=dfm[dfm['resolution']=='ROI_PVE_07mm'], x='structure', y='Volume')
ax = g.axes.flat[0];
xx = [x.replace('_vol','') for x in sel_keys]
ax.set_xticklabels(xx)
plt.grid()

sel_keys=[k for k in df.keys() if (k.find('T2')>0)]
sel_keys = ['PV_in_GM_T2', 'PV_in_CSF_T2', 'PV_in_WM_T2', 'PV_in_L_Accu_T2', 'PV_in_L_Amyg_T2', 'PV_in_L_Caud_T2', 'PV_in_L_Hipp_T2',
 'PV_in_L_Pall_T2', 'PV_in_L_Puta_T2', 'PV_in_L_Thal_T2']
sel_keys = ['PV_in_GM_T3', 'PV_in_CSF_T3', 'PV_in_WM_T3', 'PV_in_L_Accu_T3', 'PV_in_L_Amyg_T3', 'PV_in_L_Caud_T3', 'PV_in_L_Hipp_T3',
 'PV_in_L_Pall_T3', 'PV_in_L_Puta_T3', 'PV_in_L_Thal_T3']
sel_keys = ['PV_in_GM_Tot3', 'PV_in_CSF_Tot3', 'PV_in_WM_Tot3', 'PV_in_L_Accu_Tot3', 'PV_in_L_Amyg_Tot3', 'PV_in_L_Caud_Tot3', 'PV_in_L_Hipp_Tot3',
 'PV_in_L_Pall_Tot3', 'PV_in_L_Puta_Tot3', 'PV_in_L_Thal_Tot3']
hue_order = ['ROI_PVE_07mm', 'ROI_PVE_1mm', 'ROI_PVE_14mm', 'ROI_PVE_28mm']

sel_val = ['subject', 'resolution']
dfm = df.melt(id_vars=sel_val, value_vars=sel_keys, var_name='structure', value_name='Volume')
#g=sns.catplot(data=dfm[dfm['resolution']=='ROI_PVE_07mm'], x='structure', y='Volume')
g=sns.catplot(data=dfm, x='structure', y='Volume', hue='resolution', hue_order=hue_order,kind='strip', dodge=True)
xx = [x.replace('PV_in_','') for x in sel_keys]; xx = [x.replace('_Tot3','') for x in xx]
g.axes.flat[0].set_xticklabels(xx)
plt.ylabel(' tissue Partial Volume relatif to structure volume')
plt.grid()

labels = [item.get_text() for item in ax.get_xticklabels()]
ddf = pd.DataFrame()
for ii, theres in enumerate([0.7, 1, 1.4, 2.8]):
    dfsub = df[df.res==theres]
    label_list = ['GM', 'WM', 'CSF', 'L_Accu',  'L_Amyg', 'L_Caud', 'L_Hipp', 'L_Pall', 'L_Puta', 'L_Thal']
    corr_labP, corr_labS , corr_labK = [], [], []
    for lab in label_list:
        col2 = f'{lab}_vol'
        col1 = f'PV_in_{lab}_Tot3'
        corr_labP.append(dfsub[col1].corr(dfsub[col2], method='spearman')) # method='spearman'
        corr_labS.append(dfsub[col1].corr(dfsub[col2], method='spearman')) # method='spearman'
        corr_labK.append(dfsub[col1].corr(dfsub[col2], method='kendall'))
        dddict = dict({'label': lab,'corre': corr_labP[-1], 'res': theres})

        ddf = ddf.append(dddict,ignore_index=True) #.loc[ii, lab] =  corr_labP[-1]

    sns.catplot(data=ddf,x='label', y='corre' , hue='res', kind='boxen')

    plt.figure(); plt.plot(label_list,corr_labP,'x'); plt.plot(label_list,corr_labS,'x'); #plt.plot(label_list,corr_labK,'x')
    plt.grid(); plt.ylabel('Correlation total volume - total Partial Volume'); plt.legend(['pearson','spearman']) ; #, 'kendall'])
    plt.title(f'resolution = {dfsub.res.unique()}')

plt.figure();sns.scatterplot(data=df, x='CSF_vol', y = 'PV_in_CSF_Tot', hue='resolution')
plt.figure();sns.scatterplot(data=df, x='L_Caud_vol', y = 'PV_in_L_Caud_Tot', hue='resolution')
plt.figure();sns.scatterplot(data=df, x='L_Thal_vol', y = 'PV_in_L_Thal_Tot', hue='resolution')
plt.figure();sns.regplot(data=dfsub, x='CSF_vol', y = 'PV_in_CSF_Tot')
plt.figure();sns.regplot(data=dfsub, x='L_Caud_vol', y = 'PV_in_L_Caud_Tot')
sns.lmplot(data=df, x='GM_vol', y = 'PV_in_GM_Tot3', hue='res')

#explore PV proportion first try
fpv1 = gfile('/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/513130/T1w/ROI_PVE_1mm','^GM')
fpv14 = gfile('/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/513130/T1w/ROI_PVE_14mm','^GM')
fpv28 = gfile('/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/513130/T1w/ROI_PVE_28mm','^GM')
seuils = [ [0, 1], [0.001, 0.999], [0.01, 0.99], [0.05, 0.95], [0.1, 0.9] ]
seuil = [0, 1]
for ff, nn, res in zip((fpv1 + fpv14 + fpv28), ['1mm', '1.4mm', '2.8mm'], [1, 1.4, 2.8]):
    if res==1.4:
        continue
    legendstr = []
    for seuil in seuils:
        print(seuil)
        img = nb.load(ff)
        data = img.get_fdata(dtype=np.float32)
        fig = plt.figure(nn)
        data_cut = data[ (data>seuil[0]) * (data<seuil[1])]
        nGM = np.sum(data>=seuil[1])
        nPV = len(data_cut)
        pseudo_PV = np.minimum(data, (1-data))
        volPV = np.sum(data)*res*res*res
        volPseudoPV = np.sum(pseudo_PV)*res*res*res

        volPVbin = np.sum(data>=0.5)*res*res*res

        volPureGM = np.sum((data>=seuil[1])) *res*res*res
        volPV_cut = np.sum( data[ (data>seuil[0]) ])*res*res*res #almost equal to volPV
        volPurePV = volPV - volPureGM
        volPurePVcut = np.sum(data_cut)*res*res*res

        print('{} : PV vox {:.2f}  %PV vol {:.2f}   %PVcut vol {:.2f}  mm bin {:.2f} pvmm {:.2f} pseudo {:.2f} %'.format(
            nn, nPV/(nGM+nPV)*100, (volPurePV/volPV*100), (volPurePVcut/(volPurePVcut+volPureGM)*100), volPVbin, volPV, volPseudoPV/volPV * 100))

        hh = plt.hist(data_cut, bins=500, linewidth=0)
        #legendstr.append(f'GM voxel:{100 - nGM/nPV*100:.2f}% volume:{(100 - volPureGM/volPV*100):.2f}% for PV in {seuil} ')
        legendstr.append(
        f'PV GM voxel: {nPV/(nGM+nPV)*100:.0f}% volume: {(100-volPureGM/volPV*100):.0f}% for PV in {seuil} ')
    plt.legend(legendstr)


    #calcule PV conj(PV) donne les structure en contact ... ?

    pv_lin = PV.flatten(start_dim=1)
    pv_inter = torch.matmul(pv_lin,pv_lin.transpose(1,0))
    pv_inter[0]
    for i, l in enumerate(label_list):
        print('{} vol {} {}'.format(l, torch.sum(PV[i]), torch.sum(pv_inter[i])))
        inter_vol = pv_inter[i] / torch.sum(pv_inter[i]) * 100
        inter_vol[inter_vol < 0.1] = 0
        print(inter_vol)
