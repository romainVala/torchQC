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
dres = glob.glob('/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/*/T1w/ROI_PVE*')
df, df_seuil = pd.DataFrame(),  pd.DataFrame()
for dr in dres:
    subject = Path(dr).parent.parent.name
    resolution = Path(dr).name
    print("Suj {} {}".format(subject,resolution))
    dr += '/'
    label_list = ['GM', 'WM', 'CSF',  'L_Accu', 'L_Caud', 'L_Pall', 'L_Thal', 'L_Amyg', 'L_Hipp', 'L_Puta',
                  'R_Amyg', 'R_Hipp', 'R_Puta',  'R_Accu', 'R_Caud', 'R_Pall', 'R_Thal', 'BrStem', 'cereb_GM',
                 'cereb_WM',  'skull', 'skin', 'background']
    suj = [torchio.Subject (label=torchio.Image(type = torchio.LABEL, path=[dr + ll + '.nii.gz' for ll in label_list]))]
    PV = suj[0].label.data
    #dd = torchio.SubjectsDataset(suj);     ss=dd[0];     PV = ss['label']['data'] #nb.load(ff).get_fdata()  #sample0['label']['data']

    tbin = PV > 0.001
    PV[~tbin] = 0
    res = 1.4 if '14mm' in resolution else 2.8 if '28mm' in resolution else 0.7 if '07mm' in resolution else 1
    voxel_volume = res * res * res

    dd = dict(subject=subject, resolution=resolution)
    # get global volume
    for ii, ll in enumerate(label_list):
        dd[ll + '_vol'] = torch.sum(PV[ii]).numpy() * voxel_volume / 1000

    for label_index in range(0,10):
        #print('label {}'.format(label_list[label_index]) )

        volgm = dd[label_list[label_index] + '_vol'] / voxel_volume * 1000 #torch.sum(PV[label_index])
        #volgm_vox = torch.sum(tbin[0]).float()
        #volume partiel avec la GM
        vol_nb_pv=torch.zeros_like(tbin[0]).float()
        for ii, vv in enumerate(tbin) :
            if ii==label_index:
                continue
            #vol_vox = (vv * tbin[0]).sum() / volgm_vox *100
            # to get which label has pv with gm
            #vol_gm = (vv * tbin[label_index] * PV[label_index]).sum() / volgm * 100
            #print("{:.2f} % of PV voxel \t {:.2f} % vol with index {}".format(vol_vox, vol_gm, label_list[ii]))
            vol_nb_pv += (vv * tbin[label_index]).float()


        for ii in range(1,4):
            #vol_vox = (vol_nb_pv==ii).sum() / volgm_vox *100
            vol_mm = ((vol_nb_pv==ii).float() * PV[label_index]).sum() / volgm *100
            #print("{:.2f} % \t of PV voxel \t{:.2f} % of PV volume with {} tissues ".format(vol_vox, vol_mm, ii+1))
            #dd['vol_vox_T{}'.format(ii+1)] = vol_vox.numpy()
            dd['PV_in_{}_T{}'.format(label_list[label_index], ii+1)] = vol_mm.numpy()

        #vol_vox_tot_true = (vol_nb_pv > 0).sum() / volgm_vox *100
        vol_mm_tot_true = ((vol_nb_pv > 0).float() * PV[label_index]).sum() / volgm *100
        #print("{:.2f} % \t of PV voxel \t{:.2f} % of PV volume in total ".format(vol_vox_tot_true, vol_mm_tot_true))
        #dd['vol_vox_tot']  = vol_vox_tot_true.numpy()
        dd['PV_in_{}_Tot'.format(label_list[label_index])] = vol_mm_tot_true.numpy()

    df = df.append(dd, ignore_index=True)
df.to_csv('/home/romain.valabregue/datal/PVsynth/figure/csv/HCP_suj_all_vol_PV_new.csv')

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
        #df[k] = df[k].apply(lambda s: to_num(s))
        df[k] = df[k] * df['res']* df['res']* df['res']


sns.catplot(data=df, x='subject', y='vol_mm_T2', hue='resolution')
sns.catplot(data=df, x='subject', y='vol_mm_T3', hue='resolution')
sns.catplot(data=df, x='subject', y='vol_mm_tot', hue='resolution')
sns.catplot(data=df, x='subject', y='vol_vox_tot', hue='resolution')
sns.catplot(data=df, x='subject', y='GM_vol', hue='resolution')
sns.catplot(data=df, x='subject', y='CSF_vol', hue='resolution') #pourquoi CSF vol a 1mm est different des autres ???
sns.catplot(data=df, x='subject', y='WM_vol', hue='resolution')
df['GM_ratio'] = df['GM_vol'] / (df['WM_vol'] + df['CSF_vol'])
sns.catplot(data=df, x='subject', y='GM_ratio', hue='resolution')
sns.catplot(data=df_seuil, x='seuil', y='vol_gm', hue='subject', col='resolution')
sns.catplot(data=df_seuil, x='seuil', y='vol_gm', col='resolution')
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
        df.loc[ind_suj,'GM_vol_ref'] = df[ind_suj][ind_res]['GM_vol'].values #np.tile(df[ind_suj][ind_res]['GM_vol'].values, len(ind_res))

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

sel_keys =[ 'GM_vol', 'CSF_vol',  'WM_vol',  'L_Accu_vol', 'L_Amyg_vol', 'L_Caud_vol', 'L_Hipp_vol', 'L_Pall_vol', 'L_Puta_vol',
        'BrStem_vol','cereb_GM_vol', 'cereb_WM_vol',  'skin_vol', 'skull_vol','background_vol']
sel_val = ['subject', 'resolution']

dfrel = df
for k in sel_keys:
    dfrel[k] = dfrel[k] / dfrel[k].mean()

dfm = dfrel.melt(id_vars=sel_val, value_vars=sel_keys, var_name='structure', value_name='Volume')
sns.catplot(data=dfm[dfm['resolution']=='ROI_PVE_07mm'], x='structure', y='Volume')

ax.yaxis

for ax in g.axes.flat:
    ax.yaxis.
labels = [item.get_text() for item in ax.get_xticklabels()]





#explore PV proportion first try
fpv1 = gfile('/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/513130/T1w/ROI_PVE_1mm','^GM')
fpv14 = gfile('/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/513130/T1w/ROI_PVE_14mm','^GM')
seuils = [ [0, 1], [0.01, 0.99], [0.05, 0.95], [0.1, 0.9] ]
seuil = [0, 1]
for seuil in seuils:
    print(seuil)
    for ff, nn, res in zip( (fpv1 + fpv14), ['1mm', '1.4mm'], [1,1.4] ) :
        img = nb.load(ff)
        data = img.get_fdata(dtype=np.float32)
        fig = plt.figure(nn)
        dd = data[ (data>seuil[0]) * (data<seuil[1])]
        nGM = np.sum(data>=seuil[1])
        nPV = len(dd)
        volGM = np.sum(data>=0.5)
        volPV = np.sum(data)
        print('{} : pure GM {} PV {}  PV/GM {}  Vol GM vox {}  mm {} pvmm{} '.format(nn, nGM, nPV, nPV/nGM, volGM,
                                                                              volGM*res*res*res, volPV*res*res*res ))
        hh = plt.hist(dd, bins=500)

#calcule PV conj(PV) donne les structure en contact ... ?

    pv_lin = PV.flatten(start_dim=1)
    pv_inter = torch.matmul(pv_lin,pv_lin.transpose(1,0))
    pv_inter[0]
    for i, l in enumerate(label_list):
        print('{} vol {} {}'.format(l, torch.sum(PV[i]), torch.sum(pv_inter[i])))
        inter_vol = pv_inter[i] / torch.sum(pv_inter[i]) * 100
        inter_vol[inter_vol < 0.1] = 0
        print(inter_vol)
