### arg eval PV
import torchio, torch, glob
from pathlib import Path, PosixPath
import nibabel as nb
import pandas as pd
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
    label_list = ['GM', 'WM', 'CSF', 'cereb_GM',  'L_Accu', 'L_Caud', 'L_Pall', 'L_Thal', 'R_Amyg', 'R_Hipp', 'R_Puta', 'BrStem',
     'cereb_WM', 'L_Amyg', 'L_Hipp', 'L_Puta', 'R_Accu', 'R_Caud', 'R_Pall', 'R_Thal', 'skull', 'skin', 'background']
    suj = [torchio.Subject (label=torchio.Image(type = torchio.LABEL, path=[dr + ll + '.nii.gz' for ll in label_list]))]
    dd = torchio.SubjectsDataset(suj)
    ss=dd[0]
    PV = ss['label']['data'] #nb.load(ff).get_fdata()  #sample0['label']['data']

    tbin = PV > 0.001
    volgm = torch.sum(PV[0])
    volgm_vox = torch.sum(tbin[0]).float()
    #volume partiel avec la GM
    vol_nb_pv=torch.zeros_like(tbin[0]).float()
    for ii, vv in enumerate(tbin[1:]) :
        vol_vox = (vv * tbin[0]).sum() / volgm_vox *100
        vol_gm = (vv * PV[0]).sum() / volgm * 100
        #print("{:.2f} % of PV voxel \t {:.2f} % vol with index {}".format(vol_vox, vol_gm, label_list[ii+1]))
        vol_nb_pv += (vv * tbin[0]).float()

    dd = dict(subject=subject, resolution=resolution)
    #get global volume
    for ii, ll in enumerate(label_list):
        dd[ll+'_vol'] = torch.sum(PV[ii]).numpy()

    for ii in range(1,5):
        vol_vox = (vol_nb_pv==ii).sum() / volgm_vox *100
        vol_mm = ((vol_nb_pv==ii).float() * PV[0]).sum() / volgm *100
        print("{:.2f} % \t of PV voxel \t{:.2f} % of PV voxel with {} tissues ".format(vol_vox, vol_mm, ii+1))
        dd['vol_vox_T{}'.format(ii+1)] = vol_vox.numpy()
        dd['vol_mm_T{}'.format(ii+1)] = vol_mm.numpy()

    vol_vox_tot_true = (vol_nb_pv > 0).sum() / volgm_vox *100
    vol_mm_tot_true = ((vol_nb_pv > 0).float() * PV[0]).sum() / volgm *100
    print("{:.2f} % \t of PV voxel \t{:.2f} % of PV voxel in total ".format(vol_vox_tot_true, vol_mm_tot_true))
    dd['vol_vox_tot'], dd['vol_mm_tot'] = vol_vox_tot_true.numpy(), vol_mm_tot_true.numpy()
    df = df.append(dd, ignore_index=True)

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

df.to_csv('/home/romain.valabregue/datal/PVsynth/figure/csv/HCP_suj_all_vol_PV.csv')
df_seuil.to_csv('/home/romain.valabregue/datal/PVsynth/figure/csv/HCP_suj_all_vol_threshold.csv')

sys.exit()

df = pd.read_csv('/home/romain.valabregue/datal/PVsynth/figure/csv/HCP_suj7_vol_PV.csv')
df_seuil = pd.read_csv('/home/romain.valabregue/datal/PVsynth/figure/csv/HCP_suj7_vol_threshold.csv')

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

for k in df.keys():
    if '_vol' in k:
        sns.catplot(data=df, x='subject', y=k, hue='resolution')




#explore PV proportion
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
