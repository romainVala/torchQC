from segmentation.losses.dice_loss import Dice
import matplotlib.pyplot as plt, pandas as pd, seaborn as sns
pd.set_option('display.max_columns', None)
import torchio as tio
import torch, glob
from pathlib import Path, PosixPath
from nibabel.viewers import OrthoSlicer3D as ov

import copy

dice = Dice(); L1 = torch.nn.L1Loss()
t = tio.RandomAffine(default_pad_value='minimum')
t = tio.RandomAffine(default_pad_value=0)
resample_to=True
#t = tio.RandomElasticDeformation()
#t = tio.RandomBlur() #not a good idea since intensity transform won't change the label
#def apply_inverse_transform(subject, list_transfo_index = 0):
#    trsfm_hist, seeds_hist = tio.compose_from_history(history=subject.history)
#    trsfm_hist[list_transfo_index].get_inverse = True
#    return trsfm_hist[0](subject, seed=seeds_hist[list_transfo_index])
def apply_inverse_transform(subject, list_transfo_index = -1):
    trsfm_hist  = subject.history[list_transfo_index].inverse()
    return trsfm_hist(subject)


dr = '/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/727553/T1w/ROI_PVE_07mm/'
dr = '/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/727553/T1w/ROI_PVE_1mm/'
#dr = '/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/700634/T1w/ROI_PVE_1mm/'
#dr = '/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/727553/T1w/ROI_PVE_14mm/'
#dr = '/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/727553/T1w/ROI_PVE_28mm/'
dref = '/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/727553/T1w/ROI_PVE_14mm/'
dref = '/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/727553/T1w/ROI_PVE_1mm/'
dref = '/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/727553/T1w/ROI_PVE_28mm/'
label_list = ['GM', 'WM', 'CSF', 'cereb_GM',  'L_Accu', 'L_Caud', 'L_Pall', 'L_Thal', 'R_Amyg', 'R_Hipp', 'R_Puta', 'BrStem',
 'cereb_WM', 'L_Amyg', 'L_Hipp', 'L_Puta', 'R_Accu', 'R_Caud', 'R_Pall', 'R_Thal', 'skull', 'skin', 'background']

label_list = ["backNOpv", "GM", "WM", "CSF", "both_R_Accu", "both_R_Amyg", "both_R_Caud", "both_R_Hipp", "both_R_Pall", "both_R_Puta", "both_R_Thal"]
suj = tio.Subject(label=tio.ScalarImage(path=dr + '/GM.nii.gz'))
dr_list = glob.glob('/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/727553/T1w/ROI_PVE_*')
interpolation_list = ['linear', 'nearest', 'bspline'] # 'label_gaussian' not good as it binarize the inputs
save_vol=False

tcrop = tio.Crop((28, 28, 28, 28, 28, 28));
tcrop07 = tio.Crop((40, 40, 40, 40, 40, 40));
tcrop14 = tio.Crop((20, 20, 20, 20, 20, 20));
tcrop28 = tio.Crop((10, 10, 10, 10, 10, 10))

df = pd.DataFrame()

for k in range(50):
    taff = tio.RandomAffine()
    sujtmp = taff(suj)
    taff = sujtmp.history[0] #so we get the same affine
    for dr in dr_list:
        subject = Path(dr).parent.parent.name
        resolution = Path(dr).name
        res = 1.4 if '14mm' in resolution else 2.8 if '28mm' in resolution else 0.7 if '07mm' in resolution else 1
        voxel_volume = res*res*res
        print("Suj {} {}".format(subject,resolution))
        #true volumes
        sujdic = dict(subject=subject, resolution=resolution, res=res)

        suj = tio.Subject(label=tio.ScalarImage( path=dr + '/GM.nii.gz'))
        suj_orig_crop = tcrop07(suj) if res<0.9 else tcrop(suj) if res<1.1 else tcrop14(suj) if res < 2 else tcrop28(suj)
        vol_orig = suj_orig_crop.label.data.sum().numpy()

        if save_vol:
            suj.label.save(f'GM_{res}_orig.nii')

        for interp in interpolation_list:
            taff.image_interpolation = interp # tio.RandomAffine(image_interpolation=interp)
            suj_aff = taff(suj)
            suj_back = apply_inverse_transform(suj_aff)
            suj_back =  tcrop07(suj_back) if res<0.9 else  tcrop(suj_back) if res<1.1 else tcrop14(suj_back) if res < 2 else tcrop28(suj_back)

            vol_back = suj_back.label.data.sum().numpy()
            sujdic['vol_ration'] = vol_back/vol_orig * 100
            sujdic['interp'] = interp
            sujdic['dice'] = float(dice.dice_loss(suj_orig_crop.label.data, suj_back.label.data).numpy())*100
            sujdic['L1'] = float(L1(suj_orig_crop.label.data, suj_back.label.data).numpy())

            #same index
            ind_mask =  suj_orig_crop.label.data>0.0001
            data_orig = suj_orig_crop.label.data[ind_mask].unsqueeze(0)
            data_back = suj_back.label.data[ind_mask].unsqueeze(0)

            vol_orig = data_orig.sum().numpy()
            vol_back = data_back.sum().numpy()
            sujdic['vol_ration_mask'] = vol_back/vol_orig * 100
            sujdic['dice_mask'] = float(dice.dice_loss(data_orig, data_back).numpy())*100
            sujdic['L1_mask'] = float(L1(data_orig, data_back).numpy())

            df = df.append(sujdic, ignore_index=True)

            if save_vol:
                suj_aff.label.save(f'GM_{res}_aff_{interp}.nii')
                suj_back.label.save(f'GM_{res}_back_{interp}.nii')


sns.catplot(data=df, x='interp', y='dice', kind='box', hue='res'); plt.grid()







#first try ...
if resample_to:
    suj_from = suj
    pv_from = suj_from.label.data
    pv_img_from = tio.ScalarImage(tensor=pv_from, affine=suj_from.label.affine)
    label_volume_from = pv_from.argmax(dim=0, keepdim=True)
    label_from = tio.LabelMap(tensor=label_volume_from, affine=suj_from.label.affine)
    one_hot_from = torch.nn.functional.one_hot(label_volume_from[0].long()).permute(3, 0, 1, 2)
    one_hot_img_from = tio.ScalarImage(tensor=one_hot_from, affine=suj_from.label.affine)

    suj = tio.Subject(label=tio.Image(type=tio.LABEL, path=[dref + ll + '.nii.gz' for ll in label_list]))
    t = tio.Resample(target=str(suj.label.path[1]), image_interpolation=tio.Interpolation.LINEAR)

pv = suj.label.data
pv_img = tio.ScalarImage(tensor=pv, affine=suj.label.affine)
label_volume = pv.argmax(dim=0, keepdim=True)
label = tio.LabelMap(tensor=label_volume, affine=suj.label.affine)
one_hot = torch.nn.functional.one_hot(label_volume[0].long()).permute(3, 0, 1, 2)
one_hot_img = tio.ScalarImage(tensor=one_hot, affine=label.affine)

if resample_to:
    new = tio.Subject(pv=pv_img_from, label=label_from, one_hot=one_hot_img_from)
else:
    new = tio.Subject(pv=pv_img, label=label, one_hot=one_hot_img)
ref = label_volume[0]

df = pd.DataFrame()

subject = Path(dr).parent.parent.name
resolution = Path(dref).name if resample_to else Path(dr).name
res = 1.4 if '14mm' in resolution else 2.8 if '28mm' in resolution else 0.7 if '07mm' in resolution else 1
voxel_volume = res*res*res
print("Suj {} {}".format(subject,resolution))
#true volumes
sujdic = dict(subject=subject, resolution=resolution, res=res)
for i, LabName in enumerate(label_list):
    sujdic['vol_' + LabName] = float(pv[i].sum()) * voxel_volume / 1000
    sujdic['volbin_' + LabName] = float(one_hot[i].sum()) * voxel_volume /1000 / sujdic['vol_' + LabName]


max_range = 1 if resample_to else 20
for i in range(0, max_range):
    transformed = t(new)
    if not resample_to:
        transformed = apply_inverse_transform(transformed)

    #from_one_hot = transformed.one_hot.data.argmax(dim=0)  #warning if equal every where (0) -> label 10
    t_pv = copy.deepcopy(transformed.pv.data)
    t_onehot = copy.deepcopy(transformed.one_hot.data)
    t_label = copy.deepcopy(transformed.label.data[0])

    #corect border missing voxel due to padding (as much as possible)
    #but it make it worth ...
    # border = t_onehot.sum(dim=0) <0.5 #== 0 not enough at the border interpolation create voxel values in [0 1] but taking < 1 will include other point in the brain
    # t_onehot[0][border] = 1
    # t_pv[0][border] = 1 #background
    # t_label[border] = 0
    # for i in range(1,t_onehot.shape[0]):
    #     t_onehot[i][border] = 0
    #     t_pv[i][border] = 0

    from_one_hot = t_onehot.argmax(dim=0)

    dic = dict()
    dic['dice_lab'] = float(dice.dice_loss(ref, t_label).numpy())*100
    dic['dice_hot'] = float(dice.dice_loss(ref, from_one_hot).numpy() )*100
    dic['dice_pv'] =  float(dice.dice_loss(pv, transformed.pv.data).numpy() )*100
    #for j in range(0,t_onehot.shape[0]): #[0, 1, 2, 3, 4,  10]: #
    for j, LabName in enumerate(label_list):
        dic['dice_lab_' + LabName] = float(dice.dice_loss(ref == j, t_label == j).numpy())*100
        dic['dice_hot_' + LabName] = float(dice.dice_loss(ref == j, from_one_hot == j).numpy())*100
        dic['dice_pv_' + LabName]  = float(dice.dice_loss(pv[j], t_pv[j]).numpy())*100
        dic['vol_tlab_' + LabName] = float((t_label == j).sum()) * voxel_volume / 1000 / sujdic['vol_' + LabName]
        dic['vol_thot_' + LabName] = float((from_one_hot == j).sum()) * voxel_volume / 1000 / sujdic['vol_' + LabName]
        dic['vol_tpv_' + LabName]  = float(t_pv[j].sum()) * voxel_volume / 1000 / sujdic['vol_' + LabName]

    dic.update(sujdic)
    df = df.append(dic, ignore_index=True)

def print_df_mean(df, tags=[''], plot_dice=True, plot_vol=True):
    for tag in tags:
        key_list = ['dice_lab', 'dice_hot', 'dice_pv', 'vol_tlab', 'vol_thot', 'vol_tpv' ]
        key_list = [ s+'_'+tag for s in key_list]
        a = df.loc[:, key_list].describe([])

        if plot_dice:
            print('\t{} {:.2} \t{} {:.2} \t{} {:.2} '.format(
                a.keys()[0], a.iloc[1, 0], a.keys()[1], a.iloc[1, 1], a.keys()[2],a.iloc[1, 2]))
        if plot_vol:
            print('\t{} {:.4} \t{} {:.4} \t{} {:.4}'.format(
                a.keys()[3], a.iloc[1, 3], a.keys()[4], a.iloc[1, 4], a.keys()[5], a.iloc[1, 5]))

print_df_mean(df, tags=['backNOpv', 'GM', 'WM','CSF','both_R_Accu','both_R_Thal','backNOpv'],plot_vol=False)
print_df_mean(df, tags=['backNOpv', 'GM', 'WM','CSF','both_R_Accu','both_R_Thal','backNOpv'], plot_dice=False)


"""
tt = t_pv
volume=tt.permute(1,2,3,0).numpy()
v= nib.Nifti1Image(volume,affine)
nib.save(v,'/tmp/t.nii')
"""
"""
100 tirage seed 12 (20 is enough

RES 2.8
dice_lab 1.6 	dice_hot 1.8 	dice_pv 0.96
dice_lab_GM 3.2 	dice_hot_GM 4.9 	dice_pv_GM 4.3
dice_lab_WM 2.6 	dice_hot_WM 4.2 	dice_pv_WM 3.1
dice_lab_CSF 4.0 	dice_hot_CSF 5.2 	dice_pv_CSF 7.1
dice_lab_both_R_Accu 4.8 	dice_hot_both_R_Accu 6.0 	dice_pv_both_R_Accu 6.7
dice_lab_both_R_Thal 2.0 	dice_hot_both_R_Thal 0.65 	dice_pv_both_R_Thal 1.8

RES 1.4
dice_lab 0.96 	dice_hot 0.71 	dice_pv 0.5
dice_lab_GM 2.1 	dice_hot_GM 1.9 	dice_pv_GM 2.4
dice_lab_WM 1.5 	dice_hot_WM 0.92 	dice_pv_WM 1.4
dice_lab_CSF 2.5 	dice_hot_CSF 2.5 	dice_pv_CSF 3.1
dice_lab_both_R_Accu 2.6 	dice_hot_both_R_Accu 0.46 	dice_pv_both_R_Accu 2.4
dice_lab_both_R_Thal 1.0 	dice_hot_both_R_Thal 0.21 	dice_pv_both_R_Thal 0.78


RES 1mm

dice_lab 0.71 	dice_hot 0.38 	dice_pv 0.35
dice_lab_GM 1.6 	dice_hot_GM 1.0 	dice_pv_GM 1.5
dice_lab_WM 1.1 	dice_hot_WM 0.38 	dice_pv_WM 0.92
dice_lab_CSF 1.9 	dice_hot_CSF 1.4 	dice_pv_CSF 2.1
dice_lab_both_R_Accu 1.8 	dice_hot_both_R_Accu 0.18 	dice_pv_both_R_Accu 1.5
dice_lab_both_R_Thal 0.75 	dice_hot_both_R_Thal 0.14 	dice_pv_both_R_Thal 0.57

Res 07mm
0dice_lab 0.51 	dice_hot 0.19 	dice_pv 0.28
dice_lab_GM 1.2 	dice_hot_GM 0.49 	dice_pv_GM 0.97
dice_lab_WM 0.8 	dice_hot_WM 0.16 	dice_pv_WM 0.58
dice_lab_CSF 1.4 	dice_hot_CSF 0.72 	dice_pv_CSF 1.8
dice_lab_both_R_Accu 1.3 	dice_hot_both_R_Accu 0.072 	dice_pv_both_R_Accu 0.91
dice_lab_both_R_Thal 0.52 	dice_hot_both_R_Thal 0.066 	dice_pv_both_R_Thal 0.37

"""
#to reproduce, strange bug
import torchio as tio
import copy
sub = tio.datasets.Colin27()
sub.pop('brain'); sub.pop('head')
t = tio.RandomBlur(std=5)
#t = tio.RandomAffine()

label_volume = sub.t1.data > 1000000 #pv.argmax(dim=0, keepdim=True)

label = tio.LabelMap(tensor=label_volume, affine=sub.t1.affine)
#label = tio.ScalarImage(tensor=label_volume, affine=sub.t1.affine)

new_sub = tio.Subject(t1=label)
#new_sub.plot()

tsub = t(new_sub)

dd=copy.deepcopy(tsub.t1.data[0])
dd[:] = 0

new_sub.plot()


import torchio as tio

sub = tio.datasets.Colin27()
sub.pop('brain'); sub.pop('head')

t = tio.Compose([tio.Pad(padding=10, padding_mode="reflect"), tio.Crop(bounds_parameters=10)])
new_sub = t(sub)
new_sub.t1.affine
sub.t1.affine

#test inverse elastic
import torchio as tio
import SimpleITK as sitk

colin = tio.datasets.Colin27()
transform = tio.RandomElasticDeformation()
transformed = transform(colin)

trsfm_hist, seeds_hist = tio.compose_from_history(history=transformed.history)
trsfm_hist[0].get_inverse = True
colin_back = trsfm_hist[0](transformed, seed=seeds_hist[0])

transformed.t1.save('/tmp/tcolin.nii')
colin_back.t1.save('/tmp/t_back_colin.nii')
colin.t1.save('/tmp/colin.nii')

grid = -transformed.history[0][1]['coarse_grid']
