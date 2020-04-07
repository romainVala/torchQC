from utils import print_accuracy, print_accuracy_df, print_accuracy_all
from utils_file import gfile, gdir, get_parent_path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D as ov
from nilearn import plotting
import seaborn as sns

import numpy as np
import sys, os, logging

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose

from torchio.data.io import write_image, read_image
from torchio.transforms import RandomMotionFromTimeCourse, RandomAffine, \
    CenterCropOrPad, RandomElasticDeformation, RandomElasticDeformation, CropOrPad
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL, Interpolation, Subject
from utils_file import get_parent_path, gfile, gdir
from doit_train import do_training, get_motion_transform



def get_ep_iter_from_res_name(resname, nbit, remove_ext=-7, batch_size=4):
    ffn = [ff[ff.find('_ep') + 3:remove_ext] for ff in resname]
    key_list = []
    for fff, fffn in zip(ffn, resname):
        if '_it' in fff:
            ind = fff.find('_it')
            ep = int(fff[0:ind])
            it = int(fff[ind + 3:])*batch_size
            it = 4 if it==0 else it #hack to avoit 2 identical point (as val is done for it 0 and las of previous ep
        else:
            ep = int(fff)
            it = nbit
        key_list.append([fffn, ep, it])
    aa = np.array(sorted(key_list, key=lambda x: (x[1], x[2])))
    name_sorted, ep_sorted, it_sorted = aa[:, 0], aa[:, 1], aa[:, 2]
    ep_sorted = np.array([int(ee) for ee in ep_sorted])
    it_sorted = np.array([int(ee) for ee in it_sorted])
    return name_sorted, ep_sorted, it_sorted


#Explore csv results
dqc = ['/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion']
dres = gdir(dqc,'nw0.*0001')
dres = gdir(dqc,'RegMotNew.*0001')
resname = get_parent_path(dres)[1]
#sresname = [rr[rr.find('hcp400_')+7: rr.find('hcp400_')+17] for rr in resname ]; sresname[2] += 'le-4'
sresname = resname

#for ii, oneres in enumerate([dres[2]]):
for ii, oneres in enumerate(dres):
    fres=gfile(oneres,'res_val')
    #fres = gfile(oneres,'train.*csv')

    for ff in fres:
        res=pd.read_csv(ff)
        err = np.abs(res.ssim-res.model_out) #same as L1 loss
        plt.figure(sresname[ii] + '22err'); plt.plot(err)
        errcum = np.cumsum(err[50:-1])/range(1,len(err)-51+1)
        #plt.figure(sresname[ii] + '2err_cum');        plt.plot(errcum)
        N=50
        err_slidin = np.convolve(err, np.ones((N,))/N, mode='valid')
        plt.figure(sresname[ii] + '22err_slide'); plt.plot(err_slidin)
        plt.figure(sresname[ii] + '22model_out'); plt.scatter(res.model_out, res.ssim)

    legend_str = [str(ii+1) for ii in range(0,len(fres))]
    plt.legend(legend_str)
    plt.scatter(res.ssim, res.ssim)

legend_str=[]
col = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for ii, oneres in enumerate(dres):
    fresV=gfile(oneres,'res_val')
    fresT = gfile(oneres,'train.*csv')
    is_train = False if len(fresT)==0 else True
    if is_train: resT = [pd.read_csv(ff) for ff in fresT]
    resV = [pd.read_csv(ff) for ff in fresV]

    resname = get_parent_path(fresV)[1]
    nbite = len(resT[0]) if is_train else 80000
    a, b, c = get_ep_iter_from_res_name(resname, nbite)
    ite_tot = c+b*nbite
    if is_train:
        errorT = np.hstack( [ np.abs(rr.model_out.values - rr.ssim.values) for rr in resT] )
        ite_tottt = np.hstack([0, ite_tot])
        LmTrain = [ np.mean(errorT[ite_tottt[ii]:ite_tottt[ii+1]]) for ii in range(0,len(ite_tot)) ]

    LmVal = [np.mean(np.abs(rr.model_out-rr.ssim)) for rr in resV]
    if is_train: plt.figure(sresname[ii] + 'meanL1')
    plt.figure('meanL1'); legend_str.append('V{}'.format(sresname[ii]));
    if is_train: legend_str.append('T{}'.format(sresname[ii]))
    plt.plot(ite_tot, LmVal,'--',color=col[ii])
    if is_train: plt.plot(ite_tot, LmTrain,color=col[ii])

plt.legend(legend_str); plt.grid()


model = nn.Linear(10, 2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
steps = 10
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, last_epoch=-1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1)
for epoch in range(5):
    for idx in range(steps):
        scheduler.step()
        print(scheduler.get_lr())

    #print('Reset scheduler')
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)



for dd in td:
    print('shape {} path {}'.format(dd['image']['data'].shape,dd['image']['path']))

td = doit.train_dataloader
data = next(iter(td))


tensor = data['image']['data'][0].squeeze(0)  # remove channels dim
affine = data['image']['affine'].squeeze(0)

write_image(tensor, affine, '/tmp/toto.nii')
mvt = pd.read_csv('/home/romain/QCcnn/motion_cati_brain_ms/ssim_0.03557806462049484_sample00010_suj_cat12_brain_s_S02_Sag_MPRAGE_mvt.csv',header=None)
fpars = np.asarray(mvt)

transforms = get_motion_transform()

suj = [[ Image('T1', '/home/romain/QCcnn/motion_cati_brain_ms/brain_s_S02_Sag_MPRAGE.nii.gz', 'intensity'), ]]

dataset = ImagesDataset(suj, transform=transforms)
s=dataset[0]

ov(s['T1']['data'][0])
tt = dataset.get_transform()
plt.figure(); plt.plot(tt.fitpars.T)
dataset.save_sample(s, dict(T1='/home/romain/QCcnn/motion_cati_brain_ms/toto10.nii'))



#look at distribution of metric on simulate motion
dir_cache = '/network/lustre/dtlake01/opendata/data/ds000030/rrr/CNN_cache'
dd = gdir(dir_cache,'mask_mv')
fr = gfile(dd,'resul')
name_res = get_parent_path(dd)[1]
res = [ pd.read_csv(ff) for ff in fr]
sell_col = res[0].keys()

sell_col = [ 'L1', 'MSE', 'corr', 'mean_DispP', 'rmse_Disp','rmse_DispTF',
             'ssim', 'ssim_all', 'ssim_brain', 'ssim_p1', 'ssim_p2']
sell_col = [ 'L1', 'MSE', 'corr', 'ssim', 'ssim_all', 'ssim_brain', 'ssim_p1', 'ssim_p2']
sell_col = [ 'L1', 'MSE', 'corr', 'ssim', 'ssim_all']

for rr in res:
    rr=rr.loc[:,sell_col]
    sns.pairplot(rr)


#test MOTION CATI

ss = [ Image('T1', '/home/romain/QCcnn/mask_mvt_val_cati_T1/s_S07_3DT1.nii.gz', INTENSITY),
         Image('T3', '/home/romain/QCcnn/mask_mvt_val_cati_T1/s_S07_3DT1.nii.gz', INTENSITY),]

suj = [[ Image('T1', '/home/romain/QCcnn/mask_mvt_val_cati_T1/s_S07_3DT1_float.nii.gz', INTENSITY), ]]
suj = [[Image('T1', '/data/romain/HCPdata/suj_150423/T1w_1mm.nii.gz', INTENSITY), ]]

ss = [Subject(Image('T1', '/home/romain/QCcnn/mask_mvt_val_cati_T1/s_S07_3DT1.nii.gz', INTENSITY) ) ]

suj = [Subject(ss) for ss in suj]

dico_params = {"maxDisp": (1, 4), "maxRot": (1, 4), "noiseBasePars": (5, 20, 0.8),
               "swallowFrequency": (2, 6, 0.5), "swallowMagnitude": (3, 4),
               "suddenFrequency": (2, 6, 0.5), "suddenMagnitude": (3, 4),
               "verbose": False, "keep_original": True, "proba_to_augment": 1,
               "preserve_center_pct": 0.1, "keep_original": True, "compare_to_original": True,
               "oversampling_pct": 0, "correct_motion": True}

fipar = pd.read_csv('/home/romain/QCcnn/mask_mvt_val_cati_T1/ssim_0.6956839561462402_sample00220_suj_cat12_s_S07_3DT1_mvt.csv', header=None)
dico_params['fitpars'] = fipar.values
t = RandomMotionFromTimeCourse(**dico_params)

dataset = ImagesDataset(suj, transform=Compose((CenterCropOrPad(target_shape=(182, 218,182)),t)))
dataset = ImagesDataset(suj, transform=Compose((CenterCropOrPad(target_shape=(176, 240, 256)),t)))
dataset = ImagesDataset(suj, transform=Compose((CenterCropOrPad(target_shape=(182, 218, 256)),t)))
dataset = ImagesDataset(suj, transform=Compose((t,)))
s=dataset[0]
dataset.save_sample(s, dict(T1='/home/romain/QCcnn//mask_mvt_val_cati_T1/mot_float.nii'))
sample = torch.load('/home/romain/QCcnn/mask_mvt_val_cati_T1/sample00220_sample.pt')
tensor = sample['image']['data'][0]  # remove channels dim
affine = sample['image']['affine']
write_image(tensor, affine, '/home/romain/QCcnn//mask_mvt_val_cati_T1/mot_li.nii')

ff1 = t.fitpars_interp

suj = [[ Image('image', '/home/romain/QCcnn/mask_mvt_val_cati_T1/s_S07_3DT1_float.nii.gz', INTENSITY),
         Image('maskk', '/home/romain/QCcnn/mask_mvt_val_cati_T1/niw_Mean_brain_mask5k.nii.gz',  LABEL),]]

tc = CenterCropOrPad(target_shape=(182, 218,212))
tc = CropOrPad(target_shape=(182, 218,182), mode='mask',mask_key='maskk')
#dico_elast = {'num_control_points': 6, 'deformation_std': (30, 30, 30), 'max_displacement': (4, 4, 4),
#              'proportion_to_augment': 1, 'image_interpolation': Interpolation.LINEAR}
#tc = RandomElasticDeformation(**dico_elast)

dico_p = {'num_control_points': 8, 'deformation_std': (20, 20, 20), 'max_displacement': (4, 4, 4),
              'proportion_to_augment': 1, 'image_interpolation': Interpolation.LINEAR}
dico_p = { 'num_control_points': 6,
           #'max_displacement': (20, 20, 20),
           'max_displacement': (30, 30, 30),
           'proportion_to_augment': 1, 'image_interpolation': Interpolation.LINEAR }

t = Compose([ RandomElasticDeformation(**dico_p), tc])

dataset = ImagesDataset(suj, transform=t)
s = dataset[0]

for i in range(1,10):
    s=dataset[0]
    dataset.save_sample(s, dict(image='/home/romain/QCcnn//mask_mvt_val_cati_T1/elastic8_30{}.nii'.format(i)))

t = dataset.get_transform()
type(t)
isinstance(t, Compose)

