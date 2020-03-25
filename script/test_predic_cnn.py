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
from torchio.transforms import RandomMotionFromTimeCourse
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL
from utils_file import get_parent_path, gfile, gdir
from doit_train import do_training, get_motion_transform


d='/home/romain.valabregue/QCcnn/li'
fres = gfile(d,'csv')
ff = fres[-1]
sujall = []
for ff in fres[4:-1] :
    res = pd.read_csv(ff)
    res['diff'] = res.ssim - res.model_out
    res = res.sort_values('diff', ascending=False)

    sujn = get_parent_path(res.fpath[1:10].values,2)[1]
    sujall.append(sujn)

ss=np.hstack(sujall)
len(ss)
len(np.unique(ss))


f=gfile('/home/romain/QCcnn/li/','.*csv')
for ff in f:
    res=pd.read_csv(ff)
    #plt.scatter(res.ssim,res.model_out)
    #plt.plot(res.ssim, res.ssim,'k+')
    err = np.abs(res.ssim-res.model_out) #same as L1 loss
    plt.figure('err'); plt.plot(err)

    errcum = np.cumsum(err[50:-1])/range(1,len(err)-51+1)
    plt.figure('err_cum');        plt.plot(errcum)
    N=50
    err_slidin = np.convolve(err, np.ones((N,))/N, mode='valid')

    plt.figure('err_slide'); plt.plot(err_slidin)
plt.legend(('1','2','3','4','5'))

import torch.nn as nn
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


from torchio.data.io import write_image, read_image
from torchio.transforms import RandomMotionFromTimeCourse, RandomAffine, CenterCropOrPad
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL
from torchvision.transforms import Compose
from nibabel.viewers import OrthoSlicer3D as ov
from torchio.transforms.metrics import ssim3D

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
