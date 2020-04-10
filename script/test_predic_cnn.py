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
    CenterCropOrPad, RandomElasticDeformation, RandomElasticDeformation, CropOrPad, RandomNoise
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL, Interpolation, Subject
from utils_file import get_parent_path, gfile, gdir
from doit_train import do_training, get_motion_transform
from slices_2 import do_figures_from_file
from utils import reduce_name_list, get_ep_iter_from_res_name


#Explore csv results
dqc = ['/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion']
dres = gdir(dqc,'train.*_hcp')
dres = gdir(dqc,'RegMotNew.*train_hcp400_ms.*0001')
dres = gdir(dqc,'RegMotNew.*hcp400_ms.*B4.*L1.*0001')
resname = get_parent_path(dres)[1]
#sresname = [rr[rr.find('hcp400_')+7: rr.find('hcp400_')+17] for rr in resname ]; sresname[2] += 'le-4'
sresname = resname
commonstr, sresname = reduce_name_list(sresname)
print('common str {}'.format(commonstr))

target='ssim'; target_scale=1
#target='random_noise'; target_scale=10

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
        errorT = np.hstack( [ np.abs(rr.model_out.values - rr.loc[:,target].values*target_scale) for rr in resT] )
        ite_tottt = np.hstack([0, ite_tot])
        LmTrain = [ np.mean(errorT[ite_tottt[ii]:ite_tottt[ii+1]]) for ii in range(0,len(ite_tot)) ]

    LmVal = [np.mean(np.abs(rr.model_out-rr.loc[:,target].values*target_scale)) for rr in resV]
    plt.figure('meanL11'); legend_str.append('V{}'.format(sresname[ii]));
    if is_train: legend_str.append('T{}'.format(sresname[ii]))
    plt.plot(ite_tot, LmVal,'--',color=col[ii])
    if is_train: plt.plot(ite_tot, LmTrain,color=col[ii])

plt.legend(legend_str); plt.grid()

#for ii, oneres in enumerate([dres[0]]):
for ii, oneres in enumerate(dres):
    fres=gfile(oneres,'res_val')
    #fres = gfile(oneres,'train.*csv')
    if len(fres)>0:
        nb_fig = len(fres)//10 +1
        for nbf in range(0, nb_fig):
            for ff in fres[nbf*10:(nbf+1)*10]:
                #print(get_parent_path(ff)[1])
                res = pd.read_csv(ff)
                err = np.abs(res.loc[:,target]*target_scale-res.model_out) #same as L1 loss
                plt.figure('F{}_err'.format(nbf) + sresname[ii]); plt.plot(err)
                errcum = np.cumsum(err[50:-1])/range(1,len(err)-51+1)
                plt.figure(sresname[ii] + '2err_cum');        plt.plot(errcum)
                N = 50
                err_slidin = np.convolve(err, np.ones((N,))/N, mode='valid')
                #plt.figure('F{}_err_slide'.format(nbf) + sresname[ii]); plt.plot(err_slidin)
                plt.figure('F{}_model_out'.format(nbf) + sresname[ii]); plt.scatter(res.model_out, res.loc[:,target]*target_scale)

            legend_str = [str(ii+1) for ii in range(nbf*10, (nbf+1)*10)]
            plt.legend(legend_str)
            plt.scatter(res.loc[:, target]*target_scale, res.loc[:, target]*target_scale, c='k'); plt.grid()




ind_sel = (res.model_out < 0.7) & (res.ssim > 0.9)
ff=res.fpath[ind_sel].values
sujn = get_parent_path(ff,2)[1]
sujn_all = get_parent_path(res.fpath,2)[1]
print('{} suj affect over {}'.format(len(np.unique(sujn)),len(np.unique(sujn_all))))

plt.scatter(res.model_out[ind_sel], res.ssim[ind_sel])

d = '/home/romain.valabregue/datal/QCcnn/figure/bad_train'
fref = None
l_view = [("sag", "vox", 0.4), ("cor", "vox", 0.6), ("ax", "vox", 0.5), ]
display_order = np.array([1, 3])  # row and column of the montage
mask_info = [("whole", 1)]  # min max within the mask

#l_in = uf.concatenate_list([fin, fmask, faff])
l_in = ff
fig = do_figures_from_file(l_in, slices_infos=l_view, mask_info=mask_info, display_order=display_order,
                           fref=fref, out_dir=d, plot_single=True, montage_shape=(5,3), plt_ioff=True)





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

suj = [Subject(image=Image('/data/romain/HCPdata/suj_150423/mT1w_1mm.nii', INTENSITY),
         maskk=Image('/data/romain/HCPdata/suj_150423/mask_brain.nii',  LABEL))]

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
t = Compose([RandomNoise(std=(0.020,0.2)) ])


dataset = ImagesDataset(suj, transform=t)
s = dataset[0]
ov(s['image']['data'][0])

for i in range(1,50):
    s=dataset[0]
    dataset.save_sample(s, dict(image='/home/romain/QCcnn/random_motion/random{:.2}.nii'.format(100*s['random_noise'])))

t = dataset.get_transform()
type(t)
isinstance(t, Compose)

