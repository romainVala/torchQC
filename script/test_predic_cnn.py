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
    CenterCropOrPad, RandomElasticDeformation, CropOrPad, RandomNoise, ApplyMask
from torchio.transforms.augmentation.spatial.random_affine_fft import RandomAffineFFT

from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL, Interpolation, Subject
from utils_file import get_parent_path, gfile, gdir
from doit_train import do_training, get_motion_transform
from slices_2 import do_figures_from_file
from utils import reduce_name_list, get_ep_iter_from_res_name, remove_extension, remove_string_from_name_list


def get_pandadf_from_res_valOn_csv(dres, resname, csv_regex='res_valOn', data_name_list=None,
                                   select_last=None, target='ssim', target_scale=1):

    if len(dres) != len(resname) : raise('length problem between dres and resname')

    resdf_list = []
    for oneres, resn in zip(dres, resname):
        fres_valOn = gfile(oneres, csv_regex)
        print('Found {} <{}> for {} '.format(len(fres_valOn), csv_regex, resn))
        if len(fres_valOn) == 0:
            continue

        ftrain = gfile(oneres, 'res_train_ep01.csv')
        rrt = pd.read_csv(ftrain[0])
        nb_it = rrt.shape[0];

        resdir, resname_val = get_parent_path(fres_valOn)
        resname_sorted, b, c = get_ep_iter_from_res_name(resname_val, 0)

        if select_last is not None:
            if select_last<0:
                resname_sorted = resname_sorted[select_last:]
            else:
                nb_iter = b*nb_it+c
                resname_sorted = resname_sorted[np.argwhere(nb_iter > select_last)[1:8]]

        resV = [pd.read_csv(resdir[0] + '/' + ff) for ff in resname_sorted]
        resdf = pd.DataFrame()
        for ii, fres in enumerate(resname_sorted):
            iind = [i for i, s in enumerate(data_name_list) if s in fres]
            if len(iind) ==1: #!= 1: raise ("bad size do not find which sample")
                data_name = data_name_list[iind[0]]
            else:
                data_name = 'res_valds'

            iind = fres.find(data_name)
            ddn = remove_extension(fres[iind + len(data_name) + 1:])
            new_col_name = 'Mout_' + ddn
            iind = ddn.find('model_ep')
            if iind==0:
                transfo='raw'
            else:
                transfo = ddn[:iind - 1]

            if transfo[0] == '_': #if start with _ no legend ... !
                transfo = transfo[1:]

            model_name = ddn[iind:]
            aa, bb, cc = get_ep_iter_from_res_name([fres], nb_it)
            nb_iter = bb[0] * nb_it + cc[0]

            rr = resV[ii].copy()
            rr['evalOn'], rr['transfo'] = data_name, transfo
            rr['model_name'], rr['submodel_name'], rr['nb_iter'] = resn, model_name, str(nb_iter)
            rr[target] = rr[target] * target_scale
            resdf = pd.concat([resdf, rr], axis=0, sort=True)

        resdf['error'] = resdf[target] - resdf['model_out']
        resdf['error_abs'] = np.abs(resdf[target] - resdf['model_out'])
        resdf_list.append(resdf)

    return resdf_list


def plot_resdf(resdf_list, dir_fig=None,  target='ssim', split_distrib=True):

    for resdf in resdf_list :
        ee = np.unique(resdf.evalOn)
        resn = resdf['model_name'].values[0]
        zz = np.unique(resdf['model_name'])
        if len(zz)>1: raise('multiple model_name')

        if dir_fig is not None:
            dir_out_sub = dir_fig + '/' + resn +'/'
            if not os.path.isdir(dir_out_sub): os.mkdir(dir_out_sub)

        for eee in ee:
            dfsub = resdf.loc[resdf.evalOn == eee, :]
            #dfsub.transfo = dfsub.transfo.astype(str)
            fign = 'MOD_' + resn + '_ON_' + eee

            fig = plt.figure('Dist' + fign)
            #ax = sns.violinplot(x="transfo", y="error", hue="model_name", data=dfsub, palette="muted")
            ax = sns.violinplot(x="transfo", y="error", hue="transfo", data=dfsub, palette="muted")
            if split_distrib :
                nbline = int(dfsub.shape[0] / 2)
                plt.subplot(211);
                ax = sns.violinplot(x="nb_iter", y="error", hue="transfo", data=dfsub.iloc[:nbline, :], palette="muted")
                plt.grid()
                plt.subplot(212)
                ax = sns.violinplot(x="nb_iter", y="error", hue="transfo", data=dfsub.iloc[nbline:, :], palette="muted")
                plt.grid()
                ax.legend().set_visible(False);
                fig.set_size_inches([18, 6]); fig.tight_layout();   fig.suptitle(fign);

            else:
                ax = sns.violinplot(x="nb_iter", y="error", hue="transfo", data=dfsub, palette="muted")

            if dir_fig is not None:
                plt.savefig(dir_out_sub + 'Dist_' + fign + '.png');
                plt.close()

            g = sns.catplot(x="nb_iter", y="error_abs", hue="transfo", data=dfsub, palette="muted", kind="point",
                            dodge=True, legend_out=False)
            g.fig.suptitle('Error Abs' + fign)
            g.fig.set_size_inches([12, 5]);
            g.fig.tight_layout();
            if dir_fig is not None:
                plt.savefig(dir_out_sub + 'L1_' + fign + '.png');
                plt.close()

            sns.despine(offset=10, trim=True);

            g = sns.relplot(x=target, y="model_out", hue="nb_iter", data=dfsub,
                            palette=sns.color_palette("hls", dfsub.nb_iter.nunique()),
                            kind='scatter', col='transfo', col_wrap=3, alpha=0.5)
            axes = g.axes.flatten()
            for aa in axes:
                #aa.plot([0.5, 1], [0.5, 1], 'k')
                aa.plot([0.2, 2.2], [0.2, 2.2], 'k')
                plt.grid()

            g.fig.suptitle(fign, x=0.8, y=0.1)
            if dir_fig is not None:
                plt.savefig(dir_out_sub + 'Scat_' + fign + '.png');
                plt.close()

#res_valOn
dd = gfile('/network/lustre/dtlake01/opendata/data/ds000030/rrr/CNN_cache_new','_')
dir_fig = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion/figure/motion_regress/eval2/'
dir_fig = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_random_noise/figure2/'
data_name_list = get_parent_path(dd)[1]

dres_reg_exp, figname = ['Reg.*D0_DC0' ], ['noise' ]
dres_reg_exp, figname = ['.*hcp.*ms', '.*hcp.*T1', 'cati.*ms', 'cati.*T1'], ['hcp_ms', 'hcp_T1', 'cati_ms', 'cati_T1']
sns.set(style="whitegrid")

csv_regex='res_valOn_'
for rr, fign in zip(dres_reg_exp, figname):
    dres = gdir(dqc, rr)
    resname = get_parent_path(dres)[1]
    resname = remove_string_from_name_list(resname, ['RegMotNew_', 'Size182_ConvN_C16_256_Lin40_50_','_B4', '_Loss_L1_lr0.0001', '_nw0_D0'])
    resname = [fign+'_'+ zz for zz in resname]
    print('For {} found {} dir'.format(fign,len(resname)));

    if 0==1:
        for oneres, resn in zip(dres, resname):
            fres_valOn = gfile(oneres, csv_regex)
            print('Found {} <{}> for {} '.format(len(fres_valOn), csv_regex, resn))
            if len(fres_valOn) == 0:
                continue

    #resdf_list = get_pandadf_from_res_valOn_csv(dres, resname, csv_regex=csv_regex, data_name_list=data_name_list, select_last=None)
    #plot_resdf(resdf_list,  dir_fig=dir_fig)

    resdf_list = get_pandadf_from_res_valOn_csv(dres, resname, csv_regex='res_valOn_', data_name_list=data_name_list,
                                           select_last=None, target='random_noise', target_scale=10)
    plot_resdf(resdf_list, dir_fig=dir_fig, target='random_noise', split_distrib=False)




#Explore csv results
dqc = ['/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_random_noise']
dqc = ['/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion']

dres_reg_exp, figname = ['Reg.*', 'fff']
dres_reg_exp, figname = ['.*hcp.*ms', '.*hcp.*T1', 'cati.*ms', 'cati.*T1'], ['hcp_ms', 'hcp_T1', 'cati_ms', 'cati_T1']
dres_reg_exp, figname = [ 'hcp.*ms', 'hcp.*T1'] , ['hcp_ms', 'hcp_T1'] #[ 'hcp.*T1'], ['hcp_T1']
dres_reg_exp, figname = [ 'cati.*ms', 'cati.*T1'], ['cati_ms', 'cati_T1']
target = 'ssim'; target_scale = 1
#target = 'random_noise'; target_scale = 10

for rr, fign in zip(dres_reg_exp, figname):
    dres = gdir(dqc, rr)
    resname = get_parent_path(dres)[1]
    print(len(resname)); print(resname)

    #sresname = [rr[rr.find('hcp400_')+7: rr.find('hcp400_')+17] for rr in resname ]; sresname[2] += 'le-4'
    sresname = resname
    commonstr, sresname = reduce_name_list(sresname)
    print('common str {}'.format(commonstr))

    legend_str=[]
    col = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for ii, oneres in enumerate(dres):
        fresT = gfile(oneres,'res_train.*csv')
        fresV=gfile(oneres,'res_val_')

        is_train = False if len(fresT)==0 else True
        if is_train: resT = [pd.read_csv(ff) for ff in fresT]

        resdir, resname = get_parent_path(fresV)
        nbite = len(resT[0]) if is_train else 80000
        fresV_sorted, b, c = get_ep_iter_from_res_name(resname, nbite)
        resV = [pd.read_csv(resdir[0] + '/' + ff) for ff in fresV_sorted]

        ite_tot = c+b*nbite
        if is_train:
            errorT = np.hstack( [ np.abs(rr.model_out.values - rr.loc[:,target].values*target_scale) for rr in resT] )
            ite_tottt = np.hstack([0, ite_tot])
            LmTrain = [ np.mean(errorT[ite_tottt[ii]:ite_tottt[ii+1]]) for ii in range(0,len(ite_tot)) ]

        LmVal = [np.mean(np.abs(rr.model_out-rr.loc[:,target].values*target_scale)) for rr in resV]
        plt.figure('MeanL1_'+fign); legend_str.append('V{}'.format(sresname[ii]));
        if is_train: legend_str.append('T{}'.format(sresname[ii]))
        plt.plot(ite_tot, LmVal,'--',color=col[ii])
        if is_train: plt.plot(ite_tot, LmTrain,color=col[ii], linewidth=6)

    plt.legend(legend_str); plt.grid()
    ff=plt.gcf();ff.set_size_inches([15, 7]); #ff.tight_layout()
    plt.subplots_adjust(left=0.05, right=1, bottom=0.05, top=1, wspace=0, hspace=0)


for ii, oneres in enumerate([dres[0]]):
#for ii, oneres in enumerate(dres):
    fres=gfile(oneres,'res_val')
    resdir, resname = get_parent_path(fres)
    fresV_sorted, b, c = get_ep_iter_from_res_name(resname, nbite)
    fres = [rr+'/'+ff for rr,ff in zip(resdir, fresV_sorted)]
    #fres = gfile(oneres,'train.*csv')
    if len(fres)>0:
        nb_fig = len(fres)//10 +1
        for nbf in range(0, nb_fig):
            for ff in fres[nbf*10:(nbf+1)*10]:
                #print(get_parent_path(ff)[1])
                res = pd.read_csv(ff)
                err = np.abs(res.loc[:,target]*target_scale-res.model_out) #same as L1 loss
                #plt.figure('F{}_err'.format(nbf) + sresname[ii]); plt.plot(err)
                errcum = np.cumsum(err[50:-1])/range(1,len(err)-51+1)
                plt.figure(sresname[ii] + '2err_cum');        plt.plot(errcum)
                N = 50
                err_slidin = np.convolve(err, np.ones((N,))/N, mode='valid')
                #plt.figure('F{}_err_slide'.format(nbf) + sresname[ii]); plt.plot(err_slidin)
                figname = 'F{}_model_out'.format(nbf) + sresname[ii]
                plt.figure(figname); plt.scatter(res.model_out, res.loc[:,target]*target_scale)
                print('fig {} {}'.format(figname,get_parent_path(ff)[1] ))
                plt.xlabel('model prediction'); plt.ylabel('ssimm simulated motion / original image')

            legend_str = [str(ii+1) for ii in range(nbf*10, (nbf+1)*10)]
            plt.legend(legend_str)
            plt.scatter(res.loc[:, target]*target_scale, res.loc[:, target]*target_scale, c='k'); plt.grid()


#plot scatter distance
dir_fig = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion/figure/motion_regress/'
for rr, fign in zip(dres_reg_exp, figname):
    dres = gdir(dqc, rr)
    resname = get_parent_path(dres)[1]

    oneres  = dres[0]
    fresT = gfile(oneres,'res_train.*csv')
    resT = pd.read_csv(fresT[0])
    sell_col = ['L1', 'MSE', 'corr', 'ssim_brain', 'ssim_all']

    rr = resT.loc[:, sell_col]
    g=sns.pairplot(rr, corner=True, diag_kind='kde')
    g.fig.suptitle(fign)
    g.fig.set_size_inches([14, 9]); g.fig.tight_layout()
    g.savefig(dir_fig+'scatter_distance' + fign + '.png')


#results eval
prefix = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/'
# CATI label
# rlabel  = pd.read_csv(prefix+'/CATI_datasets/all_cati_mriqc_pred.csv')
rlabel = pd.read_csv(prefix + '/CATI_datasets/all_cati.csv')
rlabel.index = get_parent_path( rlabel.cenir_QC_path.values, -3)[1]
rlabel = rlabel.sort_index()

pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1, 'display.width', 400)

dres = gdir(dqc,('R.*','eva'))
sresname = get_parent_path(dres)[1]
fres = gfile(dres,'res')

resl=[]
for ff in fres:
    res = pd.read_csv(ff, header=None)
    res_filename = res.loc[:,1]
    res.index = get_parent_path(get_parent_path(res_filename,2)[0],level=-3)[1]
    res = res.sort_index()  # alphabetic order
    if len(set(res.index).difference(set(rlabel.index))) >0 : print('ERROR missing lines')
    ind_sel = res.loc[:, 2] < 10
    res = res.loc[ind_sel, :]
    resl.append(res)

plt.scatter(resl[0].loc[:,2], resl[1].loc[:,2])
res.loc[res.loc[:,2]>10,1]

rlabel = rlabel.loc[ind_sel,:]
isel = rlabel.globalQualitative==0

for ii,res in enumerate(resl):
    plt.figure('hist' + sresname[ii])
    plt.hist(res.loc[:,2], bins=100)
    plt.hist(res.loc[isel,2])
    plt.figure('Pred' + sresname[ii])
    plt.scatter(res.loc[:,2],rlabel.globalQualitative); plt.grid()
    plt.xlabel('model predictions'); plt.ylabel('note QC du CATI')






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
         maskk=Image('/data/romain/HCPdata/suj_150423/brain_T1w_1mm.nii.gz',  LABEL))]

dico_p = {'num_control_points': 8, 'deformation_std': (20, 20, 20), 'max_displacement': (4, 4, 4),
              'p': 1, 'image_interpolation': Interpolation.LINEAR}
dico_p = { 'num_control_points': 6,
           #'max_displacement': (20, 20, 20),
           'max_displacement': (30, 30, 30),
           'p': 1, 'image_interpolation': Interpolation.LINEAR }

t = Compose([ RandomElasticDeformation(**dico_p), tc])
t = Compose([RandomNoise(std=(0.020,0.2)),  RandomElasticDeformation(**dico_p) ])
t = Compose([RandomNoise(),  RandomElasticDeformation() ])
tc = ApplyMask(masking_method='maskk')
t = RandomAffine(scales=(0.8,0.8), degrees=(0,0) )

tc = CenterCropOrPad(target_shape=(182, 218,212))
tc = CropOrPad(target_shape=(182, 218,182), mask_name='maskk')
#dico_elast = {'num_control_points': 6, 'deformation_std': (30, 30, 30), 'max_displacement': (4, 4, 4),
#              'proportion_to_augment': 1, 'image_interpolation': Interpolation.LINEAR}
#tc = RandomElasticDeformation(**dico_elast)

t = Compose([RandomNoise(std=(0.05, 0.051)), RandomAffine(scales=(1,1), degrees=(10, 10),
                                                       image_interpolation=Interpolation.NEAREST )])
t = Compose( [RandomAffine(scales=(1,1), degrees=(10, 10) ),])
t = Compose([RandomNoise(std=(0.1, 0.1)),
             RandomAffineFFT(scales=(1, 1), degrees=(10, 10), oversampling_pct=0.2)])

dataset = ImagesDataset(suj, transform=t); dataset0 = ImagesDataset(suj);
s = dataset[0]; s0=dataset0[0]
#ov(s['image']['data'][0])
dataset.save_sample(s, dict(image='/home/romain/QCcnn/trot_shiftL.nii'))


for i in range(1,50):
    s=dataset[0]
    dataset.save_sample(s, dict(image='/home/romain/QCcnn/random_motion/random{:.2}.nii'.format(100*s['random_noise'])))

t = dataset.get_transform()
type(t)
isinstance(t, Compose)

from torch.utils.data import DataLoader
dl = DataLoader(dataset, batch_size=2,
                collate_fn=lambda x: x,  # this creates a list of Subjects
                )
samples = next(iter(dl))


suj = [Subject(image=Image(f1,'intensity'))]
t = torchio.Resample(target=f2)
d=torchio.ImagesDataset(suj,transform=t)
d.save_sample(s, dict(image='/tmp/t.nii'))


t = Compose([RandomNoise(seed=10)])
dataset = ImagesDataset(suj, transform=t);

sample=dataset[0]
for i in range(0, 10):
    sample = dataset[0]
    h=sample.history
    print(h[0][1]['image']['std'])

suj = [Subject(image=Image('/data/romain/data_exemple/suj_274542/T1w_acpc_dc_restore.nii.gz', INTENSITY))]
t = Compose([RandomAffineFFT(scales=(1/0.7, 1/0.7), degrees=(0, 0), oversampling_pct=0)])
t = Compose([RandomAffineFFT(scales=(1, 1), degrees=(0, 0), oversampling_pct=0)])
dataset = ImagesDataset(suj) #, transform=t);
#ov(s['image']['data'][0])

fout='/data/romain/data_exemple/suj_274542/T1w_1mm_fft2.nii'
s = dataset[0]

image=s['image']['data'][0]
ii = np.zeros((260,312,260))
#ii[:,1:,:] = image
ii[:,:-1,:] = image #this induce no shift if same resolution
output = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(ii)))).astype(np.complex128)
in_shape = np.array(image.shape)
out_shape = np.array([182, 218, 182])
diff_shape = (in_shape-out_shape)//2
oo_shape = out_shape+diff_shape
out_crop = output[diff_shape[0]:oo_shape[0], diff_shape[1]:oo_shape[1], diff_shape[2]:oo_shape[2]]
out_crop.shape

ifft = np.abs(np.fft.ifftshift(np.fft.ifftn(out_crop)))
#ifft[:,:-1,:] = ifft[:,1:,:]
#ifft[:,1:,:] = ifft[:,:-1,:]

ifft.shape

Iscale = np.prod(oo_shape) / np.prod(in_shape)
ifft = ifft * Iscale * 0.5586484991611643
affine = s['image']['affine']
affine[0,0] = -1
affine[1,1] = 1
affine[2,2] = 1
affine[0,3] = affine[0,3] - 0.15
affine[1,3] = affine[1,3] + 0.15
affine[2,3] = affine[2,3] + 0.15

nii = nib.Nifti1Image(ifft, affine)
nii.header['qform_code'] = 1
nii.header['sform_code'] = 1
nii.to_filename(str(fout))
