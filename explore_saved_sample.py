from utils_file import gfile, gdir, get_parent_path
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from doit_train import do_training
from nilearn import plotting
import time, os, sys
import torch
from slices_2 import do_figure_from_dataset

# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)gb
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048*8, rlimit[1]))
# to remove RuntimeError: received 0 items of ancdata when num_worker > 0

#RuntimeError: unable to open shared memory object </torch_34340_19699415> in read-write mode
# solution   ->   sudo  sysctl -w kernel.shmmni=8192

# torch.multiprocessing.set_sharing_strategy('file_system')
# ca enleve bien le prb ancdata mais  -> core dump

do_plot_mv, do_plot_fig = False, False
do_plot_mv, do_plot_fig = False, True

batch_size, num_workers = 1, 0
cuda, verbose = True, True

name_list_val = ['ela1_val_cati_T1', 'ela1_val_cati_ms', 'ela1_val_cati_brain_ms',
                 'ela1_val_hcp200_ms', 'ela1_val_hcp200_brain_ms', 'ela1_val_hcp200_T1']
name_list_train = ['ela1_train200_hcp400_ms', 'ela1_train_cati_ms', 'ela1_train_cati_brain',
                    'ela1_train_hcp400_ms', 'ela1_train_hcp400_brain_ms', 'ela1_train_hcp400_T1']

name_list = name_list_train# [ name_list_train[3], name_list_val[3]]
name_list = ['ela1_train200_hcp400_ms']

dir_cache = '/network/lustre/dtlake01/opendata/data/ds000030/rrr/CNN_cache/'
#dir_cache = '/data/romain/CNN_cache/'

for data_name in name_list:

    load_from_dir = '{}/{}/'.format(dir_cache, data_name)
    res_dir = load_from_dir
    res_name = 'NNN'

    doit = do_training(res_dir, res_name, verbose)
    doit.set_data_loader(batch_size=batch_size, num_workers=num_workers, load_from_dir=load_from_dir, shuffel_train=False)

    td = doit.train_dataloader

    fsaved = doit.train_csv_load_file_train
    #fname_saved = get_parent_path(fsaved)[1]
    #fcsv = gfile(gdir(load_from_dir, 'mvt_param'), '.*csv') #should be order by sample not by ssim

    plt.ioff()
    resdir_fig = res_dir + '/fig/'
    resdir_mvt = res_dir + '/mvt_param/'
    fres = res_dir + '/res_motion.csv'

    if not os.path.exists(fres):
        print('found no result file {} \n So building it'.format(fres))
        print('loading {} sample from {} \n estimated time {} h (for 2 iter/s)'.format(len(td), load_from_dir, len(td)/(2*60*60)))
        start = time.time()
        res, extra_info = pd.DataFrame(), dict()
        for ii, sample in enumerate(tqdm(td)):
            ff = sample['mvt_csv']
            fcsv_name = get_parent_path(ff)[1][0]
            extra_info = {'mvt_csv': ff, 'sample': fsaved[ii]}
            res = doit.add_motion_info(sample, res, extra_info)

        res.to_csv(fres)
        print('saving {}'.format(fres))
        print('done csv in {}'.format((time.time()-start)/60/60))

    if do_plot_fig:
        start = time.time()
        res = pd.read_csv(fres)
        param = res.ssim.values
        nb_val = 30 #96
        pind = np.argsort(param)
        p0 = np.percentile(pind, 10) #first 10
        pm1, pm2 = np.percentile(pind, 40), np.percentile(pind, 60)
        pl1, pl2 = np.percentile(pind,90), len(pind)
        indsel_list = [pind[np.random.choice(range(0,p0), size=nb_val, replace=False)],
                       pind[np.random.choice(range(pm1,p2), size=nb_val, replace=False)],
                       pind[np.random.choice(range(pl1,pl2), size=nb_val, replace=False)] ]

        param_fig = {'slices_infos': [("sag", "vox", 0.4), ("cor", "vox", 0.6), ("ax", "vox", 0.5), ],
                     'mask_info': [("mask", -1)],
                     'display_order': np.array([1, 3]),
                     'out_dir': res_dir,
                     'mask_key': 'brain',
                     'plot_single': True,
                     'montage_shape': (5, 3)}

        name_fig = [ ['fig_ssim_{:.4f}'.format(vv) for vv in res.ssim[indsel] ]
                     for indsel in indsel_list ]
        montage_basename = ["low_","medium_","high_"]

        td = doit.train_dataset
        for ii,nn, mm in zip(indsel_list, name_fig, montage_basename):
            print("plotion subset {} of images".format(ii.shape))
            do_figure_from_dataset(td, ii, nn, montage_basename=mm,  **param_fig)

        print('done Ploting in {}'.format((time.time()-start)/60/60))

test=False
if test:

    td = doit.train_dataloader
    data = next(iter(td))

    doit = do_training(res_dir, res_name, verbose)
    doit.set_data_loader(train_csv_file, val_csv_file, None, batch_size, num_workers, load_from_dir = load_from_dir)

    td = doit.train_dataloader
    data = next(iter(td))
    fs = doit.train_csv_load_file_train

    for ff in tqdm(fs):
        sample = torch.load(ff)
        if type(sample) is tuple: #don't know why?
            sample = sample[1]
            print('RRRRR SAving {}'.format(ff))
            torch.save(sample, ff)

        else:
            if 'image_orig' in sample:
                sample.pop('image_orig')
                print('saving {} type{}'.format(ff, type(sample)))
                torch.save(sample, ff)



    print('printing volume figure')

    for sample in tqdm(td):
        ff = sample['mvt_csv']
        fcsv_name = get_parent_path(ff)[1][0]

        if do_plot_mv:
            mvt = pd.read_csv(ff[0], header=None)
            fpars = np.asarray(mvt)
            fname_mvt = fcsv_name[:-4]

            fig = plt.figure(fname_mvt)
            plt.plot(fpars.T)
            plt.savefig(resdir_mvt + fname_mvt + '.png')
            plt.close(fig)

        if do_plot_fig:
            image = sample['image']['data'][0][0].numpy()
            affine = sample['image']['affine'][0]
            nii = nib.Nifti1Image(image, affine)
            fname = resdir_fig + fcsv_name[:-8] + '_fig.png'

            di = plotting.plot_anat(nii, output_file=fname, annotate=False, draw_cross=False)
