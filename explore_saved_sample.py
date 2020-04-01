from utils_file import gfile, gdir, get_parent_path
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from doit_train import do_training
from nilearn import plotting
import time
import torch

do_plot_mv, do_plot_fig = False, False
#do_plot_mv, do_plot_fig = True, True

batch_size, num_workers = 1, 0
cuda, verbose = True, True

name_list = [ 'mvt_train_cati_T1', 'mvt_train_cati_ms', 'mvt_train_cati_brain',
              'mvt_train_hcp400_ms', 'mvt_train_hcp400_brain_ms', 'mvt_train_hcp400_T1']

dir_cache = '/network/lustre/dtlake01/opendata/data/ds000030/rrr/CNN_cache/'
#dir_cache = '/data/romain/CNN_cache/'

for data_name in name_list:

    load_from_dir = '{}/{}/'.format(dir_cache, data_name)
    res_dir = load_from_dir
    res_name = 'NNN'

    doit = do_training(res_dir, res_name, verbose)
    doit.set_data_loader(None, None, None, batch_size, num_workers, load_from_dir=load_from_dir, shuffel_train=False)

    td = doit.train_dataloader
    print('loading {} sample from {} \n estimated time {} h'.format(len(td), load_from_dir, len(td)/(2*60*60)))
    start = time.time()

    #fsaved = doit.train_csv_load_file_train
    #fname_saved = get_parent_path(fsaved)[1]
    #fcsv = gfile(gdir(load_from_dir, 'mvt_param'), '.*csv') #should be order by sample not by ssim

    plt.ioff()
    resdir_fig = res_dir + '/fig/'
    resdir_mvt = res_dir + '/mvt_param/'


    print('printing volume figure')
    res, extra_info = pd.DataFrame(), dict()

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
        extra_info = {'mvt_csv':ff}
        res = doit.add_motion_info(sample, res, extra_info)

    fres = res_dir + '/res_motion.csv'
    res.to_csv(fres)

    print('saving {}'.format(fres))
    print('done in {}'.format((time.time()-start)/60/60))

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

