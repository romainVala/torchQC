from utils_file import gfile, gdir, get_parent_path
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from doit_train import do_training
from nilearn import plotting
import time

do_plot_mv, do_plot_fig = False, False
#do_plot_mv, do_plot_fig = True, True

batch_size, num_workers = 1, 0
cuda, verbose = True, True

name_list = [ 'motion_cati_T1', 'motion_cati_ms', 'motion_cati_brain_ms',
              'motion_train_hcp400_ms', 'motion_train_hcp400_brain_ms', 'motion_train_hcp400_T1']

data_name = 'motion_train_hcp400_T1' #'motion_train_hcp400_ms'#'motion_train_hcp400_brain_ms'
data_name = name_list[0]
load_from_dir = '/network/lustre/dtlake01/opendata/data/ds000030/rrr/CNN_cache/{}/'.format(data_name)
load_from_dir = '/data/romain/CNN_cache/{}/'.format(data_name)

res_dir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion/'
res_name = 'NNN'

import socket
myHostName = socket.gethostname()
if 'le53' in myHostName:
    batch_size, num_workers, max_epochs = 1, 0, 1
    cuda, verbose = False, True
    res_dir = '/home/romain/QCcnn/'
    load_from_dir = '/home/romain/QCcnn/motion_cati_brain_ms/'


doit = do_training(res_dir, res_name, verbose)
doit.set_data_loader(None, None, None, batch_size, num_workers, load_from_dir = load_from_dir)

td = doit.train_dataloader
print('loading {} sample from {} \n estimated time {} h'.format(len(td), load_from_dir, len(td)*0.002))
start = time.time()

#fsaved = doit.train_csv_load_file
#fname_saved = get_parent_path(fsaved)[1]
#fcsv = gfile(gdir(load_from_dir, 'mvt_param'), '.*csv') #should be order by sample not by ssim

plt.ioff()
res_dir = load_from_dir
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
    res = doit.add_motion_info(sample, res)

fres = res_dir + '/res_motion.csv'
res.to_csv(fres)

print('saving {}'.format(fres))
print('done in {}'.format((time.time()-start)/60/60))

test=False
if test:
    td = doit.train_dataloader
    data = next(iter(td))
