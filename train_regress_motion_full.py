from doit_train import do_training, get_motion_transform, get_train_and_val_csv, get_cache_dir
import torch
from torchio.transforms import CenterCropOrPad

torch.multiprocessing.set_sharing_strategy('file_system')

do_save, do_eval, test_sample = False, False, False
batch_size, num_workers, max_epochs = 4, 0, 10
cuda, verbose = True, True
in_size=[182, 218, 182]

name_list_train = ['mask_mvt_train_cati_T1', 'mask_mvt_train_cati_ms', 'mask_mvt_cati_train_brain_ms',
                   'mask_mvt_train_hcp400_ms', 'mask_mvt_train_hcp400_brain_ms', 'mask_mvt_train_hcp400_T1']
name_list_val = ['mask_mvt_val_cati_T1', 'mask_mvt_val_cati_ms', 'mask_mvt_val_cati_brain_ms',
                 'mask_mvt_val_hcp200_ms', 'mask_mvt_val_hcp200_brain_ms', 'mask_mvt_val_hcp200_T1']
#name_list_train = ['mask_mvt_train50_hcp400_ms', 'mask_mvt_train50_hcp400_brain_ms', 'mask_mvt_train50_hcp400_T1']
name_list_train = [ 'mvt_train_cati_T1', 'mvt_train_cati_ms', 'mvt_train_cati_brain',
              'mvt_train_hcp400_ms', 'mvt_train_hcp400_brain_ms', 'mvt_train_hcp400_T1']
name_list_val = ['mask_mvt_val_hcp200_ms', 'mask_mvt_val_hcp200_brain_ms', 'mask_mvt_val_hcp200_T1']

data_name_train = name_list_train[3]
data_name_val = name_list_val[0]

res_dir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion/'
base_name = 'RegressMot_badVal'

root_fs = 'le70' #
#root_fs = 'lustre'

par_model = {'network_name': 'ConvN',
             'losstype': 'L1',
             'lr': 1e-4,
              'conv_block': [16, 32, 64, 128, 256], 'linear_block': [40, 50],
             'in_size': in_size,
             'cuda': cuda, 'max_epochs': max_epochs}
#'conv_block':[8, 16, 32, 64, 128]

dir_cache = get_cache_dir(root_fs=root_fs)
load_from_dir = ['{}/{}/'.format(dir_cache, data_name_train), '{}/{}/'.format(dir_cache, data_name_val)]
res_name = '{}_{}'.format(base_name, data_name_train)

doit = do_training(res_dir, res_name, verbose)

if do_save:
    transforms = get_motion_transform()
    train_csv_file, val_csv_file = get_train_and_val_csv(root_fs=root_fs)
    doit.set_data_loader(train_csv_file=train_csv_file, val_csv_file=val_csv_file, transforms=transforms,
                         batch_size=batch_size, num_workers=num_workers,
                         save_to_dir = load_from_dir, replicate_suj=20)
    doit.save_to_dir(load_from_dir) # no more use, because it is much faster on cluster with job created by

elif do_eval:
    doit.set_data_loader(batch_size=batch_size, num_workers=num_workers, load_from_dir=load_from_dir)
    doit.set_model(par_model)
    doit.eval_regress_motion()

elif test_sample:
    import numpy as np
    batch_size=1
    doit.set_data_loader(batch_size=batch_size, num_workers=num_workers, load_from_dir = load_from_dir, shuffel_train=False)
    doit.set_model(par_model)
    td = doit.train_dataloader
    llog = doit.log
    for i, data in enumerate(td):
        dd = data['image']['data'].reshape(-1).numpy()
        llog.info('{} max is {}'.format(i, np.max(dd)))

else:
    #t = None #(CenterCropOrPad(target_shape=tuple(in_size)),)
    doit.set_data_loader(batch_size=batch_size, num_workers=num_workers, load_from_dir = load_from_dir)
    doit.set_model(par_model)
    doit.train_regress_motion()

