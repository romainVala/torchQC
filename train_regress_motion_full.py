from doit_train import do_training, get_motion_transform, get_train_and_val_csv, get_cache_dir
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

do_save, do_eval = False, False
batch_size, num_workers, max_epochs = 4, 4, 5
cuda, verbose = True, True
in_size=[182, 218, 182]

name_list_train = ['mask_mvt_cati_T1', 'mask_mvt_cati_ms', 'mask_mvt_cati_brain_ms',
                   'mask_mvt_train_hcp400_ms', 'mask_mvt_train_hcp400_brain_ms', 'mask_mvt_train_hcp400_T1']
name_list_val = ['mask_mvt_cati_T1', 'mask_mvt_cati_ms', 'mask_mvt_cati_brain_ms',
                 'mask_mvt_val_hcp200_ms', 'mask_mvt_val_hcp200_brain_ms', 'mask_mvt_val_hcp200_T1']

data_name_train = name_list_train[3]
data_name_val = name_list_val[3]

res_dir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion/'
base_name = 'RegressMot'

root_fs = 'le70' #'lustre'

par_model = {'network_name': 'ConvN',
             'losstype': 'L1',
             'lr': 1e-5,
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

else:
    doit.set_data_loader(batch_size=batch_size, num_workers=num_workers, load_from_dir = load_from_dir)
    doit.set_model(par_model)
    doit.train_regress_motion()

