from doit_train import do_training, get_motion_transform, get_train_and_val_csv, get_cache_dir
import torch
from torchio.transforms import CropOrPad

torch.multiprocessing.set_sharing_strategy('file_system')

make_uniform, do_eval = False, False
batch_size, num_workers, max_epochs = 4, 4, 50
cuda, verbose = True, True
in_size = [182, 218, 182]

name_list_train = ['train_cati_T1', 'train_cati_ms', 'train_cati_brain',
                   'train_hcp400_ms', 'train_hcp400_brain_ms', 'train_hcp400_T1']
name_list_val = ['val_cati_T1', 'val_cati_ms', 'val_cati_brain_ms',
                 'val_hcp200_ms', 'val_hcp200_brain_ms', 'val_hcp200_T1']

data_name_train = name_list_train[3]
data_name_val = name_list_val[3]
nb_replicate=10

if do_eval:
    nb_replicate=10;
    data_name_val = data_name_train

res_dir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_random_noise/'
base_name = 'Reg'
if make_uniform : base_name += '_uniform'

root_fs = 'le70' #
#root_fs = 'lustre'

# rr test
import socket
myHostName = socket.gethostname()
if 'le53' in myHostName:
    res_dir, res_name = '/data/romain/HCPdata/', 'toto'
    train_csv_file, val_csv_file = res_dir + '/healthy_brain_ms_train_hcp400.csv', res_dir + '/healthy_brain_ms_val_hcp200.csv'
    cuda = False
else:
    train_csv_file, val_csv_file = get_train_and_val_csv(data_name_train, root_fs=root_fs)


par_model = {'network_name': 'ConvN',
             'losstype': 'L1',
             'lr': 1e-5,
             'conv_block': [16, 32, 64, 128, 256], 'linear_block': [40, 50],
             'dropout': 0, 'batch_norm': True,
             'in_size': in_size,
             'cuda': cuda, 'max_epochs': max_epochs}
#'conv_block':[8, 16, 32, 64, 128]

dir_cache = get_cache_dir(root_fs=root_fs)
#load_from_dir = ['{}/{}/'.format(dir_cache, data_name_train), '{}/{}/'.format(dir_cache, data_name_val)]
res_name = '{}_{}'.format(base_name, data_name_train)
load_from_dir = [None]


if 'cati' in data_name_train:
    target_shape, mask_key = (182, 218, 182), 'brain'
    print('adding a CropOrPad {} with mask key {}'.format(target_shape, mask_key))
    tc = [CropOrPad(target_shape=target_shape, mode='mask', mask_key=mask_key), ]
else:
    tc = None

doit = do_training(res_dir, res_name, verbose)

transforms = get_motion_transform('random_noise_1')

if do_eval:
    val_csv_file = train_csv_file

doit.set_data_loader(train_csv_file=train_csv_file, val_csv_file=val_csv_file, transforms=transforms,
                     batch_size=batch_size, num_workers=num_workers,
                     save_to_dir=load_from_dir[0], replicate_suj=nb_replicate,
                     collate_fn=lambda x: x)

doit.set_model(par_model)
if do_eval:
    doit.val_dataloader = doit.train_dataloader
    doit.eval_regress_motion(1000, 10, target='random_noise',basename='res_val_train10')

else:
    doit.train_regress_motion(target='random_noise')

