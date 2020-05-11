from doit_train import do_training, get_motion_transform, get_train_and_val_csv, get_cache_dir
import torch
from torchio.transforms import CropOrPad, RescaleIntensity, RandomAffine, ApplyMask

torch.multiprocessing.set_sharing_strategy('file_system')

do_save, do_eval, test_sample = False, False, False
make_uniform = False
mask_brain = True
batch_size, num_workers, max_epochs = 4, 0, 50
cuda, verbose = True, True
in_size = [182, 218, 182]

name_list_train = ['mask_mvt_train_cati_T1', 'mask_mvt_train_cati_ms', 'mask_mvt_cati_train_brain_ms',
                   'mvt_train_hcp400_ms', 'mask_mvt_train_hcp400_brain_ms', 'mask_mvt_train_hcp400_T1',
                   'ela1_train200_hcp400_ms']
name_list_val = ['mask_mvt_val_cati_T1', 'mask_mvt_val_cati_ms', 'mask_mvt_val_cati_brain_ms',
                 'mvt_val_hcp200_ms', 'mask_mvt_val_hcp200_brain_ms', 'mask_mvt_val_hcp200_T1']
#name_list_train = ['mask_mvt_train50_hcp400_ms', 'mask_mvt_train50_hcp400_brain_ms', 'mask_mvt_train50_hcp400_T1']
#name_list_val = ['mask_mvt_val_hcp200_ms', 'mask_mvt_val_hcp200_brain_ms', 'mask_mvt_val_hcp200_T1']

name_list_train = [ 'ela1_train_cati_T1', 'ela1_train_cati_ms', 'ela1_train_cati_brain',
                    'ela1_train_hcp400_ms', 'ela1_train_hcp400_T1',
                'ela1_train200_hcp400_ms']
name_list_val = ['ela1_val_cati_T1', 'ela1_val_cati_ms', 'ela1_val_cati_brain_ms',
                 'ela1_val_hcp200_ms', 'ela1_val_hcp200_T1',
		'ela1_train200_hcp400_ms']

data_name_train = name_list_train[3]
data_name_val = name_list_val[3]

res_dir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion/'
base_name = 'RegMotNew_resc_Aff'
base_name = 'RegMotNew'
if make_uniform : base_name += '_uniform'

root_fs = 'le70'
root_fs = 'lustre'

par_model = {'network_name': 'ConvN',
             'losstype': 'L1',
             'lr': 1e-4,
              'conv_block': [16, 32, 64, 128, 256], 'linear_block': [40, 50],
             'dropout': 0, 'batch_norm': True, 'drop_conv':0.1,
             'in_size': in_size,
             'cuda': cuda, 'max_epochs': max_epochs}
#'conv_block':[8, 16, 32, 64, 128]

tc=[]
add_log = ''

if mask_brain and 'hcp' in data_name_train:
    add_to_load, add_to_load_regexp = 'brain', 'brain_T'
else:
    add_to_load, add_to_load_regexp = None, None

if 'cati' in data_name_train:
    target_shape, mask_key = (182, 218, 182), 'brain'
    add_log += 'adding a CropOrPad {} with mask key {}'.format(target_shape, mask_key)
    print(add_log)
    tc.append( [CropOrPad(target_shape=target_shape, mask_name=mask_key), ] )

# before RescaleIntensity for hcp le 07/04/2020 mais pas pour cati
if mask_brain:
    tc.append(ApplyMask(masking_method='brain'))
    add_log += 'adding a ApplyMask brain '
    base_name += '_Mask'

if 'T1' in data_name_train:
    tc.append(RescaleIntensity(percentiles=(0, 99)))
    #tc.append(RandomAffine())
    add_log += 'adding a RESCALE Intensity 0 99 '
    base_name += '_rescale'
    print(add_log)

if len(tc) == 0: tc = None
if len(add_log) == 0: add_log = None


dir_cache = get_cache_dir(root_fs=root_fs)
load_from_dir = ['{}/{}/'.format(dir_cache, data_name_train), '{}/{}/'.format(dir_cache, data_name_val)]
res_name = '{}_{}'.format(base_name, data_name_train)

doit = do_training(res_dir, res_name, verbose)


if do_save:
    #rr test
    load_from_dir = [None]
    res_dir = res_name='/data/romain/HCPdata'
    train_csv_file, val_csv_file = res_name + '/healthy_brain_ms_train_hcp400.csv', res_name + '/healthy_brain_ms_val_hcp200.csv'
    doit = do_training(res_dir, res_name, verbose)

    transforms = get_motion_transform()
    train_csv_file, val_csv_file = get_train_and_val_csv(data_name_train, root_fs=root_fs)
    doit.set_data_loader(train_csv_file=train_csv_file, val_csv_file=val_csv_file, transforms=transforms,
                         batch_size=batch_size, num_workers=num_workers,
                         save_to_dir = load_from_dir[0], replicate_suj=20)
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

    from plot_dataset import PlotDataset
    PlotDataset(doit.train_dataset, image_key_name='image',subject_idx=20, subject_org=(5,4),
                views=[['sag', 'vox', 50], ['cor', 'vox', 50]])
else:

    if make_uniform:
        doit.set_data_loader(batch_size=batch_size, num_workers=num_workers, load_from_dir=load_from_dir, transforms=tc,
                             get_condition_csv='res_motion.csv', get_condition_field='ssim_brain',
                             add_to_load=add_to_load, add_to_load_regexp=add_to_load_regexp)
    else:
        doit.set_data_loader(batch_size=batch_size, num_workers=num_workers, load_from_dir=load_from_dir, transforms=tc,
                             add_to_load=add_to_load, add_to_load_regexp=add_to_load_regexp)
    doit.set_model(par_model)

    if add_log is not None:
        llog = doit.log
        llog.info(add_log)

    doit.train_regress_motion()

