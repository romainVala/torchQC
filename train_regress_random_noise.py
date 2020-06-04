from doit_train import do_training, get_motion_transform, get_train_and_val_csv, get_cache_dir
from utils_cmd import get_tranformation_list
import torch
from torchio.transforms import CropOrPad
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')

make_uniform, do_eval = False, True
add_affine_zoom, add_affine_rot = 0, 0

batch_size, num_workers, max_epochs = 4, 24, 50
cuda, verbose = True, True
in_size = [182, 218, 182]

name_list_train = ['train_cati_T1', 'train_cati_ms', 'train_cati_brain',
                   'train_hcp400_ms', 'train_hcp400_brain_ms', 'train_hcp400_T1']
name_list_val = ['val_cati_T1', 'val_cati_ms', 'val_cati_brain_ms',
                 'val_hcp200_ms', 'val_hcp200_brain_ms', 'val_hcp200_T1']

data_name_train = name_list_train[3]
data_name_val = name_list_val[3]
nb_replicate=20

if do_eval:
    nb_replicate=3;

res_dir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_random_noise/'
base_name = 'Reg_AffN'
if make_uniform : base_name += '_uniform'

root_fs = 'le70' #
#root_fs = 'lustre'

train_csv_file, val_csv_file = get_train_and_val_csv(data_name_train, root_fs=root_fs)

if do_eval:
    train_csv_file = val_csv_file

par_model = {'network_name': 'ConvN',
             'losstype': 'L1',
             'lr': 1e-4,
             'conv_block': [16, 32, 64, 128, 256], 'linear_block': [40, 50],
             'dropout': 0, 'batch_norm': True,'drop_conv':0.1,
             'validation_droupout': False,
             'in_size': in_size,
             'cuda': cuda, 'max_epochs': max_epochs}
#'conv_block':[8, 16, 32, 64, 128]

dir_cache = get_cache_dir(root_fs=root_fs)
#load_from_dir = ['{}/{}/'.format(dir_cache, data_name_train), '{}/{}/'.format(dir_cache, data_name_val)]
res_name = '{}_{}'.format(base_name, data_name_train)
load_from_dir = [None]


doit = do_training(res_dir, res_name, verbose)

#transforms = get_motion_transform('random_noise_1')
transforms = get_motion_transform('AffFFT_random_noise')

if do_eval:

    from torchio.transforms import CropOrPad, RandomAffine, RescaleIntensity, ApplyMask, RandomBiasField, RandomNoise, \
        Interpolation, RandomAffineFFT

    from utils_file import get_parent_path, gfile, gdir
    from utils import get_ep_iter_from_res_name

    tc = [ RandomNoise(std=(0.020, 0.2)) ]

    if add_affine_rot>0 or add_affine_zoom >0:
        if add_affine_zoom == 0: add_affine_zoom = 1  # 0 -> no affine so 1
        # tc.append(RandomAffine(scales=(add_affine_zoom, add_affine_zoom), degrees=(add_affine_rot, add_affine_rot),
        #                        image_interpolation = Interpolation.NEAREST ))
        # name_suffix = '_tAff_nearest_S{}R{}'.format(add_affine_zoom, add_affine_rot)
        tc.append(RandomAffineFFT(scales=(add_affine_zoom, add_affine_zoom), degrees=(add_affine_rot, add_affine_rot),
                                  oversampling_pct=0.2 ))

        name_suffix = '_tAff_fft_S{}R{}'.format(add_affine_zoom, add_affine_rot)
    else:
        name_suffix = '_raw'

    transforms = tc

doit.set_data_loader(train_csv_file=train_csv_file, val_csv_file=val_csv_file, transforms=transforms,
                     batch_size=batch_size, num_workers=num_workers,
                     save_to_dir=load_from_dir[0], replicate_suj=nb_replicate,
                     collate_fn=lambda x: x)

if do_eval:

    root_dir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_random_noise/'
    model = gdir(root_dir, 'Reg.*D0_DC')

    # saved_models = []
    # for mm in model:
    #     ss_models = gfile(mm, '_ep.*pt$');
    #     nb_it=8000
    #     fresV_sorted, b, c = get_ep_iter_from_res_name(ss_models, nb_it)
    #     nb_iter = b * nb_it + c
    #     ii = np.where(nb_iter > 200000)[1:8]
    #     ss_models = list(ss_models[ii])
    #
    #     #ss_models = list(fresV_sorted[-8:])
    #
    #     saved_models = ss_models + saved_models
    saved_models = gfile(model, '_ep27_.*pt$');

    tlist, tname = get_tranformation_list(choice=[1, 2])

    for saved_model in saved_models:
        doit.set_model_from_file(saved_model, cuda=cuda)

        doit.val_dataloader = doit.train_dataloader
        basename = 'res_valOn_val_hcp_ms' + name_suffix
        #doit.eval_regress_motion(1000, 10, target='random_noise', basename=basename)
        doit.eval_multiple_transform(1000, 10, target='random_noise', basename=basename,
                                     transform_list=tlist, transform_list_name=tname)

else:
    doit.set_model(par_model)
    doit.train_regress_motion(target='random_noise')

