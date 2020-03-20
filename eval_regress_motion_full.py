from doit_train import do_training
from torchio.transforms import RandomMotionFromTimeCourse
#from nibabel.viewers import OrthoSlicer3D as ov
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

batch_size, num_workers, max_epochs = 1, 16, 10
cuda, verbose = True, True
in_size=[182, 218, 182]

res_dir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion/'
res_dir = '/data/romain/CNN_cache/motion1'
res_name = 'RegressMot_HCPbrain_ms'

import socket
myHostName = socket.gethostname()
if 'le53' in myHostName:
    batch_size, num_workers, max_epochs = 2, 2, 1
    cuda, verbose = False, True
    res_dir = '/home/romain/QCcnn/'

dico_params = {"maxDisp": (1, 6),  "maxRot": (1, 6),    "noiseBasePars": (5, 20, 0.8),
               "swallowFrequency": (2, 6, 0.5),  "swallowMagnitude": (3, 6),
               "suddenFrequency": (2, 6, 0.5),  "suddenMagnitude": (3, 6),
               "verbose": False, "keep_original": False, "compare_to_original": True, "proba_to_augment": 1,
               "res_dir": res_dir}

transforms = (RandomMotionFromTimeCourse(**dico_params),)


train_csv_file, val_csv_file = 'healthy_brain_ms_train_hcp400.csv', 'healthy_brain_ms_val_hcp200.csv'

par_model = {'network_name': 'ConvN',
             #'output_fnc': 'tanh',
             'losstype': 'MSE',
             'lr': 1e-4,
              'conv_block': [16, 32, 64, 128, 256], 'linear_block': [40, 50],
             'in_size': in_size,
             'cuda': cuda,
             'max_epochs': max_epochs}
#'conv_block':[8, 16, 32, 64, 128]


doit = do_training(res_dir, res_name, verbose)

doit.set_data_loader(train_csv_file, val_csv_file, transforms, batch_size, num_workers, save_to_dir = res_dir, replicate_suj=50)
doit.save_to_dir(res_dir)


doit.set_data_loader(train_csv_file, val_csv_file, transforms, batch_size, num_workers,load_from_dir = res_dir)


doit.set_model(par_model)

doit.train_regress_motion()



test=False
if test:
    td = doit.train_dataloader
    data = next(iter(td))

    from tqdm import tqdm
    for data in tqdm(td):
        print(data['image']['metrics']['ssim'])



