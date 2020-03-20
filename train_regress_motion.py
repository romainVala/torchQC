from doit_train import do_training
from torchio.transforms import RandomMotionFromTimeCourse
#from nibabel.viewers import OrthoSlicer3D as ov
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

par_queue = {'windows_size': [64], 'queue_length': 800,
            'samples_per_volume': 16,}

batch_size, num_workers, max_epochs = 4, 6, 100
cuda, verbose = True, False
in_size=[182, 218, 182]
in_size=[64, 64, 64]

res_dir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion/'
res_name = 'RegressMot_PATCH_HCPbrain_ms'

import socket
myHostName = socket.gethostname()
if 'le53' in myHostName:
    par_queue = {'windows_size': [64], 'queue_length': 128,
                 'samples_per_volume': 64, }
    batch_size, num_workers, max_epochs = 4, 2, 1
    cuda, verbose = False, True
    res_dir = '/home/romain/QCcnn/'


dico_params = {"maxDisp": (1, 6),  "maxRot": (1, 6),    "noiseBasePars": (5, 20, 0.8),
               "swallowFrequency": (2, 6, 0.5),  "swallowMagnitude": (3, 6),
               "suddenFrequency": (2, 6, 0.5),  "suddenMagnitude": (3, 6),
               "verbose": False, "keep_original": True, "compare_to_original":True, "proba_to_augment": 1}
transforms = (RandomMotionFromTimeCourse(**dico_params),)


train_csv_file, val_csv_file = 'healthy_brain_ms_train_hcp400.csv', 'healthy_brain_ms_val_hcp200.csv'

par_model = {'network_name': 'ConvN',
             'losstype': 'BCElogit',
             'lr': 1e-5,
              'conv_block': [32, 64, 128, 256], 'linear_block': [40, 50],
             'in_size': in_size,
             'cuda': cuda,
             'max_epochs': max_epochs}
#'conv_block':[8, 16, 32, 64, 128]


doit = do_training(res_dir, res_name, verbose)

doit.set_data_loader(train_csv_file, val_csv_file, transforms, batch_size, num_workers, par_queue=par_queue)

doit.set_model(par_model)

doit.train_regress_motion()



test=False
if test:
    td = doit.train_dataloader
    data = next(iter(td))
