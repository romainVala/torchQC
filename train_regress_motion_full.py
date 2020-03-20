from doit_train import do_training
from torchio.transforms import RandomMotionFromTimeCourse
#from nibabel.viewers import OrthoSlicer3D as ov
import torch, os
torch.multiprocessing.set_sharing_strategy('file_system')

do_save = False
batch_size, num_workers, max_epochs = 4, 0, 5
cuda, verbose = True, True
in_size=[182, 218, 182]

res_dir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion/'
load_from_dir = '/data/romain/CNN_cache/motion'
res_name = 'RegressMot_HCPbrain_ms'
if os.path.exists('/data/romain/toolbox_python/torchio'):
    data_path = '/data/romain/HCPdata/'
else:
    data_path = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/'

import socket
myHostName = socket.gethostname()
if 'le53' in myHostName:
    batch_size, num_workers, max_epochs = 4, 2, 1
    cuda, verbose = False, True
    res_dir = '/home/romain/QCcnn/'
    load_from_dir = '/home/romain/QCcnn/'

dico_params = {"maxDisp": (1, 6),  "maxRot": (1, 6),    "noiseBasePars": (5, 20, 0.8),
               "swallowFrequency": (2, 6, 0.5),  "swallowMagnitude": (3, 6),
               "suddenFrequency": (2, 6, 0.5),  "suddenMagnitude": (3, 6),
               "verbose": False, "keep_original": True, "compare_to_original":True, "proba_to_augment": 1}
transforms = (RandomMotionFromTimeCourse(**dico_params),)


train_csv_file, val_csv_file = data_path + 'healthy_brain_ms_train_hcp400.csv', data_path + 'healthy_brain_ms_val_hcp200.csv'

par_model = {'network_name': 'ConvN',
             #'output_fnc': 'tanh',
             'losstype': 'MSE',
             'lr': 1e-5,
              'conv_block': [16, 32, 64, 128, 256], 'linear_block': [40, 50],
             'in_size': in_size,
             'cuda': cuda,
             'max_epochs': max_epochs}
#'conv_block':[8, 16, 32, 64, 128]


doit = do_training(res_dir, res_name, verbose)

if do_save:
    doit.set_data_loader(train_csv_file, val_csv_file, transforms, batch_size, num_workers, save_to_dir = load_from_dir, replicate_suj=50)
    doit.save_to_dir(load_from_dir)
else:
    doit.set_data_loader(train_csv_file, val_csv_file, transforms, batch_size, num_workers, load_from_dir = load_from_dir)

    doit.set_model(par_model)
    doit.train_regress_motion()



test=False
if test:
    td = doit.train_dataloader
    data = next(iter(td))

    f=gfile('/home/romain/QCcnn/li/','.*csv')
    for ff in f:
        res=pd.read_csv(ff)
        #plt.scatter(res.ssim,res.model_out)
        #plt.plot(res.ssim, res.ssim,'k+')
        err = np.abs(res.ssim-res.model_out) #same as L1 loss
        plt.figure('err'); plt.plot(err)

        errcum = np.cumsum(err[50:-1])/range(1,len(err)-51+1)
        plt.figure('err_cum');        plt.plot(errcum)
        N=50
        err_slidin = np.convolve(err, np.ones((N,))/N, mode='valid')

        plt.figure('err_slide'); plt.plot(err_slidin)
    plt.legend(('1','2','3','4','5'))

    import torch.nn as nn
    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    steps = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, last_epoch=-1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1)
    for epoch in range(5):
        for idx in range(steps):
            scheduler.step()
            print(scheduler.get_lr())

        #print('Reset scheduler')
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
