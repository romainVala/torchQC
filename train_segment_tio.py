from smallunet_pytorch import SmallUnet, dice_loss, dice_coef_loss, print_metrics, calc_loss, \
    load_existing_weights_if_exist, summary
import torch.nn as tnn
import torch, os, logging, sys
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from collections import defaultdict
from utils_file import get_log_file, gfile, get_parent_path

from torchio import ImagesDataset, Queue
from torchvision.transforms import Compose
from torchio.data import ImageSampler, get_subject_list_and_csv_info_from_data_prameters
from torchvision.transforms import Compose
from unet import unet

#from nibabel.viewers import OrthoSlicer3D as ov

csv_file = '/data/romain/data_exemple/filename.csv'
sampling_met = 'weighted'
#sampling_met ='uniform'
winsize = 64
windows_size = (winsize, winsize, winsize )
queue_length, samples_per_volume = 1600,  160
#queue_length, samples_per_volume = 1000,  100
batch_size, num_workers, max_epochs = 4, 10, 100
bin_label = None# 0.5
cuda = True
losstype = 'BCElogit'
lr = 1e-4

model_name = 'model_unet'
if bin_label: model_name += '_labelBin{}'.format(bin_label)
model_name += '_{}_lr{}_B{}_W{}_spv{}_nw{}'.format( losstype, lr, batch_size, windows_size[0], samples_per_volume, num_workers)

resdir = "/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/UNET_saved_pytorch/" + model_name

if not os.path.isdir(resdir): os.mkdir(resdir)
log = get_log_file(resdir + '/training.log')

transforms = None

data_parameters = {'image': {'csv_file':'/data/romain/data_exemple/file_ms.csv'}, 'label1': {'csv_file':'/data/romain/data_exemple/file_p1.csv'},
            'label2': {'csv_file': '/data/romain/data_exemple/file_p2.csv'}, 'label3': {'csv_file':'/data/romain/data_exemple/file_p3.csv'},
                   'sampler': {'csv_file': '/data/romain/data_exemple/file_mask.csv'}}
roi_path = None #'/data/romain/data_exemple/roi16_weighted.txt'

subject_list, res_info = get_subject_list_and_csv_info_from_data_prameters(data_parameters)

train_dataset = ImagesDataset(subject_list, transform = transforms)
train_queue = Queue(train_dataset, queue_length, samples_per_volume, windows_size,
                    ImageSampler, num_workers=num_workers, shuffle_patches=True, verbose=False)

train_dataloader = DataLoader(train_queue, batch_size=batch_size, shuffle=True)

# d=next(iter(train_dataloader))
# d['image'].shape
# d['label'].shape


#loss = BCEWithLogitsLoss()  # sigmoid + bcel
#loss = dice_coef_loss #dice_loss
#loss = dice_loss()
if losstype == 'BCE': loss = tnn.BCELoss()
elif losstype == 'dice': loss = dice_loss(type=1)
elif losstype == 'BCElogit': loss = tnn.BCEWithLogitsLoss()


model = SmallUnet(in_channels=1, out_channels=3)

if cuda:
    model = model.cuda()
    loss = loss.cuda()
    device = "cuda"
else: device = 'cpu'

ep_start, last_model = load_existing_weights_if_exist(resdir, model, model_name='seg_unet', log=log)

max_epochs += ep_start

log.info(summary(model, (1, windows_size[0], windows_size[1], windows_size[1]), device=device, batch_size=batch_size))

optimizer = optim.Adam(model.parameters(), lr=lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)



for ep in range(ep_start, max_epochs):
    model.train()
    #exp_lr_scheduler.step() #to change learning rate ... ?
    metrics = defaultdict(float)
    epoch_samples = 0

    for iteration, data in enumerate(train_dataloader):
        optimizer.zero_grad()

        inputs = data['image']['data']

        lk = [kk for kk in data.keys() if ('label' in kk) ]
        list_label = [data[kkk]['data'].squeeze(1) for kkk in lk]
        labels = torch.stack(list_label, dim=1)

        #labels = data['label']
        if bin_label:
            labels = (labels > bin_label).float()

        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        with torch.set_grad_enabled(True):

            outputs = model(inputs)

            #outputs = torch.nn.Sigmoid()(outputs)
            l_tmp = loss(outputs, labels)
            calc_loss(outputs, labels, l_tmp.item(), metrics)
            epoch_samples += inputs.size(0)

            l_tmp.backward()
            optimizer.step()

        if iteration % 100 == 0:
            log.info(print_metrics(metrics, epoch_samples, 'train'))


    epoch_loss = metrics['loss'] / epoch_samples
    log.info("Ep: {} Iteration: {} Loss: {} mean {}".format(ep, iteration, l_tmp.item(), epoch_loss))

    if ep % 4 == 0:
        resname = resdir + "/unet_ep{}.pt".format(ep)
        torch.save({"seg_unet": model.state_dict()}, resname)
        log.info('saving model to %s' % (resname))


