import os
from torchio import ImagesDataset, Queue
from torchio.data import ImagesClassifDataset, get_subject_list_and_csv_info_from_data_prameters
from torchio.data.sampler import ImageSampler
from torchio.utils import is_image_dict
from torchio import INTENSITY

from torch.utils.data import DataLoader
import torch.nn as tnn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.transforms import Compose

from utils_file import get_log_file, gfile, get_parent_path, gdir

from torchio.transforms.metrics import SSIM3D, ssim3D
from smallunet_pytorch import ConvN_FC3, SmallUnet, load_existing_weights_if_exist, summary
from unet import UNet, UNet3D

import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt

#from collections import defaultdict

class do_training():
    def __init__(self, res_dir, res_name='', verbose=False):

        self.res_name = res_name
        self.res_dir = res_dir
        if not os.path.isdir(res_dir): os.mkdir(res_dir)
        self.verbose = verbose
        #self.log_file = self.res_dir + '/training.log'
        self.log_string = ''
        #self.log = get_log_file(self.log_file)


    def set_data_loader(self, train_csv_file, val_csv_file, transforms=None,
                        batch_size=1, num_workers=0,
                        par_queue=None, save_to_dir=None, load_from_dir=None, replicate_suj=0 ):

        if load_from_dir is not None :
            fsample = gfile(load_from_dir, 'sample.*pt')
            self.log_string += '\nloading {} sample from {}'.format(len(fsample), load_from_dir)
            train_dataset = ImagesDataset(fsample, load_from_dir=load_from_dir)
            val_dataset = train_dataset

        else :
            data_parameters = {'image': {'csv_file': train_csv_file}, }
            data_parameters_val = {'image': {'csv_file': val_csv_file}, }

            paths_dict, info = get_subject_list_and_csv_info_from_data_prameters(data_parameters, fpath_idx='filename')
            paths_dict_val, info_val = get_subject_list_and_csv_info_from_data_prameters(
                data_parameters_val, fpath_idx='filename', shuffle_order=False)

            if replicate_suj:
                lll = []
                for i in range(0, replicate_suj):
                    lll.extend(paths_dict)
                paths_dict = lll
                self.log_string += 'Replicating train dataSet {} times, new length is {}'.format(replicate_suj,len(lll))

            train_dataset = ImagesDataset(paths_dict, transform=Compose(transforms), save_to_dir=save_to_dir)
            val_dataset = ImagesDataset(paths_dict_val, transform=Compose(transforms), save_to_dir=save_to_dir)

        self.res_name += '_B{}_nw{}'.format(batch_size, num_workers)

        if par_queue is not None:
            self.patch = True
            windows_size = par_queue['windows_size']
            if len(windows_size) == 1:
                windows_size = [windows_size[0], windows_size[0], windows_size[0]]

            train_queue = Queue(train_dataset,
                                par_queue['queue_length'],
                                par_queue['samples_per_volume'],
                                windows_size,
                                ImageSampler,
                                num_workers=num_workers, verbose=self.verbose)

            val_queue = Queue(val_dataset, par_queue['queue_length'], 1, windows_size,
                              ImageSampler, num_workers=num_workers,
                              shuffle_subjects=False, shuffle_patches=False, verbose=self.verbose)
            self.res_name += '_spv{}'.format(par_queue['samples_per_volume'])

            self.train_dataloader = DataLoader(train_queue, batch_size=batch_size, shuffle=True)
            self.val_dataloader = DataLoader(val_queue, batch_size=batch_size, shuffle=False)

        else:
            self.patch = False
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)



    def set_model(self, par_model):

        network_name = par_model['network_name']
        losstype = par_model['losstype']
        lr = par_model['lr']
        in_size = par_model['in_size']
        self.cuda = par_model['cuda']
        self.max_epochs = par_model['max_epochs' ]

        if network_name == 'unet_f':
            self.model = UNet(in_channels=1, dimensions=3, out_classes=1, num_encoding_blocks=3, out_channels_first_layer=16,
                         normalization='batch', padding=True,
                         pooling_type='max',  # max avg AdaptiveMax AdaptiveAvg
                         upsampling_type='trilinear', residual=False,
                         dropout=False, monte_carlo_dropout=0.5)

        elif network_name == 'unet':
            self.model = SmallUnet(in_channels=1, out_channels=1)

        elif network_name == 'ConvN':
            conv_block = par_model['conv_block']
            linear_block = par_model['linear_block']
            output_fnc = par_model['output_fnc'] if 'output_fnc' in par_model else None
            self.model = ConvN_FC3(in_size=in_size, conv_block=conv_block, linear_block=linear_block, output_fnc=output_fnc)
            network_name += '_C{}_{}_Lin{}_{}'.format(np.abs(conv_block[0]), conv_block[-1], linear_block[0], linear_block[-1])
            if output_fnc is not None:
                network_name += '_fnc_{}'.format(output_fnc)

        self.res_name += '_Size{}_{}_Loss_{}_lr{}'.format(in_size[0], network_name, losstype, lr)

        self.res_dir += self.res_name + '/'
        if not os.path.isdir(self.res_dir): os.mkdir(self.res_dir)

        self.log = get_log_file(self.res_dir + '/training.log')
        self.log.info(self.log_string)

        if losstype == 'MSE':
            self.loss = tnn.MSELoss()
        elif losstype == 'L1':
            self.loss = tnn.L1Loss()
        elif losstype == 'ssim':
            self.loss = SSIM3D()
        elif losstype == 'ssim_dist':
            self.loss = SSIM3D(distance=2)
        elif losstype == 'BCE':
            self.loss = tnn.BCELoss()
        elif losstype == 'BCElogit':
            self.loss = tnn.BCEWithLogitsLoss()

        if self.cuda:
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            device = "cuda"
        else: device = 'cpu'

        self.log.info(summary(self.model, (1, in_size[0], in_size[1], in_size[2]), device=device, batch_size=1))

        self.ep_start = load_existing_weights_if_exist(self.res_dir, self.model, log=self.log, device=device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)


    def train_regress_motion(self):

        for ep in range(self.ep_start, self.max_epochs + self.ep_start):
            self.model.train()

            # exp_lr_scheduler.step() #to change learning rate ... ?
            epoch_samples, epoch_loss, sliding_loss = 0, 0, 0
            res, extra_info = pd.DataFrame(), dict()

            for iteration, data in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                inputs = data['image']['data']

                if self.patch: #compute ssim for the patch
                    inputs_orig = data['image_orig']['data']
                    if self.cuda:
                        inputs, inputs_orig = inputs.cuda(), inputs_orig.cuda()
                    labels = ssim3D(inputs, inputs_orig, verbose=self.verbose)
                    labels = labels.unsqueeze(1)
                    extra_info['ssim_patch'] = labels.squeeze().cpu().detach()

                else:
                    labels = data['image']['metrics']['ssim'].unsqueeze(1)
                    if self.cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()

                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    l_tmp = self.loss(outputs, labels)
                    epoch_samples += 1 #inputs.size(0)
                    epoch_loss += l_tmp.item()
                    sliding_loss += l_tmp.item()

                    l_tmp.backward()
                    self.optimizer.step()

                extra_info['model_out'] = outputs.squeeze().cpu().detach()
                res = self.add_motion_info(data, res, extra_info)

                if (iteration==10) :
                    self.log.info("Ep: {} Iteration: {} Loss: {} mean10 {} mean {}".format(
                        ep, iteration, l_tmp.item(), sliding_loss/10, epoch_loss / epoch_samples))

                if (iteration % 100 == 0) :
                    self.log.info("Ep: {} Iteration: {} Loss: {} mean100 {} mean {}".format(
                        ep, iteration, l_tmp.item(), sliding_loss/100 ,epoch_loss / epoch_samples))
                    sliding_loss = 0

                if (iteration % 500 == 0) :
                    resname = self.res_dir + "/model_ep{}_it{}.pt".format(ep,iteration)
                    torch.save({"model": self.model.state_dict()}, resname)
                    self.log.info('saving model to %s' % (resname))

            self.log.info("Ep: {} Iteration: {} Loss: {} mean {}".format(ep, iteration, l_tmp.item(),
                                                                         epoch_loss / epoch_samples))
            fres = self.res_dir + '/res_train_ep{:02d}.csv'.format(ep)
            res.to_csv(fres)

            if ep % 4 == 0:
                resname = self.res_dir + "/model_ep{}.pt".format(ep)
                torch.save({"model": self.model.state_dict()}, resname)
                self.log.info('saving model to %s' % (resname))

    def eval_regress_motion(self):

        self.model.eval()
        epoch_samples, epoch_loss = 0, 0
        res, extra_info = pd.DataFrame(), dict()
        ep = self.ep_start

        for iteration, data in enumerate(self.val_dataloader):
            inputs = data['image']['data']

            if self.patch: #compute ssim for the patch
                inputs_orig = data['image_orig']['data']
                if self.cuda:
                    inputs, inputs_orig = inputs.cuda(), inputs_orig.cuda()
                labels = ssim3D(inputs, inputs_orig, verbose=self.verbose)
                labels = labels.unsqueeze(1)
                extra_info['ssim_patch'] = labels.squeeze().cpu().detach()

            else:
                labels = data['image']['metrics']['ssim'].unsqueeze(1)
                if self.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

            with torch.no_grad():
                outputs = self.model(inputs)
                l_tmp = self.loss(outputs, labels)
                epoch_samples += 1  # inputs.size(0)
                epoch_loss += l_tmp.item()

            extra_info['model_out'] = outputs.squeeze().cpu().detach()
            res = self.add_motion_info(data, res, extra_info)

            if (iteration % 100 == 0) or (iteration==10) :
                self.log.info("Ep: {} Iteration: {} Loss: {} mean {}".format(ep, iteration, l_tmp.item(),
                                                                        epoch_loss / epoch_samples))

        self.log.info("Ep: {} Iteration: {} Loss: {} mean {}".format(ep, iteration, l_tmp.item(),
                                                                     epoch_loss / epoch_samples))
        fres = self.res_dir + '/res_val_ep{:02d}.csv'.format(ep)
        res.to_csv(fres)


    def add_motion_info(self, data, res, extra_info=None):

        batch_size = data['image']['data'].size(0)
        dicm = data['image']['metrics']
        dics = data['image']['simu_param']
        dicm.update(dics)
        if extra_info is not None:
            for k, v in extra_info.items():
                dicm[k] = v

        if 'index_ini' in data: dicm['index_patch'] = data['index_ini']
        dicm['fpath'] = data['image']['path']

        for nb_batch in range(0, batch_size):
            one_dict = dict()
            for key, vals in dicm.items():
                val = vals[nb_batch]
                if type(val) is list:
                    one_dict[key] = [x.numpy() for x in val]
                elif type(val) is str:
                    one_dict[key] = val
                else:
                    one_dict[key] = val.numpy()
            res = res.append(one_dict, ignore_index=True)

        return res


    def save_to_dir(self, res_dir):
        #self.nb_saved += 1 does not work with multiple dataloader
        from tqdm import tqdm

        if not os.path.isdir(res_dir): os.mkdir(res_dir)

        res = pd.DataFrame()

        for data in tqdm(self.train_dataloader):
            inputs = data['image']['data']
            res = self.add_motion_info(data, res)

        fres = self.res_dir + '/res_data_set.csv'
        res.to_csv(fres)
