import os
from torchio import ImagesDataset, Queue
from torchio.data import ImagesClassifDataset, get_subject_list_and_csv_info_from_data_prameters
#from torchio.data.sampler import ImageSampler
from torchio import INTENSITY, LABEL, Interpolation, Image, Subject

from torchio.transforms import RandomMotionFromTimeCourse, RandomElasticDeformation, RandomNoise, RandomAffineFFT, RandomAffine

from torch.utils.data import DataLoader
import torch.nn as tnn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.transforms import Compose
import torchvision

from utils_file import get_log_file, gfile, get_parent_path, gdir
#from utils import apply_conditions_on_dataset

from torchio.transforms.metrics import SSIM3D, ssim3D
from smallunet_pytorch import ConvN_FC3, SmallUnet, load_existing_weights_if_exist
from torch_summary import summary
from unet import UNet, UNet3D

import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import time, random
import socket


#from collections import defaultdict

class do_training():
    def __init__(self, res_dir, res_name='', verbose=False):

        self.res_name = res_name
        self.res_dir = res_dir
        if not os.path.isdir(res_dir): os.mkdir(res_dir)
        self.verbose = verbose
        #self.log_file = self.res_dir + '/training.log'
        myHostName = socket.gethostname()
        self.log_string = '\n working on {} \n'.format(myHostName)
        #self.log = get_log_file(self.log_file)
        self.patch = False

    def set_data_loader_from_file_list(self, fin, transforms=None, mask_key=None, mask_regex=None,
                                       batch_size=1, num_workers=0, shuffel=True):

        suj_list = get_subject_list_from_file_list(fin, mask_regex=mask_regex, mask_key=mask_key)

        if not isinstance(transforms, torchvision.transforms.transforms.Compose) and transforms is not None:
            transforms = Compose(transforms)

        train_dataset = ImagesDataset(suj_list, transform=transforms)

        self.train_dataset = train_dataset
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffel,
                                           num_workers=num_workers)
        self.val_dataloader = self.train_dataloader

    def set_data_loader(self, train_csv_file='', val_csv_file='', transforms=None,
                        batch_size=1, num_workers=0,
                        par_queue=None, save_to_dir=None, load_from_dir=None,
                        replicate_suj=0, shuffel_train=True,
                        get_condition_csv=None, get_condition_field='', get_condition_nb_wanted=1/4,
                        collate_fn=None, add_to_load=None, add_to_load_regexp=None ):

        if not isinstance(transforms, torchvision.transforms.transforms.Compose) and transforms is not None:
            transforms = Compose(transforms)

        if load_from_dir is not None :
            if type(load_from_dir) == str:
                load_from_dir = [load_from_dir, load_from_dir]
            fsample_train, fsample_val = gfile(load_from_dir[0], 'sample.*pt'), gfile(load_from_dir[1], 'sample.*pt')
            #random.shuffle(fsample_train)
            #fsample_train = fsample_train[0:10000]

            if get_condition_csv is not None:
                res = pd.read_csv(load_from_dir[0]+'/'+get_condition_csv)
                cond_val = res[get_condition_field].values

                y = np.linspace(np.min(cond_val), np.max(cond_val), 101)
                nb_wanted_per_interval = int(np.round(len(cond_val) * get_condition_nb_wanted / 100))
                y_select = []
                for i in range(len(y)-1):
                    indsel = np.where((cond_val > y[i]) & (cond_val < y[i+1]))[0]
                    nb_select = len(indsel)
                    if nb_select < nb_wanted_per_interval:
                        print(' only {} / {} for interval {} {:,.3f} |  {:,.3f} '.format(nb_select, nb_wanted_per_interval, i, y[i], y[i+1]))
                        y_select.append(indsel)
                    else:
                        pind = np.random.permutation(range(0,nb_select))
                        y_select.append(indsel[pind[0:nb_wanted_per_interval]])
                        #print('{} selecting {}'.format(i, len(y_select[-1])))
                ind_select = np.hstack(y_select)
                y = cond_val[ind_select]
                fsample_train = [fsample_train[ii] for ii in ind_select]
                self.log_string += '\nfinal selection {} soit {:,.3f} % instead of {:,.3f} %'.format(
                    len(y), len(y)/len(cond_val)*100, get_condition_nb_wanted*100)

                #conditions = [("MSE", ">", 0.0028),]
                #select_ind = apply_conditions_on_dataset(res,conditions)
                #fsel = [fsample_train[ii] for ii,jj in enumerate(select_ind) if jj]

            self.log_string += '\nloading {} train sample from {}'.format(len(fsample_train), load_from_dir[0])
            self.log_string += '\nloading {} val   sample from {}'.format(len(fsample_val), load_from_dir[1])
            train_dataset = ImagesDataset(fsample_train, load_from_dir=load_from_dir[0], transform=transforms,
                                          add_to_load=add_to_load, add_to_load_regexp=add_to_load_regexp)
            self.train_csv_load_file_train = fsample_train

            val_dataset = ImagesDataset(fsample_val, load_from_dir=load_from_dir[1], transform=transforms,
                                        add_to_load=add_to_load, add_to_load_regexp=add_to_load_regexp)
            self.train_csv_load_file_train = fsample_val

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

            train_dataset = ImagesDataset(paths_dict, transform=transforms, save_to_dir=save_to_dir)
            val_dataset = ImagesDataset(paths_dict_val, transform=transforms, save_to_dir=save_to_dir)

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

            self.train_dataloader = DataLoader(train_queue, batch_size=batch_size, shuffle=shuffel_train, collate_fn=collate_fn)
            self.val_dataloader = DataLoader(val_queue, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        else:
            self.train_dataset = train_dataset
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffel_train,
                                               num_workers=num_workers, collate_fn=collate_fn)
            self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, collate_fn=collate_fn)


    def set_model_from_file(self, file_name, cuda):

        resdir_name = get_parent_path(file_name, 2)[1]
        print('loading {} \nfrom dir {}'.format( get_parent_path(file_name)[1], resdir_name) )
        if 'ConvN_C16_256_Lin40_50' in resdir_name:
            conv_block, linear_block = [16, 32, 64, 128, 256], [40, 50]
            network_name = 'ConvN'
        else:
            raise('did not recognise model type')

        if 'L1' in resdir_name:
            losstype = 'L1'
        elif 'MSE' in resdir_name:
            losstype = 'MSE'

        if '_Size182' in resdir_name:
            in_size = [182, 218, 182]

        batch_norm = True if '_BN_' in resdir_name else False

        ind_drop = resdir_name.find('_D')
        substr = resdir_name[ind_drop+2:]
        ii = substr.find('_')
        dropout = float(substr[:ii])

        drop_conv = 0
        if '_DC' in resdir_name:
            ind_drop = resdir_name.find('_DC')
            substr = resdir_name[ind_drop+3:]
            ii = substr.find('_')
            drop_conv = float(substr[:ii])

        ind_lr = resdir_name.find('_lr')
        lr = float(resdir_name[ind_lr+3:])

        par_model = {'network_name': network_name,
                     'losstype': losstype,
                     'lr': lr,
                     'conv_block': conv_block, 'linear_block': linear_block,
                     'dropout': dropout, 'drop_conv': drop_conv, 'batch_norm': batch_norm,
                     'in_size': in_size,
                     'cuda': cuda, 'max_epochs': 1}
        self.set_model(par_model, res_model_file=file_name, verbose=False, log_filename='eval.log')


    def set_model(self, par_model, res_model_file=None, verbose=True, log_filename='training.log'):

        network_name = par_model['network_name']
        losstype = par_model['losstype']
        lr = par_model['lr']
        in_size = par_model['in_size']
        self.cuda = par_model['cuda']
        self.max_epochs = par_model['max_epochs']
        optim_name = par_model['optim'] if 'optim' in par_model else 'Adam'
        self.validation_droupout = par_model['validation_droupout'] if 'validation_droupout' in par_model else False

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
            dropout, drop_conv, batch_norm = par_model['dropout'], par_model['drop_conv'], par_model['batch_norm']
            linear_block = par_model['linear_block']
            output_fnc = par_model['output_fnc'] if 'output_fnc' in par_model else None
            self.model = ConvN_FC3(in_size=in_size, conv_block=conv_block, linear_block=linear_block,
                                   dropout=dropout, drop_conv=drop_conv, batch_norm=batch_norm, output_fnc=output_fnc)
            network_name += '_C{}_{}_Lin{}_{}_D{}_DC{}'.format(np.abs(conv_block[0]), conv_block[-1], linear_block[0],
                                                             linear_block[-1], dropout, drop_conv)
            if output_fnc is not None:
                network_name += '_fnc_{}'.format(output_fnc)
            if batch_norm:
                network_name += '_BN'
            if self.validation_droupout :
                network_name += '_VD'
        self.res_name += '_Size{}_{}_Loss_{}_lr{}'.format(in_size[0], network_name, losstype, lr)

        if 'Adam' not in optim_name: #only write if not default Adam
            self.res_name += '_{}'.format(optim_name)

        self.res_dir += self.res_name + '/'

        if res_model_file is not None:  #to avoid handeling batch size and num worker used for model training
            self.res_dir, self.res_name = get_parent_path(res_model_file)

        if not os.path.isdir(self.res_dir): os.mkdir(self.res_dir)

        self.log = get_log_file(self.res_dir + '/' + log_filename)
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

        if verbose:
            self.log.info(summary(self.model, (1, in_size[0], in_size[1], in_size[2]), device=device, batch_size=1))

        self.ep_start, self.last_model_saved = load_existing_weights_if_exist(self.res_dir, self.model, log=self.log,
                                                                              device=device, res_model_file=res_model_file)
        if "Adam" in optim_name:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif "SGD" in optim_name:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.5)

        #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    def get_inputs_labels_from_sample(self, data, target):

        if isinstance(data, list):  # case where callate_fn is used
            inputs = torch.cat([sample['image']['data'].unsqueeze(0) for sample in data])
        else:
            inputs = data['image']['data']

        if self.patch:  # compute ssim for the patch
            inputs_orig = data['image_orig']['data']
            if self.cuda:
                inputs, inputs_orig = inputs.cuda(), inputs_orig.cuda()
            labels = ssim3D(inputs, inputs_orig, verbose=self.verbose)
            labels = labels.unsqueeze(1)

        else:
            if target == 'ssim':
                labels = data['image']['metrics']['ssim'].unsqueeze(1)
            elif target == 'random_noise':
                lab=[]
                for sample in data:
                    historys = sample.history
                    for hh in historys: #len depend of number of transform
                        if 'RandomNoise' in hh:
                            lab.append( torch.tensor(hh[1]['image']['std']).unsqueeze(0) * 10  )
                labels = torch.cat(lab).unsqueeze(1)  #data['random_noise'].unsqueeze(1).float() * 10

            if self.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

        return inputs, labels


    def train_regress_motion(self, target='ssim'):
        max_iteration = len(self.train_dataloader)
        for ep in range(self.ep_start, self.max_epochs + self.ep_start):
            self.model.train()

            # exp_lr_scheduler.step() #to change learning rate ... ?
            epoch_samples, epoch_loss, sliding_loss = 0, 0, 0
            res, extra_info = pd.DataFrame(), dict()
            start = time.time()
            for iteration, data in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                inputs, labels = self.get_inputs_labels_from_sample(data, target)

                if self.patch:  # compute ssim for the patch
                    extra_info['ssim_patch'] = labels.squeeze().cpu().detach()

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
                    duration = (time.time() - start) / iteration * max_iteration / 60 / 60 #hours for on epochs
                    self.log.info("train start Ep: {} It: {} Loss: {} mean10 {} mean {}".format(
                        ep, iteration, l_tmp.item(), sliding_loss/10, epoch_loss / epoch_samples))
                    self.log.info(' estimate duration {:.2f} hours for one epoch '.format(duration))


                if (iteration % 100 == 0) and (iteration > 0) :
                    self.log.info("Train Ep: {} It: {} Loss: {} mean100 {} mean {}".format(
                        ep, iteration, l_tmp.item(), sliding_loss/100 ,epoch_loss / epoch_samples))
                    sliding_loss = 0
                    if (iteration == 100):
                        duration = (time.time() - start) / iteration * max_iteration / 60 / 60 #hours for on epochs
                        self.log.info(' estimate duration {:.2f} hours for one epoch '.format(duration))

                if (iteration % 500 == 0) and (iteration > 0):
                    if (iteration == 500):
                        duration = (time.time() - start) / iteration * max_iteration / 60 / 60 #hours for on epochs
                        self.log.info(' estimate duration {:.2f} hours for one epoch '.format(duration))
                    self.save_model(ep, iteration, fct_eval=self.eval_regress_motion, target=target)

            self.log.info("Train Ep: {} It {} Loss: {} mean {}".format(ep, iteration, l_tmp.item(),
                                                                         epoch_loss / epoch_samples))

            duration = (time.time() - start) / 60 / 60  # hours for on epochs
            self.log.info(' performed duration {:.2f} hours for one epoch '.format(duration))

            fres = self.res_dir + '/res_train_ep{:02d}.csv'.format(ep)
            res.to_csv(fres)

            if ep % 4 == 0:
                self.save_model(ep, iteration, fct_eval=self.eval_regress_motion, target=target)

        self.save_model(ep, iteration, fct_eval=self.eval_regress_motion, target=target)

    def save_model(self, ep, iteration=None, fct_eval=None, target='ssim'):
        if iteration is not None:
            resname = "model_ep{}_it{}.pt".format(ep, iteration)
        else:
            resname = "model_ep{}.pt".format(ep)

        torch.save({"model": self.model.state_dict()}, self.res_dir + resname)
        self.log.info('saving model to %s' % (resname))
        self.last_model_saved = resname
        if fct_eval is not None:
            fct_eval(ep, iteration, target=target)
            self.model.train()

    def eval_regress_motion(self, epTrain, iterationTrain, target='ssim',
                            basename='res_val', subdir=None):
        start = time.time()
        self.model.eval()
        if self.validation_droupout:
            self.model.enable_dropout()

        epoch_samples, epoch_loss = 0, 0
        res, extra_info = pd.DataFrame(), dict()

        for iteration, data in enumerate(self.val_dataloader):

            inputs, labels = self.get_inputs_labels_from_sample(data, target)

            with torch.no_grad():
                outputs = self.model(inputs)
                if labels is None:
                    labels = outputs

                l_tmp = self.loss(outputs, labels)
                epoch_samples += 1  # inputs.size(0)
                epoch_loss += l_tmp.item()

            extra_info['model_out'] = outputs.squeeze().cpu().detach()
            res = self.add_motion_info(data, res, extra_info)

            if (iteration % 100 == 0) and (iteration > 0):
                self.log.info("VAL data Ep_it: {}_{} It {} Loss: {} mean {}".format(epTrain, iterationTrain, iteration, l_tmp.item(),
                                                                        epoch_loss / epoch_samples))

        self.log.info("VAL data Ep_it: {}_{} It {} Loss: {} mean {}".format(epTrain, iterationTrain, iteration, l_tmp.item(),
                                                                epoch_loss / epoch_samples))
        duration = (time.time() - start) / 60 / 60
        self.log.info(' validation duration for {} iter {:.2f} hours {:.2f} mn '.format(iteration,duration, duration*60))

        fres = self.res_dir + '/{}_{}.csv'.format(basename, self.last_model_saved[:-3])
        if subdir is not None:
            fres = self.res_dir + '/' + subdir
            if not os.path.isdir(fres): os.mkdir(fres)
            fres += '/{}_{}.csv'.format(basename, self.last_model_saved)

        res.to_csv(fres)


    def eval_multiple_transform(self, epTrain, iterationTrain, target='ssim', basename='res_val', subdir=None,
                                transform_list=None, transform_list_name=None):
        start = time.time()
        self.model.eval()
        if self.validation_droupout:
            self.model.enable_dropout()

        epoch_samples, epoch_loss = 0, 0
        res, extra_info = pd.DataFrame(), dict()

        #transform_list = self.eval_transform_list
        #transform_list_name = self.eval_transform_list_name

        for iteration, data in enumerate(self.val_dataloader):

            inputs, labels = self.get_inputs_labels_from_sample(data, target)

            with torch.no_grad():
                outputs = self.model(inputs)
                if labels is None:
                    labels = outputs

                l_tmp = self.loss(outputs, labels)
                epoch_samples += 1  # inputs.size(0)
                epoch_loss += l_tmp.item()

            extra_info['model_out'] = outputs.squeeze().cpu().detach()
            for trans, trans_name in zip(transform_list, transform_list_name):
                tinputs = torch.empty(inputs.shape, dtype=torch.float)

                #arge c'est pas le bon endroit si les transform sont en cpu grrr should be handel in data handeling
                for ii in range(inputs.shape[0]):
                    data_n = inputs[ii].cpu().detach() if self.cuda else inputs[ii]
                    tinputs[ii] = trans(data_n)
                tttinputs = tinputs.cuda() if self.cuda else tinputs

                with torch.no_grad():
                    outputs = self.model(tttinputs)

                extra_info[trans_name + 'model_out'] = outputs.squeeze().cpu().detach()

            res = self.add_motion_info(data, res, extra_info)

            if (iteration % 100 == 0) and (iteration > 0):
                self.log.info("VAL data Ep_it: {}_{} It {} Loss: {} mean {}".format(epTrain, iterationTrain, iteration, l_tmp.item(),
                                                                        epoch_loss / epoch_samples))

        self.log.info("VAL data Ep_it: {}_{} It {} Loss: {} mean {}".format(epTrain, iterationTrain, iteration, l_tmp.item(),
                                                                epoch_loss / epoch_samples))
        duration = (time.time() - start) / 60 / 60
        self.log.info(' validation duration for {} iter {:.2f} hours {:.2f} mn '.format(iteration,duration, duration*60))

        fres = self.res_dir + '/{}_{}.csv'.format(basename, self.last_model_saved[:-3])
        if subdir is not None:
            fres = self.res_dir + '/' + subdir
            if not os.path.isdir(fres): os.mkdir(fres)
            fres += '/{}_{}.csv'.format(basename, self.last_model_saved)

        res.to_csv(fres)


    def add_motion_info(self, data, res, extra_info=None):

        if isinstance(data, list):  # case where callate_fn is used
            batch_size = len(data)

            for ii, sample in enumerate(data):
                one_dict = dict()
                historys = sample.history
                for hh in historys: #len depend of number of transform
                    if 'RandomNoise' in hh:
                        one_dict['random_noise'] = hh[1]['image']['std']
                    if 'RandomAffine' in hh: #if 'RandomAffineFFT' in hh:
                        one_dict.update(hh[1])

                if extra_info is not None:
                    for k, v in extra_info.items():
                        one_dict[k] = v[ii].numpy() if torch.is_tensor(v[ii]) else v[ii]

                one_dict['fpath'] = sample['image']['path']
                res = res.append(one_dict, ignore_index=True)

        else:
            if data['image']['data'].ndim == 4:  # no batch
                batch_size = 0
            else:
                batch_size = data['image']['data'].size(0)

            dicm = {}
            if 'metrics' in data['image']:
                dicm = data['image']['metrics']
                dics = data['image']['simu_param']
                dicm.update(dics)

            if 'random_noise' in data:
                dicm['random_noise'] = data['random_noise']

            if extra_info is not None:
                for k, v in extra_info.items():
                    dicm[k] = v

            if 'index_ini' in data: dicm['index_patch'] = data['index_ini']
            if 'mvt_csv' in data: dicm['mvt_csv'] = data['mvt_csv']
            dicm['fpath'] = data['image']['path']

            if batch_size == 0:
                res = res.append(dicm, ignore_index=True)
            else:
                for nb_batch in range(0, batch_size):
                    one_dict = dict()
                    for key, vals in dicm.items():
                        if isinstance(vals, list):
                            val = vals[nb_batch]
                        elif isinstance(vals, str):
                            val = vals
                        else:
                            val = vals[nb_batch] if len(vals.size()) > 0 else vals

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

def get_motion_transform(type='motion1'):
    if 'motion1' in type:
        from torchio.metrics import SSIM3D, MetricWrapper, MapMetricWrapper
        from torchio.metrics.ssim import functional_ssim
        from torch.nn import MSELoss, L1Loss


        metrics = {
            "L1": MetricWrapper("L1", L1Loss()),
            "L1_map": MapMetricWrapper("L1_map", lambda x, y: torch.abs(x - y), average_method="mean",
                                       mask_keys=['brain']),
            "L2": MetricWrapper("L2", MSELoss()),
            # "SSIM": SSIM3D(average_method="mean"),
            "SSIM_mask": SSIM3D(average_method="mean", mask_keys=["brain"]),
            "SSIM_Wrapped": MetricWrapper("SSIM_wrapped", lambda x, y: functional_ssim(x, y, return_map=False),
                                          use_mask=True, mask_key="brain"),
            "ssim_base": MapMetricWrapper('SSIM_base', lambda x, y: ssim3D(x, y, size_average=True), average_method="mean",
                                          mask_keys=['brain'])
        }

        dico_params_mot = {"maxDisp": (1, 6), "maxRot": (1, 6), "noiseBasePars": (5, 20, 0.8),
                       "swallowFrequency": (2, 6, 0.5), "swallowMagnitude": (3, 6),
                       "suddenFrequency": (2, 6, 0.5), "suddenMagnitude": (3, 6),
                       "verbose": False, "proba_to_augment": 1,
                       "preserve_center_pct": 0.1, "compare_to_original": True,
                       "oversampling_pct": 0, "correct_motion": False}

        dico_params_mot = {"maxDisp": (1, 4), "maxRot": (1, 4), "noiseBasePars": (5, 20, 0.8),
                       "swallowFrequency": (2, 6, 0.5), "swallowMagnitude": (3, 4),
                       "suddenFrequency": (2, 6, 0.5), "suddenMagnitude": (3, 4),
                       "verbose": False, "proba_to_augment": 1,
                       "preserve_center_pct": 0.1, "compare_to_original": True, "metrics": metrics,
                       "oversampling_pct": 0, "correct_motion": False}

    if 'elastic1' in type:
        dico_elast = { 'num_control_points': 6, 'max_displacement': (30, 30, 30),
           'p': 1, 'image_interpolation': Interpolation.LINEAR }

    if type == 'motion1':
        transforms = Compose([ RandomMotionFromTimeCourse(**dico_params_mot),])

    if type == 'elastic1':
        transforms = Compose([ RandomElasticDeformation(**dico_elast),])

    elif type == 'elastic1_and_motion1':
        transforms = Compose([ RandomElasticDeformation(**dico_elast),
                               RandomMotionFromTimeCourse(**dico_params_mot) ] )
    if type == 'random_noise_1':
        transforms = Compose([RandomNoise(std=(0.020, 0.2))])

    if type == 'AffFFT_random_noise':
        transforms = Compose([RandomAffineFFT(scales=(0.8, 1.2), degrees=10, oversampling_pct=0.2, p=0.75),
                               RandomNoise(std=(0.020, 0.2))])
    if type == 'AffFFT_random_noise':
        transforms = Compose([RandomAffine(scales=(0.8, 1.2), degrees=10, p=0.75, image_interpolation=Interpolation.NEAREST),
                              RandomNoise(std=(0.020, 0.2))])

    return transforms

def get_cache_dir(root_fs = 'lustre'):
    if root_fs == 'lustre':
        #dir_cache = '/network/lustre/dtlake01/opendata/data/ds000030/rrr/CNN_cache/'
        dir_cache = '/network/lustre/dtlake01/opendata/data/ds000030/rrr/CNN_cache_new/'
    elif root_fs == 'le70':
        dir_cache = '/data/romain/CNN_cache/'
    return dir_cache

def get_train_and_val_csv(names='', root_fs = 'lustre'):

    return_list = True
    if isinstance(names, str):
        names = [names]
        return_list = False

    print('name is {}'.format(type(names)))

    if root_fs == 'lustre':
        data_path_hcp = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/'
    elif root_fs == 'le70':
        data_path_hcp = '/data/romain/HCPdata/'
    data_path_cati = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/CATI_datasets/'

    fcsv_train, fcsv_val = [], []
    for name in names:
        if 'hcp' in name:
            if 'T1' in name:
                fname_train, fname_val = 'Motion_T1_train_hcp400.csv',  'Motion_T1_val_hcp200.csv'
            elif 'brain_ms' in name:
                fname_train, fname_val = 'healthy_brain_ms_train_hcp400.csv', 'healthy_brain_ms_val_hcp200.csv'
            elif 'ms' in name:
                fname_train, fname_val = 'healthy_ms_train_hcp400.csv', 'healthy_ms_val_hcp200.csv'
            else:
                print('can not guess which DATA from {}'.format(name))
                raise

            file_train = data_path_hcp + fname_train
            file_val   = data_path_hcp + fname_val

        elif 'cati' in name:
            if 'T1' in name:
                fname_train, fname_val = 'cati_cenir_QC4_train_T1.csv',  'cati_cenir_QC4_val_T1.csv'
            elif 'brain' in name:
                fname_train, fname_val = 'cati_cenir_QC4_train_brain.csv',  'cati_cenir_QC4_val_brain.csv'
            elif 'i_ms' in name:
                fname_train, fname_val = 'cati_cenir_QC4_train_ms.csv',  'cati_cenir_QC4_val_ms.csv'
            else:
                print('can not guess which DATA from {}'.format(name))
                raise

            file_train = data_path_cati + fname_train
            file_val   = data_path_cati + fname_val

        else:
            print('can not guess which DATA from {}'.format(name))
            raise

        print('data {}\nfor {} \t found {} {} '.format(get_parent_path([file_train])[0][0], name, fname_train, fname_val))
        fcsv_train.append(file_train)
        fcsv_val.append(file_val)

    if return_list:
        return fcsv_train, fcsv_val
    else:
        return fcsv_train[0], fcsv_val[0]

def write_cati_csv():
    import pandas as pd
    data_path = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/CATI_datasets/'
    fcsv = data_path + 'all_cati.csv';
    res = pd.read_csv(fcsv)

    ser_dir = res.cenir_QC_path
    ser_dir = res.cenir_QC_path[res.globalQualitative > 3].values
    dcat = gdir(ser_dir, 'cat12')
    fT1 = gfile(dcat, '^s.*nii')
    fms = gfile(dcat, '^ms.*nii')
    fs_brain = gfile(dcat, '^brain_s.*nii')
    # return fT1, fms, fs_brain

    ind_perm = np.random.permutation(range(0, len(fT1)))
    itrain = ind_perm[0:100]
    ival = ind_perm[100:]

    dd = pd.DataFrame({'filename': fT1})
    dd.to_csv(data_path + 'cati_cenir_QC4_all_T1.csv', index=False)
    dd.loc[ival, :].to_csv(data_path + 'cati_cenir_QC4_val_T1.csv', index=False)
    dd.loc[itrain, :].to_csv(data_path + 'cati_cenir_QC4_train_T1.csv', index=False)

    dd = pd.DataFrame({'filename': fms})
    dd.to_csv(data_path + 'cati_cenir_QC4_all_ms.csv', index=False)
    dd.loc[ival, :].to_csv(data_path + 'cati_cenir_QC4_val_ms.csv', index=False)
    dd.loc[itrain, :].to_csv(data_path + 'cati_cenir_QC4_train_ms.csv', index=False)

    dd = pd.DataFrame({'filename': fs_brain})
    dd.to_csv(data_path + 'cati_cenir_QC4_all_brain.csv', index=False)
    dd.loc[ival, :].to_csv(data_path + 'cati_cenir_QC4_val_brain.csv', index=False)
    dd.loc[itrain, :].to_csv(data_path + 'cati_cenir_QC4_train_brain.csv', index=False)


    dd = pd.DataFrame({'filename': fT1})
    dd.to_csv(data_path + 'cati_cenir_all_T1.csv', index=False)
    dd = pd.DataFrame({'filename': fms})
    dd.to_csv(data_path + 'cati_cenir_all_ms.csv', index=False)
    dd = pd.DataFrame({'filename': fs_brain})
    dd.to_csv(data_path + 'cati_cenir_all_brain.csv', index=False)

    #add brain mask in csv
    allcsv = gfile('/home/romain.valabregue/datal/QCcnn/CATI_datasets','cati_cenir.*csv')

    for onecsv in allcsv:
        res = pd.read_csv(onecsv)
        resout = onecsv[:-4] + '_mask.csv'
        fmask=[]
        for ft1 in res.filename:
            d = get_parent_path(ft1)[0]
            fmask += gfile(d,'^mask',opts={"items":1})
        res['brain_mask'] = fmask
        res.to_csv(resout, index=False)




def get_subject_list_from_file_list(fin, mask_regex=None, mask_key='brain'):
    subjects_list=[]
    for ff in fin:
        one_suj = {'image': Image(ff, INTENSITY)}
        if mask_regex is not None:
            dir_file = get_parent_path(ff)[0]
            fmask = gfile(dir_file, mask_regex, {"items": 1})
            one_suj[mask_key] = Image(fmask[0], LABEL)

        subjects_list.append(Subject(one_suj))

    return subjects_list
