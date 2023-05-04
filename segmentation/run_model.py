""" Run model and save it """

import json
import torch
import time
import pickle
import logging
import glob
import os, tarfile
import numpy as np
import pandas as pd
import nibabel as nib
import torchio
import torchio as tio
import resource
import warnings
from torch.utils.data import DataLoader
from segmentation.utils import to_var, summary, save_checkpoint, to_numpy, get_largest_connected_component
from apex import amp
from torch.utils.tensorboard import SummaryWriter


class ArrayTensorJSONEncoder(json.JSONEncoder):
    """
    JSONEncoder extension to be able to stringify NumPy arrays and Torch
    tensors.
    """
    def default(self, o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()
        elif isinstance(o, float):
            return str(o)
        else:
            return json.JSONEncoder.default(self, o)


class RunModel:
    """
    Handle training, evaluation and saving of a model from a json
    configuration file.
    """
    def __init__(self, model, device, train_loader, val_loader, val_set,
                 test_set, image_key_name, label_key_name, labels,
                 logger, debug_logger, results_dir, batch_size,
                 patch_size, struct, post_transforms, model_name=None):
        self.model = model
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_set = val_set
        self.test_set = test_set

        self.image_key_name = image_key_name
        self.label_key_name = label_key_name
        self.labels = labels

        self.logger = logger
        self.debug_logger = debug_logger
        self.results_dir = results_dir
        self.log_on_tensorboard = struct['log_on_tensorboard']
        if self.log_on_tensorboard is True:
            self.logs_dir = results_dir + '/tb_logs'
            self.create_folder_if_not_exists(self.logs_dir)
            self.tb_logger = SummaryWriter(self.logs_dir)
            self.tb_df_train = pd.DataFrame()
            self.tb_df_val = pd.DataFrame()

        self.batch_size = batch_size
        self.patch_size = patch_size

        self.post_transforms = post_transforms

        # Set attributes to keep track of information during training
        self.epoch = struct['current_epoch']
        self.iteration = 0
        self.val_iteration = 0

        # Retrieve information from structure
        self.criteria = struct['criteria']
        self.log_frequency = struct['log_frequency']
        self.record_frequency = struct['save']['record_frequency']
        self.eval_frequency = struct['validation']['eval_frequency']
        self.whole_image_inference_frequency = struct['validation'][
            'whole_image_inference_frequency']
        self.metrics = struct['validation']['reporting_metrics']
        self.patch_overlap = struct['validation']['patch_overlap']
        self.save_predictions = struct['validation']['save_predictions']
        self.eval_results_dir = struct['validation']['eval_results_dir']
        self.dense_patch_eval = struct['validation']['dense_patch_eval']
        self.eval_patch_size = struct['validation']['eval_patch_size']
        self.save_labels = struct['validation']['save_labels']
        self.save_data = struct['validation']['save_data']
        self.eval_dropout = struct['validation']['eval_dropout']
        self.split_batch_gpu = struct['validation']['split_batch_gpu']
        self.eval_repeate =  struct['validation']['repeate_eval']
        self.n_epochs = struct['n_epochs']
        self.seed = struct['seed']
        self.activation = struct['activation']
        self.save_bin = struct['save']['save_bin']
        self.split_channels = struct['save']['split_channels']
        self.save_channels = struct['save']['save_channels']
        self.save_threshold = struct['save']['save_threshold']
        self.save_biggest_comp = struct['save']['save_biggest_comp']
        self.save_volume_name = struct['save']['save_volume_name']
        self.save_label_name = struct['save']['save_label_name']
        self.save_struct = struct['save']
        self.apex_opt_level = struct['apex_opt_level']
        self.no_blocking = struct['no_blocking']

        # Keep information to load optimizer and learning rate scheduler
        self.optimizer, self.lr_scheduler = None, None
        self.optimizer_dict = struct['optimizer']

        # Define which methods will be used to retrieve data and record
        # information
        function_datagetter = getattr(self, struct['data_getter']['name'])
        attributes = struct['data_getter']['attributes']
        self.data_getter = lambda sample: function_datagetter(
            sample, **attributes)
        self.batch_recorder = struct['save']['batch_recorder']
        if self.batch_recorder is not None:
            self.batch_recorder = getattr(self, struct['save']['batch_recorder'])
        self.prediction_saver = getattr(
            self, struct['save']['prediction_saver'])
        self.label_saver = getattr(self, struct['save']['label_saver'])

        self.eval_csv_basename = None
        self.save_transformed_samples = False
        self.model_name = model_name

    @staticmethod
    def create_folder_if_not_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def log(self, info):
        if self.logger is not None:
            self.logger.log(logging.INFO, info)

    def debug(self, info):
        if self.debug_logger is not None:
            self.debug_logger.log(logging.DEBUG, info)

    def log_peak_CPU_memory(self):
        # Get max CPU memory usage
        main_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        child_memory = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        self.log('******** CPU Memory Usage  ********')
        self.log(f'Peak: {main_memory + child_memory} kB')

    def get_optimizer(self, optimizer_dict):
        optimizer_dict['attributes'].update({'params': list(self.model.parameters())})
        for cc in self.criteria:
            if cc['criterion'].additional_learned_param is not None:
                #optimizer.add_param_group({'params': cc['criterion'].additional_learned_param})
                #print(f"is leaf {cc['criterion'].additional_learned_param.is_leaf}")
                optimizer_dict['attributes']['params'] += [cc['criterion'].additional_learned_param]

        optimizer = optimizer_dict['optimizer_class'](**optimizer_dict['attributes'])

        scheduler = None
        if optimizer_dict['lr_scheduler'] is not None:
            optimizer_dict['lr_scheduler']['attributes'].update(
                {'optimizer': optimizer}
            )
            scheduler = optimizer_dict['lr_scheduler']['class'](
                **optimizer_dict['lr_scheduler']['attributes']
            )

        #in nn unet they use
            # self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
              #                           momentum=0.99, nesterov=True)

        return optimizer, scheduler

    def get_segmentation_data(self, sample):
        volumes = sample[self.image_key_name]
        volumes = to_var(volumes[torchio.DATA].float(), self.device, self.no_blocking)

        targets = None
        if self.label_key_name in sample:
            targets = sample[self.label_key_name]
            targets = to_var(targets[torchio.DATA].float(), self.device, self.no_blocking)
        return volumes, targets

    def get_segmentation_data_and_regress_key(self, sample, regress_key):
        volumes = sample[self.image_key_name]
        volumes = to_var(volumes[torchio.DATA].float(), self.device, self.no_blocking)

        targets = None
        if self.label_key_name in sample:
            targets = sample[self.label_key_name]
            targets = to_var(targets[torchio.DATA].float(), self.device, self.no_blocking)
        if regress_key in sample:
            targets_to_regress = to_var(sample[regress_key][torchio.DATA].float(), self.device, self.no_blocking)
            targets = torch.cat((targets, targets_to_regress), dim=1)
        return volumes, targets

    def train(self):
        def get_loader_from_arg(arg_dict):
            #add for data set with a lot of files, (previously list of dataloader ... with load_from_dir)
            sample_files = eval(arg_dict['subjects'])
            train_set =  torchio.SubjectsDataset(sample_files,
                                              load_from_dir=arg_dict['load_from_dir'],
                                              transform=arg_dict['transform'],
                                              add_to_load=arg_dict['add_to_load'],
                                              add_to_load_regexp=arg_dict['add_to_load_regexp'])

            train_queue = torchio.Queue(train_set,  start_background=True,
                                        **arg_dict['queue_param'] )

            dataloader = DataLoader(train_queue, self.batch_size, num_workers=0,
                                           collate_fn=arg_dict['collate_fn'])

            return dataloader
        """ Train the model on the training set and evaluate it on
        the validation set. """
        # Set seed for reproducibility
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Get optimizer and scheduler
        self.optimizer, self.lr_scheduler = self.get_optimizer(
            self.optimizer_dict)

        # Initialize Apex
        if self.apex_opt_level is not None:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.apex_opt_level
            )

        # Try to load optimizer state
        opt_files = glob.glob(
            os.path.join(self.results_dir, f'opt_ep{self.epoch - 1}*.pth.tar')
        )
        if len(opt_files) > 0:
            self.log('RRR WARNING hard coded skip of optimize load')
            #self.log(f'loading optimizer from {opt_files[-1]}')
            #self.optimizer.load_state_dict(torch.load(opt_files[-1]))

        # Try to load scheduler state
        if self.lr_scheduler is not None and self.val_loader is None:
            warnings.warn('A learning rate scheduler is set but there is no'
                          'validation set, removing scheduler.')
            self.lr_scheduler = None
        if self.lr_scheduler is not None:
            sch_files = glob.glob(
                os.path.join(self.results_dir,
                             f'sch_ep{self.epoch - 1}*.pth.tar')
            )
            if len(sch_files) > 0:
                self.lr_scheduler.load_state_dict(torch.load(sch_files[-1]))
                self.log(f'loading shedulder from {sch_files[-1]}')
            else:
                self.log(f'warning no shedulder file found to load from ... ')
        else:
            self.log('No shedulder load from file')

        # Try to load Apex
        if self.apex_opt_level is not None:
            amp_files = glob.glob(
                os.path.join(
                    self.results_dir,
                    f'amp_ep{self.epoch - 1}*.pth.tar'
                )
            )
            if len(amp_files) > 0:
                amp.load_state_dict(torch.load(amp_files[-1]))
            self.log(f'working with Apex level {self.apex_opt_level}')
        else:
            self.log('No apex optim')

        if isinstance(self.train_loader, list):
            dataloader_list = self.train_loader
            starting_epoch = self.epoch
            self.n_epochs = len(dataloader_list)

            #for epoch, train_loader in enumerate(dataloader_list):
            for i in range(0, starting_epoch-1):  #first tentative to remove memory but not enoug
                # no more usefull since now the dataloader is build for the current epoch

                self.log(f'droping ep {i+1}')
                dataloader_list.pop(0)

            for ind_epoch in  range(starting_epoch-1, self.n_epochs):
                self.epoch = ind_epoch + 1
                self.log(f'******** Epoch from dir [{self.epoch}/{self.n_epochs}]  ********')

                #train_loader = dataloader_list[ind_epoch]
                train_loader = get_loader_from_arg(dataloader_list[0])

                if isinstance(train_loader.dataset,tio.data.queue.Queue):
                    self.log(f'first data is {train_loader.dataset.subjects_dataset._subjects[0]}')
                else:
                    self.log(f'first data is {train_loader.dataset._subjects[0]}')

                #not sure if need to set here because it is the "main' loader optim are whihtin queue dataloader
                # train_loader.pin_memory=True
                self.train_loader = train_loader
                # Train for one epoch
                self.model.train()
                self.train_loop()

                # Evaluate on whole images of the validation set
                with torch.no_grad():
                    self.model.eval()
                    if self.patch_size is not None \
                            and self.whole_image_inference_frequency is not None:
                        if self.epoch % self.whole_image_inference_frequency == 0:
                            self.log('Validation on whole images')
                            self.whole_image_evaluation_loop()

                            # Save model after inference
                            self.save_checkpoint()

                # Log memory consumption
                self.log_peak_CPU_memory()
                del train_loader
                dataloader_list.pop(0)

        else:

            for epoch in range(self.epoch, self.n_epochs + 1):
                self.epoch = epoch
                self.log(f'******** Epoch [{self.epoch}/{self.n_epochs}]  ********')

                # Train for one epoch
                self.model.train()
                self.train_loop()

                # Evaluate on whole images of the validation set
                with torch.no_grad():
                    self.model.eval()
                    if self.patch_size is not None \
                            and self.whole_image_inference_frequency is not None:
                        if self.epoch % self.whole_image_inference_frequency == 0:
                            self.log('Validation on whole images')
                            self.whole_image_evaluation_loop()

                            # Save model after inference
                            self.save_checkpoint()

                # Log memory consumption
                self.log_peak_CPU_memory()

        # Save model at the end of training
        self.save_checkpoint()

    def eval(self, eval_csv_basename=None, save_transformed_samples=False):
        """ Evaluate the model on the validation set. """
        self.epoch -= 1
        if eval_csv_basename:
            self.eval_csv_basename = eval_csv_basename

        self.log('Evaluation mode')
        self.log_peak_CPU_memory()
        for nb_eval in range(self.eval_repeate):
            with torch.no_grad():
                if isinstance(self.model, list):
                    print(f'List model size {len(self.model)}')
                    for model in self.model:
                        model.eval()
                else:
                    self.model.eval()
                if nb_eval > 0:
                    self.eval_csv_basename = f'V{nb_eval}'+ self.eval_csv_basename if self.eval_csv_basename else f'V{nb_eval}'
                patch_size = self.patch_size or self.eval_patch_size
                if self.eval_frequency is not None:
                    if patch_size is not None:
                        self.log('Evaluation on patches')
                    self.train_loop(save_model=False)

                do_whole_image_loop = self.dense_patch_eval \
                    or self.whole_image_inference_frequency is not None
                if patch_size is not None and do_whole_image_loop:
                    self.log('Evaluation on whole images')
                    self.whole_image_evaluation_loop(save_transformed_samples)

        # Log memory consumption
        self.log_peak_CPU_memory()
        self.log('Evaluation mode')

    def infer(self):
        """ Use the model to make predictions on the test set. """
        self.epoch -= 1
        self.log('Inference')
        with torch.no_grad():
            self.model.eval()
            self.inference_loop()

        # Log memory consumption
        self.log_peak_CPU_memory()

    def train_loop(self, save_model=True):
        eval_dropout = 0
        is_model_training = self.model[0].training if isinstance(self.model, list) else self.model.training
        if is_model_training:
            self.log('Training')
            model_mode = 'Train'
            loader = self.train_loader
        else:
            self.log('Validation')
            if self.eval_dropout:
                eval_dropout = self.eval_dropout
            model_mode = 'Val'
            loader = self.val_loader
            if loader is None:
                # Save model after an evaluation on the whole validation set
                if save_model:
                    self.save_checkpoint(11)
                return 1

        df = pd.DataFrame()
        start = time.time()
        time_sum, loss_sum, reporting_time_sum = 0, 0, 0
        average_loss, max_loss = None, None

        for i, sample in enumerate(loader, 1):
            if is_model_training:
                self.iteration = i
            else:
                self.val_iteration = i
            if isinstance(sample, list):
                self.batch_size = self.batch_size * len(sample)
                for ii, ss in enumerate(sample):
                    ss['list_idx'] = ii #add this key to use in save_volume (having different volume name)
                sample = self.sample_list_to_batch(sample)

            # Take variables and make sure they are tensors on the right device
            volumes, targets = self.data_getter(sample)

            hack_plot=False
            if hack_plot:
                import matplotlib.pyplot as plt
                dv = volumes.cpu().numpy()
                fig, axs = plt.subplots(self.batch_size,3)
                fig.set_size_inches(4.8, 6.4) #batch size 4 (so 4.8 4.8*4/3)
                axs = axs.flatten()
                nimg=-1
                for ii, ax in enumerate(axs):
                    axnum = ii%3
                    if axnum==0 :
                        nimg +=1
                        slice = dv[nimg,0, 64, :, :]
                    elif axnum == 1:
                        slice = dv[nimg, 0, :, 64, :]
                    elif axnum == 2:
                        slice = dv[nimg,0, :, :, 64]
                    ax.imshow(slice.T, origin='lower' )
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                plt.tight_layout()
                plt.savefig(f'{self.results_dir}/fig_it{i}.png' )
                if i>10 :
                    qsdfstop
                continue

            # Compute output
            if eval_dropout:
                self.model.enable_dropout()
                if self.split_batch_gpu and self.batch_size > 1:
                    predictions_dropout = [torch.cat([self.activation(self.model(v.unsqueeze(0))) for v in volumes])
                                           for ii in range(0, eval_dropout)]
                else:
                    predictions_dropout = [self.activation(self.model(volumes)) for ii in range(0, eval_dropout)]
                predictions = torch.mean(torch.stack(predictions_dropout), axis=0)
            else:
                if self.split_batch_gpu and self.batch_size > 1: #usefull, when using ListOf transform that pull sample in batch to avoid big full volume batch in gpu
                    predictions = torch.cat([self.activation(self.model(v.unsqueeze(0))) for v in volumes])
                else:
                    if isinstance(self.model, list):
                        predictions = [self.activation(mmm(volumes)) for mmm in self.model]
                    else:
                        azer
                        predictions = self.activation(self.model(volumes))

            if eval_dropout:
                predictions_dropout = [self.apply_post_transforms(pp, sample)[0] for pp in predictions_dropout]
                predictions = torch.mean(torch.stack(predictions_dropout), axis=0)
            else:
                if isinstance(predictions, list):
                    predictions = [ self.apply_post_transforms(pp, sample)[0] for pp in predictions]
                else:
                    predictions, _ = self.apply_post_transforms(predictions, sample)

            targets, new_affine = self.apply_post_transforms(targets, sample)

            # Compute loss
            loss = 0
            if targets is None:
                targets = predictions

            for criterion in self.criteria:
                if isinstance(predictions, list):
                    one_loss = criterion['criterion'](predictions[0], targets) #only the loss of the firts model ...
                else:
                    one_loss = criterion['criterion'](predictions, targets)
                if isinstance(one_loss, tuple): #multiple task loss may return tuple to report each loss
                    one_loss = one_loss[0]
                loss += criterion['weight'] * one_loss

            # Compute gradient and do SGD step
            if is_model_training:
                self.optimizer.zero_grad()
                if self.apex_opt_level is not None:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()
                loss = float(loss)
                predictions = predictions.detach()
            if self.save_predictions and not is_model_training:
                if eval_dropout:
                    predictions_std = torch.stack([pp.cpu().detach() for pp in predictions_dropout])
                    predictions_std = torch.std(predictions_std, axis=0) #arggg

                if isinstance(predictions, list):
                    for nb_pred, ppp in enumerate(predictions):
                        for j, prediction in enumerate(ppp):
                            n = i * self.batch_size + j
                            vname = f'{self.save_volume_name}_M{self.model_name[nb_pred]}'
                            self.prediction_saver(sample, prediction.unsqueeze(0), n, j, affine=new_affine,
                                                  volume_name=vname, save_biggest_comp=self.save_biggest_comp)

                else:
                    for j, prediction in enumerate(predictions):
                        n = i * self.batch_size + j
                        self.prediction_saver(sample, prediction.unsqueeze(0), n, j, affine=new_affine,
                                              save_biggest_comp=self.save_biggest_comp)
                        if eval_dropout:
                            volume_name = self.save_volume_name + "_std" or "Dp_std"
                            aaa = self.save_bin
                            self.save_bin = False
                            self.prediction_saver(sample,  predictions_std[j].unsqueeze(0), n, j, volume_name=volume_name,
                            affine=new_affine, save_biggest_comp=self.save_biggest_comp)
                            self.save_bin = aaa

            if self.save_labels and not is_model_training:
                for j, target in enumerate(targets):
                    n = i * self.batch_size + j
                    self.label_saver(sample, target.unsqueeze(0), n, j, volume_name=self.save_label_name, affine=new_affine)
            if self.save_data and not is_model_training:
                volumes, _ = self.apply_post_transforms(volumes, sample)
                for j, volumes in enumerate(volumes):
                    n = i * self.batch_size + j
                    self.label_saver(sample, volumes.unsqueeze(0), n, j, volume_name='data', affine=new_affine)

            # Measure elapsed time
            batch_time = time.time() - start

            time_sum += batch_time
            loss_sum += loss
            average_loss = loss_sum / i
            average_time = time_sum / i
            reporting_time = 0

            if max_loss is None or max_loss < loss:
                max_loss = loss

            # Update DataFrame and record it every record_frequency iterations
            write_csv_file = i % self.record_frequency == 0 or i == len(loader)

            if self.batch_recorder is not None:
                if eval_dropout:
                    if self.eval_results_dir != self.results_dir: #todo case where not csv_name == 'eval':
                        df = pd.DataFrame()
                    for ii in range(0, eval_dropout):
                        df, reporting_time = self.batch_recorder(
                            df, sample, predictions_dropout[ii], targets, batch_time, write_csv_file, append_in_df=True)
                else:
                    if isinstance(predictions, list):
                        for iii, ppp in enumerate(predictions):
                            df, reporting_time = self.batch_recorder(
                                df, sample, ppp, targets, batch_time, write_csv_file,  append_in_df=True, model_name=self.model_name[iii])
                    else:
                        df, reporting_time = self.batch_recorder(
                            df, sample, predictions, targets, batch_time, write_csv_file)
                if write_csv_file: self.log_peak_CPU_memory()

                reporting_time_sum += reporting_time
                average_reporting_time = reporting_time_sum / i
            # Log training or validation information every
            # log_frequency iterations
            if i % self.log_frequency == 0:
                to_log = summary(
                    self.epoch, i, len(loader), max_loss, batch_time,
                    average_loss, average_time, model_mode,
                    reporting_time, average_reporting_time
                )
                self.log(to_log)

            # Run model on validation set every eval_frequency iteration
            if is_model_training and \
                    (i % self.eval_frequency == 0 or i == len(loader)):
                with torch.no_grad():
                    self.model.eval()
                    validation_loss = self.train_loop()
                    self.model.train()
                    if self.log_on_tensorboard:
                        self.tb_logger.add_scalars('Average Train/Val loss',
                                                   {'Train': average_loss,
                                                    'Val': validation_loss}, self.epoch)

                # Update scheduler at the end of the epoch
                if i == len(loader) and self.lr_scheduler is not None:
                    self.lr_scheduler.step(validation_loss)

                self.log('reinitialise loss / time ')
                time_sum, loss_sum, reporting_time_sum = 0, 0, 0
                average_loss, max_loss = None, None


            start = time.time()

        # Save model after an evaluation on the whole validation set
        if save_model and not is_model_training:
            self.save_checkpoint(average_loss)

        return average_loss

    def whole_image_evaluation_loop(self, save_transformed_samples=False):
        df, patch_df = pd.DataFrame(), pd.DataFrame()
        start = time.time()
        time_sum, loss_sum, reporting_time_sum = 0, 0, 0
        average_loss = None

        for i, sample in enumerate(self.val_set, 1):

            # Load target for the whole image
            volume, target = self.data_getter(sample)

            if save_transformed_samples:
                self.prediction_saver(sample, volume.unsqueeze(0), i)

            predictions, patch_df = self.make_prediction_on_whole_volume(
                sample, patch_df)
            sample_loss = 0

            # As direct inference on whole volume is preferred, aggregated
            # prediction is not used if a patch evaluation is run
            if not self.dense_patch_eval:
                predictions = self.apply_post_transforms(
                    predictions.unsqueeze(0), sample)[0][0]

                target = self.apply_post_transforms(
                    target.unsqueeze(0), sample)[0][0]

                if self.save_predictions:
                    self.prediction_saver(sample, predictions.unsqueeze(0), i)
                    
                if self.save_labels:
                    self.label_saver(sample, target.unsqueeze(0), i, 'label')

                # Compute loss
                for criterion in self.criteria:
                    sample_loss += criterion['weight'] * \
                                   criterion['criterion'](
                                       predictions.unsqueeze(0),
                                       target.unsqueeze(0)
                                   )

            # Measure elapsed time
            sample_time = time.time() - start

            time_sum += sample_time
            loss_sum += sample_loss
            average_loss = loss_sum / i
            average_time = time_sum / i

            reporting_time = 0
            if not self.dense_patch_eval:
                # Record information about the sample and the performances of
                # the model on this sample after every iteration
                df, reporting_time = self.batch_recorder(
                    df, sample, predictions.unsqueeze(0), target.unsqueeze(0),
                    sample_time, True)

            reporting_time_sum += reporting_time
            average_reporting_time = reporting_time_sum / i

            # Log validation information every log_frequency iterations
            if i % self.log_frequency == 0:
                to_log = summary(self.epoch, i, len(self.val_set),
                                 sample_loss, sample_time, average_loss,
                                 average_time, 'Val', reporting_time,
                                 average_reporting_time, 'Sample')
                self.log(to_log)

            start = time.time()
        return average_loss

    def inference_loop(self):
        start = time.time()
        time_sum, saving_time_sum = 0, 0

        for i, sample in enumerate(self.test_set, 1):
            if self.patch_size is not None or self.eval_patch_size is not None:
                predictions, _ = self.make_prediction_on_whole_volume(
                    sample, None)
            else:
                volume, _ = self.data_getter(sample)
                predictions = self.activation( self.model(volume.unsqueeze(0))[0] )

            predictions, new_affine = self.apply_post_transforms(
                predictions.unsqueeze(0), sample)
            predictions = predictions[0]

            # Measure elapsed time
            sample_time = time.time() - start

            time_sum += sample_time
            average_time = time_sum / i

            saving_start = time.time()
            self.prediction_saver(sample, predictions.unsqueeze(0), i, affine=new_affine)
            saving_time = time.time() - saving_start

            saving_time_sum += saving_time
            average_saving_time = saving_time_sum / i

            # Log time information every log_frequency iterations
            if i % self.log_frequency == 0:
                to_log = summary('/', i, len(self.test_set), '/', sample_time,
                                 '/', average_time, 'Val', saving_time,
                                 average_saving_time, 'Sample', 'saving')
                self.log(to_log)

            start = time.time()

    def get_affine(self, sample):
        affine = sample[self.image_key_name]['affine']
        if affine.ndim == 3:
            affine = to_numpy(affine[0])
        return affine

    def apply_post_transforms(self, tensors, sample):
        affine = self.get_affine(sample)
        if not self.post_transforms:
            return tensors, affine
        if len(self.post_transforms) == 0:
            return tensors, affine
        # Transforms apply on TorchIO subjects and TorchIO images require
        # 4D tensors
        transformed_tensors = []
        for i, tensor in enumerate(tensors):
            subject = torchio.Subject(
                pred=torchio.LabelMap(
                    tensor=to_var(tensor, 'cpu'),
                    affine=affine)
            )
            transformed = self.post_transforms(subject)
            tensor = transformed['pred']['data']
            transformed_tensors.append(tensor)
        new_affine = transformed['pred']['affine']
        transformed_tensors = torch.stack(transformed_tensors)
        return to_var(transformed_tensors, self.device, self.no_blocking), new_affine

    def save_checkpoint(self, loss=None):
        optimizer_dict = None
        scheduler_dict = None

        if self.optimizer is not None:
            optimizer_dict = self.optimizer.state_dict()

        if self.lr_scheduler is not None:
            scheduler_dict = self.lr_scheduler.state_dict()

        state = {'epoch': self.epoch,
                 'iterations': self.iteration,
                 'val_loss': loss,
                 'state_dict': self.model.state_dict(),
                 'optimizer': optimizer_dict,
                 'scheduler': scheduler_dict}
        if self.apex_opt_level is not None:
            state['amp'] = amp.state_dict()

        save_checkpoint(state, self.results_dir, self.model)

    def record_simple(self, df, sample, predictions, targets,
                                  batch_time, save=False):

        """
        Record information about the batches the model was trained or evaluated
        on during the segmentation task.
        At evaluation time, additional reporting metrics are recorded.
        """
        start = time.time()

        is_batch = not isinstance(sample, torchio.Subject)
        if self.model.training:
            mode = 'Train'
        else:
            if self.eval_results_dir != self.results_dir:
                df = pd.DataFrame()
            mode = 'Val' if is_batch else 'Whole_image'

        shape = targets.shape
        #size = np.product(shape[2:])
        location = sample.get('index_ini')

        batch_size = shape[0]

        sample_time = batch_time / batch_size

        time_sum = 0

        for idx in range(batch_size):
            if is_batch:
                image_path = sample[self.image_key_name]['path'][idx]
            else:
                image_path = sample[self.image_key_name]['path']
            info = {
                'image_filename': image_path,
                'shape': to_numpy(shape[2:]),
                'sample_time': sample_time
            }

            if is_batch:
                info['label_filename'] = sample[self.label_key_name][
                    'path'][idx]
            else:
                info['label_filename'] = sample[self.label_key_name]['path']

            if location is not None:
                info['location'] = to_numpy(location[idx])

            loss = 0
            if predictions is not None:
                for criterion in self.criteria:
                    loss += criterion['weight'] * criterion['criterion'](
                        predictions[idx].unsqueeze(0),
                        targets[idx].unsqueeze(0)
                    )
                info['loss'] = to_numpy(loss)

            if not self.model.training:
                for metric in self.metrics:
                    name = f'metric_{metric["name"]}'

                    info[name] = json.dumps(to_numpy(
                        metric['criterion'](
                            predictions[idx].unsqueeze(0),
                            targets[idx].unsqueeze(0)
                        )
                    ), cls=ArrayTensorJSONEncoder)
            if 'metrics' in sample[self.image_key_name]:
                dics = sample[self.image_key_name]['metrics']
                dicm = {}
                for key, val in dics.items():
                    dicm[key] = to_numpy(val[idx])
                    # if isinstance(val,dict): # hmm SSIM_wrapped still contains dict
                    #     for kkey, vval in val.items():
                    #         dicm[key + '_' + kkey] = to_numpy(vval[idx])
                    # else:
                    #     dicm[key] = to_numpy(val[idx])
                info.update(dicm)

            reporting_time = time.time() - start
            time_sum += reporting_time
            info['reporting_time'] = reporting_time
            start = time.time()

            self.record_history(info, sample, idx)

            df = df.append(info, ignore_index=True)

        if save:
            self.save_info(mode, df, sample)

        return df, time_sum

    def record_segmentation_batch(self, df, sample, predictions, targets,
                                  batch_time, save=False, csv_name='eval', append_in_df=False, model_name='model'):
        """
        Record information about the batches the model was trained or evaluated
        on during the segmentation task.
        At evaluation time, additional reporting metrics are recorded.
        """
        start = time.time()

        is_batch = not isinstance(sample, torchio.Subject)
        is_model_training = self.model[0].training if isinstance(self.model, list) else self.model.training
        if is_model_training:
            mode = 'Train'
        else:
            if self.eval_results_dir != self.results_dir and csv_name == 'eval' and append_in_df is False:
                df = pd.DataFrame()
            mode = 'Val' if is_batch else 'Whole_image'
            if csv_name == 'patch_eval':
                mode = 'patch_eval'

        shape = targets.shape
        #size = np.product(shape[2:])
        location = sample.get('index_ini') if 'index_ini' in sample else sample.get('location')
        if location is not None:
            location = location[:, :3]
        affine = self.get_affine(sample)
        # M is the product between a scaling and a rotation
        M = affine[:3, :3]
        voxel_size = np.sqrt( np.diagonal(M @ M.T).prod()) #np.diagonal(np.sqrt(M @ M.T)).prod()

        batch_size = shape[0]

        sample_time = batch_time / batch_size

        time_sum = 0

        for idx in range(batch_size):
            if is_batch:
                image_path = sample[self.image_key_name]['path'][idx]
            else:
                image_path = sample[self.image_key_name]['path']
            info = {
                'image_filename': image_path,
                'shape': to_numpy(shape[2:]),
                'sample_time': sample_time
            }
            if is_batch:
                info['batch_size'] = batch_size

            if 'name' in sample:
                info['name'] = sample['name'][idx]

            if self.label_key_name is not None:
                if is_batch:
                    info['label_filename'] = sample[self.label_key_name][
                        'path'][idx]
                else:
                    info['label_filename'] = sample[self.label_key_name]['path']

            if self.criteria[0]['criterion'].mixt_activation:
                max_chanel = shape[1] - self.criteria[0]['criterion'].mixt_activation
            else:
                max_chanel = shape[1]
            if self.labels is not None : #bad fix, just to make eval mode possible without target labels
                for channel in list(range(max_chanel)):
                    suffix = self.labels[channel]
                    info[f'occupied_volume_{suffix}'] = to_numpy(
                       targets[idx, channel].sum() * voxel_size
                    )
                    info[f'predicted_occupied_volume_{suffix}'] = to_numpy(
                        predictions[idx, channel].sum() * voxel_size
                    )

            if location is not None:
                info['location'] = to_numpy(location[idx])

            info['model_name'] = model_name

            loss = 0 ;
            for criterion in self.criteria:
                if 'return_loss_dict' in criterion['criterion'].metric.__self__.__dict__: #check if class retunr_loss_dict in atribut
                    criterion['criterion'].metric.__self__.return_loss_dict=True #call with extra info !
                    one_loss = criterion['criterion'](predictions[idx].unsqueeze(0), targets[idx].unsqueeze(0))
                    criterion['criterion'].metric.__self__.return_loss_dict=False #call with extra info !

                else:
                    one_loss = criterion['criterion'](predictions[idx].unsqueeze(0), targets[idx].unsqueeze(0))

                if isinstance(one_loss, tuple): #multiple task loss may return tuple to report each loss
                    for i in range(1,len(one_loss)):
                        aaa = one_loss[i]
                        if not isinstance(aaa,dict):
                            if aaa.requires_grad:
                                aaa = aaa.detach()
                            info[f'loss_{i}'] = to_numpy(aaa)
                        else: # UniVarGaussianLogLkd return a dict of different loss_metric
                            info = dict(info, **aaa)

                    one_loss = one_loss[0]
                elif isinstance(one_loss, list):
                    one_loss, loss_dict = one_loss[0], one_loss[1]
                loss += criterion['weight'] * one_loss

            info['loss'] = to_numpy(loss)

            if not is_model_training:
                for metric in self.metrics:
                    name = f'metric_{metric["name"]}'

                    info[name] = json.dumps(to_numpy(
                        metric['criterion'](
                            predictions[idx].unsqueeze(0),
                            targets[idx].unsqueeze(0)
                        )
                    ), cls=ArrayTensorJSONEncoder)

            reporting_time = time.time() - start
            time_sum += reporting_time
            info['reporting_time'] = reporting_time
            start = time.time()

            self.record_history(info, sample, idx)

            df = pd.concat([df, pd.DataFrame([info])], ignore_index=True)

        if self.log_on_tensorboard is True:
            save_frequency = 30  # Log on TB every (save_frequency / batch_size) iterations
            train_len = len(self.train_loader) if self.train_loader else 0
            val_len = len(self.val_loader) if self.val_loader else 0

            total_iter = {'Train': (self.epoch - 1) * train_len + self.iteration,
                          'Val': (self.epoch - 1) * val_len + self.val_iteration}

            if self.criteria[0]['criterion'].mixt_activation:
                max_channel = shape[1] - self.criteria[0]['criterion'].mixt_activation
            else:
                max_channel = shape[1]

            if (mode == 'Train' and self.tb_df_train.shape[0] < save_frequency) or \
                    (mode == 'Val' and self.tb_df_val.shape[0] < save_frequency):
                current_iter = self.iteration if mode == 'Train' else self.val_iteration
                start = current_iter * self.batch_size - self.batch_size
                end = current_iter * self.batch_size
                if df.shape[0] < end:
                    end = df.shape[0]
                for idx in range(start, end):
                    values = {'loss': df['loss'].iloc[idx]}
                    for channel in list(range(max_channel)):
                        suffix = self.labels[channel]
                        values[f'predicted_occupied_volume_{suffix}'] = df[f'predicted_occupied_volume_{suffix}'].iloc[idx]
                        values[f'occupied_volume_{suffix}'] = df[f'occupied_volume_{suffix}'].iloc[idx]
                    if not self.model.training:
                        for metric in self.metrics:
                            name = f'metric_{metric["name"]}'
                            values[name] = float(df[name].iloc[idx])
                    if mode == 'Train':
                        self.tb_df_train = self.tb_df_train.append(values, ignore_index=True)
                    elif mode == 'Val':
                        self.tb_df_val = self.tb_df_val.append(values, ignore_index=True)

            if (mode == 'Train' and self.tb_df_train.shape[0] >= save_frequency) or \
                    (mode == 'Val' and self.tb_df_val.shape[0] >= save_frequency):
                tb_values = self.tb_df_train if mode == 'Train' else self.tb_df_val

                self.tb_logger.add_scalar('Total {} loss'.format(mode),
                                          tb_values['loss'].mean(),
                                          total_iter[mode])
                for channel in list(range(max_channel)):
                    suffix = self.labels[channel]
                    self.tb_logger.add_scalar('Total {} {} volume ratio'.format(mode, suffix),
                                              (tb_values[f'predicted_occupied_volume_{suffix}'] /
                                              tb_values[f'occupied_volume_{suffix}']).mean(),
                                              total_iter[mode])

                if not self.model.training:
                    for metric in self.metrics:
                        name = f'metric_{metric["name"]}'
                        self.tb_logger.add_scalar(name,
                                                  tb_values[name].mean(),
                                                  total_iter[mode])
                if mode == 'Train':
                    self.tb_df_train = pd.DataFrame()
                elif mode == 'Val':
                    self.tb_df_val = pd.DataFrame()

        if save:
            self.save_info(mode, df, sample, csv_name)

        return df, time_sum

    def make_prediction_on_whole_volume(self, sample, df):
        patch_size = self.eval_patch_size or self.patch_size
        grid_sampler = torchio.inference.GridSampler(
            sample, patch_size, self.patch_overlap, padding_mode='reflect'
        )
        patch_loader = DataLoader(grid_sampler, batch_size=self.batch_size)
        aggregator = torchio.inference.GridAggregator(grid_sampler)

        if self.results_dir != self.eval_results_dir:
            df = pd.DataFrame()

        for patches_batch in patch_loader:
            # Take variables and make sure they are tensors on the right device
            volumes, targets = self.data_getter(patches_batch)
            locations = patches_batch[torchio.LOCATION]

            # Compute output
            predictions = self.activation(self.model(volumes))
            aggregator.add_batch(predictions, locations)

            if self.dense_patch_eval and targets is not None:
                df, _ = self.batch_recorder(
                    df, patches_batch, predictions, targets,
                    0, True, csv_name='patch_eval'
                )

        # Aggregate predictions for the whole image
        predictions = to_var(aggregator.get_output_tensor(), self.device, self.no_blocking)
        return predictions, df

    def save_volume(self, sample, volume, idx=0, batch_idx=0, affine=None, volume_name=None,
                    save_biggest_comp = None):
        volume_name = volume_name or self.save_volume_name
        name = sample.get('name') or f'{idx:06d}'
        if affine is None:
            affine = self.get_affine(sample)

        if isinstance(name, list):
            name = name[batch_idx]
            name = name[0] if isinstance(name,list) else name

        resdir = f'{self.eval_results_dir}/{name}/'
        if not os.path.isdir(resdir):
            os.makedirs(resdir)
        if 'list_idx' in sample:
            volume_name = 'l{}_'.format(batch_idx) + volume_name

        if self.save_bin:
            bin_volume = torch.argmax(volume, dim=1)
            bin_volume = nib.Nifti1Image(
                to_numpy(bin_volume[0]).astype(np.uint8), affine
            )
            nib.save(bin_volume, f'{resdir}/bin_{volume_name}.nii.gz')

        volume[volume < self.save_threshold] = 0.

        if save_biggest_comp:
            volume = to_numpy(volume)
            #find label index
            index_label=[ self.labels.index(nn) for nn in self.save_biggest_comp]
            volume_mask = volume > 0.1 #billot use 0.25 ... why so big ?
            for i_label in index_label:
                tmp_mask = get_largest_connected_component(volume_mask[:, i_label, ...])
                volume[:, i_label, ...] *= tmp_mask
            #renomalize posteriors todo what if sum over proba is null after connected compo ... ?
            # if np.sum() == 0
            volume /= np.sum(volume, axis=1)[:,np.newaxis,...]

        if self.save_channels is not None:
            channels = [self.labels.index(c) for c in self.save_channels]
            channel_names = self.save_channels
            volume = volume[:, channels, ...]
        else:
            channel_names = self.labels

        if self.split_channels:
            for channel in range(volume.shape[1]):
                label = channel_names[channel]
                v = nib.Nifti1Image(
                    to_numpy(volume[0, channel, ...]), affine
                )
                nib.save(v, f'{resdir}/{label}.nii.gz')

        else:
            if isinstance(volume, np.ndarray):
                volume = np.squeeze( np.transpose(volume, (0, 2, 3, 4, 1) ) )
            else:
                volume = volume.permute(0, 2, 3, 4, 1).squeeze()
            volume = nib.Nifti1Image(
                to_numpy(volume), affine
            )
            nib.save(volume, f'{resdir}/{volume_name}.nii.gz')
            self.debug('saving {}'.format(f'{resdir}/{volume_name}.nii.gz'))


    def get_regression_data(self, data, target=None, scale_label=[1], default_missing_label = 0 ):

        if isinstance(data, list):  # case where callate_fn is used
            #inputs = torch.cat([sample[self.image_key_name]['data'].unsqueeze(0) for sample in data]) #this was wen lamba collate x:x
            #inputs = torch.cat([sample[self.image_key_name]['data'] for sample in data])
            # #this happen when ListOf transform
            input_list, labels_list = [], []
            for dd in data:
                ii, ll = self.get_regression_data(dd, target)
                input_list.append(ii)
                labels_list.append(ll)
            inputs = torch.cat(input_list)
            labels = torch.cat(labels_list)
            return inputs, labels

        else:
            inputs = data[self.image_key_name]['data']

        targets = [target] if isinstance(target, str) else target
        default_missing_label = [default_missing_label] if not isinstance(default_missing_label, list) else default_missing_label
        self.target_name = targets
        self.scale_label = [scale_label] if not isinstance(scale_label, list) else scale_label
        #default values for missing label
        labels = torch.cat([torch.ones(inputs.shape[0],1) * default_lab for default_lab in default_missing_label], dim=1)
        for target_idx, target in enumerate(targets):
            if target == "is_motion":
                histo = data['history']
                for batch_idx, hh in enumerate(histo): #length = batch size
                    for hhh in hh : #length: number of transfo that lead history info
                        if isinstance(hhh, tio.transforms.augmentation.intensity.random_motion_from_time_course.MotionFromTimeCourse)  :
                            labels[batch_idx, 0] = 1 #; labels[batch_idx, 1] = 0
            elif target == 'random_noise':
                histo = data['history']
                for batch_idx, hh in enumerate(histo): #length = batch size
                    for hhh in hh : #length: number of transfo that lead history info
                        if isinstance(hhh, tio.transforms.augmentation.intensity.random_noise.Noise)  :
                            labels[batch_idx, target_idx] = hhh.std[self.image_key_name] * scale_label[target_idx]
            else:
                histo = data['transforms_metrics'] #  data['history']
                for batch_idx, dicm in enumerate(histo): #length = batch size
                    if len(dicm) >0 : # a transform with _metrics exist (motion ...)
                        dict_metrics = dicm[0][1][self.image_key_name]
                        labels[batch_idx, target_idx] = dict_metrics[target] * scale_label[target_idx]

        #print(f'label ar {labels}')
        inputs = to_var(inputs.float(), self.device, self.no_blocking)
        labels = to_var(labels.float(), self.device, self.no_blocking)

        return inputs, labels

    def record_regression_batch(self, df, sample, predictions, targets, batch_time, save=False, extra_info=None):
        """
        Record information about the the model was trained or evaluated on during the regression task.
        At evaluation time, additional reporting metrics are recorded.
        """
        start = time.time()
        mode = 'Train' if self.model.training else 'Val'
        if self.eval_results_dir != self.results_dir:
            df = pd.DataFrame()
            save=True

        location = sample.get('index_ini')
        shape = sample[self.image_key_name]['data'].shape
        batch_size = shape[0]
        sample_time = batch_time / batch_size
        time_sum = 0
        is_batch = not isinstance(sample, torchio.Subject)

        for idx in range(batch_size):
            info = {
                'image_filename': sample[self.image_key_name]['path'][idx] if is_batch else sample[self.image_key_name]['path'],
                'shape': to_numpy(shape[2:]),
                'sample_time': sample_time,
                'batch_size': batch_size,
            }

            if location is not None:
                info['location'] = to_numpy(location[idx])

            if self.label_key_name in sample :
                info['label_filename'] = sample[self.label_key_name]['path'][idx] if is_batch else sample[self.label_key_name]['path']
            if 'name' in sample:
                info['subject_name'] = sample['name'][idx] if is_batch else sample['name']

            with torch.no_grad():
                loss = 0
                for criterion in self.criteria:
                    loss += criterion['weight'] * criterion['criterion'](predictions[idx].unsqueeze(0), targets[idx].unsqueeze(0))
                info['loss'] = to_numpy(loss)
                info['prediction'] = to_numpy(predictions[idx])
                info['targets'] = to_numpy(targets[idx])
                if 'target_name' in self.__dict__.keys():
                    for i_target, tgn in enumerate(self.target_name):
                        info['tar_' + tgn] = to_numpy(targets[idx][i_target])
                        info['pred_' + tgn] = to_numpy(predictions[idx][i_target])
                if 'scale_label' in self.__dict__.keys():
                    for i_target, tgn in enumerate(self.target_name):
                        info['scale_'+tgn] = self.scale_label[i_target]


            if 'simu_param' in sample[self.image_key_name]:
                #dicm = sample[self.image_key_name]['metrics']
                dics = sample[self.image_key_name]['simu_param']
                dicm={}
                for key, val in dics.items():
                    dicm[key] = to_numpy(val[idx])
                info.update(dicm)

            if 'metrics' in sample[self.image_key_name]:
                dics = sample[self.image_key_name]['metrics']
                dicm = {}
                """
                for key, val in dics.items():
                    dicm[key] = to_numpy(val[idx])
                    # if isinstance(val,dict): # hmm SSIM_wrapped still contains dict
                    #     for kkey, vval in val.items():
                    #         dicm[key + '_' + kkey] = to_numpy(vval[idx])
                    # else:
                    #     dicm[key] = to_numpy(val[idx])
                """
                info.update({"metrics": {self.image_key_name: dics[idx]}})

            if not self.model.training:
                for metric in self.metrics:
                    name = f'metric_{metric["name"]}'

                    info[name] = json.dumps(to_numpy(
                        metric['criterion'](
                            predictions[idx].unsqueeze(0),
                            targets[idx].unsqueeze(0)
                        )
                    ), cls=ArrayTensorJSONEncoder)

            self.record_history(info, sample, idx)

            reporting_time = time.time() - start
            time_sum += reporting_time
            info['reporting_time'] = reporting_time
            start = time.time()

            if extra_info is not None:
                info = dict(info, ** extra_info)

            df = pd.concat([df, pd.DataFrame([info])], ignore_index=True)
            #df = df.append(info, ignore_index=True)

        if save:
            self.save_info(mode, df, sample)

        return df, time_sum

    @staticmethod
    def record_history(info, sample, idx=None):
        is_batch = not isinstance(sample, torchio.Subject)
        order = []
        history = sample.get('history') if is_batch else sample.history
        transforms_metrics = sample.get("transforms_metrics") if is_batch else sample.transforms_metrics
        if history is None or len(history) == 0:
            return
        relevant_history = history[idx] if is_batch else history
        info["history"] = relevant_history

        relevant_metrics = transforms_metrics[idx] if is_batch else transforms_metrics

        if len(relevant_metrics) == 1 and isinstance(relevant_metrics[0], list):
            relevant_metrics = relevant_metrics[0]
        info["transforms_metrics"] = relevant_metrics
        if len(relevant_history)==1 and isinstance(relevant_history[0], list):
            relevant_history = relevant_history[0] #because ListOf transfo to batch make list of list ...

        for hist in relevant_history:
            if isinstance(hist, dict) :
                histo_name = hist['name']
                for key, val in hist.items():
                    if callable(val):
                        hist[key] = str(val)
                str_hist = str( hist )
            else:
                histo_name = hist.name #arg bad idea to mixt transfo and dict
                str_hist = dict()
                for name in  hist.args_names :
                    val = getattr(hist, name)
                    if callable(val):
                        val = str(val)
                    str_hist[name] = val
#               str_hist = {name: str() if isinstance(getattr(hist, name),funtion) else getattr(hist, name) for name in hist.args_names}
            #instead of using str(hist) wich is not correct as a python eval, make a dict of input_param
            if f'T_{histo_name}' in info:
                histo_name = f'{histo_name}_2'
            info[f'T_{histo_name}'] = json.dumps(
                str_hist, cls=ArrayTensorJSONEncoder)
            order.append(histo_name)

        info['transfo_order'] = '_'.join(order)

    def save_info(self, mode, df, sample, csv_name='eval'):
        name = self.eval_csv_basename or mode
        if mode == 'Train':
            filename = f'{self.results_dir}/{name}_ep{self.epoch:03d}'
        else:
            if self.eval_results_dir == self.results_dir:
                filename = f'{self.results_dir}/{name}_ep{self.epoch:03d}' \
                           f'_it{self.iteration:04d}'
            else:
                name = sample.get('name')
                if isinstance(name, list):
                    name = name[0][0] if isinstance(name[0],list) else name[0]

                resdir = f'{self.eval_results_dir}/{name}/'
                if not os.path.isdir(resdir):
                    os.makedirs(resdir)
                if self.eval_csv_basename is not None:
                    csv_name = self.eval_csv_basename + '_' + csv_name #if repeate_eval the val loop change eval_csv_basename
                filename = f'{resdir}/{csv_name}'
        df = df.drop(columns=["history"])
        df.to_csv(filename+".csv")

    def sample_list_to_batch(self,sample):
        #similare to collate function but instead of adding the batch dim to tensor we just do a cat
        elem = sample[0]
        keys = elem.keys()
        new_sample = dict()
        for k in keys:
            elem2 = elem[k]
            if isinstance(elem2, torch.Tensor):
                new_sample[k] = torch.cat([e[k] for e in sample])
            elif isinstance(elem2, dict):
                new_sample[k] = self.sample_list_to_batch([e[k] for e in sample])
            else:
                new_sample[k] = [e[k] for e in sample]

        return new_sample

    def torch_save_suj(self, subject, result_dir, fname, CAST_TO=torch.float16):
        #result_dir = result_dir + '/tio/'
        #self.my_mkdir(result_dir)
        if CAST_TO is not None:
            for sujkey in subject.get_images_names():
                subject[sujkey]['data'] = subject[sujkey]['data'].to(dtype=CAST_TO)
                if self.save_threshold:
                    volume = subject[sujkey]['data']
                    volume[volume < self.save_threshold] = 0.
                    subject[sujkey]['data'] = volume

        start = time.time()

        os.chdir(result_dir)
        fname += '.pt'
        torch.save(subject, fname)
        #with tarfile.open(f'{fname}.tar.gz', 'w:gz') as tar:
        #    tar.add(fname)

        os.system(f'tar -czf {fname}.tar.gz {fname}')
        os.system(f'rm -f {fname}')
        end_time = time.time() - start
        self.log(f'{fname}.tar.gz is saved in {end_time}')

    def get_transfo_short_name(self, in_str):
        ll = in_str.split('_')
        name = ''
        for sub in ll:
            name += sub[:3]
        return name

    def do_plot_volume_tio(self, suj, img_key, result_dir, fname, fname_prefix):
        result_dir = result_dir + '/fig/'
        self.my_mkdir(result_dir)
        if fname_prefix:
            fname = fname_prefix + fname

        if not isinstance(img_key, list): img_key = [img_key]
        for ikey in img_key:
            # hape = suj[img_key].data.shape[-3:]
            # fig_size = [np.array(shape).max() + 40 , np.array(shape).sum()+40]
            suj[ikey].plot(show=False, output_path=result_dir + fname + '_tio.png',
                           xlabels=False, percentiles=(0, 100))  # , figsize=fig_size)

    def plot_motion(self, suj, result_dir, fname, key_prefix):
        import matplotlib.pyplot as plt
        result_dir = result_dir + '/fig/'
        self.my_mkdir(result_dir)
        fname_prefix = None

        tmot = None
        for hist in suj.history:
            if isinstance(hist, tio.transforms.augmentation.intensity.random_motion_from_time_course.MotionFromTimeCourse):
                tmot = hist
        if tmot :
            transfo_met = suj.transforms_metrics[0][1]
            fitpar_dict = tmot.euler_motion_params
            for key,fitpars in fitpar_dict.items():
                image_metrics = int( transfo_met[key][key_prefix] * 100)
                fname_prefix = f'{key_prefix}_0{image_metrics}'
                #print(key)
                #print(fitpar.shape)
                plt.ioff()
                fig = plt.figure()
                plt.plot(fitpars.T)
                plt.savefig(result_dir + fname_prefix + fname + '_mvt.png')
                plt.close(fig)
        return fname_prefix

    def save_nii_volume(self, suj, img_key, result_dir, fname):
        #outdir = result_dir + '/nii/'
        #self.my_mkdir(result_dir)

        if not isinstance(img_key, list): img_key = [img_key]

        for ikey in img_key:
            suj[ikey].save(result_dir + '/' + ikey + '_' + fname + '.nii.gz')

    def my_mkdir(self, result_dir):
        try : #on cluster, all job are doing the mkdir at the same time ...
            if not os.path.isdir(result_dir): os.mkdir(result_dir)
        except:
            pass

    def save_sample(self, dataset):
        main_result_dir = self.eval_results_dir + '/'

        nb_sample = self.n_epochs
        nb_suj = len(dataset)
        for suj_num in range(0, nb_suj ):
            for i_sample in range(0, nb_sample):
                result_dir = main_result_dir + f'ep_{i_sample:03d}/'
                self.my_mkdir(result_dir)

                #i_sample += sample_num_offset
                suj = dataset[suj_num] #same suj but new transform
                df = pd.DataFrame(); #sample = dict()
                df, reporting_time = self.record_simple(df, suj, None,suj[self.image_key_name],0, False)

                transfo_name = self.get_transfo_short_name(df.transfo_order.values[0])
                sujname = suj.name.replace('/','_')
                fname  = f'suj_{sujname}_{transfo_name}_S{i_sample:03d}'

                df = df.drop(columns=["history"])
                if os.path.isfile(fname+'.csv'):
                    raise(f'Error file {fname}.csv exist ... grrr ...')
                df.to_csv(result_dir+fname+".csv")
                save_option = self.save_struct

                plot_volume = True if "plot_volume" in save_option else False
                plot_motion = True if "plot_motion" in save_option else False
                save_tio = True if "save_tio" in save_option else False
                save_data = True if "save_nii_img_key" in save_option else False

                fname_prefix = None
                if plot_motion:
                    fname_prefix = self.plot_motion(suj,result_dir,fname, save_option['plot_motion'])

                if plot_volume:
                    plot_key = save_option['plot_volume']
                    self.do_plot_volume_tio(suj,plot_key,result_dir, fname, fname_prefix)

                if save_data :
                    nii_key = save_option['save_nii_img_key']
                    self.save_nii_volume(suj, nii_key, result_dir, fname)
                if save_tio:
                    if 'save_tio_drop_key' in save_option:
                        for dkey in  save_option['save_tio_drop_key']:
                            suj.remove_image(dkey)

                    suj.clear_history()
                    self.torch_save_suj(suj, result_dir, fname, CAST_TO=torch.float16)

