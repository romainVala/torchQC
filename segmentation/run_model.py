""" Run model and save it """

import json
import torch
import time
import pickle
import logging
import glob
import os
import numpy as np
import pandas as pd
import nibabel as nib
import torchio
import torchio as tio
import resource
import warnings
from torch.utils.data import DataLoader
from segmentation.utils import to_var, summary, save_checkpoint, to_numpy
from apex import amp


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
                 patch_size, struct, post_transforms):
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

        self.batch_size = batch_size
        self.patch_size = patch_size

        self.post_transforms = post_transforms

        # Set attributes to keep track of information during training
        self.epoch = struct['current_epoch']
        self.iteration = 0

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
        self.save_volume_name = struct['save']['save_volume_name']
        self.apex_opt_level = struct['apex_opt_level']

        # Keep information to load optimizer and learning rate scheduler
        self.optimizer, self.lr_scheduler = None, None
        self.optimizer_dict = struct['optimizer']

        # Define which methods will be used to retrieve data and record
        # information
        function_datagetter = getattr(self, struct['data_getter']['name'])
        attributes = struct['data_getter']['attributes']
        self.data_getter = lambda sample: function_datagetter(
            sample, **attributes)

        self.batch_recorder = getattr(self, struct['save']['batch_recorder'])
        self.prediction_saver = getattr(
            self, struct['save']['prediction_saver'])
        self.label_saver = getattr(self, struct['save']['label_saver'])

        self.eval_csv_basename = None
        self.save_transformed_samples = False

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

        return optimizer, scheduler

    def get_segmentation_data(self, sample):
        volumes = sample[self.image_key_name]
        volumes = to_var(volumes[torchio.DATA].float(), self.device)

        targets = None
        if self.label_key_name in sample:
            targets = sample[self.label_key_name]
            targets = to_var(targets[torchio.DATA].float(), self.device)
        return volumes, targets

    def get_segmentation_data_and_regress_key(self, sample, regress_key):
        volumes = sample[self.image_key_name]
        volumes = to_var(volumes[torchio.DATA].float(), self.device)

        targets = None
        if self.label_key_name in sample:
            targets = sample[self.label_key_name]
            targets = to_var(targets[torchio.DATA].float(), self.device)
        if regress_key in sample:
            targets_to_regress = to_var(sample[regress_key][torchio.DATA].float(), self.device)
            targets = torch.cat((targets, targets_to_regress), dim=1)
        return volumes, targets

    def train(self):
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
            self.optimizer.load_state_dict(torch.load(opt_files[-1]))

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
        if self.model.training:
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
            if self.model.training:
                self.iteration = i
            if isinstance(sample, list):
                self.batch_size = self.batch_size * len(sample)
                for ii, ss in enumerate(sample):
                    ss['list_idx'] = ii #add this key to use in save_volume (having different volume name)
                sample = self.sample_list_to_batch(sample)

            # Take variables and make sure they are tensors on the right device
            volumes, targets = self.data_getter(sample)

            # Compute output
            if eval_dropout:
                self.model.enable_dropout()
                if self.split_batch_gpu and self.batch_size > 1:
                    predictions_dropout = [torch.cat([self.model(v.unsqueeze(0)) for v in volumes]) for ii in range(0, eval_dropout)]
                else:
                    predictions_dropout = [self.model(volumes) for ii in range(0, eval_dropout)]
                predictions = torch.mean(torch.stack(predictions_dropout), axis=0)
            else:
                if self.split_batch_gpu and self.batch_size > 1: #usefull, when using ListOf transform that pull sample in batch to avoid big full volume batch in gpu
                    predictions = torch.cat([self.model(v.unsqueeze(0)) for v in volumes])
                else:
                    predictions = self.model(volumes)

            # Compute loss
            loss = 0
            if targets is None:
                targets = predictions

            for criterion in self.criteria:
                one_loss = criterion['criterion'](predictions, targets)
                if isinstance(one_loss, tuple): #multiple task loss may return tuple to report each loss
                    one_loss = one_loss[0]
                loss += criterion['weight'] * one_loss

            # Compute gradient and do SGD step
            if self.model.training:
                self.optimizer.zero_grad()
                if self.apex_opt_level is not None:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()
                loss = float(loss)
                predictions = predictions.detach()

            if eval_dropout:
                predictions_dropout = [self.apply_post_transforms(pp, sample)[0] for pp in predictions_dropout]
                predictions = torch.mean(torch.stack(predictions_dropout), axis=0)
            else:
                predictions, _ = self.apply_post_transforms(predictions, sample)
            targets, new_affine = self.apply_post_transforms(targets, sample)

            if self.save_predictions and not self.model.training:
                if eval_dropout:
                    predictions_std = torch.stack([self.activation(pp.cpu().detach()) for pp in predictions_dropout])
                    predictions_std = torch.std(predictions_std, axis=0) #arggg
                for j, prediction in enumerate(predictions):
                    n = i * self.batch_size + j
                    self.prediction_saver(sample, prediction.unsqueeze(0), n, j, affine=new_affine)
                    if eval_dropout:
                        volume_name = self.save_volume_name + "_std" or "Dp_std"
                        aaa = self.save_bin
                        self.save_bin = False
                        self.prediction_saver(sample,  predictions_std[j].unsqueeze(0), n, j, volume_name=volume_name,
                        apply_activation=False, affine=new_affine)
                        self.save_bin = aaa

            if self.save_labels and not self.model.training:
                for j, target in enumerate(targets):
                    n = i * self.batch_size + j
                    self.label_saver(sample, target.unsqueeze(0), n, j, volume_name='label', affine=new_affine, apply_activation=False,)
            if self.save_data and not self.model.training:
                volumes, _ = self.apply_post_transforms(volumes, sample)
                for j, volumes in enumerate(volumes):
                    n = i * self.batch_size + j
                    self.label_saver(sample, volumes.unsqueeze(0), n, j, volume_name='data', apply_activation=False, affine=new_affine)

            # Measure elapsed time
            batch_time = time.time() - start

            time_sum += batch_time
            loss_sum += loss
            average_loss = loss_sum / i
            average_time = time_sum / i

            if max_loss is None or max_loss < loss:
                max_loss = loss

            # Update DataFrame and record it every record_frequency iterations
            write_csv_file = i % self.record_frequency == 0 or i == len(loader)

            if eval_dropout:
                if self.eval_results_dir != self.results_dir: #todo case where not csv_name == 'eval':
                    df = pd.DataFrame()
                for ii in range(0, eval_dropout):
                    df, reporting_time = self.batch_recorder(
                        df, sample, predictions_dropout[ii], targets, batch_time, write_csv_file, append_in_df=True)
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
            if self.model.training and \
                    (i % self.eval_frequency == 0 or i == len(loader)):
                with torch.no_grad():
                    self.model.eval()
                    validation_loss = self.train_loop()
                    self.model.train()

                # Update scheduler at the end of the epoch
                if i == len(loader) and self.lr_scheduler is not None:
                    self.lr_scheduler.step(validation_loss)

            start = time.time()

        # Save model after an evaluation on the whole validation set
        if save_model and not self.model.training:
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
                predictions = self.model(volume.unsqueeze(0))[0]

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
                pred=torchio.ScalarImage(
                    tensor=to_var(tensor, 'cpu'),
                    affine=affine)
            )
            transformed = self.post_transforms(subject)
            tensor = transformed['pred']['data']
            transformed_tensors.append(tensor)
        new_affine = transformed['pred']['affine']
        transformed_tensors = torch.stack(transformed_tensors)
        return to_var(transformed_tensors, self.device), new_affine

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
                                  batch_time, save=False, csv_name='eval', append_in_df=False):
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
        voxel_size = np.diagonal(np.sqrt(M @ M.T)).prod()

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
            for channel in list(range(max_chanel)):
                suffix = self.labels[channel]
                info[f'occupied_volume_{suffix}'] = to_numpy(
                   targets[idx, channel].sum() * voxel_size
                )
                info[f'predicted_occupied_volume_{suffix}'] = to_numpy(
                    self.activation(predictions)[idx, channel].sum() * voxel_size
                )

            if location is not None:
                info['location'] = to_numpy(location[idx])

            loss = 0
            for criterion in self.criteria:
                one_loss = criterion['criterion'](predictions[idx].unsqueeze(0), targets[idx].unsqueeze(0))
                if isinstance(one_loss, tuple): #multiple task loss may return tuple to report each loss
                    for i in range(1,len(one_loss)):
                        aaa = one_loss[i]
                        if aaa.requires_grad:
                            aaa = aaa.detach()
                        info[f'loss_{i}'] = to_numpy(aaa)

                    one_loss = one_loss[0]
                loss += criterion['weight'] * one_loss

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

            reporting_time = time.time() - start
            time_sum += reporting_time
            info['reporting_time'] = reporting_time
            start = time.time()

            self.record_history(info, sample, idx)

            df = df.append(info, ignore_index=True)

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
            predictions = self.model(volumes)
            aggregator.add_batch(predictions, locations)

            if self.dense_patch_eval and targets is not None:
                df, _ = self.batch_recorder(
                    df, patches_batch, predictions, targets,
                    0, True, csv_name='patch_eval'
                )

        # Aggregate predictions for the whole image
        predictions = to_var(aggregator.get_output_tensor(), self.device)
        return predictions, df

    def save_volume(self, sample, volume, idx=0, batch_idx=0, affine=None, volume_name=None, apply_activation=True):
        volume_name = volume_name or self.save_volume_name
        name = sample.get('name') or f'{idx:06d}'
        if affine is None:
            affine = self.get_affine(sample)

        if isinstance(name, list):
            name = name[batch_idx]
            name = name[0] if isinstance(name,list) else name
        if apply_activation:
            if self.criteria[0]['criterion'].mixt_activation: #softmax apply only on segmentation not regression
                skip_vol = self.criteria[0]['criterion'].mixt_activation
                vv= self.activation(volume[0,:-skip_vol,...].unsqueeze(0))
                volume[0,:-skip_vol,...] = vv[0]
            else:
                volume = self.activation(volume)

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
            volume = nib.Nifti1Image(
                to_numpy(volume.permute(0, 2, 3, 4, 1).squeeze()), affine
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
            if target == 'random_noise':
                histo = data['history']
                for batch_idx, hh in enumerate(histo): #length = batch size
                    for hhh in hh : #length: number of transfo that lead history info
                        if isinstance(hhh, tio.transforms.augmentation.intensity.random_noise.Noise)  :
                            labels[batch_idx, target_idx] = hhh.std[self.image_key_name] * scale_label[target_idx]
            else:
                histo = data['history']
                for batch_idx, hh in enumerate(histo): #length = batch size
                    for hhh in hh : #length: number of transfo that lead history info
                        #if '_metrics' in hhh[1].keys():
                        if isinstance(hhh,dict): #hhh.name == 'RandomMotionFromTimeCourse':
                            #dict_metrics = hhh[1]["_metrics"][self.image_key_name]
                            if '_metrics' in hhh:
                                dict_metrics = hhh['_metrics'][self.image_key_name]
                                labels[batch_idx, target_idx] = dict_metrics[target] * scale_label[target_idx]

        inputs = to_var(inputs.float(), self.device)
        labels = to_var(labels.float(), self.device)

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

            df = df.append(info, ignore_index=True)

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
        df.to_pickle(filename+'.gz', protocol=3)
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
