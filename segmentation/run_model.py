""" Run model and save it """

import json
import torch
import time
import logging
import glob
import os
import numpy as np
import pandas as pd
import nibabel as nib
import torchio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from segmentation.utils import to_var, summary, save_checkpoint, to_numpy


class ArrayTensorJSONEncoder(json.JSONEncoder):
    """
    JSONEncoder extension to be able to stringify NumPy arrays and Torch
    tensors.
    """
    def default(self, o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()
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
                 patch_size, struct):
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
        self.n_epochs = struct['n_epochs']
        self.seed = struct['seed']
        self.activation = struct['activation']

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

        self.eval_model_name = None
        self.eval_csv_basename = None
        self.save_transformed_samples = False

    def log(self, info):
        if self.logger is not None:
            self.logger.log(logging.INFO, info)

    def get_optimizer(self, optimizer_dict):
        optimizer_dict['attributes'].update({'params': self.model.parameters()})
        optimizer = optimizer_dict['optimizer_class'](
            **optimizer_dict['attributes'])

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

    def train(self):
        """ Train the model on the training set and evaluate it on
        the validation set. """
        # Set seed for reproducibility
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Get optimizer and scheduler
        self.optimizer, self.lr_scheduler = self.get_optimizer(
            self.optimizer_dict)

        # Try to load optimizer state
        opt_files = glob.glob(
            os.path.join(self.results_dir, f'opt_ep{self.epoch - 1}*.pth.tar')
        )
        if len(opt_files) > 0:
            self.optimizer.load_state_dict(torch.load(opt_files[-1]))

        # Try to load scheduler state
        if self.lr_scheduler is not None:
            sch_files = glob.glob(
                os.path.join(self.results_dir,
                             f'sch_ep{self.epoch - 1}*.pth.tar')
            )
            if len(sch_files) > 0:
                self.lr_scheduler.load_state_dict(torch.load(sch_files[-1]))

        for epoch in range(self.epoch, self.n_epochs + 1):
            self.epoch = epoch
            self.log(f'******** Epoch [{self.epoch}/{self.n_epochs}]  ********')

            # Train for one epoch
            self.model.train()
            self.train_loop()

            # Evaluate on whole images of the validation set
            with torch.no_grad():
                self.model.eval()
                if self.patch_size is not None and self.epoch % \
                        self.whole_image_inference_frequency == 0:
                    self.log('Validation on whole images')
                    self.whole_image_evaluation_loop()

                    # Save model after inference
                    self.save_checkpoint()

        # Save model at the end of training
        self.save_checkpoint()

    def eval(self, model_name=None, eval_csv_basename=None,
             save_transformed_samples=False):
        """ Evaluate the model on the validation set. """
        self.epoch -= 1
        self.eval_model_name = model_name
        if eval_csv_basename:
            self.eval_csv_basename = eval_csv_basename

        self.log('Evaluation mode')
        with torch.no_grad():
            self.model.eval()
            if self.eval_frequency is not None:
                self.log('Evaluation on patches')
                self.train_loop(save_model=False)

            if self.patch_size is not None and \
                    self.whole_image_inference_frequency is not None:
                self.log('Evaluation on whole images')
                self.whole_image_evaluation_loop(save_transformed_samples)

    def infer(self):
        """ Use the model to make predictions on the test set. """
        self.epoch -= 1
        self.log('Inference')
        with torch.no_grad():
            self.model.eval()
            self.inference_loop()

    def train_loop(self, save_model=True):
        if self.model.training:
            self.log('Training')
            model_mode = 'Train'
            loader = self.train_loader
        else:
            self.log('Validation')
            model_mode = 'Val'
            loader = self.val_loader

        df = pd.DataFrame()
        start = time.time()
        time_sum, loss_sum = 0, 0
        average_loss, max_loss = None, None

        for i, sample in enumerate(loader, 1):
            if self.model.training:
                self.iteration = i

            # Take variables and make sure they are tensors on the right device
            volumes, targets = self.data_getter(sample)

            # Compute output
            predictions = self.model(volumes)

            # Compute loss
            loss = 0
            for criterion in self.criteria:
                loss += criterion['weight'] * criterion['criterion'](
                    predictions, targets, activation=self.activation)

            # Compute gradient and do SGD step
            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss = float(loss)

            # Measure elapsed time
            batch_time = time.time() - start

            time_sum += batch_time
            loss_sum += loss
            average_loss = loss_sum / i
            average_time = time_sum / i

            if max_loss is None or max_loss < loss:
                max_loss = loss

            # Log training or validation information every
            # log_frequency iterations
            if i % self.log_frequency == 0:
                to_log = summary(
                    self.epoch, i, len(loader), max_loss, batch_time,
                    average_loss, average_time, model_mode
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

            # Update DataFrame and record it every record_frequency iterations
            if i % self.record_frequency == 0 or i == len(loader):
                df = self.batch_recorder(
                    df, sample, predictions, targets, batch_time, True)
            else:
                df = self.batch_recorder(
                    df, sample, predictions, targets, batch_time, False)

            start = time.time()

        # Save model after an evaluation on the whole validation set
        if save_model and not self.model.training:
            self.save_checkpoint(average_loss)

        return average_loss

    def whole_image_evaluation_loop(self, save_transformed_samples=False):
        df = pd.DataFrame()
        start = time.time()
        time_sum, loss_sum = 0, 0
        average_loss = None

        for i, sample in enumerate(self.val_set, 1):
            # Load target for the whole image
            volume, target = self.data_getter(sample)

            if save_transformed_samples:
                self.prediction_saver(sample, volume, i)

            predictions = self.make_prediction_on_whole_volume(sample)

            # Compute loss
            sample_loss = 0
            for criterion in self.criteria:
                sample_loss += criterion['weight'] * \
                               criterion['criterion'](
                                   predictions.unsqueeze(0), target.unsqueeze(0)
                               )

            # Measure elapsed time
            sample_time = time.time() - start

            time_sum += sample_time
            loss_sum += sample_loss
            average_loss = loss_sum / i
            average_time = time_sum / i

            start = time.time()

            # Log validation information every log_frequency iterations
            if i % self.log_frequency == 0:
                to_log = summary(self.epoch, i, len(self.val_set),
                                 sample_loss, sample_time, average_loss,
                                 average_time, 'Val', 'Sample')
                self.log(to_log)

            # Record information about the sample and the performances of
            # the model on this sample after every iteration
            df = self.batch_recorder(df, sample, predictions.unsqueeze(0),
                                     target.unsqueeze(0), sample_time, True)
        return average_loss

    def inference_loop(self):
        start = time.time()
        time_sum = 0

        for i, sample in enumerate(self.test_set, 1):
            if self.patch_size is not None:
                predictions = self.make_prediction_on_whole_volume(sample)
            else:
                volume, _ = self.data_getter(sample)
                predictions = self.model(volume.unsqueeze(0))[0]

            # Measure elapsed time
            sample_time = time.time() - start
            time_sum += sample_time
            average_time = time_sum / i
            start = time.time()

            # Log time information every log_frequency iterations
            if i % self.log_frequency == 0:
                to_log = summary('/', i, len(self.test_set), '/', sample_time,
                                 '/', average_time, 'Val', 'Sample')
                self.log(to_log)

            self.prediction_saver(sample, predictions, i)

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

        save_checkpoint(state, self.results_dir, self.model)

    def record_segmentation_batch(self, df, sample, predictions, targets,
                                  batch_time, save=False):
        """
        Record information about the batches the model was trained or evaluated
        on during the segmentation task.
        At evaluation time, additional reporting metrics are recorded.
        """
        is_batch = not isinstance(sample, torchio.Subject)
        if self.model.training:
            mode = 'Train'
        else:
            mode = 'Val' if is_batch else 'Whole_image'

        shape = targets.shape
        size = np.product(shape[2:])
        location = sample.get('index_ini')
        batch_size = shape[0]

        sample_time = batch_time / batch_size

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

            if is_batch:
                info['label_filename'] = sample[self.label_key_name][
                    'path'][idx]
            else:
                info['label_filename'] = sample[self.label_key_name]['path']

            for channel in list(range(shape[1])):
                suffix = self.labels[channel]
                info[f'occupied_volume_{suffix}'] = to_numpy(
                   targets[idx, channel].sum() / size
                )
                if self.activation == 'softmax':
                    info[f'predicted_occupied_volume_{suffix}'] = to_numpy(
                        F.softmax(predictions, dim=1)[idx, channel].sum() / size
                    )
                else:
                    info[f'predicted_occupied_volume_{suffix}'] = to_numpy(
                        predictions[idx, channel].sum() / size
                    )

            if location is not None:
                info['location'] = to_numpy(location[idx])

            loss = 0
            for criterion in self.criteria:
                loss += criterion['weight'] * criterion['criterion'](
                    predictions[idx].unsqueeze(0),
                    targets[idx].unsqueeze(0),
                    activation=self.activation
                )
            info['loss'] = to_numpy(loss)

            if not self.model.training:
                for metric in self.metrics:
                    name = f'metric_{metric["name"]}'
                    kwargs = {
                        'prediction': predictions[idx].unsqueeze(0),
                        'target': targets[idx].unsqueeze(0),
                        'activation': self.activation
                    }

                    if metric['mask'] is not None:
                        kwargs['mask'] = to_var(
                            sample[metric['mask']]['data'][idx, 0], self.device)

                    channels = []
                    for key in metric['channels']:
                        channels.append(self.labels.index(key))
                    if len(channels) > 0:
                        kwargs['channels'] = channels

                    info[name] = to_numpy(metric['criterion'](**kwargs))

            self.record_history(info, sample, idx)

            df = df.append(info, ignore_index=True)

        if save:
            self.save_info(mode, df)

        return df

    def make_prediction_on_whole_volume(self, sample):
        grid_sampler = torchio.inference.GridSampler(
            sample, self.patch_size, self.patch_overlap, padding_mode='reflect'
        )
        patch_loader = DataLoader(grid_sampler, batch_size=self.batch_size)
        aggregator = torchio.inference.GridAggregator(grid_sampler)

        for patches_batch in patch_loader:
            # Take variables and make sure they are tensors on the right device
            volumes, _ = self.data_getter(patches_batch)
            locations = patches_batch[torchio.LOCATION]

            # Compute output
            predictions = self.model(volumes)
            aggregator.add_batch(predictions, locations)

        # Aggregate predictions for the whole image
        predictions = to_var(aggregator.get_output_tensor(), self.device)
        return predictions

    def save_volume(self, sample, volume, idx=0):
        affine = sample[self.image_key_name]['affine']
        name = sample.get('name') or f'{idx:06d}'
        volume = nib.Nifti1Image(to_numpy(volume.squeeze()), affine)
        nib.save(volume, f'{self.results_dir}/{name}.nii.gz')

    def get_regress_random_noise_data(self, data):
        return self.get_regression_data(data, 'random_noise')

    def get_regress_motion_data(self, data):
        return self.get_regression_data(data, 'ssim')

    def get_regression_data(self, data, target):

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

        labels = torch.zeros(inputs.shape[0], 1) #for eval case without label
        if 'metrics' in data[self.image_key_name]:
            if target in data[self.image_key_name]['metrics']:
                labels = data[self.image_key_name]['metrics'][target].unsqueeze(1)

            # if target == 'ssim':
            # #labels = data[self.image_key_name]['metrics']['ssim'].unsqueeze(1) \
            # labels = data[self.image_key_name]['metrics']['SSIM_base_brain'].unsqueeze(1) \
            #     if 'metrics' in  data[self.image_key_name] else torch.zeros(inputs.shape[0],1)
            #     #0 tensor with dim batch size for eval case without ssim

        elif target == 'random_noise':
            histo = data['history']
            lab = []
            for hh in histo: #length = batch size
                for hhh in hh : #length: number of transfo that lead history info
                    if 'RandomNoise' in hhh:
                        lab.append(hhh[1][self.image_key_name]['std'])
            labels = torch.Tensor(lab).unsqueeze(1)

        inputs = to_var(inputs.float(), self.device)
        labels = to_var(labels.float(), self.device)

        return inputs, labels

    def record_regression_batch(self, df, sample, predictions, targets, batch_time, save=False):
        """
        Record information about the the model was trained or evaluated on during the regression task.
        At evaluation time, additional reporting metrics are recorded.
        """

        if isinstance(sample, list):
            batch_size = predictions.shape[0] // len(sample)
            targets_split = torch.split(targets, batch_size)
            pred_split = torch.split(predictions, batch_size)
            for ss, pp, tt in zip(sample, pred_split, targets_split):
                df = self.record_regression_batch(df, ss, pp, tt, batch_time, save)
            return df

        mode = 'Train' if self.model.training else 'Val'

        location = sample.get('index_ini')
        shape = sample[self.image_key_name]['data'].shape
        batch_size = shape[0]
        sample_time = batch_time / batch_size

        for idx in range(batch_size):
            info = {
                'image_filename': sample[self.image_key_name]['path'][idx],
                'shape': to_numpy(shape[2:]),
                'sample_time': sample_time,
                'batch_size': batch_size
            }

            if location is not None:
                info['location'] = to_numpy(location[idx])

            with torch.no_grad():
                loss = 0
                for criterion in self.criteria:
                    loss += criterion['weight'] * criterion['criterion'](predictions[idx].unsqueeze(0), targets[idx].unsqueeze(0))
                info['loss'] = to_numpy(loss)
                info['prediction'] = to_numpy(predictions[idx])[0]
                info['targets'] = to_numpy(targets[idx])[0]

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
                for key, val in dics.items():
                    dicm[key] = to_numpy(val[idx])
                    # if isinstance(val,dict): # hmm SSIM_wrapped still contains dict
                    #     for kkey, vval in val.items():
                    #         dicm[key + '_' + kkey] = to_numpy(vval[idx])
                    # else:
                    #     dicm[key] = to_numpy(val[idx])
                info.update(dicm)

            if not self.model.training:
                for metric in self.metrics:
                    name = f'metric_{metric["name"]}'
                    kwargs = {'prediction': predictions[idx].unsqueeze(0), 'target': targets[idx].unsqueeze(0)}

                    if metric['mask'] is not None:
                        kwargs['mask'] = to_var(sample[metric['mask']]['data'][idx, 0], self.device)

                    channels = []
                    for key in metric['channels']:
                        channels.append(self.labels.index(key))
                    if len(channels) > 0:
                        kwargs['channels'] = channels

                    info[name] = to_numpy(metric['criterion'](**kwargs))

            self.record_history(info, sample, idx)

            df = df.append(info, ignore_index=True)

        if save:
            self.save_info(mode, df)

        return df

    @staticmethod
    def record_history(info, sample, idx=None):
        is_batch = not isinstance(sample, torchio.Subject)
        order = []
        history = sample.get('history') if is_batch else sample.history
        if history is None or len(history) == 0:
            return
        relevant_history = history[idx] if is_batch else history
        for hist in relevant_history:
            info[f'T_{hist[0]}'] = json.dumps(
                hist[1], cls=ArrayTensorJSONEncoder)
            order.append(hist[0])
        info['transfo_order'] = '_'.join(order)

    def save_info(self, mode, df):
        name = self.eval_csv_basename or mode
        if mode == 'Train':
            filename = f'{self.results_dir}/{name}_ep{self.epoch:03d}.csv'
        elif self.eval_model_name is not None:
            filename = f'{self.results_dir}/' \
                       f'{name}_from_{self.eval_model_name}.csv'
        else:
            filename = f'{self.results_dir}/' \
                       f'{name}_ep{self.epoch:03d}_it{self.iteration:04d}.csv'
        df.to_csv(filename)
