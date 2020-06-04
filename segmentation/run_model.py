""" Run model and save it """

import os
import json
import torch
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import torchio
from torch.utils.data import DataLoader
from segmentation.utils import parse_object_import, parse_function_import, to_var, summary, save_checkpoint, \
    instantiate_logger, to_numpy, set_dict_value, check_mandatory_keys, parse_method_import
from segmentation.visualization import report_loss


class ArrayTensorJSONEncoder(json.JSONEncoder):
    """
    JSONEncoder extension to be able to stringify NumPy arrays and Torch tensors.
    """
    def default(self, o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()
        else:
            return json.JSONEncoder.default(self, o)


RUN_KEYS = ['criteria', 'optimizer', 'logger', 'save', 'validation', 'n_epochs', 'image_key_name', 'label_key_name']
OPTIMIZER_KEYS = ['name', 'module']
LOGGER_KEYS = ['name', 'log_frequency', 'filename']
SAVE_KEYS = ['save_model', 'save_path', 'save_frequency', 'record_frequency']
VALIDATION_KEYS = ['batch_size', 'patch_size', 'eval_frequency', 'out_channels']


class RunModel:
    """
    Handle training, evaluation and saving of a model from a json configuration file.
    """
    def __init__(self, model, train_loader, val_loader, val_set, folder, run_filename='run.json'):
        with open(folder + run_filename) as file:
            info = json.load(file)

        check_mandatory_keys(info, RUN_KEYS, folder + run_filename)
        set_dict_value(info, 'seed')
        set_dict_value(info, 'title', 'Session')

        self.model = model
        self.device = next(model.parameters()).device

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_set = val_set

        # Parse information from json
        self.criteria = self.parse_criteria(info['criteria'])
        self.optimizer, self.lr_strategy, self.lr_strategy_attributes = self.parse_optimizer(info['optimizer'])
        self.logger, self.logger_dict = self.parse_logger(info['logger'])
        self.save_dict = self.parse_save(info['save'])
        self.validation_dict, self.metrics = self.parse_validation(info['validation'])

        self.n_epochs = info['n_epochs']
        self.seed = info['seed']
        self.image_key_name = info['image_key_name']
        self.label_key_name = info['label_key_name']
        self.title = info['title']

        # Define which methods will be used to retrieve data
        self.data_getter = getattr(self, info.get('data_getter') or 'get_segmentation_data')

        # Define which methods will be used to record information
        self.batch_recorder = getattr(self, info['batch_recorder'] or 'record_segmentation_batch')
        self.inference_recorder = getattr(self, info['inference_recorder'] or 'record_segmentation_inference')

        # Set attributes to keep track of information during training
        self.epoch = 0
        self.iteration = 0
        self.train_df = None
        self.val_df = None
        self.save = False

    @staticmethod
    def parse_criteria(criterion_list):
        c_list = []
        for criterion in criterion_list:
            c = parse_method_import(criterion)
            c_list.append(c)
        return c_list

    def parse_optimizer(self, optimizer_dict):
        check_mandatory_keys(optimizer_dict, OPTIMIZER_KEYS, 'optimizer dict')
        set_dict_value(optimizer_dict, 'attributes', {})
        set_dict_value(optimizer_dict, 'learning_rate_strategy')
        set_dict_value(optimizer_dict, 'learning_rate_strategy_attributes', {})

        optimizer_dict['attributes'].update({'params': self.model.parameters()})
        optimizer, _ = parse_object_import(optimizer_dict)

        strategy = optimizer_dict['learning_rate_strategy']
        if strategy is not None:
            strategy = parse_function_import(strategy)
        return optimizer, strategy, optimizer_dict['learning_rate_strategy_attributes']

    @staticmethod
    def parse_logger(logger_dict):
        check_mandatory_keys(logger_dict, LOGGER_KEYS, 'logger dict')
        set_dict_value(logger_dict, 'level', logging.INFO)

        logger_object = instantiate_logger(logger_dict['name'], logger_dict['level'], logger_dict['filename'])
        return logger_object, logger_dict

    @staticmethod
    def parse_save(save_dict):
        check_mandatory_keys(save_dict, SAVE_KEYS, 'save dict')
        set_dict_value(save_dict, 'custom_save')
        return save_dict

    def parse_validation(self, validation_dict):
        check_mandatory_keys(validation_dict, VALIDATION_KEYS, 'validation dict')
        set_dict_value(validation_dict, 'whole_image_inference_frequency', np.inf)
        set_dict_value(validation_dict, 'patch_overlap', 0)
        set_dict_value(validation_dict, 'reporting_metrics', [])

        metrics = self.parse_criteria(validation_dict['reporting_metrics'])
        return validation_dict, metrics

    def get_segmentation_data(self, sample):
        volumes = sample[self.image_key_name]
        targets = sample[self.label_key_name]

        volumes = to_var(volumes[torchio.DATA].float(), self.device)
        targets = to_var(targets[torchio.DATA].float(), self.device)
        return volumes, targets

    def train(self):
        # Log session title
        session_name = f'{self.title}_{time.strftime("%m.%d %Hh%M")}'
        self.logger.log(logging.INFO, session_name)

        # Log model
        self.logger.log(logging.INFO, '******** Model  ********')
        self.logger.log(logging.INFO, self.model)

        # Set seed for reproducibility
        if self.seed is not None:
            torch.manual_seed(self.seed)

        for epoch in range(1, self.n_epochs + 1):
            self.epoch = epoch
            self.logger.log(logging.INFO, '******** Epoch [{}/{}]  ********'.format(self.epoch, self.n_epochs))

            # Train for one epoch
            self.model.train()
            self.logger.log(logging.INFO, 'Training')
            self.train_loop()

            # Evaluate on whole images of the validation set
            with torch.no_grad():
                self.model.eval()
                if self.epoch % self.validation_dict['whole_image_inference_frequency'] == 0:
                    self.logger.log(logging.INFO, 'Validation')
                    self.whole_image_loop()

            # Save model
            if self.save_dict['save_model'] and self.epoch % self.save_dict['save_frequency'] == 0:
                state = {'epoch': self.epoch,
                         'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict()}
                save_checkpoint(state, self.save_dict['save_path'], self.save_dict['custom_save'], self.model)

            # Update learning rate
            if self.lr_strategy is not None:
                self.optimizer = self.lr_strategy(self.optimizer, self.logger_dict['log_filename'],
                                                  **self.lr_strategy_attributes)

        # Report loss
        report_loss(self.save_dict['save_path'])

    def train_loop(self, iterations=None):
        if self.model.training:
            model_mode = 'Train'
            loader = self.train_loader
            self.train_df = pd.DataFrame()
        else:
            model_mode = 'Val'
            loader = self.val_loader
            self.val_df = pd.DataFrame()

        batch_size = loader.batch_size

        start = time.time()
        time_sum, loss_sum = 0, 0

        average_loss = None

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
                loss += criterion(predictions, targets)

            # Compute gradient and do SGD step
            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Measure elapsed time
            batch_time = time.time() - start

            time_sum += batch_size * batch_time
            loss_sum += batch_size * loss
            average_loss = loss_sum / (i * batch_size)
            average_time = time_sum / (i * batch_size)

            start = time.time()

            # Log training or validation information every log_frequency iterations
            if i % self.logger_dict['log_frequency'] == 0:
                to_log = summary(self.epoch, i, len(loader), loss, batch_time, average_loss, average_time, model_mode)
                self.logger.log(logging.INFO, to_log)

            # Run model on validation set every eval_frequency iteration
            if self.model.training and i % self.validation_dict['eval_frequency'] == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.train_loop()
                    self.model.train()

            # Update DataFrame and record it every record_frequency iterations or every iteration at validation time
            if self.model.training and i % self.save_dict['record_frequency'] == 0:
                self.batch_recorder(sample, predictions, targets, batch_time, batch_size, True)
                self.train_df = pd.DataFrame()
            elif self.model.training:
                self.train_df = self.batch_recorder(sample, predictions, targets, batch_time, batch_size, False)
            else:
                self.val_df = self.batch_recorder(sample, predictions, targets, batch_time, batch_size, True)

        # Save model after an evaluation on the whole validation set
        if not self.model.training:
            state = {'epoch': self.epoch,
                     'iterations': iterations,
                     'val_loss': average_loss,
                     'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
            save_checkpoint(state, self.save_dict['save_path'], self.save_dict['custom_save'], self.model)

        return average_loss

    def whole_image_loop(self):
        start = time.time()
        time_sum, loss_sum = 0, 0

        average_loss = None

        for i, sample in enumerate(self.val_set, 1):
            grid_sampler = torchio.inference.GridSampler(
                sample, self.validation_dict['patch_size'], self.validation_dict['patch_overlap']
            )
            patch_loader = DataLoader(grid_sampler, batch_size=self.validation_dict['batch_size'])
            aggregator = torchio.inference.GridAggregator(
                sample, self.validation_dict['patch_overlap'], self.validation_dict['out_channels']
            )

            for patches_batch in patch_loader:
                # Take variables and make sure they are tensors on the right device
                volumes, _ = self.data_getter(patches_batch)
                locations = patches_batch[torchio.LOCATION]

                # Compute output
                predictions = self.model(volumes)
                aggregator.add_batch(predictions, locations)

            # Aggregate predictions for the whole image
            predictions = to_var(aggregator.get_output_tensor(), self.device)

            # Load target for the whole image
            _, target = self.data_getter(sample)

            # Compute loss
            sample_loss = 0
            for criterion in self.criteria:
                sample_loss += criterion(predictions.unsqueeze(0), target.unsqueeze(0))

            # Measure elapsed time
            sample_time = time.time() - start

            time_sum += sample_time
            loss_sum += sample_loss
            average_loss = loss_sum / i
            average_time = time_sum / i

            start = time.time()

            # Log validation information every log_frequency iterations
            if i % self.logger_dict['log_frequency'] == 0:
                to_log = summary(self.epoch, i, len(self.val_set), sample_loss, sample_time, average_loss,
                                 average_time, 'Val', 'Sample')
                self.logger.log(logging.INFO, to_log)

            # Record information about the sample and the performances of the model on this sample after every iteration
            self.inference_recorder(sample, predictions, target, sample_time)
        return average_loss

    def record_segmentation_batch(self, sample, predictions, targets, batch_time, batch_size, save=False):
        """
        Record information about the patches the model was trained or evaluated on during the segmentation task.
        At evaluation time, additional reporting metrics are recorded.
        """
        if self.model.training:
            model_mode = 'Train'
            df = self.train_df
        else:
            model_mode = 'Val'
            df = self.val_df

        location = sample.get('index_ini')
        shape = sample[self.label_key_name]['data'].shape
        size = np.product(shape[2:])
        history = sample.get('history')

        for idx in range(batch_size):
            info = {
                'name': sample['name'][idx],
                'image_filename': sample[self.image_key_name]['path'][idx],
                'label_filename': sample[self.label_key_name]['path'][idx],
                'shape': to_numpy(shape[2:]),
                'batch_time': batch_time,
                'batch_size': batch_size
            }

            for channel in list(range(shape[1])):
                info[f'occupied_volume{channel}'] = to_numpy(
                    sample[self.label_key_name]['data'][idx, channel].sum() / size
                )

            if location is not None:
                info['location'] = to_numpy(location[idx])

            loss = 0
            for criterion in self.criteria:
                loss += criterion(predictions[idx].unsqueeze(0), targets[idx].unsqueeze(0))
            info['loss'] = to_numpy(loss)

            if not self.model.training:
                for metric in self.metrics:
                    info[f'metric_{metric.__name__}'] = to_numpy(
                        metric(predictions[idx].unsqueeze(0), targets[idx].unsqueeze(0))
                    )

            if history is not None:
                for hist in history[idx]:
                    info[f'history_{hist[0]}'] = json.dumps(hist[1], cls=ArrayTensorJSONEncoder)

            df = df.append(info, ignore_index=True)

        if save:
            save_path = self.save_dict['save_path']
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            filename = f'{save_path}/{model_mode}_ep{self.epoch}_it{self.iteration}.csv'
            df.to_csv(filename)

        return df

    def record_segmentation_inference(self, sample, predictions, target, sample_time):
        """
        Record information about the samples the model was evaluated on during the segmentation task.
        """
        save_path = self.save_dict['save_path']
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        filename = f'{save_path}/Val_inference_ep{self.epoch}.csv'
        if Path(filename).is_file():
            df = pd.read_csv(filename, index_col=0)
        else:
            df = pd.DataFrame()

        shape = sample[self.label_key_name]['data'].shape
        size = np.product(shape[1:])

        info = {
            'name': sample['name'],
            'image_filename': sample[self.image_key_name]['path'],
            'label_filename': sample[self.label_key_name]['path'],
            'shape': to_numpy(shape[1:]),
            'sample_time': sample_time
        }

        for channel in list(range(shape[0])):
            info[f'occupied_volume{channel}'] = to_numpy(
                sample[self.label_key_name]['data'][channel].sum() / size
            )

        loss = 0
        for criterion in self.criteria:
            loss += criterion(predictions.unsqueeze(0), target.unsqueeze(0))
        info['loss'] = to_numpy(loss)

        for metric in self.metrics:
            info[f'metric_{metric.__name__}'] = to_numpy(metric(predictions.unsqueeze(0), target.unsqueeze(0)))

        df = df.append(info, ignore_index=True)
        df.to_csv(filename)
