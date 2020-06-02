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
    instantiate_logger, to_numpy, set_dict_value, check_mandatory_keys
from segmentation.visualization import report_loss


class ArrayTensorJSONEncoder(json.JSONEncoder):
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

    @staticmethod
    def parse_criteria(criterion_list):
        c_list = []
        for criterion in criterion_list:
            c = parse_function_import(criterion)
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

    def train(self):
        # Log session title
        session_name = f'{self.title}_{time.strftime("%m.%d %Hh%M")}'
        self.logger.log(logging.INFO, session_name)

        # Set seed for reproducibility
        if self.seed is not None:
            torch.manual_seed(self.seed)

        for epoch in range(1, self.n_epochs + 1):
            self.logger.log(logging.INFO, '******** Epoch [{}/{}]  ********'.format(epoch, self.n_epochs))

            # Train for one epoch
            self.model.train()
            self.logger.log(logging.INFO, 'Training')
            self.train_loop(epoch)

            # Evaluate on whole images of the validation set
            with torch.no_grad():
                self.model.eval()
                if epoch % self.validation_dict['whole_image_inference_frequency'] == 0:
                    self.logger.log(logging.INFO, 'Validation')
                    self.whole_image_loop(epoch)

            # Save model
            if self.save_dict['save_model'] and epoch % self.save_dict['save_frequency'] == 0:
                state = {'epoch': epoch,
                         'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict()}
                save_checkpoint(state, self.save_dict['save_path'], self.save_dict['custom_save'], self.model)

            # Update learning rate
            if self.lr_strategy is not None:
                self.optimizer = self.lr_strategy(self.optimizer, self.logger_dict['log_filename'],
                                                  **self.lr_strategy_attributes)

        # Report loss
        report_loss(self.save_dict['save_path'])

    def train_loop(self, epoch, iterations=None):
        if self.model.training:
            model_mode = 'Train'
            loader = self.train_loader
        else:
            model_mode = 'Val'
            loader = self.val_loader

        batch_size = loader.batch_size

        start = time.time()
        time_sum, loss_sum = 0, 0

        average_loss = None
        df = pd.DataFrame()

        for i, sample in enumerate(loader, 1):
            # Take variables and make sure they are tensors on the right device
            volumes = sample[self.image_key_name]
            targets = sample[self.label_key_name]

            volumes = to_var(volumes[torchio.DATA].float(), self.device)
            targets = to_var(targets[torchio.DATA].float(), self.device)

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
                to_log = summary(epoch, i, len(loader), loss, batch_time, average_loss, average_time, model_mode)
                self.logger.log(logging.INFO, to_log)

            # Run model on validation set every eval_frequency iteration
            if self.model.training and i % self.validation_dict['eval_frequency'] == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.train_loop(epoch, iterations=i)
                    self.model.train()

            # Update DataFrame and record it every record_frequency iterations or every iteration at validation time
            if self.model.training and i % self.save_dict['record_frequency'] == 0:
                self.record_batch(df, i, sample, predictions, targets, batch_time, batch_size, epoch, True)
                df = pd.DataFrame()
            elif self.model.training:
                df = self.record_batch(df, i, sample, predictions, targets, batch_time, batch_size, epoch, False)
            else:
                df = self.record_batch(
                    df, iterations, sample, predictions, targets, batch_time, batch_size, epoch, True
                )

        # Save model after an evaluation on the whole validation set
        if not self.model.training:
            state = {'epoch': epoch,
                     'iterations': iterations,
                     'val_loss': average_loss,
                     'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
            save_checkpoint(state, self.save_dict['save_path'], self.save_dict['custom_save'], self.model)

        return average_loss

    def whole_image_loop(self, epoch):
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
                volumes = patches_batch[self.image_key_name]
                locations = patches_batch[torchio.LOCATION]

                volumes = to_var(volumes[torchio.DATA].float(), self.device)

                # Compute output
                predictions = self.model(volumes)
                aggregator.add_batch(predictions, locations)

            # Aggregate predictions for the whole image
            predictions = to_var(aggregator.get_output_tensor(), self.device)

            # Load target for the whole image
            target = sample[self.label_key_name]
            target = to_var(target[torchio.DATA].float(), self.device)

            # Compute loss
            sample_loss = 0
            for criterion in self.criteria:
                sample_loss += criterion(predictions, target)

            # Measure elapsed time
            sample_time = time.time() - start

            time_sum += sample_time
            loss_sum += sample_loss
            average_loss = loss_sum / i
            average_time = time_sum / i

            start = time.time()

            # Log validation information every log_frequency iterations
            if i % self.logger_dict['log_frequency'] == 0:
                to_log = summary(
                    epoch, i, len(self.val_set), sample_loss, sample_time, average_loss, average_time, 'Val', 'Sample'
                )
                self.logger.log(logging.INFO, to_log)

            # Record information about the sample and the performances of the model on this sample after every iteration
            self.record_whole_image(sample, predictions, target, sample_time, epoch)
        return average_loss

    def record_batch(self, df, i, sample, predictions, targets, batch_time, batch_size, epoch, save=False):
        model_mode = 'Train' if self.model.training else 'Val'
        location = sample.get('index_ini')
        shape = sample[self.label_key_name]['data'].shape[2:]
        size = np.product(shape)
        history = sample.get('history')

        for idx in range(batch_size):
            info = {
                'name': sample['name'][idx],
                'image_filename': sample[self.image_key_name]['path'][idx],
                'label_filename': sample[self.label_key_name]['path'][idx],
                'shape': to_numpy(shape),
                'occupied_volume': to_numpy(sample[self.label_key_name]['data'][idx].sum() / size),
                'batch_time': batch_time,
                'batch_size': batch_size
            }

            if location is not None:
                info['location'] = to_numpy(location[idx])

            loss = 0
            for criterion in self.criteria:
                loss += criterion(predictions[idx], targets[idx])
            info['loss'] = to_numpy(loss)

            if not self.model.training:
                for metric in self.metrics:
                    info[f'metric_{metric.__name__}'] = to_numpy(metric(predictions[idx], targets[idx]))

            if history is not None:
                for hist in history[idx]:
                    info[f'history_{hist[0]}'] = json.dumps(hist[1], cls=ArrayTensorJSONEncoder)

            df = df.append(info, ignore_index=True)

        if save:
            save_path = self.save_dict['save_path']
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            filename = f'{save_path}/{model_mode}_ep{epoch}_it{i}.csv'
            df.to_csv(filename)

        return df

    def record_whole_image(self, sample, predictions, target, sample_time, epoch):
        save_path = self.save_dict['save_path']
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        filename = f'{save_path}/Val_inference_ep{epoch}.csv'
        if Path(filename).is_file():
            df = pd.read_csv(filename, index_col=0)
        else:
            df = pd.DataFrame()

        shape = sample[self.label_key_name]['data'].shape[1:]
        size = np.product(shape)

        info = {
            'name': sample['name'],
            'image_filename': sample[self.image_key_name]['path'],
            'label_filename': sample[self.label_key_name]['path'],
            'shape': to_numpy(shape),
            'occupied_volume': to_numpy(sample[self.label_key_name]['data'].sum() / size),
            'sample_time': sample_time
        }

        loss = 0
        for criterion in self.criteria:
            loss += criterion(predictions, target)
        info['loss'] = to_numpy(loss)

        for metric in self.metrics:
            info[f'metric_{metric.__name__}'] = to_numpy(metric(predictions, target))

        df = df.append(info, ignore_index=True)
        df.to_csv(filename)
