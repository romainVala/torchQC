""" Run model and save it """

import json
import torch
import time
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import torchio
from torch.utils.data import DataLoader
from segmentation.utils import to_var, summary, save_checkpoint, to_numpy, get_class_name_from_method


class ArrayTensorJSONEncoder(json.JSONEncoder):
    """
    JSONEncoder extension to be able to stringify NumPy arrays and Torch tensors.
    """
    def default(self, o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()
        else:
            return json.JSONEncoder.default(self, o)


class RunModel:
    """
    Handle training, evaluation and saving of a model from a json configuration file.
    """
    def __init__(self, model, train_loader, val_loader, val_set, test_set, image_key_name, label_key_name,
                 logger, debug_logger, results_dir, batch_size, patch_size, struct):
        self.model = model
        self.device = next(model.parameters()).device

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_set = val_set
        self.test_set = test_set

        self.image_key_name = image_key_name
        self.label_key_name = label_key_name

        self.logger = logger
        self.debug_logger = debug_logger
        self.results_dir = results_dir

        self.batch_size = batch_size
        self.patch_size = patch_size

        # Retrieve information from structure
        self.criteria = struct['criteria']
        self.optimizer, self.lr_strategy, self.lr_strategy_attributes = self.get_optimizer(struct['optimizer'])
        self.log_frequency = struct['log_frequency']
        self.record_frequency = struct['save']['record_frequency']
        self.eval_frequency = struct['validation']['eval_frequency']
        self.whole_image_inference_frequency = struct['validation']['whole_image_inference_frequency']
        self.metrics = struct['validation']['reporting_metrics']
        self.patch_overlap = struct['validation']['patch_overlap']
        self.n_epochs = struct['n_epochs']
        self.seed = struct['seed']

        # Define which methods will be used to retrieve data and record information
        self.data_getter = getattr(self, struct['data_getter'])
        self.batch_recorder = getattr(self, struct['save']['batch_recorder'])
        self.prediction_saver = getattr(self, struct['save']['prediction_saver'])

        # Set attributes to keep track of information during training
        self.epoch = struct['current_epoch']
        self.iteration = 0

    def get_optimizer(self, optimizer_dict):
        optimizer_dict['attributes'].update({'params': self.model.parameters()})
        optimizer = optimizer_dict['optimizer_class'](**optimizer_dict['attributes'])
        return optimizer, optimizer_dict['learning_rate_strategy'], optimizer_dict['learning_rate_strategy_attributes']

    def get_segmentation_data(self, sample):
        volumes = sample[self.image_key_name]
        volumes = to_var(volumes[torchio.DATA].float(), self.device)

        if self.label_key_name in sample:
            targets = sample[self.label_key_name]
            targets = to_var(targets[torchio.DATA].float(), self.device)
        else:
            targets = None
        return volumes, targets

    def train(self):
        """ Train the model on the training set and evaluate it on the validation set. """
        # Set seed for reproducibility
        if self.seed is not None:
            torch.manual_seed(self.seed)

        for epoch in range(self.epoch, self.n_epochs + 1):
            self.epoch = epoch
            self.logger.log(logging.INFO, '******** Epoch [{}/{}]  ********'.format(self.epoch, self.n_epochs))

            # Train for one epoch
            self.model.train()
            self.train_loop()

            # Evaluate on whole images of the validation set
            with torch.no_grad():
                self.model.eval()
                if self.patch_size is not None and self.epoch % self.whole_image_inference_frequency == 0:
                    self.logger.log(logging.INFO, 'Validation')
                    self.whole_image_evaluation_loop()

                    # Save model after inference
                    state = {'epoch': self.epoch,
                             'iterations': self.iteration,
                             'val_loss': None,
                             'state_dict': self.model.state_dict(),
                             'optimizer': self.optimizer.state_dict() if self.optimizer is not None else {}}
                    save_checkpoint(state, self.results_dir, self.model)

            # Update learning rate
            if self.lr_strategy is not None:
                self.optimizer = self.lr_strategy(self.optimizer, **self.lr_strategy_attributes)

        # Save model at the end of training
        state = {'epoch': self.epoch,
                 'iterations': self.iteration,
                 'val_loss': None,
                 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict() if self.optimizer is not None else {}}
        save_checkpoint(state, self.results_dir, self.model)

    def eval(self):
        """ Evaluate the model on the validation set. """
        self.epoch -= 1
        self.logger.log(logging.INFO, 'Evaluation')
        with torch.no_grad():
            self.model.eval()
            if self.eval_frequency != np.inf:
                self.logger.log(logging.INFO, 'Evaluation on patches')
                self.train_loop(save_model=False)

            if self.patch_size is not None and self.whole_image_inference_frequency != np.inf:
                self.logger.log(logging.INFO, 'Evaluation on whole images')
                self.whole_image_evaluation_loop()

    def infer(self):
        """ Use the model to make predictions on the test set. """
        self.epoch -= 1
        self.logger.log(logging.INFO, 'Inference')
        with torch.no_grad():
            self.model.eval()
            self.inference_loop()

    def train_loop(self, save_model=True):
        if self.model.training:
            self.logger.log(logging.INFO, 'Training')
            model_mode = 'Train'
            loader = self.train_loader
        else:
            self.logger.log(logging.INFO, 'Validation')
            model_mode = 'Val'
            loader = self.val_loader

        df = pd.DataFrame()

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
            if i % self.log_frequency == 0:
                to_log = summary(self.epoch, i, len(loader), loss, batch_time, average_loss, average_time, model_mode)
                self.logger.log(logging.INFO, to_log)

            # Run model on validation set every eval_frequency iteration
            if self.model.training and i % self.eval_frequency == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.train_loop()
                    self.model.train()

            # Update DataFrame and record it every record_frequency iterations or every iteration at validation time
            if i % self.record_frequency == 0 or i == len(loader):
                df = self.batch_recorder(df, sample, predictions, targets, batch_time, True)
            else:
                df = self.batch_recorder(df, sample, predictions, targets, batch_time, False)

        # Save model after an evaluation on the whole validation set
        if save_model and not self.model.training:
            state = {'epoch': self.epoch,
                     'iterations': self.iteration,
                     'val_loss': average_loss,
                     'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict() if self.optimizer is not None else {}}
            save_checkpoint(state, self.results_dir, self.model)

        return average_loss

    def whole_image_evaluation_loop(self):
        df = pd.DataFrame()
        start = time.time()
        time_sum, loss_sum = 0, 0
        average_loss = None

        for i, sample in enumerate(self.val_set, 1):
            predictions = self.make_prediction_on_whole_volume(sample)

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
            if i % self.log_frequency == 0:
                to_log = summary(self.epoch, i, len(self.val_set), sample_loss, sample_time, average_loss,
                                 average_time, 'Val', 'Sample')
                self.logger.log(logging.INFO, to_log)

            # Record information about the sample and the performances of the model on this sample after every iteration
            df = self.batch_recorder(df, sample, predictions.unsqueeze(0), target.unsqueeze(0), sample_time, True)
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
                to_log = summary('/', i, len(self.test_set), '/', sample_time, '/', average_time, 'Val', 'Sample')
                self.logger.log(logging.INFO, to_log)

            self.prediction_saver(sample, predictions)

    def record_segmentation_batch(self, df, sample, predictions, targets, batch_time, save=False):
        """
        Record information about the batches the model was trained or evaluated on during the segmentation task.
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
        history = sample.get('history')
        batch_size = shape[0]

        for idx in range(batch_size):
            name = sample['name'][idx] if is_batch else sample['name']
            image_path = sample[self.image_key_name]['path'][idx] if is_batch else sample[self.image_key_name]['path']
            label_path = sample[self.label_key_name]['path'][idx] if is_batch else sample[self.label_key_name]['path']
            time_key = 'batch_time' if is_batch else 'sample_time'
            info = {
                'name': name,
                'image_filename': image_path,
                'label_filename': label_path,
                'shape': to_numpy(shape[2:]),
                time_key: batch_time
            }
            if is_batch:
                info['batch_size'] = batch_size

            for channel in list(range(shape[1])):
                info[f'occupied_volume{channel}'] = to_numpy(
                   targets[idx, channel].sum() / size
                )

            if location is not None:
                info['location'] = to_numpy(location[idx])

            loss = 0
            for criterion in self.criteria:
                loss += criterion(predictions[idx].unsqueeze(0), targets[idx].unsqueeze(0))
            info['loss'] = to_numpy(loss)

            if not self.model.training:
                for metric in self.metrics:
                    name = f'metric_{get_class_name_from_method(metric)}{metric.__name__}'
                    value = to_numpy(metric(predictions[idx].unsqueeze(0), targets[idx].unsqueeze(0)))
                    if value.size == 1:
                        info[name] = value
                    else:
                        for i, v in enumerate(value):
                            info[name] = v

            if history is not None:
                for hist in history[idx]:
                    info[f'history_{hist[0]}'] = json.dumps(hist[1], cls=ArrayTensorJSONEncoder)

            df = df.append(info, ignore_index=True)

        if save:
            if mode == 'Train':
                filename = f'{self.results_dir}/{mode}_ep{self.epoch}.csv'
            else:
                filename = f'{self.results_dir}/{mode}_ep{self.epoch}_it{self.iteration}.csv'
            df.to_csv(filename)

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

    def save_segmentation_prediction(self, sample, prediction):
        affine = sample[self.image_key_name]['affine']
        prediction = nib.Nifti1Image(to_numpy(prediction), affine)
        nib.save(prediction, f'{self.results_dir}/predictions_suj{sample["name"]}.nii.gz')

    def get_regress_random_noise_data(self, data):
        return self.get_regression_data(data, 'random_noise')

    def get_regress_motion_data(self, data):
        return self.get_regression_data(data, 'ssim')

    def get_regression_data(self, data, target):

        if isinstance(data, list):  # case where callate_fn is used
            inputs = torch.cat([sample[self.image_key_name]['data'].unsqueeze(0) for sample in data])
        else:
            inputs = data[self.image_key_name]['data']

        if target == 'ssim':
            labels = data[self.image_key_name]['metrics']['ssim'].unsqueeze(1)
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

        mode = 'Train' if self.model.training else 'Val'

        location = sample.get('index_ini')
        shape = sample[self.image_key_name]['data'].shape
        history = sample.get('history')
        batch_size = shape[0]

        for idx in range(batch_size):
            info = {
                'name': sample['name'][idx],
                'image_filename': sample[self.image_key_name]['path'][idx],
                'shape': to_numpy(shape[2:]),
                'batch_time': batch_time,
                'batch_size': batch_size
            }

            if location is not None:
                info['location'] = to_numpy(location[idx])

            loss = 0
            for criterion in self.criteria:
                loss += criterion(predictions[idx].unsqueeze(0), targets[idx].unsqueeze(0))
            info['loss'] = to_numpy(loss)
            info['prediction'] = to_numpy(predictions[idx])[0]
            info['targets'] = to_numpy(targets[idx])[0]

            if 'metrics' in sample[self.image_key_name]:
                #dicm = sample[self.image_key_name]['metrics']
                dics = sample[self.image_key_name]['simu_param']
                dicm={}
                for key,val in dics.items():
                    dicm[key] = to_numpy(val[idx])
                info.update(dicm)

            if not self.model.training:
                for metric in self.metrics:
                    info[f'metric_{metric.__name__}'] = to_numpy(
                        metric(predictions[idx].unsqueeze(0), targets[idx].unsqueeze(0))
                    )

            if history is not None:
                for hist in history[idx]:
                    info['T_{}'.format(hist[0])] = json.dumps(hist[1], cls=ArrayTensorJSONEncoder)

            df = df.append(info, ignore_index=True)

        if save:
            if mode == 'Train':
                filename = '{}/{}_ep{:03d}.csv'.format(self.results_dir, mode, self.epoch)
            else:
                filename = '{}/{}_ep{:03d}_it{:04d}.csv'.format(self.results_dir, mode, self.epoch, self.iteration)

            df.to_csv(filename)

        return df
