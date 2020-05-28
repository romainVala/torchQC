""" Load model, use it and save it """

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
    instantiate_logger, to_numpy
from segmentation.visualization import report_loss


def load_model(folder, model_filename='model.json'):
    with open(folder + model_filename) as file:
        info = json.load(file)

    model = info.get('model')
    load = model.get('load')

    device_name = info.get('device') or 'cuda'
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model, model_class = parse_object_import(model)

    if load is not None:
        if load.get('custom'):
            model, params = model_class.load(load.get('path'))
        else:
            model.load_state_dict(torch.load(load.get('path')))

    return model.to(device)


def train_loop(loader, model, criteria, optimizer, epoch, device, log_frequency, logger, save_path,
               image_key_name='image', label_key_name='label', reporting_metrics=None, eval_loader=None,
               eval_frequency=None, iterations=None, record_frequency=1, custom_save=False):
    batch_size = loader.batch_size
    model_mode = 'Train' if model.training else 'Val'

    if eval_frequency is None:
        eval_frequency = np.inf

    start = time.time()
    time_sum, loss_sum = 0, 0

    average_loss = None
    df = pd.DataFrame()

    for i, sample in enumerate(loader, 1):
        # Take variables and make sure they are tensors on the right device
        volumes = sample[image_key_name]
        targets = sample[label_key_name]

        volumes = to_var(volumes[torchio.DATA].float(), device)
        targets = to_var(targets[torchio.DATA].float(), device)

        # Compute output
        pred_targets = model(volumes)

        # Compute loss
        loss = 0
        for criterion in criteria:
            loss += criterion(pred_targets, targets)

        # Compute gradient and do SGD step
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Measure elapsed time
        batch_time = time.time() - start

        time_sum += batch_size * batch_time
        loss_sum += batch_size * loss
        average_loss = loss_sum / (i * batch_size)
        average_time = time_sum / (i * batch_size)

        start = time.time()

        if i % log_frequency == 0:
            to_log = summary(epoch, i, len(loader), loss, batch_time, average_loss, average_time, model_mode)
            logger.log(logging.INFO, to_log)

        if i % eval_frequency == 0:
            with torch.no_grad():
                model.eval()
                train_loop(eval_loader, model, criteria, optimizer, epoch, device, log_frequency, logger, save_path,
                           image_key_name, label_key_name, reporting_metrics, iterations=i, custom_save=custom_save)
                model.train()

        if i % record_frequency == 0:
            df = record_batch(df, save_path, i, sample, pred_targets, targets, batch_time, batch_size, model_mode,
                              criteria, epoch, image_key_name, label_key_name, reporting_metrics, iterations, save=True)

            if model_mode == 'Train':
                df = pd.DataFrame()
        else:
            df = record_batch(df, save_path, i, sample, pred_targets, targets, batch_time, batch_size, model_mode,
                              criteria, epoch, image_key_name, label_key_name, reporting_metrics, iterations,
                              save=False)

    if model_mode == 'Val':
        state = {'epoch': epoch,
                 'iterations': iterations,
                 'val_loss': average_loss,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        save_checkpoint(state, save_path, custom_save, model)

    return average_loss


def whole_image_loop(dataset, model, criteria, optimizer, epoch, device, log_frequency, logger, batch_size, patch_size,
                     patch_overlap, out_channels, save_path, image_key_name='image', label_key_name='label',
                     reporting_metrics=None):
    model_mode = 'Val'

    start = time.time()
    time_sum, loss_sum = 0, 0

    average_loss = None

    for i, sample in enumerate(dataset, 1):
        grid_sampler = torchio.inference.GridSampler(sample, patch_size, patch_overlap)
        patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
        aggregator = torchio.inference.GridAggregator(sample, patch_overlap, out_channels)

        for patches_batch in patch_loader:
            # Take variables and make sure they are tensors on the right device
            volumes = patches_batch[image_key_name]
            locations = patches_batch[torchio.LOCATION]

            volumes = to_var(volumes[torchio.DATA].float(), device)

            # Compute output
            pred_targets = model(volumes)
            aggregator.add_batch(pred_targets, locations)

        # Aggregate predictions for the whole image
        pred_targets = to_var(aggregator.get_output_tensor(), device)

        # Load target for the whole image
        target = sample[label_key_name]
        target = to_var(target[torchio.DATA].float(), device)

        # Compute loss
        sample_loss = 0
        for criterion in criteria:
            sample_loss += criterion(pred_targets, target)

        # Compute gradient and do SGD step
        if model.training:
            optimizer.zero_grad()
            sample_loss.backward()
            optimizer.step()

        # Measure elapsed time
        sample_time = time.time() - start

        time_sum += sample_time
        loss_sum += sample_loss
        average_loss = loss_sum / i
        average_time = time_sum / i

        start = time.time()

        if i % log_frequency == 0:
            to_log = summary(
                epoch, i, len(dataset), sample_loss, sample_time, average_loss, average_time, model_mode, 'Sample'
            )
            logger.log(logging.INFO, to_log)

        record_whole_image(save_path, sample, pred_targets, target, sample_time, model_mode, criteria, epoch,
                           image_key_name, label_key_name, reporting_metrics)
    return average_loss


def train(model, train_loader, val_loader, val_set, folder, train_filename='train.json'):
    def parse_criteria(criterion_list):
        c_list = []
        for criterion in criterion_list:
            c = parse_function_import(criterion)
            c_list.append(c)
        return c_list

    def parse_optimizer(optimizer_dict, model):
        attributes = optimizer_dict.get('attributes') or {}
        attributes.update({'params': model.parameters()})
        o, _ = parse_object_import(optimizer_dict)
        strategy = optimizer_dict.get('learning_rate_strategy')
        strategy_attributes = optimizer_dict.get('learning_rate_strategy_attributes') or {}
        if strategy is not None:
            strategy = parse_function_import(strategy)
        return o, strategy, strategy_attributes

    def parse_logger(logger_dict):
        frequency = logger_dict.get('log_frequency')
        filename = logger_dict.get('filename')
        name = logger_dict.get('name')
        level = logger_dict.get('level') or logging.INFO
        logger_object = instantiate_logger(name, level, filename)
        return logger_object, frequency, filename

    def parse_save(save_dict):
        save = save_dict.get('save_model')
        frequency = save_dict.get('save_frequency')
        path = save_dict.get('save_path')
        custom = save_dict.get('custom_save')
        rec_frequency = save_dict.get('record_frequency')
        return save, frequency, path, custom, rec_frequency

    def parse_validation(validation_dict):
        inference_frequency = validation_dict.get('whole_image_inference_frequency') or np.inf
        size = validation_dict.get('patch_size')
        overlap = validation_dict.get('patch_overlap')
        channels = validation_dict.get('out_channels')
        batch = validation_dict.get('batch_size')
        metrics = validation_dict.get('reporting_metrics') or []
        metrics = parse_criteria(metrics)
        val_frequency = validation_dict.get('eval_frequency')
        return inference_frequency, size, overlap, channels, batch, metrics, val_frequency

    device = next(model.parameters()).device
    with open(folder + train_filename) as file:
        info = json.load(file)

    criteria = parse_criteria(info.get('criteria'))
    optimizer, learning_rate_strategy, learning_rate_strategy_attributes = parse_optimizer(info.get('optimizer'), model)
    logger, log_frequency, log_filename = parse_logger(info.get('logger'))
    save_model, save_frequency, save_path, custom_save, record_frequency = parse_save(info.get('save'))
    whole_image_inference_frequency, patch_size, patch_overlap, out_channels, \
        batch_size, reporting_metrics, eval_frequency = parse_validation(info.get('validation'))
    n_epochs = info.get('n_epochs')
    title = info.get('title') or 'Session'
    seed = info.get('seed')
    image_key_name = info.get('image_key_name')
    label_key_name = info.get('label_key_name')

    session_name = f'{title}_{time.strftime("%m.%d %Hh%M")}'
    logger.log(logging.INFO, session_name)

    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    for epoch in range(1, n_epochs + 1):
        logger.log(logging.INFO, '******** Epoch [{}/{}]  ********'.format(epoch, n_epochs))

        # Train for one epoch
        model.train()
        logger.log(logging.INFO, 'Training')
        train_loop(train_loader, model, criteria, optimizer, epoch, device, log_frequency, logger, save_path,
                   image_key_name, label_key_name, reporting_metrics, val_loader, eval_frequency,
                   record_frequency=record_frequency, custom_save=custom_save)

        # Evaluate on whole images of the validation set
        with torch.no_grad():
            model.eval()
            if epoch % whole_image_inference_frequency == 0:
                logger.log(logging.INFO, 'Validation')
                whole_image_loop(val_set, model, criteria, optimizer, epoch, device, log_frequency, logger, batch_size,
                                 patch_size, patch_overlap, out_channels, save_path, image_key_name, label_key_name,
                                 reporting_metrics)

        # Save model
        if save_model and epoch % save_frequency == 0:
            state = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            save_checkpoint(state, save_path, custom_save, model)

        # Update learning rate
        if learning_rate_strategy is not None:
            optimizer = learning_rate_strategy(optimizer, log_filename, **learning_rate_strategy_attributes)

    # Report loss
    report_loss(save_path)


def record_batch(df, save_path, i, sample, pred_targets, targets, batch_time, batch_size, mode, criteria, epoch,
                 image_key_name='t1', label_key_name='label', reporting_metrics=None, iterations=None, save=False):
    location = sample.get('index_ini')
    shape = sample[label_key_name]['data'].shape[2:]
    size = np.product(shape)

    for idx in range(batch_size):
        info = {
            'name': sample['name'][idx],
            'image_filename': sample[image_key_name]['path'][idx],
            'label_filename': sample[label_key_name]['path'][idx],
            'shape': to_numpy(shape),
            'occupied_volume': to_numpy(sample[label_key_name]['data'][idx].sum() / size),
            'batch_time': batch_time,
            'batch_size': batch_size
        }

        if location is not None:
            info['location'] = to_numpy(location[idx])

        loss = 0
        for criterion in criteria:
            loss += criterion(pred_targets[idx], targets[idx])
        info['loss'] = to_numpy(loss)

        if mode == 'Val':
            for metric in reporting_metrics:
                info[f'metric_{metric.__name__}'] = to_numpy(metric(pred_targets[idx], targets[idx]))

        df = df.append(info, ignore_index=True)

    if save:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        filename = f'{save_path}/{mode}_ep{epoch}_it{i if iterations is None else iterations}.csv'
        df.to_csv(filename)

    return df


def record_whole_image(save_path, sample, pred_targets, targets, sample_time, mode, criteria, epoch,
                       image_key_name='t1', label_key_name='label', reporting_metrics=None):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    filename = f'{save_path}/{mode}_inference_ep{epoch}.csv'
    if Path(filename).is_file():
        df = pd.read_csv(filename, index_col=0)
    else:
        df = pd.DataFrame()

    shape = sample[label_key_name]['data'].shape[1:]
    size = np.product(shape)

    info = {
        'name': sample['name'],
        'image_filename': sample[image_key_name]['path'],
        'label_filename': sample[label_key_name]['path'],
        'shape': to_numpy(shape),
        'occupied_volume': to_numpy(sample[label_key_name]['data'].sum() / size),
        'sample_time': sample_time
    }

    loss = 0
    for criterion in criteria:
        loss += criterion(pred_targets, targets)
    info['loss'] = to_numpy(loss)

    if mode == 'Val':
        for metric in reporting_metrics:
            info[f'metric_{metric.__name__}'] = to_numpy(metric(pred_targets, targets))

    df = df.append(info, ignore_index=True)

    df.to_csv(filename)
