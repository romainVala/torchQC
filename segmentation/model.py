""" Load model, use it and save it """

import json
import torch
import time
import logging
import torchio
from torch.utils.data import DataLoader
from segmentation.utils import parse_object_import, parse_function_import, to_var, summary, save_checkpoint, \
    instantiate_logger


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


def train_loop(loader, model, criteria, optimizer, epoch, device, log_frequency, logger, image_key_name='image',
               label_key_name='label'):
    batch_size = loader.batch_size
    model_mode = 'Train' if model.training else 'Val'

    start = time.time()
    time_sum, loss_sum = 0, 0

    average_loss = None

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
            to_log = summary(epoch + 1, i, len(loader), loss, batch_time, average_loss, average_time, model_mode)
            logger.log(logging.INFO, to_log)
    return average_loss


def validation_loop(dataset, model, criteria, optimizer, epoch, device, log_frequency, logger, batch_size, patch_size,
                    patch_overlap, out_channels, image_key_name='image', label_key_name='label'):
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
                epoch + 1, i, len(dataset), sample_loss, sample_time, average_loss, average_time, model_mode, 'Sample'
            )
            logger.log(logging.INFO, to_log)
    return average_loss


def train(model, train_loader, val_loader, folder, train_filename='train.json'):
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
        return save, frequency, path, custom

    def parse_validation(validation_dict):
        infer = validation_dict.get('infer_on_whole_image')
        size = validation_dict.get('patch_size')
        overlap = validation_dict.get('patch_overlap')
        channels = validation_dict.get('out_channels')
        batch = validation_dict.get('batch_size')
        return infer, size, overlap, channels, batch

    device = next(model.parameters()).device
    with open(folder + train_filename) as file:
        info = json.load(file)

    criteria = parse_criteria(info.get('criteria'))
    optimizer, learning_rate_strategy, learning_rate_strategy_attributes = parse_optimizer(info.get('optimizer'), model)
    logger, log_frequency, log_filename = parse_logger(info.get('logger'))
    save_model, save_frequency, save_path, custom_save = parse_save(info.get('save'))
    infer_on_whole_image, patch_size, patch_overlap, out_channels, batch_size = parse_validation(info.get('validation'))
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

    for epoch in range(n_epochs):
        logger.log(logging.INFO, '******** Epoch [{}/{}]  ********'.format(epoch + 1, n_epochs))

        # Train for one epoch
        model.train()
        logger.log(logging.INFO, 'Training')
        train_loop(train_loader, model, criteria, optimizer, epoch, device, log_frequency, logger, image_key_name,
                   label_key_name)

        # Evaluate on validation set
        logger.log(logging.INFO, 'Validation')
        with torch.no_grad():
            model.eval()
            if infer_on_whole_image:
                val_loss = validation_loop(val_loader, model, criteria, optimizer, epoch, device, log_frequency, logger,
                                           batch_size, patch_size, patch_overlap, out_channels, image_key_name,
                                           label_key_name)
            else:
                val_loss = train_loop(val_loader, model, criteria, optimizer, epoch, device, log_frequency, logger,
                                      image_key_name, label_key_name)
            if save_model and epoch % save_frequency == 0:
                state = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'val_loss': val_loss,
                         'optimizer': optimizer.state_dict()}
                save_checkpoint(state, save_path, custom_save, model)

        # Update learning rate
        if learning_rate_strategy is not None:
            optimizer = learning_rate_strategy(optimizer, log_filename, **learning_rate_strategy_attributes)
