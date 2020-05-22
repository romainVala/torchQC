""" Load model, use it and save it """

import json
import torch
import time
import logging
from segmentation.utils import parse_object_import, parse_function_import, to_var, summary, save_checkpoint,\
    instantiate_logger


default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(folder, model_filename='model.json', device=default_device):
    with open(folder + model_filename) as file:
        info = json.load(file)

    model = info.get('model')
    load = model.get('load')

    model, model_class = parse_object_import(model)

    if load is not None:
        if load.get('custom'):
            model, params = model_class.load(load.get('path'))
        else:
            model.load_state_dict(torch.load(load.get('path')))

    return model.to(device)


def train_loop(loader, model, criteria, optimizer, epoch, device, batch_size, log_frequency, logger):
    model_mode = 'Train' if model.training else 'Val'

    start = time.time()
    time_sum, loss_sum = 0, 0

    loss, volumes, masks, pred_masks = None, None, None, None

    for i, sample in enumerate(loader, 1):
        # Take variable and put them to GPU
        (volumes, masks, patients) = sample

        volumes = to_var(volumes.float(), device)
        masks = to_var(masks.float(), device)

        # compute output
        pred_masks = model(volumes)

        # compute loss
        loss = torch.tensor(0)
        for criterion in criteria:
            loss += criterion(pred_masks, masks)

        # compute gradient and do SGD step
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time = time.time() - start

        time_sum += batch_size * batch_time
        loss_sum += batch_size * loss
        average_loss = loss_sum / (i * batch_size)
        average_time = time_sum / (i * batch_size)

        start = time.time()

        if i % log_frequency == 0:
            to_log = summary(epoch + 1, i, len(loader), loss, batch_time, average_loss, average_time, model_mode)
            logger.log(logging.INFO, to_log)
    return loss


def train(model, train_loader, val_loader, batch_size, folder, train_filename='train.json', n_epochs=10):
    def parse_criteria(criterion_list):
        c_list = []
        for criterion in criterion_list:
            c = parse_function_import(criterion)
            c_list.append(c)
        return c_list

    def parse_optimizer(optimizer_dict):
        o = parse_object_import(optimizer_dict)
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

    device = next(model.parameters()).device
    with open(folder + train_filename) as file:
        info = json.load(file)

    criteria = parse_criteria(info.get('criteria'))
    optimizer, learning_rate_strategy, learning_rate_strategy_attributes = parse_optimizer(info.get('optimizer'))
    logger, log_frequency, log_filename = parse_logger(info.get('logger'))
    save_model, save_frequency, save_path, custom_save = parse_save(info.get('save'))
    title = info.get('title') or 'Session'
    seed = info.get('seed')

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
        train_loop(train_loader, model, criteria, optimizer, epoch, device, batch_size, log_frequency, logger)

        # Evaluate on validation set
        logger.log(logging.INFO, 'Validation')
        with torch.no_grad():
            model.eval()
            val_loss = train_loop(
                val_loader, model, criteria, optimizer, epoch, device, batch_size, log_frequency, logger
            )
            if save_model and epoch % save_frequency == 0:
                state = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'val_loss': val_loss,
                         'optimizer': optimizer.state_dict()}
                save_checkpoint(state, save_path, custom_save, model)

        # Update learning rate
        if learning_rate_strategy is not None:
            optimizer = learning_rate_strategy(optimizer, log_filename, **learning_rate_strategy_attributes)
