""" Load and preprocess data """

import glob
import json
import re
import multiprocessing
import numpy as np
import torchio
from torch.utils.data import DataLoader


def load_data(folder, data_filename='data.json'):
    def update_subject(sub, mods, mod_name, mod_path):
        subject_type = mods.get(mod_name).get('type')
        subject_attributes = mods.get(mod_name).get('attributes') or {}

        sub.update({
            mod_name: torchio.Image(mod_path, subject_type, **subject_attributes)
        })

    def get_relevant_list(default_list, train_list, val_list, test_list, list_name=None):
        if list_name == 'train':
            return train_list
        if list_name == 'val':
            return val_list
        if list_name == 'test':
            return test_list
        return default_list

    def dict_to_subject_list(subject_dict):
        return list(map(lambda s: torchio.Subject(s), subject_dict.values()))

    def get_name(name_pattern, string):
        if name_pattern is None:
            string_split = string.split('/')
            return string_split[-1] if len(string_split[-1]) > 0 else string_split[-2]
        else:
            matches = re.findall(name_pattern, string)
            return matches[-1]

    with open(folder + data_filename) as file:
        info = json.load(file)

    modalities = info.get('modalities')
    patterns = info.get('patterns') or []
    paths = info.get('paths') or []
    train_val_test_repartition = info.get('repartition') or [0.7, 0.15, 0.15]
    assert(sum(train_val_test_repartition) == 1)
    seed = info.get('seed')

    subjects, train_subjects, val_subjects, test_subjects = {}, {}, {}, {}

    # Using patterns
    for pattern in patterns:
        root = pattern.get('root')
        list_name = pattern.get('list_name')
        relevant_dict = get_relevant_list(subjects, train_subjects, val_subjects, test_subjects, list_name)
        for folder_path in glob.glob(root):
            name = get_name(pattern.get('name_pattern'), folder_path)
            subject = relevant_dict.get(name) or {}

            for modality_name, modality_path in pattern.get('modalities').items():
                modality_path = glob.glob(folder_path + modality_path)[0]
                update_subject(subject, modalities, modality_name, modality_path)

            subject['name'] = name
            relevant_dict[name] = subject

    # Using paths
    for path in paths:
        name = path.get('name') or f'{len(subjects) + len(train_subjects) + len(val_subjects) + len(test_subjects):0>6}'
        list_name = path.get('list_name')
        relevant_dict = get_relevant_list(subjects, train_subjects, val_subjects, test_subjects, list_name)
        subject = relevant_dict.get(name) or {}

        for modality_name, modality_path in path.get('modalities').items():
            update_subject(subject, modalities, modality_name, modality_path)

        subject['name'] = name
        relevant_dict[name] = subject

    subjects = dict_to_subject_list(subjects)
    train_subjects = dict_to_subject_list(train_subjects)
    val_subjects = dict_to_subject_list(val_subjects)
    test_subjects = dict_to_subject_list(test_subjects)

    np.random.seed(seed)
    np.random.shuffle(subjects)
    n_subjects = len(subjects)

    end_train = int(round(train_val_test_repartition[0] * n_subjects))
    end_val = end_train + int(round(train_val_test_repartition[1] * n_subjects))
    train_subjects += subjects[:end_train]
    val_subjects += subjects[end_train:end_val]
    test_subjects += subjects[end_val:]

    return train_subjects, val_subjects, test_subjects


def generate_dataset(subjects, folder, transform_filename='transform.json'):
    with open(folder + transform_filename) as file:
        info = json.load(file)
        transforms = info.get('transforms')
        selection = info.get('selection')

    transform_list = []
    for transform in transforms:
        transform_name = transform.get('name')
        transform_attributes = transform.get('attributes') or {}
        transform_list.append(getattr(torchio.transforms, transform_name)(**transform_attributes))

    selection_name = selection.get('name')
    selection_attributes = selection.get('attributes') or {}
    transform = getattr(torchio.transforms, selection_name)(transform_list, **selection_attributes)

    dataset = torchio.ImagesDataset(subjects, transform=transform)

    return dataset


def generate_dataloader(dataset, folder, loader_filename='loader.json', train=True):
    with open(folder + loader_filename) as file:
        info = json.load(file)

    batch_size = info.get('batch_size')
    num_workers = info.get('num_workers') or 0
    queue = info.get('queue')
    shuffle = info.get('shuffle') or True

    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()

    if queue is None:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        if train:
            queue_attributes = queue.get('attributes')
            sampler_class = queue.get('sampler_class')
            sampler_class = getattr(torchio.data.sampler, sampler_class)
            queue_attributes.update({'num_workers': num_workers, 'sampler_class': sampler_class})
            queue = torchio.Queue(dataset, **queue_attributes)
            loader = DataLoader(queue, batch_size)
        else:
            loader = dataset

    return loader
