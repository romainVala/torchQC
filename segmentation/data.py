""" Load and preprocess data """

import glob
import json
import re
import multiprocessing
import numpy as np
import torchio
from torch.utils.data import DataLoader
from segmentation.utils import parse_function_import, set_dict_value, check_mandatory_keys


DATA_LOADING_KEYS = ['modalities']
PATTERN_KEYS = ['root', 'modalities']
PATH_KEYS = ['name', 'modalities']
DATASET_KEYS = ['train_transforms', 'val_transforms']
DATA_LOADER_KEYS = ['batch_size']


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

    def dict_to_subject_list(subject_dict, ref_mods):
        subject_list = []
        for n, s in subject_dict.items():
            if not set(s.keys()).issuperset(ref_mods.keys()):
                raise KeyError(f'A modality is missing for subject {n}, {s.keys()} were found but '
                               f'at least {ref_mods.keys()} were expected')
            subject_list.append(torchio.Subject(s))
        return subject_list

    def get_name(name_pattern, string):
        if name_pattern is None:
            string_split = string.split('/')
            return string_split[-1] if len(string_split[-1]) > 0 else string_split[-2]
        else:
            matches = re.findall(name_pattern, string)
            return matches[-1]

    def check_modalities(ref_mods, mods, subject_name):
        if not set(mods.keys()).issubset(ref_mods.keys()):
            raise KeyError(f'At least one modality of {mods.keys()} from {subject_name}  is not in the reference '
                           f'modalities {ref_mods.keys()}')

    with open(folder + data_filename) as file:
        info = json.load(file)

    # Make sure keys are present in the info dictionary
    check_mandatory_keys(info, DATA_LOADING_KEYS, folder + data_filename)
    set_dict_value(info, 'patterns', [])
    set_dict_value(info, 'paths', [])
    set_dict_value(info, 'shuffle')
    set_dict_value(info, 'seed')
    set_dict_value(info, 'repartition', [0.7, 0.15, 0.15])
    assert (sum(info['repartition']) == 1)

    subjects, train_subjects, val_subjects, test_subjects = {}, {}, {}, {}

    # Retrieve subjects using patterns
    for pattern in info['patterns']:
        check_mandatory_keys(pattern, PATTERN_KEYS, 'pattern dictionary')
        set_dict_value(pattern, 'list_name')
        set_dict_value(pattern, 'name_pattern')
        check_modalities(info['modalities'], pattern['modalities'], pattern['root'])

        relevant_dict = get_relevant_list(subjects, train_subjects, val_subjects, test_subjects, pattern['list_name'])
        for folder_path in glob.glob(pattern['root']):
            name = get_name(pattern['name_pattern'], folder_path)
            subject = relevant_dict.get(name) or {}

            for modality_name, modality_path in pattern['modalities'].items():
                modality_path = glob.glob(folder_path + modality_path)[0]
                update_subject(subject, info['modalities'], modality_name, modality_path)

            subject['name'] = name
            relevant_dict[name] = subject

    # Retrieve subjects using paths
    for path in info['paths']:
        check_mandatory_keys(path, PATH_KEYS, 'path dictionary')
        default_name = f'{len(subjects) + len(train_subjects) + len(val_subjects) + len(test_subjects):0>6}'
        set_dict_value(path, 'name', default_name)
        set_dict_value(path, 'list_name')
        check_modalities(info['modalities'], path['modalities'], path['name'])

        name = path['name']
        relevant_dict = get_relevant_list(subjects, train_subjects, val_subjects, test_subjects, path['list_name'])
        subject = relevant_dict.get(name) or {}

        for modality_name, modality_path in path['modalities'].items():
            update_subject(subject, info['modalities'], modality_name, modality_path)

        subject['name'] = name
        relevant_dict[name] = subject

    # Create torchio.Subjects from dictionaries
    subjects = dict_to_subject_list(subjects, info['modalities'])
    train_subjects = dict_to_subject_list(train_subjects, info['modalities'])
    val_subjects = dict_to_subject_list(val_subjects, info['modalities'])
    test_subjects = dict_to_subject_list(test_subjects, info['modalities'])

    if info['shuffle']:
        np.random.seed(info['seed'])
        np.random.shuffle(subjects)
    n_subjects = len(subjects)

    # Split between train, validation and test sets
    end_train = int(round(info['repartition'][0] * n_subjects))
    end_val = end_train + int(round(info['repartition'][1] * n_subjects))
    train_subjects += subjects[:end_train]
    val_subjects += subjects[end_train:end_val]
    test_subjects += subjects[end_val:]

    return train_subjects, val_subjects, test_subjects


def generate_dataset(subjects, folder, transform_filename='transform.json', prefix='train'):
    def parse_transform(t):
        attributes = t.get('attributes') or {}
        if t.get('is_custom'):
            t_class = parse_function_import(t)
        else:
            t_class = getattr(torchio.transforms, t['name'])
        if t.get('is_selection'):
            t_dict = {}
            for p_and_t in t['transforms']:
                t_dict[parse_transform(p_and_t['transform'])] = p_and_t['proba']
            return t_class(t_dict, **attributes)
        else:
            return t_class(**attributes)

    with open(folder + transform_filename) as file:
        info = json.load(file)

    check_mandatory_keys(info, DATASET_KEYS, folder + transform_filename)

    transform_list = []
    for transform in info[f'{prefix}_transforms']:
        transform_list.append(parse_transform(transform))

    transform = torchio.transforms.Compose(transform_list)
    dataset = torchio.ImagesDataset(subjects, transform=transform)

    return dataset


def generate_dataloader(dataset, folder, loader_filename='loader.json'):
    with open(folder + loader_filename) as file:
        info = json.load(file)

    check_mandatory_keys(info, DATA_LOADER_KEYS, folder + loader_filename)
    set_dict_value(info, 'num_workers', 0)
    set_dict_value(info, 'queue')
    set_dict_value(info, 'shuffle', True)

    num_workers = info['num_workers']
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()

    if info['queue'] is None:
        loader = DataLoader(dataset, batch_size=info['batch_size'], shuffle=info['shuffle'], num_workers=num_workers)
    else:
        queue_attributes = info['queue']['attributes']
        sampler_class = getattr(torchio.data.sampler, info['queue']['sampler_class'])
        queue_attributes.update({'num_workers': num_workers, 'sampler_class': sampler_class})
        queue = torchio.Queue(dataset, **queue_attributes)
        loader = DataLoader(queue, info['batch_size'])

    return loader
