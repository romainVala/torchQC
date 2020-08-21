import commentjson as json
import numpy as np
import glob
import os
import logging
import re
import torchio
import torch
import multiprocessing
import warnings
from pathlib import Path
from inspect import signature
from copy import deepcopy
from torch.utils.data import DataLoader
import torch.nn.functional as F
from segmentation.utils import parse_object_import, parse_function_import, \
    parse_method_import, generate_json_document
from segmentation.run_model import RunModel
from segmentation.metrics.fuzzy_overlap_metrics import minimum_t_norm
from torch_summary import summary_string
from plot_dataset import PlotDataset
import pandas as pd

MAIN_KEYS = ['data', 'transform', 'model']

DATA_KEYS = ['images', 'batch_size', 'image_key_name', 'label_key_name',
             'labels']
IMAGE_KEYS = ['type', 'components']
PATTERN_KEYS = ['root', 'components']
PATH_KEYS = ['name', 'components']
LOAD_FROM_DIR_KEYS = ['root', 'list_name']
QUEUE_KEYS = ['sampler']
SAMPLER_KEYS = ['name', 'module', 'attributes']
SAMPLER_ATTRIBUTES_KEYS = ['patch_size']

TRANSFORM_KEYS = ['train_transforms', 'val_transforms']

MODEL_KEYS = ['name', 'module']

RUN_KEYS = ['criteria', 'optimizer', 'save', 'validation', 'n_epochs']
OPTIMIZER_KEYS = ['name', 'module']
SCHEDULER_KEYS = ['name', 'module']
SAVE_KEYS = ['record_frequency']


class Config:
    def __init__(self, main_file, results_dir, logger=None, debug_logger=None,
                 mode='train', viz=0, extra_file=None,
                 safe_mode=False):
        self.mode = mode
        self.logger = logger
        self.debug_logger = debug_logger
        self.viz = viz
        self.safe_mode = safe_mode

        self.results_dir = results_dir
        self.main_structure = self.parse_main_file(main_file)
        self.json_config = {}

        data_structure, transform_structure, model_structure, \
            run_structure = self.parse_extra_file(extra_file)

        data_structure, labels, patch_size, sampler = self.parse_data_file(
            data_structure)
        transform_structure = self.parse_transform_file(transform_structure)

        self.labels = labels
        self.batch_size = data_structure['batch_size']
        self.patch_size = self.parse_patch_size(patch_size)
        self.sampler = sampler
        self.image_key_name = data_structure['image_key_name']
        self.label_key_name = data_structure['label_key_name']

        self.train_set, self.val_set, self.test_set = self.load_subjects(
            data_structure, transform_structure)
        self.train_loader, self.val_loader = self.generate_data_loaders(
            data_structure)

        self.save_transformed_samples = transform_structure['save']
        self.loaded_model_name = None

        if 'model' in self.main_structure:
            self.model_structure = self.parse_model_file(model_structure)
        if 'run' in self.main_structure:
            self.run_structure = self.parse_run_file(run_structure)
        if 'visualisation' in self.main_structure:
            self.viz_structure = self.parse_visualization_file(
                self.main_structure['visualisation'])

        self.save_json(self.json_config, 'config_all.json',
                       compare_existing=False)
        self.log(
            '******** Result_dir is ******** \n  {}'.format(self.results_dir))

    @staticmethod
    def check_mandatory_keys(struct, keys, name):
        """
        Check that all keys mentioned as mandatory are present
        in a given dictionary.
        """
        for key in keys:
            if key not in struct.keys():
                raise KeyError(
                    f'Mandatory key {key} not in {struct.keys()} from {name}')

    @staticmethod
    def set_struct_value(struct, key, default_value=None):
        """
        Give a default value to a key of a dictionary is this
        key was not in the dictionary.
        """
        if struct.get(key) is None:
            struct[key] = default_value

    @staticmethod
    def read_json(file):
        if isinstance(file, str):
            with open(file) as f:
                return json.load(f)
        else:
            return file

    def compare_structs(self, struct1, struct2):
        def dict_diff(d1, d2):
            diff = {}

            # Keys present only in d1
            for key in set(d1) - set(d2):
                diff[key] = d1[key]

            # Common keys between d1 and d2
            for key in set(d1) - (set(d1) - set(d2)):
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    difference = dict_diff(d1[key], d2[key])
                    if difference is not None:
                        diff[key] = difference

                elif d1[key] != d2[key]:
                    diff[key] = d1[key]

            if len(diff) == 0:
                return None
            else:
                return diff

        old_version = dict_diff(struct1, struct2)
        new_version = dict_diff(struct2, struct1)

        has_changed = old_version is not None or new_version is not None
        if has_changed:
            self.log('Old version:')
            self.log(json.dumps(old_version, indent=4, sort_keys=True))

            self.log('New version:')
            self.log(json.dumps(new_version, indent=4, sort_keys=True))
        return has_changed

    def save_json(self, struct, name, compare_existing=True):
        self.debug(f'******** {name.upper()} ********')
        self.debug(json.dumps(struct, indent=4, sort_keys=True))
        file_path = os.path.join(self.results_dir, name)
        if os.path.exists(file_path) and compare_existing:
            self.debug('WARNING file {} exist'.format(file_path))
            old_struct = self.read_json(file_path)
            has_changed = self.compare_structs(old_struct, struct)
            if has_changed and self.safe_mode:
                proceed = input('Do you want to proceed? (Y/n)')
                print(proceed)
                if proceed.upper() in ['N', 'NO']:
                    raise KeyboardInterrupt('User did not want to proceed.')
            if not has_changed:
                self.debug('No differences found between old and new versions.')
            else:
                self.log('over-writing {}'.format(file_path))
        else:
            self.debug('writing {}'.format(file_path))
        generate_json_document(file_path, **struct)

    def log(self, info):
        if self.logger is not None:
            self.logger.log(logging.INFO, info)

    def debug(self, info):
        if self.debug_logger is not None:
            self.debug_logger.log(logging.DEBUG, info)

    def parse_main_file(self, file):
        struct = self.read_json(file)

        additional_key = []
        if self.mode in ['train', 'eval', 'infer']:
            additional_key = ['run']
        self.check_mandatory_keys(struct, MAIN_KEYS + additional_key,
                                  'MAIN CONFIG FILE')

        # Replace relative path if needed
        dir_file = os.path.dirname(file)
        for key, val in struct.items():
            if isinstance(val, str):
                if os.path.dirname(val) == '':
                    struct[key] = os.path.realpath(os.path.join(dir_file, val))

        # self.save_json(struct, 'main.json') #performe in parse_extra_file, since the result_dir can change

        return struct

    def parse_extra_file(self, file):
        data_structure = self.parse_data_file(
            self.main_structure['data'], return_string=True)
        transform_structure = self.parse_transform_file(
            self.main_structure['transform'], return_string=True)
        if 'model' in self.main_structure:
            model_structure = self.parse_model_file(
                self.main_structure['model'], return_string=True)
        else:
            model_structure = {}
        if 'run' in self.main_structure:
            run_structure = self.parse_run_file(
                self.main_structure['run'], return_string=True)
        else:
            run_structure = {}

        if file is not None:
            struct = self.read_json(file)

            if struct.get('data') is not None:
                data_structure.update(struct['data'])
            if struct.get('transform') is not None:
                transform_structure.update(struct['transform'])
            if struct.get('model') is not None:
                model_structure.update(struct['model'])
            if struct.get('run') is not None:
                run_structure.update(struct['run'])
            if struct.get('results_dir') is not None:
                results_dir = struct['results_dir']

                # Replace relative path if needed
                if Path(results_dir).parent.anchor == '':
                    results_dir = os.path.join(os.path.dirname(file),
                                               results_dir)

                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)
                self.results_dir = results_dir

            self.save_json(struct, 'extra_file.json')

        # Save main_struct with relative path and generic name for future use
        main_struct = self.main_structure.copy()
        for key, val in main_struct.items():
            main_struct[key] = '{}.json'.format(key)

        self.save_json(self.main_structure, 'main_orig.json')
        self.save_json(main_struct, 'main.json')

        return data_structure, transform_structure, model_structure, \
            run_structure

    def parse_data_file(self, file, return_string=False):
        struct = self.read_json(file)

        self.check_mandatory_keys(struct, DATA_KEYS, 'DATA CONFIG FILE')
        self.set_struct_value(struct, 'patterns', [])
        self.set_struct_value(struct, 'paths', [])
        self.set_struct_value(struct, 'load_sample_from_dir', [])
        self.set_struct_value(struct, 'csv_file', [])
        self.set_struct_value(struct, 'subject_shuffle')
        self.set_struct_value(struct, 'subject_seed')
        self.set_struct_value(struct, 'repartition', [0.7, 0.15, 0.15])
        self.set_struct_value(struct, 'raise_error', True)

        self.set_struct_value(struct, 'num_workers', 0)
        self.set_struct_value(struct, 'queue')
        self.set_struct_value(struct, 'batch_shuffle')
        self.set_struct_value(struct, 'collate_fn')
        self.set_struct_value(struct, 'batch_seed')

        total = sum(struct['repartition'])
        struct['repartition'] = list(
            map(lambda x: x / total, struct['repartition']))

        for image in struct['images'].values():
            self.check_mandatory_keys(image, IMAGE_KEYS, 'IMAGE')
            self.set_struct_value(image, 'attributes', {})

        for pattern in struct['patterns']:
            self.check_mandatory_keys(pattern, PATTERN_KEYS, 'PATTERN')
            self.set_struct_value(pattern, 'list_name')
            self.set_struct_value(pattern, 'name_pattern')

        for path in struct['paths']:
            self.check_mandatory_keys(path, PATH_KEYS, 'PATH')
            self.set_struct_value(path, 'name')
            self.set_struct_value(path, 'list_name')

        for directory in struct['load_sample_from_dir']:
            self.check_mandatory_keys(
                directory, LOAD_FROM_DIR_KEYS, 'DIRECTORY')
            self.set_struct_value(directory, 'add_to_load_regexp')
            self.set_struct_value(directory, 'add_to_load')

        patch_size, sampler = None, None
        if struct['queue'] is not None:
            self.check_mandatory_keys(struct['queue'], QUEUE_KEYS, 'QUEUE')
            self.check_mandatory_keys(
                struct['queue']['sampler'], SAMPLER_KEYS, 'SAMPLER')
            self.check_mandatory_keys(
                struct['queue']['sampler']['attributes'],
                SAMPLER_ATTRIBUTES_KEYS, 'SAMPLER ATTRIBUTES'
            )
            self.set_struct_value(struct['queue'], 'attributes', {})

            patch_size = struct['queue']['sampler']['attributes']['patch_size']

        if return_string:
            return struct
        self.json_config['data'] = deepcopy(struct)  # struct.copy()
        self.save_json(struct, 'data.json')

        # Make imports
        if struct['collate_fn'] is not None:
            struct['collate_fn'] = parse_function_import(struct['collate_fn'])

        if struct['queue'] is not None:
            if 'label_probabilities' in struct['queue']['sampler'][
                    'attributes']:
                for key, value in struct['queue']['sampler']['attributes'][
                        'label_probabilities'].items():
                    if isinstance(key, str) and key.isdigit():
                        del struct['queue']['sampler']['attributes'][
                            'label_probabilities'][key]
                        struct['queue']['sampler']['attributes'][
                            'label_probabilities'][int(key)] = value

            sampler, _ = parse_object_import(struct['queue']['sampler'])
            struct['queue']['sampler'] = sampler
            struct['queue']['attributes'].update(
                {'num_workers': struct['num_workers'], 'sampler': sampler})

        return struct, struct['labels'], patch_size, sampler

    def parse_transform_file(self, file, return_string=False):
        def parse_metric_wrapper(w):
            try:
                from torchio.metrics import MetricWrapper, MapMetricWrapper
            except Exception as e:
                self.debug(
                    'Could not import MetricWrapper from torchio.metrics . Skipping wrapped metrics.')
                return None
            wrapper_attrs = w['attributes']
            wrapper_attrs['metric_func'], _ = parse_object_import(
                wrapper_attrs['metric_func'])
            if not callable(wrapper_attrs['metric_func']):
                self.debug(
                    'Specified func in metric is not callable: {}'.format(
                        wrapper_attrs['metric_func']))
                return None
            if w['type'] == 'mapmetricwrapper':
                return MapMetricWrapper(**wrapper_attrs)
            elif w['type'] == 'metricwrapper':
                return MetricWrapper(**wrapper_attrs)
            self.debug('Found unknown wrapper type: {}'.format(w['type']))
            return None

        def parse_transform_metrics(m):
            m_dict = dict()
            for metric in m:
                for m_name, m_struct in metric.items():
                    if m_struct.get('wrapper'):
                        wrapper_attrs = m_struct['wrapper']
                        wrapper_metric = parse_metric_wrapper(wrapper_attrs)
                        m_dict[m_name] = wrapper_metric
                    else:
                        m_dict[m_name], _ = parse_object_import(m_struct)
            return m_dict

        def parse_transform(t):
            attributes = t.get('attributes') or {}
            if attributes.get('metrics'):
                t_metrics = parse_transform_metrics(attributes['metrics'])
                attributes['metrics'] = t_metrics
            if t.get('is_custom'):
                t_class = parse_function_import(t)
            else:
                t_class = getattr(torchio.transforms, t['name'])
            if t.get('is_selection'):
                if 'prob' in t['transforms'][0]:
                    t_dict = {
                        parse_transform(p_and_t['transform']): p_and_t['prob']
                        for p_and_t in t['transforms']
                    }
                else:
                    t_dict = [
                        parse_transform(p_and_t['transform']) for p_and_t in
                        t['transforms']
                    ]
                return t_class(t_dict, **attributes)
            else:
                return t_class(**attributes)

        struct = self.read_json(file)

        self.check_mandatory_keys(struct, TRANSFORM_KEYS,
                                  'TRANSFORM CONFIG FILE')
        self.set_struct_value(struct, 'save', False)

        if return_string:
            return struct
        self.json_config['transform'] = deepcopy(struct)  # struct.copy()
        self.save_json(struct, 'transform.json')

        # Make imports
        transform_list = []
        for transform in struct['train_transforms']:
            transform_list.append(parse_transform(transform))
        struct['train_transforms'] = transform_list

        transform_list = []
        for transform in struct['val_transforms']:
            transform_list.append(parse_transform(transform))
        struct['val_transforms'] = transform_list

        return struct

    def parse_model_file(self, file, return_string=False):
        struct = self.read_json(file)

        self.check_mandatory_keys(struct, MODEL_KEYS, 'MODEL CONFIG FILE')

        self.set_struct_value(struct, 'last_one', True)
        self.set_struct_value(struct, 'path')
        self.set_struct_value(struct, 'device', 'cuda')
        self.set_struct_value(struct, 'input_shape')
        self.set_struct_value(struct, 'eval_csv_basename')

        if return_string:
            return struct
        self.json_config['model'] = deepcopy(struct)  # struct.copy()
        self.save_json(struct, 'model.json')

        if struct['device'] == 'cuda' and torch.cuda.is_available():
            struct['device'] = torch.device('cuda')
        else:
            struct['device'] = torch.device('cpu')

        # Make imports
        struct['model'], struct['model_class'] = parse_object_import(struct)

        return struct

    def parse_run_file(self, file, return_string=False):
        def parse_criteria(criterion_list):
            c_list = []
            for criterion in criterion_list:
                self.set_struct_value(criterion, 'weight', 1)
                self.set_struct_value(criterion, 'mask')
                self.set_struct_value(criterion, 'channels', [])
                self.set_struct_value(criterion, 'reported_name',
                                      f'{criterion["name"]}_{criterion["method"]}')

                c = parse_method_import(criterion)
                c_list.append({
                    'criterion': c,
                    'weight': criterion['weight'],
                    'mask': criterion['mask'],
                    'channels': criterion['channels'],
                    'name': criterion['reported_name']
                })
            return c_list

        struct = self.read_json(file)

        self.check_mandatory_keys(struct, RUN_KEYS, 'RUN CONFIG FILE')
        self.set_struct_value(struct, 'data_getter', 'get_segmentation_data')
        self.set_struct_value(struct, 'seed')
        self.set_struct_value(struct, 'current_epoch')
        self.set_struct_value(struct, 'log_frequency', 10)
        self.set_struct_value(struct, 'activation', 'softmax')

        files = glob.glob(os.path.join(self.results_dir, 'model_ep*'))
        if len(files) == 0:
            struct['current_epoch'] = 1
        else:
            last_model = max(files, key=os.path.getctime)
            matches = re.findall('ep([0-9]+)', last_model)
            struct['current_epoch'] = int(matches[-1]) + 1 if matches else 1

        # Optimizer
        self.check_mandatory_keys(struct['optimizer'], OPTIMIZER_KEYS,
                                  'OPTIMIZER')
        self.set_struct_value(struct['optimizer'], 'attributes', {})
        self.set_struct_value(struct['optimizer'], 'lr_scheduler')
        if struct['optimizer']['lr_scheduler'] is not None:
            self.check_mandatory_keys(struct['optimizer']['lr_scheduler'],
                                      SCHEDULER_KEYS, 'SCHEDULER')
            self.set_struct_value(struct['optimizer']['lr_scheduler'],
                                  'attributes', {})

        # Save
        self.check_mandatory_keys(struct['save'], SAVE_KEYS, 'SAVE')
        self.set_struct_value(struct['save'], 'batch_recorder',
                              'record_segmentation_batch')
        self.set_struct_value(struct['save'], 'prediction_saver', 'save_volume')

        if isinstance(struct['data_getter'],
                      str):  # let some lazzy definition if no attribute
            struct['data_getter'] = {"name": struct['data_getter']}
        self.check_mandatory_keys(struct['data_getter'], ["name"],
                                  'data_getter')
        self.set_struct_value(struct['data_getter'], 'attributes', {})

        # Validation
        self.set_struct_value(struct['validation'], 'eval_frequency')
        self.set_struct_value(struct['validation'],
                              'whole_image_inference_frequency')
        self.set_struct_value(struct['validation'], 'patch_overlap', 8)
        self.set_struct_value(struct['validation'], 'reporting_metrics', [])

        if return_string:
            return struct
        self.json_config['run'] = deepcopy(struct)  # struct.copy()
        self.save_json(struct, 'run.json')

        # Make imports
        # Criteria
        struct['criteria'] = parse_criteria(struct['criteria'])

        # Optimizer
        struct['optimizer']['optimizer_class'] = parse_function_import(
            struct['optimizer'])
        if struct['optimizer']['lr_scheduler'] is not None:
            struct['optimizer']['lr_scheduler'][
                'class'] = parse_function_import(
                struct['optimizer']['lr_scheduler'])

        # Validation
        struct['validation']['reporting_metrics'] = parse_criteria(
            struct['validation']['reporting_metrics'])

        return struct

    def parse_visualization_file(self, file):
        struct = self.read_json(file)

        self.set_struct_value(struct, 'kwargs', {})
        sig = signature(PlotDataset)
        for key in struct['kwargs'].keys():
            if key not in sig.parameters:
                del struct['kwargs'][key]

        self.set_struct_value(struct, 'set', 'train')
        self.set_struct_value(struct, 'sample', 0)
        self.save_json(struct, 'visualization.json')

        return struct

    @staticmethod
    def parse_patch_size(patch_size):
        if isinstance(patch_size, int):
            return patch_size, patch_size, patch_size
        return patch_size

    def load_subjects(self, data_struct, transform_struct):
        def update_subject(subj, ref_images, comp_name, comp_path, img_name):
            if img_name not in ref_images:
                raise ValueError(
                    f'Try to provide component {comp_name} for image {img_name}'
                    f' but {img_name} not in reference images: {ref_images}'
                )
            if comp_name not in ref_images[img_name]['components']:
                raise ValueError(
                    f'Try to provide component {comp_name} for image {img_name}'
                    f' but {comp_name} not in listed components: '
                    f'{ref_images[img_name]["components"]}'
                )
            if img_name not in subj:
                subj[img_name] = {}
            subj[img_name][comp_name] = comp_path

        def get_relevant_dict(default_dict, train_dict, val_dict, test_dict,
                              dict_name=None):
            if dict_name == 'train':
                return train_dict
            if dict_name == 'val':
                return val_dict
            if dict_name == 'test':
                return test_dict
            return default_dict

        def dict2subjects(subject_dict, ref_images):
            subject_list = []
            for n, s in subject_dict.items():
                if not (set(s.keys())).issuperset(ref_images.keys()):
                    warnings.warn(
                        f'An image is missing for subject {n}, '
                        f'{s.keys()} were found but '
                        f'at least {ref_images.keys()} were expected.'
                        f'Dropping subject {n}')
                    continue
                for img_name in ref_images.keys():
                    image_attributes = ref_images[img_name]['attributes']
                    img = torchio.Image(
                        type=ref_images[img_name]['type'],
                        path=[s[img_name][c]
                              for c in ref_images[img_name]['components']],
                        **image_attributes
                    )
                    s[img_name] = img
                s['name'] = n
                subject_list.append(torchio.Subject(s))
            return subject_list

        def get_name(name_pattern, string):
            if name_pattern is None:
                return os.path.relpath(string, Path(string).parent)
            else:
                matches = re.findall(name_pattern, string)
                return matches[-1]

        def create_dataset(subject_list, transforms):
            if len(subject_list) == 0:
                return []
            final_transform = torchio.transforms.Compose(transforms)
            return torchio.SubjectsDataset(
                subject_list, transform=final_transform)

        train_set, val_set, test_set = [], [], []

        # Retrieve subjects using load_sample_from_dir
        if len(data_struct['load_sample_from_dir']) > 0:
            for sample_dir in data_struct['load_sample_from_dir']:
                # print('parsing sample dir addin {}'.format(sample_dir['root']))

                sample_files = glob.glob(
                    os.path.join(sample_dir['root'], 'sample*pt'))
                self.logger.log(logging.INFO,
                                f'{len(sample_files)} subjects in the '
                                f'{sample_dir["list_name"]} set')
                transform = torchio.transforms.Compose(
                    transform_struct[f'{sample_dir["list_name"]}_transforms'])
                dataset = torchio.SubjectsDataset(sample_files,
                                                load_from_dir=True,
                                                transform=transform,
                                                add_to_load=sample_dir[
                                                    'add_to_load'],
                                                add_to_load_regexp=sample_dir[
                                                    'add_to_load_regexp'])
                if sample_dir["list_name"] == 'train':
                    train_set = dataset
                elif sample_dir["list_name"] == 'val':
                    val_set = dataset
                else:
                    raise ValueError(
                        'list_name attribute from load_from_dir must be '
                        'either train or val')

            return train_set, val_set, test_set

        subjects, train_subjects, val_subjects, test_subjects = {}, {}, {}, {}

        # Retrieve subjects using csv file
        for csv_file in data_struct['csv_file']:
            relevant_dict = get_relevant_dict(subjects, train_subjects,
                                              val_subjects, test_subjects,
                                              csv_file['list_name'])
            res = pd.read_csv(csv_file["root"])

            for suj_idx in range(len(res)):
                subject = {'name': res['name'][suj_idx]}
                for component_name, component in csv_file['components'].items():
                    component_path = res[component['column_name']][suj_idx]
                    image_name = component['image']
                    update_subject(subject, data_struct['images'],
                                   component_name, component_path, image_name)

                relevant_dict[suj_idx] = subject

        # Retrieve subjects using patterns
        for pattern in data_struct['patterns']:
            relevant_dict = get_relevant_dict(
                subjects, train_subjects, val_subjects, test_subjects,
                pattern['list_name'])
            # Sort to get alphabetic order if not shuffle
            for folder_path in sorted(glob.glob(pattern['root'])):
                name = get_name(pattern['name_pattern'], folder_path)
                subject = relevant_dict.get(name) or {}

                for component_name, component in pattern['components'].items():
                    component_path = os.path.join(
                        folder_path, component['path'])
                    image_name = component['image']
                    update_subject(subject, data_struct['images'],
                                   component_name, component_path, image_name)

                relevant_dict[name] = subject

        # Retrieve subjects using paths
        for path in data_struct['paths']:
            subject_number = len(subjects) + len(train_subjects) \
                             + len(val_subjects) + len(test_subjects)
            default_name = f'{subject_number:0>6}'
            name = path['name'] or default_name
            relevant_dict = get_relevant_dict(
                subjects, train_subjects, val_subjects, test_subjects,
                path['list_name'])
            subject = relevant_dict.get(name) or {}

            for component_name, component in path['components'].items():
                component_path = component['path']
                image_name = component['image']
                update_subject(subject, data_struct['images'],
                               component_name, component_path, image_name)

            relevant_dict[name] = subject

        if self.mode == 'eval' and len(val_subjects) > 0:
            subjects, train_subjects, test_subjects = {}, {}, {}

        # Create torchio.Subjects from dictionaries
        subjects = dict2subjects(subjects, data_struct['images'])
        train_subjects = dict2subjects(train_subjects, data_struct['images'])
        val_subjects = dict2subjects(val_subjects, data_struct['images'])
        test_subjects = dict2subjects(test_subjects, data_struct['images'])

        if data_struct['subject_shuffle']:
            np.random.seed(data_struct['subject_seed'])
            np.random.shuffle(subjects)
        n_subjects = len(subjects)

        # Split between train, validation and test sets
        end_train = int(round(data_struct['repartition'][0] * n_subjects))
        end_val = end_train + int(
            round(data_struct['repartition'][1] * n_subjects))
        train_subjects += subjects[:end_train]
        val_subjects += subjects[end_train:end_val]
        test_subjects += subjects[end_val:]

        self.log(f'{len(train_subjects)} subjects in the train set')
        self.log(f'{len(val_subjects)} subjects in the validation set')
        self.log(f'{len(test_subjects)} subjects in the test set')

        train_set = create_dataset(train_subjects,
                                   transform_struct['train_transforms'])
        val_set = create_dataset(val_subjects,
                                 transform_struct['val_transforms'])
        test_set = create_dataset(test_subjects,
                                  transform_struct['val_transforms'])
        return train_set, val_set, test_set

    def generate_data_loaders(self, struct):
        train_loader, val_loader = None, None

        if struct['num_workers'] == -1:
            struct['num_workers'] = multiprocessing.cpu_count()

        if struct['batch_seed'] is not None:
            torch.manual_seed(struct['batch_seed'])

        if struct['queue'] is None:
            if len(self.train_set) > 0:
                train_loader = DataLoader(self.train_set, self.batch_size,
                                          shuffle=struct['batch_shuffle'],
                                          num_workers=struct['num_workers'],
                                          collate_fn=struct['collate_fn'])
            if len(self.val_set) > 0:
                val_loader = DataLoader(self.val_set, self.batch_size,
                                        shuffle=struct['batch_shuffle'],
                                        num_workers=struct['num_workers'],
                                        collate_fn=struct['collate_fn'])
        else:
            if len(self.train_set) > 0:
                train_queue = torchio.Queue(self.train_set,
                                            **struct['queue']['attributes'])
                train_loader = DataLoader(train_queue, self.batch_size,
                                          collate_fn=struct['collate_fn'])
            if len(self.val_set) > 0:
                val_queue = torchio.Queue(self.val_set,
                                          **struct['queue']['attributes'])
                val_loader = DataLoader(val_queue, self.batch_size,
                                        collate_fn=struct['collate_fn'])
        return train_loader, val_loader

    def load_model(self, struct):
        def return_model(m, f=None):
            if f is not None:
                self.log(f'Using model from file {f}')
            else:
                self.log('Using new model')
            device = struct['device']
            self.debug('Model description:')
            self.debug(model)

            m.to(device)

            input_shape = self.patch_size or struct['input_shape']
            if input_shape is not None:
                summary, _ = summary_string(model, (1, *input_shape),
                                            self.batch_size, device)
                self.log('Model summary:')
                self.log(summary)
            return m, device

        self.log('******** Model  ********')

        model = struct['model']
        model_class = struct['model_class']

        if struct['last_one']:
            files = glob.glob(os.path.join(self.results_dir, 'model_ep*'))
            if len(files) == 0:
                return return_model(model)
            file = max(files, key=os.path.getctime)

        else:
            file = struct['path']
            if file is None or not Path(file).exists():
                raise ValueError(
                    f'Impossible to load model from {file}, '
                    f'this file does not exist')

        if hasattr(model_class, 'load'):
            model, _ = model_class.load(file)
        else:
            model.load_state_dict(torch.load(file))

        self.loaded_model_name = os.path.basename(file)[
                                 :-8]  # to remove .pth.tar

        return return_model(model, file)

    def run(self):
        if self.mode in ['train', 'eval', 'infer']:
            model, device = self.load_model(self.model_structure)

            model_runner = RunModel(model, device, self.train_loader,
                                    self.val_loader, self.val_set,
                                    self.test_set, self.image_key_name,
                                    self.label_key_name, self.labels,
                                    self.logger, self.debug_logger,
                                    self.results_dir, self.batch_size,
                                    self.patch_size, self.run_structure)

            if self.mode == 'train':
                model_runner.train()
            elif self.mode == 'eval':
                model_runner.eval(
                    model_name=self.loaded_model_name,
                    eval_csv_basename=self.model_structure['eval_csv_basename'],
                    save_transformed_samples=self.save_transformed_samples)
            else:
                model_runner.infer()

        # Other case would typically be visualization for example
        if self.mode == 'visualization':
            self.viz_structure['kwargs'].update(
                {'image_key_name': self.image_key_name[0]})

            if self.viz_structure['set'] == 'train':
                viz_set = self.train_set
            else:
                viz_set = self.val_set

            if 0 <= self.viz < 4:
                if self.viz == 1:
                    self.viz_structure['kwargs'].update({
                        'label_key_name': self.label_key_name
                    })

                elif self.viz == 2:
                    self.viz_structure['kwargs'].update({
                        'patch_sampler': self.sampler
                    })

                elif self.viz == 3:
                    self.viz_structure['kwargs'].update({
                        'label_key_name': self.label_key_name,
                        'patch_sampler': self.sampler
                    })

            # Move this to run_model.py and update it to handle C channels
            if self.viz >= 4:
                model_structure = self.parse_model_file(self.model_structure)
                run_structure = self.parse_run_file(self.main_structure['run'])
                model, device = self.load_model(model_structure)

                sample = viz_set[self.viz_structure['sample']]
                viz_set = [sample]

                model_runner = RunModel(model, [], None, [], viz_set,
                                        self.image_key_name,
                                        self.label_key_name, None, None,
                                        self.results_dir, self.batch_size,
                                        self.patch_size, run_structure)

                with torch.no_grad():
                    model_runner.model.eval()
                    volume, target = model_runner.data_getter(sample)
                    if self.patch_size is not None:
                        prediction = model_runner.make_prediction_on_whole_volume(sample)
                    else:
                        prediction = model_runner.model(volume.unsqueeze(0))[0]

                prediction = F.softmax(prediction, dim=0).to('cpu')

                self.viz_structure['kwargs'].update({
                    'label_key_name': self.label_key_name
                })

                # TODO: Define which key is (/ keys are) used to create FP maps using config files
                if self.viz == 4:
                    false_positives = minimum_t_norm(prediction[0], target,
                                                     True)
                    sample[self.label_key_name]['data'] = false_positives
                    viz_set = [sample]

                elif self.viz == 5:
                    ground_truth = deepcopy(sample)
                    sample[self.label_key_name]['data'] = prediction[
                        0].unsqueeze(0)
                    viz_set = [ground_truth, sample]

            return PlotDataset(viz_set, **self.viz_structure['kwargs'])
