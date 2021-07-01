import commentjson as json
import numpy as np
import glob
import os
import sys
import logging
import re
import torchio
import torch
import multiprocessing
import warnings
from pathlib import Path
from inspect import signature
from copy import deepcopy
from itertools import product
from torch.utils.data import DataLoader
from segmentation.utils import parse_object_import, parse_function_import, parse_class_and_method_import, \
    parse_method_import, generate_json_document, custom_import, CustomDataset
from segmentation.run_model import RunModel
from segmentation.metrics.utils import MetricOverlay
from segmentation.metrics.fuzzy_overlap_metrics import minimum_t_norm
from torch_summary import summary_string
from plot_dataset import PlotDataset
import pandas as pd

MAIN_KEYS = ['data', 'transform', 'model']

DATA_KEYS = ['images', 'batch_size', 'image_key_name' ]\
            #, 'label_key_name','labels']
IMAGE_KEYS = ['type', 'components']
PATTERN_KEYS = ['root', 'components']
CSV_KEYS = ['root', 'components', 'name']
PATH_KEYS = ['name', 'components']
LOAD_FROM_DIR_KEYS = ['root', 'list_name']
QUEUE_KEYS = ['sampler']
SAMPLER_KEYS = ['name', 'module', 'attributes']
SAMPLER_ATTRIBUTES_KEYS = ['patch_size']

TRANSFORM_KEYS = ['train_transforms', 'val_transforms']

MODEL_KEYS = ['name', 'module']

RUN_KEYS = ['criteria', 'optimizer', 'save', 'validation', 'n_epochs']
ACTIVATION_KEYS = ['name', 'module']
OPTIMIZER_KEYS = ['name', 'module']
SCHEDULER_KEYS = ['name', 'module']
SAVE_KEYS = ['record_frequency']


class Config:
    def __init__(self, main_file, results_dir, logger=None, debug_logger=None,
                 mode='train', viz=0, extra_file=None, safe_mode=False,
                 create_jobs_file=None, gs_keys=None, gs_values=None,
                 max_subjects_per_job=None, save_files=True):
        self.main_file = main_file
        self.results_dir = results_dir
        self.logger = logger
        self.debug_logger = debug_logger
        self.mode = mode
        self.viz = viz
        self.extra_file = extra_file
        self.safe_mode = safe_mode
        self.create_jobs_file = create_jobs_file
        self.gs_keys = gs_keys
        self.gs_values = gs_values
        self.max_subjects_per_job = max_subjects_per_job
        self.save_files = save_files

        self.main_structure = None
        self.json_config = {}

        self.labels = None
        self.batch_size = None
        self.patch_size = None
        self.sampler = None
        self.image_key_name = None
        self.label_key_name = None
        self.train_set, self.val_set, self.test_set = None, None, None
        self.train_subjects = None
        self.val_subjects = None
        self.test_subjects = None
        self.train_loader, self.val_loader = None, None
        self.save_transformed_samples = None
        self.post_transforms = None
        self.model_structure = None
        self.run_structure = None
        self.viz_structure = None
    """
    suject_seed used only if subject_shufflie is not null
    batch_seed used if not None, (torch manual seed) used in init
    batch_shuffle, only used for torch dataloader, when using the queue
    seed keywork (defin in train.json) is the one set before training (or eval ... test ?)
    """
    def init(self):
        if os.path.dirname(self.main_file) == self.results_dir:
            self.save_files = False
            self.debug('Forcing no json save because input json  {} are in the result_dir {} '.format(self.main_file,self.results_dir))

        self.main_structure = self.parse_main_file(self.main_file)

        data_structure, transform_structure, model_structure, \
            run_structure = self.parse_extra_file(self.extra_file)

        data_structure, transform_structure, model_structure, \
            run_structure = self.parse_gs_items(
                self.gs_keys, self.gs_values, data_structure,
                transform_structure, model_structure, run_structure)

        data_structure, labels, patch_size, sampler = self.parse_data_file(
            data_structure)
        transform_structure = self.parse_transform_file(transform_structure)

        self.labels = labels
        self.batch_size = data_structure['batch_size']
        self.patch_size = self.parse_patch_size(patch_size)
        self.sampler = sampler
        self.image_key_name = data_structure['image_key_name']
        self.label_key_name = data_structure['label_key_name']

        self.train_set, self.val_set, self.test_set, self.train_subjects, \
            self.val_subjects, self.test_subjects = self.load_subjects(
                data_structure, transform_structure)
        self.train_loader, self.val_loader = self.generate_data_loaders(
            data_structure)
        self.save_transformed_samples = transform_structure['save']
        self.post_transforms = transform_structure['post_transforms']

        if 'model' in self.main_structure:
            self.model_structure = self.parse_model_file(model_structure)
        if 'run' in self.main_structure:
            self.run_structure = self.parse_run_file(run_structure)
        if 'visualization' in self.main_structure:
            self.viz_structure = self.parse_visualization_file(
                self.main_structure['visualization'])

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
        if key not in struct:
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
        if not self.save_files:
            return
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
        else:
            print(info)

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

            self.save_json(struct, 'extra_file.json')

        # Save main_struct with relative path and generic name for future use
        main_struct = self.main_structure.copy()
        for key, val in main_struct.items():
            main_struct[key] = '{}.json'.format(key)

        self.save_json(self.main_structure, 'main_orig.json')
        self.save_json(main_struct, 'main.json')

        return data_structure, transform_structure, model_structure, \
            run_structure

    @staticmethod
    def parse_gs_items(gs_keys, gs_values, data_structure,
                       transform_structure, model_structure, run_structure):
        def _set_value(dictionary, key_list, val):
            new_key = key_list.pop(0)
            if new_key.isdigit():
                new_key = int(new_key)
            element = dictionary[new_key]
            if len(key_list) == 0:
                dictionary[new_key] = val
            else:
                _set_value(element, key_list, val)

        if gs_keys is None:
            return data_structure, transform_structure, model_structure, \
                run_structure
        for key, value in zip(gs_keys, gs_values):
            key_fragments = key.split('.')
            struct_key = key_fragments.pop(0)
            if struct_key == 'data':
                _set_value(data_structure, key_fragments, value)
            if struct_key == 'transform':
                _set_value(transform_structure, key_fragments, value)
            if struct_key == 'model':
                _set_value(model_structure, key_fragments, value)
            if struct_key == 'run':
                _set_value(run_structure, key_fragments, value)
        return data_structure, transform_structure, model_structure, \
            run_structure

    def parse_data_file(self, file, return_string=False):
        struct = self.read_json(file)

        self.check_mandatory_keys(struct, DATA_KEYS, 'DATA CONFIG FILE')
        self.set_struct_value(struct, 'label_key_name')
        self.set_struct_value(struct, 'labels')
        self.set_struct_value(struct, 'patterns', [])
        self.set_struct_value(struct, 'csv', [])
        self.set_struct_value(struct, 'paths', [])
        self.set_struct_value(struct, 'load_sample_from_dir', [])
        self.set_struct_value(struct, 'csv_file', [])
        self.set_struct_value(struct, 'subject_shuffle')
        self.set_struct_value(struct, 'subject_seed')
        self.set_struct_value(struct, 'repartition', [0.7, 0.15, 0.15])

        self.set_struct_value(struct, 'num_workers', 0)
        self.set_struct_value(struct, 'queue')
        self.set_struct_value(struct, 'batch_shuffle')
        self.set_struct_value(struct, 'collate_fn')
        self.set_struct_value(struct, 'batch_seed')
        self.set_struct_value(struct, 'epoch_length')

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
            self.set_struct_value(pattern, 'prefix', '')
            self.set_struct_value(pattern, 'suffix', '')

        for csv in struct['csv_file']:
            self.check_mandatory_keys(csv, CSV_KEYS, 'CSV_FILE')
            self.set_struct_value(csv, 'name')
            self.set_struct_value(csv, 'list_name')

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
                label_probabilities = {}
                for key, value in struct['queue']['sampler']['attributes'][
                        'label_probabilities'].items():
                    try:
                        label_probabilities[int(key)] = value
                    except ValueError:
                        raise ValueError(f'Label must be integers, not {key}')
                struct['queue']['sampler']['attributes'][
                    'label_probabilities'] = label_probabilities

            sampler, _ = parse_object_import(struct['queue']['sampler'])
            struct['queue']['sampler'] = sampler
            struct['queue']['attributes'].update(
                {'num_workers': struct['num_workers'], 'sampler': sampler})
        return struct, struct['labels'], patch_size, sampler

    def parse_transform_file(self, file, return_string=False):
        def parse_metric_wrapper(w):
            try:
                from torchio.metrics import MapMetricWrapper
            except (ModuleNotFoundError, ImportError):
                self.debug(
                    'Could not import MetricWrapper from torchio.metrics . '
                    'Skipping wrapped metrics.')
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
        self.set_struct_value(struct, 'post_transforms', [])

        if return_string:
            return struct
        self.json_config['transform'] = deepcopy(struct)  # struct.copy()
        self.save_json(struct, 'transform.json')

        # Make imports
        transform_list = []
        for transform in struct['train_transforms']:
            transform_list.append(parse_transform(transform))
        self.train_transfo_list = transform_list
        struct['train_transforms'] = torchio.transforms.Compose(transform_list)

        transform_list = []
        for transform in struct['val_transforms']:
            transform_list.append(parse_transform(transform))
        struct['val_transforms'] = torchio.transforms.Compose(transform_list)
        self.val_transfo_list = transform_list
        transform_list = []
        for transform in struct['post_transforms']:
            transform_list.append(parse_transform(transform))
        struct['post_transforms'] = torchio.transforms.Compose(transform_list)

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
        def parse_criteria(criterion_list, activation):
            c_list = []
            for criterion in criterion_list:
                self.set_struct_value(criterion, 'channels', None)
                self.set_struct_value(criterion, 'mask')
                self.set_struct_value(criterion, 'mask_cut', [0.99, 1])
                self.set_struct_value(criterion, 'binarize_target', False)
                self.set_struct_value(criterion, 'binarize_prediction', False)
                self.set_struct_value(criterion, 'binary_volumes', False)
                self.set_struct_value(criterion, 'activation', activation)
                self.set_struct_value(criterion, 'weight', 1)
                self.set_struct_value(criterion, 'band_width')
                self.set_struct_value(criterion, 'use_far_mask', False)
                self.set_struct_value(criterion, 'mixt_activation', 0)
                self.set_struct_value(
                    criterion, 'reported_name',
                    f'{criterion["name"]}_{criterion["method"]}')

                if criterion['channels'] is not None:
                    criterion['channels'] = [
                        self.labels.index(channel)
                        for channel in criterion['channels']
                    ]

                if criterion['mask'] is not None:
                    criterion['mask'] = self.labels.index(criterion['mask'])

                a = parse_function_import(criterion['activation'])

                class_instance, c = parse_class_and_method_import(criterion)
                additional_learned_param = class_instance.log_vars if hasattr(class_instance, 'log_vars') else None

                c_list.append({
                    'criterion': MetricOverlay(
                        metric=c,
                        channels=criterion['channels'],
                        mask=criterion['mask'],
                        mask_cut=criterion['mask_cut'],
                        binarize_target=criterion['binarize_target'],
                        activation=a,
                        binary_volumes=criterion['binary_volumes'],
                        binarize_prediction=criterion['binarize_prediction'],
                        band_width=criterion['band_width'],
                        use_far_mask=criterion['use_far_mask'],
                        mixt_activation=criterion['mixt_activation'],
                        additional_learned_param=additional_learned_param
                    ),
                    'weight': criterion['weight'],
                    'name': criterion['reported_name']
                })
            return c_list

        struct = self.read_json(file)

        self.check_mandatory_keys(struct, RUN_KEYS, 'RUN CONFIG FILE')
        self.set_struct_value(struct, 'data_getter', 'get_segmentation_data')
        self.set_struct_value(struct, 'seed')
        self.set_struct_value(struct, 'current_epoch')
        self.set_struct_value(struct, 'log_frequency', 10)
        self.set_struct_value(
            struct,
            'activation',
            {
                'name': 'softmax',
                'module': 'torch.nn.functional',
                'attributes': {'dim': 1}
            }
        )
        self.set_struct_value(struct, 'apex', {})

        files = glob.glob(os.path.join(self.results_dir, 'model_ep*'))
        if len(files) == 0:
            struct['current_epoch'] = 1
        else:
            last_model = max(files, key=os.path.getmtime)
            matches = re.findall('ep([0-9]+)', last_model)
            struct['current_epoch'] = int(matches[-1]) + 1 if matches else 1

        # Activation
        self.check_mandatory_keys(struct['activation'], ACTIVATION_KEYS,
                                  'ACTIVATION')

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
        self.set_struct_value(struct['save'], 'label_saver', 'save_volume')
        self.set_struct_value(struct['save'], 'save_bin', False)
        self.set_struct_value(struct['save'], 'split_channels', False)
        self.set_struct_value(struct['save'], 'save_channels')
        self.set_struct_value(struct['save'], 'save_threshold', 0)
        self.set_struct_value(struct['save'], 'save_volume_name', 'prediction')

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
        self.set_struct_value(struct['validation'], 'save_predictions', False)
        self.set_struct_value(
            struct['validation'], 'prefix_eval_results_dir', None)
        self.set_struct_value(struct['validation'], 'dense_patch_eval', False)
        self.set_struct_value(struct['validation'], 'eval_patch_size')
        self.set_struct_value(struct['validation'], 'save_labels', False)
        self.set_struct_value(struct['validation'], 'save_data', False)
        self.set_struct_value(struct['validation'], 'eval_dropout', 0)
        self.set_struct_value(struct['validation'], 'split_batch_gpu', False)
        self.set_struct_value(struct['validation'], 'repeate_eval', 1)

        if struct['validation']['prefix_eval_results_dir'] is None:
            struct['validation']['eval_results_dir'] = self.results_dir
        else:
            struct['validation']['eval_results_dir'] = os.path.join(
                struct['validation']['prefix_eval_results_dir'],
                Path(self.results_dir).name
            )

        # Apex
        self.set_struct_value(struct, 'apex_opt_level')

        if return_string:
            return struct
        self.json_config['run'] = deepcopy(struct)  # struct.copy()
        self.save_json(struct, 'run.json')

        # Make imports
        # Criteria
        struct['criteria'] = parse_criteria(struct['criteria'],
                                            struct['activation'])

        # Validation
        struct['validation']['reporting_metrics'] = parse_criteria(
            struct['validation']['reporting_metrics'], struct['activation'])

        # Activation
        struct['activation'] = parse_function_import(struct['activation'])

        # Optimizer
        struct['optimizer']['optimizer_class'] = custom_import(
            struct['optimizer'])
        if struct['optimizer']['lr_scheduler'] is not None:
            struct['optimizer']['lr_scheduler'][
                'class'] = custom_import(struct['optimizer']['lr_scheduler'])

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
                    components = ref_images[img_name]['components']
                    image_attributes = ref_images[img_name]['attributes']
                    image_attributes.update(
                        {'components': [c for c in components]})
                    #force nibable reader
                    image_attributes.update({'reader': torchio.io._read_nibabel})
                    if ref_images[img_name]['type'] == 'label':
                        img = torchio.LabelMap( path=[s[img_name][c] for c in components], **image_attributes)
                    elif ref_images[img_name]['type'] == 'label4D':
                        image_attributes.update({'channels_last':True})
                        img = torchio.LabelMap(path=[s[img_name][c] for c in components], **image_attributes)
                    elif  ref_images[img_name]['type'] == 'intensity':
                        img = torchio.ScalarImage( path=[s[img_name][c] for c in components], **image_attributes)
                    else:
                        raise (f'error image type {ref_images[img_name]["type"]} is not know either intensity or label')

                    s[img_name] = img
                if 'name' not in s:
                    s['name'] = n
                subject_list.append(torchio.Subject(s))
            return subject_list

        def get_name(name_pattern, string, prefix, suffix):
            if name_pattern is None:
                core = os.path.relpath(string, Path(string).parent)
            else:
                core = '_'.join(re.findall(name_pattern, string))
            return prefix + core + suffix

        def create_dataset(subject_list, transforms, epoch_length=None):
            if len(subject_list) == 0:
                return []
            if epoch_length is not None:
                return CustomDataset(subject_list, transforms, epoch_length)
            return torchio.SubjectsDataset(
                subject_list, transform=transforms)

        train_set, val_set, test_set = [], [], []

        # Retrieve subjects using load_sample_from_dir
        if len(data_struct['load_sample_from_dir']) > 0:
            for sample_dir in data_struct['load_sample_from_dir']:
                # print('parsing sample dir addin {}'.format(
                #   sample_dir['root']))

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
                subject = {'name': f'S{suj_idx:03d}_' + str(res[csv_file['name']][suj_idx])}
                for component_name, component in csv_file['components'].items():
                    component_path = res[component['column_name']][suj_idx]
                    image_name = component['image']
                    dds =  data_struct['images']
                    if "attributes" in component:
                        ccname = component['attributes']['column_name']
                        if "attributes" in dds[image_name]:
                            dds[image_name]['attributes'].update({ccname : res[ccname][suj_idx]})
                        else:
                            dds[image_name]['attributes'] = {ccname: res[ccname][suj_idx]}

                    update_subject(subject, dds,
                                   component_name, component_path, image_name)
                relevant_dict[suj_idx] = subject

        # Retrieve subjects using patterns
        for pattern in data_struct['patterns']:
            relevant_dict = get_relevant_dict(
                subjects, train_subjects, val_subjects, test_subjects,
                pattern['list_name'])
            # Sort to get alphabetic order if not shuffle
            for folder_path in sorted(glob.glob(pattern['root'])):
                name = get_name(
                    pattern['name_pattern'],
                    folder_path,
                    pattern['prefix'],
                    pattern['suffix']
                )
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
            seed_to_set = data_struct['subject_seed']
            self.log(f'SETING numpy seed (for subject shuffle) to {seed_to_set}')
            np.random.seed(seed_to_set)
            np.random.shuffle(subjects)
            np.random.shuffle(train_subjects)
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
        if len(train_subjects) > 2:
            self.log('first 3 Train suj : {} {}  {} '.format(train_subjects[0]['name'], train_subjects[1]['name'], train_subjects[2]['name']))
        elif len(val_subjects) > 2:
            self.log('first 3 Val suj : {} {}  {} '.format(val_subjects[0]['name'], val_subjects[1]['name'], val_subjects[2]['name']))
        elif len(test_subjects) > 2:
            self.log('first 3 test suj : {} {}  {} '.format(test_subjects[0]['name'], test_subjects[1]['name'], test_subjects[2]['name']))

        train_set = create_dataset(train_subjects,
                                   transform_struct['train_transforms'],
                                   data_struct['epoch_length'])
        val_set = create_dataset(val_subjects,
                                 transform_struct['val_transforms'])
        test_set = create_dataset(test_subjects,
                                  transform_struct['val_transforms'])
        return train_set, val_set, test_set, \
            train_subjects, val_subjects, test_subjects

    def generate_data_loaders(self, struct):
        train_loader, val_loader = None, None

        if struct['num_workers'] == -1:
            struct['num_workers'] = multiprocessing.cpu_count()

        if struct['batch_seed'] is not None:
            seed_to_set = struct['batch_seed']
            self.log(f'SETING torch seed (from batch seed) to {seed_to_set}')
            torch.manual_seed(seed_to_set)

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
            if True:
                self.log('no PARA_QUEUE ')
                self.log(struct['queue']['attributes'])
                train_queue = torchio.Queue(self.train_set,
                                            **struct['queue']['attributes'])
                train_loader = DataLoader(train_queue, self.batch_size, collate_fn=struct['collate_fn'])

            else:
                if len(self.train_set) > 0:
                    self.log('PARA_QUEUE 16 16 64 numworker 1')
                    train_queue = torchio.ParallelQueue(self.train_set, struct['queue']['sampler'], num_patches=16,
                                                        patch_queue_size=64, max_no_subjects_in_mem=16)
                    train_loader = DataLoader(train_queue, self.batch_size, num_workers=1,
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
            file = max(files, key=os.path.getmtime)
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

        return return_model(model, file)

    def create_cmd(self, main_file='main.json'):
        torchQC_path = next(filter(lambda x: 'torchQC' == x[-7:], sys.path))
        cmd = os.path.join(
            torchQC_path, 'segmentation/segmentation_pipeline.py')
        params = ' '.join([
            '-f', os.path.join(self.results_dir, main_file),
            '-r', self.results_dir,
            '-m', self.mode,
            '-viz', str(self.viz)
        ])
        full_cmd = ' '.join(['python', cmd, params])
        return full_cmd

    def split_subjects(self):
        def create_path(subject, list_name):
            path_object = {
                'name': subject['name'],
                'list_name': list_name,
                'components': {}
            }
            for image_name, image in subject.get_images_dict(False).items():
                for component, path in zip(image['components'], image.path):
                    path_object['components'][component] = {
                        'path': str(path), 'image': image_name
                    }
            return path_object

        # Get data.json and main.json structures
        data_ref = deepcopy(self.json_config['data'])
        main_ref = self.read_json(os.path.join(self.results_dir, 'main.json'))

        # Remove former paths
        if 'patterns' in data_ref:
            del data_ref['patterns']
        if 'paths' in data_ref:
            del data_ref['paths']
        if 'csv_file' in data_ref:
            del data_ref['csv_file']

        # Construct paths
        paths = []
        for subj in self.val_subjects:
            paths.append(create_path(subj, 'val'))
        for subj in self.test_subjects:
            paths.append(create_path(subj, 'test'))

        # Create data and main structures
        main_files = []
        n = self.max_subjects_per_job
        chunks = [paths[i:i + n] for i in range(0, len(paths), n)]
        for count, chunk in enumerate(chunks):
            data_structure = deepcopy(data_ref)
            data_structure['paths'] = chunk
            data_name = f'split_data_{count}.json'
            main_structure = deepcopy(main_ref)
            main_structure['data'] = data_name
            main_name = f'split_main_{count}.json'
            self.save_json(data_structure, data_name)
            self.save_json(main_structure, main_name)
            main_files.append(main_name)
        return main_files

    def get_runner(self):
        model, device = self.load_model(self.model_structure)

        model_runner = RunModel(model, device, self.train_loader,
                                self.val_loader, self.val_set,
                                self.test_set, self.image_key_name,
                                self.label_key_name, self.labels,
                                self.logger, self.debug_logger,
                                self.results_dir, self.batch_size,
                                self.patch_size, self.run_structure,
                                self.post_transforms)
        return model_runner

    def run(self):
        if self.create_jobs_file is not None and self.max_subjects_per_job is None:
            return [self.create_cmd()]
        elif self.create_jobs_file is not None:
            return [self.create_cmd(f) for f in self.split_subjects()]
        if self.mode in ['train', 'eval', 'infer']:
            model, device = self.load_model(self.model_structure)

            model_runner = RunModel(model, device, self.train_loader,
                                    self.val_loader, self.val_set,
                                    self.test_set, self.image_key_name,
                                    self.label_key_name, self.labels,
                                    self.logger, self.debug_logger,
                                    self.results_dir, self.batch_size,
                                    self.patch_size, self.run_structure,
                                    self.post_transforms)

            if self.mode == 'train':
                model_runner.train()
            elif self.mode == 'eval':
                model_runner.eval(
                    eval_csv_basename=self.model_structure['eval_csv_basename'],
                    save_transformed_samples=self.save_transformed_samples)
            else:
                model_runner.infer()

        # Other case would typically be visualization for example
        if self.mode == 'visualization':
            self.viz_structure['kwargs'].update(
                {'image_key_name': self.image_key_name})

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

                model_runner = RunModel(model, device, [], None, [], viz_set,
                                        self.image_key_name,
                                        self.label_key_name, self.labels, None,
                                        None, self.results_dir, self.batch_size,
                                        self.patch_size, run_structure,
                                        self.post_transforms)

                with torch.no_grad():
                    model_runner.model.eval()
                    volume, target = model_runner.data_getter(sample)
                    if self.patch_size is not None:
                        prediction = model_runner\
                            .make_prediction_on_whole_volume(sample)
                    else:
                        prediction = model_runner.model(volume.unsqueeze(0))[0]

                prediction = run_structure['activation'](
                    prediction.unsqueeze(0))[0].to('cpu')

                self.viz_structure['kwargs'].update({
                    'label_key_name': self.label_key_name
                })

                # TODO: Define which key is (/ keys are) used to
                #  create FP maps using config files
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


def parse_grid_search_file(file):
    struct = Config.read_json(file)
    for key, value in struct.items():
        Config.check_mandatory_keys(value, ['values', 'prefix'],
                                    f'Grid search file {key}'.upper())
        Config.set_struct_value(value, 'names', value['values'])
        assert len(value['names']) == len(value['values'])

    # Flatten cartesian product
    product_struct = {
        'keys': struct.keys(),
        'prefixes': [struct[key]['prefix'] for key in struct],
        'values': product(*[struct[key]['values'] for key in struct.keys()]),
        'names': product(*[struct[key]['names'] for key in struct.keys()]),
    }

    results_dirs = []
    for names in product_struct['names']:
        results_dir = []
        for i, name in enumerate(names):
            prefix = product_struct['prefixes'][i]
            results_dir.append(f'{prefix}_{name}')
        results_dirs.append('_'.join(results_dir))
    product_struct['results_dirs'] = results_dirs

    return product_struct


def parse_create_jobs_file(file):
    struct = Config.read_json(file)
    Config.check_mandatory_keys(
        struct, ['job_name', 'output_directory'], 'CREATE JOBS FILE')
    Config.set_struct_value(struct, 'cluster_queue', 'bigmem,normal')
    Config.set_struct_value(struct, 'cpus_per_task', 1)
    Config.set_struct_value(struct, 'mem', 4000)
    Config.set_struct_value(struct, 'walltime', '12:00:00')
    Config.set_struct_value(struct, 'job_pack', 1)
    folder = struct['output_directory']
    if Path(folder).parent.anchor == '':
        folder = os.path.join(os.path.dirname(file), folder)
        struct['output_directory'] = folder

    return struct
