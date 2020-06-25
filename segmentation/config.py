import commentjson as json
import numpy as np
import glob
import os
import logging
import re
import torchio
import torch
import multiprocessing
from pathlib import Path
from inspect import signature
from copy import deepcopy
from torch.utils.data import DataLoader
import torch.nn.functional as F
from segmentation.utils import parse_object_import, parse_function_import, parse_method_import, generate_json_document
from segmentation.run_model import RunModel
from segmentation.metrics.fuzzy_overlap_metrics import minimum_t_norm
from torch_summary import summary_string
from plot_dataset import PlotDataset
import pandas as pd

MAIN_KEYS = ['data', 'transform', 'model']

DATA_KEYS = ['modalities', 'batch_size', 'image_key_name', 'label_key_name']
MODALITY_KEYS = ['type']
PATTERN_KEYS = ['root', 'modalities']
PATH_KEYS = ['name', 'modalities']
LOAD_FROM_DIR_KEYS = ['root', 'list_name']
QUEUE_KEYS = ['sampler']
SAMPLER_KEYS = ['name', 'module', 'attributes']
SAMPLER_ATTRIBUTES_KEYS = ['patch_size']

TRANSFORM_KEYS = ['train_transforms', 'val_transforms']

MODEL_KEYS = ['name', 'module']

RUN_KEYS = ['criteria', 'optimizer', 'save', 'validation', 'n_epochs']
OPTIMIZER_KEYS = ['name', 'module']
SAVE_KEYS = ['record_frequency']


class Config:
    def __init__(self, main_file, results_dir, logger=None, debug_logger=None, mode='train', viz=0, extra_file=None):
        self.mode = mode
        self.logger = logger
        self.debug_logger = debug_logger
        self.viz = viz

        self.results_dir = results_dir
        self.main_structure = self.parse_main_file(main_file)
        self.json_config = {}

        data_structure, transform_structure, model_structure = self.parse_extra_file(extra_file)

        data_structure, patch_size, sampler = self.parse_data_file(data_structure)
        transform_structure = self.parse_transform_file(transform_structure)

        self.batch_size = data_structure['batch_size']
        self.patch_size = self.parse_patch_size(patch_size)
        self.sampler = sampler
        self.image_key_name = data_structure['image_key_name']
        self.label_key_name = data_structure['label_key_name']

        self.train_set, self.val_set, self.test_set = self.load_subjects(data_structure, transform_structure)
        self.train_loader, self.val_loader = self.generate_data_loaders(data_structure)

        self.loaded_model_name = None

        if 'model' in self.main_structure:
            self.model_structure = self.parse_model_file(model_structure)
        if 'run' in self.main_structure:
            self.run_structure = self.parse_run_file(self.main_structure['run'])

        self.save_json(self.json_config, 'config_all.json')

    @staticmethod
    def check_mandatory_keys(struct, keys, name):
        """
        Check that all keys mentioned as mandatory are present in a given dictionary.
        """
        for key in keys:
            if key not in struct.keys():
                raise KeyError(f'Mandatory key {key} not in {struct.keys()} from {name}')

    @staticmethod
    def set_struct_value(struct, key, default_value=None):
        """
        Give a default value to a key of a dictionary is this key was not in the dictionary.
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

    def save_json(self, struct, name):
        self.debug(f'******** {name.upper()} ********')
        self.debug(json.dumps(struct, indent=4, sort_keys=True))
        file_path = os.path.join(self.results_dir, name)
        if os.path.exists(file_path):
            self.log('WARNING file {} exist'.format(file_path))
        else:
            self.log('writing {}'.format(file_path))
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
        self.check_mandatory_keys(struct, MAIN_KEYS + additional_key, 'MAIN CONFIG FILE')

        # Replace relative path if needed
        dir_file = os.path.dirname(file)
        for key, val in struct.items():
            if isinstance(val, str):
                if os.path.dirname(val) == '':
                    struct[key] = os.path.realpath(os.path.join(dir_file, val))

        #self.save_json(struct, 'main.json') #performe in parse_extra_file, since the result_dir can change

        return struct

    def parse_extra_file(self, file):
        data_structure = self.parse_data_file(self.main_structure['data'], return_string=True)
        transform_structure = self.parse_transform_file(self.main_structure['transform'], return_string=True)
        model_structure = self.parse_model_file(self.main_structure['model'], return_string=True)
        results_dir = self.results_dir

        if file is not None:
            struct = self.read_json(file)

            if struct.get('data') is not None:
                data_structure.update(struct['data'])
            if struct.get('transform') is not None:
                transform_structure.update(struct['transform'])
            if struct.get('model') is not None:
                model_structure.update(struct['model'])
            if struct.get('results_dir') is not None:
                results_dir = struct['results_dir']

                # Replace relative path if needed
                if Path(results_dir).parent.anchor == '':
                    results_dir = os.path.join(os.path.dirname(file), results_dir)

                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)
                self.results_dir = results_dir

            self.save_json(struct, 'extra_file.json')

        #save main_struct with relative path and generic name future use
        main_struct = self.main_structure.copy()
        for key, val in main_struct.items():
            main_struct[key] = '{}.json'.format(key)

        self.save_json(self.main_structure, 'main_orig.json')
        self.save_json(main_struct, 'main.json')

        return data_structure, transform_structure, model_structure

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

        self.set_struct_value(struct, 'num_workers', 0)
        self.set_struct_value(struct, 'queue')
        self.set_struct_value(struct, 'batch_shuffle')
        self.set_struct_value(struct, 'collate_fn')
        self.set_struct_value(struct, 'batch_seed')

        total = sum(struct['repartition'])
        struct['repartition'] = list(map(lambda x: x / total, struct['repartition']))

        for modality in struct['modalities'].values():
            self.check_mandatory_keys(modality, MODALITY_KEYS, 'MODALITY')
            self.set_struct_value(modality, 'attributes', {})

        for pattern in struct['patterns']:
            self.check_mandatory_keys(pattern, PATTERN_KEYS, 'PATTERN')
            self.set_struct_value(pattern, 'list_name')
            self.set_struct_value(pattern, 'name_pattern')

        for path in struct['paths']:
            self.check_mandatory_keys(path, PATH_KEYS, 'PATH')
            self.set_struct_value(path, 'name')
            self.set_struct_value(path, 'list_name')

        for directory in struct['load_sample_from_dir']:
            self.check_mandatory_keys(directory, LOAD_FROM_DIR_KEYS, 'DIRECTORY')
            self.set_struct_value(directory, 'add_to_load_regexp')
            self.set_struct_value(directory, 'add_to_load')

        patch_size, sampler = None, None
        if struct['queue'] is not None:
            self.check_mandatory_keys(struct['queue'], QUEUE_KEYS, 'QUEUE')
            self.check_mandatory_keys(struct['queue']['sampler'], SAMPLER_KEYS, 'SAMPLER')
            self.check_mandatory_keys(
                struct['queue']['sampler']['attributes'], SAMPLER_ATTRIBUTES_KEYS, 'SAMPLER ATTRIBUTES'
            )
            self.set_struct_value(struct['queue'], 'attributes', {})

            patch_size = struct['queue']['sampler']['attributes']['patch_size']

            if 'label_probabilities' in struct['queue']['sampler']['attributes']:
                for key, value in struct['queue']['sampler']['attributes']['label_probabilities'].items():
                    if isinstance(key, str) and key.isdigit():
                        del struct['queue']['sampler']['attributes']['label_probabilities'][key]
                        struct['queue']['sampler']['attributes']['label_probabilities'][int(key)] = value

        if return_string:
            return struct
        self.json_config['data'] = deepcopy(struct) #struct.copy()
        self.save_json(struct, 'data.json')

        # Make imports
        if struct['collate_fn'] is not None:
            struct['collate_fn'] = parse_function_import(struct['collate_fn'])

        if struct['queue'] is not None:
            sampler, _ = parse_object_import(struct['queue']['sampler'])
            struct['queue']['sampler'] = sampler
            struct['queue']['attributes'].update({'num_workers': struct['num_workers'], 'sampler': sampler})

        return struct, patch_size, sampler

    def parse_transform_file(self, file, return_string=False):
        def parse_transform(t):
            attributes = t.get('attributes') or {}
            if t.get('is_custom'):
                t_class = parse_function_import(t)
            else:
                t_class = getattr(torchio.transforms, t['name'])
            if t.get('is_selection'):
                t_dict = {}
                for p_and_t in t['transforms']:
                    t_dict[parse_transform(p_and_t['transform'])] = p_and_t['prob']
                return t_class(t_dict, **attributes)
            else:
                return t_class(**attributes)

        struct = self.read_json(file)

        self.check_mandatory_keys(struct, TRANSFORM_KEYS, 'TRANSFORM CONFIG FILE')

        if return_string:
            return struct
        self.json_config['transform'] = deepcopy(struct) #struct.copy()
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
        self.json_config['model'] = deepcopy(struct) #struct.copy()
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
                c = parse_method_import(criterion)
                c_list.append(c)
            return c_list

        struct = self.read_json(file)

        self.check_mandatory_keys(struct, RUN_KEYS, 'RUN CONFIG FILE')
        self.set_struct_value(struct, 'data_getter', 'get_segmentation_data')
        self.set_struct_value(struct, 'seed')
        self.set_struct_value(struct, 'current_epoch')
        self.set_struct_value(struct, 'log_frequency', 10)

        files = glob.glob(os.path.join(self.results_dir, 'model_ep*'))
        if len(files) == 0:
            struct['current_epoch'] = 1
        else:
            last_model = max(files, key=os.path.getctime)
            matches = re.findall('ep([0-9]+)', last_model)
            struct['current_epoch'] = int(matches[-1]) + 1 if matches else 1

        # Optimizer
        self.check_mandatory_keys(struct['optimizer'], OPTIMIZER_KEYS, 'OPTIMIZER')
        self.set_struct_value(struct['optimizer'], 'attributes', {})
        self.set_struct_value(struct['optimizer'], 'learning_rate_strategy')
        self.set_struct_value(struct['optimizer'], 'learning_rate_strategy_attributes', {})

        # Save
        self.check_mandatory_keys(struct['save'], SAVE_KEYS, 'SAVE')
        self.set_struct_value(struct['save'], 'batch_recorder', 'record_segmentation_batch')
        self.set_struct_value(struct['save'], 'prediction_saver', 'save_segmentation_prediction')

        # Validation
        self.set_struct_value(struct['validation'], 'eval_frequency')
        self.set_struct_value(struct['validation'], 'whole_image_inference_frequency')
        self.set_struct_value(struct['validation'], 'patch_overlap', 8)
        self.set_struct_value(struct['validation'], 'reporting_metrics', [])

        if return_string:
            return struct
        self.json_config['run'] = deepcopy(struct) #struct.copy()
        self.save_json(struct, 'run.json')

        # Make imports
        # Criteria
        struct['criteria'] = parse_criteria(struct['criteria'])

        # Optimizer
        struct['optimizer']['optimizer_class'] = parse_function_import(struct['optimizer'])
        strategy = struct['optimizer']['learning_rate_strategy']
        if strategy is not None:
            struct['optimizer']['learning_rate_strategy'] = parse_function_import(strategy)

        # Validation
        struct['validation']['reporting_metrics'] = parse_criteria(struct['validation']['reporting_metrics'])

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
        def update_subject(subject_to_update, ref_modalities, mod_name, mod_path):
            image_type = ref_modalities[mod_name]['type']
            image_attributes = ref_modalities[mod_name]['attributes']
            subject_to_update.update({
                mod_name: torchio.Image(mod_path, image_type, **image_attributes)
            })

        def get_relevant_dict(default_dict, train_dict, val_dict, test_dict, dict_name=None):
            if dict_name == 'train':
                return train_dict
            if dict_name == 'val':
                return val_dict
            if dict_name == 'test':
                return test_dict
            return default_dict

        def dict2subjects(subject_dict, ref_modalities):
            subject_list = []
            for n, s in subject_dict.items():
                if not (set(s.keys())).issuperset(ref_modalities.keys()):
                    raise KeyError(f'A modality is missing for subject {n}, {s.keys()} were found but '
                                   f'at least {ref_modalities.keys()} were expected')
                subject_list.append(torchio.Subject(s))
            return subject_list

        def get_name(name_pattern, string):
            if name_pattern is None:
                return os.path.relpath(string, Path(string).parent)
            else:
                matches = re.findall(name_pattern, string)
                return matches[-1]

        def check_modalities(ref_modalities, modalities, subject_name):
            if not set(modalities.keys()).issubset(ref_modalities.keys()):
                raise KeyError(f'At least one modality of {modalities.keys()} from {subject_name} is not in the '
                               f'reference modalities {ref_modalities.keys()}')

        def create_dataset(subject_list, transforms):
            if len(subject_list) == 0:
                return []
            final_transform = torchio.transforms.Compose(transforms)
            return torchio.ImagesDataset(subject_list, transform=final_transform)

        train_set, val_set, test_set = [], [], []

        # Retrieve subjects using load_sample_from_dir
        if len(data_struct['load_sample_from_dir']) > 0:
            for sample_dir in data_struct['load_sample_from_dir']:
                #print('parsing sample dir addin {}'.format(sample_dir['root']))

                sample_files = glob.glob(os.path.join(sample_dir['root'], 'sample*pt'))
                self.logger.log(logging.INFO, f'{len(sample_files)} subjects in the {sample_dir["list_name"]} set')
                transform = torchio.transforms.Compose(transform_struct[f'{sample_dir["list_name"]}_transforms'])
                dataset = torchio.ImagesDataset(sample_files,
                                                load_from_dir=True, transform=transform,
                                                add_to_load=sample_dir['add_to_load'],
                                                add_to_load_regexp=sample_dir['add_to_load_regexp'])
                if sample_dir["list_name"] == 'train':
                    train_set = dataset
                elif sample_dir["list_name"] == 'val':
                    val_set = dataset
                else:
                    raise ValueError('list_name attribute from load_from_dir must be either train or val')

            return train_set, val_set, test_set

        subjects, train_subjects, val_subjects, test_subjects = {}, {}, {}, {}

        # Retrieve subjects using csv file
        for csv_file in data_struct['csv_file']:
            check_modalities(data_struct['modalities'], csv_file['modalities'], csv_file['root'])

            relevant_dict = get_relevant_dict(subjects, train_subjects, val_subjects, test_subjects,
                                              csv_file['list_name'])
            res = pd.read_csv(csv_file["root"])

            for suj_idx in range(len(res)):
                subject = {}
                for modality_name, modality_column_name in csv_file['modalities'].items():
                    subject_file_path = res[modality_column_name][suj_idx]
                    update_subject(subject, data_struct['modalities'], modality_name, subject_file_path)

                relevant_dict[suj_idx] = subject

        # Retrieve subjects using patterns
        for pattern in data_struct['patterns']:
            check_modalities(data_struct['modalities'], pattern['modalities'], pattern['root'])

            relevant_dict = get_relevant_dict(subjects, train_subjects, val_subjects, test_subjects,
                                              pattern['list_name'])
            for folder_path in sorted(glob.glob(pattern['root'])):  # so that we get alphabetic order if no shuffle
                name = get_name(pattern['name_pattern'], folder_path)
                subject = relevant_dict.get(name) or {}

                for modality_name, modality_path in pattern['modalities'].items():
                    modality_path = glob.glob(os.path.join(folder_path, modality_path))[0]
                    update_subject(subject, data_struct['modalities'], modality_name, modality_path)

                relevant_dict[name] = subject

        # Retrieve subjects using paths
        for path in data_struct['paths']:
            default_name = f'{len(subjects) + len(train_subjects) + len(val_subjects) + len(test_subjects):0>6}'
            name = path['name'] or default_name
            check_modalities(data_struct['modalities'], path['modalities'], name)
            relevant_dict = get_relevant_dict(subjects, train_subjects, val_subjects, test_subjects, path['list_name'])
            subject = relevant_dict.get(name) or {}

            for modality_name, modality_path in path['modalities'].items():
                update_subject(subject, data_struct['modalities'], modality_name, modality_path)

            relevant_dict[name] = subject

        # Create torchio.Subjects from dictionaries
        subjects = dict2subjects(subjects, data_struct['modalities'])
        train_subjects = dict2subjects(train_subjects, data_struct['modalities'])
        val_subjects = dict2subjects(val_subjects, data_struct['modalities'])
        test_subjects = dict2subjects(test_subjects, data_struct['modalities'])

        if data_struct['subject_shuffle']:
            np.random.seed(data_struct['subject_seed'])
            np.random.shuffle(subjects)
        n_subjects = len(subjects)

        # Split between train, validation and test sets
        end_train = int(round(data_struct['repartition'][0] * n_subjects))
        end_val = end_train + int(round(data_struct['repartition'][1] * n_subjects))
        train_subjects += subjects[:end_train]
        val_subjects += subjects[end_train:end_val]
        test_subjects += subjects[end_val:]

        self.log(f'{len(train_subjects)} subjects in the train set')
        self.log(f'{len(val_subjects)} subjects in the validation set')
        self.log(f'{len(test_subjects)} subjects in the test set')

        train_set = create_dataset(train_subjects, transform_struct['train_transforms'])
        val_set = create_dataset(val_subjects, transform_struct['val_transforms'])
        test_set = create_dataset(test_subjects, transform_struct['val_transforms'])
        return train_set, val_set, test_set

    def generate_data_loaders(self, struct):
        train_loader, val_loader = None, None

        if struct['num_workers'] == -1:
            struct['num_workers'] = multiprocessing.cpu_count()

        if struct['batch_seed'] is not None:
            torch.manual_seed(struct['batch_seed'])

        if struct['queue'] is None:
            if len(self.train_set) > 0:
                train_loader = DataLoader(self.train_set, self.batch_size, shuffle=struct['batch_shuffle'],
                                          num_workers=struct['num_workers'], collate_fn=struct['collate_fn'])
            if len(self.val_set) > 0:
                val_loader = DataLoader(self.val_set, self.batch_size, shuffle=struct['batch_shuffle'],
                                        num_workers=struct['num_workers'], collate_fn=struct['collate_fn'])
        else:
            if len(self.train_set) > 0:
                train_queue = torchio.Queue(self.train_set, **struct['queue']['attributes'])
                train_loader = DataLoader(train_queue, self.batch_size, collate_fn=struct['collate_fn'])
            if len(self.val_set) > 0:
                val_queue = torchio.Queue(self.val_set, **struct['queue']['attributes'])
                val_loader = DataLoader(val_queue, self.batch_size, collate_fn=struct['collate_fn'])
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
                summary, _ = summary_string(model, (1, *input_shape), self.batch_size, device)
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
                raise ValueError(f'Impossible to load model from {file}, this file does not exist')

        if hasattr(model_class, 'load'):
            model, _ = model_class.load(file)
        else:
            model.load_state_dict(torch.load(file))

        self.loaded_model_name = os.path.basename(file)[:-8] #to remove .pth.tar

        return return_model(model, file)

    def run(self):
        if self.mode in ['train', 'eval', 'infer']:
            model, device = self.load_model(self.model_structure)

            model_runner = RunModel(model, self.train_loader, self.val_loader, self.val_set, self.test_set,
                                    self.image_key_name, self.label_key_name, self.logger, self.debug_logger,
                                    self.results_dir, self.batch_size, self.patch_size, self.run_structure)

            if self.mode == 'train':
                model_runner.train()
            elif self.mode == 'eval':
                model_runner.eval(model_name=self.loaded_model_name,
                                  eval_csv_basename=self.model_structure['eval_csv_basename'])
            else:
                model_runner.infer()

        # Other case would typically be visualization for example
        if self.mode == 'visualization':
            viz_structure = self.parse_visualization_file(self.main_structure['visualization'])
            viz_structure['kwargs'].update({'image_key_name': self.image_key_name})

            if viz_structure['set'] == 'train':
                viz_set = self.train_set
            else:
                viz_set = self.val_set

            if 0 <= self.viz < 4:
                if self.viz == 1:
                    viz_structure['kwargs'].update({
                        'label_key_name': self.label_key_name
                    })

                elif self.viz == 2:
                    viz_structure['kwargs'].update({
                        'patch_sampler': self.sampler
                    })

                elif self.viz == 3:
                    viz_structure['kwargs'].update({
                        'label_key_name': self.label_key_name,
                        'patch_sampler': self.sampler
                    })

            if self.viz >= 4:
                model_structure = self.parse_model_file(self.model_structure)
                run_structure = self.parse_run_file(self.main_structure['run'])
                model, device = self.load_model(model_structure)

                sample = viz_set[viz_structure['sample']]
                viz_set = [sample]

                model_runner = RunModel(model, [], None, [], viz_set,
                                        self.image_key_name, self.label_key_name, None, None,
                                        self.results_dir, self.batch_size, self.patch_size, run_structure)

                with torch.no_grad():
                    model_runner.model.eval()
                    volume, target = model_runner.data_getter(sample)
                    if self.patch_size is not None:
                        prediction = model_runner.make_prediction_on_whole_volume(sample)
                    else:
                        prediction = model_runner.model(volume.unsqueeze(0))[0]

                prediction = F.softmax(prediction, dim=0).to('cpu')

                viz_structure['kwargs'].update({
                    'label_key_name': self.label_key_name
                })

                if self.viz == 4:
                    false_positives = minimum_t_norm(prediction[0], target, True)
                    sample[self.label_key_name]['data'] = false_positives
                    viz_set = [sample]

                elif self.viz == 5:
                    ground_truth = deepcopy(sample)
                    sample[self.label_key_name]['data'] = prediction[0].unsqueeze(0)
                    viz_set = [ground_truth, sample]

            return PlotDataset(viz_set, **viz_structure['kwargs'])
