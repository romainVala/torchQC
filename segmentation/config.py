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
from torch.utils.data import DataLoader
from segmentation.utils import parse_object_import, parse_function_import, parse_method_import
from segmentation.run_model import RunModel
from torch_summary import summary_string


MAIN_KEYS = ['data', 'transform', 'model']

DATA_KEYS = ['modalities', 'batch_size', 'image_key_name', 'label_key_name']
MODALITY_KEYS = ['type']
PATTERN_KEYS = ['root', 'modalities']
PATH_KEYS = ['name', 'modalities']
QUEUE_KEYS = ['sampler']
SAMPLER_KEYS = ['name', 'module', 'attributes']
SAMPLER_ATTRIBUTES_KEYS = ['patch_size']

TRANSFORM_KEYS = ['train_transforms', 'val_transforms']

MODEL_KEYS = ['name', 'module']

RUN_KEYS = ['criteria', 'optimizer', 'save', 'validation', 'n_epochs']
OPTIMIZER_KEYS = ['name', 'module']
SAVE_KEYS = ['record_frequency']


class Config:
    def __init__(self, main_file, results_dir, logger, debug_logger=None, mode='train'):
        self.mode = mode
        self.logger = logger
        self.debug_logger = debug_logger
        self.main_structure = self.parse_main_file(main_file)

        self.results_dir = results_dir
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)

        data_structure, patch_size = self.parse_data_file(self.main_structure['data'])
        transform_structure = self.parse_transform_file(self.main_structure['transform'])

        self.batch_size = data_structure['batch_size']
        self.patch_size = self.parse_patch_size(patch_size)
        self.image_key_name = data_structure['image_key_name']
        self.label_key_name = data_structure['label_key_name']

        self.train_subjects, self.val_subjects, self.test_subjects = self.load_subjects(data_structure)

        self.train_set, self.val_set, self.test_set = self.generate_datasets(transform_structure)
        self.train_loader, self.val_loader = self.generate_data_loaders(data_structure)

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
        value = struct.get(key) or default_value
        struct[key] = value

    def load_json(self, file, name):
        with open(file) as f:
            struct = json.load(f)
            self.logger.log(logging.INFO, f'******** {name} ********')
            self.logger.log(logging.INFO, json.dumps(struct, indent=4, sort_keys=True))
        return struct

    def debug(self, info):
        if self.debug_logger is not None:
            self.debug_logger.log(logging.DEBUG, info)

    def parse_main_file(self, file):
        struct = self.load_json(file, 'MAIN CONFIG FILE')
        additional_key = []
        if self.mode in ['train', 'eval', 'infer']:
            additional_key = ['run']
        self.check_mandatory_keys(struct, MAIN_KEYS + additional_key, 'MAIN CONFIG FILE')
        return struct

    def parse_data_file(self, file):
        struct = self.load_json(file, 'DATA CONFIG FILE')
        self.check_mandatory_keys(struct, DATA_KEYS, 'DATA CONFIG FILE')
        self.set_struct_value(struct, 'patterns', [])
        self.set_struct_value(struct, 'paths', [])
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

        if struct['num_workers'] == -1:
            struct['num_workers'] = multiprocessing.cpu_count()

        if struct['collate_fn'] is not None:
            struct['collate_fn'] = parse_function_import(struct['collate_fn'])

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

        patch_size = None
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

            sampler, _ = parse_object_import(struct['queue']['sampler'])
            struct['queue']['sampler'] = sampler
            struct['queue']['attributes'].update({'num_workers': struct['num_workers'], 'sampler': sampler})

        return struct, patch_size

    def parse_transform_file(self, file):
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

        struct = self.load_json(file, 'TRANSFORM CONFIG FILE')
        self.check_mandatory_keys(struct, TRANSFORM_KEYS, 'TRANSFORM CONFIG FILE')
        transform_list = []
        for transform in struct['train_transforms']:
            transform_list.append(parse_transform(transform))
        struct['train_transforms'] = transform_list

        transform_list = []
        for transform in struct['val_transforms']:
            transform_list.append(parse_transform(transform))
        struct['val_transforms'] = transform_list

        return struct

    def parse_model_file(self, file):
        struct = self.load_json(file, 'MODEL CONFIG FILE')
        self.check_mandatory_keys(struct, MODEL_KEYS, 'MODEL CONFIG FILE')

        self.set_struct_value(struct, 'last_one', True)
        self.set_struct_value(struct, 'path', None)
        self.set_struct_value(struct, 'device', 'cuda')

        if struct['device'] == 'cuda' and torch.cuda.is_available():
            struct['device'] = torch.device('cuda')
        else:
            struct['device'] = torch.device('cpu')

        struct['model'], struct['model_class'] = parse_object_import(struct)

        return struct

    def parse_run_file(self, file):
        def parse_criteria(criterion_list):
            c_list = []
            for criterion in criterion_list:
                c = parse_method_import(criterion)
                c_list.append(c)
            return c_list

        struct = self.load_json(file, 'RUN CONFIG FILE')
        self.check_mandatory_keys(struct, RUN_KEYS, 'RUN CONFIG FILE')
        self.set_struct_value(struct, 'data_getter', 'get_segmentation_data')
        self.set_struct_value(struct, 'seed')
        self.set_struct_value(struct, 'current_epoch')
        self.set_struct_value(struct, 'log_frequency', 10)

        struct['criteria'] = parse_criteria(struct['criteria'])

        files = glob.glob(self.results_dir + '/model*')
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

        struct['optimizer']['optimizer_class'] = parse_function_import(struct['optimizer'])
        strategy = struct['optimizer']['learning_rate_strategy']
        if strategy is not None:
            struct['optimizer']['learning_rate_strategy'] = parse_function_import(strategy)

        # Save
        self.check_mandatory_keys(struct['save'], SAVE_KEYS, 'SAVE')
        self.set_struct_value(struct['save'], 'batch_recorder', 'record_segmentation_batch')

        # Validation
        self.set_struct_value(struct['validation'], 'eval_frequency', np.inf)
        self.set_struct_value(struct['validation'], 'whole_image_inference_frequency', np.inf)
        self.set_struct_value(struct['validation'], 'patch_overlap', 8)
        self.set_struct_value(struct['validation'], 'reporting_metrics', [])

        struct['validation']['reporting_metrics'] = parse_criteria(struct['validation']['reporting_metrics'])

        return struct

    @staticmethod
    def parse_patch_size(patch_size):
        if isinstance(patch_size, int):
            return patch_size, patch_size, patch_size
        return patch_size

    @staticmethod
    def load_subjects(struct):
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
                if not (set(subject.keys())).issuperset(ref_modalities.keys()):
                    raise KeyError(f'A modality is missing for subject {n}, {s.keys()} were found but '
                                   f'at least {ref_modalities.keys()} were expected')
                subject_list.append(torchio.Subject(s))
            return subject_list

        def get_name(name_pattern, string):
            if name_pattern is None:
                string_split = string.split('/')
                return string_split[-1] if len(string_split[-1]) > 0 else string_split[-2]
            else:
                matches = re.findall(name_pattern, string)
                return matches[-1]

        def check_modalities(ref_modalities, modalities, subject_name):
            if not set(modalities.keys()).issubset(ref_modalities.keys()):
                raise KeyError(f'At least one modality of {modalities.keys()} from {subject_name} is not in the '
                               f'reference modalities {ref_modalities.keys()}')

        subjects, train_subjects, val_subjects, test_subjects = {}, {}, {}, {}

        # Retrieve subjects using patterns
        for pattern in struct['patterns']:
            check_modalities(struct['modalities'], pattern['modalities'], pattern['root'])

            relevant_dict = get_relevant_dict(subjects, train_subjects, val_subjects, test_subjects,
                                              pattern['list_name'])
            for folder_path in sorted(glob.glob(pattern['root'])):  # so that we get alphabetic order if no shuffle
                name = get_name(pattern['name_pattern'], folder_path)
                subject = relevant_dict.get(name) or {}

                for modality_name, modality_path in pattern['modalities'].items():
                    modality_path = glob.glob(folder_path + modality_path)[0]
                    update_subject(subject, struct['modalities'], modality_name, modality_path)

                subject['name'] = name
                relevant_dict[name] = subject

        # Retrieve subjects using paths
        for path in struct['paths']:
            default_name = f'{len(subjects) + len(train_subjects) + len(val_subjects) + len(test_subjects):0>6}'
            name = path['name'] or default_name
            check_modalities(struct['modalities'], path['modalities'], name)
            relevant_dict = get_relevant_dict(subjects, train_subjects, val_subjects, test_subjects, path['list_name'])
            subject = relevant_dict.get(name) or {}

            for modality_name, modality_path in path['modalities'].items():
                update_subject(subject, struct['modalities'], modality_name, modality_path)

            subject['name'] = name
            relevant_dict[name] = subject

        # Create torchio.Subjects from dictionaries
        subjects = dict2subjects(subjects, struct['modalities'])
        train_subjects = dict2subjects(train_subjects, struct['modalities'])
        val_subjects = dict2subjects(val_subjects, struct['modalities'])
        test_subjects = dict2subjects(test_subjects, struct['modalities'])

        if struct['subject_shuffle']:
            np.random.seed(struct['subject_seed'])
            np.random.shuffle(subjects)
        n_subjects = len(subjects)

        # Split between train, validation and test sets
        end_train = int(round(struct['repartition'][0] * n_subjects))
        end_val = end_train + int(round(struct['repartition'][1] * n_subjects))
        train_subjects += subjects[:end_train]
        val_subjects += subjects[end_train:end_val]
        test_subjects += subjects[end_val:]

        return train_subjects, val_subjects, test_subjects

    def generate_datasets(self, struct):
        def create_dataset(subjects, transforms):
            if len(subjects) == 0:
                return
            transform = torchio.transforms.Compose(transforms)
            return torchio.ImagesDataset(subjects, transform=transform)
        train_set = create_dataset(self.train_subjects, struct['train_transforms'])
        val_set = create_dataset(self.val_subjects, struct['val_transforms'])
        test_set = create_dataset(self.test_subjects, struct['val_transforms'])
        return train_set, val_set, test_set

    def generate_data_loaders(self, struct):
        if struct['batch_seed'] is not None:
            torch.manual_seed(struct['batch_seed'])

        if struct['queue'] is None:
            train_loader = DataLoader(self.train_set, self.batch_size, shuffle=struct['batch_shuffle'],
                                      num_workers=struct['num_workers'], collate_fn=struct['collate_fn'])
            val_loader = DataLoader(self.val_set, self.batch_size, shuffle=struct['batch_shuffle'],
                                    num_workers=struct['num_workers'], collate_fn=struct['collate_fn'])
        else:
            train_queue = torchio.Queue(self.train_set, **struct['queue']['attributes'])
            train_loader = DataLoader(train_queue, self.batch_size, collate_fn=struct['collate_fn'])
            val_queue = torchio.Queue(self.val_set, **struct['queue']['attributes'])
            val_loader = DataLoader(val_queue, self.batch_size, collate_fn=struct['collate_fn'])
        return train_loader, val_loader

    def load_model(self, struct):
        def return_model(m, f=None):
            if f is not None:
                self.logger.log(logging.INFO, f'Using model from file {f}')
            else:
                self.logger.log(logging.INFO, 'Using new model')
            device = struct['device']
            self.logger.log(logging.INFO, 'Model description:')
            self.logger.log(logging.INFO, model)

            m.to(device)

            if self.patch_size is not None:
                summary, _ = summary_string(model, (1, *self.patch_size), self.batch_size, device)
                self.logger.log(logging.INFO, 'Model summary:')
                self.logger.log(logging.INFO, summary)
            return m, device

        self.logger.log(logging.INFO, '******** Model  ********')

        model = struct['model']
        model_class = struct['model_class']

        if struct['last_one']:
            files = glob.glob(self.results_dir + '/model*.pth*')
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

        return return_model(model, file)

    def run(self):
        if self.mode in ['train', 'eval', 'infer']:
            model_structure = self.parse_model_file(self.main_structure['model'])
            run_structure = self.parse_run_file(self.main_structure['run'])
            model, device = self.load_model(model_structure)

            model_runner = RunModel(model, self.train_loader, self.val_loader, self.val_set, self.test_set,
                                    self.image_key_name, self.label_key_name, self.logger, self.debug_logger,
                                    self.results_dir, self.batch_size, self.patch_size, run_structure)

            if self.mode == 'train':
                model_runner.train()
            elif self.mode == 'eval':
                model_runner.eval()
            else:
                model_runner.infer()

        # Other case would typically be visualization for example
        else:
            pass
