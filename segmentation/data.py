import glob
import json
import torchio


def load_data(folder, data_filename='data.json'):
    def update_subject(sub, mods, mod_name, mod_path):
        subject_type = mods.get(mod_name).get('type')
        subject_attributes = mods.get(mod_name).get('attributes') or {}

        sub.update({
            mod_name: torchio.Image(mod_path, subject_type, **subject_attributes)
        })

    with open(folder + data_filename) as file:
        info = json.load(file)
        modalities = info.get('modalities')
        patterns = info.get('patterns') or []
        paths = info.get('paths') or []

    subjects = []

    # Using patterns
    for pattern in patterns:
        root = pattern.get('root')
        for folder_path in glob.glob(root):
            path_split = folder_path.split('/')
            name = path_split[-1] if len(path_split[-1]) > 0 else path_split[-2]
            subject = {'name': name}

            for modality_name, modality_path in pattern.get('modalities').items():
                modality_path = glob.glob(folder_path + modality_path)[0]
                update_subject(subject, modalities, modality_name, modality_path)

            subjects.append(torchio.Subject(subject))

    # Using paths
    for path in paths:
        name = path.get('name') or f'{len(subjects):0>6}'
        subject = {'name': name}

        for modality_name, modality_path in path.get('modalities').items():
            update_subject(subject, modalities, modality_name, modality_path)

        subjects.append(torchio.Subject(subject))

    return subjects


def generate_dataset(visualize=False):
    pass


def generate_json_document(filename, **kwargs):
    with open(filename, 'w') as file:
        json.dump(kwargs, file)
