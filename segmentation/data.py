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


def generate_dataset(folder, data_filename='data.json', transform_filename='transform.json'):
    subjects = load_data(folder, data_filename)

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
