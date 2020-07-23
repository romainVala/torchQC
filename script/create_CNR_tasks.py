import os
import argparse
import commentjson as json
from pathlib import Path
from itertools import product
from copy import deepcopy
from segmentation.utils import generate_json_document


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-ef', '--experiment_folder', type=str, help='Path to a folder containing a main.json file'
                                                                     'to run segmentation experiments')
    parser.add_argument('-rf', '--results_folder', type=str,
                        help='Path to the folder in which the results will be saved')
    parser.add_argument('-ms', '--main_structure', type=str, default='GM', help='Structure of interest')
    parser.add_argument('-r', '--references', type=str, nargs='+', help='Structures to compare to main structure')
    parser.add_argument('-nr', '--noise_range', type=float, nargs='+', help='Range of noise levels')
    parser.add_argument('-sl', '--signal_level', type=float, help='Level of signal of the main structure')
    parser.add_argument('-sr', '--signal_range', type=float, nargs='+',
                        help='Range of signal levels for the other structures')
    parser.add_argument('-s', '--save', type=bool, default=False, help='If True, generated images will be saved '
                                                                       'in results_dir')

    args = parser.parse_args()

    experiment_folder = args.experiment_folder
    results_folder = args.results_folder
    main_structure = args.main_structure
    references = args.references
    noise_range = args.noise_range
    signal_level = args.signal_level
    signal_range = args.signal_range
    save = args.save

    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    # Find transform file
    with open(os.path.join(experiment_folder, 'main.json')) as f:
        transform_file_path = json.load(f)['transform']

    # Replace relative path if needed
    if Path(transform_file_path).parent.anchor == '':
        transform_file_path = os.path.join(experiment_folder, transform_file_path)

    # Read transform file
    with open(transform_file_path) as f:
        initial_transform_struct = json.load(f)

    # Expect to find RandomLabelsToImage and RandomNoise in the transform file
    if 'val_transforms' not in initial_transform_struct:
        raise ValueError(f'"val_transforms" must be in transform structure {initial_transform_struct}')

    try:
        image_transform = next(
            filter(lambda x: x['name'] == 'RandomLabelsToImage', initial_transform_struct['val_transforms'])
        )
    except StopIteration:
        raise ValueError(f'A "RandomLabelsToImage" transform must be present in '
                         f'validation transforms {initial_transform_struct["val_transforms"]}')
    image_transform_idx = initial_transform_struct['val_transforms'].index(image_transform)

    try:
        noise_transform = next(
            filter(lambda x: x['name'] == 'RandomNoise', initial_transform_struct['val_transforms'])
        )
    except StopIteration:
        raise ValueError(f'A "RandomNoise" transform must be present in '
                         f'validation transforms {initial_transform_struct["val_transforms"]}')
    noise_transform_idx = initial_transform_struct['val_transforms'].index(noise_transform)

    # Modify main structure signal level in image transform
    image_transform['attributes']['gaussian_parameters'][main_structure]['mean'] = signal_level

    initial_transform_struct['save'] = save

    commands = []

    # Create extra transform files from transform file
    for ref, ref_signal_level, noise_level in product(references, signal_range, noise_range):
        new_struct = deepcopy(initial_transform_struct)

        new_noise_transform = deepcopy(noise_transform)
        new_noise_transform['attributes']['std'] = [noise_level, noise_level]

        new_image_transform = deepcopy(image_transform)
        new_image_transform['attributes']['gaussian_parameters'][ref]['mean'] = ref_signal_level

        new_struct['val_transforms'][noise_transform_idx] = new_noise_transform
        new_struct['val_transforms'][image_transform_idx] = new_image_transform

        suffix = f'_ms_{main_structure}_sl_{signal_level}_r_{ref}_rl_{ref_signal_level}_nl_{noise_level}'

        # Save new file
        file_path = os.path.join(
            results_folder,
            f'extra_transform{suffix}'
        )
        generate_json_document(file_path, **{
            'transform': new_struct,
            'results_dir': os.path.join(results_folder, f'results{suffix}')
        })

        cmd = f'python segmentation/segmentation_pipeline.py -f {os.path.join(experiment_folder, "main.json")} ' \
              f'-m eval -e {os.path.join(results_folder, "extra_transform" + suffix)}'

        commands.append(cmd)

    # Create text file with the commands to run all experiments
    with open(os.path.join(results_folder, 'commands.txt'), 'w') as f:
        f.write('\n'.join(commands))
