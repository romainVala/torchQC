""" End-to-end segmentation pipeline """
import commentjson as json
import argparse
import matplotlib.pyplot as plt
from segmentation.utils import check_mandatory_keys
from segmentation.data import load_data, generate_dataset, generate_dataloader
from segmentation.model import load_model
from segmentation.run_model import RunModel
from segmentation.visualization import parse_visualization_config_file
from plot_dataset import PlotDataset


MAIN_KEYS = ['folder', 'data', 'transform', 'loader', 'model', 'train']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='Path to configuration file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='Training or inference mode')
    parser.add_argument('-viz', '--visualization', type=int, default=0,
                        help='Visualization, 1 means that full images will be shown, 2 means that patches will be '
                             'shown, higher values mean full images with patches will be shown')
    parser.add_argument('-v', '--verbose', type=int, default=0, help='Verbosity, the higher the more verbose')
    args = parser.parse_args()

    # Load main configuration file
    with open(args.file) as file:
        info = json.load(file)

    # Check that all configuration files are listed in main file
    check_mandatory_keys(info, MAIN_KEYS, file)

    # Generate datasets and data loaders
    train_subjects, val_subjects, test_subjects = load_data(info['folder'], info['data'])

    train_set = generate_dataset(train_subjects, info['folder'], info['transform'])
    val_set = generate_dataset(val_subjects, info['folder'], info['transform'], prefix='val')
    test_set = generate_dataset(test_subjects, info['folder'], info['transform'], prefix='val')

    train_loader = generate_dataloader(train_set, info['folder'], info['loader'])
    val_loader = generate_dataloader(val_set, info['folder'], info['loader'])
    test_loader = generate_dataloader(test_set, info['folder'], info['loader'])

    # Visualize data
    if args.visualization > 0:
        kwargs = parse_visualization_config_file(info['folder'], info['visualization'])
        if args.visualization == 1:
            fig = PlotDataset(train_set, **kwargs)
        elif args.visualization == 2:
            _, batch = next(enumerate(train_loader))
            fig = PlotDataset(batch, **kwargs)
        elif args.visualization > 2:
            print('Be careful, if random transforms are applied without seed, patches will not match full images.')
            _, batch = next(enumerate(train_loader))
            fig = PlotDataset(train_set, batch=batch, batch_mapping_key='name', **kwargs)
        plt.show()

    # Load model
    model = load_model(info['folder'], info['model'])

    # Train model
    if args.mode == 'train':
        run_model = RunModel(model, train_loader, val_loader, val_set, info['folder'], info['train'])
        run_model.train()

    # Infer results on test data
    else:
        pass
