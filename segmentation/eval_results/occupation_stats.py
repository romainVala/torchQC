import os
import glob
import pandas as pd
import numpy as np
import commentjson as json


def compute_occupation_percentage(results_dir, use_all_files=False):
    # Get labels
    data_file = os.path.join(results_dir, 'data.json')
    with open(data_file) as f:
        labels = json.load(f)['label_key_name']

    occupations = {label: 0 for label in labels}

    # Get file
    files = glob.glob(os.path.join(results_dir, 'Train_*.csv'))

    if not use_all_files:
        files = files[:1]

    for file in files:
        # Create dataframe out of CSV file
        df = pd.read_csv(file, index_col=0)

        # Compute occupation
        for label in labels:
            occupied = df[f'occupied_volume_{label}']
            occupations[label] += (occupied == 0).sum() / len(occupied)

    for label in labels:
        occupations[label] *= 100 / len(files)

    print(occupations)


def compute_real_occupation(results_dir, patch=False):
    # Get labels
    data_file = os.path.join(results_dir, 'data.json')
    with open(data_file) as f:
        labels = json.load(f)['label_key_name']

    real_occupations = {label: 0 for label in labels}

    # Get file
    csv_pattern = 'Whole_image_*.csv' if patch else 'Val_*.csv'
    file = glob.glob(os.path.join(results_dir, csv_pattern))[-1]

    # Create dataframe out of CSV file
    df = pd.read_csv(file, index_col=0)

    # Compute occupation
    for label in labels:
        really_occupied = df[f'occupied_volume_{label}']
        real_occupations[label] = really_occupied.sum() / len(really_occupied) * 100

    print(real_occupations)


def compute_predicted_occupation(results_dir, patch=False):
    # Get labels
    data_file = os.path.join(results_dir, 'data.json')
    with open(data_file) as f:
        labels = json.load(f)['label_key_name']

    predicted_occupations = {label: 0 for label in labels}

    # Get file
    csv_pattern = 'Whole_image_*.csv' if patch else 'Val_*.csv'
    file = glob.glob(os.path.join(results_dir, csv_pattern))[-1]

    # Create dataframe out of CSV file
    df = pd.read_csv(file, index_col=0)

    # Compute occupation
    for label in labels:
        predicted_occupied = df[f'predicted_occupied_volume_{label}']
        predicted_occupations[label] = predicted_occupied.sum() / len(predicted_occupied) * 100

    print(predicted_occupations)


def compute_occupation_stats(results_dir, patch=False):
    # Get labels
    data_file = os.path.join(results_dir, 'data.json')
    with open(data_file) as f:
        labels = json.load(f)['label_key_name']

    volume_differences = {label: 0 for label in labels}

    # Get file
    csv_pattern = 'Whole_image_*.csv' if patch else 'Val_*.csv'
    file = glob.glob(os.path.join(results_dir, csv_pattern))[-1]

    # Create dataframe out of CSV file
    df = pd.read_csv(file, index_col=0)

    # Compute occupation
    for label in labels:
        occupied = df[f'occupied_volume_{label}']
        predicted_occupied = df[f'predicted_occupied_volume_{label}']
        differences = occupied - predicted_occupied
        volume_differences[label] = {
            'mean': np.mean(differences),
            'std': np.std(differences),
            'min': np.min(differences),
            'max': np.max(differences),
        }

    # print(volume_differences)
    return volume_differences


def compute_abs_occupation_stats(results_dir, patch=False):
    # Get labels
    data_file = os.path.join(results_dir, 'data.json')
    with open(data_file) as f:
        labels = json.load(f)['label_key_name']

    volume_differences = {label: 0 for label in labels}

    # Get file
    csv_pattern = 'Whole_image_*.csv' if patch else 'Val_*.csv'
    file = glob.glob(os.path.join(results_dir, csv_pattern))[-1]

    # Create dataframe out of CSV file
    df = pd.read_csv(file, index_col=0)

    # Compute occupation
    for label in labels:
        occupied = df[f'occupied_volume_{label}']
        predicted_occupied = df[f'predicted_occupied_volume_{label}']
        differences = occupied - predicted_occupied
        volume_differences[label] = {
            'mean': np.mean(np.abs(differences)),
            'std': np.std(np.abs(differences)),
            'min': np.min(np.abs(differences)),
            'max': np.max(np.abs(differences)),
        }

    # print(volume_differences)
    return volume_differences


def compute_dice_score_stats(results_dir, patch=False):
    # Get labels
    data_file = os.path.join(results_dir, 'data.json')
    with open(data_file) as f:
        labels = json.load(f)['label_key_name']

    dice_scores = {label: 0 for label in labels}

    # Get file
    csv_pattern = 'Whole_image_*.csv' if patch else 'Val_*.csv'
    file = glob.glob(os.path.join(results_dir, csv_pattern))[-1]

    # Create dataframe out of CSV file
    df = pd.read_csv(file, index_col=0)

    # Compute occupation
    for label in labels:
        dice_loss = df[f'metric_dice_loss_{label}']
        dice_scores[label] = {
            'mean': 1 - np.mean(dice_loss),
            'std': np.std(dice_loss),
            'min': 1 - np.max(dice_loss),
            'max': 1 - np.min(dice_loss)
        }

    # print(dice_scores)
    return dice_scores


def compute_bin_dice_score_stats(results_dir, patch=False):
    # Get labels
    data_file = os.path.join(results_dir, 'data.json')
    with open(data_file) as f:
        labels = json.load(f)['label_key_name']

    dice_scores = {label: 0 for label in labels}

    # Get file
    csv_pattern = 'Whole_image_*.csv' if patch else 'Val_*.csv'
    file = glob.glob(os.path.join(results_dir, csv_pattern))[-1]

    # Create dataframe out of CSV file
    df = pd.read_csv(file, index_col=0)

    # Compute occupation
    for label in labels:
        dice_loss = df[f'metric_bin_dice_loss_{label}']
        dice_scores[label] = {
            'mean': 1 - np.mean(dice_loss),
            'std': np.std(dice_loss),
            'min': 1 - np.max(dice_loss),
            'max': 1 - np.min(dice_loss)
        }

    # print(dice_scores)
    return dice_scores
