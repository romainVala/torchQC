""" Compute differences of volumes between PV and discretized PV maps."""
import torch
import numpy as np
import tqdm


def compute_differences(config):
    labels = config.label_key_name
    differences = {label: [] for label in labels}

    for sample in tqdm.tqdm(config.val_set):
        discretized_volumes = []
        for label in labels:
            discretized_volumes.append(sample[label]['data'])
        discretized_volumes = torch.cat(discretized_volumes, dim=-4)
        discretized_volumes = torch.argmax(discretized_volumes, dim=-4)

        for i, label in enumerate(labels):
            pv_volume = sample[label]['data'].sum()
            bin_volume = (discretized_volumes == i).sum()
            differences[label].append((pv_volume - bin_volume) / pv_volume)

    stats = {
        label: {
            'mean_diff': np.mean(differences[label]),
            'std_diff': np.std(differences[label]),
            'min_diff': np.min(differences[label]),
            'max_diff': np.max(differences[label]),
            'mean_abs_diff': np.mean(np.abs(differences[label])),
            'std_abs_diff': np.std(np.abs(differences[label])),
            'min_abs_diff': np.min(np.abs(differences[label])),
            'max_abs_diff': np.max(np.abs(differences[label])),
        } for label in labels
    }

    return stats
