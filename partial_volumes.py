"""
Python script to produce grey and white matter partial volume maps from HCP subject.
This scripts assumes that reconstructed volumes are available.

Input: subject folder path
Output: results folder path
"""

import numpy as np
import nibabel as nib
import os
from skimage import measure


def treat_volume(volume):
    """
    Remove isolated voxels from a volume using skimage function.

    :param volume: a nifty volume
    :return: a new nifty volume
    """
    labels = measure.label(volume.dataobj, background=0, connectivity=2)
    new_volume = np.asarray(volume.dataobj)
    new_volume[labels > 1] = 0
    new_volume = nib.Nifti1Image(new_volume, volume.affine)
    return new_volume


def process_volumes(left_white_volume, right_white_volume, left_pial_volume, right_pial_volume, results_folder,
                    subject_number=None):
    """
    Process reconstructed volumes by removing isolated voxels, aggregate left and white parts, substract
    white matter from pial volume, record quality information and clip values to [0, 1].

    :param left_white_volume: left hemisphere white matter volume
    :param right_white_volume: right hemisphere white matter volume
    :param left_pial_volume: left hemisphere pial volume
    :param right_pial_volume: right hemisphere pial volume
    :param results_folder: path to which save the processed volumes
    :param subject_number: number of the subject from which the volumes come
    """
    # Remove isolated voxels
    left_pial_volume = treat_volume(left_pial_volume)
    right_pial_volume = treat_volume(right_pial_volume)

    # Aggregate volumes
    white_volume_data = np.asarray(left_white_volume.dataobj) + np.asarray(right_white_volume.dataobj)
    grey_volume_data = np.asarray(left_pial_volume.dataobj) + np.asarray(right_pial_volume.dataobj) - white_volume_data

    # Record information
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    with open(results_folder + '/info.csv', 'a') as file:
        to_write = ['volume', 'nb_negative_voxel', 'nb_greater_than_one_voxel', 'nb_grey_voxels', '\n']
        if subject_number is not None:
            to_write.insert(0, 'subject_name')
        file.write(','.join(to_write))

        to_write = [
            'white',
            str((white_volume_data < 0).sum()),
            str((white_volume_data > 1).sum()),
            str(((white_volume_data > 0) * (white_volume_data <= 1)).sum()),
            '\n'
        ]
        if subject_number is not None:
            to_write.insert(0, str(subject_number))
        file.write(','.join(to_write))

        to_write = [
            'grey',
            str((grey_volume_data < 0).sum()),
            str((grey_volume_data > 1).sum()),
            str(((grey_volume_data > 0) * (grey_volume_data <= 1)).sum()),
            '\n'
        ]
        if subject_number is not None:
            to_write.insert(0, str(subject_number))
        file.write(','.join(to_write))
        file.write(','.join(to_write))

    # Clip values
    white_volume = nib.Nifti1Image(np.clip(white_volume_data, 0, 1), left_white_volume.affine)
    grey_volume = nib.Nifti1Image(np.clip(grey_volume_data, 0, 1), left_white_volume.affine)

    # Save volumes
    nib.save(white_volume, results_folder + '/white.nii.gz')
    nib.save(grey_volume, results_folder + '/grey.nii.gz')
