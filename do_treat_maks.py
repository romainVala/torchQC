"""
Python script to clean partial volumes masks.

Input: subject folder path
Output: results folder path
"""

import nibabel as nib
import os
import numpy as np
import argparse
from nibabel.processing import resample_from_to


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-sf', '--subject_folder', type=str, help='Path to the folder of a subject')
    parser.add_argument('-rf', '--results_folder', type=str,
                        help='Path to the folder in which the results will be saved')
    parser.add_argument('-r', '--resolution', type=str, help='Resolution in millimeters', choices=['07', '1', '14'])
    args = parser.parse_args()
    subject_folder, results_folder = args.subject_folder, args.results_folder
    resolution = args.resolution

    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    res_path = f'PVE_{resolution}mm'

    paths_to_main_structures = {
        'GM': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/cortex_GM.nii.gz'),
        'WM': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/cortex_WM.nii.gz'),
        'CSF': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/cortex_nonbrain.nii.gz'),
    }

    paths = {
        'L_Accu': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/L_Accu.nii.gz'),
        'L_Amyg': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/L_Amyg.nii.gz'),
        'L_Caud': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/L_Caud.nii.gz'),
        'L_Hipp': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/L_Hipp.nii.gz'),
        'L_Pall': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/L_Pall.nii.gz'),
        'L_Puta': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/L_Puta.nii.gz'),
        'L_Thal': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/L_Thal.nii.gz'),
        'R_Accu': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/R_Accu.nii.gz'),
        'R_Amyg': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/R_Amyg.nii.gz'),
        'R_Caud': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/R_Caud.nii.gz'),
        'R_Hipp': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/R_Hipp.nii.gz'),
        'R_Pall': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/R_Pall.nii.gz'),
        'R_Puta': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/R_Puta.nii.gz'),
        'R_Thal': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/R_Thal.nii.gz'),
        'BrStem': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/BrStem.nii.gz'),
        'cereb_GM': os.path.join(subject_folder, 'ROI/cereb_GM.nii.gz'),
        'cereb_WM': os.path.join(subject_folder, 'ROI/cereb_WM.nii.gz'),
        'skin': os.path.join(subject_folder, 'ROI/skin.nii.gz'),
        'skull': os.path.join(subject_folder, 'ROI/skull.nii.gz'),
        'background': os.path.join(subject_folder, 'ROI/Background.nii.gz'),
        'CSF': os.path.join(subject_folder, f'Native/{res_path}/intermediate_pvs/FAST_CSF.nii.gz')
    }

    special_keys = ['cereb_GM', 'cereb_WM', 'skin', 'skull', 'background']

    # Load volumes
    volumes = []
    idx2key = []
    affine = None
    shape = None

    for key, value in paths.items():
        volume = nib.load(value)
        if key in special_keys and resolution != '07':
            volume = resample_from_to(volume, (shape, affine), order=3)
        volumes.append(np.array(volume.dataobj).clip(0., 1.))
        idx2key.append(key)
        affine = volume.affine
        shape = volume.shape

    main_volumes = []
    main_idx2key = []

    for key, value in paths_to_main_structures.items():
        volume = nib.load(value)
        main_volumes.append(np.array(volume.dataobj).clip(0., 1.))
        main_idx2key.append(key)

    # Deal with FAST_CSF
    volumes[-1][(main_volumes[-1] > 0) + (main_volumes[0] > 0)] = 0

    # Clean structures
    volumes = np.stack(volumes)
    volumes = volumes / np.maximum(1, volumes.sum(axis=0))

    for i in range(len(volumes) - 1):
        volume = nib.Nifti1Image(volumes[i], affine)
        nib.save(volume, os.path.join(results_folder, f'{idx2key[i]}.nii.gz'))

    # Clean main structures
    main_volumes = np.stack(main_volumes)
    main_volumes = (main_volumes * (1 - volumes.sum(axis=0)) / main_volumes.sum(axis=0)).clip(0, 1)

    for i in range(len(main_volumes) - 1):
        volume = nib.Nifti1Image(main_volumes[i], affine)
        nib.save(volume, os.path.join(results_folder, f'{main_idx2key[i]}.nii.gz'))

    volume = nib.Nifti1Image(main_volumes[-1] + volumes[-1], affine)
    nib.save(volume, os.path.join(results_folder, f'{main_idx2key[-1]}.nii.gz'))
