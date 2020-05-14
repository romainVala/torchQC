"""
Python script to produce grey and white matter partial volume maps from HCP subject.
This scripts assumes that reconstructed volumes are available.

Input: subject folder path
Output: results folder path
"""

import nibabel as nib
import argparse
from partial_volumes import process_volumes


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-sf', '--subject_folder', type=str, help='Path to the folder of a subject')
    parser.add_argument('-rf', '--results_folder', type=str,
                        help='Path to the folder in which the results will be saved')
    args = parser.parse_args()
    subject_folder, results_folder = args.subject_folder, args.results_folder

    # Get subject number
    folder_split = subject_folder.split('/')
    subject_number = folder_split[-1] if len(folder_split[-1]) > 0 else folder_split[-2]

    # Set all paths
    left_white_volume_path = subject_folder + f'/T1w/Native/{subject_number}.L.white.native.surf.vtk.nii.gz'
    right_white_volume_path = subject_folder + f'/T1w/Native/{subject_number}.R.white.native.surf.vtk.nii.gz'

    left_pial_volume_path = subject_folder + f'/T1w/Native/{subject_number}.L.pial.native.surf.vtk.nii.gz'
    right_pial_volume_path = subject_folder + f'/T1w/Native/{subject_number}.R.pial.native.surf.vtk.nii.gz'

    # Load volumes
    left_white_volume = nib.load(left_white_volume_path)
    right_white_volume = nib.load(right_white_volume_path)

    left_pial_volume = nib.load(left_pial_volume_path)
    right_pial_volume = nib.load(right_pial_volume_path)

    # Process volumes
    process_volumes(left_white_volume, right_white_volume, left_pial_volume, right_pial_volume, results_folder,
                    subject_number)
