"""
Python script to extract grey and white matter partial volume maps from FreeSurfer surface files of a subject.
This scripts uses mrtrix mesh2voxel function to construct volumes from surfaces.

Input: subject folder path
Output: results folder path
"""

import subprocess
import nibabel as nib
from nibabel.freesurfer.io import read_geometry
import os
import shutil
import tempfile
from pathlib import Path
import argparse
from partial_volumes import process_volumes


def mrtrix_mesh2vox(surface_path, template_path, temp_dir, output_prefix):
    """
    Create a partial volume map from a surface and a reference template using mrtrix mesh2voxel command.

    :param surface_path: path to the surface file
    :param template_path: path to the template file
    :param temp_dir: path to temporary directory to which temporary files are saved
    :param output_prefix: prefix to output file
    """
    # Adapt affine translation using metadata
    template = nib.load(template_path)
    _, _, meta = read_geometry(surface_path, read_metadata=True)

    template = nib.as_closest_canonical(template)
    affine = template.affine.copy()
    affine[:-1, -1] = template.affine[:-1, -1] - meta['cras']

    new_template = nib.Nifti1Image(template.dataobj, affine)
    new_template_path = temp_dir / 'template.mgz'
    nib.save(new_template, new_template_path)

    # Reconstruct volume from mesh
    subprocess.run(['mesh2voxel', surface_path, new_template_path, temp_dir / f'{output_prefix}_output.mgz'])

    # Save the reconstructed volume with the right affine
    output = nib.load(temp_dir / f'{output_prefix}_output.mgz')
    new_output = nib.Nifti1Image(output.dataobj, template.affine)
    # nib.save(new_output, output_path)

    return new_output


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-sf', '--subject_folder', type=str, help='Path to the folder of a subject')
    parser.add_argument('-rf', '--results_folder', type=str,
                        help='Path to the folder in which the results will be saved')
    args = parser.parse_args()
    subject_folder, results_folder = args.subject_folder, args.results_folder

    # Create temporary folder
    temp_directory = Path(os.path.join(tempfile.gettempdir(), os.urandom(24).hex()))
    temp_directory.mkdir(exist_ok=True)

    # Set all paths
    template_path = subject_folder + '/mri/T1.mgz'

    left_white_surface_path = subject_folder + '/surf/lh.white'
    right_white_surface_path = subject_folder + '/surf/rh.white'

    left_pial_surface_path = subject_folder + '/surf/lh.pial'
    right_pial_surface_path = subject_folder + '/surf/rh.pial'

    # Generate volumes
    left_white_volume = mrtrix_mesh2vox(left_white_surface_path, template_path, temp_directory, 'lh_white')
    right_white_volume = mrtrix_mesh2vox(right_white_surface_path, template_path, temp_directory, 'rh_white')

    left_pial_volume = mrtrix_mesh2vox(left_pial_surface_path, template_path, temp_directory, 'lh_pial')
    right_pial_volume = mrtrix_mesh2vox(right_pial_surface_path, template_path, temp_directory, 'rh_pial')

    # Process volumes
    process_volumes(left_white_volume, right_white_volume, left_pial_volume, right_pial_volume, results_folder)

    # Remove temporary files
    shutil.rmtree(temp_directory)
