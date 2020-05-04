import os
import glob
import numpy as np
import pandas as pd
import torchio
from torchio.transforms import RescaleIntensity, Resample
from torchio import Interpolation
from torchvision.transforms import Compose
# from resample_with_fov_transform import ResampleWithFoV
from plot_dataset import PlotDataset
import nibabel as nib
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def generate_dataset(data_path, data_root='', ref_path=None, nb_subjects=5, resampling='mni', masking_method='label'):
    """
    Generate a torchio dataset from a csv file defining paths to subjects.

    :param data_path: path to a csv file
    :param data_root:
    :param ref_path:
    :param nb_subjects:
    :param resampling:
    :param masking_method:
    :return:
    """
    ds = pd.read_csv(data_path)
    ds = ds.dropna(subset=['suj'])
    np.random.seed(0)
    subject_idx = np.random.choice(range(len(ds)), nb_subjects, replace=False)
    directories = ds.iloc[subject_idx, 1]
    dir_list = directories.tolist()
    dir_list = map(lambda partial_dir: data_root + partial_dir, dir_list)

    subject_list = []
    for directory in dir_list:
        img_path = glob.glob(os.path.join(directory, 's*.nii.gz'))[0]

        mask_path = glob.glob(os.path.join(directory, 'niw_Mean*'))[0]
        coregistration_path = glob.glob(os.path.join(directory, 'aff*.txt'))[0]

        coregistration = np.loadtxt(coregistration_path, delimiter=' ')
        coregistration = np.linalg.inv(coregistration)

        subject = torchio.Subject(
            t1=torchio.Image(img_path, torchio.INTENSITY, coregistration=coregistration),
            label=torchio.Image(mask_path, torchio.LABEL),
            #ref=torchio.Image(ref_path, torchio.INTENSITY)
            # coregistration=coregistration,
        )
        print('adding img {} \n mask {}\n'.format(img_path,mask_path))
        subject_list.append(subject)

    transforms = [
        # Resample(1),
        RescaleIntensity((0, 1), (0, 99), masking_method=masking_method),
    ]

    if resampling == 'mni':
        # resampling_transform = ResampleWithFoV(
        #     target=nib.load(ref_path), image_interpolation=Interpolation.BSPLINE, coregistration_key='coregistration'
        # )
        resampling_transform = Resample(
            target='ref', image_interpolation=Interpolation.BSPLINE, coregistration='coregistration'
        )
        transforms.insert(0, resampling_transform)
    elif resampling == 'mm':
        # resampling_transform = ResampleWithFoV(target=nib.load(ref_path), image_interpolation=Interpolation.BSPLINE)
        resampling_transform = Resample(target=ref_path, image_interpolation=Interpolation.BSPLINE)
        transforms.insert(0, resampling_transform)

    transform = Compose(transforms)

    return torchio.ImagesDataset(subject_list, transform=transform)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ns', '--nb_subjects', type=int, default=5, help='Number of subjects in the dataset')
    parser.add_argument('-rs', '--resampling_strategy', type=str, choices=[None, 'mni', 'mm'], default=None,
                        help='Resampling strategy applied to the data among None, "mni" and "mm"')
    parser.add_argument('-nd', '--nb_display', type=int, default=5, help='Number of subjects to display')
    args = parser.parse_args()

    data_path = '/home/romain.valabregue/datal/QCcnn/res/res_cat12seg_18999.csv'
    data_root = '/network/lustre/iss01'
    ref_path = Path('/home/romain.valabregue/datal/HCPdata/suj_100307/T1w_1mm.nii.gz')

    dataset = generate_dataset(data_path, data_root, nb_subjects=args.nb_subjects,
                               resampling=args.resampling_strategy)
    int_plot = PlotDataset(dataset, subject_idx=args.nb_display, update_all_on_scroll=True)

    plt.show()
