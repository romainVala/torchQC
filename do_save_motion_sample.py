#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Aug  29  09:11:47 2019

@author: romain
"""

import nibabel as nb
import numpy as np
import sys, os, logging

import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D as ov
import nibabel as nib
import torch
import torch.nn as nn
from doit_train import do_training
from torchio.transforms import RandomMotionFromTimeCourse
#from nibabel.viewers import OrthoSlicer3D as ov
import torch, os
from nilearn import plotting
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL

if __name__ == '__main__':
    from optparse import OptionParser
    usage= "usage: %prog [options] run a model on a file "

    # Parse input arguments
    parser=OptionParser(usage=usage)
    #parser.add_option("-h", "--help", action="help")
    parser.add_option("-i", "--image_in", action="store", dest="image_in", default='',
                                help="full path to the image to add motion to ")
    parser.add_option("-s", "--seed", action="store", dest="seed", default='1',
                                help="random seed ")
    parser.add_option("-r", "--res_dir", action="store", dest="res_dir", default='/tmp/',
                                help="result dir ")
    parser.add_option("-n", "--index_num", action="store", dest="index_num", default=1,
                                help="num given to sample saved file")
    parser.add_option("--nb_sample", action="store", dest="nb_sample", default=1,
                                help="number of sample to generate")
    parser.add_option("--plot_volume", action="store_true", dest="plot_volume", default=False,
                                help="if spefifyed a 3 slice png of the transform volume wil be created ")

    (options, args) = parser.parse_args()

    fin, seed, res_dir = options.image_in, np.int(options.seed), options.res_dir
    index, nb_sample = np.int(options.index_num),  np.int(options.nb_sample)
    plot_volume = options.plot_volume

    import os

    resdir_mvt = res_dir + '/mvt_param/'
    resdir_fig = res_dir + '/fig/'
    if not os.path.isdir(resdir_mvt): os.mkdir(resdir_mvt)
    if not os.path.isdir(resdir_fig): os.mkdir(resdir_fig)

    dico_params = {"maxDisp": (1, 6), "maxRot": (1, 6), "noiseBasePars": (5, 20, 0.8),
                   "swallowFrequency": (2, 6, 0.5), "swallowMagnitude": (3, 6),
                   "suddenFrequency": (2, 6, 0.5), "suddenMagnitude": (3, 6),
                   "verbose": False, "keep_original": True, "proba_to_augment": 1,
                   "preserve_center_pct": 0.1, "keep_original": True, "compare_to_original": True,
                   "oversampling_pct": 0, "correct_motion": True}

    transforms = RandomMotionFromTimeCourse(**dico_params)


    torch.manual_seed(seed)
    np.random.seed(seed)
    subject = [[ Image('image', fin, INTENSITY),]]

    dataset = ImagesDataset(subject, transform=transforms)

    for i in range(0, nb_sample):

        sample = dataset[0]
        fname = res_dir + '/sample{:05d}'.format(index)
        if 'image_orig' in sample: sample.pop('image_orig')
        torch.save(sample, fname + '_sample.pt')

        image_dict = sample['image']
        volume_path = image_dict['path']
        dd = volume_path.split('/')
        volume_name = dd[len(dd)-2] + '_' + image_dict['stem']
        nb_saved = image_dict['index']

        fname = resdir_mvt + 'ssim_{}_sample{:05d}_suj_{}'.format(image_dict['metrics']['ssim'],
                                                    index, volume_name)
        t=dataset.get_transform()
        fitpars = t.fitpars
        np.savetxt(fname + '_mvt.csv', fitpars, delimiter=',')

        if plot_volume:
            plt.ioff()
            fig = plt.figure()
            plt.plot(fitpars.T)
            plt.savefig(fname + '_mvt.png')
            plt.close(fig)

            resdir_fig = res_dir + '/fig/'

            image = sample['image']['data'][0].numpy()
            affine = sample['image']['affine']
            nii = nib.Nifti1Image(image, affine)

            fname = resdir_fig + 'ssim_{}_N{:04d}_suj_{}'.format(image_dict['metrics']['ssim'],
                                                        index, volume_name)

            di = plotting.plot_anat(nii, output_file=fname+'_fig.png',annotate=False, draw_cross = False)

        index += 1
