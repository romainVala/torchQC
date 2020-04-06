#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Aug  29  09:11:47 2019

@author: romain
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch
from nilearn import plotting
from torchio import Image, ImagesDataset, INTENSITY, LABEL, Subject
from utils_file import get_parent_path, gfile, gdir
from doit_train import get_motion_transform
from torchvision.transforms import Compose
from torchio.transforms import RandomMotionFromTimeCourse

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
    parser.add_option("--motion_type", action="store", dest="motion_type", default='motion1',
                                help=" chose type of deformation  motion1 | elastic1_and_motion1  ")
    parser.add_option("--keep_all ", action="store_true", dest="keep_all", default=False,
                                help="if not specifie it will remove the imag_orig p1 and p2 from sample ")

    (options, args) = parser.parse_args()

    fin, seed, res_dir = options.image_in, np.int(options.seed), options.res_dir
    index, nb_sample = np.int(options.index_num),  np.int(options.nb_sample)
    plot_volume, keep_all = options.plot_volume, options.keep_all
    motion_type = options.motion_type

    import os

    resdir_mvt = res_dir + '/mvt_param/'
    resdir_fig = res_dir + '/fig/'
    try : #on cluster, all job are doing the mkdir at the same time ...
        if not os.path.isdir(resdir_mvt): os.mkdir(resdir_mvt)
        if not os.path.isdir(resdir_fig): os.mkdir(resdir_fig)
    except:
        pass

    transfo = get_motion_transform(type=motion_type)

    torch.manual_seed(seed)
    np.random.seed(seed)

    dir_img = get_parent_path([fin])[0]
    fm = gfile(dir_img, '^mask', {"items":1})
    fp1 = gfile(dir_img,'^p1', {"items":1})
    fp2 = gfile(dir_img,'^p2', {"items":1})
    if len(fm)==0: #may be in cat12 subdir (like for HCP)
        fm = gfile(dir_img, '^brain_T1', {"items": 1})
        #dir_cat = gdir(dir_img,'cat12')
        #fm = gfile(dir_cat, '^mask_brain', {"items": 1})
        #fp1 = gfile(dir_cat, '^p1', {"items": 1})
        #fp2 = gfile(dir_cat, '^p2', {"items": 1})

    one_suj = [ Image('image', fin, INTENSITY),
                Image('brain', fm[0], LABEL), ]
    if len(fp1)==1:
        one_suj.append(Image('p1', fp1[0], LABEL))
    if len(fp2) == 1:
        one_suj.append(Image('p2', fp2[0], LABEL))

    subject = [ Subject(one_suj[0]) for i in range(0,nb_sample) ]
    #subject = [ one_suj for i in range(0,nb_sample) ]
    print('input list is duplicated {} '.format(len(subject)))
    #subject = Subject(subject)
    dataset = ImagesDataset(subject, transform=transfo)

    for i in range(0, nb_sample):

        sample = dataset[i]  #in n time sample[0] it is cumulativ

        image_dict = sample['image']
        volume_path = image_dict['path']
        dd = volume_path.split('/')
        volume_name = dd[len(dd)-2] + '_' + image_dict['stem']
        #nb_saved = image_dict['index'] #

        fname = resdir_mvt + 'ssim_{}_sample{:05d}_suj_{}_mvt.csv'.format(image_dict['metrics']['ssim'],
                                                    index, volume_name)

        t = dataset.get_transform()
        if isinstance(t, Compose):
            tt = t.transforms
            for ttt in tt:
                if isinstance(ttt, RandomMotionFromTimeCourse):
                    t = ttt
                    break

        fitpars = t.fitpars
        np.savetxt(fname , fitpars, delimiter=',')

        sample['mvt_csv'] = fname
        fname_sample = res_dir + '/sample{:05d}'.format(index)

        if keep_all is False:
            if 'image_orig' in sample: sample.pop('image_orig')
            if 'p1' in sample: sample.pop('p1')
            if 'p2' in sample: sample.pop('p2')
            if 'brain' in sample: sample.pop('brain')

        torch.save(sample, fname_sample + '_sample.pt')


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
