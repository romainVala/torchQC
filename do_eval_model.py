#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Aug  29  09:11:47 2019

@author: romain
"""

import nibabel as nb
import numpy as np
import pandas as pd
import sys, os, logging

from utils_file import get_parent_path
from doit_train import do_training, get_motion_transform, get_train_and_val_csv, get_cache_dir
from torchio.transforms import CropOrPad, RandomAffine, RescaleIntensity, ApplyMask, RandomBiasField

formatter = logging.Formatter('%(asctime)-2s: %(levelname)-2s : %(message)s')

console = logging.StreamHandler()
console.setFormatter(formatter)
log = logging.getLogger('test_model')

def get_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

if __name__ == '__main__':

    from optparse import OptionParser
    usage= "usage: %prog [options] run a model on a file "

    # Parse input arguments
    parser=OptionParser(usage=usage)
    #parser.add_option("-h", "--help", action="help")
    #parser.add_option("-i", "--image_in", action="store", dest="image_in", default='',
    #                            help="full path to the image to test ")
    parser.add_option("-i", "--image_in", action="callback", dest="image_in", default='', callback=get_comma_separated_args,
                                type='string', help="full path to the image to test, list separate path by , ")
    parser.add_option("--sample_dir", action="store", dest="sample_dir", default='',
                                type='string', help="instead of -i specify dir of saved sample ")
    parser.add_option("-n", "--out_name", action="store", dest="out_name", default='res_val',
                                help="name to be append to the results ")
    parser.add_option("--val_number", action="store", dest="val_number", default='-1', type="int",
                                help="number to be prepend to out name default 1 ")
    parser.add_option("-w", "--saved_model", action="store", dest="saved_model", default='',
                                help="full path of the model's weights file ")
    parser.add_option("--use_gpu", action="store", dest="use_gpu", default=0, type="int",
                                help="0 means no gpu 1 to 4 means gpu device (0) ")
    parser.add_option("--add_cut_mask", action="store_true", dest="add_cut_mask", default=False,
                                help="if specifie it will adda cut mask (brain) transformation default False ")
    parser.add_option("--add_affine_zoom", action="store", dest="add_affine_zoom", default=0, type="float",
                      help=">0 means we add an extra affine transform with zoom value and if define rotations default 0")
    parser.add_option("--add_affine_rot", action="store", dest="add_affine_rot", default=0, type="float",
                      help=">0 means we add an extra affine transform with rotation values and if define rotations default 0")
    parser.add_option("--add_rescal_Imax", action="store_true", dest="add_rescal_Imax", default=False,
                                help="if specifie it will add a rescale intensity transformation default False ")
    parser.add_option("--add_mask_brain", action="store_true", dest="add_mask_brain", default=False,
                                help="if specifie it will add a apply_mask (name brain) transformation default False ")
    parser.add_option("--add_elastic1", action="store_true", dest="add_elastic1", default=False,
                                help="if specifie it will add a elastic1 transformation default False ")
    parser.add_option("--add_bias", action="store_true", dest="add_bias", default=False,
                                help="if specifie it will add a bias transformation default False ")

    (options, args) = parser.parse_args()

    log = logging.getLogger('do_eval')

    formatter = logging.Formatter("%(asctime)-2s: %(levelname)-2s : %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    log.addHandler(console)

    log.setLevel(logging.INFO)

    name, val_number = options.out_name, options.val_number
    saved_model = options.saved_model

    cuda = True if options.use_gpu > 0 else False

    #fixed param
    batch_size, num_workers = 2, 0

    fin = options.image_in
    dir_sample = options.sample_dir
    add_affine_zoom, add_affine_rot = options.add_affine_zoom, options.add_affine_rot


    doit = do_training('/tmp/', 'not_use', verbose=True)
    # adding transformation
    tc = []
    name_suffix = ''

    if options.add_cut_mask > 0:
        target_shape, mask_key = (182, 218, 182), 'brain'
        tc = [CropOrPad(target_shape=target_shape, mask_name=mask_key), ]
        name_suffix += '_tCrop_brain'

    if add_affine_rot>0 or add_affine_zoom >0:
        if add_affine_zoom==0: add_affine_zoom=1 #0 -> no affine so 1
        tc.append( RandomAffine(scales=(add_affine_zoom, add_affine_zoom), degrees=(add_affine_rot, add_affine_rot) ) )
        name_suffix += '_tAffineS{}R{}'.format(add_affine_zoom, add_affine_rot)

    if options.add_rescal_Imax:
        tc.append(RescaleIntensity(percentiles=(0, 99)))
        name_suffix += '_tRescale_0_99'

    # TODO should be before RescaleIntensity when done in train_regres_motion_full
    if options.add_mask_brain:
        tc.append(ApplyMask(masking_method='brain'))
        name_suffix += '_tMaskBrain'

    if options.add_elastic1:
        tc.append(get_motion_transform(type='elastic1'))
        name_suffix += '_tElastic1'

    if options.add_bias:
        tc.append(RandomBiasField())
        name_suffix += '_tElastic1'

    if len(name_suffix)==0:
        name_suffix = '_Raw'

    target = None
    if len(tc)==0: tc = None

    if len(dir_sample) > 0:
        print('loading from {}'.format(dir_sample))
        doit.set_data_loader(batch_size=batch_size, num_workers=num_workers, load_from_dir=dir_sample, transforms=tc)
        name += 'On_' + get_parent_path(dir_sample)[1]
        target='ssim' #suppose that if from sample, it should be simulation so set target
    else :
        print('working on ')
        for ff in fin:
            print(ff)

        doit.set_data_loader_from_file_list(fin, transforms=tc,
                                            batch_size=batch_size, num_workers=num_workers,
                                            mask_key=mask_key, mask_regex='^mask')

    if val_number<0:
        out_name = name
        subdir = None #'eval_rrr__{}_{}'.format(name)

    else:
        out_name = 'eval_num_{:04d}'.format(val_number)
        subdir = 'eval_{}_{}'.format(name, get_parent_path(saved_model)[1][:-3])

    out_name += name_suffix

    doit.set_model_from_file(saved_model, cuda=cuda)

    doit.eval_regress_motion(999, 99, basename=out_name, subdir=subdir, target=target)
