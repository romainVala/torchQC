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
from torchio.transforms import CropOrPad

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
    parser.add_option("-n", "--out_name", action="store", dest="out_name", default='',
                                help="name to be append to the results ")
    parser.add_option("--val_number", action="store", dest="val_number", default='1',
                                help="number to be prepend to out name default 1 ")
    parser.add_option("-w", "--saved_model", action="store", dest="saved_model", default='',
                                help="full path of the model's weights file ")
    parser.add_option("--use_gpu", action="store", dest="use_gpu", default=0, type="int",
                                help="0 means no gpu 1 to 4 means gpu device (0) ")

    (options, args) = parser.parse_args()

    log = logging.getLogger('do_eval')

    formatter = logging.Formatter("%(asctime)-2s: %(levelname)-2s : %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    log.addHandler(console)

    log.setLevel(logging.INFO)

    name, val_number = options.out_name, options.val_number
    saved_model = options.saved_model
    gpu = options.use_gpu
    #let's force absolute path if not (weights[0] == '/') : weights1 = prefix + weights
    cuda = True if gpu > 0 else False

    #fixed param
    batch_size, num_workers = 2, 0

    fin = options.image_in
    dir_sample = options.sample_dir


    target_shape, mask_key = (182, 218, 182), 'brain'
    tc = [CropOrPad(target_shape=target_shape, mask_name=mask_key), ]

    doit = do_training('/tmp/', 'not_use', verbose=True)

    if len(dir_sample) > 0:
        print('loading from {}'.format(dir_sample))
        doit.set_data_loader(batch_size=batch_size, num_workers=num_workers, load_from_dir=dir_sample, transforms=tc)
        name += '_' + get_parent_path(dir_sample)[1]
    else :
        print('working on ')
        for ff in fin:
            print(ff)

        doit.set_data_loader_from_file_list(fin, transforms=tc,
                                            batch_size=batch_size, num_workers=num_workers,
                                            mask_key=mask_key, mask_regex='^mask')

    out_name = 'eval_num_{:04d}'.format(int(val_number))
    subdir = 'eval_{}_{}'.format(name, get_parent_path(saved_model)[1][:-3])

    doit.set_model_from_file(saved_model, cuda=cuda)

    doit.eval_regress_motion(999, 99, basename=out_name, subdir=subdir, target=None)
