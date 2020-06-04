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

from utils_file import get_parent_path, get_log_file
from utils_cmd import get_cmd_select_data_option, get_dataset_from_option, get_tranformation_list

formatter = logging.Formatter('%(asctime)-2s: %(levelname)-2s : %(message)s')

console = logging.StreamHandler()
console.setFormatter(formatter)
log = logging.getLogger('test_model')

if __name__ == '__main__':

    #get option for dataset selection
    parser = get_cmd_select_data_option()

    #option for model evaluation
    parser.add_option("-n", "--out_name", action="store", dest="out_name", default='res_val',
                                help="name to be append to the results ")
    parser.add_option("--val_number", action="store", dest="val_number", default='-1', type="int",
                                help="number to be prepend to out name default 1 ")
    parser.add_option("-w", "--saved_model", action="store", dest="saved_model", default='',
                                help="full path of the model's weights file ")
    parser.add_option("--use_gpu", action="store", dest="use_gpu", default=0, type="int",
                                help="0 means no gpu 1 to 4 means gpu device (0) ")
    parser.add_option("--validation_dropout", action="store_true", dest="validation_dropout", default=False,
                                help="if specifie it will perform validation with dropout enable ")
    parser.add_option("--transfo_list", action="store", dest="transfo_list", default=0, type="int",
                                help="integer to set a predefined transfo list default 0 means no list ")


    (options, args) = parser.parse_args()

    log = get_log_file()

    name, val_number = options.out_name, options.val_number
    saved_model = options.saved_model

    cuda = True if options.use_gpu > 0 else False


    if val_number<0:
        out_name = name
        subdir = None #'eval_rrr__{}_{}'.format(name)

    else:
        out_name = 'eval_num_{:04d}'.format(val_number)
        subdir = 'eval_{}_{}'.format(name, get_parent_path(saved_model)[1][:-3])

    doit, name_suffix, target = get_dataset_from_option(options)
    out_name += name_suffix

    doit.set_model_from_file(saved_model, cuda=cuda)

    if options.validation_dropout:
        doit.validation_droupout = True

    if options.transfo_list > 0:
        tlist, tname = get_tranformation_list(options.transfo_list)
        tname_all = [out_name + tt for tt in tname]
        doit.eval_multiple_transform(999, 99, basename=out_name, subdir=subdir, target=target,
                                     transform_list=tlist, transform_list_name=tname_all)

    else:
        doit.eval_regress_motion(999, 99, basename=out_name, subdir=subdir, target=target)
