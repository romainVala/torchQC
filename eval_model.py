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

from utils_file import get_parent_path, gfile
from doit_train import do_training, get_motion_transform, get_train_and_val_csv, get_cache_dir
from torchio.transforms import CropOrPad

root_dir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion/'


if __name__ == '__main__':
    model = root_dir + 'RegMotNew_ela1_train200_hcp400_ms_B4_nw0_Size182_ConvN_C16_256_Lin40_50_D0_BN_Loss_L1_lr0.0001/'
    saved_models = gfile(model, '_ep9_.*000.pt$')

    name_list_val = ['mvt_val_hcp200_ms', 'ela1_val_hcp200_ms']
    dir_cache = '/network/lustre/dtlake01/opendata/data/ds000030/rrr/CNN_cache/'

    batch_size, num_workers = 4, 0
    cuda, verbose = True, True

    target_shape, mask_key = (182, 218, 182), 'brain'
    tc = None  # [CropOrPad(target_shape=target_shape, mask_name=mask_key), ]

    for data_name_val in name_list_val :
        dir_sample = '{}/{}/'.format(dir_cache, data_name_val)

        doit = do_training('/tmp/', 'not_use', verbose=True)

        doit.set_data_loader(batch_size=batch_size, num_workers=num_workers, load_from_dir=dir_sample, transforms=tc)
        name = '{}'.format(data_name_val)

        for saved_model in saved_models:
            out_name = 'res_valOn_{}'.format( data_name_val)
            subdir = None #'eval_rrr__{}_{}'.format(name)

            doit.set_model_from_file(saved_model, cuda=cuda)

            doit.eval_regress_motion(999, 99, basename=out_name, subdir=subdir, target='ssim')
