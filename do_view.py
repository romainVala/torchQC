#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Aug  29  09:11:47 2019

@author: romain
"""

import sys, os, logging
import numpy as np
from utils_file import get_parent_path, get_log_file
from utils_cmd import get_cmd_select_data_option, get_dataset_from_option
from plot_dataset import PlotDataset
import matplotlib.pyplot as plt

formatter = logging.Formatter('%(asctime)-2s: %(levelname)-2s : %(message)s')

console = logging.StreamHandler()
console.setFormatter(formatter)
log = logging.getLogger('test_model')

if __name__ == '__main__':

    #get option for dataset selection
    parser = get_cmd_select_data_option()

    #option for visualisation
    #parser.add_option('-ns', '--nb_subjects', type=int, default=10,
    #                  help='Number of subjects in the dataset default 10')
    # this should be done in get_cmd_select_data_option
    #parser.add_option('-rs', '--resampling_strategy', type=str, choices=[None, 'mni', 'mm'], default=None,
    #                  help='Resampling strategy applied to the data among None, "mni" and "mm"')

    parser.add_option('-n', '--nb_display', type=int, default=5, dest="nb_display",
                      help='Number of subjects to display')

    parser.add_option("-s", "--seed", action="store", dest="seed", default=-1, type="int",
                      help="seed numer, to select the same subject default is -1 which means do not set seed")

    (options, args) = parser.parse_args()

    if options.seed >= 0:
        np.random.seed(options.seed)

    log = get_log_file()

    doit, name_suffix, target = get_dataset_from_option(options)
    dataset = doit.train_dataset
    int_plot = PlotDataset(dataset, subject_idx=options.nb_display, update_all_on_scroll=True)

    plt.show()
