""" End-to-end segmentation pipeline """
import logging
import argparse
import os
import resource
from pathlib import Path
from segmentation.utils import instantiate_logger
from segmentation.config import Config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='Path to main configuration file')
    parser.add_argument('-r', '--results_dir', type=str,default='result',
                        help='Path to results directory if it does not start with the config file dir is prepend')
    parser.add_argument('-m', '--mode', type=str, default='train', help='Training, visualization or inference mode')
    parser.add_argument('-e', '--extra_file', type=str, help='Extra configuration file')
    parser.add_argument('-d', '--debug', type=int, default=0, help='Debug option, value different from 0 means that '
                                                                   'debug messages will be printed in the console')
    parser.add_argument('-viz', '--visualization', type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                        help='Visualization mode \n'
                             '\t0: whole images are shown, \n'
                             '\t1: whole images with labels are shown, \n '
                             '\t2: patches are shown, \n '
                             '\t3: patches with labels are shown, \n'
                             '\t4: fuzzy false positives and false negatives maps between prediction and ground truth '
                             'are shown on a given sample, \n'
                             '\t5: prediction and ground truth are shown on a given sample.')
    args = parser.parse_args()

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048*8, rlimit[1]))

    results_dir = args.results_dir
    file = args.file
    extra_file = args.extra_file

    if extra_file is not None and os.path.dirname(extra_file) == '':
        extra_file = os.path.join(os.path.dirname(file), extra_file)

    if extra_file is not None:
        estruct = Config.read_json(extra_file)
        if 'results_dir' in estruct:
            results_dir = estruct["results_dir"]

    # Replace relative path if needed
    if Path(results_dir).parent.anchor == '':
        results_dir = os.path.join(os.path.dirname(file), results_dir)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)


    logger = instantiate_logger('info', logging.INFO, results_dir + '/info.txt')
    debug_logger = instantiate_logger('debug', logging.DEBUG, results_dir + '/debug.txt', args.debug != 0)

    config = Config(file, results_dir, logger, debug_logger, args.mode, args.visualization, extra_file)
    config.run()
