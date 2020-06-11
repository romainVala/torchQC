""" End-to-end segmentation pipeline """
import logging
import argparse
import os
from segmentation.utils import instantiate_logger
from segmentation.config import Config
from utils_file import get_parent_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='Path to main configuration file')
    parser.add_argument('-r', '--results_dir', type=str, help='Path to results directory if it does not start with / '
                                                              'the config file dir is prepend')
    parser.add_argument('-m', '--mode', type=str, default='train', help='Training, visualization or inference mode')
    parser.add_argument('-d', '--debug', type=int, default=0, help='Debug option, value different from 0 means that '
                                                                   'debug messages will be printed in the console')
    args = parser.parse_args()

    resdir = args.results_dir
    conf_file = args.file
    if not resdir[0]=='/':
        resdir = get_parent_path(conf_file)[0] + '/' + resdir + '/'

    if not os.path.isdir(resdir):
        os.makedirs(resdir)

    logger = instantiate_logger('info', logging.INFO,resdir + '/info.txt')
    debug_logger = instantiate_logger('debug', logging.DEBUG, resdir + '/debug.txt', args.debug != 0)

    config = Config(conf_file, resdir, logger, debug_logger, args.mode)
    config.run()
