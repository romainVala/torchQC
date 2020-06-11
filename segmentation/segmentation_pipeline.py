""" End-to-end segmentation pipeline """
import logging
import argparse
import os
from segmentation.utils import instantiate_logger
from segmentation.config import Config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='Path to main configuration file')
    parser.add_argument('-r', '--results_dir', type=str, help='Path to results directory')
    parser.add_argument('-m', '--mode', type=str, default='train', help='Training, visualization or inference mode')
    parser.add_argument('-d', '--debug', type=int, default=0, help='Debug option, value different from 0 means that '
                                                                   'debug messages will be printed in the console')
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    logger = instantiate_logger('info', logging.INFO, args.results_dir + '/info.txt')
    debug_logger = instantiate_logger('debug', logging.DEBUG, args.results_dir + '/debug.txt', args.debug != 0)

    config = Config(args.file, args.results_dir, logger, debug_logger, args.mode)
    config.run()
