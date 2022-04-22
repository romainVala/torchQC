""" End-to-end segmentation pipeline """
import logging
import argparse
import os
import resource
import matplotlib.pyplot as plt
from pathlib import Path
from segmentation.utils import instantiate_logger
from segmentation.config import Config, parse_grid_search_file, \
    parse_create_jobs_file
from script.create_jobs import create_jobs


def handle_results_dir(folder, ref_file):
    # Replace relative path if needed
    if Path(folder).parent.anchor == '':
        folder = os.path.join(os.path.dirname(ref_file), folder)

    # Create dir
    if not os.path.isdir(folder):
        os.makedirs(folder)

    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str,
                        help='Path to main configuration file')
    parser.add_argument('-r', '--results_dir', type=str, default='result',
                        help='Path to results directory if it does not start '
                             'with the config file dir is prepend')
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help='possible mode :  train, val, infer, visualization ')
    parser.add_argument('-e', '--extra_file', type=str,
                        help='Extra configuration file')
    parser.add_argument('-d', '--debug', type=int, default=0,
                        help='Debug option, value different from 0 means that '
                        'debug messages will be printed in the console')
    parser.add_argument('-s', '--safe_mode', type=bool, default=False,
                        help='Whether to ask confirmation or not before'
                        'overwritting a configuration file.')
    parser.add_argument('-viz', '--visualization', type=int, default=0,
                        choices=[0, 1, 2, 3, 4, 5],
                        help='Visualization mode \n'
                             '\t0: whole images are shown, \n'
                             '\t1: whole images with labels are shown, \n '
                             '\t2: patches are shown, \n '
                             '\t3: patches with labels are shown, \n'
                             '\t4: fuzzy false positives and false negatives '
                             'maps between prediction and ground truth '
                             'are shown on a given sample, \n'
                             '\t5: prediction and ground truth are shown '
                             'on a given sample.')
    parser.add_argument('-cj', '--create_jobs_file', type=str, default=None,
                        help='Create job file, if None the script is run,'
                             'otherwise the configuration files are created'
                             'and the command line to run the script is '
                             'saved according to this configuration file')
    parser.add_argument('-gs', '--grid_search_file', type=str,
                        help='Grid search file')
    parser.add_argument('-ms', '--max_subjects_per_job', type=int, default=None,
                        help='Maximum number of subjects per job.')
    args = parser.parse_args()

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    #resource.setrlimit(resource.RLIMIT_NOFILE, (2048*8, rlimit[1]))

    results_dir = args.results_dir
    file = args.file
    extra_file = args.extra_file
    gs_file = args.grid_search_file
    create_jobs_file = args.create_jobs_file

    jobs, jobs_struct = [], {}

    current_dir = os.getcwd()
    if os.path.dirname(file) == '':
        file = os.path.join(current_dir, file)

    if extra_file is not None and os.path.dirname(extra_file) == '':
        extra_file = os.path.join(os.path.dirname(file), extra_file)

    if create_jobs_file is not None and os.path.dirname(create_jobs_file) == '':
        create_jobs_file = os.path.join(os.path.dirname(file), create_jobs_file)
        jobs_struct = parse_create_jobs_file(create_jobs_file)

    if gs_file is not None:
        if os.path.dirname(gs_file) == '':
            gs_file = os.path.join(os.path.dirname(file), gs_file)

        gs_struct = parse_grid_search_file(gs_file)

        for results_dir, values in zip(
                gs_struct['results_dirs'], gs_struct['values']
        ):
            results_dir = handle_results_dir(results_dir, file)
            logger = instantiate_logger(
                f'info_{results_dir}', logging.INFO, results_dir + '/info.txt')
            debug_logger = instantiate_logger(
                f'debug_{results_dir}', logging.DEBUG,
                results_dir + '/debug.txt', args.debug != 0)
            config = Config(file, results_dir, logger, debug_logger, args.mode,
                            args.visualization, extra_file, args.safe_mode,
                            args.create_jobs_file, gs_struct['keys'], values,
                            args.max_subjects_per_job)
            config.init()
            result = config.run()
            if isinstance(result, list):
                jobs += result

    else:
        if extra_file is not None:
            estruct = Config.read_json(extra_file)
            if 'results_dir' in estruct:
                results_dir = estruct["results_dir"]

        results_dir = handle_results_dir(results_dir, file)

        logger = instantiate_logger(
            'info', logging.INFO, results_dir + '/info.txt')
        debug_logger = instantiate_logger(
            'debug', logging.DEBUG, results_dir + '/debug.txt', args.debug != 0)

        config = Config(file, results_dir, logger, debug_logger, args.mode,
                        args.visualization, extra_file, args.safe_mode,
                        args.create_jobs_file,
                        max_subjects_per_job=args.max_subjects_per_job)
        config.init()
        result = config.run()
        if isinstance(result, str):
            jobs += result

    if create_jobs_file is not None:
        jobs_struct['jobs'] = jobs
        create_jobs(jobs_struct)

    plt.show()
