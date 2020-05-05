#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to generate jobs
"""
import pandas as pd
from script.create_jobs import create_jobs
from utils_file import get_parent_path, gfile

# parameters
root_dir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion/'
prefix = "/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/job/job_eval/"

model = root_dir + 'RegMotNew_ela1_train200_hcp400_ms_B4_nw0_Size182_ConvN_C16_256_Lin40_50_D0_BN_Loss_L1_lr0.0001/'
saved_models = gfile(model, '_ep9_.*000.pt$')
saved_models = saved_models[0:10]

name_list_val = ['mvt_val_hcp200_ms', 'ela1_val_hcp200_ms']
name_list_val = ['ela1_val_hcp200_ms']

job_id = "eval_sample_ela1"

dir_cache = '/network/lustre/dtlake01/opendata/data/ds000030/rrr/CNN_cache_new/'

jobs = []
scriptsDir = '/network/lustre/iss01/cenir/software/irm/toolbox_python/romain/torchQC'

options_test = ['', '--add_affine_zoom 0.8', '--add_affine_zoom 1.2', '--add_affine_rot 10' ]

for data_name_val in name_list_val :
    dir_sample = '{}/{}/'.format(dir_cache, data_name_val)

    for saved_model in saved_models:
        for opt_test in options_test:

            py_options = '--use_gpu 0 --saved_model {} {} '.format(saved_model, opt_test)

            model_name = get_parent_path(saved_model)[1]

            cmd_init = '\n'.join(["#source /network/lustre/iss01/cenir/software/irm/bin/python_path3.6",
                                  "#source activate pytorch1.2",
                                  "python " + scriptsDir + "/do_eval_model.py \\",
                                  py_options + " \\"] )

            job = '\n'.join([cmd_init, ' --sample_dir ' + dir_sample ])
            jobs.append(job)


params = dict()
params['output_directory'] = prefix + '/jobs/' + job_id
params['scripts_to_copy'] = scriptsDir #+ '/*.py'

params['jobs'] = jobs
params['job_name'] = job_id
params['cluster_queue'] = 'bigmem,normal'
params['cpus_per_task'] = 1
params['mem'] = 4000
params['walltime'] = '12:00:00'
params['job_pack'] = 1

create_jobs(params)

