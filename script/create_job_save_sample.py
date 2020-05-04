#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to generate jobs
"""
import pandas as pd
from script.create_jobs import create_jobs
from doit_train import get_train_and_val_csv

# parameters

prefix = "/network/lustre/dtlake01/opendata/data/ds000030/rrr/CNN_cache_new/"

name_list = [ 'mvt_train_hcp400_ms', 'mvt_train_hcp400_brain_ms', 'mvt_train_hcp400_T1',
              '', 'mvt_val_hcp200_brain_ms', 'mvt_val_hcp200_T1']
name_list = [ 'mvt_train_cati_T1', 'mvt_train_cati_ms', 'mvt_train_cati_brain',
              'mvt_val_cati_T1', 'mvt_val_cati_ms', 'mvt_val_cati_brain',]
name_list = [ 'ela1_train_cati_T1', 'ela1_train_cati_ms', 'ela1_train_cati_brain',
              'ela1_val_cati_T1', 'ela1_val_cati_ms', 'ela1_val_cati_brain',]
name_list =[  'ela1_train_hcp400_brain_ms', 'ela1_train_hcp400_T1',
              'ela1_val_hcp200_brain_ms', 'ela1_val_hcp200_T1']
# name_list = [ 'ela1_train_cati_T1', 'ela1_train_cati_ms',
#               'ela1_val_cati_T1', 'ela1_val_cati_ms',]
name_list = ['mvt_train_hcp400_ms', 'mvt_val_hcp200_ms', 'mvt_val_hcp200_T1', 'mvt_val_cati_T1', 'mvt_val_cati_ms',
             'mvt_train_cati_T1', 'mvt_train_cati_ms']
nb_motions_list = [20, 5, 5, 10, 10, 5, 5]

nb_motions_list = [20, 20, 20, 10, 10, 10] #[5, 5, 5]
nb_motions_list = [50, 50, 50, 5, 5, 5]
#nb_motions_list = [50, 50, 50, 20, 20, 20]
#nb_motions_list = [200]
nb_motions_list = [20, 5, 5, 10, 10, 5,5]


do_plotting = False

fin_list_train, fin_list_val = get_train_and_val_csv(name_list) #

fin_choose = []
for ii, nn in enumerate(name_list):
    if 'train' in nn: fin_choose.append(fin_list_train[ii])
    elif 'val' in nn: fin_choose.append(fin_list_val[ii])

fin_list =  [ pd.read_csv(ff).filename for ff in fin_choose]


for name, fin, nb_motions in zip(name_list, fin_list, nb_motions_list):
    resdir = prefix  + name + '/'

    scriptsDir = '/network/lustre/iss01/cenir/software/irm/toolbox_python/romain/torchQC'

    py_options = '--motion_type elastic1_and_motion1 --nb_sample={} --res_dir={}'.format(nb_motions, resdir)

    job_id = name #+'_redo'
    params = dict()
    params['output_directory'] = prefix + '/jobs/' + job_id
    params['scripts_to_copy'] = scriptsDir #+ '/*.py'
    params['output_results'] = resdir  # just to do the mkdir -p

    cmd_init = '\n'.join(["#source /network/lustre/iss01/cenir/software/irm/bin/python_path3.6",
                          "#source activate pytorch1.2",
                          "#module load fftw/3.3.8-g7jugkl ",
                          "python " + scriptsDir + "/do_save_motion_sample.py \\",
                          py_options + " \\"] )
    jobs = []

    if 'fin_redo' not in locals():
        fin_redo = fin
    print('fin {} redo {}'.format(len(fin),len(fin_redo)))
    for ii, ff in enumerate(fin):
        #if ff in fin_redo:
        index = ii*nb_motions
        job = '\n'.join([cmd_init,
                         ' -i ' + ff + ' \\',
                         ' --seed={} '.format(ii) + ' \\',
                         ' --index_num={} '.format(index) ])
        if do_plotting:
            job += ' --plot_volume '

        jobs.append(job)

    params['jobs'] = jobs
    params['job_name'] = job_id
    params['cluster_queue'] = 'bigmem,normal'
    params['cpus_per_task'] = 1
    params['mem'] = 6000
    params['walltime'] = '12:00:00'
    params['job_pack'] = 1

    create_jobs(params)

