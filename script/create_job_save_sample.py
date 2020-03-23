#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to generate jobs
"""

import nibabel as nb
import pandas as pd
import sys, os
from script.create_jobs import create_jobs
from utils_file import gfile, gdir, get_parent_path
def get_cati_sample():
    fcsv='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/CATI_datasets/all_cati.csv';
    res = pd.read_csv(fcsv)

    ser_dir = res.cenir_QC_path[res.globalQualitative>3].values
    dcat = gdir(ser_dir, 'cat12')
    fT1 = gfile(dcat, '^s.*nii')
    fms = gfile(dcat, '^ms.*nii')
    fs_brain = gfile(dcat, '^brain_s.*nii')
    return fT1, fms, fs_brain

# parameters

name_list = [ 'motion_cati_T1', 'motion_cati_ms', 'motion_cati_brain_ms', 'motion_train_hcp400_ms', 'motion_train_hcp400_brain_ms', 'motion_train_hcp400_T1']
prefix = "/network/lustre/dtlake01/opendata/data/ds000030/rrr/CNN_cache/"
data_path = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/'
nb_motions_list = [10, 10, 10, 20, 20, 20]
do_plotting = False

fcsv =  [ data_path+ 'healthy_ms_train_hcp400.csv', data_path+ 'healthy_brain_ms_train_hcp400.csv', data_path+ 'Motion_T1_train_hcp400.csv']
fincsv = [ pd.read_csv(ff).filename for ff in fcsv]
#res=pd.read_csv(fcsv); fin=res.filename

#fT1, fms, fs_brain = get_cati_sample()
fout = list( get_cati_sample() )
fin_list = fout + fincsv

for name, fin, nb_motions in zip(name_list, fin_list, nb_motions_list):
    resdir = prefix  + name + '/'

    scriptsDir = '/network/lustre/iss01/cenir/software/irm/toolbox_python/romain/torchQC/'

    py_options = '--nb_sample={} --res_dir={}'.format(nb_motions, resdir)

    job_id = name
    params = dict()
    params['output_directory'] = prefix + '/jobs/' + job_id
    params['scripts_to_copy'] = scriptsDir #+ '/*.py'
    params['output_results'] = resdir  # just to do the mkdir -p


    cmd_init = '\n'.join(["python " + scriptsDir + "/do_save_motion_sample.py " +py_options + " \\"] )
    jobs = []


    for ii, ff in enumerate(fin):
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
    params['mem'] = 12096
    params['walltime'] = '12:00:00'
    params['job_pack'] = 1

    create_jobs(params)

