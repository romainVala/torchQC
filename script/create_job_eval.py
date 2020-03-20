#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug  29  09:11:47 2019

@author: romain
"""

import nibabel as nb
import pandas as pd
import sys, os

sys.path.extend(['/network/lustre/iss01/cenir/software/irm/toolbox_python/romain/cnnQC_pytorch'])

import create_jobs
from utils_file import gfile, gdir, get_parent_path


pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1, 'display.width', 400)

prefix = "/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/"
weights1 = prefix + "NN_saved_pytorch/torcho2_msbrain098_equal_BN05_b4_BCEWithLogitsLoss_Adam/quadriview_ep21.pt"
weights1 = prefix + "NN_saved_pytorch/torcho2_msbrain098_equal_BN05_b4_BCEWithLogitsLoss_SDG/quadriview_ep29.pt"
weights1 = prefix + "NN_saved_pytorch/torcho2_full_098_equal_BN05_b4_BCEWithLogitsLoss_SDG/quadriview_ep10.pt"
weights1 = prefix + "NN_saved_pytorch/modelV2_msbrain_098_equal_BN05_b1_BCEWithLogitsLoss_SDG/quadriview_ep10.pt"
weights1 = prefix + "NN_saved_pytorch/modelV2_msbrain_098_equal_BN05_b4_BCEWithLogitsLoss_SDG/quadriview_ep10.pt"
weights1 = prefix + "NN_saved_pytorch/modelV2_one256_msbrain_098_equal_BN0_b4_BCEWithLogitsLoss_SDG/quadriview_ep10.pt"
weights1 = prefix + "modelV2_last128_msbrain_098_equal_BN0_b4_BCEWithLogitsLoss_SDG/quadriview_ep10.pt"

name = "cati_modelV2_last128_msbrain_098_equal_BN05_b4_BCEWithLogitsLoss_SDG"
resdir = prefix + "predict_torch/" + name + '/'

py_options = ' --BN_momentum 0.5  --do_reslice --apply_mask --model_type=2'  # --use_gpu 0 '

# for CATI
tab = pd.read_csv(prefix + "CATI_datasets/all_cati.csv", index_col=0)
clip_val = tab.meanW.values + 3 * tab.stdW.values

dcat = gdir(tab.cenir_QC_path, 'cat12')
# dspm = gdir(tab.cenir_QC_path,'spm' )

fms = gfile(dcat, '^ms.*ni', opts={"items": 1})
fmask = gfile(dcat, '^mask_brain.*gz', opts={"items": 1})
# fms = gfile(dspm,'^ms.*ni',opts={"items":1})
faff = gfile(dcat, '^aff.*txt', opts={"items": 1})
fref = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/HCPdata/suj_100307/T1w_1mm.nii.gz'

# for ABIDE
dcat = gdir('/network/lustre/dtlake01/opendata/data/ABIDE/cat12', ['^su', 'anat'])
# for ds30
dcat = gdir('/network/lustre/dtlake01/opendata/data/ds000030/cat12', ['^su', 'anat'])

# for validation
tab = pd.read_csv(prefix + "Motion_brain_ms_val_hcp200.csv", index_col=0)
fms = tab.img_file.values
fmask = fms
faff = fms

fin = fms;

sujid = []
for ff in fin:
    dd = ff.split('/')
    nn = len(dd)
    sujid.append(dd[nn - 5] + '+' + dd[nn - 4] + '+' + dd[nn - 3])  # for CATI
    # sujid.append(dd[nn - 3] + '+' + dd[nn - 2] + '+' + dd[nn - 1 ])  #for simulation dataset

scriptsDir = '/network/lustre/iss01/cenir/software/irm/toolbox_python/romain/cnnQC_pytorch'

job_id = name
params = dict()
params['output_directory'] = prefix + '/jobs/' + job_id
params['scripts_to_copy'] = scriptsDir + '/*.py'
params['output_results'] = resdir  # just to do the mkdir -p

cmd_init = '\n'.join(["python " + scriptsDir + "/test_model.py " + py_options + " \\",
                      ' -o ' + resdir + ' \\',
                      ' -w ' + weights1 + ' \\',
                      ' --fref ' + fref + ' \\'])
jobs = []

for ii, ff in enumerate(fin):
    job = '\n'.join([cmd_init,
                     ' -i ' + ff + ' \\',
                     ' --fmask ' + fmask[ii] + ' \\',
                     ' --faff ' + faff[ii] + ' \\',
                     ' --clip 1.3 '
                     #                     ' --clip %f'%(clip_val[ii]) + ' \\',
                     ' -n ' + sujid[ii], ' '])
    jobs.append(job)

params['jobs'] = jobs
params['job_name'] = job_id
params['cluster_queue'] = 'bigmem,normal'
params['cpus_per_task'] = 1
params['mem'] = 8096
params['walltime'] = '2:00:00'
params['job_pack'] = 2

create_jobs.create_jobs(params)
