import matplotlib.pyplot as plt, pandas as pd, seaborn as sns
import torchio as tio, torch, time
from segmentation.config import Config
from segmentation.run_model import RunModel
from nibabel.viewers import OrthoSlicer3D as ov
import glob, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1, 'display.width', 400)
from util_affine import perform_motion_step_loop, product_dict, create_motion_job, select_data, corrupt_data, apply_motion
import nibabel as nib
from read_csv_results import ModelCSVResults
from types import SimpleNamespace
from kymatio import HarmonicScattering3D
from types import SimpleNamespace
from script.create_jobs import create_jobs

dircati = '/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cati_full/delivery_new/'
allfitpars_preproc = glob.glob(dircati+'/*/*/*/*/fitpars_preproc.txt')
allfitpars_raw = glob.glob(dircati+'/*/*/*/*/fitpars.txt')
fp_path = allfitpars_preproc[0]
fp_paths = allfitpars_preproc
res_name = 'CATI_fitpar_one_suj_noise001'
fjson = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/test1/main.json'
out_path = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/fit_parmCATI/'

split_length = 1
nb_param = len(fp_path)
nb_job = int(np.ceil(nb_param / split_length))


cmd_init = '\n'.join(['python -c "', "from util_affine import perform_one_motion "])
jobs = []
for fp_p in fp_paths:
    # ind_start = nj * split_length;
    # ind_end = np.min([(nj + 1) * split_length, nb_param])
    # print(f'{nj} {ind_start} {ind_end} ')
    # print(f'param = {params[ind_start:ind_end]}')

    cmd = '\n'.join([cmd_init,  # f'params = {params[ind_start:ind_end]}',
                     f'out_path = \'{out_path}\'',
                     f'json_file = \'{fjson}\'',
                     f'fp_path = \'{fp_p}\'',
                     '_ = perform_one_motion(fp_path,json_file, out_dir=out_path) "'])
    jobs.append(cmd)

job_params = dict()
job_params[
    'output_directory'] = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/' + f'{res_name}_job'
job_params['jobs'] = jobs
job_params['job_name'] = 'motion'
job_params['cluster_queue'] = 'bigmem,normal'
job_params['cpus_per_task'] = 4
job_params['mem'] = 8000
job_params['walltime'] = '12:00:00'
job_params['job_pack'] = 10

create_jobs(job_params)


def get_sujname_from_path(ff):
    name=[]; dn = os.path.dirname(ff)
    for k in range(3):
        name.append(os.path.basename(dn))
        dn = os.path.dirname(dn)
    return '_'.join(reversed(name))

df = pd.DataFrame()
mvt_name = ['tx','ty','tz','rx','ry','rz']
for idx, ff in enumerate(allfitpars_preproc):
    mydict = dict()
    mydict['sujname'] = get_sujname_from_path(ff)
    fitpars = np.loadtxt(ff)
    for nb_mvt in range(6):
        one_mvt = fitpars[nb_mvt,:]
        one_mvt = one_mvt - one_mvt[125]
        mydict[mvt_name[nb_mvt] + '_mean'] = np.mean(one_mvt)
        mydict[mvt_name[nb_mvt] + '_max'] = np.max(np.abs(one_mvt))
    mydict['nb_pts'] = len(one_mvt)

    df = df.append(mydict, ignore_index=True)


fjson = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/test1/main.json'

param = dict()
param['suj_contrast'] = 1
param['suj_noise'] = 0.01
param['suj_index'] = 0
param['suj_deform'] = 0

sdata, tmot, mr = select_data(fjson, param)
