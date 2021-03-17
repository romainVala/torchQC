import matplotlib.pyplot as plt, pandas as pd, seaborn as sns
import torchio as tio, torch, time
from segmentation.config import Config
from segmentation.run_model import RunModel
from nibabel.viewers import OrthoSlicer3D as ov
import glob, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, numpy.linalg as npl
import scipy.linalg as scl, scipy.stats as ss
pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1, 'display.width', 400)
from util_affine import perform_one_motion, product_dict, create_motion_job, select_data, corrupt_data, apply_motion, spm_matrix, spm_imatrix
import nibabel as nib
from read_csv_results import ModelCSVResults
from types import SimpleNamespace
from kymatio import HarmonicScattering3D
from types import SimpleNamespace
from script.create_jobs import create_jobs
import glob
from torchio.transforms.augmentation.intensity.random_motion_from_time_course import _interpolate_space_timing, _tile_params_to_volume_dims

def change_root_path(f_path, root_path='/data/romain/PVsynth/motion_on_synth_data/delivery_new'):
    common_dir = os.path.basename(root_path)
    ss = f_path.split('/')
    for k,updir in enumerate(ss):
        if common_dir in updir:
            break
    snew = ss[k:]
    snew[0] = root_path
    return '/'.join(snew)
def get_sujname_from_path(ff):
    name = [];
    dn = os.path.dirname(ff)
    for k in range(3):
        name.append(os.path.basename(dn))
        dn = os.path.dirname(dn)
    return '_'.join(reversed(name))

dircati = '/data/romain/PVsynth/motion_on_synth_data/delivery_new'
dircati = '/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cati_full/delivery_new/'
allfitpars_preproc = glob.glob(dircati+'/*/*/*/*/fitpars_preproc.txt')
allfitpars_raw = glob.glob(dircati+'/*/*/*/*/fitpars.txt')
fp_paths = allfitpars_raw

res_name = 'CATI_fitpar_one_suj_noise001'
fjson = '/data/romain/PVsynth/motion_on_synth_data/test1/main.json'
fjson = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/test1/main.json'
out_path = '/data/romain/PVsynth//motion_on_synth_data/fit_parmCATI/'
out_path = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/fit_parmCATI_raw/'


### ###### run motion on all fitpar
split_length = 10
create_motion_job(fp_paths, split_length, fjson, out_path, res_name='fit_parmCATI_raw', type='one_motion')
##############"


#read from global csv (to get TR)
dfall = pd.read_csv(dircati+'/description_copie.csv')
dfall['Fitpars_path'] = dfall['Fitpars_path'].apply(change_root_path)
dfall['resdir'] = dfall['Fitpars_path'].apply(get_sujname_from_path); dfall['resdir'] = out_path+dfall['resdir']

allfitpars_raw = dfall['Fitpars_path']
afffitpars_preproc = [os.path.basename(p) + '/fitpars_preproc.txt' for p in allfitpars_raw]

#read results
fcsv = glob.glob(out_path+'/*/*csv')
df = [pd.read_csv(ff) for ff in fcsv]
df1 = pd.concat(df, ignore_index=True); dfall = df1
df1['fp'] = df1['fp'].apply(change_root_path)

df1['srot'] = abs(df1['shift_R1']) + abs(df1['shift_R2']) + abs(df1['shift_R3'])
df1['stra'] = abs(df1['shift_T1']) + abs(df1['shift_T2']) + abs(df1['shift_T3'])

df_coreg = df1[df1.flirt_coreg==1] ; df_coreg.index = range(len(df_coreg))
df_nocoreg = df1[df1.flirt_coreg==0]; df_nocoreg.index = range(len(df_nocoreg))
#plt.scatter(df_coreg.m_t1_L1_map,df_nocoreg.m_t1_L1_map); plt.plot([0,5],[0,5])
df_nocoreg.columns = ['noShift_' + k for k in  df_nocoreg.columns]
df_coreg = pd.concat([df_coreg, df_nocoreg], sort=True, axis=1); del(df_nocoreg)

#add max amplitude and select data
for ii, fp_path in enumerate(df_coreg.fp.values):
    fitpar = np.loadtxt(fp_path)
    amplitude_max = fitpar.max(axis=1) - fitpar.min(axis=1)
    for i in range(6):
        cname = f'amp_max{i}'
        df_coreg.loc[ii, cname] = amplitude_max[i]
    df_coreg.loc[ii, 'amp_max_max'] = amplitude_max.max()

df_coreg = df_coreg[ (df_coreg.amp_max_max>2) ];  df_coreg.index = range(len(df_coreg))

#get the data
param = dict();param['suj_contrast'] = 1;param['suj_noise'] = 0.01;param['suj_index'] = 0;param['suj_deform'] = 0;param['displacement_shift_strategy']=None
sdata, tmot, config_runner = select_data(fjson, param, to_canonical=False)
image = sdata.t1.data[0]
fi = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)

#compute FFT weights for image (here always the same)
w_coef = np.abs(fi)
w_coef_flat = w_coef.reshape(-1)

# w TF coef approximation at fitpar resolution (== assuming fitparinterp constant )
w_coef_short, w_coef_shaw = np.zeros_like(fitpar[0]), np.zeros_like(fitpar[0])
step_size = w_coef_flat.shape[0] / w_coef_short.shape[0]
for kk in range(w_coef_short.shape[0]):
    ind_start = int(kk * step_size)
    ind_end = ind_start + int(step_size)
    w_coef_short[kk] = np.sum(w_coef_flat[ind_start:ind_end])  # sum or mean is equivalent for the weighted mean

    # in shaw article, they sum the voxel in image domain (iFFT(im conv mask)) but 256 fft is too much time ...
    fft_mask = np.zeros_like(w_coef_flat).astype(complex)
    fft_mask[ind_start:ind_end] = w_coef_flat[ind_start:ind_end]
    fft_mask = fft_mask.reshape(fi.shape)
    ffimg = np.fft.ifftshift(np.fft.ifftn(fft_mask))
    w_coef_shaw[kk] = np.sum(np.abs(ffimg))

w_coef_short = w_coef_short / np.sum(w_coef_short)  # nomalize the weigths
w_coef_shaw = w_coef_shaw / np.sum(w_coef_shaw)  # nomalize the weigths
plt.figure(); plt.plot(w_coef_short); plt.plot(w_coef_shaw)


for ii, fp_path in enumerate(df_coreg.fp.values):
    fitpar = np.loadtxt(fp_path)

    fitparinter = _interpolate_space_timing(fitpar, 0.004, 2.3,[218, 182],0)
    fitparinter = _tile_params_to_volume_dims(fitparinter, list(image.shape))

    Aff_mean = np.zeros((4, 4))
    for nb_mvt in range(fitpar.shape[1]):
        trans_rot = np.hstack((fitpar[:, nb_mvt], np.array([1, 1, 1, 0, 0, 0])))
        Aff = spm_matrix(trans_rot)
        Aff_mean = Aff_mean + w_coef_short[nb_mvt] * scl.logm(Aff)
    Aff_mean = scl.expm(Aff_mean)
    wshift = spm_imatrix(Aff_mean, order=0)

    Aff_mean = np.zeros((4, 4))
    for nb_mvt in range(fitpar.shape[1]):
        trans_rot = np.hstack((fitpar[:, nb_mvt], np.array([1, 1, 1, 0, 0, 0])))
        Aff = spm_matrix(trans_rot)
        Aff_mean = Aff_mean + w_coef_shaw[nb_mvt] * scl.logm(Aff)
    Aff_mean = scl.expm(Aff_mean)
    wshift_shaw = spm_imatrix(Aff_mean, order=0)

    print(f'suj {ii}  working on {fp_path}')
    for i in range(0, 6):
        ffi = fitparinter[i].reshape(-1)
        rrr = np.sum(ffi * w_coef_flat) / np.sum(w_coef_flat)
        #already sifted    rcheck2 = df_coreg.loc[ii,f'm_wTF_Disp_{i}']
        #rcheck = df_nocoreg.loc[ii,f'm_wTF_Disp_{i}']
        #print(f'n={i} mean disp {i} = {rrr}  / {rcheck} after shift {rcheck2}')
        cname = f'before_coreg_wTF_Disp_{i}';        df_coreg.loc[ii, cname] = rrr

        rrr2 = np.sum(fitpar[i]*w_coef_short) #/ np.sum(w_coef_short)
        fsl_shift = df_coreg.loc[ii,f'shift_T{i+1}'] if i<3 else df_coreg.loc[ii,f'shift_R{i-2}']
        #print(f'n={i} mean disp {i} (full/approx) = {rrr}  / {rrr2} after shift {rcheck2}')
        cname = f'before_coreg_short_wTF_Disp_{i}';        df_coreg.loc[ii, cname] = rrr2

        print(f'n={i} fsl shift {fsl_shift}  wTF shift {wshift[i]}  wshaw shift {wshift_shaw[i]}')
        #spm_imatrix(npl.inv(Aff_mean),order=0)
        cname = f'w_expTF_disp{i}';        df_coreg.loc[ii, cname] = wshift[i]
        cname = f'w_shaw_disp{i}';        df_coreg.loc[ii, cname] = wshift_shaw[i]


df_coreg.to_csv(out_path+'/df_core_sub.csv')

fig, axs = plt.subplots(nrows=2,ncols=3)
max_errors=0
for  i, ax in enumerate(axs.flatten()):
    fsl_shift = df_coreg[ f'shift_T{i + 1}'] if i < 3 else df_coreg[ f'shift_R{i - 2}']
    xname = 'fsl_shift' #'w_expTF_disp' #'before_coreg_short_wTF_Disp_' #'m_wTF2_Disp_'
    yname = 'w_shaw_disp' #'before_coreg_wTF_Disp_'  #
    #x = df_coreg[f'm_wTF2_Disp_{i}'] + fsl_shift

    x = fsl_shift if xname=='fsl_shift' else df_coreg[f'{xname}{i}'];
    y = df_coreg[f'{yname}{i}'];
    #y=df_coreg[f'm_wTF_Disp_{i}'] + fsl_shift

    ax.scatter(x,y);ax.plot([x.min(),x.max()],[x.min(),x.max()]);
    max_error = np.max(np.abs(y-x)); mean_error = np.mean(np.abs(y-x))
    max_errors = max_error if max_error>max_errors else max_errors
    corrPs, Pval = ss.pearsonr(x, y)
    print(f'cor is {corrPs} P {Pval} max error for {i} is {max_error}')
    ax.title.set_text(f'R = {corrPs:.2f} err mean {mean_error:.2f} max {max_error:.2f}')
fig.text(0.5, 0.04, xname, ha='center')
fig.text(0.04, 0.5, yname, va='center', rotation='vertical')
print(f'max  is {max_errors}')
# m_wTF_Disp_ +fsl_shift ==  before_coreg_wTF_Disp_


ds = df1[(df1['srot']>10) | (df1['stra']>20)]
fp_path = ds.fp.values[1]
for fp_path in ds.fp.values:
    fitpar = np.loadtxt(fp_path)
    #plt.figure();plt.plot(fitpar.T); plt.legend({'tx','ty','tz','rx','ry','rz'})
    print(fp_path)
    print(fitpar.min(axis=1))


sdata, tmot, config_runner = select_data(fjson, param, to_canonical=False)
image = sdata.t1.data[0]
fi = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)

one_df, smot = perform_one_motion(fp_path, fjson, coreg_nifti_reg=False, just_motion=True)
tma = smot.history[2]

mres = ModelCSVResults(df_data=one_df, out_tmp="/tmp/rrr")
keys_unpack = ['transforms_metrics', 'm_t1'];
suffix = ['m', 'm_t1']
one_df1 = mres.normalize_dict_to_df(keys_unpack, suffix=suffix);
one_df1.m_wTF_Disp_1

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


sdata, tmot, mr = select_data(fjson, param)
