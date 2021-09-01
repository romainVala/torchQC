from util_affine import *
import matplotlib.pyplot as plt, pandas as pd, seaborn as sns
import torchio as tio, torch, time
from nibabel.viewers import OrthoSlicer3D as ov
import glob
sns.set_style("darkgrid")
def parse_string_array(ss):
    ss=ss.replace('[','')
    ss=ss.replace(']','')
    dd = ss.split(' ')
    dl=[]
    for ddd in dd:
        if len(ddd)>1:
            dl.append(float(ddd))
    return np.array(dl)
def get_fp_path(x,outdir):
    return outdir + '/' + x + '/fitpars_orig.txt'
def get_history_path(x,outdir):
    dirout = outdir + '/' + x + '/*gz'
    fhist = glob.glob(dirout)
    return fhist[0]
def get_num_fitpar(x):
    return int(x[-1:])
def disp_to_vect(s,key,type):
    if type=='trans':
        k1 = key[:-1] + '0'; k2 = key[:-1] + '1'; k3 = key[:-1] + '2';
    else:
        k1 = key[:-1] + '3'; k2 = key[:-1] + '4'; k3 = key[:-1] + '5';
    return np.array([s[k1], s[k2], s[k3]])

Lustre=True
if Lustre:
    fjson = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/test1/main.json'
    rootdir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/'
else:
    fjson = '/data/romain/PVsynth/motion_on_synth_data/test1/main.json'
    rootdir = '/data/romain/PVsynth/'
res_name = 'random_mot_amp_noseed_2'
res_name = 'random_mot_amp_noseed_newSuj'
out_path =  rootdir + res_name

param = dict();param['suj_contrast'] = 1;param['suj_noise'] = 0.01;param['suj_index'] = 0;param['suj_deform'] = 0;param['displacement_shift_strategy']=None
sdata, tmot, config_runner = select_data(fjson, param, to_canonical=False)
image = sdata.t1.data[0]; brain_mask = sdata.brain.data[0]

tmot.preserve_center_frequency_pct = 0;tmot.nT=218
tmot.maxGlobalDisp, tmot.maxGlobalRot = (4,4), (3,3)
tmot._simulate_random_trajectory()
plt.figure();plt.plot(tmot.fitpars.T)

all_params = dict(
    amplitude = [0.5, 1, 2, 4, 8],
    nb_x0s = [50],
    x0_min = [0],
    cor_disp = [False,],
    disp_str =  ['no_shift'],
    suj_index = [ 475, 478, 492, 497, 499, 500], #[474, 475, 477, 478, 485, 492, 497, 499, 500, 510],
    #suj_seed = [0,2,4,7,9], #[0,1,2,4,5,6,7,8,9],
    suj_seed = [None], # if multiple the same get in same dir , None, None, None, None], #[0,1,2,4,5,6,7,8,9],
    suj_contrast = [1, 2, 3],
    suj_deform = [False, ],
    suj_noise = [0.01, 0.05 ],
    clean_output = [0],
    new_suj = [True],
)
params = product_dict(**all_params)
nb_x0s = all_params['nb_x0s']; nb_sim = len(params) * nb_x0s[0]
print(f'performing loop of {nb_sim} iter 10s per iter is {nb_sim*10/60/60} Hours {nb_sim*10/60} mn ')
print(f'{nb_x0s[0]} nb x0 and {len(params)} params')

split_length = 1
create_motion_job(params, split_length, fjson, out_path, res_name=res_name, type='one_motion_simulated',job_pack=1)

#read results
fcsv = glob.glob(out_path+'/*/*csv')
df = [pd.read_csv(ff) for ff in fcsv]
df1 = pd.concat(df, ignore_index=True); # dfall = df1

array_keys = [ 'max_disp', 'mean_disp_mask', 'min_disp']
for kk in array_keys:
    df1[kk] = df1[kk].apply(lambda x: parse_string_array(x))
df1['fp'] = df1.suj_name_fp.apply(lambda x: get_fp_path(x,out_path))
df1['f_hist'] = df1.suj_name_fp.apply(lambda x: get_history_path(x,out_path))
df1['num_fp'] = df1.suj_name_fp.apply(lambda x: get_num_fitpar(x))

key_disp = [k for k in df1.keys() if 'isp_1' in k]; key_replace_length = 7  # computed in torchio
key_disp += ['shift_0'];
for k in key_disp:
    if 'shift_0' in k:
        key_replace_length = 2
    new_key = k[:-key_replace_length] +'_trans'
    df1[new_key] = df1.apply(lambda s: disp_to_vect(s, k, 'trans'), axis=1)
    df1[f'{new_key}N'] = df1[new_key].apply(lambda x: npl.norm(x)) / df1.amplitude
    new_key = k[:-key_replace_length] +'_rot'
    df1[new_key] = df1.apply(lambda s: disp_to_vect(s, k, 'rot'),  axis=1)
    df1[f'{new_key}N'] = df1[new_key].apply(lambda x: npl.norm(x)) / df1.amplitude
    for ii in range(6):
        key_del = f'{k[:-1]}{ii}';  print(f'create {new_key}  delete {key_del}') #del(df1[key_del])


#ploting
df1['diffL1'] = df1.no_shift_L1_map - df1.m_L1_map
df1['diffL1'] = df1.no_shift_NCC - df1.m_NCC
sns.catplot(data=df1,x='amplitude', y='diffL1', col='suj_contrast',hue='suj_noise', kind='boxen', col_wrap=2, dodge=True)


xx, yy ='MD_mask_wTF', 'meanDispJ_wTF'
xx, yy ='MD_mask_mean', 'mean_DispJ'
grid = sns.FacetGrid(df1, col = "suj_contrast", hue = "suj_noise", col_wrap=2)
grid.map(sns.scatterplot,xx, yy, alpha=0.2)
grid.add_legend()

plt.show()
sns.catplot(data=df1, x='m_L1_map', y='m_NCC', col='suj_contrast',hue='suj_noise', kind='scatter', col_wrap=2 )

dfsub = df1[(df1.suj_index==500) & (df1.suj_seed==0) & (df1.num_fp==1) & (df1.suj_contrast==1) & (df1.amplitude==1)]
dfsub = df1[(df1.suj_index==500) & (df1.suj_seed==0) & (df1.num_fp==1)  ]
for i in range(16):
    fp_path = dfsub.fp.values[i]
    fitpar = np.loadtxt(fp_path)
    plt.figure(); plt.plot(fitpar.T)
#ok because of seeding I get the same fitar (different scalling) so only nb_seed * nb_x0s different fitpar (* nb_ampl diff)
len(df1.no_shift_center_Disp_1.unique())


#name from torchio metric
ynames = [ 'wTFshort_trans', 'wTFshort2_trans', 'wSH_trans', 'wSH2_trans'] #'wTF_trans', 'wTF2_trans',
ynames += [ 'wTFshort_rot', 'wTFshort2_rot', 'wSH_rot', 'wSH2_rot'] #'wTF_rot', 'wTF2_rot',
#ynames = [ 'no_shift_wTF_trans', 'no_shift_wTF2_trans', 'no_shift_wTFshort_trans', 'no_shift_wTFshort2_trans', 'no_shift_wSH_trans', 'no_shift_wSH2_trans']
#ynames += [ 'no_shift_wTF_rot', 'no_shift_wTF2_rot', 'no_shift_wTFshort_rot', 'no_shift_wTFshort2_rot', 'no_shift_wSH_rot', 'no_shift_wSH2_rot']
ynames += [ 'center_trans', 'mean_trans','center_rot', 'mean_rot']

e_yname,d_yname=[],[]
e_yname += [f'error_{yy}' for yy in ynames ]
d_yname += [f'{yy}' for yy in ynames ]
for yname in ynames:
    xname ='shift_rot' if  'rot' in yname else 'shift_trans';
    x = df1[f'{xname}'];  y = df1[f'{yname}'];
    cname = f'error_{yname}';
    df1[cname] = (y).apply(lambda x: npl.norm(x)) / df1.amplitude #
    #df1[cname] = (y-x).apply(lambda x: npl.norm(x))

ind_sel = range(df1.shape[0]) #(df1.trans_max<10) & (df1.rot_max<10)
dfm = df1.melt(id_vars=['fp','amplitude','suj_noise','suj_contrast'], value_vars=e_yname, var_name=['shift'], value_name='error')
dfm["ei"] = 0
for kk in  dfm['shift'].unique() :
    dfm.loc[dfm['shift'] == kk, 'ei'] = 'trans' if 'trans' in kk else 'rot'  #int(kk[-1])
    dfm.loc[dfm['shift'] == kk, 'shift'] = kk[6:-6] if 'trans' in kk else kk[6:-4]
    #dfm.loc[dfm['shift'] == kk, 'shift'] = kk[6:] if 'trans' in kk else kk[6:]

sns.catplot(data=dfm,x='shift', y='error', hue='ei', col='amplitude', kind='boxen', col_wrap=2, dodge=True)
sns.catplot(data=dfm,x='shift', y='error', hue='ei', col='suj_contrast', kind='boxen', col_wrap=2, dodge=True)

import statsmodels.api as sm
ys = ['wSH_transN','wTFshort_transN','wSH2_transN','wTFshort2_transN']
x='shift_transN' ; y ='wSH2_transN'
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 8)); axs = axs.flatten()
for ii,y in enumerate(ys):
    ax = axs[ii]
    sm.graphics.mean_diff_plot(df1[x], df1[y], ax=ax); plt.tight_layout()


#strange big L1 variations
dfsub = df1[df1.m_L1_map>10]
#reproduce !
from read_gz_results import GZReader
gzr = GZReader(gz_path=dfsub.f_hist.values[0])
v=gzr.get_volume_torchio(0)
vi,tmot = gzr.get_volume_torchio_without_motion(0)
fitpar = tmot.fitpars['t1']; fitpar = fitpar - np.mean(fitpar,axis=1, keepdims=True)
tmot.fitpars['t1'] = fitpar
tmot.frequency_encoding_dim = dict(t1=tmot.frequency_encoding_dim)
resdir = '/data/romain/PVsynth/random_mot_amp_noseed_img'
apply_motion(vi, tmot, fitpar, root_out_dir=resdir, config_runner=config_runner,suj_name='toto', suj_orig_name='toto_orig')

fitpar=v.history[2].fitpars['t1']
fdj = np.zeros(fitpar.shape[1])
for ii,P in enumerate(fitpar.T):
    aff = spm_matrix( np.hstack((P, np.array([1, 1, 1, 0, 0, 0]))))
    fdj[ii] = compute_FD_J(aff)


#Testing motion one shot ... but convention error with torchio randomMotion, angle in oposit direction
image = sdata.t1.data[0]
image_shape = image.shape
center = [xx//2 for xx in image_shape ]

lin_spaces = [np.linspace(-center[ii],center[ii]-1 , x) for ii,x in enumerate(image_shape)]
meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
meshgrids.append(np.ones_like(image))
grid_coords = np.array([mg.flatten() for mg in meshgrids])
grid_out = np.matmul(aff,grid_coords)
vector_field = grid_out - grid_coords

iv = np.reshape(vector_field, [4] + list(image_shape))
ivnor = npl.norm(iv,axis=0)

freq_domain = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)

phase_shift = grid_out[:3,:].sum(axis=0)  # phase shift is added

exp_phase_shift = np.exp(-1j *2*np.pi * phase_shift)
freq_domain_translated = exp_phase_shift * freq_domain.reshape(-1)

iout = abs( np.fft.ifftshift(np.fft.ifftn(freq_domain_translated.reshape(image.shape))))
ov(iout)

#kspace grid
lin_spaces = [np.linspace(-0.5, 0.5, x)*2*math.pi for x in freq_domain.shape]  # todo it suposes 1 vox = 1mm
meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
meshgrids.append(np.zeros_like(image))
kspace_coords = np.array([mg.flatten() for mg in meshgrids])

image_flat = image.numpy().flatten()
grid_out_img = grid_out.reshape([4] + list(image_shape))
kspace_coords_img  = kspace_coords.reshape([4] + list(image_shape))
freq_trans=np.zeros_like(image).astype(complex)

for kk in range(grid_out_img.shape[2]):
    phase_s = ( kspace_coords_img[:3,:,kk,:] * grid_out_img[:3, :,kk,:] ).sum(axis=0)
    freq_trans += image[:,kk,:] * np.exp(1j * phase_s )
    if kk%100 == 0:
        print(kk)





aff = spm_matrix([20, 0, 0, 0, 0, 10,  1, 1, 1, 0, 0, 0])
freq_domain = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)
aff = spm_matrix([20, 0, 0, 0, 0, 0,  1, 1, 1, 0, 0, 0])
#aff = aff.transpose([-1, 0, 1])

#lin_spaces = [np.linspace(-0.5, 0.5, x) for x in freq_domain.shape]  # todo it suposes 1 vox = 1mm
lin_spaces = [np.linspace(-0.5, 0.5, x)*2*math.pi for x in freq_domain.shape]  # todo it suposes 1 vox = 1mm
meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
meshgrids.append(np.zeros_like(image))
grid_coords = np.array([mg.flatten(order='F') for mg in meshgrids])

grid_out = np.matmul(aff.T,grid_coords)

import finufft
eps=1E-7

f = np.zeros(image.shape, dtype=np.complex128, order='F')

#freq_domain_data_flat = np.asfortranarray(freq_domain.flatten(order='F'))
freq_domain_data_flat = freq_domain.flatten(order='F')

#phase_shift = grid_out[:3].sum(axis=0)  # phase shift is added
phase_shift = grid_out[3]
exp_phase_shift = np.exp(-1j * phase_shift)
freq_domain_data_flat = freq_domain_data_flat * exp_phase_shift

finufft.nufft3d1(grid_out[0], grid_out[1], grid_out[2], freq_domain_data_flat,
                 eps=eps, out=f, debug=0, spread_debug=0, spread_sort=2, fftw=0, modeord=0,
                 chkbnds=0, upsampfac=1.25)  # upsampling at 1.25 saves time at low precisions
im_out = f.reshape(image.shape, order='F')
ov(abs(im_out))

#compare with tio implementation
fitpar = np.array([20, 0, 0, 0, 0, 30])
tmot.fitpars = np.tile(fitpar[np.newaxis].T,[1,200])
tmot.simulate_displacement = False;  # tmot.oversampling_pct = 1

smot = tmot(sdata)
ov(smot.t1.data[0])
nufft3d1


grid_out = np.matmul(aff.T,grid_coords)

f = np.zeros(grid_out[0].shape, dtype=np.complex128, order='F')
ip = np.asfortranarray(image.numpy().astype(complex) )
finufft.nufft3d2(grid_out[0], grid_out[1], grid_out[2], ip,
                 eps=eps, out=f, debug=0, spread_debug=0, spread_sort=2, fftw=0, modeord=0,
                 chkbnds=0, upsampfac=1.25, isign=-1)  # upsampling at 1.25 saves time at low precisions
f = f * np.exp(1j * grid_out[3])
iout = abs( np.fft.ifftshift(np.fft.ifftn(f.reshape(image.shape, order='F'))))
ov(iout)

