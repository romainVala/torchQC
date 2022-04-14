from util_affine import *
import matplotlib.pyplot as plt, pandas as pd, seaborn as sns
import torchio as tio, torch, time
from nibabel.viewers import OrthoSlicer3D as ov
import glob
sns.set_style("darkgrid")


def correct_string_bool(x):
    if x=='0.0':
        x = 'False'
    if x=='1.0':
        x = 'True'
    return x
def split_df_unique_vals(df, list_keys):
    dfout = dict()
    k = list_keys[0]
    vals = df[k].unique()
    for vv in vals:
        out_name = f'{k}_{vv}'
        dfsub = df[df[k]==vv]
        dfout[out_name] = dfsub

    if len(list_keys)>1:
        list_keys = list_keys[1:]
        for k_filter in list_keys:
            New_dfout = dict()
            for k, v in dfout.items():
                dd=split_df_unique_vals(v, [k_filter])
                for kk, vv in dd.items():
                    out_name = f'{k}_{kk}'
                    New_dfout[out_name] = vv
            dfout = New_dfout

    return dfout
def show_df(df, keys):
    rout, rout_unique = dict(),dict()
    if isinstance(keys,dict):
        keys = keys.keys()
    for k in keys:
        if k not in df:
            print(f'skiping missing {k}')
            continue
        vars = df[k].unique()
        if len(vars)==1:
            rout_unique[k] = vars
        else:
            vars = np.sort(vars)
            rout[k] = vars
    if 'xend' in df:
        if df.nb_x0s.values[0]>1:
            show_x0(df)
    print('unique values')
    for k, v in rout_unique.items():
        print(f'  {k} \t: {v}')
    print('multiple values')
    for k, v in rout.items():
        print(f'  {k} \t N:{len(v)}\t: {v}')
def show_x0(df1):
    for s in np.sort(df1.sigma.unique()):
        dfs = df1[df1.sigma==s]
        nbx0_strn, nbX0 = '',0
        for sym in np.sort(dfs.sym.unique()):
            dfss = dfs[dfs.sym==sym]
            nbX0 += len(dfss.xend.unique())
            nbx0_strn = f'{nbx0_strn} {len(dfss.xend.unique())} Sym={sym}'
        nb_vol = len(dfs)/nbX0
        print(f'Sig {s} \t nb Vol {nb_vol} = {len(dfs)} pts / {nbX0} X0 ({nbx0_strn}) [{min(dfs.x0.unique())} '
              f'{max(dfs.x0.unique())}]  Xend [{min(dfs.xend.unique())} {max(dfs.xend.unique())}]')

    #return rout, rout_unique
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
def disp_to_vect(s,key,type, remove_ref=False):
    if type=='trans':
        k1 = key[:-1] + '0'; k2 = key[:-1] + '1'; k3 = key[:-1] + '2';
        res = np.array([s[k1], s[k2], s[k3]])
        if remove_ref:
            res = res - s['init_trans_vect']
    else:
        k1 = key[:-1] + '3'; k2 = key[:-1] + '4'; k3 = key[:-1] + '5';
        res = np.array([s[k1], s[k2], s[k3]])
        if remove_ref:
            res = res - s['init_rot_vect']
    return res
def correct_amplitude(s): #not ideal, as need to know the disp vector, should be done before in corup_motion
    return np.round(2* s['amplitude']/np.sqrt(3) if (s['mvt_axe']=='transXtransYtransZ') | (s['mvt_axe']=='rotXrotYrotZ')  else 2*s['amplitude'])/2
def compute_disp(df1, normalize_amplitude=True, corect_initial_disp=False):
    key_disp = [k for k in df1.keys() if 'isp_1' in k];
    key_replace_length = 7  # computed in torchio
    key_disp += ['shift_0'];
    for k in key_disp:
        if 'shift_0' in k:
            key_replace_length = 2
        if corect_initial_disp:
            remove_ref = True if 'shift' in k else False #only sihift_ and no_shift metrics should be change
        else:
            remove_ref = False
        new_key = k[:-key_replace_length] + '_trans'
        df1[new_key] = df1.apply(lambda s: disp_to_vect(s, k, 'trans', remove_ref), axis=1)
        if normalize_amplitude:
            df1[f'{new_key}N'] = df1[new_key].apply(lambda x: npl.norm(x)) / df1.amplitude
        else:
            df1[f'{new_key}N'] = df1[new_key].apply(lambda x: npl.norm(x))

        new_key = k[:-key_replace_length] + '_rot'
        df1[new_key] = df1.apply(lambda s: disp_to_vect(s, k, 'rot', remove_ref), axis=1)
        if normalize_amplitude:
            df1[f'{new_key}N'] = df1[new_key].apply(lambda x: npl.norm(x)) / df1.amplitude
        else:
            df1[f'{new_key}N'] = df1[new_key].apply(lambda x: npl.norm(x))
        for ii in range(6):
            key_del = f'{k[:-1]}{ii}';
            print(f'create {new_key}  delete {key_del} remove res is {remove_ref}')  # del(df1[key_del])

    return df1
#hack for negative amp
def correct_neg_amp(s):
    if s['amplitude']<0:
        s['amplitude'] = -s['amplitude']
        s['mvt_axe'] = s['mvt_axe'] + '-'
    return s
def correct_sigma_sym(s):
    if s['sym']==1:
        s['sigma'] = s['sigma'] * 2
    return s

def read_csv_motion(out_path, corect_initial_disp=False, correct_amplitude=False,
                    csv_search = '/*/*csv'):

    fcsv = glob.glob(out_path + csv_search)
    print(f'reading {len(fcsv)} csv files')
    df = [pd.read_csv(ff) for ff in fcsv]
    df1 = pd.concat(df, ignore_index=True);  # dfall = df1
    print(f'conversions / corrections')

    # after 11 sep 2021, correction is done in correct_data for xyz direction only
    if correct_amplitude:
        df1['amplitude'] = df1.apply(lambda s: correct_amplitude(s), axis=1)

    if 'max_disp' in df1:
        array_keys = ['max_disp', 'mean_disp_mask', 'min_disp']
        for kk in array_keys:
            df1[kk] = df1[kk].apply(lambda x: parse_string_array(x))
    # df1['fp'] = df1.suj_name_fp.apply(lambda x: get_fp_path(x,out_path))
    if 'out_dir' in df1:
        df1['fp'] = df1.out_dir.apply(lambda x: get_fp_path(x, out_path))
    if 'suj_name_fp' in df1:
        df1['f_hist'] = df1.suj_name_fp.apply(lambda x: get_history_path(x, out_path))
        df1['num_fp'] = df1.suj_name_fp.apply(lambda x: get_num_fitpar(x))

    if np.sum(df1.amplitude<0):
        df1 = df1.apply(lambda x: correct_neg_amp(x), axis=1)

    if corect_initial_disp:
        # only for fsl_coreg_along_x0_rotZ_origY_suj_2_con
        # because fp has been center to mean, max shift will change with sigma
        # to get the shift without center_to_mean we recompute this mean by taking the first value of fp_orig
        # assign column array df1['init_trans_vect'] = df1.apply(lambda x: np.array([0,0,0]),axis=1)
        # but then this does not work: cname = f'init_trans_vect'    #df1.loc[ii, cname] =  np.array([fitpar[0,0],fitpar[1,0],fitpar[2,0]])
        c1, c2 = [], []
        for ii, fp_path in enumerate(df1.fp.values):
            fitpar = np.loadtxt(fp_path)
            c1.append(np.array([fitpar[0, -1], fitpar[1, -1], fitpar[2, -1]]))  # at the end should be 0 IF sigma motion
            c2.append(np.array([fitpar[3, -1], fitpar[4, -1], fitpar[5, -1]]))
        df1['init_trans_vect'] = c1
        df1['init_rot_vect'] = c2
        df1['init_rotN'] = df1.init_rot_vect.apply(lambda x: npl.norm(x))  # to check display

    df1 = compute_disp(df1, normalize_amplitude=False, corect_initial_disp=corect_initial_disp)

    all_params = dict(
        amplitude=0, sigma=0,  nb_x0s=0, x0_min=0, x0_step=0,  sym=0, mvt_type=0, mvt_axe=0, displacement_shift_strategy=0,
        suj_index=0, suj_seed=0, suj_contrast=0, suj_deform=0, suj_noise=0, clean_output=0, new_suj=0,
         )
    show_df(df1, all_params)

    return df1

Lustre=False
if Lustre:
    fjson = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/test1/main.json'
    rootdir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion_elastix/'
else:
    fjson = '/data/romain/PVsynth/motion_on_synth_data/test1/main.json'
    rootdir = '/data/romain/PVsynth/'
res_name = 'random_mot_amp_noseed_2'
res_name = 'random_mot_amp_noseed_newSuj'
res_name = 'random_mot_amp_noseed_newSuj_center_wTF2'
res_name = 'random_mot_amp_noseed_newSuj_no_noise_center_wTF_TF2'
out_path =  rootdir + res_name

param = dict();param['suj_contrast'] = 1;param['suj_noise'] = 0.01;param['suj_index'] = 0;param['suj_deform'] = 0;param['displacement_shift_strategy']=None
sdata, tmot, config_runner = select_data(fjson, param, to_canonical=True)
image = sdata.t1.data[0]; brain_mask = sdata.brain.data[0]
freq_domain = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)

all_params = dict(
    amplitude = [1, 2, 4, 8], #[0.5, 1, 2, 4, 8],
    nb_x0s = [50],
    x0_min = [0],
    cor_disp = [False,],
    suj_index = [ 475, 478, 492, 497, 499, 500], #[474, 475, 477, 478, 485, 492, 497, 499, 500, 510],
    #suj_seed = [0,2,4,7,9], #[0,1,2,4,5,6,7,8,9],
    suj_seed = [None], # if multiple the same get in same dir , None, None, None, None], #[0,1,2,4,5,6,7,8,9],
    suj_contrast = [1, 3],
    suj_deform = [False, True],
    suj_noise = [0], #[0.01 ],
    clean_output = [0],
    new_suj = [True],
    displacement_shift_strategy = ['1D_wTF' ,'1D_wTF2'],
)
params = product_dict(**all_params)
nb_x0s = all_params['nb_x0s']; nb_sim = len(params) * nb_x0s[0]
print(f'performing loop of {nb_sim} iter 10s per iter is {nb_sim*10/60/60} Hours {nb_sim*10/60} mn ')
print(f'{nb_x0s[0]} nb x0 and {len(params)} params')

split_length = 1
create_motion_job(params, split_length, fjson, out_path, res_name=res_name, type='one_motion_simulated', job_pack=1,
                  jobdir='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion_elastix/')

#read results
df1 = read_csv_motion(out_path, corect_initial_disp=False)

df1.to_csv(out_path+'res_all.csv')
#ploting
df1['diffL1'] = df1.no_shift_L1_map - df1.m_L1_map
df1['diffL1'] = df1.no_shift_NCC - df1.m_NCC
sns.catplot(data=df1,x='amplitude', y='diffL1', col='suj_contrast',hue='suj_noise', kind='boxen', col_wrap=2, dodge=True)


xx, yy ='MD_mask_wTF', 'meanDispJ_wTF'
xx, yy ='MD_mask_mean', 'MD_wimg_mean' #mean_DispJ'
grid = sns.FacetGrid(df1, col = "suj_contrast", hue = "suj_noise", col_wrap=2)
grid.map(sns.scatterplot,xx, yy, alpha=0.2)
grid.add_legend()

plt.show()
sns.relplot(data=df1, x='m_L1_map', y='m_NCC', col='suj_contrast',hue='suj_noise', kind='scatter', col_wrap=2 )

dfsub = df1[(df1.suj_index==500) & (df1.suj_seed==0) & (df1.num_fp==1) & (df1.suj_contrast==1) & (df1.amplitude==1)]
dfsub = df1[(df1.suj_index==500) & (df1.suj_seed==0) & (df1.num_fp==1)  ]
dfs = df1[(df1.suj_contrast==1) &(df1.suj_noise>0.05) & (df1.suj_deform==0) & (df1.suj_index==500) &
            (df1.sigma==128.)& (df1.mvt_axe=='rotZ') &(df1.amplitude==8) & (df1.x0==256.0)]
dfs=df1[(df1.mvt_type=='Ustep') & (df1.x0==256.0) & (df1.sigma==64.) ]

for i in range(16):
    fp_path = dfsub.fp.values[i]
    fitpar = np.loadtxt(fp_path)
    plt.figure(); plt.plot(fitpar.T)
#ok because of seeding I get the same fitar (different scalling) so only nb_seed * nb_x0s different fitpar (* nb_ampl diff)
len(df1.no_shift_center_Disp_1.unique())


#name from torchio metric
dfsub = df1[(df1.suj_noise==0.01) & (df1.suj_contrast<5) &( df1.amplitude>0.6)].copy()

ynames = [  'wSH_trans', 'wSH2_trans'] #'wTF_trans', 'wTF2_trans','wTFshort_trans', 'wTFshort2_trans',
ynames += [  'wSH_rot', 'wSH2_rot'] #'wTF_rot', 'wTF2_rot','wTFshort_rot', 'wTFshort2_rot',
#ynames = [ 'no_shift_wTF_trans', 'no_shift_wTF2_trans', 'no_shift_wTFshort_trans', 'no_shift_wTFshort2_trans', 'no_shift_wSH_trans', 'no_shift_wSH2_trans']
#ynames += [ 'no_shift_wTF_rot', 'no_shift_wTF2_rot', 'no_shift_wTFshort_rot', 'no_shift_wTFshort2_rot', 'no_shift_wSH_rot', 'no_shift_wSH2_rot']
ynames += [ 'center_trans', 'mean_trans','center_rot', 'mean_rot']
ynames += [ 'center_rot', 'mean_rot']

e_yname,d_yname=[],[]
e_yname += [f'error_{yy}' for yy in ynames ]
d_yname += [f'{yy}' for yy in ynames ]
for yname in ynames:
    xname ='shift_rot' if  'rot' in yname else 'shift_trans';
    x = dfsub[f'{xname}'];  y = dfsub[f'{yname}'];
    cname = f'error_{yname}';
    dfsub[cname] = (y).apply(lambda x: npl.norm(x)) / dfsub.amplitude #
    #dfsub[cname] = (y-x).apply(lambda x: npl.norm(x))

ind_sel = range(dfsub.shape[0]) #(dfsub.trans_max<10) & (dfsub.rot_max<10)
dfm = dfsub.melt(id_vars=['fp','amplitude','suj_noise','suj_contrast'], value_vars=e_yname, var_name=['shift'], value_name='error')
dfm["ei"] = 0
for kk in  dfm['shift'].unique() :
    dfm.loc[dfm['shift'] == kk, 'ei'] = 'trans' if 'trans' in kk else 'rot'  #int(kk[-1])
    dfm.loc[dfm['shift'] == kk, 'shift'] = kk[6:-6] if 'trans' in kk else kk[6:-4]
    #dfm.loc[dfm['shift'] == kk, 'shift'] = kk[6:] if 'trans' in kk else kk[6:]

sns.catplot(data=dfm,x='shift', y='error', hue='ei', col='amplitude', kind='boxen', col_wrap=2, dodge=True)
sns.catplot(data=dfm,x='shift', y='error', hue='ei', col='suj_contrast', kind='boxen', col_wrap=2, dodge=True)
sns.relplot(data=dfsub, x='wSH_rotN', y='wSH2_rotN', col='suj_contrast',hue='suj_noise', kind='scatter', col_wrap=2 )
sns.relplot(data=dfsub, x='no_shift_wSH_rotN', y='no_shift_wSH2_rotN', col='suj_contrast',hue='suj_noise', kind='scatter', col_wrap=2 )

dfsub = df1[(df1.suj_noise==0.01) & (df1.suj_contrast<5) &( df1.amplitude>0.6)].copy()
kdisp =  [k for k in df1.keys() if 'rotN' in k];
kdisp =  [k for k in df1.keys() if 'transN' in k];
for kk in kdisp:
    dfsub[kk] = dfsub[kk]*dfsub['amplitude'] #undo the amplitude normaizatin

fig = sns.relplot(data=dfsub, y='no_shift_wSH_rotN', x='shift_rotN', col='amplitude', kind='scatter', col_wrap=2 )
for aa in fig.axes:
    aa.plot([0,0.5],[0,0.5],'k--');


ykey='wSH_rotN'
dfsub = df1[(df1.suj_noise==0.01) & (df1.suj_contrast==1) &( df1.amplitude==8.)].copy()
isort = dfsub[ykey].sort_values(ascending=False, axis=0).index
ii=np.argsort(dfsub.loc[isort[:50],'m_L1_map_brain'].values )

for fp in dfsub.loc[isort[ii[:10]],'fp'].values:
    ff = np.loadtxt(fp)
    plt.figure(); plt.plot(ff[3:,:].T); center = 218/2-1; plt.plot([center, center],[np.min(ff[3:,:]), np.max(ff[3:,:])],'k')
#yes avec wSH2 on voit tous les plot avec un swallow au centre !

dfsub.loc[isort[:10],'shift_rot'].values ;  dfsub.loc[isort[:10],'no_shift_wSH2_rotN'].values

dfsub=df3[(df3.amplitude>6) & (df3.suj_contrast==1)]
ykey='no_shift_L1_map_brain'
isort = dfsub[ykey].sort_values(ascending=False, axis=0).index


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
    get_matrix_from_euler_and_trans(P)
    fdj[ii] = compute_FD_J(aff)








#Testing motion one shot ... but convention error with torchio randomMotion, angle in oposit direction


tmot.displacement_shift_strategy = '1D_wTF'
tmot.nufft_type = '1D_type2'
tmot.preserve_center_frequency_pct = 0;tmot.nT=218
tmot.maxGlobalDisp, tmot.maxGlobalRot = (4,4), (3,3)

tmot._simulate_random_trajectory()
plt.figure();plt.plot(tmot.fitpars.T)
fitpar = tmot.fitpars

tmot.fitpars =fitpar
tmot.simulate_displacement = False# True;  # tmot.oversampling_pct = 1

smot = tmot(sdata)
fitpar = tmot.fitpars
ov(smot.t1.data[0])
#fitpar =  interpolate_fitpars(tmot.fitpars, len_output=image.shape[1])


import finufft
def nufft_type1(freq_domain, fitpar, trans_last=False):
    #trans_last=False #weird idea while looking to match tio.Affine ... no more usefull
    eps = 1E-7
    f = np.zeros(image.shape, dtype=np.complex128, order='F')

    lin_spaces = [np.linspace(-0.5, 0.5-1/x, x)*2*math.pi for x in freq_domain.shape]  # todo it suposes 1 vox = 1mm
    #remove 1/x to avoid small scaling
    meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
    # pour une affine on ajoute de 1, dans les coordone du point, mais pour le augmented kspace on ajoute la phase initial, donc 0 ici
    meshgrids.append(np.zeros_like(image))

    grid_coords = np.array([mg for mg in meshgrids]) #grid_coords = np.array([mg.flatten(order='F') for mg in meshgrids])
    grid_out = grid_coords
    #applied motion at each phase step (on the kspace grid plan)
    for nnp in range(fitpar.shape[1]):
        aff = get_matrix_from_euler_and_trans(fitpar[:,nnp])
        aff = np.linalg.inv(aff)
        grid_plane = grid_out[:,:,nnp,:]
        shape_mem = grid_plane.shape
        grid_plane_moved = np.matmul(aff.T, grid_plane.reshape(4,shape_mem[1]*shape_mem[2])) #equ15 A.T * k0
        #grid_plane_moved = np.matmul( grid_plane.reshape(4,shape_mem[1]*shape_mem[2]).T, aff.T).T # r0.T * A.T
        grid_out[:, :, nnp, :] = grid_plane_moved.reshape(shape_mem)
    print(f'nufft 1 apply {aff.T}')

    phase_shift = grid_out[3].flatten(order='F')
    exp_phase_shift = np.exp( 1j * phase_shift)  #+1j -> x z == tio, y inverse

    if trans_last:
        freq_domain_data_flat = freq_domain.flatten(order='F')
    else:
        freq_domain_data_flat = freq_domain.flatten(order='F')* exp_phase_shift # same F order as phase_shift if not inversion x z

    finufft.nufft3d1(grid_out[0].flatten(order='F'), grid_out[1].flatten(order='F'),
                     grid_out[2].flatten(order='F'), freq_domain_data_flat,
                     eps=eps, out=f, debug=0, spread_debug=0, spread_sort=2, fftw=0, modeord=0,
                     chkbnds=0, upsampfac=2, isign= 1)  # upsampling at 1.25 saves time at low precisions
    #im_out = f.reshape(image.shape, order='F')
    #im_out = f.flatten().reshape(image.shape)
    im_out = f / f.size
    #im_out /= im_out.size

    if trans_last:
        freq_domain = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(im_out)))).astype(np.complex128)
        freq_domain_data_flat = freq_domain.flatten(order='F')* exp_phase_shift # same F order as phase_shift if not inversion x z
        im_out = abs( np.fft.ifftshift(np.fft.ifftn(freq_domain_data_flat.reshape(image.shape,order='F'))))

    return abs(im_out)


def nufft_type2(image, fitpar):
    eps = 1E-7
    lin_spaces = [np.linspace(-0.5, 0.5-1/(x-1), x)*2*math.pi for x in image.shape]  # todo it suposes 1 vox = 1mm
    #lin_spaces = [np.linspace(-0.5 -1/x, 0.5 -1/x, x) * 2 * math.pi for x in image.shape]  # todo it suposes 1 vox = 1mm
    off_center = np.array([(x/2-x//2)*2 for x in image.shape]) #one voxel shift if odd ! todo resolution ?
    #aff_offenter = np.eye(4); aff_offenter[:3,3] = -off_center
    fitpar[:3,:] = fitpar[:3,:] - np.repeat(np.expand_dims(off_center,1),fitpar.shape[1], axis=1)

    meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
    meshgrids.append(np.zeros_like(image))

    grid_coords = np.array([mg for mg in meshgrids]) #grid_coords = np.array([mg.flatten(order='F') for mg in meshgrids])
    grid_out = grid_coords
    #applied motion at each phase step (on the kspace grid plan)
    for nnp in range(fitpar.shape[1]):

        aff = get_matrix_from_euler_and_trans(fitpar[:,nnp])
        #aff = np.matmul(aff_offenter, aff)
        grid_plane = grid_out[:,:,nnp,:]
        shape_mem = grid_plane.shape
        grid_plane_moved = np.matmul(aff.T, grid_plane.reshape(4,shape_mem[1]*shape_mem[2])) #equ15 A.T * k0
        grid_out[:, :, nnp, :] = grid_plane_moved.reshape(shape_mem)

    print(f'nufft 2 apply {aff.T}')

    f = np.zeros(grid_out[0].shape, dtype=np.complex128, order='F').flatten() #(order='F')
    ip = np.asfortranarray(image.numpy().astype(complex) )

    finufft.nufft3d2(grid_out[0].flatten(order='F'), grid_out[1].flatten(order='F'), grid_out[2].flatten(order='F'), ip,
                     eps=eps, out=f, debug=0, spread_debug=0, spread_sort=2, fftw=0, modeord=0,
                     chkbnds=0, upsampfac=2, isign=-1)  # upsampling at 1.25 saves time at low precisions

    f = f * np.exp(-1j * grid_out[3].flatten(order='F'))
    f = f.reshape(ip.shape,order='F')
    #f = np.ascontiguousarray(f)  #pas l'aire de changer grand chose
    iout = abs( np.fft.ifftshift(np.fft.ifftn(f)))
    #iout = abs( np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(f))))
    return iout
    #ov(iout)



## direct problem, registration with SimpleElastix
#create affine


#tes single affine
sdata = tio.datasets.Colin27(); sdata.pop('head'); sdata.pop('brain')
tmot = tio.RandomMotionFromTimeCourse()
tmot.simulate_displacement = False# True;  # tmot.oversampling_pct = 1
image = sdata.t1.data[0]; #brain_mask = sdata.brain.data[0]
freq_domain = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)
#freq_domain = (np.fft.fftshift(np.fft.fftn(image))).astype(np.complex128)

sdata.pop('label'); sdata.pop('brain')
np.set_printoptions(2)

sdata.t1.save('orig.nii')
P = np.zeros(6)
for k in range(0,6):
    #P = np.zeros(6)
    P[k] = 10 + k #(k - 2) * 10 + 5
    Euler_angle, Translation = P[3:] , P[:3] ;
    fitpar =  np.tile(np.array([P]).T,[1,200])

    #taff = tio.Affine([1,1,1],Euler_angle, Translation, center='origin');  img_center_ras = np.array(sdata.t1.get_center())
    taff = tio.Affine([1,1,1],Euler_angle, Translation, center='image');img_center_ras=[0,0,0]

    aff = get_matrix_from_euler_and_trans(P) # default rot_order = 'yxz', rotation_center=img_center_ras)
    affitk = itk_euler_to_affine(Euler_angle, Translation, img_center_ras, make_RAS=False, set_ZYX=False)

    srot = taff(sdata)
    sname = f'tio_A_ind{k}.nii'
    srot.t1.save(sname)

    if 1==0:
        aff_elastix, elastix_trf = ElastixRegister(sdata.t1, srot.t1)
        aff_elastix[abs(aff_elastix)<0.0001]=0
        if np.allclose(affitk, aff_elastix, atol=1e-1):
            print(f'k:{k} almost equal ')
        else:
            print(f'k:{k} transform tioAff (then elastix) \n {affitk} \n Elastix {aff_elastix}  ')

    if np.allclose(affitk, aff, atol=1e-1):
        print(f'k:{k} almost equal ')
    else:
        print(f'k:{k} transform tioAff (then spm) \n {affitk} \n {aff}  ')

    #tmot.fitpars = fitpar
    #smot = tmot(sdata)
    #smot.t1.save(f'tio_{sname}')

    im_out = nufft_type1(freq_domain, fitpar, trans_last=False)
    srot.t1.data[0] = torch.tensor(im_out)
    srot.t1.save(f'nufft_type1_u2{sname}')


    im_out = nufft_type2(image, fitpar)
    srot.t1.data[0] = torch.tensor(im_out)
    srot.t1.save(f'nufft_type2_u2{sname}')




#https://github.com/SimpleITK/ISBI2018_TUTORIAL/blob/master/python/utilities.py

#test filtre motion
fitpar = corrupt_data(108, resolution=217, mvt_axes=[2], return_all6=True, center=None)
fitpar = corrupt_data(108, resolution=217, mvt_axes=[1], return_all6=True, center=None, sigma= 10,
                      amplitude=4, method='Ustep')

tmot.simulate_displacement = False
tmot.euler_motion_params = fitpar
tmot.kspace_order = [0, 2, 1]
tmot.oversampling_pct = 0 #.2
#tmot.displacement_shift_strategy='1D_wTF'
smot = tmot(sdata)
smot.t1.save('/home/romain.valabregue/tmp/mot_fred012_rot_moti_n1_type2.nii')

P=fitpar.max(axis=1)
taff = tio.Affine(scales=(1,1,1), degrees=P[3:], translation=P[:3], center="image")
saff=taff(sdata);
saff.t1.save('/home/romain.valabregue/tmp/aff.nii')
#tmot.euler_motion_params = np.tile(np.expand_dims(P,1),[200]) #constant motion

#test torchio motion
tmi = tio.RandomMotion(num_transforms=1)
smoti = tmi(sdata)
smoti.t1.save('/home/romain.valabregue/tmp/moti_n1.nii')
#reporduce the same time course
ttt = smoti.history[3]
deg, trans, times = ttt.degrees['t1'],ttt.translation['t1'],ttt.times['t1'],
fitpar = np.zeros([6,512])
tx = np.linspace(0,1,512)
time_index = [ np.argmin(np.abs(tx - tt)) for tt in times]; time_index.append(512)
time_index.insert(0,0);
t_list = []
k=0
nbpts = fitpar[:3, time_index[k]:time_index[k + 1]].shape[1]
fitparl = np.zeros([6, nbpts])
t_list.append(fitparl)
time_index.pop(0)
for k in range(len(times)):
    nbpts = fitpar[:3,time_index[k]:time_index[k+1]].shape[1]
    fitparl = np.zeros([6,nbpts])
    fitparl[:3,:] = np.tile(np.expand_dims(trans[k,:].T, axis=1), reps = [1,nbpts])
    fitparl[3:,:] = np.tile(np.expand_dims(deg[k,:].T, axis=1), reps = [1,nbpts])
    t_list.append(fitparl)

    #fitpar[:3,time_index[k]:time_index[k+1]] = np.tile(np.expand_dims(trans[k,:].T, axis=1), reps = [1,nbpts])
    #fitpar[3:,time_index[k]:time_index[k+1]] = np.tile(np.expand_dims(deg[k,:].T, axis=1), reps = [1,nbpts])

index = np.where(times > 0.5)[0].min()
t_list[0], t_list[index] = t_list[index], t_list[0]
for ii,tt in enumerate(t_list):
    if ii:
        f = np.hstack([f, tt])
    else:
        f = tt
tmot.euler_motion_params = f

plt.plot(f.T)





plt.plot(fitpar.T)
lin_spaces = [np.linspace(-0.5, 0.5 - 1 / x, x) * 2 * math.pi for x in image.shape]  # todo it suposes 1 vox = 1mm
#lin_spaces = [np.linspace(-0.5 -1/x, 0.5 , x) * 2 * math.pi for x in image.shape]  # todo it suposes 1 vox = 1mm

#lin_spaces = [np.linspace(0, x-1, x)*2*math.pi / x for x in freq_domain.shape]  # todo it suposes 1 vox = 1mm

meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
meshgrids.append(np.zeros_like(image))

grid_coords = np.array([mg for mg in meshgrids]) #grid_coords = np.array([mg.flatten(order='F') for mg in meshgrids])
grid_out = grid_coords
#applied motion at each phase step (on the kspace grid plan)
for nnp in range(fitpar.shape[1]):
    aff = get_matrix_from_euler_and_trans(fitpar[:,nnp])
    grid_plane = grid_out[:,:,nnp,:]
    shape_mem = grid_plane.shape
    grid_plane_moved = np.matmul(aff.T, grid_plane.reshape(4,shape_mem[1]*shape_mem[2])) #equ15 A.T * k0
    grid_out[:, :, nnp, :] = grid_plane_moved.reshape(shape_mem)

phase_shift = grid_out[3].flatten()
exp_phase_shift = np.exp( 1j * phase_shift)  #+1j -> x z == tio, y inverse

im_out = abs(np.fft.ifftshift(np.fft.ifftn(exp_phase_shift.reshape(image.shape))))

srot.t1.data[0] = torch.tensor(abs(im_out))
srot.t1.save(f'mot_filter_tZ_{sname}')

