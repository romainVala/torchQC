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
def get_num_fitpar(x):
    return int(x[-1:])
def disp_to_vect(s,key,type):
    if type=='trans':
        k1 = key[:-1] + '0'; k2 = key[:-1] + '1'; k3 = key[:-1] + '2';
    else:
        k1 = key[:-1] + '3'; k2 = key[:-1] + '4'; k3 = key[:-1] + '5';
    return np.array([s[k1], s[k2], s[k3]])

fjson = '/data/romain/PVsynth/motion_on_synth_data/test1/main.json'
fjson = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/test1/main.json'
res_name = 'random_mot_amp_noseed'
out_path = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/' + res_name
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
    clean_output = [2]
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
df1['num_fp'] = df1.suj_name_fp.apply(lambda x: get_num_fitpar(x))


#ploting
sns.catplot(data=df1,x='amplitude', y='m_L1_map', col='suj_contrast',hue='suj_noise', kind='boxen', col_wrap=2, dodge=True)

dfsub = df1[(df1.suj_index==500) & (df1.suj_seed==0) & (df1.num_fp==1) & (df1.suj_contrast==1) & (df1.amplitude==1)]
dfsub = df1[(df1.suj_index==500) & (df1.suj_seed==0) & (df1.num_fp==1)  ]
for i in range(16):
    fp_path = dfsub.fp.values[i]
    fitpar = np.loadtxt(fp_path)
    plt.figure(); plt.plot(fitpar.T)
#ok because of seeding I get the same fitar (different scalling) so only nb_seed * nb_x0s different fitpar (* nb_ampl diff)
len(df1.no_shift_center_Disp_1.unique())



key_disp = [k for k in df1.keys() if 'isp_1' in k]; key_replace_length = 7  # computed in torchio
key_disp = ['shift_0']; key_replace_length = 2
for k in key_disp:
    new_key = k[:-key_replace_length] +'_trans'
    df1[new_key] = df1.apply(lambda s: disp_to_vect(s, k, 'trans'), axis=1)
    new_key = k[:-key_replace_length] +'_rot'
    df1[new_key] = df1.apply(lambda s: disp_to_vect(s, k, 'rot'),  axis=1)
    for ii in range(6):
        key_del = f'{k[:-1]}{ii}';  print(f'create {new_key}  delete {key_del}') #del(df1[key_del])
df1['shift_transN'] =df1.shift_trans.apply(lambda x: npl.norm(x))
df1['shift_rotN'] =df1.shift_rot.apply(lambda x: npl.norm(x))


#name from torchio metric
ynames = ['wTF_trans', 'wTF2_trans', 'wTFshort_trans', 'wTFshort2_trans', 'wSH_trans', 'wSH2_trans']
ynames += ['wTF_rot', 'wTF2_rot', 'wTFshort_rot', 'wTFshort2_rot', 'wSH_rot', 'wSH2_rot']
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
    df1[cname] = (y).apply(lambda x: npl.norm(x)) #
    #df1[cname] = (y-x).apply(lambda x: npl.norm(x))

ind_sel = range(df1.shape[0]) #(df1.trans_max<10) & (df1.rot_max<10)
dfm = df1.melt(id_vars=['fp'], value_vars=e_yname, var_name='shift', value_name='error')
dfm["ei"] = 0
for kk in  dfm['shift'].unique() :
    dfm.loc[dfm['shift'] == kk, 'ei'] = 'trans' if 'trans' in kk else 'rot'  #int(kk[-1])
    dfm.loc[dfm['shift'] == kk, 'shift'] = kk[6:-6] if 'trans' in kk else kk[6:-4]
    #dfm.loc[dfm['shift'] == kk, 'shift'] = kk[6:] if 'trans' in kk else kk[6:]

sns.catplot(data=dfm,x='shift', y='error', col='ei', kind='boxen', col_wrap=2, dodge=True)
