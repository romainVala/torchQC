import matplotlib.pyplot as plt, pandas as pd, seaborn as sns
import torchio as tio, torch, time
from segmentation.config import Config
from segmentation.run_model import RunModel
from nibabel.viewers import OrthoSlicer3D as ov
import glob, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1, 'display.width', 400)
from util_affine import perform_motion_step_loop, product_dict, create_motion_job
import nibabel as nib
from read_csv_results import ModelCSVResults
from types import SimpleNamespace
from kymatio import HarmonicScattering3D

def show_df(df, keys):
    rout, rout_unique = dict(),dict()
    if isinstance(keys,dict):
        keys = keys.keys()
    for k in keys:
        vars = df[k].unique()
        if len(vars)==1:
            rout_unique[k] = vars
        else:
            vars = np.sort(vars)
            rout[k] = vars
    print('unique values')
    for k, v in rout_unique.items():
        print(f'  {k} \t: {v}')
    print('multiple values')
    for k, v in rout.items():
        print(f'  {k} \t: {v}')

    #return rout, rout_unique
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


#not working, too bad ... from torchio.transforms.augmentation.intensity.torch_random_motion import TorchRandomMotionFromTimeCourse
#volume=tt.permute(1,2,3,0).numpy()
#v= nib.Nifti1Image(volume,affine); nib.save(v,'/tmp/t.nii')

file='/data/romain/PVsynth/motion_on_synth_data/test1/main.json'
file = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/test1/main.json'
result_dir='/data/romain/PVsynth/motion_on_synth_data/test1/rrr/'
config = Config(file, result_dir, mode='eval')
config.init()
mr = config.get_runner()
mr.epoch=4

s1 = config.train_subjects[0]
transfo_list = config.train_transfo_list

suj_name = ['715041', '713239', '770352', '749058', '765056', '766563'] #3/3 min/max GM_vol
suj_name +=['707749','765056','789373','704238','733548','770352']  #3/3 min/max vol_mm_tot
indsuj = [474, 475, 477, 478, 485, 492, 497, 499, 500, 510]
#     '704238', '707749', '713239', '715041', '733548', '749058', '765056', '766563', '770352', '789373']
#for ii,ss in enumerate(config.train_subjects):    if  ss.name in suj_name:        indsuj.append(ss.name)

#same label, random motion
tsynth = transfo_list[0]
tsynth.mean = [(0.6, 0.6), (0.1, 0.1), (1, 1), (0.6, 0.6), (0.6, 0.6), (0.6, 0.6), (0.6, 0.6), (0.6, 0.6), (0.6, 0.6), (0.6, 0.6),
 (0.9, 0.9), (0.6, 0.6), (1, 1), (0.2, 0.2), (0.4, 0.4), (0, 0)]
ssynth = tsynth(s1)
tmot = transfo_list[1]


#same label random step gaussian motion
shifts = range(-15, 15, 1)
l1_loss = torch.nn.L1Loss()

disp_str_list = ['no_shift', 'center_zero', 'demean', 'demean_half' ] # [None 'center_zero', 'demean']
disp_str = disp_str_list[0];
mvt_axe_str_list = ['transX', 'transY','transZ', 'rotX', 'rotY', 'rotZ']

all_params = dict(
    amplitude=[1,2,4,8],
    sigma =  [2, 4,  8,  16,  32, 64, 128], #[2, 4,  8,  16,  32, 44, 64, 88, 128], #np.linspace(4,128,60),
    nb_x0s = [65],
    x0_min = [0],
    sym = [False, ],
    mvt_type= ['Ustep'],
    mvt_axe = [[1]] ,
    cor_disp = [True,],
    disp_str =  ['no_shift'],
    suj_index = [ 475,  478, 492, 497, 499, 500], #[474, 475, 477, 478, 485, 492, 497, 499, 500, 510],
    suj_seed = [0,2,4,7,9], #[0,1,2,4,5,6,7,8,9],
    suj_contrast = [1, 2,3],
    suj_deform = [False, True,],
    suj_nois = [0.01, 0.1]

)
all_params['suj_index'] = [0]

# all_params['sigmas'] = [4,  8,  16,  32, 64, 128]
#amplitudes, sigmas, nb_x0s, x0_min = [1,2,4,8], [4,  8,  16, 24, 32, 40, 48, 64, 72, 80, 88, 128] ,21, 120;
#syms, mvt_types, mvt_axes, cor_disps = [True, False], ['Ustep'], [[0],[1],[3]], [False,]

#params = list(product(amplitudes,sigmas, syms, mvt_types, mvt_axes, cor_disps, disp_strs, nb_x0s,x0_min))
params = product_dict(**all_params)
nb_x0s = all_params['nb_x0s']; nb_sim = len(params) * nb_x0s[0]
print(f'performing loop of {nb_sim} iter 10s per iter is {nb_sim*10/60/60} Hours {nb_sim*10/60} mn ')
print(f'{nb_x0s[0]} nb x0 and {len(params)} params')
resolution=512

#df1, res, res_fitpar = perform_motion_step_loop(params)

file = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/test1/main.json'
res_name = 'transX_ustep_sym_nosym';res_name = 'transX_ustep_sym_10suj'
res_name = 'transX_ustep_65X0_5suj_10seed_3contrast_2deform'
res_name = 'noise_transX_ustep_65X0_1suj_5seed_3contrast_2deform'
out_path = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/' + res_name

create_motion_job(params,1,file,out_path,res_name)

mres = ModelCSVResults(df_data=df1, out_tmp="/tmp/rrr")
keys_unpack = ['T_LabelsToImage'];
suffix = ['l']
df1 = mres.normalize_dict_to_df(keys_unpack, suffix=suffix);

csv_file = glob.glob(out_path+'/res_metrics*')
df = [pd.read_csv(ff) for ff in csv_file]
df1 = pd.concat(df, ignore_index=True); dfall = df1
show_df(df1,all_params)

df1['m_wTF_Disp_1']=df1['m_wTF_Disp_1'] + df1['shift']
df1['m_wTF2_Disp_1']=df1['m_wTF2_Disp_1'] + df1['shift']

df1 = dfall[dfall['mvt_type']=='Ustep_sym']
df1 = df1[df1['xend']==256]

dfsub = df1[df1.mvt_axe=='transY'] ; # df1[df1.mvt_axe=='transX'] ;
dfsub = dfsub[dfsub.mvt_type=='Ustep'] #dfsub[dfsub.mvt_type=='Ustep_sym']
dfsub = df1[df1.suj_contrast==3]; dfsub = dfsub[dfsub.suj_deform==1];
dfsub = df1[ (df1.amplitude==4) & (df1.suj_contrast==1)];
ds1 = dfsub[(dfsub.sigma==64) & (dfsub.x0==256) & (dfsub.suj_deform==1)]; ds1.shape
dfsub_dict = split_df_unique_vals(df1,['amplitude'])
dfsub = split_df_unique_vals(df1,['suj_seed'])

ykeys=['m_t1_L1_map', 'm_t1_NCC', 'm_t1_ssim_SSIM', 'm_t1_PSNR','m_t1_grad_ratio','m_t1_lab_std0','m_t1_lab_mean0',
         'm_t1_nRMSE', 'm_t1_grad_nMI2', 'm_t1_grad_Eratio','m_t1_grad_EGratio','m_t1_grad_ratio']
#ykeys = ['m_t1_PSNR', 'm_t1_grad_ratio']  # ,'m_t1_nRMSE'] #,'m_t1_L1_map'] m_t1_grad_grad_ration
'mean_DispP', 'm_meanDispP_wTF', 'm_meanDispP_wTF2', 'm_wTF_Disp_1', 'm_wTF2_Disp_1',

figres = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/figure/'
prefix = 'contrast_'
for k, dfsub in dfsub_dict.items():
    cmap = sns.color_palette("coolwarm", len(dfsub.sigma.unique()))
    for ykey in ykeys:
        #fig = sns.relplot(data=dfsub, x="xend", y=ykey, hue="sigma", legend='full', kind="line",palette=cmap, col='amplitude',col_wrap=2)
        fig = sns.relplot(data=dfsub, x="xend", y=ykey, hue="sigma", legend='full', kind="line",palette=cmap, col='suj_contrast',col_wrap=2,style='suj_deform')
        for aa in fig.axes: aa.grid()
        plt.text(0.8,0.3,ykey,transform=plt.gcf().transFigure,bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        plt.text(0.8,0.2,k,transform=plt.gcf().transFigure,bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        figname = f'{prefix}_{ykey}_{k}.png'
        fig.savefig(figres+figname)
        plt.close()

for suj in df1['sujname'].unique():
    dfsub = df1[ df1['sujname'] == suj ]
    fig = sns.relplot(data=dfsub, x="xend", y=ykey, hue="sigma", legend='full', kind="line",palette=cmap, col='amplitude',col_wrap=2)
    for aa in fig.axes: aa.grid()

fig = sns.relplot(data=dfsub, x="xend", y=ykey, hue="sigma", legend='full', kind="line",palette=cmap, col='suj_seed',col_wrap=3, style='suj_deform')
for aa in fig.axes: aa.grid()

cmap = sns.color_palette("coolwarm", len(df1.suj_seed.unique()))
fig = sns.relplot(data=dfsub[dfsub['sigma']==64], x="xend", y=ykey, hue="suj_seed", legend='full', kind="line", col='amplitude',col_wrap=2)

sns.relplot(data=df1, x="xend", y="m_t1_L1_map", hue="sigma", legend='full', kind="line",palette=cmap, col='amplitude',col_wrap=2)
fig=sns.relplot(data=df1, x="shift", y="m_wTF_Disp_1", hue="sigma", legend='full', kind="scatter",palette=cmap, col='amp',col_wrap=2)
for aa in fig.axes:
    aa.plot([0,10], [0,10])
sns.relplot(data=df1, x="xend", y="shift", hue="sigma", legend='full', kind="line")
sns.relplot(data=res, x="vox_disp", y="L1", hue="x0", legend='full', col="sigma", kind="line", col_wrap=2)
sns.relplot(data=res_fitpar, x="nbt", y="trans_y", hue="x0", legend='full', col="sigma", kind="line", col_wrap=2)
sns.relplot(data=df1, x="xend", y="m_wTF_Disp_1", hue="sigma", legend='full', kind="line")

rsub = res_fitpar[res_fitpar['mvt_type']=='Ustep'];
rsub = res_fitpar[res_fitpar['xend']==250]; rsub = rsub[rsub['sigma']==64]; rsub = rsub[rsub['amp']==4];rsub = rsub[rsub['mvt_axe']=='transY'];
sns.relplot(data=rsub, x="nbt", y="trans_y", legend='full', col="mvt_type", kind="line")

#variing sigma at center
cmap = sns.color_palette("coolwarm", len(df1.amplitude.unique()))
sns.relplot(data=df1, x="sigma", y="m_t1_L1_map", hue="amplitude", legend='full', kind="line",palette=cmap)
fig=sns.relplot(data=df1, x="shift", y="m_wTF_Disp_1", hue="sigma", legend='full', kind="scatter",palette=cmap, col='amp',col_wrap=2)
for aa in fig.axes:
    aa.plot([0,10], [0,10])

sel_key=['m_wTF_Disp_1', 'm_wTF2_Disp_1', 'shift']
sel_key=['m_t1_L1_map', 'm_t1_NCC', 'm_t1_ssim_SSIM', 'm_t1_PSNR','m_t1_grad_ratio','m_t1_lab_std0',
         'm_meanDispP_wTF2','m_rmse_Disp_wTF2']
sel_key=['m_t1_PSNR', 'm_t1_NCC', 'm_t1_grad_Eratio','m_t1_grad_EGratio','m_t1_grad_ratio']
sel_key=['m_t1_L1_map', 'm_t1_NCC', 'm_t1_ssim_SSIM', 'm_t1_PSNR','m_t1_grad_ratio','m_t1_lab_std0','m_t1_nRMSE']
fig = sns.pairplot(df1[sel_key], kind="scatter", corner=True)

cmap = sns.color_palette("coolwarm", len(df1.sigma.unique()))
fig = sns.relplot(data=dfsub, x="shift", y='m_wTF_Disp_1', hue="sigma", legend='full', kind="scatter",palette=cmap, col='amplitude',col_wrap=2,style='suj_seed')
cmap = sns.color_palette("coolwarm", len(df1.amplitude.unique()))
fig = sns.relplot(data=dfsub, x="shift", y='m_wTF_Disp_1', hue="amplitude", legend='full', kind="scatter",palette=cmap,
                  col='sigma',col_wrap=2,style='suj_seed')
for aa in fig.axes:
    aa.plot([0,8], [0,8]); aa.grid()


d= result_dir + '/'
f1= d+ 'Train_ep001.csv'
f1 = d + 'Train_random_label_one_small_motion.csv'
f2 = d + 'Train_ep003.csv'
f2 = d + 'Train_random_label_affine_one_small_motion.csv'
f2 = d + 'Train_one_contrast_radom_motion_small.csv'

mres = ModelCSVResults(f1,  out_tmp="/tmp/rrr")
mres2 = ModelCSVResults(f2,  out_tmp="/tmp/rrr")
keys_unpack = ['T_RandomLabelsToImage','T_RandomMotionFromTimeCourse_metrics_t1','T_RandomAffine', 'T_RandomMotionFromTimeCourse']
suffix = ['Tl', '', 'Tr', 'Tm']
keys_unpack = ['T_Affine','T_ElasticDeformation', 'T_Noise','T_RandomMotionFromTimeCourse', 'T_BiasField','T_LabelsToImage']
suffix = ['Taff', 'Tela','Tnoi', 'Tmot', 'Tbias', 'Tlab']
df1 = mres.normalize_dict_to_df(keys_unpack, suffix=suffix);

df1 = mres.normalize_dict_to_df(keys_unpack, suffix=suffix); #df1 = df1.rename(columns = {"sample_time":"meanS"})
df2 = mres2.normalize_dict_to_df(keys_unpack, suffix=suffix); #df2 = df2.rename(columns = {"sample_time":"meanS"})

keys = df1.keys()
sel_key=[]
for k in keys:
    #if k.find('_metri')>0:
    if k.endswith('SSIM'):
        print(k); sel_key.append(k)
sel_key = ['luminance_SSIM', 'structure_SSIM', 'contrast_SSIM', 'ssim_SSIM', 'L1','NCC' ]
sel_key = ['nL2e', 'pSNR', 'metric_ssim_old', 'L1_map', 'NCC', 'meanS', 'Tm_mean_DispP']
sel_key = ['nL2e', 'pSNRm', 'L1', 'L2', 'NCC', 'ssim_SSIM','l2ns_SSIM'] #'meanS'
sel_key = ['ssim_SSIM', 'ssim_SSIM_brain', 'NCC', 'NCC_brain','contrast_SSIM', 'contrast_SSIM_brain' ]
sel_key =  ['Tm_mean_DispP', 'Tm_rmse_Disp', 'Tm_meanDispP_wTF2', 'Tm_rmse_Disp_wTF2', 'NCC']

#sur le graphe one motion different contrast,
#L1 L2 pSNR tres correle
#nL2e nL2m pSNRm tres correle (nL2m et pSNRm sont tres tres correle) NCC plus proche de nL2e

sns.pairplot(df1[sel_key], kind="scatter", corner=True)
sns.pairplot(df2[sel_key], kind="scatter", corner=True)

plt.scatter(df1['L1_map'], df1['NCC'])
plt.figure();plt.scatter(df1['metric_ssim_old'], df1['ssim_SSIMr'])
plt.figure();plt.scatter(df1['SSIM_ssim_SSIM'], df1['NCC'])
plt.figure();plt.scatter(df1['SSIM_ssim_SSIM'], df1['SSIM_contrast_SSIM'])
plt.figure();plt.scatter(df1['SSIM_ssim_SSIM'], df1['SSIM_structure_SSIM'])
plt.figure();plt.scatter(df1['SSIM_ssim_SSIM'], df1['SSIM_luminance_SSIM'])
plt.figure();plt.scatter(df1['NCC'], df1['NCC_c'])
plt.figure();plt.scatter(df1['SSIM_contrast_SSIM'], df1['metric_ssim_old'])


mres.scatter('L1_map','NCC')





mr.epoch=3; df = pd.DataFrame()
for i in range(5):
    start = time.time()
    s=ssynth
    sm = tmot(s)
    batch_time = time.time() - start
    df, reporting_time = mr.record_regression_batch( df, sm, torch.zeros(1).unsqueeze(0), torch.zeros(1).unsqueeze(0),
                                                     batch_time, save=True)

    print(f'{i} in {batch_time} repor {reporting_time}')



main_structure = config.parse_main_file(file)
transform_structure = config.parse_transform_file(main_structure['transform'], return_string=True)



transfo = transform_structure['train_transforms']
st = transfo(s1)
hist = st.history
hist[4][1].pop('_metrics')
trsfm_hist, seeds_hist = tio.compose_from_history(history=hist)


trsfm_hist[0].get_inverse = True
colin_back = trsfm_hist[0](transformed, seed=seeds_hist[0])


data = ssynth.t1.data
data_shape = data.shape[1:]
M, N, O =  128, 128, 128
J = 2
L = 2
integral_powers = [1., 2.]
sigma_0 = 1
scattering = HarmonicScattering3D(J, shape=data_shape, L=L, sigma_0=sigma_0)
scattering.method = 'integral'
scattering.integral_powers = integral_powers
s=time.time()
res = scattering(data)
s=time.time()-s

ldata = ssynth.label.data
ind = ldata>0.5
#ldata[ind]=1;ldata[~ind]=0
data = ssynth.t1.data
meanlab = [data[li].mean() for li in ind]
stdlab = [data[li].std() for li in ind]

data = ssynth.t1.data[0]
datas = smot.t1.data[0]
from scipy.signal import fftconvolve
dd = data[::-1,::-1,::-1]
rr = fftconvolve(data,data)

############## CLUSTERING

ykeys=['m_t1_L1_map', 'm_t1_NCC', 'm_t1_ssim_SSIM', 'm_t1_PSNR','m_t1_grad_ratio','m_t1_nRMSE','m_t1_grad_Eratio','m_t1_grad_EGratio','m_t1_grad_ratio']
dsub = df1.loc[:,ykeys]
data = dsub.to_numpy()
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
 # A list holds the SSE values for each k
sse = [];silhouette_coefficients = []
kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42 }
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)
    score = silhouette_score(data, kmeans.labels_)
    silhouette_coefficients.append(score)
plt.figure();plt.plot(sse)
plt.figure();plt.plot(silhouette_coefficients)

from sklearn.decomposition import PCA, KernelPCA
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(data)
X_back = kpca.inverse_transform(X_kpca)
pca = PCA()
X_pca = pca.fit_transform(data)

from sklearn import manifold
tsne = manifold.TSNE(n_components=3, init='random',random_state=0, perplexity=50)
Y = tsne.fit_transform(data)

#and so what ???





##############retrieve one suj
from util_affine import select_data, corrupt_data, apply_motion
from types import SimpleNamespace

json_file='/data/romain/PVsynth/motion_on_synth_data/test1/main.json'
df=pd.DataFrame()

param = {'amplitude': 4, 'sigma': 64, 'nb_x0s': 65, 'x0_min': 0, 'sym': False, 'mvt_type': 'Ustep',
 'mvt_axe': [1], 'cor_disp': True, 'disp_str': 'no_shift', 'suj_index': 0, 'suj_seed': 1, 'suj_contrast': 1,
 'suj_deform': False}
pp = SimpleNamespace(**param)

amplitude, sigma, sym, mvt_type, mvt_axe, cor_disp, disp_str, nb_x0s, x0_min =  pp.amplitude, pp.sigma, pp.sym, pp.mvt_type, pp.mvt_axe, pp.cor_disp, pp.disp_str, pp.nb_x0s, pp.x0_min
resolution, sigma, x0 = 512, int(sigma), 212
extra_info = param; extra_info['mvt_axe']=1

ssynth, tmot, mr = select_data(json_file, param)
fp = corrupt_data(x0, sigma=sigma, method=mvt_type, amplitude=amplitude, mvt_axes=mvt_axe,center='none', return_all6=True, sym=sym, resolution=resolution)
smot, df, res_fitpar, res = apply_motion(ssynth, tmot, fp, df,pd.DataFrame(),pd.DataFrame(), extra_info, config_runner=mr,correct_disp=cor_disp)


param['suj_seed'] = 7
s7, tmot, mr = select_data(json_file, param)
fp = corrupt_data(x0, sigma=sigma, method=mvt_type, amplitude=amplitude, mvt_axes=mvt_axe,center='none', return_all6=True, sym=sym, resolution=resolution)
smot7, df, res_fitpar, res = apply_motion(s7, tmot, fp, df,pd.DataFrame(),pd.DataFrame(), extra_info, config_runner=mr,correct_disp=cor_disp)


mres = ModelCSVResults(df_data=df, out_tmp="/tmp/rrr")
keys_unpack = ['transforms_metrics', 'm_t1'];
suffix = ['m', 'm_t1']
df1 = mres.normalize_dict_to_df(keys_unpack, suffix=suffix);


for i,pp in enumerate(params):
    all_equal=True
    for k,v in param.items():
        if not (v== pp[k]):
            all_equal=False
    if all_equal:
        print(i)


############################ contrast
label_list = ['GM', 'CSF', 'WM',  'L_Accu','L_Amyg', 'L_Caud', 'L_Hipp', 'L_Pall', 'L_Puta', 'L_Thal',  'BrStem', 'cereb_GM','cereb_WM', 'skin',  'skull', 'background']
PV = ssynth.label.data
pv_lin = PV.flatten(start_dim=1)
pv_inter = torch.matmul(pv_lin,pv_lin.transpose(1,0))
pv_inter_norm = torch.zeros_like(pv_inter)
pv_inter[0]
for i, l in enumerate(label_list):
    #print('{} vol {} {}'.format(l, torch.sum(PV[i]), torch.sum(pv_inter[i])))
    inter_vol = pv_inter[i] / torch.sum(pv_inter[i]) * 100
    inter_vol[inter_vol < 0.01] = 0
    pv_inter_norm[i] = inter_vol; pv_inter_norm[i,i] = 0
    ind_pv = torch.where(inter_vol>0)[0].numpy()
    strr=f'{l}\t '
    for found_ind, ii in enumerate(ind_pv):
        strr += f' {label_list[ii]}  {inter_vol[ii]}'
    print(strr)


############## autocorrelation
data=ssynth.t1.data[0]
datam = smot.t1.data[0]

c1,c2,c3,cdiff = get_autocor(data,nb_off_center = 5)
c1m,c2m,c3m,cdiffm = get_autocor(datam,nb_off_center = 3)
c1/c1m, c2/c2m, c3/c3m, cdiffm/cdiff

plt.figure(); plt.plot(x_dis, y_cor); plt.plot(x_dism, y_corm);
x=unique_dist
plt.plot(x, m*x + b)