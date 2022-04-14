import matplotlib.pyplot as plt, pandas as pd, seaborn as sns
import torchio as tio, torch, time
from segmentation.config import Config
from segmentation.run_model import RunModel
from nibabel.viewers import OrthoSlicer3D as ov
import glob, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, numpy.linalg as npl
pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1, 'display.width', 400)
from util_affine import perform_motion_step_loop, product_dict, create_motion_job, select_data, corrupt_data, apply_motion
import nibabel as nib
from read_csv_results import ModelCSVResults
from types import SimpleNamespace
from kymatio import HarmonicScattering3D
from types import SimpleNamespace
sns.set_style("darkgrid")
from test_motion_on_synth3 import *

disp_str_list = ['no_shift', 'center_zero', 'demean', 'demean_half' ] # [None 'center_zero', 'demean']
disp_str = disp_str_list[0];

xend = np.array([1,2,4,8,12,16,32,64,128]); xee = np.sort(-xend)
xend = np.hstack([xee+256,256, xend+256])
#xend = np.array([128, 384])

mvt_axe_all = [[0],[1],[2]] # #[[0],[1],[2],[0,1,2], [3,4,5], [7] ]#[[0,1,2], [3,4,5]], #, [7] ],#[[1], [4]], #[[0,1,2], ], #[[5],[6]],  #[[0,1,2]], # [[4],[1]] ,
#amp3 = list(np.sqrt(np.array(amp)**2/3))
amp = [1,2,4,8]; sigmas = [  2.,   4.,   6.,   8.,  12.,  16., 22., 32,  48.,  58., 64.,  76.,  86.,  98.,  108., 128., 138.,  148.,  158.,  168., 188., 208., 228., 256.] #
# now done in coupt_data  ... if len(mvt_axe==3  amp3 = list(np.sqrt(np.array(amp)**2/3))
all_params = dict(
    amplitude=amp,
    sigma = [ 4,  8,  16,  32, 64, 128], #sigmas, # np.linspace(2,256,128).astype(int), # [2, 4,  8,  16,  32, 44, 64, 88, 128], ,
    nb_x0s = [19], #[16], #[65],
    x0_min = [0], # [0],
    #x0_step = [xend] , #[8], # this give max(sigma/x0_step (then x0_min is not used,  start from center
    xend_steps = [list(xend)] ,
    sym = [False],#[ True, False],
    mvt_type= ['Ustep'],
    mvt_axe = mvt_axe_all, #[[0],[2]],#[[0,1,2], [3,4,5]], #, [7] ],#[[1], [4]], #[[0,1,2], ], #[[5],[6]],  #[[0,1,2]], # [[4],[1]] ,
    cor_disp = [True,], #no more used
    suj_index = [475,500], #[475, 500], #[ 475,  478, 492, 497, 499, 500], #[474, 475, 477, 478, 485, 492, 497, 499, 500, 510],
    suj_seed = [None],# [0, 1, 2, 3, 5], #[0,1,2,4,5,6,7,8,9],  [0,2,4,7,9]
    suj_contrast = [1,3], #[1, 3],
    suj_deform = [ True],
    suj_noise = [0], #[0.01 ],
    displacement_shift_strategy = ['nodisp', '1D_wTF','1D_wTF2'], #['nodisp'], #
    #freq_encoding_dim = [0,1,2]
)
#all_params['suj_index'] = [0]
# all_params['sigmas'] = [4,  8,  16,  32, 64, 128]
#amplitudes, sigmas, nb_x0s, x0_min = [1,2,4,8], [4,  8,  16, 24, 32, 40, 48, 64, 72, 80, 88, 128] ,21, 120;
#syms, mvt_types, mvt_axes, cor_disps = [True, False], ['Ustep'], [[0],[1],[3]], [False,]

#params = list(product(amplitudes,sigmas, syms, mvt_types, mvt_axes, cor_disps, disp_strs, nb_x0s,x0_min))
params = product_dict(**all_params)
nb_x0s = all_params['nb_x0s']; nb_sim = len(params) * nb_x0s[0]
print(f'performing loop of {nb_sim} iter 10s per iter is {nb_sim*10/60/60} Hours {nb_sim*10/60} mn ')
print(f'{nb_x0s[0]} nb x0 and {len(params)} params')
resolution=218; #512

#df1, res, res_fitpar = perform_motion_step_loop(params)

file = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/test1/main.json'
res_name = 'transX_ustep_sym_nosym';res_name = 'transX_ustep_sym_10suj'
res_name = 'transX_ustep_65X0_5suj_10seed_3contrast_2deform'
res_name = 'noise_transX_ustep_65X0_1suj_5seed_3contrast_2deform'
res_name = 'noise_abs_transX_ustep_65X0_1suj_5seed_3contrast_2deform'
res_name = 'constant_motion_noise_7suj_5seed_3contrast_2deform_3noise'
res_name = 'constant_motion_noise_abs_7suj_5seed_3contrast_2deform_3noise'
res_name = 'new_noise_abs_transX_ustep_65X0_1suj_5seed_3contrast_2deform'
res_name = 'x0_256_2noise_abs_transX_ustep_5suj_5seed_3contrast_2deform'
res_name = 'sigma_2-256_x0_256_4noise_abs_transX_ustep_2suj_5seed_2contrast_deform'
res_name = 'rot4_sigma_2-256_x0_256_4noise_abs_transX_ustep_2suj_5seed_2contrast_deform'
res_name = 'fsl_coreg_rot_trans_sigma_2-256_x0_256_suj_0'
res_name = 'fsl_coreg_along_x0_transXYZ_suj_0'
res_name = 'fsl_coreg_along_x0_rot_origY_suj_0'
res_name = 'fsl_coreg_along_x0_rotZ_origY_suj_2_con'
res_name = 'fsl_coreg_along_xend_transY_plusmoins_suj_0_def_con'
res_name = 'fsl_coreg_along_xend_transX_Z_XYZ_suj_0_def_con'
res_name = 'fsl_coreg_along_xend_rotXYZ_oy2_suj_0_def_con'
res_name = 'fsl_coreg_along_xend_trans_FreAxe2'
res_name = 'fsl_coreg_along_xend_transZ_disp_shift'
res_name = 'fsl_coreg_along_xend_all_3_trans_3_shift_2c'

out_path = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion_elastix/' + res_name

create_motion_job(params,10,file,out_path,res_name, type='motion_loop')


#read results
df1 = read_csv_motion(out_path, corect_initial_disp=False)

df1 = pd.read_csv('fsl_coreg_along_xend_transX_Z_XYZ_suj_0_def_con.csv')
df1=compute_disp(df1, normalize_amplitude=True)


outlocal = '/data/romain/PVsynth/motion_on_synth_data/fsl_coreg_along_xend_all_suj_0_def_con/'
df1.to_csv(outlocal+'res_all_2ROT_3trans.csv')  #does not work on lustre !!!
dflist = [df1, df2]
for dff in dflist:
    dff = compute_disp(dff, normalize_amplitude=True)
    dff['direction'] = dff.mvt_axe;
    dff.direction.replace('transY', 'translation Y', inplace=True);dff.direction.replace('transXtransYtransZ', 'translation XYZ', inplace=True);dff.direction.replace('rotY', 'rotation Y', inplace=True);dff.direction.replace('rotXrotYrotZ', 'rotation XYZ', inplace=True);
    dff['amp_mvt'] = dff.direction + ' A ' + dff.amplitude.astype(str)
    value_name='Displacement / Amplitude ';
    mixt_variable = ['shift_rotN' ,'shift_transN']; style_order = [ '|Trans|','|Rot|']

df1 = df1.rename({'m_L1_map':'L1', 'm_NCC':'NCC', 'm_grad_nMI2':'MI', 'm_PSNR':'PSNR','m_nRMSE':'nL2',
                 'm_ssim_SSIM':'SSIM', 'm_grad_ratio':'grad_ratio','m_grad_cor_diff_ratio':'auto_cor_ratio'}, axis='columns')

dfall  = pd.concat([df1,df2], ignore_index=True);  # dfall = df1

df2.sort_values(by=['xend', 'amplitude', 'sigma', 'suj_index', 'mvt_axe', 'suj_contrast'], inplace=True)

# sinon le no_shift mais qui est presque equivalen le wTF == wTF_short mais pas les 2 wTF2 == wshaw2 !!!
# for i in range(6):
#     k = f'wTF_Disp_{i}'; df1[k]=df1[k] + df1[f'shift_{i}']
#     k = f'wTF2_Disp_{i}'; df1[k]=df1[k] + df1[f'shift_{i}']

#transform disp_to vec see test_motion_on_synth3.py

### plot displacement
ykey='shift_rotN'
#along x0
cmap = sns.color_palette(n_colors= len(df1.sigma.unique()))
fig = sns.relplot(data=dfsub, x="xend", y=ykey, hue="sigma", legend='full', kind='line',palette=cmap, col='amplitude',col_wrap=2, ci=100)
#along sigma
cmap = sns.color_palette(n_colors= len(df1.amplitude.unique()))
fig = sns.relplot(data=dfsub, x="sigma", hue="amplitude", y=ykey, legend='full', kind='line',palette=cmap,  ci=100)

value_name='rotation (degre) '; shift_type='rotN'
value_name='translation (mm)'; shift_type='transN'
mixt_variable = [k for k in df1.keys() if shift_type in k];
del mixt_variable[0:8];  del mixt_variable[2] # ['wTF_rotN', 'wTF2_rotN', 'srot']
hue_order = [f'shift_{shift_type}', f'center_{shift_type}', f'mean_{shift_type}', f'wTFshort_{shift_type}', f'wSH_{shift_type}', f'wSH2_{shift_type}']
mixt_variable = ['no_shift_wTF_transN', 'no_shift_wTF2_transN', 'no_shift_wTFshort_transN', 'no_shift_wTFshort2_transN', 'no_shift_wSH_transN',
 'no_shift_wSH2_transN', 'no_shift_mean_transN', 'no_shift_center_transN', 'shift_transN']
mixt_variable = ['no_shift_wTF_transN', 'no_shift_wSH_transN',
 'no_shift_wSH2_transN', 'no_shift_mean_transN', 'no_shift_center_transN']

dfsub = df1[(df1.mvt_axe=='rotZ') ] ; # df1[df1.mvt_axe=='transX'] ;
dfsub = df1[(df1.mvt_type=='Ustep_sym') & (df1.sigma>12)]# & (df1.amplitude>0.6)] ; # df1[df1.mvt_axe=='transX'] ;
dfsub = df1[(df1.mvt_axe=='oy2') & (df1.sigma>12)] ; # df1[df1.mvt_axe=='transX'] ;
dfsub=df1
# different sigma x0 center
dfm = dfsub.melt(id_vars=['fp','sigma', 'amplitude','mvt_axe'], value_vars=mixt_variable, var_name='mean', value_name=value_name)
fig = sns.relplot(data=dfm, x="sigma", y=value_name,hue='mean', legend='full', kind="line", style='amplitude',
                  col='mvt_axe',col_wrap=2, hue_order=hue_order)
# different x0
dfm = dfsub.melt(id_vars=['fp','sigma', 'amplitude','mvt_axe','x0','xend'], value_vars=mixt_variable, var_name='mean', value_name=value_name)
fig = sns.relplot(data=dfm, x="xend", y=value_name,hue='mean', legend='full', kind="line", col='sigma', col_wrap=2, hue_order=hue_order,style='amplitude')
plt.text(0.4,0.5,'transXYZ ',transform=plt.gcf().transFigure,bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})

#Figure
#plot fsl_coreg_rot_trans_sigma_2-256_x0_256_suj_0 or newer res fsl_coreg_along_sigma_rot_trans_suj_0_def_con
#center 256 many sigma


ykey='shift_rotN' # ykey='shift_transN'
cmap = sns.color_palette("coolwarm", len(dfsub.suj_index.unique()))
cmap = sns.color_palette(n_colors=len(dfsub.amplitude.unique()))
fig = sns.relplot(data=df1, x="sigma", y=ykey, hue="amplitude", legend='full', kind="line",palette=cmap,
                  col='direction',col_wrap=2)

dfm = dfsub.melt(id_vars=['sigma', 'amplitude','direction', xend], value_vars=mixt_variable, var_name='Displacement', value_name=value_name)
dfm['Displacement'].replace('shift_rotN','|Rot|', inplace=True); dfm['Displacement'].replace('shift_transN','|Trans|', inplace=True)
fig = sns.relplot(data=dfm, x="sigma", y=value_name,hue='amplitude', legend='full', kind="line",
                  style='Displacement',palette=cmap, col='direction',col_wrap=4,
                  style_order=style_order, ci=100,
                  col_order=['translation Y','translation XYZ', 'rotation Y' , 'rotation XYZ' ])
#save as ~/datal/PVsynth/job/motion/figure/resume/displacement_along_sigma_trans_rot_fsl_shift.png

#same for Xend but hue is needed for sigma, so we need one dim more
dfsub = df1[(df1.mvt_type=='Ustep') & (df1.mvt_axe != 'oy2' )]
dfsub = df1[((df1.mvt_axe=='oy2') | (df1.mvt_axe=='rotXrotYrotZ') )& (df1.mvt_type=='Ustep') ]
dfsub = df1[(df1.displacement_shift_strategy=='nodisp') & ((df1.amplitude==1) |(df1.amplitude==8)) &
            ((df1.mvt_axe=='transX')|(df1.mvt_axe=='transZ')) & (df1.suj_contrast==1) ]

dfsub=compute_disp(dfsub.copy(), normalize_amplitude=True)

cmap = sns.color_palette(n_colors=len(dfsub.sigma.unique()))
col_order = np.sort(dfsub.amp_mvt.unique())

dfsub = df1[(df1.displacement_shift_strategy=='nodisp') & ((df1.amplitude==1) |(df1.amplitude==8)) & (df1.suj_contrast==1)]
ykeys = ['L1','nL2','NCC','MI','PSNR','SSIM', 'grad_ratio',
         'no_shift_L1_map_brain', 'no_shift_nRMSE_brain', 'no_shift_ssim_SSIM_brain']
ykeys = ['no_shift_PSNR_brain','no_shift_l2ns_SSIM_brain']
for ykey in ykeys:
    fig = sns.relplot(data=dfsub, x="xend", y=ykey, hue="sigma", legend='full', kind="line", palette=cmap,
                      col='amp_mvt', col_wrap=4, ci=100, col_order=col_order)  # , marker='*')
    figname = f'm_C1_{ykey}.svg'
    fig.savefig(figname)
    plt.close()


fig = sns.relplot(data=dfsub, x="xend", y=ykey, hue="sigma", legend='full', kind="line", palette=cmap,
                  col='amp_mvt', col_wrap=4, ci=100, col_order=col_order )#, marker='*')
#or direction with style ...
style_order = np.sort(dfsub.direction.unique())
fig = sns.relplot(data=dfsub, x="xend", y=ykey, hue="sigma", legend='full', kind="line", palette=cmap,
                  style='direction', ci=100, col='amplitude',col_wrap=2, style_order=style_order)#, marker='*')
#dfsub['ind'] =  dfsub.suj_index+dfsub.suj_contrast
 # units='ind', estimator=None
dfsub.direction.replace('transZ','Z',inplace=True)
fig.set_ylabels('Displacement / Amplitude'); fig.axes[1].set(ylim=[-0.05 , 1.05])


#wTF metrics
dfsub = df1[(df1.mvt_type=='Ustep_sym')& (df1.mvt_axe=='rotXrotYrotZ') ]
mixt_variable=['no_shift_wSH_rotN',  'no_shift_wSH2_rotN',  'no_shift_center_rotN',  'no_shift_mean_rotN']
mixt_variable=['no_shift_wSH_transN',  'no_shift_wSH2_transN',  'no_shift_center_transN',  'no_shift_mean_transN']
dfm = dfsub.melt(id_vars=['sigma', 'amplitude','direction', 'xend', 'mvt_type'], value_vars=mixt_variable,
                 var_name='predicted_shift', value_name=value_name)
dfm['predicted_shift'].replace('no_shift_wSH_transN', 'wTF', inplace=True);
dfm['predicted_shift'].replace('no_shift_wSH2_transN', 'wTF2', inplace=True);
dfm['predicted_shift'].replace('no_shift_center_transN', 'center', inplace=True);
dfm['predicted_shift'].replace('no_shift_mean_transN', 'mean', inplace=True);

fig = sns.relplot(data=dfm, x="xend", y=value_name, hue="sigma", legend='full', kind="line", palette=cmap,
                  col='predicted_shift',col_wrap=3 ,   ci=100)

#wTF and fsl shift
mixt_variable = ['shift_transN', 'no_shift_wSH2_transN']
value_name='Displacement / Amplitude'
dfm = dfsub.melt(id_vars=['sigma', 'amplitude','direction', 'xend', 'mvt_type'], value_vars=mixt_variable, var_name='Displacement', value_name=value_name)
fig = sns.relplot(data=dfm, x="xend", y=value_name, hue="sigma", legend='full', kind="line", palette=cmap,
                  col='amplitude',col_wrap=2, style = 'Displacement' ,   ci=100)

dfsub.shift_diff = dfsub.shift_transN - dfsub.no_shift_wSH2_transN
fig = sns.relplot(data=dfsub, x="xend", y=ykey, hue="sigma", legend='full', kind="line", palette=cmap,
                  ci=100, col='amplitude',col_wrap=2)#, marker='*')

mixt_variable = ['no_shift_wTF_transN', 'no_shift_wTF_rotN','no_shift_wTFshort2_transN','no_shift_wTFshort2_rotN','no_shift_wTFshort_transN','no_shift_wTFshort_rotN']
#mixt_variable += ['no_shift_wTF2_transN', 'no_shift_wTF2_rotN','no_shift_wTFshort_transN','no_shift_wTFshort_rotN'] ]# TF2 == SH2 TFshort==TF !
mixt_variable = ['no_shift_wSH_transN', 'no_shift_wSH_rotN', 'no_shift_wSH2_transN', 'no_shift_wSH2_rotN', ]
mixt_variable += ['no_shift_center_transN', 'no_shift_center_rotN', 'no_shift_mean_transN', 'no_shift_mean_rotN', ]
#mixt_variable=['no_shift_wSH_transN',  'no_shift_wSH2_transN',  'no_shift_center_transN',  'no_shift_mean_transN']

#dfs = df1[(df1.mvt_axe=='rotXrotYrotZ')| (df1.mvt_axe=='transXtransYtransZ')]
dfm = dfs.melt(id_vars=['sigma', 'amplitude','direction'], value_vars=mixt_variable, var_name='Displacement', value_name=value_name)
dfm['predicted_shift'] = dfm.Displacement
for mv in mixt_variable:
    if 'rotN' in mv:
        str_rep = mv[9:-5]; print(f'rep {mv} -> {str_rep} |rot|')
        dfm['Displacement'].replace(mv, '|Rot|', inplace=True);
        dfm['predicted_shift'].replace(mv, str_rep, inplace=True);
    if 'transN' in mv:
        str_rep = mv[9:-7];
        print(f'rep {mv} -> {str_rep} |trans|')
        dfm['Displacement'].replace(mv, '|Trans|', inplace=True);
        dfm['predicted_shift'].replace(mv, str_rep, inplace=True);


cmap = sns.color_palette(n_colors=len(dfm.predicted_shift.unique()))
fig = sns.relplot(data=dfm, x="sigma", y=value_name,hue='predicted_shift', legend='full', kind="line",
                  palette=cmap, col='direction',col_wrap=2,style='Displacement',style_order=style_order,
                  ci=100)
fig = sns.relplot(data=dfm, x="sigma", y=value_name,hue='predicted_shift', legend='full', kind="line",
                  palette=cmap,style_order=style_order,
                  ci=100)

### L1 metrics
dfsub = df1[(df1.mvt_type=='Ustep')& (df1.mvt_axe=='rotXrotYrotZ')&(df1.suj_contrast==1) ]
dfsub=dfsub.copy()
mixt_variable = [k for k in df1.keys() if 'L1_map_brain' in k];
#mixt_variable=['m_L1_map','no_shift_L1_map']

for k in mixt_variable:
    #dfsub[k] = dfsub[k]/dfsub['amplitude'] # dfsub[k]/np.sqrt(dfsub['amplitude'])
    dfsub[k] = dfsub.groupby('amplitude')[k].transform(lambda x: x/x.max())

dfm = dfsub.melt(id_vars=['sigma', 'amplitude','xend'], value_vars=mixt_variable, var_name='shift_cor', value_name='L1')
k='L1'; dfm[k] = dfm.groupby('amplitude')[k].transform(lambda x: x / x.max())

fig = sns.relplot(data=dfm, x="xend", hue="sigma", col="amplitude", y='L1', legend='full', kind='line',
                   palette=cmap,  ci=100, col_wrap=2, style='shift_cor')

#diff L1 before and after coreg
dfsub = df1[ (df1.suj_contrast==1) & ((df1.amplitude==1) | (df1.amplitude==8)) & ((df1.mvt_axe=='transX') | (df1.mvt_axe=='transZ'))].copy() # compute_disp(df1)
dfsub = df1[ (df1.suj_contrast==1) & ((df1.mvt_axe=='transX') | (df1.mvt_axe=='transZ'))].copy() # compute_disp(df1)
#dfs1 = dfsub[(dfsub.displacement_shift_strategy=="nodisp")].copy()
dfs2 = dfsub[(dfsub.displacement_shift_strategy=="1D_wTF") ].copy()
dfs3 = dfsub[(dfsub.displacement_shift_strategy=="1D_wTF2")].copy()
#dfs1.sort_values(by=['xend', 'amplitude', 'sigma', 'suj_index'], inplace=True)
dfs2.sort_values(by=['xend', 'amplitude', 'sigma', 'suj_index'], inplace=True)
dfs3.sort_values(by=['xend', 'amplitude', 'sigma', 'suj_index'], inplace=True)
#dfs1.sort_values(by='fp', inplace=True);dfs2.sort_values(by='fp', inplace=True);dfs3.sort_values(by='fp', inplace=True)
dfsub =dfs2.copy()
dfsub['wTF'] = ( dfs2['no_shift_L1_map_brain'].values - dfs2['m_L1_map_brain'].values)/ dfs2['m_L1_map_brain'].values # dfs2['shift_transN'].values
dfsub['wTF2']= ( dfs3['no_shift_L1_map_brain'].values - dfs3['m_L1_map_brain'].values)/ dfs3['m_L1_map_brain'].values #  dfs3['shift_transN'].values

mixt_variable=['wTF','wTF2']; yvaluename = 'L1 diff %' #'shift error'
dfm = dfsub.melt(id_vars=['sigma', 'amplitude','xend'], value_vars=mixt_variable,
                 var_name='Disp', value_name=yvaluename)
def concat_str_col(s,k1,k2):
    return f'{s[k1]} | Amplitude = {s[k2]}'
dfm['Displacement'] = dfm.apply(lambda x: concat_str_col(x,'Disp','amplitude'), axis=1)

cmap = sns.color_palette(n_colors=len(dfsub.sigma.unique()));ykey = yvaluename
fig = sns.relplot(data=dfm, x="xend", hue="sigma", col="Displacement", y=ykey, legend='full',
                  kind='line', palette=cmap,  ci=100, col_wrap=4)

#scatter plot for random motion
dfa = pd.concat([df1, df2])
dfa['predicted shift']=dfa.displacement_shift_strategy
dfa['predicted shift'].replace('1D_wTF', 'wTF', inplace=True);
cmap = sns.color_palette(n_colors=len(dfsub.amplitude.unique()))
fig = sns.relplot(data=dfa, x='m_L1_map_brain', hue="amplitude", col="predicted shift", y='no_shift_L1_map_brain',
                  legend='full',kind='scatter', palette=cmap,  ci=100, col_wrap=2)
for ax in fig.axes:
    ax.plot([0,10],[0,10],'k--')

aa = dfsub['m_L1_map_brain']-dfsub['no_shift_L1_map_brain']
index_bad=dfsub[aa>1].index
for ii in index_bad:
    print(ii)
    print(f'outdir {dfsub.loc[ii,"out_dir"]} {dfsub.loc[ii,"m_L1_map_brain"]}  {dfsub.loc[ii, "no_shift_L1_map_brain"]}')
    #print(f'Suj_{dfsub.loc[ii,"subject_name"]}_I{dfsub.loc[ii,"suj_index"]}_C{dfsub.loc[ii,"suj_contrast"]}_SNone_Disp_{dfsub.loc[ii,"displacement_shift_strategy"]}_Amp{dfsub.loc[ii,"amplitude"]}_N')


### plot images metrics
df1 = df1.rename({'m_t1_L1_map':'L1', 'm_t1_NCC':'NCC', 'm_t1_grad_nMI2':'MI', 'm_t1_PSNR':'PSNR','m_t1_nRMSE':'nL2',
                 'm_t1_ssim_SSIM':'SSIM', 'm_t1_grad_ratio':'grad_ratio','m_t1_grad_cor_diff_ratio':'auto_cor_ratio'}, axis='columns')
df1 = df1.rename({'m_L1_map':'L1', 'm_NCC':'NCC', 'm_grad_nMI2':'MI', 'm_PSNR':'PSNR','m_nRMSE':'nL2',
                 'm_ssim_SSIM':'SSIM', 'm_grad_ratio':'grad_ratio','m_grad_cor_diff_ratio':'auto_cor_ratio'}, axis='columns')

df1 = dfall[dfall['mvt_type']=='Ustep_sym']
df1 = df1[df1['xend']==256]

dfsub = df1[df1.mvt_axe=='transY'] ; # df1[df1.mvt_axe=='transX'] ;
dfsub = dfsub[dfsub.mvt_type=='Ustep'] #dfsub[dfsub.mvt_type=='Ustep_sym']
dfsub = df1[df1.suj_contrast==1]; dfsub = dfsub[dfsub.suj_noise==0.01];
dfsub = df1[ (df1.amplitude==4) & (df1.suj_contrast==1)];
ds1 = dfsub[(dfsub.sigma==64) & (dfsub.x0==256) & (dfsub.suj_deform==1)]; ds1.shape
dfsub_dict = split_df_unique_vals(df1,['amplitude'])
dfsub_dict = split_df_unique_vals(df1,['suj_noise', 'suj_deform', 'suj_contrast'])
dfsub = split_df_unique_vals(df1,['suj_noise', 'suj_contrast'])

ykeys = ['L1','nL2','NCC','MI','PSNR','SSIM', 'grad_ratio']#,'auto_cor_ratio']
ykeys = ['L1','nL2','NCC','MI','PSNR','SSIM', 'grad_ratio','auto_cor_ratio',
         'shift', 'm_t1_grad_EGratio','m_t1_grad_Eratio', 'm_t1_grad_scat', 'm_t1_grad_cor1_ratio',
         'm_t1_lab_mean2','m_t1_lab_mean0','m_t1_lab_std2','m_t1_lab_std0',]
ykeys = ['m_mean_DispP', 'shift', 'm_wTF_Disp_1', 'm_wTF2_Disp_1', 'm_mean_DispP', 'm_meanDispP_wTF', 'm_wTF_absDisp_t',
'm_t1_grad_Eratio', 'm_t1_grad_ratio_bin', 'm_t1_grad_cor1_ratio', 'm_t1_grad_cor3_ratio']

ykeys=['m_t1_L1_map', 'm_t1_NCC', 'm_t1_ssim_SSIM', 'm_t1_PSNR','m_t1_grad_ratio','m_t1_grad_ratio_bin','m_t1_lab_std0','m_t1_lab_mean0',
         'm_t1_nRMSE', 'm_t1_grad_nMI2', 'm_t1_grad_EGratio','m_t1_grad_cor_diff_ratio','m_t1_grad_cor2_ratio'] #'m_t1_grad_Eratio' is null with noise !!!
#ykeys = ['m_t1_PSNR', 'm_t1_grad_ratio']  # ,'m_t1_nRMSE'] #,'m_t1_L1_map'] m_t1_grad_grad_ration
ykeys=['mean_DispP', 'm_meanDispP_wTF', 'm_meanDispP_wTF2', 'm_wTF_Disp_1', 'm_wTF2_Disp_1',]
ykeys=[];
for k in df1.keys():
    if 'm_t1_lab_' in  k: ykeys.append(k)
ykeys = ykeys[1:6]

figres = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/figure/roty_center_sigma/'
prefix = 'abs_noise' #'sigma_x256' #'abs_noise_colAmp' # 'contrast_'
for k, dfsub in dfsub_dict.items():
    cmap = sns.color_palette("coolwarm", len(dfsub.sigma.unique()))
    #cmap = sns.color_palette("coolwarm", len(dfsub.suj_index.unique()))
    for ykey in ykeys:
        if ykey in dfsub:
            fig = sns.relplot(data=dfsub, x="xend", y=ykey, hue="sigma", legend='full', kind="line",palette=cmap, col='amplitude',col_wrap=2)
            #fig = sns.relplot(data=dfsub, x="xend", y=ykey, hue="sigma", legend='full', kind="line",palette=cmap, col='suj_contrast',col_wrap=2,style='suj_deform')
            #fig = sns.relplot(data=dfsub, x="sigma", y=ykey, hue="suj_index", legend='full', kind="line", palette=cmap, col='amplitude', col_wrap=2)

            for aa in fig.axes: aa.grid()
            plt.text(0.1,0.9,ykey,transform=plt.gcf().transFigure,bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
            plt.text(0.1,0.8,k,transform=plt.gcf().transFigure,bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
            figname = f'{prefix}_{ykey}_{k}.png'
            fig.savefig(figres+figname)
            plt.close()
        else:
            print(f'missing keys {ykey}')

ykeys = ['L1','nL2','NCC','MI','PSNR','SSIM', 'grad_ratio', 'm_t1_grad_EGratio']#,'auto_cor_ratio']
sns.pairplot(df1[ykeys], kind="scatter", corner=True)


plt.figure();plt.scatter(df1.L1,df1.nL2); plt.xlabel('L1'); plt.ylabel('nL2')
ind = (df1.nL2<0.155) & (df1.nL2>0.15); np.sum(ind)
di = df1.loc[ind,:]
il1min,il1max = di.L1.argmin(), di.L1.argmax()
dmin, dmax = di.iloc[il1min,:],di.iloc[il1max,:]



#show metric when constant translation
dfsub = df1[ (df1.mvt_axe=='transY')];
cmap = sns.color_palette("coolwarm", len(dfsub.suj_noise.unique()))
prefix = 'constant_disp_transY' # 'contrast_'

ykey=ykeys[0]
for ykey in ykeys:
    fig = sns.catplot(data=dfsub, x="amplitude", y=ykey, hue="suj_noise",  col='suj_contrast', height=4, aspect=.9)
    for aa in fig.axes[0]: aa.grid()
    figname = f'{prefix}_{ykey}.png'
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

json_file='/data/romain/PVsynth/motion_on_synth_data/test1/main.json'
df=pd.DataFrame()

param = {'amplitude': 4, 'sigma': 64, 'nb_x0s': 1, 'x0_min': 206, 'sym': False, 'mvt_type': 'Ustep',
 'mvt_axe': [1], 'cor_disp': False, 'disp_str': 'no_shift', 'suj_index': 0, 'suj_seed': 1, 'suj_contrast': 1,
 'suj_deform': False, 'suj_noise' : 0.01}
pp = SimpleNamespace(**param)
dfone = dmin #dmax #dmin
for k in param.keys():
    if k in dfone:
        print(f'{k} : {dfone[k]}')
        param[k] = dfone[k]
    else:
        print(f'no {k} taking 0')
        param[k] = 0;

x0 = int(dfone['x0']); param['suj_seed'] = int(param['suj_seed']);param['suj_index'] = int(param['suj_index'])
param['sigma'] = int(param['sigma']) ; param['mvt_axe'] = [1]
pp = SimpleNamespace(**param)

amplitude, sigma, sym, mvt_type, mvt_axe, cor_disp, disp_str, nb_x0s, x0_min =  pp.amplitude, pp.sigma, pp.sym, pp.mvt_type, pp.mvt_axe, pp.cor_disp, pp.disp_str, pp.nb_x0s, pp.x0_min
#resolution, sigma, x0 = 512, int(sigma), 256
extra_info = param; extra_info['mvt_axe']=1

ssynth, tmot, mr = select_data(json_file, param)
fp = corrupt_data(x0, sigma=sigma, method=mvt_type, amplitude=amplitude, mvt_axes=mvt_axe,center='none', return_all6=True, sym=sym, resolution=resolution)
smot, df, res_fitpar, res = apply_motion(ssynth, tmot, fp, df,pd.DataFrame(),pd.DataFrame(), extra_info, config_runner=mr,correct_disp=cor_disp)


param['suj_noise'] = 0.01
s7, tmot, mr = select_data(json_file, param)

smot7, df, res_fitpar, res = apply_motion(s7, tmot, fp, df,pd.DataFrame(),pd.DataFrame(), extra_info, config_runner=mr,correct_disp=cor_disp)

param['suj_noise'] = 0.1
s8, tmot, mr = select_data(json_file, param)
fp = corrupt_data(x0, sigma=sigma, method=mvt_type, amplitude=amplitude, mvt_axes=mvt_axe,center='none', return_all6=True, sym=sym, resolution=resolution)
smot8, df, res_fitpar, res = apply_motion(s8, tmot, fp, df,pd.DataFrame(),pd.DataFrame(), extra_info, config_runner=mr,correct_disp=cor_disp)

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

tnoise = tio.RandomNoise(std=(0.05, 0.05))
ssynth = tnoise(ssynth)

#motion and save ex
resolution=512
dout = '/data/romain/data_exemple/motion_step2/'
for ii,x0 in enumerate( np.arange(356,150,-10) ):
    fp = corrupt_data(x0, sigma=1, method='step', amplitude=4, mvt_axes=[1], center='none', return_all6=True, resolution=resolution)
    fig = plt.figure(); plt.plot(fp.T);
    fig.savefig(dout+f'step_{ii:03d}_x0_{x0}.png')
    plt.close()

    tmot.nT = fp.shape[1];    tmot.simulate_displacement = False
    tmot.fitpars = fp;    tmot.displacement_shift_strategy = None
    sout = tmot(ssynth)
    sout.t1.save(dout + f'v_{ii:03d}_step_x0{x0:03d}.nii')


dout = '/data/romain/data_exemple/motion_const/'
fp = corrupt_data(x0, sigma=1, method='Const', amplitude=0.1, mvt_axes=[1], center='none', return_all6=True, resolution=resolution)
tmot.nT = fp.shape[1];    tmot.simulate_displacement = False
tmot.fitpars = fp;    tmot.displacement_shift_strategy = None;tmot.metrics['grad'].metric_kwargs=None
df=pd.DataFrame()
tnoise = tio.RandomNoise(std=(0.01, 0.01))
ssynth = tnoise(ssynth)
del(sout)
for ii in range(10):
    if 'sout' not in locals():
        sout = tmot(ssynth)
    else:
        sout, df, res_fitpar, res = apply_motion(sout, tmot, fp, df,pd.DataFrame(),pd.DataFrame(), extra_info, config_runner=mr,correct_disp=cor_disp)
    sout.t1.save(dout + f'n00v_y06rot_fft_{ii:03d}.nii')

del(sout)
tt = tio.Affine(scales=(1,1,1),degrees=(0,0.6,0), translation = (0,0,0))
for ii in range(10):
    if 'sout' not in locals():
        sout = tt(ssynth)
    else:
        sout = tt(sout)
    sout.t1.save(dout + f'n05v_y06rot_aff_{ii:02d}.nii')



tmot = tio.RandomMotionFromTimeCourse(displacement_shift_strategy='center', correct_motion=True,
                                      oversampling_pct = 0,  preserve_center_frequency_pct=0.1)
tmot._simulate_random_trajectory(); tmot.simulate_displacement=False; tmot.fitpars[:3,:]=0;
smot = tmot(ssynth)

tmot.fitpars = - tmot.fitpars
smotinv = tmot(smot)
smot.t1.save('mot.nii'); smotinv.t1.save('motinv.nii');ssynth.t1.save('orig.nii');
smotinv.t1['data'] = smot.t1['data_cor']; smotinv.t1.save('mot_complex_inv.nii')

#read fmri fitparts
csv_file = '/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cati_full/delivery_new/description_filter.csv'
df = pd.read_csv(csv_file)
fpars_paths = df["Fitpars_path"]




#not working, too bad ... from torchio.transforms.augmentation.intensity.torch_random_motion import TorchRandomMotionFromTimeCourse
#volume=tt.permute(1,2,3,0).numpy()
#v= nib.Nifti1Image(volume,affine); nib.save(v,'/tmp/t.nii')
fjson = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/test1/main.json' # '/data/romain/PVsynth/motion_on_synth_data/test1/main.json'
param = dict();param['suj_contrast'] = 1;param['suj_noise'] = 0.01;param['suj_index'] = 0;param['suj_deform'] = 0;param['displacement_shift_strategy']=None
sdata, tmot, config_runner = select_data(fjson, param, to_canonical=False)
image = sdata.t1.data[0]; brain_mask = sdata.brain.data[0]
fi = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)

df1['fp'] =  [os.path.dirname(ff)  + '/fitpars_orig.txt'  for ff in csv_file]
### recompute motion metrics
for ii, fp_path in enumerate(df1.fp.values):
    fitpar = np.loadtxt(fp_path)
    tmot = tio.transforms.MotionFromTimeCourse(frequency_encoding_dim=0,fitpars=fitpar, tr=1, es=0.001, oversampling_pct=0, displacement_shift_strategy=None)
    tmot._calc_dimensions(image.shape,0)
    f_interp = np.zeros([6] + list(image.shape))
    tmot._compute_motion_metrics(fitpar, f_interp, fi)
    res = tmot._metrics
    for k in res.keys():
        if 'Disp_' in k :
            df1.loc[ii, k] = res[k]

dfall = pd.DataFrame(); sujcon =3;
for k in range(500):
    param['suj_index'] = k;
    param['suj_contrast'] = sujcon;
    param['suj_deform'] = 1;
    sdata, tmot, config_runner = select_data(fjson, param, to_canonical=False)
    image = sdata.t1.data[0]
    fi = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)

    coef_shaw = np.sqrt(np.sum(abs(fi) ** 2, axis=(0, 2)));
    coef_shaw /= np.sum(coef_shaw)
    dff = pd.DataFrame(coef_shaw, columns=['wFT'])
    dff['ky'] = range(coef_shaw.shape[0])
    dff['wFT2'] = coef_shaw**2 / np.sum(coef_shaw**2)
    dff['suj_index'] = k;
    dff['suj_contrast'] = sujcon
    dfall = pd.concat([dfall, dff])

dfm = dfall.melt(id_vars=['suj_contrast','ky','suj_index'], value_vars=['wFT','wFT2'],
                 var_name ='weight', value_name='weigths')
dfm.ky = dfm.ky*512/218

fig = sns.relplot(data=dfm, x="ky", y='weigths', hue='weight',  legend='full', kind="line", ci=100,
                  units='suj_index', estimator=None)
                   err_style='bars')

def scale_ky():

