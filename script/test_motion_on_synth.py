import matplotlib.pyplot as plt, pandas as pd, seaborn as sns
import torchio as tio, torch
from segmentation.config import Config
from segmentation.run_model import RunModel
from nibabel.viewers import OrthoSlicer3D as ov
import glob, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1, 'display.width', 400)

import nibabel as nib

#volume=tt.permute(1,2,3,0).numpy()
#v= nib.Nifti1Image(volume,affine); nib.save(v,'/tmp/t.nii')

def corrupt_data( x0, sigma= 5, amplitude=20, method='gauss', mvt_axes=[1] ):
    fp = np.zeros((6, 200))
    x = np.arange(0,200)
    if method=='gauss':
        y = np.exp(-(x - x0) ** 2 / float(2 * sigma ** 2))*amplitude
    elif method == 'step':
        if x0<100:
            y = np.hstack((np.zeros((1,(x0-sigma))),
                            np.linspace(0,amplitude,2*sigma+1).reshape(1,-1),
                            np.ones((1,((200-x0)-sigma-1)))*amplitude ))
        else:
            y = np.hstack((np.zeros((1,(x0-sigma))),
                            np.linspace(0,-amplitude,2*sigma+1).reshape(1,-1),
                            np.ones((1,((200-x0)-sigma-1)))*-amplitude ))
    elif method == 'sin':
        fp = np.zeros((6, 182*218))
        x = np.arange(0,182*218)
        y = np.sin(x/x0 * 2 * np.pi)
        #plt.plot(x,y)

    for xx in mvt_axes:
        fp[xx,:] = y
    return fp

torch.manual_seed(12)

file='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion_New/train_random_synth/test/main.json'
result_dir='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion_New/train_random_synth/test/rrr/'
#file = '/data/romain/PVsynth/ex_transfo/pve_synth_drop01_aug_mot/main.json'
#result_dir = '/data/romain/PVsynth/ex_transfo/pve_synth_drop01_aug_mot/rrr'
config = Config(file, result_dir, mode='eval')
config.init()
mr = config.get_runner()

s1 = config.train_subjects[0]
transfo_list = config.train_transfo_list
df = pd.DataFrame()

#same motion random label
for i in range(500):
    s=s1
    for t in transfo_list:
        if isinstance(t, tio.transforms.augmentation.composition.OneOf): #skip affine elastic
            continue
        if isinstance(t, tio.transforms.augmentation.intensity.RandomMotionFromTimeCourse):
            s = t(s, seed=5555)
        else:
            s = t(s)
        if isinstance(t, tio.transforms.augmentation.intensity.random_labels_to_image.RandomLabelsToImage):
            mean_S = s.t1.data.mean().numpy()
    df, batch_time = mr.record_regression_batch( df, s, torch.zeros(1).unsqueeze(0), torch.zeros(1).unsqueeze(0), mean_S, save=True)
    print(i)

#same label, random motion
trl = transfo_list[0]
trl.mean = [(0.6, 0.6), (0.1, 0.1), (1, 1), (0.6, 0.6), (0.6, 0.6), (0.6, 0.6), (0.6, 0.6), (0.6, 0.6), (0.6, 0.6), (0.6, 0.6),
 (0.9, 0.9), (0.6, 0.6), (1, 1), (0.2, 0.2), (0.4, 0.4), (0, 0)]
mr.epoch=3
df = pd.DataFrame()
for i in range(500):
    s=s1
    for t in transfo_list:
        if isinstance(t, tio.transforms.augmentation.composition.OneOf): #skip affine elastic
            continue
        s = t(s)
    df, batch_time = mr.record_regression_batch( df, s, torch.zeros(1).unsqueeze(0), torch.zeros(1).unsqueeze(0), 1, save=True)
    print(i)

#same label random step gaussian motion
shifts, dimy = range(-15, 15, 1), 218
l1_loss = torch.nn.L1Loss()
out_path = '/data/romain/data_exemple/motion_synt_gaus/'
amplitudes = [1, 2 ,4, 10, 20]
positions = [10, 30, 50, 60, 70, 80, 90, 95, 100] #[ 90, 95, 99];
sigmas = [2, 4, 8, 12, 20]
nb_sim = len(amplitudes)* len(positions) * len(sigmas)
print("Nb simu {}".format(nb_sim))
disp_str_list = ['no_shift', 'center_zero', 'demean', 'demean_half' ] # [None 'center_zero', 'demean']
disp_str = disp_str_list[0];
mvt_types=['step', 'gauss']
mvt_type =mvt_types[1]
mvt_axe_str_list = ['transX', 'transY','transZ', 'rotX', 'rotY', 'rotZ']
mvt_axes = [1]
mvt_axe_str = mvt_axe_str_list[mvt_axes[0]]
mr.epoch=4
df, res, res_fitpar, extra_info, i = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), dict(), 0

for amplitude in amplitudes:
    for sigma in sigmas:
        for x0 in positions:
            s = s1
            fp = corrupt_data(x0, sigma=sigma, method=mvt_type, amplitude=amplitude, mvt_axes=mvt_axes)
            for t in transfo_list:
                if isinstance(t, tio.transforms.augmentation.composition.OneOf):  # skip affine elastic
                    continue
                if isinstance(t, tio.transforms.augmentation.intensity.RandomMotionFromTimeCourse):
                    t.nT = fp.shape[1]
                    t.simulate_displacement = False
                    t.fitpars = fp
                    t.displacement_shift_strategy = 'center_zero'
                    data_ref = s.t1.data
                    s = t(s)
                    break
                s = t(s)
            df, batch_time = mr.record_regression_batch(df, s, torch.zeros(1).unsqueeze(0), torch.zeros(1).unsqueeze(0),
                                                        1, save=True)

            fout = out_path + '/{}_{}_{}_A{}_s{}_freq{}_{}'.format('synthT1', mvt_axe_str, mvt_type, amplitude, sigma, x0, disp_str)
            fit_pars = t.fitpars - np.tile(t.to_substract[..., np.newaxis],(1,200))
            #fig = plt.figure();plt.plot(fit_pars.T);plt.savefig(fout+'.png');plt.close(fig)
            #s.t1.save(fout+'.nii')

            extra_info['x0'], extra_info['mvt_type'], extra_info['mvt_axe']= x0, mvt_type, mvt_axe_str
            extra_info['shift_type'], extra_info['sigma'], extra_info['amp'] = disp_str, sigma, amplitude
            extra_info['disp'] = np.sum(t.to_substract)

            dff = pd.DataFrame(fit_pars.T); dff.columns = ['x', 'trans_y', 'z', 'r1', 'r2', 'r3']; dff['nbt'] = range(0,200)
            for k,v in extra_info.items():
                dff[k] = v
            res_fitpar = res_fitpar.append(dff, sort=False)

            data = s.t1.data
            for shift in shifts:
                if shift < 0:
                    d1 = data[:, :, dimy + shift:, :]
                    d2 = torch.cat([d1, data[:, :, :dimy + shift, :]], dim=2)
                else:
                    d1 = data[:, :, 0:shift, :]
                    d2 = torch.cat([data[:, :, shift:, :], d1], dim=2)
                extra_info['L1'] , extra_info['vox_disp'] = float(l1_loss(data_ref, d2).numpy()), shift
                res = res.append(extra_info, ignore_index=True, sort=False)

            print('{} / {}'.format(i, nb_sim))
            i += 1

res.to_csv(out_path+'/res_shift.csv')
res_fitpar.to_csv(out_path + '/res_fitpars.csv')

pp = sns.relplot(data=res, x="vox_disp", y="L1", row='x0', col='sigma', kind='line')

from read_csv_results import ModelCSVResults
d= result_dir + '/'
f1= d+ 'Train_ep001.csv'
f1 = d + 'Train_random_label_one_small_motion.csv'
f2 = d + 'Train_ep002.csv'
f2 = d + 'Train_random_label_affine_one_small_motion.csv'
f2 = d + 'Train_one_contrast_radom_motion_small.csv'

mres = ModelCSVResults(f1,  out_tmp="/tmp/rrr")
mres2 = ModelCSVResults(f2,  out_tmp="/tmp/rrr")
keys_unpack = ['T_RandomLabelsToImage','T_RandomMotionFromTimeCourse_metrics_t1','T_RandomAffine', 'T_RandomMotionFromTimeCourse']
suffix = ['Tl', '', 'Tr', 'Tm']
df1 = mres.normalize_dict_to_df(keys_unpack, suffix=suffix); df1 = df1.rename(columns = {"sample_time":"meanS"})
df2 = mres2.normalize_dict_to_df(keys_unpack, suffix=suffix); df2 = df2.rename(columns = {"sample_time":"meanS"})

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








main_structure = config.parse_main_file(file)
transform_structure = config.parse_transform_file(main_structure['transform'], return_string=True)



transfo = transform_structure['train_transforms']
st = transfo(s1)
hist = st.history
hist[4][1].pop('_metrics')
trsfm_hist, seeds_hist = tio.compose_from_history(history=hist)


trsfm_hist[0].get_inverse = True
colin_back = trsfm_hist[0](transformed, seed=seeds_hist[0])




