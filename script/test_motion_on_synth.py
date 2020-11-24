import matplotlib.pyplot as plt, pandas as pd
import torchio as tio, torch
from segmentation.config import Config
from segmentation.run_model import RunModel
from nibabel.viewers import OrthoSlicer3D as ov
import nibabel as nib

#volume=tt.permute(1,2,3,0).numpy()
#v= nib.Nifti1Image(volume,affine); nib.save(v,'/tmp/t.nii')


file='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion_New/train_random_synth/test/main.json'
result_dir='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion_New/train_random_synth/test/rrr/'

config = Config(file, result_dir, mode='eval')
config.init()
mr = config.get_runner()

s1 = config.train_subjects[0]


transfo_list = config.train_transfo_list
df = pd.DataFrame()
for i in range(500):
    s=s1
    for ii, t in enumerate(transfo_list):
        s = t(s, seed=555) if ii==1 else  t(s)
    df, batch_time = mr.record_regression_batch( df, s, torch.zeros(1).unsqueeze(0), torch.zeros(1).unsqueeze(0), 1, save=True)
    print(i)

main_structure = config.parse_main_file(file)
transform_structure = config.parse_transform_file(main_structure['transform'], return_string=True)



transfo = transform_structure['train_transforms']
st = transfo(s1)
hist = st.history
hist[4][1].pop('_metrics')
trsfm_hist, seeds_hist = tio.compose_from_history(history=hist)


trsfm_hist[0].get_inverse = True
colin_back = trsfm_hist[0](transformed, seed=seeds_hist[0])





import numpy as np
def interpolate_fitpars(fpars, tr_fpars, tr_to_interpolate=2.4):
    fpars_length = fpars.shape[1]
    x = np.asarray(range(fpars_length))*tr_fpars
    xp = np.asarray(range(250))*tr_to_interpolate
    interpolated_fpars = np.asarray([np.interp(xp, x, fp) for fp in fpars])
    return interpolated_fpars

data_to_interpolate = np.loadtxt("/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cati_full/delivery_new/MEMENTO/Nantes/0180020FAJE/M024/fitpars.txt")
tr_fpars = 2659.05151367187/1000
interpolated = interpolate_fitpars(data_to_interpolate, tr_fpars)

plt.figure()
plt.plot(np.asarray(range(data_to_interpolate.shape[1]))*tr_fpars, data_to_interpolate.T)
plt.title(f"Fpars original: TR: {tr_fpars} nb_pts: {data_to_interpolate.shape[1]}")

plt.figure()
plt.plot(np.asarray(range(250))*2.4, interpolated.T)
plt.title(f"Fpars interpolé: TR: {2.4} nb_pts: {interpolated.shape[1]}")