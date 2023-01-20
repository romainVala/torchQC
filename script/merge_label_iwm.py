import torchio as tio
from utils_file import get_parent_path, gfile, gdir
import torch
from scipy.ndimage.morphology import  binary_erosion, binary_dilation
import os, pandas as pd

suj = gdir('/data/romain/baby/training_suj',['.*','ses','ana'])
sujp = gdir('/data/romain/PVsynth/eval_cnn/baby/Article/train/eval_T2train_model_15suj_ep75','.*')

flab = gfile(suj,'^nw_draw.*thin')
flab = gfile(suj,'PVlabel')
fpred = gfile(sujp, 'pred')
fout = [os.path.dirname(ff) + '/iWM' + os.path.basename(ff) for ff in flab]
datatype = 'float32'
iteration = 1

for f1, f2, f3 in zip(flab, fpred, fout):
    print(f'{f1} ')

    suj = tio.Subject({'lab': tio.LabelMap(f1)})
    tc = tio.CropOrPad(target_shape=suj.lab.shape[1:], include='pred')

    suj = tio.Subject({ 'pred': tio.LabelMap(f2)})
    sujt = tc(suj)
    sujt.add_image(tio.LabelMap(f1),'lab')

    if sujt.lab.data.shape[0] > 1: #4D label
        WMl = sujt.lab.data[[3]]
    else:
        WMl = sujt.lab.data==3
    GMp = sujt.pred.data==2
    GMp_in_WM = GMp * WMl
    skul_p = sujt.pred.data==4
    skull_in_WM = skul_p * WMl

    ino_wm = torch.zeros_like(GMp)
    ino_wm[GMp_in_WM>0] = 1 # GMp_in_WM[GMp_in_WM>0] # 1

    data = (ino_wm[0].numpy()).astype(datatype)
    data = binary_erosion(data, iterations=iteration).astype(datatype)
    data = binary_dilation(data, iterations=iteration).astype(datatype)
    data = torch.tensor(data);
    data = data.unsqueeze(0)

    if sujt.lab.data.shape[0] > 1: #4D label
        data[data>0] = WMl[data>0]
        sujt.lab.data = torch.cat([sujt.lab.data, data], axis=0)
    else:
        sujt.lab.data[data > 0] = 19
    print(f'GM in wm {data.sum()/WMl.sum()*100}')

    ino_wm = torch.zeros_like(GMp)
    ino_wm[skull_in_WM>0] = 1

    data = (ino_wm[0].numpy()).astype(datatype)
    data = binary_erosion(data, iterations=iteration).astype(datatype)
    data = binary_dilation(data, iterations=iteration).astype(datatype)
    data = torch.tensor(data);
    data = data.unsqueeze(0)
    if sujt.lab.data.shape[0] > 1: #4D label
        data[data>0] = WMl[data>0]
        sujt.lab.data = torch.cat([sujt.lab.data, data], axis=0)
    else:
        sujt.lab.data[data>0] = 20

    print(f'skul in wm {data.sum()/WMl.sum()*100}')

    if sujt.lab.data.shape[0] > 1: #remove WM label
        added_data = sujt.lab.data[[19]] + sujt.lab.data[[20]]
        WMl[added_data>0] = 0
        sujt.lab.data[3] = WMl[0]

    sujt.lab.save(f3)


#update
df=pd.read_csv('/data/romain/baby/suj_hcp_15_suj_jzay.csv')
fT2 = df.vol_T2.values
fnew = [os.path.dirname(ff) + '/iWMPVlabel.nii.gz' for ff in fT2]
df['label_mida_PV_iWM'] = fnew
fnew = [os.path.dirname(ff) + '/nw_drawem9_mida_ext_thin_dseg.nii.gz' for ff in fT2]
df['label_mida_thin'] = fnew
fnew = [os.path.dirname(ff) + '/iWMnw_drawem9_mida_ext_thin_dseg.nii.gz' for ff in fT2]
df['label_mida_thin_iWM'] = fnew
#fout = [os.path.dirname(ff) + '/iWM' + os.path.basename(ff) for ff in df.label_name]
#df['label_iWM'] = fout

df.to_csv('/data/romain/baby/suj_hcp_15_suj_jzay.csv', index=False)

df=pd.read_csv('/data/romain/baby/suj_hcp_15_suj_lustre.csv')
df.to_csv('/data/romain/baby/suj_hcp_15_suj_lustre.csv', index=False)

df=pd.read_csv('/data/romain/baby/suj_hcp_15_suj.csv')
df.to_csv('/data/romain/baby/suj_hcp_15_suj.csv', index=False)

nw drawem9

    index	name
1	CSF
2	Cortical gray matter
3	White matter
4	Background
5	Ventricles
6	Cerebellum
7	Deep Gray Matter
8	Brainstem
9	Hippocampi and Amygdala

BG,0
Air,10
dura_brain,11
eyes,12
muco,13
muscle,14
nerve,15
skin,16
skull,17
vessel,18

