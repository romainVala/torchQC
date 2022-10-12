
import torch, torchio as tio
import torch.nn.functional as F

import glob
import pandas as pd
from segmentation.config import Config
from segmentation.utils import to_numpy
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D as ov

device='cuda'
nb_test_transform = 3

fsuj_csv = pd.read_csv('/data/romain/baby/suj_hcp_feta_T2_5test_suj.csv')

model_dir = '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/jzay/training/baby/'
models=[]
models.append(model_dir + 'bin_dseg9_5suj_hcp_T2/results_cluster_nextela/model_ep1_it640_loss11.0000.pth.tar')
models.append(model_dir + 'bin_dseg9_5suj_hcp_T2/results_cluster_nextela_fromAffBig/model_ep1_it640_loss11.0000.pth.tar')

model = models[1]
fin = fsuj_csv.vol_name[0]
flab = fsuj_csv.label_name[0]

model_struct = {'module': 'unet',    'name': 'UNet',    'last_one': False,    'path': model,    'device': device }
config = Config(None, None, save_files=False)
model_struct = config.parse_model_file(model_struct)
model, device = config.load_model(model_struct)
model.eval()

suj = tio.Subject({'t1': tio.ScalarImage(fin), 'label': tio.ScalarImage(flab)})
treshape = tio.EnsureShapeMultiple(2**4)
tscale = tio.RescaleIntensity(out_min_max=(0,1), percentiles=(0,99))
tpreproc = tio.Compose([treshape, tscale])
suj = tpreproc(suj)

test_transfo = tio.RandomAffine(scales=0.1, degrees=180 , translation=0)

for nb_transfo in range(nb_test_transform):
    sujt = test_transfo(suj)
    with torch.no_grad():
        prediction = model(sujt.t1.data.unsqueeze(0).float().to(device))
        prediction = F.softmax(prediction, dim=1)

