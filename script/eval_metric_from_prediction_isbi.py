
import torch,numpy as np,  torchio as tio
from utils_metrics import get_tio_data_loader, predic_segmentation, load_model
from timeit import default_timer as timer
import json, os, seaborn as sns

import pandas as pd
from nibabel.viewers import OrthoSlicer3D as ov


sns.set_style("darkgrid")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

labels_name = np.array(["bg","vessel",])
selected_index = [1]
selected_label = torch.zeros(labels_name.shape); selected_label[selected_index]=1; selected_label = selected_label>0
labels_name = labels_name[selected_index]

scale_to_volume = 0 # 350000
device='cuda'
nb_test_transform = 0

#fsuj_csv = pd.read_csv('/data/romain/ISBI2023Challenges/SHINY-ICARUS/test_s7.csv')
fsuj_csv = pd.read_csv('/data/romain/ISBI2023Challenges/SMILE-UHURA/test_3suj.csv')
t1_column_name = "vol"; label_column_name = "label"; sujname_column_name = "sujname"

resname = 'suj_test_spline'; #'suj_old'#'hcp10' #mar_12suj_tpm_masked_hast'#'res_Flip' #'hcp_next' #'res_Aff_suj80' #'suj_mar_from_template_tru_masked'
result_dir = '/data/romain/ISBI2023Challenges/SHINY-Eval/' + resname
result_dir = '/data/romain/ISBI2023Challenges/SMILE-UHURA/eval/' + resname

save_data = 2
#result_dir = None

model_dir = '/data/romain/ISBI2023Challenges/training/bin_2class_onData/'
model_dir = '/data/romain/ISBI2023Challenges/training_jzay/'
models, model_name = [], []
#models.append(model_dir + 'result_reslice/model_ep60_it360_loss11.0000.pth.tar');    model_name.append('CTA_reslice_ep60')
models.append(model_dir + '/MRA_bin_unet/results_cluster_r05mm/model_ep120_it180_loss11.0000.pth.tar'); model_name.append('MRA_r0_ep120')


for mm in models:
    if not os.path.exists(mm):
        print(f'WARNING model {mm} does not exist ')

tresamp = tio.Resample(target=0.5)
thot = tio.OneHot()
treshape = tio.EnsureShapeMultiple(2 ** 4)
tscale = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 99))

if nb_test_transform:
    test_transfo = tio.RandomAffine(scales=0.1, degrees=180, translation=0, default_pad_label=0)
    test_transfor_name = 'rdAff'
    test_transfo = tio.RandomAnisotropy(axes=[0,1,2], downsampling=[3, 6])
    test_transfor_name = 'rdAniso'
    test_transfo = tio.RandomAffine(scales=0.1, degrees=10, translation=0, default_pad_label=0)
    test_transfor_name = 'Aff10'
    test_transfo = tio.RandomFlip(axes=[0, 1, 2], flip_probability=1)
    test_transfor_name = 'Flip'

    tpreproc = tio.Compose([treshape, tscale, thot, test_transfo])
else:
    tpreproc = tio.Compose([tresamp, treshape, tscale, thot])
    test_transfor_name = 'none'


tio_data = get_tio_data_loader(fsuj_csv, tpreproc, replicate=nb_test_transform ,get_dataset=True,
                               t1_column_name=t1_column_name, label_column_name=label_column_name, sujname_column_name=sujname_column_name)

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

df = pd.DataFrame()

for nb_model, model_path in enumerate(models):
    model = load_model(model_path, device)
    for suj in tio_data:
        suj_name = suj.name if isinstance(suj,tio.Subject) else suj["name"][0]
        print(f'Suj {suj_name}')

        if scale_to_volume :
            head_volume = ((suj.label.data[1:,...]>0).sum() * 0.5**3 ).numpy()

            scale_factor = (scale_to_volume/head_volume)**(1/3)
            trescale = tio.Affine(scales=scale_factor,degrees=0, translation=0)
            suj = trescale(suj)

        res_dict = {'sujname': suj_name, 'model_name':  model_name[nb_model], 'test_transfo': test_transfor_name, 'model_path': model_path}

        df = predic_segmentation(suj, model, df, res_dict, device, labels_name, selected_label=selected_label,
                                 out_dir=result_dir, save_data=save_data, resample_back=True)

#df.to_csv(f'res_all_metric_hcp80_2model_mot.csv')

df.to_csv(f'res_{resname}_3dice.csv')

