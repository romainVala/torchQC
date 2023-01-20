
import torch,numpy as np,  torchio as tio
import json, os, seaborn as sns
import tqdm
import pandas as pd
from utils_file import get_parent_path, gfile, gdir

from utils_metrics import computes_all_metric, binarize_5D, get_results_dir

sns.set_style("darkgrid")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


labels_name=['GM']
selected_label = [1]
ind_pGM, ind_lGM = 1,0


ress, resname = get_results_dir('eval_T2')
ress, resname = get_results_dir('eval_T1')

ress, resname = ['/home/romain.valabregue/datal/PVsynth/eval_cnn/RES_prob_tissue/retest_HCP1mm_T2/SNclean_ep120/'], ['SNclean_ep120']
do_coreg = True  #if 'eval_T1' in resdir else False

df = pd.DataFrame()

for nbres, resdir in enumerate(ress):
    print(f'reading result {resname[nbres]} ')

    sujs = gdir(resdir,'.*')

    for sujdir in sujs:
        print(f'suj {sujdir} ')
        sujname = get_parent_path(sujdir)[1]
        label_file = gfile(sujdir, 'bin_label', opts={"items": 1})
        pred_file = gfile(sujdir, 'bin_pred', opts={"items": 1})
        csv_file = sujdir +'/coreg_metrics.csv'

        l_img = tio.ScalarImage(label_file)
        p_img = tio.LabelMap(pred_file)

        res_dict = {'sujname': sujname, 'model_name':  resname[nbres] }
        df_one = pd.DataFrame([res_dict])
        if do_coreg:
            print(f' dice after coregistration ')
            suj = tio.Subject(dict(lab=l_img, pred=p_img))

            pred_GM = torch.zeros_like(suj.pred.data)
            pred_GM[suj.pred.data==ind_pGM] = 1

            suj.add_image(tio.ScalarImage(tensor=pred_GM, affine=suj.pred.affine), 'predGM')
            #suj.add_image(tio.ScalarImage(tensor=suj.lab.data[ind_lGM].unsqueeze(0), affine=suj.pred.affine), 'labGM')
            #tcoreg = tio.Coregister(target='predGM', default_parameter_map='affine',estimation_mapping={'labGM': ['lab']})
            tcoreg = tio.Coregister(target='predGM', default_parameter_map='affine',estimation_mapping={'lab': ['lab']})
            sujcoreg = tcoreg(suj)

            #lab_bin = binarize_5D(sujcoreg.lab.data.unsqueeze(0), add_extra_to_class=0)
            #sujcoreg['lab']['data'] = lab_bin[0]
            res_dict = {'sujname': sujname, 'model_name': f'coreg_{resname[nbres]}'}

            dice = computes_all_metric(pred_GM.unsqueeze(0), sujcoreg.lab.data.unsqueeze(0), labels_name, selected_label=selected_label)
            fout = sujdir + '/label_coreg.nii.gz'
            sujcoreg.lab.save(fout)
            #dicec = {f'coreg_{k}': v for k,v in dice.items() }
            res_dict.update(dice)

        else :
            res_dict.update(computes_all_metric(p_img.data.unsqueeze(0), l_img.data.unsqueeze(0), labels_name,
                                                selected_label=selected_label))

        df_one = pd.concat([df_one,  pd.DataFrame([res_dict]) ], ignore_index=True)
        df_one.to_csv(csv_file)
        df = pd.concat([df, df_one], ignore_index=True)

df.to_csv('all_metric_res.csv')

test=False
if test:
    import glob
    dff=glob.glob('/data/romain/PVsynth/eval_cnn/baby/eval_T1_model_fetaBgT2_hcp_ep1/*/met*csv')
    df=pd.concat( [ pd.read_csv(fff) for fff in dff])
