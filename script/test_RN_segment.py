
#apply sitk affine
import torchio as tio
import torch, numpy as np, pandas as pd
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation as dill
import glob

sujdir = glob.glob('/home/romain.valabregue/datal/segment_RedNucleus/ToRomain/suj_orig/*')
#cd /home/romain.valabregue/datal/segment_RedNucleus/ToRomain/suj_orig/RJ_100
df = pd.DataFrame()

for one_suj in sujdir:
    f_label = glob.glob(one_suj + '/red.nii')
    f_qsm = glob.glob(one_suj + '/*QSM_meas*')
    f_mag = glob.glob(one_suj + '/*iMag_meas*')
    f_r2s = glob.glob(one_suj + '/*R2star_meas*')

    suj = tio.Subject({'qsm':tio.ScalarImage(f_qsm[0]),
                       'rn':tio.ScalarImage(f_label[0]),
                       'mag':tio.ScalarImage(f_mag[0]),
                       'r2':tio.ScalarImage(f_r2s[0])})

    data = suj.rn.data[0].numpy()
    datal=dill(data)
    datal2 = dill(datal)
    RN_dill1, RN_dill2 = np.zeros_like(data),np.zeros_like(data)

    RN_dill1[datal] = 1
    RN_dill2[datal2] = 1

    border1 = torch.BoolTensor( RN_dill1-data)
    border2 = torch.BoolTensor(RN_dill2-RN_dill1)
    label = torch.BoolTensor(suj.rn.data[0]>0)
    res = {"suj_name" : os.path.basename(one_suj)}

    for contr in ['qsm', 'mag','r2']:
        dqsm = suj[contr].data[0]
        print(f'contrast is {contr}')    #dqsm = suj.mag.data[0]

        tb =  dqsm.masked_select(border1)
        tb2 =  dqsm.masked_select(border2)
        tl =  dqsm.masked_select(label)

        cnr1 = torch.abs(tb.mean() -tl.mean()) / torch.sqrt(tb.std() * tl.std())
        #cnr1 = torch.abs(tb.mean() -tl.mean()) / tb.std() / tl.std()
        print(f'cnr border 1 is {cnr1:.3} diff is {tl.mean() -tb.mean():.3} std lab {tl.std():.4} std border {tb.std():.4}',)
        print(f'         volume rn is {label.sum()} border volume is {border1.sum()}')
        cnr2 = torch.abs(tb2.mean() -tl.mean()) / torch.sqrt(tb2.std() * tl.std())
        print(f'cnr border 2 is  {cnr2:.3} diff is {tl.mean() -tb2.mean():.3} std lab {tl.std():.4} std border {tb2.std():.4}',)
        print(f'         border volume is {border2.sum()}')

        res['cmr1'] = float(cnr1)
        res['cmr2'] = float(cnr2)
        res['image'] = contr
        df = pd.concat([ df , pd.DataFrame.from_dict([res])], ignore_index=True)

df=df.rename(columns={"suj_name":'sujname'})
df = df.rename(columns={"cmr1":'cnr1'})
dfsub = df[df.image=='qsm']
dfsub = df[df.image=='mag']
dfsub = df[df.image=='r2']
dfres1 = pd.read_csv('/home/romain.valabregue/datal/segment_RedNucleus/ToRomain/pred_orig/dices_eval_QSM_model_Yeb_erode_crop_ep60.csv')
dfres2 = pd.read_csv('/home/romain.valabregue/datal/segment_RedNucleus/ToRomain/pred_orig/dices_eval_iMag_model_Yeb_crop_ep30.csv')
dfres3 = pd.read_csv('/home/romain.valabregue/datal/segment_RedNucleus/ToRomain/pred_orig/dices_eval_R2s_model_Yeb_erode_crop_ep60.csv')
dfres1['image'] = 'qsm'
dfres2['image'] = 'mag'
dfres3['image'] = 'r2'
dfres = pd.concat([dfres1, dfres2, dfres3])

dfres = pd.merge(dfres,df, on=['sujname', 'image'])
sns.set_style("whitegrid")
sns.lmplot(data=dfres, x='cnr1', y='dice', col='image')

