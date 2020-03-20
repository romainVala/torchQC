from utils import print_accuracy, print_accuracy_df, print_accuracy_all
from utils_file import gfile, gdir, get_parent_path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


d='/home/romain.valabregue/QCcnn/li'
fres = gfile(d,'csv')
ff = fres[-1]
sujall = []
for ff in fres[4:-1] :
    res = pd.read_csv(ff)
    res['diff'] = res.ssim - res.model_out
    res = res.sort_values('diff', ascending=False)

    sujn = get_parent_path(res.fpath[1:10].values,2)[1]
    sujall.append(sujn)

ss=np.hstack(sujall)
len(ss)
len(np.unique(ss))
