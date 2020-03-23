from utils import print_accuracy, print_accuracy_df, print_accuracy_all
from utils_file import gfile, gdir, get_parent_path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim


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


f=gfile('/home/romain/QCcnn/li/','.*csv')
for ff in f:
    res=pd.read_csv(ff)
    #plt.scatter(res.ssim,res.model_out)
    #plt.plot(res.ssim, res.ssim,'k+')
    err = np.abs(res.ssim-res.model_out) #same as L1 loss
    plt.figure('err'); plt.plot(err)

    errcum = np.cumsum(err[50:-1])/range(1,len(err)-51+1)
    plt.figure('err_cum');        plt.plot(errcum)
    N=50
    err_slidin = np.convolve(err, np.ones((N,))/N, mode='valid')

    plt.figure('err_slide'); plt.plot(err_slidin)
plt.legend(('1','2','3','4','5'))

import torch.nn as nn
model = nn.Linear(10, 2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
steps = 10
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, last_epoch=-1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1)
for epoch in range(5):
    for idx in range(steps):
        scheduler.step()
        print(scheduler.get_lr())

    #print('Reset scheduler')
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
