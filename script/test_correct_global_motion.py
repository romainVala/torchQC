from pprint import pprint
from torchio import Image, transforms, INTENSITY, LABEL, Subject, SubjectsDataset
import torchio
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from torchio.transforms import RandomMotionFromTimeCourse, RandomAffine, CenterCropOrPad
from copy import deepcopy
from nibabel.viewers import OrthoSlicer3D as ov
from torchvision.transforms import Compose
import sys
from torchio.data.image import read_image
import torch
import seaborn as sns
sns.set(style="whitegrid")
pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1, 'display.width', 400)

#from torchQC import do_training
#dt = do_training('/tmp')
l1_loss = torch.nn.L1Loss()

"""
Comparing result with retromocoToolbox
"""
from utils_file import gfile, get_parent_path
import pandas as pd
from doit_train import do_training

def corrupt_data( x0, sigma= 5, amplitude=20, method='gauss', mvt_axes=[1], center='zero' ):
    fp = np.zeros((6, 200))
    x = np.arange(0,200)
    if method=='gauss':
        y = np.exp(-(x - x0) ** 2 / float(2 * sigma ** 2))*amplitude
    elif method == 'step':
        if x0 < 100:
            y = np.hstack((np.zeros((1, (x0 - sigma))),
                           np.linspace(0, amplitude, 2 * sigma + 1).reshape(1, -1),
                           np.ones((1, ((200 - x0) - sigma - 1))) * amplitude))
        else:
            y = np.hstack((np.zeros((1, (x0 - sigma))),
                           np.linspace(0, -amplitude, 2 * sigma + 1).reshape(1, -1),
                           np.ones((1, ((200 - x0) - sigma - 1))) * -amplitude))
        y = y[0]
    elif method == 'Ustep':
        y = np.zeros(200)
        y[x0-(sigma//2):x0+(sigma//2)] = 1
    elif method == 'sin':
        #fp = np.zeros((6, 182*218))
        #x = np.arange(0,182*218)
        y = np.sin(x/x0 * 2 * np.pi)
        #plt.plot(x,y)
    if center=='zero':
        print(y.shape)
        y = y -y[100]
    for xx in mvt_axes:
        fp[xx,:] = y
    return y

import math
def _translate_freq_domain( freq_domain, translations, inv_transfo=False):
    translations = -translations if inv_transfo else translations

    lin_spaces = [np.linspace(-0.5, 0.5, x) for x in freq_domain.shape] #todo it suposes 1 vox = 1mm
    meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
    grid_coords = np.array([mg.flatten() for mg in meshgrids])

    phase_shift = np.multiply(grid_coords, translations).sum(axis=0)  # phase shift is added
    exp_phase_shift = np.exp(-2j * math.pi * phase_shift)
    freq_domain_translated = exp_phase_shift * freq_domain.reshape(-1)

    return freq_domain_translated.reshape(freq_domain.shape)
def print_fft(Fi):
    s1 = np.sum(np.imag(Fi[0:100]))
    s2 = np.sum(np.imag(Fi[101:]))
    print('ks1 {} ks2 {} ks1+ks2 {} sum {}'.format(s1,s2,s1+s2,np.sum(np.imag(Fi))))

def sym_imag(Fi):
    lin_spaces = [np.linspace(-0.5, 0.5, x) for x in Fi.shape] #todo it suposes 1 vox = 1mm
    meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
    grid_coords = np.array([mg.flatten() for mg in meshgrids])
    sum_list=[]
    sum_ini = np.sum(np.imag(Fi[0:100])) + np.sum(np.imag(Fi[101:]));
    print(f'sum_ini is {sum_ini}, ')
    resolution=1000
    xx = np.arange(-10000,30000)
    for i in xx:
        t1 = np.ones(200) * i /resolution
        t2 = np.ones(200) * (i+1)/resolution
        phase_shift1 = np.multiply(grid_coords, t1).sum(axis=0)  # phase shift is added
        phase_shift2 = np.multiply(grid_coords, t2).sum(axis=0)  # phase shift is added
        exp_phase_shift1 = np.exp(-2j * math.pi * phase_shift1)
        exp_phase_shift2 = np.exp(-2j * math.pi * phase_shift2)
        #exp_phase_shift1 = np.exp(-2j * math.pi * i/4000)
        Fit1 = exp_phase_shift1 * Fi
        Fit2 = exp_phase_shift2 * Fi
        s1 = np.sum(np.imag(Fit1[0:100])) + np.sum(np.imag(Fit1[101:]));
        s2 = np.sum(np.imag(Fit2[0:100])) + np.sum(np.imag(Fit2[101:]));
        sum_list.append(s1)
        #s2 = np.sum(np.imag(Fit2)) #marche pas pour sinus
        #print(f's1 {s1} s2 {s2}')
        if s2*s1 <0 or s1*s2 < 1e-4:
            if np.abs(s1) < np.abs(s2):
                Fmin = Fit1; phase_shift = 1/resolution #phase_shift1
            else:
                Fmin = Fit2; phase_shift = (i+1)/resolution #phase_shift2
            print_fft(Fmin)
            print(f'phase shift {phase_shift}')
            xx = xx / resolution
            plt.figure();
            plt.plot(xx[0:len(sum_list)], sum_list)
            return Fmin
    print('warning no change of sign')
    xx = xx/resolution
    plt.figure();plt.plot(xx, sum_list)
    return Fi
#fmc = sym_imag(fm)

fp = corrupt_data(100, sigma=2,center='zerio')
#fp = corrupt_data(100, sigma=20, method='step')
so = corrupt_data(50,sigma=30, method='Ustep',center='None')
#so = corrupt_data(50,sigma=20, method='sin')+1
fi = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(so)).astype(np.complex))
fm =_translate_freq_domain(fi, fp)
print_fft(fm);
# fm = sym_imag(fm)
so2 = np.fft.ifftshift(np.fft.ifftn(fm))
plt.figure();plt.plot(fp.T)
plt.figure();plt.plot(abs(so2)); plt.plot(so)
plt.figure(); plt.plot(np.real(fi)); plt.plot(np.imag(fi));plt.plot(np.real(fm)); plt.plot(np.imag(fm)); plt.legend(['Sr','Sim','Tr','Tim'])

#output = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)
