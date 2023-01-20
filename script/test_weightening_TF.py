import matplotlib.pyplot as plt, pandas as pd, seaborn as sns
import glob, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, numpy.linalg as npl
from nibabel.viewers import OrthoSlicer3D as ov
from util_affine import *


fjson = '/data/romain/PVsynth/motion_on_synth_data/test1/main.json'
param = dict();param['suj_contrast'] = 1;param['suj_noise'] = 0.01;param['suj_index'] = 0;param['suj_deform'] = 0;param['displacement_shift_strategy']=None
sdata, tmot, config_runner = select_data(fjson, param, to_canonical=False)
image = sdata.t1.data[0]; brain_mask = sdata.brain.data[0]
fi = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)
fi_phase = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(image), axis=1)) #phase in y why it is the case in random_motion ?

wsh_bad = np.sqrt( np.sum( abs(fi**2),axis=(0,2) ) )
wsh_ok  = np.sqrt( np.sum( abs(fi)**2,axis=(0,2) ) )


#wcfft = abs(np.sum(np.sum(fi_phase,axis=0),axis=1))
wafft =  np.sum(abs(fi_phase),axis=(0,2))
wa2fft = np.sum(abs(fi_phase**2),axis=(0,2))

#I do not know why but it seems not correct (to small ponderation),
coef_TF_3D = np.sum(abs(fi), axis=(0,2)) # easier to comute than with interval (but equivalent)

coef_TF_3D = np.sqrt(np.sum(abs(fi**2), axis=(0,2))) # easier to comute than with interval (but equivalent)

coef2_TF_3D = np.sum(abs(fi**2), axis=(0,2)) # this one is equivalent to wafft**2
coef2_TF_3D = abs( np.sqrt(np.sum(fi*np.conjugate(fi) , axis=(0,2))) ) # easier to comute than with interval (but equivalent)
arg = np.sqrt(np.sum(abs(fi_phase**2),axis=(0,2)))
arg = np.sqrt(abs(np.sum(fi_phase*np.conjugate(fi_phase),axis=(0,2))))

w_coef_shaw = wafft/np.sum(wafft) #since it is equivalent
w_coef_short = coef_TF_3D/np.sum(coef_TF_3D)
w_coef_shaw2 = np.sqrt(coef2_TF_3D) #an other way then from 3D fft

plt.figure(); plt.plot(wafft/np.sum(wafft)); plt.plot(wa2fft/np.sum(wa2fft))
plt.plot(coef_TF_3D/np.sum(coef_TF_3D)); plt.plot(coef2_TF_3D/np.sum(coef2_TF_3D))
plt.plot(wafft**2 / np.sum(wafft**2)); plt.plot(coef_TF_3D**2/np.sum(coef_TF_3D**2));
plt.legend(['wa', 'wa2', 'wtf', 'wtf2', 'wa**2', 'wtf**2'])
# wa2 == wa**2 == wtf2
#ov(abs(fi_phase))
ff_phase = np.sum(np.sum(abs(fi_phase),axis=0),axis=1)


# test different computation for the weigths
fi_phase = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(image), axis=1)) #phase in y why it is the case in random_motion ?
center=[ii//2 for ii in fi_phase.shape]
#fi_phase[:,:center[1],:] = 0;fi_phase[:,center[1]+1:,:] = 0
fisum=np.zeros_like(image).astype(complex)
wc,wa=np.zeros(fi_phase.shape[1]).astype(complex),np.zeros(fi_phase.shape[1])
for x in range(fi_phase.shape[1]):
    fi_phase_cut = np.zeros_like(fi_phase).astype(complex)
    fi_phase_cut[:,x,:] = fi_phase[:,x,:]
    fi_image = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(fi_phase_cut), axis=1))
    fisum += fi_image # this reconstruct the image but not abs(fi_image)
    wc[x] = np.sum(fi_image*np.conj(fi_image)); wa[x] = np.sum(abs(fi_image))
    #ov(np.real(fi_image))

ov(abs(fisum))

plt.figure(); plt.plot(coef_TF_3D/np.sum(coef_TF_3D)) ; plt.plot(w_coef_short/np.sum(w_coef_short)) #ok
plt.figure();plt.plot(w_coef_shaw/np.sum(w_coef_shaw));plt.plot(wafft/np.sum(wafft));plt.legend(['nfft','fft'])

plt.figure(); plt.plot(coef_TF_3D**2/np.sum(coef_TF_3D**2)) ; plt.plot(w_coef_short**2/np.sum(w_coef_short**2)) #ok
plt.figure();plt.plot(w_coef_shaw**2/np.sum(w_coef_shaw**2));plt.plot(wafft**2/np.sum(wafft**2));plt.legend(['nfft','fft'])
plt.plot(coef2_TF_3D/np.sum(coef2_TF_3D))
plt.plot(wa2fft/np.sum(wa2fft)) #that strange somme des carre ou carre de la somme , equivalent (after norm)

plt.figure();plt.plot(abs(wc)/np.sum(abs(wc)));plt.plot(wa/np.sum(wa));plt.legend(['c','a'])
plt.figure();plt.plot(abs(wc)/np.sum(abs(wc)));plt.plot(wa**2/np.sum(wa**2));plt.legend(['c','a'])
plt.figure();plt.plot(abs(wcfft)/np.sum(abs(wcfft)));plt.plot(wafft/np.sum(wafft));plt.legend(['c','a'])
#computing image power with 3D nftt or fft
plt.figure();plt.plot(w_coef_shaw/np.sum(w_coef_shaw));plt.plot(wa/np.sum(wa));plt.legend(['nfft','fft'])
plt.figure();plt.plot(w_coef_short/np.sum(w_coef_short));plt.plot(wafft/np.sum(wafft));plt.legend(['nfft','fft'])
plt.figure();plt.plot(w_coef_TF2_short/np.sum(w_coef_TF2_short));plt.plot(wa2fft/np.sum(wa2fft));plt.legend(['nfft','fft'])
plt.figure();plt.plot(wafft/np.sum(wafft));plt.plot(wa/np.sum(wa));plt.legend(['fft','fft_ima']) #identical

ff_fi = np.sum(np.sum(abs(fi),axis=0),axis=1)
plt.figure(); plt.plot(ff_phase/npl.sum(ff_phase)); plt.plot(ff_fi/npl.sum(ff_fi))

fi_flat = np.reshape(np.transpose(fi,[0,2,1]),-1,order='F')


#compute FFT weights for image (here always the same)
w_coef = np.abs(fi)
w_coef_flat = np.transpose(w_coef,[0,2,1]).reshape(-1, order='F') #ordering is important ! to the the phase in y !
fitpar=interpolate_fitpars(fitpar,len_output=218)
# w TF coef approximation at fitpar resolution (== assuming fitparinterp constant )
w_coef_short, w_coef_TF2_short, w_coef_shaw = np.zeros_like(fitpar[0]), np.zeros_like(fitpar[0]), np.zeros_like(fitpar[0])
step_size = w_coef_flat.shape[0] / w_coef_short.shape[0]
for kk in range(w_coef_short.shape[0]):
    ind_start = int(kk * step_size)
    ind_end = ind_start + int(step_size)
    w_coef_short[kk] = np.sum(w_coef_flat[ind_start:ind_end])  # sum or mean is equivalent for the weighted mean
    w_coef_TF2_short[kk] = np.sum(w_coef_flat[ind_start:ind_end]**2)  # sum or mean is equivalent for the weighted mean

    # in shaw article, they sum the voxel in image domain (iFFT(im conv mask)) but 256 fft is too much time ...
    # fft_mask = np.zeros_like(w_coef_flat).astype(complex)
    # fft_mask[ind_start:ind_end] = w_coef_flat[ind_start:ind_end]
    # fft_mask = fft_mask.reshape(fi.shape, order='F')
    # ffimg = np.fft.ifftshift(np.fft.ifftn(fft_mask))
    # w_coef_shaw[kk] = np.sum(np.abs(ffimg))
    # to fix there is problem in th index that make if differ from wa following solution make it identical

    fft_mask = np.zeros_like(fi).astype(complex)
    fft_mask[:,kk,:] = fi[:,kk,:]
    ffimg = np.fft.ifftshift(np.fft.ifftn(fft_mask))
    w_coef_shaw[kk] = np.sum(np.abs(ffimg))

w_coef_short = w_coef_short / np.sum(w_coef_short)  # nomalize the weigths
w_coef_TF2_short = w_coef_TF2_short / np.sum(w_coef_TF2_short)  # nomalize the weigths
w_coef_shaw = w_coef_shaw / np.sum(w_coef_shaw)  # nomalize the weigths
plt.figure(); plt.plot(w_coef_short); plt.plot(w_coef_shaw); plt.plot(w_coef_TF2_short);
plt.legend(['wTF', 'shaw', 'wTF2']) #but shaw method is identical to compute directly from 1D fft
