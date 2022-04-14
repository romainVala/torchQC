# calcul metrique dans le plan de fourier
# decomposition wawelet, pour characterise l'effet du motion (base versus haute frequence perturbation)
# parametre geometri et contrast de l'object ...
# corection global displacement dans le plan de fourier, (subvoxel)
import torch
import matplotlib.pyplot as plt, seaborn as sns, pandas as pd
import matplotlib.ticker as plticker
import numpy as np
import os, sys, math
from nibabel.viewers import OrthoSlicer3D as ov
import torchio as tio
from utils_file import gfile, get_parent_path
sns.set(style="whitegrid")
pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1, 'display.width', 400)

from torchio.metrics.normalized_cross_correlation import th_pearsonr, normalize_cc
from torchio.metrics.ssim import functional_ssim
from scipy.interpolate import pchip_interpolate
from kymatio.numpy import Scattering1D

def get_mshape( resolution=512, nb_tissue=(3,5), background_size=(20,40), min_size=10, max_tissue_size=40):
    s=np.zeros((1,resolution))
    bg_size = np.random.uniform(background_size[0],background_size[1],1).astype(int)[0]
    nbt = np.rint(np.random.uniform(nb_tissue[0],nb_tissue[1],1)).astype(int)[0]
    max_size=resolution-2*bg_size - nbt*min_size
    tissue_size=[]
    for _ in range(0,nbt-1):
        tsize = np.rint(np.random.uniform(min_size,max_size,1)).astype(int)[0]
        tsize = min(tsize, max_tissue_size)
        tissue_size.append(tsize)
        max_size = max_size+min_size - tsize
    tissue_size.append(max_size+min_size)

    return s

def corrupt_data( x0, sigma= 5, amplitude=20, method='gauss', mvt_axes=[1], center='zero', resolution=200, sym=False ):
    fp = np.zeros((6, resolution))
    x = np.arange(0, resolution)
    if method=='gauss':
        y = np.exp(-(x - x0) ** 2 / float(2 * sigma ** 2))*amplitude
    elif method == '2step':
        y = np.hstack((np.zeros((1, (x0 - sigma[0]))),
                       np.linspace(0, amplitude[0], 2 * sigma[0] + 1).reshape(1, -1),
                       np.ones((1, sigma[1]-1)) * amplitude[0],
                       np.linspace(amplitude[0], amplitude[1], 2 * sigma[0] + 1).reshape(1, -1),
                       np.ones((1, sigma[2]-1)) * amplitude[1],
                       np.linspace(amplitude[1], 0 , 2 * sigma[0] + 1).reshape(1, -1)) )
        remain = resolution - y.shape[1]
        if remain<0:
            y = y[:,:remain]
            print(y.shape)
            print("warning seconf step is too big taking cutting {}".format(remain))
        else:
            y = np.hstack([y, np.zeros((1,remain))])
        y=y[0]

    elif method == 'step':
        if x0 < 100:
            y = np.hstack((np.zeros((1, (x0 - sigma))),
                           np.linspace(0, amplitude, 2 * sigma + 1).reshape(1, -1),
                           np.ones((1, ((resolution - x0) - sigma - 1))) * amplitude))
        else:
            y = np.hstack((np.zeros((1, (x0 - sigma))),
                           np.linspace(0, -amplitude, 2 * sigma + 1).reshape(1, -1),
                           np.ones((1, ((resolution - x0) - sigma - 1))) * -amplitude))
        y = y[0]
    elif method == 'Ustep':
        y = np.zeros(resolution)
        y[x0-(sigma//2):x0+(sigma//2)] = 1
        y = y * amplitude
    elif method == 'sin':
        #fp = np.zeros((6, 182*218))
        #x = np.arange(0,182*218)
        y = np.sin(x/x0 * 2 * np.pi)
        #plt.plot(x,y)
    if center=='zero':
        #print(y.shape)
        y = y -y[resolution//2]
    for xx in mvt_axes:
        fp[xx,:] = y
    if sym:
        center = y.shape[0]//2
        y = np.hstack([y[0:center], np.flip(y[0:center])])

    return y

def get_random_2step(rampe=0, sym=False, resolution=512, shape=None, intensity=None, norm=None):
    if shape is not None:
        x0 = shape[0]
        sigma = shape[1:]
    else:
        sigma = [rampe, np.random.randint(10,resolution//4 -1), np.random.randint(10,200)]
        x0 = np.random.randint(rampe,resolution//4 -1 )
    if intensity is not None:
        ampli = intensity
    else:
        ampli = [np.random.uniform(0.1,1,1)[0], np.random.uniform(0.1,1,1)[0]]

    if sym:
        #let force the second sigma to be more than the half (ie no noise in the middle)
        last_start = x0 + sigma[0] + sigma[1]
        sigma[2] = resolution//2 + 2 - last_start
        if sigma[2]<=0:
            x0 -= sigma[2] -5
            sigma[2]=10
        print(f'x0={x0}; sigma={sigma}; ampli={ampli}')
        so = corrupt_data(x0, sigma=sigma, amplitude=ampli, method='2step', center='None', resolution=resolution)
        center = so.shape[0]//2
        so = np.hstack([so[0:center], np.flip(so[0:center])])
    else:
        so = corrupt_data(x0, sigma=sigma, amplitude=ampli, method='2step', center='None', resolution=resolution)
    if norm is not None:
        so = so / np.sum(so) * norm
    print(f'objet x {sigma[0]} freq2 {1/sigma[1]} {1/sigma[1]/(1/resolution)} freq2 {1/(sigma[2]-2)/2} {1/(sigma[2]-2)/2/(1/resolution)} ')
    return so, rampe, sigma, ampli

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
    center=Fi.shape[0]//2
    s1 = np.sum(np.imag(Fi[0:center]))
    s2 = np.sum(np.imag(Fi[center+1:]))
    print('IMAG ks1 {} ks2 {} ks1+ks2 {} sum {}'.format(s1,s2,s1+s2,np.sum(np.imag(Fi))))
    s1 = np.sum(np.angle(Fi[0:center]))
    s2 = np.sum(np.angle(Fi[center+1:]))
    print('ANGLE ks1 {} ks2 {} ks1+ks2 {} sum {}'.format(s1,s2,s1+s2,np.sum(np.angle(Fi))))
    s1 = np.sum(Fi[0:center])
    s2 = np.sum(Fi[center+1:])
    print('COMP ks1 {} ks2 {} ks1+ks2 {} sum {}'.format(s1,s2,s1+s2,np.sum(Fi)))

def l1_shfit(y1,y2,shifts, do_plot=True, fp=None, plot_diff=False):
    l1 = []
    #shifts = np.arange(-30, 30, 1)
    for shift in shifts:
        y = np.hstack([y1[-shift:], y1[0:-shift]])
        # plt.plot(y)
        ll1 = np.sum(np.abs(y - y2))
        l1.append(ll1)

    disp = shifts[np.argmin(l1)]
    if do_plot:
        if fp is not None:
            fig,axs = plot_obj(fp,y2, y1, nb_subplot=3, plot_diff=plot_diff)
            fig.set_size_inches([6, 7])
            ax = axs[2]
        else:
            f,ax=plt.subplots(1)
        ax.plot(shifts, l1)
        print('displacement from L1 {}'.format(disp))
        ax.set_ylabel('L1 norm'), ax.set_xlabel('voxel displacement shift')
        ax.set_title('min L1 for a shift of {} voxels'.format(disp))
    return disp

def l1_shfit_fft(y1,y2,shifts, do_plot=True, fp=None, loss='L1'):
    l1 = []

    resolution = y1.shape[0]
    fi = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(y1)).astype(np.complex128))
    # fm =_translate_freq_domain(fi, fp)
    kx = np.arange(-1, 1 + 2 / resolution, 2 / (resolution - 1))
    for shift in shifts:
        fp_kspace = np.exp(-1j * math.pi * kx * shift)
        fm = fi * fp_kspace
        ym = np.abs(np.fft.ifftshift(np.fft.ifftn(fm)))
        if loss=='L1':
            ll1 = np.sum(np.abs(ym - y2))
        if loss=='L2':
            ll1 =  np.sum((ym -y2)**2)
        l1.append(ll1)

    disp = shifts[np.argmin(l1)]
    if do_plot:
        if fp is not None:
            fig,axs = plot_obj(fp,y2, y1, nb_subplot=3)
            ax = axs[2]
        else:
            f,ax=plt.subplots(1)
        ax.plot(shifts, l1)
        print('displacement from L1 {}'.format(disp))
        ax.set_ylabel('L1 norm')
        ax.set_title('max from L1 is {}'.format(disp))
    return disp

def shift_fft_imag(x,fmove, fref, type='complex2'):
    kx = np.arange(-1, 1 + 2 / resolution, 2 / (resolution - 1))
    fp_kspace = np.exp(-1j * math.pi * kx * x)
    fm_corect = fmove * fp_kspace
    if type =='imag':
        imag_diff = -np.imag(fm_corect) + np.imag(fref)
    if type == 'imag2': #symetrie
        i1 = np.sum(np.imag(fm_corect[:resolution//2]))
        i2 = np.sum(np.imag(fm_corect[resolution // 2:]))
        #i2 = np.sum(np.imag(np.conjugate(fm_corect)[:resolution//2 ]))
        imag_diff = abs(i1-i2)
        return imag_diff
    elif type == 'real':
        imag_diff = -np.real(fm_corect) + np.real(fref)
    elif type == 'phase':
        imag_diff = np.angle(np.exp(-1j * (-np.angle(fm_corect) + np.angle(fref))))
    elif type == 'phase2':
        imag_diff = np.angle(np.exp(-1j * (-np.angle(fm_corect) + np.angle(fref))))
        return np.sum(np.abs(imag_diff))   #does not work if too diff outside first lobe
    elif type == 'complex':
        imag_diff = np.abs(fm_corect - fref)
        return np.sum(np.abs(imag_diff))
    elif type == 'complex2':
        imag_diff = np.abs((fm_corect - fref)*np.conj(fm_corect - fref))
        return np.sqrt(np.sum(np.abs(imag_diff)))
    weights = np.abs(fref); #weights[resolution//2]=0
    weighted_sum = (np.abs(imag_diff ) * weights).sum()
    #return (np.abs(imag_diff)).sum()
    return weighted_sum

def coreg_fft(y1, fp, shifts):
    #taking fp as argument to get som in complex ...
    fy1 = np.fft.fftshift(np.fft.fftn(y1))
    resolution = fp.shape[0]
    kx = np.arange(-1, 1 + 2 / resolution, 2 / (resolution - 1))
    fp_kspace = np.exp(-1j * math.pi * kx * fp)
    # shift_cor=-0.6;fp_shift = np.exp(-1j * math.pi * kx * np.ones_like(fp)*shift_cor)
    fy2 = fy1 * fp_kspace  # pl* fp_shift

    Cdiff = [shift_fft_imag(x, fy2, fy1) for x in shifts]
    disp = shifts[np.argmin(Cdiff)]
    return -disp

def plot_obj(fp, so, som, nb_subplot=2, plot_diff=False, axs=None, fig=None):

    if axs is None:
        fig, axs = plt.subplots(nb_subplot);
        plt.subplots_adjust(hspace=0.5)

    fi = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(so)).astype(np.complex128))
    axs[0].plot(abs(fi) * 10 / max(abs(fi)));
    #axs[0].plot(abs(fi) );

    #axs[0].set_xlim([220, 280]);
    #axs[0].set_ylim([0, 1]);
    axs[0].set_ylabel('trans Y'); axs[0].set_xlabel('k-space'); axs[0].plot(fp); axs[0].legend(['Object fft','motion'])

    axs[1].plot(so);
    axs[1].plot(som); #axs[1].plot(abs(som));
    axs[1].legend(['orig object', 'artefacted object'])

    if plot_diff:
        dso = np.diff(so, prepend=so[0])
        dsom = np.diff(som, prepend=som[0])
        _ = plot_obj(fp,dso, dsom, axs=axs)
        #dso = np.diff(dso, prepend=dso[0])
        #dsom = np.diff(dsom, prepend=dsom[0])
        #_ = plot_obj(fp,dso, dsom, axs=axs)

    return fig,axs

def simu_motion(fp, so, return_abs=True):
    resolution = fp.shape[0]
    if so.dtype==np.complex128:
        #fi =  np.fft.fftn(np.fft.ifftshift(so))
        fi = np.fft.fftshift(np.fft.fftn(so))
    else:
        #fi = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(so)).astype(np.complex))
        #fi =  np.fft.fftn(np.fft.ifftshift(so))
        fi = np.fft.fftshift(np.fft.fftn(so))
    resolution = so.shape[0]
    # if upsample:
    #     fi = np.pad(fi,resolution//2)
    #     resolution = fi.shape[0]

    # fm =_translate_freq_domain(fi, fp)
    kx = np.arange(-1, 1 + 2 / resolution, 2 / (resolution - 1))
    fp_kspace = np.exp(-1j * math.pi * kx * fp)
    fm = fi * fp_kspace
    #som = np.fft.ifftshift(np.fft.ifftn(fm))
    som = np.fft.ifftn(np.fft.ifftshift(fm))

    if return_abs:
        som = np.abs(som)
    return som

def sym_imag(Fi, Fo=None):
    lin_spaces = [np.linspace(-0.5, 0.5, x) for x in Fi.shape] #todo it suposes 1 vox = 1mm
    meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
    grid_coords = np.array([mg.flatten() for mg in meshgrids])
    sum_list=[]
    sum_ini = np.sum(np.imag(Fi[0:100])) + np.sum(np.imag(Fi[101:]));
    print(f'sum_ini is {sum_ini}, ')
    resolution=1000
    xx = np.arange(-30000,30000)
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
        if s2*s1 <0 :#or s1*s2 < 1e-4:
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

def rand_uniform( min=0.0, max=1.0, shape=1):
    rand = torch.FloatTensor(shape).uniform_(min, max)
    if shape == 1:
        return rand.item()
    return rand.numpy()

def get_perlin(resolution, freq=16, amplitude=10, center='zerro', sigma=0, x0=0):
    #b = perlinNoise1D(freq, [5, 20])
    #x = np.linspace(0,1,b.shape[0])
    #xt = np.linspace(0,1,resolution)
    #bi = np.interp(xt,x,b)

    #bi = perlinNoise1D(resolution, [1, 3, 9, 3 ** 3, 3**4, 3**5, 3**6]) #even smoother
    bi = perlinNoise1D(resolution, [1, 3, 9, 3 ** 3])
    bi = bi * amplitude #because [-0.5 0.5]
    if center=='zero':
        #print(y.shape)
        bi = bi -bi[resolution//2]

    if sigma:
        ind_min = np.max( [int(x0 - sigma/2), 0])
        ind_max = int(x0 + sigma/2)
        bi_band = np.zeros_like(bi)
        bi_band[ind_min:ind_max] = bi[ind_min:ind_max]
        bi = bi_band

    return bi

def perlinNoise1D(npts, weights):
    if not isinstance(weights, list):
        weights = range(int(round(weights)))
        weights = np.power([2] * len(weights), weights)

    n = len(weights)
    xvals = np.linspace(0, 1, npts)
    total = np.zeros((npts, 1))

    for i in range(n):
        frequency = 2 ** i
        this_npts = round(npts / frequency)

        if this_npts > 1:
            total += weights[i] * pchip_interpolate(np.linspace(0, 1, this_npts),
                                                    rand_uniform(shape=this_npts)[..., np.newaxis],
                                                    xvals)
    #            else:
    # TODO does it matter print("Maxed out at octave {}".format(i))

    total = total - np.min(total)
    total = total / np.max(total)
    return total.reshape(-1) - 0.5 #add -0.5 from torchio version

def get_metric(s1,s2, mask=None, prefix='', scattering=None):
    if mask is None:
        mask = np.ones_like(s1)
    s1 = s1[mask>0]
    s2 = s2[mask>0]

    l1loss = torch.nn.L1Loss()
    l2loss = torch.nn.MSELoss()
    l1 = l1loss(torch.tensor(s1).unsqueeze(0), torch.tensor(s2).unsqueeze(0)).numpy() * 100
    l2 = l2loss(torch.tensor(s1).unsqueeze(0), torch.tensor(s2).unsqueeze(0)).numpy() * 1000
    thp = float(th_pearsonr(torch.tensor(s1), torch.tensor(s2)).numpy())
    ncc = float(normalize_cc(torch.tensor(s1), torch.tensor(s2)).numpy())
    ssim = functional_ssim(torch.tensor(so).unsqueeze(0).unsqueeze(0), torch.tensor(som).unsqueeze(0).unsqueeze(0))
    ssim = {prefix + k: float(v.numpy()) for k,v in ssim.items()}
    grad1, grad2 = np.convolve(s1,[1,-1]), np.convolve(s2,[1,-1])
    dgrad = (np.sum(grad1**2)/np.sum(np.abs(grad1))**2) / (np.sum(grad2**2)/np.sum(np.abs(grad2))**2)
    mdict = {prefix + "L1": l1, prefix + "L2" : l2, prefix + "th_p": thp, prefix + "ncc": ncc ,
             prefix + "dgrad": dgrad}
    if scattering is not None:
        meta = scattering.meta()
        order1 = np.where(meta['order'] == 1)
        order2 = np.where(meta['order'] == 2)

        sx1 = scattering(s1)
        sx2 = scattering(s2)

        sxa1 = np.sum(sx1, axis=1)
        sxa2 = np.sum(sx2, axis=1)
        scat1 = np.sum(np.abs(sxa1[order1] - sxa2[order1]))
        scat2 = np.sum(np.abs(sxa1[order2] - sxa2[order2]))
        scat1L2 = np.sqrt(np.sum((sxa1[order1] - sxa2[order1])**2))
        scat2L2 = np.sqrt(np.sum((sxa1[order2] - sxa2[order2])**2))
        scat1L2n = scat1L2 / np.sqrt( np.sum( (sxa1[order1])**2 + sxa2[order1])**2   )
        #scat11 = np.sum(np.abs(sxa1[order1[:10]] - sxa2[order1[:10]]))
        #scat12 = np.sum(np.abs(sxa1[order1[10:20]] - sxa2[order1[10:20]]))
        #scat13 = np.sum(np.abs(sxa1[order1[20:]] - sxa2[order1[20:]]))

        mdict['scat1L2'], mdict['scat2L2'], mdict['scat1L2n'] = scat1L2, scat2L2, scat1L2n
        mdict['scat1'], mdict['scat2'] = scat1, scat2
        #mdict['scat11'], mdict['scat12'], mdict['scat13']   = scat11, scat12, scat13

    return dict(mdict, **ssim)

def get_metric_fp(fp, tf_s1, diff_tf=None, prefix='', shift=0):
    meanD = np.mean(fp)  - shift
    zeroD = fp[fp.shape[0]//2] -shift
    meanDTFA =  np.sum( fp*np.abs(tf_s1) ) / np.sum(np.abs(tf_s1))  - shift
    meanDTFA2 =  np.sum( fp*np.abs(tf_s1)**2 ) / np.sum(np.abs(tf_s1)**2)  - shift
    kx = np.abs(np.linspace(-1,1,tf_s1.shape[0]))
    meanDTFAkx =  np.sum( fp*np.abs(tf_s1)*kx ) / np.sum(np.abs(tf_s1)*kx)  - shift
    #meanD2TFA2 =  np.sqrt(np.sum( fp*fp*np.abs(tf_s1)**2 ) / np.sum(np.abs(tf_s1)**2))  - shift
    meanDisp    = np.mean(np.abs(fp))
    rmseDisp    = np.sqrt(np.mean(fp ** 2))
    meanDispTFA = np.sum(np.abs(fp) * np.abs(tf_s1)) / np.sum(np.abs(tf_s1))
    meanDispTFA2 = np.sum(np.abs(fp) * np.abs(tf_s1 ** 2)) / np.sum(np.abs(tf_s1 ** 2)),
    rmseDispTF  = np.sqrt(np.sum(np.abs(tf_s1) * fp ** 2) / np.sum(np.abs(tf_s1)))
    rmseDispTF2 =  np.sqrt(np.sum(np.abs(tf_s1 ** 2) * fp ** 2) / np.sum(np.abs(tf_s1 ** 2)))
    #tf_s1[tf_s1.shape[0]//2] = 0
    #meanDTFzA =  np.sum( fp*np.abs(tf_s1) ) / np.sum(np.abs(tf_s1))
    #meanDTFzA2 =  np.sum( fp*np.abs(tf_s1)**2 ) / np.sum(np.abs(tf_s1)**2)

    dict_disp = {
        prefix + "meanD": meanD,
        prefix + "zeroD": zeroD,
        prefix + "meanDTFA": meanDTFA,
        prefix + "meanDTFA2": meanDTFA2,
        prefix + "meanDTFAkx": meanDTFAkx,
        #prefix + "meanD2TFA2": meanD2TFA2,
        #prefix + "meanDTFzA": meanDTFzA,
        #prefix + "meanDTFzA2": meanDTFzA2,
        prefix + "meanDisp": meanDisp,
        prefix + "rmseDisp": rmseDisp,
        prefix + "meanDispTFA": meanDispTFA,
        # "meanDispTFA2": np.sum(np.abs(fp) * np.abs(tf_s1 ** 2)) / np.sum(np.abs(tf_s1 ** 2)),
        # "meanDispTFP" : np.sum(np.abs(fp) * np.angle(tf_s1)) / np.sum(np.angle(tf_s1)),  #poid negatif
        # "meanDispTFC" : np.abs( np.sum(np.abs(fp) * tf_s1) / np.sum(tf_s1) ),
        prefix + "rmseDispTF": rmseDispTF,
        prefix + "rmseDispTF2": rmseDispTF2
    }
    if diff_tf is not None:
        dict_disp[prefix + 'meanDTFdifA'] = np.sum(fp * np.abs(diff_tf)) / np.sum(np.abs(diff_tf)) - shift
        dict_disp[prefix + 'meanDTFdifA2'] = np.sum(fp * np.abs(diff_tf) ** 2) / np.sum(np.abs(diff_tf) ** 2) - shift
        dict_disp[prefix +'meanDispTFdiffA'] = np.sum(np.abs(fp) * np.abs(diff_tf)) / np.sum(np.abs(diff_tf))
        dict_disp[prefix +'meanDispTFdiffA2'] = np.sum(np.abs(fp) * np.abs(diff_tf ** 2)) / np.sum(np.abs(diff_tf ** 2))
        dict_disp[prefix +'rmseDispTFdiff' ] = np.sqrt(np.sum(np.abs(diff_tf) * fp ** 2) / np.sum(np.abs(diff_tf)))
        dict_disp[prefix +'rmseDispTFdiff2'] =  np.sqrt(np.sum(np.abs(diff_tf ** 2) * fp ** 2) / np.sum(np.abs(diff_tf ** 2)))
        weigths = np.abs(diff_tf.copy()) #* np.abs(diff_tf)
        low_freq = resolution//2//8; center = resolution//2
        weigths[:center-low_freq] = 0 ;weigths[center+low_freq:] = 0
        dict_disp[prefix + 'r32'] = np.sum(fp * weigths**2) / np.sum(weigths**2) - shift
        low_freq = resolution//2//8//2; center = resolution//2
        weigths = np.abs(diff_tf.copy()); weigths[:center-low_freq] = 0 ;weigths[center+low_freq:] = 0
        dict_disp[prefix + 'r16'] = np.sum(fp * weigths**2) / np.sum(weigths**2) - shift
        tf_s1[resolution//2] = 0
        #dict_disp[prefix +'TFzA'] = np.sum( fp*np.abs(tf_s1) ) / np.sum(np.abs(tf_s1))  - shift
        #dict_disp[prefix +'TFzA2'] = np.sum( fp*np.abs(tf_s1)**2 ) / np.sum(np.abs(tf_s1)**2)  - shift
        from scipy.signal import savgol_filter #g numpy.convolve().?
        yd = savgol_filter(abs(tf_s1), 51, 5)
        yd=np.flip(yd[:resolution//2]) #begin by the center, (and take one half)
        ind_first_min = np.argwhere(np.r_[True, yd[1:] < yd[:-1]] & np.r_[yd[:-1] < yd[1:], True])[0][0]
        print(f'cuting at {ind_first_min} ')
        low_freq = ind_first_min; center = resolution//2
        weigths = np.abs(diff_tf.copy()); weigths[:center-low_freq] = 0 ;weigths[center+low_freq:] = 0
        dict_disp[prefix + 'diff_center'] = np.sum(fp * weigths**2) / np.sum(weigths**2) - shift

    return dict_disp

def get_metrics(s1, s2, fp=None, scatt=None, shift=0):
    mdict = get_metric(s1, s2, scattering=scatt)
    mdict_brain = get_metric(s1, s2, mask=s1, prefix='brain_')
    mdict = dict(mdict, ** mdict_brain)

    tf_s1 = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(s1)).astype(np.complex128))
    tf_s2 = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(s2)).astype(np.complex128))

    dict_allA = get_metric(np.abs(tf_s1), np.abs(tf_s2), prefix='tf_abs_')
    dict_allP = get_metric(np.angle(tf_s1), np.angle(tf_s2), prefix='tf_pha_')

    dict_all = dict(dict_allA, ** dict_allP)

    if fp is not None:
        s1dif = np.diff(s1, prepend=s1[0])
        s2dif = np.diff(s2, prepend=s2[0])
        tf_diff = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(s1dif)).astype(np.complex128))
        dict_disp = get_metric_fp(fp, tf_s1, diff_tf=tf_diff, shift=shift)
        dict_all = dict(dict_all, ** dict_disp)

        dict_disp_diff = get_metric(s1dif, s2dif , prefix='tf_abs_')
        dict_all = dict(dict_all, **dict_disp_diff)

    return dict(mdict, **dict_all)


resolution=512
shifts = np.arange(-30,30,1); shifts_small = np.arange(-2,2,0.01)
shifts_fft = np.linspace(-15,15,500)

T = resolution
J = 5  # The averaging scale is specified as a power of two, 2**J. Here, we set J = 5 to get an averaging, or maximum,
# scattering scale of 2**5 = 32 samples.
Q = 8  # we set the number of wavelets per octave, Q, to 8. This lets us resolve frequencies at a resolution of 1/8 octaves.
scattering = None#Scattering1D(J, T, Q)

so,_,_,_ = get_random_2step(rampe=4, sym=True); soTF = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(so)).astype(np.complex))
amplitudes, sigmas, x0s = [2,5,10,20], [2, 4, 10, 20, 40, 80, 120, 160, 200]  , np.hstack([np.linspace(10, 120, 10), np.linspace(130, 256, 30)])
amplitudes, sigmas, x0s = [10, 10, 10, 10], [5, 10, 20, 40, 80, 120, 160, 200]  , np.hstack([np.linspace(10, 120, 10), np.linspace(130, 256, 30)])
amplitudes, sigmas, x0s = [2,5,10,20], [50, 100, 150, 200]  ,np.linspace(10, 256, 20);amplitudes = np.tile(amplitudes,[10])
amplitudes, sigmas, rampes, nb_x0s, x0_min, sym = [1,2,4,8], [2,4,8,16,32,64,128], [2,5,10,20], 1, 256, False
rampes = [0, 1, 2, 5]; sigmas = np.hstack((np.linspace(2,64,16),np.linspace(74,256,16))) # np.linspace(2,256, 64);
nb_obj = 5;
np.random.seed(12)
#amplitudes, sigmas, rampes, sym, nb_x0s, x0_min = [10,10,10,10,10,10,10,10], [2,4,8,16,32,64], [2,5,10,20,2,5,10,20], True, 10, 20
#amplitudes, sigmas, rampes, sym, nb_x0s, x0_min = [10,10,10,10,10,10,10,10], [2,4,8,16,32,64], [2,5,10,20,2,5,10,20], False, 10, 200

df, dfmot = pd.DataFrame(), pd.DataFrame()
for rampe in rampes:
    for nboj in range(nb_obj):
        #so, _, _, _ = get_random_2step(rampe=rampe, sym=True, norm=256)
        for ind,a in enumerate(amplitudes):
            #rampe = 5# rampes[ind] #np.random.randint(2, 30, 1)[0]
            # so,_,_,_  = get_random_2step(rampe=rampe, sym=True, norm=256) befor 11/2021 but confond object shape and amplitude ???
            # if rampe==2:
            #     shape = [100,5, 25, 60]; amp = [0.5, 1]
            # if rampe==5:
            #     shape = [100,5, 50, 120]; amp = [0.5, 1]
            # so, _, _, _ = get_random_2step(rampe=rampe, sym=True, norm=256, shape=shape, intensity=amp)
            #plot_obj(fp,so,so,plot_diff=True)
            #plt.plot(so)
            for s in sigmas:  # np.linspace(2,200,10):
                if sym:
                    xcenter = resolution // 2 - s // 2;
                    x0s = np.floor(np.linspace(xcenter - x0_min, xcenter, nb_x0s))
                else:
                    xcenter = resolution // 2;
                    x0s = np.floor(np.linspace(x0_min, xcenter, nb_x0s))
                    x0s = x0s[x0s>=s//2] #remove first point to not reduce sigma

                for x0 in x0s:
                    so,_,_,_  = get_random_2step(rampe=rampe, sym=True, norm=256)
                    s=int(s); x0 = int(x0); xend = x0+s//2
                    print(f'Amplitude {a} sigma {s} X0 {x0} rampe {rampe}')
                    #fp = get_perlin(resolution, x0=x0, sigma=s, amplitude=a, center='zero')
                    fp = corrupt_data(x0, sigma=s, amplitude=a, method='Ustep', mvt_axes=[1], center='zero', resolution=resolution, sym=True)
                    som = simu_motion(fp, so)
                    mydict_cor_before = get_metric(so, som, scattering=scattering, prefix='before')

                    disp_fft = coreg_fft(so, fp, shifts_fft)
                    print(disp_fft)
                    disp = l1_shfit(som,so,shifts, do_plot=False,fp=fp); Dzero = fp[resolution//2]
                    #mydict = {"sigma":s, "x0":x0, "amplitude": a, "rampe": rampe, "shift": -disp, "Dzero": Dzero }
                    #mydict_mot = dict(get_metrics(so,som,fp=fp, scatt=scattering) , **mydict)
                    #dfmot = dfmot.append(mydict_mot, ignore_index=True)
                    if np.abs(disp)>=0:
                        fp = fp + disp
                        som = simu_motion(fp, so)
                        disp2 = l1_shfit_fft(som, so, shifts_small, do_plot=False, fp=fp, loss='L2')
                        fp = fp + disp2;  som = simu_motion(fp, so);  disp+=disp2
                    mydict = {"sigma":s, "x0":x0, "xend":xend, "amplitude": a, "num_obj": nboj, "shift": -disp, "rampe": rampe,
                              "Dzero": Dzero, 'disp_fft': disp_fft}
                    mydict_cor = dict(get_metrics(so,som,fp=fp, scatt=scattering, shift=disp) , **mydict)
                    mydict_cor = dict(mydict_cor, **mydict_cor_before)
                    df = df.append(mydict_cor, ignore_index=True)
                    if disp < 1 and  mydict_cor['meanDTFdifA2']>3:
                        pass

keyn = ['shift', 'meanDisp','meanDispTFA','meanDispTFdiffA','meanDispTFdiffA2','rmseDisp','rmseDispTF','rmseDispTF2','rmseDispTFdiff','rmseDispTFdiff2']
keyn += ['meanD', 'zeroD'  ,'meanDTFA' ,'meanDTFA2', 'meanDTFdifA2', 'diff_center']
for k in keyn:
    df[k] = df[k] / df.amplitude

plt.figure();plt.plot(df['xend'],df['shift']);plt.plot(df['xend'],df['meanDTFA']);plt.plot(df['xend'],df['meanDTFdifA2']);
plt.plot(df['xend'],df['r16']) ; plt.plot(df['xend'],df['TFzA']) ;plt.legend(['shif','tfa','difTFA2','r16','tfzA'])
sns.relplot(data=df, x="xend", y="shift", hue="sigma", legend='full', palette=cmap, col="rampe", kind="line", col_wrap=2)

cmap = sns.color_palette("coolwarm", len(df.amplitude.unique()))
sns.relplot(data=df, x="sigma", y="disp_fft", hue="amplitude", legend='full', palette=cmap, col='rampe', col_wrap=2, kind="line")

fig = sns.relplot(data=df, x='shift', y='meanDTFA2kx', col='sigma', kind='scatter', col_wrap=3, hue='rampe')
fig = sns.relplot(data=df, x='shift', y='disp_fft', col='rampe', kind='scatter', col_wrap=2, hue='sigma', legend='full')
fig = sns.relplot(data=df, x='shift', y='r16', col='sigma', kind='scatter', col_wrap=3, hue='rampe')
fig = sns.relplot(data=df, x='shift', y='diff_center', col='sigma', kind='scatter', col_wrap=3, hue='rampe')
for aa in fig.axes:    aa.plot([0,8], [0,8])
plot_obj(fp, so, som)

sel_key=['L1', 'L2', 'ncc', 'ssim', 'dgrad', 'beforeL1', 'beforeL2', 'beforencc', 'beforessim','shift'] #,'scat1L2n' 'structure', 'contrast','luminance'] 'scat1L2',
#sel_key=['brain_L1', 'brain_L2', 'brain_ncc', 'brain_ssim'] #, 'structure', 'contrast','luminance']
sel_key += ['meanDispTFA', 'rmseDispTF'] #,  'rmseDispTF2'['meanDispTFA', 'meanDispTFP', 'meanDispTFC', 'rmseDispTF']
sel_key = ['meanDisp','meanDispTFA', 'meanDispTFdiffA', 'meanDispTFdiffA2'] #,  'rmseDispTF2'['meanDispTFA', 'meanDispTFP', 'meanDispTFC', 'rmseDispTF']
sel_key = ['meanD', 'zeroD'  ,'meanDTFA' ,'meanDTFA2', 'meanDTFdifA2', 'diff_center']

figres = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/figure/'
figres = '/data/romain/PVsynth/figure/synth_motion/synth1D_fp_zero_center_sigma_new'
prefix = 'object_random'#'object_by_rampe'
#for k, dfsub in dfsub_dict.items():
cmap = sns.color_palette("coolwarm", len(df.sigma.unique()))
cmap = sns.color_palette(n_colors= len(df.amplitude.unique()))
#see  https://seaborn.pydata.org/tutorial/color_palettes.html
before_after=True
for ykey in sel_key:
    #fig = sns.relplot(data=dfsub, x="xend", y=ykey, hue="sigma", legend='full', kind="line",palette=cmap, col='amplitude',col_wrap=2)
    #fig = sns.relplot(data=df, x="xend", y=ykey, hue="sigma", legend='full', kind="line",palette=cmap, col='rampe',col_wrap=2 )
    #fig = sns.relplot(data=df, x="shift", y=ykey, hue="sigma", legend='full', kind="scatter",palette=cmap, col='rampe',col_wrap=2 )
    if before_after:
        dfm = df.melt(id_vars=['sigma','amplitude','rampe',], value_vars=[ykey, f'before{ykey}'],
                     var_name='coreg', value_name=ykey)

        fig = sns.relplot(data=dfm, x="sigma", y=ykey, hue="amplitude", legend='full', palette=cmap,
                          col='rampe', col_wrap=2, kind="line", style='coreg')
    else:
        fig = sns.relplot(data=df, x="sigma", y=ykey, hue="amplitude", legend='full', palette=cmap,
                          col='rampe', col_wrap=2, kind="line")
    #for aa in fig.axes: aa.grid();aa.plot([0,10], [0,10])
    plt.text(0.8,0.3,ykey,transform=plt.gcf().transFigure,bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
    #plt.text(0.8,0.2,k,transform=plt.gcf().transFigure,bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
    figname = f'{prefix}_{ykey}.png'
    fig.savefig(figres+figname)
    plt.close()

#againt but melt before and after coreg
sel_key=['shift', 'meanDisp','meanDispTFA', 'meanDispTFdiffA',]

freq_res = 1/512
resolution=512
shifts = np.arange(-30,30,1)
a=10;s=80; x0=180
#a=10;s=10; x0=resolution//2
for rampe in [5]:
    so,_,_,_ = get_random_2step(rampe=3, sym=True, norm=256, resolution=resolution)
    for s in [10, 80]:
        fp = corrupt_data(x0, sigma=s, amplitude=a, method='Ustep', mvt_axes=[1], center='zerfo', resolution=resolution)
        som = simu_motion(fp, so)
        disp = l1_shfit(som, so, shifts, do_plot=False, fp=fp)
        fpp = fp + disp
        somm = simu_motion(fpp, so)
        disp2 = l1_shfit_fft(somm, so, shifts_small, do_plot=False, fp=fpp, loss='L2')

        md = get_metrics(so,som,fp=fp, scatt=None)
        print(f'Amplitude {a} sigma {s} X0 {x0} rampe {rampe}')
        print(f' L1/L2 shift {-disp-disp2}\nmeanDTFdifA2 {md["meanDTFdifA2"]} \nmeanDTFA2 {md["meanDTFA2"]}  \nrrr {md["r32"]}  \nrrr2 {md["r32"]} ')
        plot_obj(fp,so,som, plot_diff=True)


cmap = sns.color_palette("coolwarm", len(df.sigma.unique()))
sns.relplot(data=df, x="xend", y="shift", hue="sigma", legend='full', palette=cmap, col="rampe", kind="line", col_wrap=2)
plt.figure();sns.lineplot(data=df, x="xend", y="L2", hue="sigma", legend='full', palette=cmap, style="amplitude")
plt.figure();sns.lineplot(data=df, x="x0", y="shift", hue="sigma",  legend='full', palette=cmap)
plt.figure();sns.scatterplot(data=df, x="L2", y="L1", size="x0", hue="sigma", legend='full')
plt.figure();sns.scatterplot(data=df, x="meanDispTFA", y="structure", size="x0", hue="sigma", legend='full')

#i1 = df.L2 > 15


sel_key=['tf_abs_L1', 'tf_abs_L2', 'tf_abs_ncc', 'tf_abs_ssim']
sel_key=['tf_pha_L1', 'tf_pha_L2', 'tf_pha_ncc', 'tf_pha_ssim']
sel_key=['L1', 'L2', 'ncc', 'ssim', 'scat1', 'scat2'] #, 'structure', 'contrast','luminance']
sel_key=['L1', 'L2', 'ncc', 'ssim', 'dgrad'] #,  'scat1L2n' 'structure', 'contrast','luminance'] 'scat1L2',
sel_key=['brain_L1', 'brain_L2', 'brain_ncc', 'brain_ssim'] #, 'structure', 'contrast','luminance']
sel_key += ['meanDispTFA', 'rmseDispTF'] #,  'rmseDispTF2'['meanDispTFA', 'meanDispTFP', 'meanDispTFC', 'rmseDispTF']
sel_key += ['meanDispTFA', 'meanDispTFdiffA', 'meanDispTFdiffA2'] #,  'rmseDispTF2'['meanDispTFA', 'meanDispTFP', 'meanDispTFC', 'rmseDispTF']
#sel_key += ['meanDisp', 'rmseDisp']
sns.pairplot(df[sel_key], kind="scatter", corner=True)
sns.pairplot(df[sel_key + ['sel']], kind="scatter", corner=True, hue='sel')

sel_key=[]
for k in df.keys():
    if "Disp" in k :
        print(k); sel_key.append(k)

dfall=df;
df = dfall[dfall['rampe']<6]
plt.figure()
plt.scatter(df['shift'], df['meanDTFA']) #df['meanDTFA']-df['shift']
plt.scatter(df['shift'], df['meanDTFA2']) #df['meanDTFA']-df['shift']
plt.scatter(df['shift'], df['zeroD']) #df['meanDTFA']-df['shift']
plt.scatter(df['shift'], df['diff_center'])
#plt.scatter(df['shift'], df['meanDTFdifA2'])
#plt.legend(['meanDTFA','meanDTFA2', 'meanDTFdiffA', 'meanDTFdiffA2'])
plt.legend(['meanDTFA','meanDTFA2','Dzero', 'diff_center'])
plt.plot([0,10],[0,10]);plt.xlabel('L1 shift');plt.ylabel('weighted mean dispalcement')

sns.scatterplot(data=df, x='shift', y='meanDTFdifA2', hue='sigma')
sns.scatterplot(data=df, x='meanD', y='meanDTFdifA2', hue='sigma')
fig = sns.relplot(data=df, x='shift', y='meanDTFdifA2', col='sigma', kind='scatter', col_wrap=3, hue='rampe')
fig = sns.relplot(data=df, x='shift', y='r16', col='sigma', kind='scatter', col_wrap=3, hue='rampe')
for aa in fig.axes:
    aa.plot([0,10], [0,10])
cmap = sns.color_palette("coolwarm", len(df.sigma.unique()))
plt.figure();sns.lineplot(data=df, x="x0", y="meanDTFdifA2", hue="sigma",  legend='full', palette=cmap)
sns.relplot(data=df, x="x0", y="L2", hue="sigma", legend='full', palette=cmap, kind="line")
sns.relplot(data=df, x="x0", y="scat1L2n", hue="sigma", legend='full', palette=cmap, kind="line")
sns.relplot(data=df, x="x0", y="shift", hue="sigma", legend='full', palette=cmap, kind="line")
sns.relplot(data=df, x="rso", y="shift", hue="sigma", legend='full', palette=cmap, kind="line")

#one_suj example
rampe=2; a=10; s=1; x0=256; meth='step'; sym=False
rampe=5; a=5; s=64; x0=210; meth='Ustep'; sym=True
rampe=20; a=8; s=64; x0=256; meth='Ustep'; sym=True
rampe=0; a=8; s=70; x0=256; meth='Ustep'; sym=False
rampe=10; a=10; s=20; x0=256; meth='Ustep'; sym=False

np.random.seed(12)
so,_,_,_  = get_random_2step(rampe=rampe, sym=True, intensity=[0.5, 1])
fp = corrupt_data(x0, sigma=s, amplitude=a, method=meth, mvt_axes=[1], center='none',
                  resolution=resolution, sym=sym)
#fp=np.ones_like(fp)*3.3#fp = np.ones_like(fp)*10 # fp = fp - 2.61
som = simu_motion(fp, so)
disp = l1_shfit(som, so, shifts, do_plot=True, fp=fp)
disp_fft = coreg_fft(so,fp,shifts_fft)
fp = fp + disp; som = simu_motion(fp, so)
disp2 = l1_shfit_fft(som, so, shifts_small, do_plot=True, fp=fp, loss='L1')
fp = fp + disp2;disp += disp2
som = simu_motion(fp, so);
dd=get_metrics(so,som,fp=fp, scatt=scattering, shift=disp)
print(f'mean TFA {dd["meanDTFA"]} meandTFA2 {dd["meanDTFA2"]} zeroD {dd["zeroD"]} meanD {dd["meanD"]}')
print(f'disp {disp} fft_coreg {disp_fft} ')

#inversion
#som = simu_motion(fp, so, return_abs=False)
#fpinv = -fp
#soinv = simu_motion(fpinv, som)
#disp = l1_shfit(soinv, so, shifts, do_plot=True, fp=fpinv)

fi = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(so)).astype(np.complex))
fi = np.fft.fftshift(np.fft.fftn(so))

# fm =_translate_freq_domain(fi, fp)
kx = np.arange(-1, 1 + 2 / resolution, 2 / (resolution - 1))
fp_kspace = np.exp(-1j * math.pi * kx * fp)
#shift_cor=-0.6;fp_shift = np.exp(-1j * math.pi * kx * np.ones_like(fp)*shift_cor)
fm = fi * fp_kspace #pl* fp_shift

som = np.fft.ifftshift(np.fft.ifftn(fm))
plot_obj(fp,so,abs(som))
fig, axs = plt.subplots(4);
axs[0].plot(so); axs[0].plot(abs(som)); axs[0].legend(['original', 'motion corrupted'])
axs[1].plot(np.abs(fi)); axs[1].plot(np.abs(fm)); #axs[1].set_xlim([200, 306]);
axs[1].set_ylabel('abs FFT')# axs[0].plot(np.abs(fp))
axs[1].plot(fp); axs[1].legend(['original', 'motion corrupted', 'motion time course'])
axs[2].plot(np.imag(fi)); axs[2].plot(np.imag(fm)); #axs[2].set_xlim([200, 306]);
axs[2].set_ylabel('imag FFT')
#axs[0].legend(['original', 'motion corrupted'])
axs[3].plot(np.angle(fi)); axs[3].plot(np.angle(fm)); #axs[3].set_xlim([200, 306]);
axs[3].set_ylabel('phase FFT')
axs[3].plot(np.angle(fp_kspace)); axs[3].legend(['original', 'motion corrupted', 'motion phase shift'],bbox_to_anchor=[0.2, 0.2, 0.2, 0.25])

phase_diff =np.angle(np.exp(-1j *( -np.angle(fm) + np.angle(fi) )))
phase_diff =np.angle(fm) - np.angle(fi)
plt.figure();plt.plot(phase_diff); plt.plot(np.angle(fp_kspace))

plt.figure();
plt.plot(np.angle(fp_kspace))

from scipy import optimize

y=[]; x0 = np.linspace(-11,11,500)
yp = [shift_fft_imag(x, fm, fi, type='phase') for x in x0]
yp2 = [shift_fft_imag(x, fm, fi, type='phase2') for x in x0]
yi = [shift_fft_imag(x, fm, fi, type='imag') for x in x0]
yi2 = [shift_fft_imag(x, fm, fi, type='imag2') for x in x0]
yc = [shift_fft_imag(x, fm, fi, type='complex2') for x in x0]
#yr = [shift_fft_imag(x, fm, fi,type='real') for x in x0]
plt.figure();plt.plot(x0, yp); plt.plot(x0, yi);plt.plot(x0, yi2); #plt.plot(x0, yp2);
plt.plot(x0, yc)
plt.legend(['phase','imag','imag2','comp'])
x0[np.argmin(yc)]



result = optimize.minimize_scalar(shift_fft_imag,bounds=[-10, 10],method="Brent", args=(fm,fi))
result = optimize.minimize(shift_fft_imag,0,method="Nelder-Mead", args=(fm,fi))




# shape and contrast
fig,axs = plt.subplots(2)
for k in range(0,15):
    so,_,_,_ = get_random_2step(5, sym=True, resolution=512, intensity=[0.5, 1],  norm=256)
    axs[0].plot(so)
    fi = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(so)).astype(np.complex))
    axs[1].plot(abs(fi))

fig,axs = plt.subplots(2)
for k in range(0,15):
    so,_,_,_ = get_random_2step(5, sym=True, resolution=512, shape=[100, 5, 50, 300], norm=256)
    axs[0].plot(so)
    fi = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(so)).astype(np.complex))
    axs[1].plot(abs(fi))





############# scattering
from kymatio.numpy import Scattering1D

T = 512
so,_,_,_ = get_random_2step(rampe=2, sym=True)
plot_obj(fp,so,so)
J = 5 #The averaging scale is specified as a power of two, 2**J. Here, we set J = 5 to get an averaging, or maximum,
# scattering scale of 2**5 = 32 samples.
Q = 8 # we set the number of wavelets per octave, Q, to 8. This lets us resolve frequencies at a resolution of 1/8 octaves.
sot = np.hstack([so[-50:], so[0:-50]])
scattering = Scattering1D(J, T, Q)

sx = scattering(sot)
Sx  = scattering(so)
sxt = scattering(sot)
sxm = scattering(som)

sxa = np.sum(sx,axis=1)
sxat = np.sum(sxt,axis=1)
sxam = np.sum(sxm,axis=1)

meta = scattering.meta()
order0 = np.where(meta['order'] == 0)
order1 = np.where(meta['order'] == 1)
order2 = np.where(meta['order'] == 2)
plt.figure(figsize=(8, 2))
plt.plot(so)
plt.title('Original signal')

plt.figure(figsize=(8, 8))
plt.subplot(3, 1, 1)
plt.plot(Sx[order0][0])
plt.title('Zeroth-order scattering')

plt.subplot(3, 1, 2)
plt.imshow(Sx[order1], aspect='auto')
plt.title('First-order scattering')

plt.subplot(3, 1, 3)
plt.imshow(Sx[order2], aspect='auto')
plt.title('Second-order scattering')

###################################### WAVELET Transform
import pywt

def signal_wnrmse(s1, s2, wavelet, level=None, eps=10e-8):
    s1_detail_coeffs = pywt.wavedec(s1, wavelet, level=level)[1:]
    s2_detail_coeffs = pywt.wavedec(s2, wavelet, level=level)[1:]
    #Flatten coeffs
    s1_detail_coeffs = np.concatenate(s1_detail_coeffs)[..., np.newaxis]
    s2_detail_coeffs = np.concatenate(s2_detail_coeffs)[..., np.newaxis]
    numerator = np.linalg.norm(s1_detail_coeffs - s2_detail_coeffs, axis=1)
    denominator = np.sqrt( np.linalg.norm(s1_detail_coeffs, axis=1)**2 + np.linalg.norm(s2_detail_coeffs, axis=1)**2 + eps )
    wnrmse = numerator/denominator
    return wnrmse.sum()

def signal_wnrmse_cwt(s1, s2, waveletname='cmor1.5-1.0', level=None, eps=10e-8):
    scales = np.arange(1, 64)
    dt = 1,
    [s1_coeffs, frequencies] = pywt.cwt(s1, scales, waveletname, dt)
    [s2_coeffs, frequencies] = pywt.cwt(s2, scales, waveletname, dt)

    #sum over time
    s1_coeffs = np.sum( (abs(s1_coeffs)) , axis=1)
    s2_coeffs = np.sum( (abs(s2_coeffs)) , axis=1)

    numerator = np.linalg.norm(s1_coeffs - s2_coeffs)
    denominator = np.sqrt( np.linalg.norm(s1_coeffs)**2 + np.linalg.norm(s2_coeffs)**2 + eps )
    wnrmse = numerator/denominator
    return wnrmse.sum()

def compute_wavelets_wnrmse(s1, s2, wavelet_lists=None):
    if not wavelet_lists:
        wavelet_lists = ["db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8", "db9", "db10", "sym2",
                         "sym3", "sym4", "sym5", "sym6", "sym7", "sym8", "coif1", "coif2", "coif3", "coif4", "coif5",
                         "dmey"]
    return {wavelet_name: signal_wnrmse(s1, s2, wavelet=wavelet_name) for wavelet_name in wavelet_lists}

cw = compute_wavelets_wnrmse(so,som)


def plot_wavelet(time, signal, scales,
                 waveletname = 'cmor',
                 cmap = plt.cm.seismic,
                 title = 'Wavelet Power',
                 ylabel = 'Period ',
                 xlabel = 'Time'):

    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)

    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)

    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)

    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()

time = np.arange(0,512)
scales = np.arange(1,128)
plot_wavelet(time, sot, scales )
plot_wavelet(time, som, scales )
