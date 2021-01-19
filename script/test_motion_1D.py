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

def corrupt_data( x0, sigma= 5, amplitude=20, method='gauss', mvt_axes=[1], center='zero', resolution=200 ):
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
    return y

def get_random_2step(rampe=0, sym=False, resolution=512):
    sigma = [rampe, np.random.randint(10,100), np.random.randint(10,200)]
    ampli = [np.random.rand(1,1), np.random.rand(1,1)]
    x0 = np.random.randint(rampe,200)

    so = corrupt_data(x0, sigma=sigma, amplitude=ampli, method='2step', center='None', resolution=resolution)
    if sym:
        center = so.shape[0]//2
        so = np.hstack([so[0:center], np.flip(so[0:center])])
    return so

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
    print('IMAG ks1 {} ks2 {} ks1+ks2 {} sum {}'.format(s1,s2,s1+s2,np.sum(np.imag(Fi))))
    s1 = np.sum(np.angle(Fi[0:100]))
    s2 = np.sum(np.angle(Fi[101:]))
    print('ANGLE ks1 {} ks2 {} ks1+ks2 {} sum {}'.format(s1,s2,s1+s2,np.sum(np.angle(Fi))))
    s1 = np.sum(Fi[0:100])
    s2 = np.sum(Fi[101:])
    print('COMP ks1 {} ks2 {} ks1+ks2 {} sum {}'.format(s1,s2,s1+s2,np.sum(Fi)))

def l1_shfit(y1,y2,shifts, do_plot=True, fp=None):
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
            fig,axs = plot_obj(fp,so, som, nb_subplot=3)
            ax = axs[2]
        else:
            f,ax=plt.subplots(1)
        ax.plot(shifts, l1)
        print('displacement from L1 {}'.format(disp))
        ax.set_ylabel('L1 norm')
        ax.set_title('max from L1 is {}'.format(disp))
    return disp

def l1_shfit_fft(y1,y2,shifts, do_plot=True, fp=None, loss='L1'):
    l1 = []

    resolution = y1.shape[0]
    fi = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(y1)).astype(np.complex))
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
            fig,axs = plot_obj(fp,so, som, nb_subplot=3)
            ax = axs[2]
        else:
            f,ax=plt.subplots(1)
        ax.plot(shifts, l1)
        print('displacement from L1 {}'.format(disp))
        ax.set_ylabel('L1 norm')
        ax.set_title('max from L1 is {}'.format(disp))
    return disp

def plot_obj(fp, so, som, nb_subplot=2):
    fig, axs = plt.subplots(nb_subplot);
    axs[0].plot(fp); axs[0].legend(['motion'])
    axs[0].set_ylabel('trans Y')
    axs[1].plot(so);
    axs[1].plot(abs(som));
    axs[1].legend(['orig object', 'artefacted object'])

    return fig,axs

def simu_motion(fp, so, return_abs=True):
    resolution = fp.shape[0]
    fi = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(so)).astype(np.complex))
    # fm =_translate_freq_domain(fi, fp)
    kx = np.arange(-1, 1 + 2 / resolution, 2 / (resolution - 1))
    fp_kspace = np.exp(-1j * math.pi * kx * fp)
    fm = fi * fp_kspace
    som = np.fft.ifftshift(np.fft.ifftn(fm))
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

def get_perlin(resolution, freq=16):
    b = perlinNoise1D(freq, [5, 20])
    x = np.linspace(0,1,b.shape[0])
    xt = np.linspace(0,1,resolution)
    bi = np.interp(xt,x,b)
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

def corrupt_data( x0, sigma= 5, amplitude=20, method='gauss', mvt_axes=[1], center='zero', resolution=200 ):
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
        left = np.max([0, x0-(sigma//2)])
        y[left:x0+(sigma//2)] = 1
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
    return y

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

    mdict = {prefix + "L1": l1, prefix + "L2" : l2, prefix + "th_p": thp, prefix + "ncc": ncc }
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
        scat11 = np.sum(np.abs(sxa1[order1[:10]] - sxa2[order1[:10]]))
        scat12 = np.sum(np.abs(sxa1[order1[10:20]] - sxa2[order1[10:20]]))
        scat13 = np.sum(np.abs(sxa1[order1[20:]] - sxa2[order1[20:]]))

        mdict['scat1L2'], mdict['scat2L2'] = scat1L2, scat2L2
        mdict['scat1'], mdict['scat2'] = scat1, scat2
        mdict['scat11'], mdict['scat12'], mdict['scat13']   = scat11, scat12, scat13

    return dict(mdict, **ssim)

def get_metric_fp(fp, tf_s1, prefix=''):
    meanDisp    = np.mean(np.abs(fp))
    rmseDisp    = np.sqrt(np.mean(fp ** 2))
    meanDispTFA = np.sum(np.abs(fp) * np.abs(tf_s1)) / np.sum(np.abs(tf_s1))
    meanDispTFA2 = np.sum(np.abs(fp) * np.abs(tf_s1 ** 2)) / np.sum(np.abs(tf_s1 ** 2)),
    rmseDispTF  = np.sqrt(np.sum(np.abs(tf_s1) * fp ** 2) / np.sum(np.abs(tf_s1)))
    rmseDispTF2 =  np.sqrt(np.sum(np.abs(tf_s1 ** 2) * fp ** 2) / np.sum(np.abs(tf_s1 ** 2)))

    dict_disp = {
        prefix + "meanDisp": meanDisp,
        prefix + "rmseDisp": rmseDisp,
        prefix + "meanDispTFA": meanDispTFA,
        # "meanDispTFA2": np.sum(np.abs(fp) * np.abs(tf_s1 ** 2)) / np.sum(np.abs(tf_s1 ** 2)),
        # "meanDispTFP" : np.sum(np.abs(fp) * np.angle(tf_s1)) / np.sum(np.angle(tf_s1)),  #poid negatif
        # "meanDispTFC" : np.abs( np.sum(np.abs(fp) * tf_s1) / np.sum(tf_s1) ),
        prefix + "rmseDispTF": rmseDispTF,
        prefix + "rmseDispTF2": rmseDispTF2
    }
    return dict_disp

def get_metrics(s1, s2, fp=None, scatt=None):
    mdict = get_metric(s1, s2, scattering=scatt)
    mdict_brain = get_metric(s1, s2, mask=s1, prefix='brain_')
    mdict = dict(mdict, ** mdict_brain)

    tf_s1 = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(s1)).astype(np.complex))
    tf_s2 = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(s2)).astype(np.complex))

    dict_allA = get_metric(np.abs(tf_s1), np.abs(tf_s2), prefix='tf_abs_')
    dict_allP = get_metric(np.angle(tf_s1), np.angle(tf_s2), prefix='tf_pha_')

    dict_all = dict(dict_allA, ** dict_allP)

    if fp is not None:
        dict_disp = get_metric_fp(fp, tf_s1)
        dict_all = dict(dict_all, ** dict_disp)
    return dict(mdict, **dict_all)


resolution=512
shifts = np.arange(-30,30,1); shifts_small = np.arange(-1,1,0.01)

T = resolution
J = 5  # The averaging scale is specified as a power of two, 2**J. Here, we set J = 5 to get an averaging, or maximum,
# scattering scale of 2**5 = 32 samples.
Q = 8  # we set the number of wavelets per octave, Q, to 8. This lets us resolve frequencies at a resolution of 1/8 octaves.
scattering = Scattering1D(J, T, Q)

so = get_random_2step(rampe=10, sym=True)
df = pd.DataFrame()
for a in [2,5,10,20]:
    for s in [2,4,10, 20, 40, 80, 120, 160, 200]: #np.linspace(2,200,10):
        for x0 in np.hstack([np.linspace(10,120,10), np.linspace(130,256,30)]):
            #so = get_random_2step(rampe=2, sym=True)
            s=int(s); x0 = int(x0)
            print(f'sigma {s} X0 {x0}')
            fp = corrupt_data(x0, sigma=s, amplitude=a, method='Ustep', mvt_axes=[1], center='zero', resolution=resolution)
            #fp = get_perlin(resolution=resolution, freq=16) * a
            som = simu_motion(fp, so)
            disp = l1_shfit(som,so,shifts, do_plot=False,fp=fp)
            if np.abs(disp)>0:
                fp = fp + disp
                som = simu_motion(fp, so)
                #disp2 = l1_shfit_fft(som, so, shifts_small, do_plot=False, fp=fp, loss='L2')
                #fp = fp + disp2;  som = simu_motion(fp, so);  disp+=disp2
            #plot_obj(fp, so, som)
            mydict = {"sigma":s, "x0":x0, "amplitude": a, "shift": disp}
            mydict = dict(get_metrics(so,som,fp=fp, scatt=scattering) , **mydict)
            df = df.append(mydict, ignore_index=True)

plot_obj(fp, so, som)

resolution=512
shifts = np.arange(-30,30,1)
a=10;s=80; x0=220
so = get_random_2step(rampe=2, sym=True)
fp = corrupt_data(x0, sigma=s, amplitude=a, method='Ustep', mvt_axes=[1], center='zero', resolution=resolution)
som = simu_motion(fp, so)
disp = l1_shfit(som, so, shifts, do_plot=True, fp=fp)
fp = fp + disp
som = simu_motion(fp, so)
disp = l1_shfit(som, so, shifts, do_plot=True, fp=fp)

shifts_small = np.arange(-1,1,0.01)
disp = l1_shfit_fft(som, so, shifts_small, do_plot=True, fp=fp, loss='L1')

som = simu_motion(fp, so)
s = get_metric(so,som)
plot_obj(fp, so, som)



cmap = sns.color_palette("coolwarm", len(df.sigma.unique()))
plt.figure();sns.lineplot(data=df, x="x0", y="L1", hue="sigma", legend='full', palette=cmap, style="amplitude")
sns.relplot(data=df, x="x0", y="L2", hue="sigma", legend='full', palette=cmap, col="amplitude",
                         kind="line", col_wrap=2)
plt.figure();sns.lineplot(data=df, x="x0", y="L2", hue="sigma", legend='full', palette=cmap)
plt.figure();sns.lineplot(data=df, x="x0", y="shift", hue="sigma",  legend='full', palette=cmap)
plt.figure();sns.scatterplot(data=df, x="L2", y="L1", size="x0", hue="sigma", legend='full')
plt.figure();sns.scatterplot(data=df, x="L2", y="ncc", size="x0", hue="sigma", legend='full')
plt.figure();sns.scatterplot(data=df, x="L2", y="ssim", size="x0", hue="sigma", legend='full')
plt.figure();sns.scatterplot(data=df, x="ssim", y="contrast", size="x0", hue="sigma", legend='full')
plt.figure();sns.scatterplot(data=df, x="ssim", y="structure", size="x0", hue="sigma", legend='full')

i1 = df.L2 > 15


sel_key=['tf_abs_L1', 'tf_abs_L2', 'tf_abs_ncc', 'tf_abs_ssim']
sel_key=['tf_pha_L1', 'tf_pha_L2', 'tf_pha_ncc', 'tf_pha_ssim']
sel_key=['L1', 'L2', 'ncc', 'ssim', 'scat1', 'scat2'] #, 'structure', 'contrast','luminance']
sel_key=['L1', 'L2', 'ncc', 'ssim', 'scat1L2', 'scat2L2'] #, 'structure', 'contrast','luminance']
sel_key=['brain_L1', 'brain_L2', 'brain_ncc', 'brain_ssim'] #, 'structure', 'contrast','luminance']
sel_key += ['meanDispTFA', 'rmseDispTF'] #,  'rmseDispTF2'['meanDispTFA', 'meanDispTFP', 'meanDispTFC', 'rmseDispTF']
#sel_key += ['meanDisp', 'rmseDisp']
sns.pairplot(df[sel_key], kind="scatter", corner=True)
sns.pairplot(df[sel_key + ['sel']], kind="scatter", corner=True, hue='sel')

sel_key=[]
for k in df.keys():
    if "Disp" in k :
        print(k); sel_key.append(k)



t=tio.transforms.RandomMotionFromTimeCourse(displacement_shift_strategy="center_zero", maxRot=(2,10), maxDisp=(2,10),
                                            suddenMagnitude=(2,10), swallowMagnitude=(2,10))
t.nT = resolution

fp = fitpar[1]
plt.figure()
for i in range(0,10):
    so = get_random_2step(rampe=2, sym=True)
    plt.plot(so)

for i in range(0,20):
    t._simulate_random_trajectory()
    fitpar = t.fitpars
    fp = fitpar[2] - fitpar[2,resolution//2]

    som = simu_motion(fp, so)
    plot_obj(fp, so, som)


resolution=64
for a2 in [0.1, 1, 10, 100]: #[1, 5, 10, 50]:
    plt.figure()
    leg=[]
    for a1 in [0.1, 1, 10, 100]:
        b = perlinNoise1D(32, [a1, a2])
        leg.append(f'a1 {a1}, a2 {a2}')
        x = np.linspace(0, 1, b.shape[0])
        xt = np.linspace(0, 1, resolution)
        bi = np.interp(xt, x, b)
        plt.plot(bi)
    plt.legend(leg)

plt.figure()
for i in range(0,10):
    b = perlinNoise1D(32, [5, 20])
    x = np.linspace(0,1,b.shape[0])
    xt = np.linspace(0,1,resolution)
    bi = np.interp(xt,x,b)
    plt.plot(b)


# calcul metrique dans le plan de fourier
# decomposition wawelet, pour characterise l'effet du motion (base versus haute frequence perturbation)
# parametre geometri et contrast de l'object ...
# corection global displacement dans le plan de fourier, (subvoxel)


from kymatio.numpy import Scattering1D

T = 512
so = get_random_2step(rampe=2, sym=True)
J = 5 #The averaging scale is specified as a power of two, 2**J. Here, we set J = 5 to get an averaging, or maximum,
# scattering scale of 2**5 = 32 samples.
Q = 8 # we set the number of wavelets per octave, Q, to 8. This lets us resolve frequencies at a resolution of 1/8 octaves.
sot = np.hstack([so[-50:], so[0:-50]])
scattering = Scattering1D(J, T, Q)

Sx = scattering(sot)
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
plt
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
    numerator = np.linalg.norm(s1_detail_coeffs - s2_detail_coeffs, axis=1) ** 2
    denominator = np.linalg.norm(s1_detail_coeffs, axis=1)**2 + np.linalg.norm(s2_detail_coeffs, axis=1)**2 + eps
    wnrmse = (numerator/denominator)**2
    return wnrmse.sum()

def compute_wavelets_wnrmse(s1, s2, wavelet_lists=None):
    if not wavelet_lists:
        wavelet_lists = ["db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8", "db9", "db10", "sym2",
                         "sym3", "sym4", "sym5", "sym6", "sym7", "sym8", "coif1", "coif2", "coif3", "coif4", "coif5",
                         "dmey"]
    return {wavelet_name: signal_wnrmse(s1, s2, wavelet=wavelet_name) for wavelet_name in wavelet_lists}
