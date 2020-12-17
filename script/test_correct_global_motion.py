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

def get_random_2step(rampe=0):
    sigma = [rampe, np.random.randint(10,100), np.random.randint(10,200)]
    ampli = [np.random.rand(1,1), np.random.rand(1,1)]
    x0 = np.random.randint(rampe,200)

    so = corrupt_data(x0, sigma=sigma, amplitude=ampli, method='2step', center='None', resolution=resolution)
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
#fmc = sym_imag(fm)
resolution=512

#exp 1 for different sigma gaussian motion
so = corrupt_data(50,sigma=[1, 30, 40], amplitude=[0.3, 0.5], method='2step',center='None',resolution=resolution)
#so = corrupt_data(50,sigma=30, method='Ustep',center='None',resolution=resolution)
#so = corrupt_data(50,sigma=20, method='sin')+1
do_plot=False; all_disp=[]; all_disp_d=[]; disp_sigma=[]; sigma=6
sigmas= np.hstack((np.arange(1,10,1),np.arange(10,20,1), [35, 50, 100]))
for sigma in sigmas:
    all_disp = [];    all_disp_d = [];
    fp = corrupt_data(256, sigma=sigma, center='zero', resolution=resolution, amplitude=10)
    for i in range(0,5):
        so = get_random_2step(rampe=20)
        som = simu_motion(fp,so)
        #if do_plot: plot_obj(fp, so, som)
        #plt.figure();plt.plot(fp.T)
        shifts = np.arange(-30,30,1)
        disp = l1_shfit(som,so,shifts, do_plot=do_plot,fp=fp)
        all_disp.append(disp)

        #siconv = np.convolve(so, so, mode='full'); soconva = np.abs(np.convolve(np.abs(som), so, mode='full'))
        #disp_conv = np.argmax(siconv) - np.argmax(soconva);        all_disp_d.append(disp_conv)
        #sod = np.diff(so, 1);    somd = np.diff(som, 1); disp = l1_shfit(somd,sod,shifts, do_plot=do_plot); all_disp_d.append(disp)
        if disp< -200 : #222:# or disp<=15:
            disp = l1_shfit(som, so, shifts, do_plot=True)
            plt.figure();        plt.plot(so);        plt.plot(abs(som));
            #plt.plot(abs(sod));plt.plot(abs(somd))
    plt.figure();plt.plot(all_disp); plt.plot(all_disp_d); plt.title('sigma is {}'.format(sigma))
    disp_sigma.append(max(all_disp,key=all_disp.count))
fig = plt.figure();plt.plot(sigmas,disp_sigma)
ax = ax=fig.get_axes()[0]
intervals = 2
loc = plticker.MultipleLocator(base=intervals)
ax.yaxis.set_major_locator(loc)
# Add the grid
ax.grid(True,which='major', axis='both', linestyle='-')


#TESTING realign with convolve, il amplitude=20, L1 computation is wrong
for rr in [1, 20]:
    resolution=512;        shifts = np.arange(-30,30,1)
    fp = corrupt_data(256, sigma=6, center='zerro', resolution=resolution,amplitude=10)
    so = corrupt_data(100,sigma=[rr, 50, 70], amplitude=[0.3,0.9], method='2step',center='None',resolution=resolution)
    som = simu_motion(fp, so)
    disp = l1_shfit(som, so, shifts, do_plot=True, fp=fp)

    fi = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(so)).astype(np.complex))
    # fm =_translate_freq_domain(fi, fp)
    kx = np.arange(-1, 1 + 2 / resolution, 2 / (resolution - 1))
    fp_kspace = np.exp(-1j * math.pi * kx * fp)
    fm = fi * fp_kspace
    print_fft(fm)


f,a = plt.subplots(2); a[0].plot(np.angle(fi)); a[1].plot(np.abs(fi)) ; a[1].plot(np.imag(fi))
a[0].plot(np.angle(fp_kspace)); a[1].plot(np.imag(fp_kspace))

print_fft(fm)
np.sum(fp*fi) / np.sum(fi) # ok for cst
np.sum(fp* np.abs(fi)) / np.sum(np.abs(fi)) # ok for cst
np.sum(fp * np.imag(fi)) / np.sum(np.image(fi))   # not ok for cst
np.sum(fp * np.angle(fi)) / np.sum(np.angle(fi))  # not ok for cst
np.sum(fp * np.abs(np.angle(fi))) / np.sum(np.abs(np.angle(fi))) # ok for cst

np.sum(np.angle(fi)*fp) / np.sum(fp) # not ok for cst
np.sum(np.angle(fi)* np.angle(fp_kspace)) / np.sum(np.angle(fp_kspace)) /math.pi /2 # not ok for cst
np.sum(np.imag(fi)*fp) / np.sum(fp) # not ok for cst
np.sum(np.imag(fi)* np.imag(fp_kspace)) / np.sum(np.imag(fp_kspace))  # not ok for cst

np.sum(fp_kspace*fi) / np.sum(fi)
np.sum(np.abs(fp_kspace)*np.abs(fi)) / np.sum(np.abs(fi))
np.sum(np.imag(fp_kspace) * np.imag(fi)) / np.sum(np.imag(fi))
np.sum(np.angle(fp_kspace) * np.angle(fi)) / np.sum(np.angle(fi))
np.sum(fp_kspace * np.abs(np.angle(fi))) / np.sum(np.abs(np.angle(fi)))
np.sum(np.angle(fp_kspace*fi))
np.sum( fp * np.angle(fi))
np.sum(np.angle(fp_kspace) + np.angle(fi))

#constant motion
fp = np.ones_like(so)*10
fp_kspace = np.exp(-1j * math.pi * kx * fp)

#ne marche pas
siconv = np.convolve(so,so, mode='full')
soconva = np.abs(np.convolve(np.abs(som), so, mode='full'))
disp_conv = np.argmax(siconv) - np.argmax(soconva)
plt.figure();plt.plot(siconv); plt.plot(soconva); plt.legend(['so auto convolv', 'so conv som']); plt.title('disp from max = {}'.format(disp_conv))
plt.figure(); plt.plot(so); plt.plot(abs(som));
print('displacement from convolve {}'.format(np.argmax(siconv)-np.argmax(soconva)))

centers = ['none', 'zero']
sigmas = [2,20]
for sigma in sigmas:
    for center in centers:
        fp = corrupt_data(100, sigma=sigma, center=center)
        fp_kspace = np.exp(-1j * math.pi * kx * fp)
        fp_im = np.fft.ifftshift(np.fft.ifftn(fp_kspace))
        print(f'sigma {sigma}, center {center}')
        #print_fft(fp_im)  # fp_kspace = sym_imag(fp_kspace)  ->NO
        #print_fft(fp_kspace)  # fp_kspace = sym_imag(fp_kspace) -> NO

fp = corrupt_data(256, sigma=6,center='zerro')
#fp = corrupt_data(90, sigma=20, method='step',center='zero')

#fp = np.ones_like(kx)*1
fp_kspace = np.exp(-1j * math.pi * kx * fp)
fp_im = np.fft.ifftshift(np.fft.ifftn(fp_kspace))
#print_fft(fp_kspace) #fp_kspace = sym_imag(fp_kspace)

#plt.figure(); plt.plot(np.imag(fp_kspace))
#plt.plot(np.abs(fp_im))
#print_fft(fp_im)

fm = fi * fp_kspace
som = np.fft.ifftshift(np.fft.ifftn(fm))
sconv_fft = np.fft.ifftshift(np.fft.ifftn(fi*fm))

plt.figure(); plt.plot(so); plt.plot(abs(som));
plt.figure();plt.plot(fp.T)
plt.figure(); plt.plot(np.real(fi)); plt.plot(np.imag(fi));plt.plot(np.real(fm)); plt.plot(np.imag(fm)); plt.legend(['Sr','Sim','Tr','Tim'])

#output = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)
siconv = np.convolve(so,so, mode='full')
#soconv = np.abs(np.convolve(som, so, mode='full'))
soconva = np.abs(np.convolve(np.abs(som), so, mode='full'))

plt.figure();plt.plot(siconv);
plt.plot(soconv);plt.plot(soconva);
#plt.plot(np.abs(sconv_fft)); plt.legend(['auto','conv_img','conv_fft'])
