import matplotlib.pyplot as plt, pandas as pd, seaborn as sns
import torchio as tio, torch, time
from segmentation.config import Config
from segmentation.run_model import RunModel
from nibabel.viewers import OrthoSlicer3D as ov
import glob, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, numpy.linalg as npl
import scipy.linalg as scl, scipy.stats as ss #, quaternion as nq
from scipy.spatial import distance_matrix
from util_affine import *

import nibabel as nib
from read_csv_results import ModelCSVResults
from types import SimpleNamespace
#from kymatio import HarmonicScattering3D
from types import SimpleNamespace
from script.create_jobs import create_jobs
import subprocess
from dual_quaternions import DualQuaternion
#np.set_printoptions(precision=2)
#pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1, 'display.width', 400)

def change_root_path(f_path, root_path='/data/romain/PVsynth/motion_on_synth_data/delivery_new'):
    common_dir = os.path.basename(root_path)
    ss = f_path.split('/')
    for k,updir in enumerate(ss):
        if common_dir in updir:
            break
    snew = ss[k:]
    snew[0] = root_path
    return '/'.join(snew)
def makeArray(text): #convert numpy array store in csv, back to array
    array_string = ','.join(text.replace('[ ', '[').split())
    array_string = array_string.replace('[,','[')
    return np.array(eval(array_string))
def parse_string_array(ss):
    ss=ss.replace('[','')
    ss=ss.replace(']','')
    dd = ss.split(' ')
    dl=[]
    for ddd in dd:
        if len(ddd)>1:
            dl.append(float(ddd))
    return np.array(dl)
#reasign each 6 disp key to rot_or rot vector
def disp_to_vect(s,key,type):
    if type=='trans':
        k1 = key[:-1] + '0'; k2 = key[:-1] + '1'; k3 = key[:-1] + '2';
    else:
        k1 = key[:-1] + '3'; k2 = key[:-1] + '4'; k3 = key[:-1] + '5';
    return np.array([s[k1], s[k2], s[k3]])

dircati = '/data/romain/PVsynth/motion_on_synth_data/delivery_new'
fjson = '/data/romain/PVsynth/motion_on_synth_data/test1/main.json'
out_path = '/data/romain/PVsynth//motion_on_synth_data/fit_parmCATI_raw/'
out_path = '/data/romain/PVsynth/motion_on_synth_data/fsl_coreg_rot_trans_sigma_2-256_x0_256_suj_0'
out_path = '/data/romain/PVsynth/motion_on_synth_data/fsl_coreg_along_x0_rot_origY_suj_0'#fsl_coreg_along_x0_transXYZ_suj_0'

dircati = '/network/lustre/iss02/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cati_full/delivery_new/'
fjson = '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/test1/main.json'
fjson = '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/test1/main_hcpT1.json'

out_path = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/fit_parmCATI_raw/'
out_path = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/fit_parmCATI_raw_new/'
out_path = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/fsl_coreg_rot_trans_sigma_2-256_x0_256_suj_0/'
out_path = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/fsl_coreg_along_x0_transXYZ_suj_0'
out_path = '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion_elastix/fit_parmCATI_raw_wTF2_noCano/'
resname = 'fit_parmCATI_raw_wTF2_sujT1_Amp'
resname = 'fit_parmCATI_raw_wTF_sujSynth_Amp_multiSuj'
out_path = '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion_elastix_bug_fix/' + resname

import dill
#dill.dump_session('globalsave.pkl')
#dill.load_session('globalsave.pkl')

""" run motion """
allfitpars_preproc = glob.glob(dircati+'/*/*/*/*/fitpars_preproc.txt')
allfitpars_raw = glob.glob(dircati+'/*/*/*/*/fitpars.txt')
fp_paths = allfitpars_raw  #fp_paths = dfsub.fp.values
dfsub = pd.read_csv('/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion_elastix/cati_fitpar_max_2_10.csv'); fp_paths = dfsub.fp.values

fjson = '/data/romain/PVsynth/motion_on_synth_data/test1/main.json'
param = dict();param['suj_contrast'] = 1;param['suj_noise'] = 0.01;param['suj_index'] = 0;param['suj_deform'] = 0;
param['displacement_shift_strategy']=None
sdata, tmot, config_runner = select_data(fjson, param, to_canonical=False)
image = sdata.t1.data[0]; brain_mask = sdata.brain.data[0]
fi = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)

### ###### run motion on all fitpar
split_length = 1
#param = dict();param['suj_contrast'] = [1, 3];param['suj_noise'] = [0.01];param['suj_index'] = [9999]; param['suj_deform'] = [0];
param = dict();param['suj_contrast'] = [1];param['suj_noise'] = [0.01];param['suj_index'] = [-10]; param['suj_deform'] = [0];
param['amplitude'] = [1,2,4,8];param['displacement_shift_strategy']=['1D_wFT']
params = product_dict(**param)
for ind, param in enumerate(params):
    res_name = f'{resname}_Param_{ind}'
    create_motion_job(param, split_length, fjson, out_path,fp_paths=fp_paths, res_name=res_name, type='one_motion',job_pack=10,
                      jobdir='/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion_elastix_bug_fix/')

##############"

#read results
fcsv = glob.glob(out_path+'/*/*csv')
df = [pd.read_csv(ff) for ff in fcsv]
df1 = pd.concat(df, ignore_index=True); # dfall = df1
#df1['fp'] = df1['fp'].apply(change_root_path)

#for sigma fitpars
fp_paths = [os.path.dirname(ff)  + '/fitpars_orig.txt'  for ff in fcsv]
df1['fp'] = fp_paths

#df1['srot'] = abs(df1['shift_R1']) + abs(df1['shift_R2']) + abs(df1['shift_R3']);df1['stra'] = abs(df1['shift_T1']) + abs(df1['shift_T2']) + abs(df1['shift_T3'])
df_coreg=df1
df_coreg['stra'] = np.sqrt(abs(df_coreg['shift_0'])**2 + abs(df_coreg['shift_1'])**2 + abs(df_coreg['shift_2'])**2)
df_coreg['srot'] = np.sqrt(abs(df_coreg['shift_3'])**2 + abs(df_coreg['shift_4'])**2 + abs(df_coreg['shift_5'])**2)
#abs(df1['shift_3']) + abs(df1['shift_4']) + abs(df1['shift_5'])

df_coreg=df1.sort_values(by='fp')
df_coreg['TR'] = dfall['TR']
#no more use as concatenation is done in apply_motion
#df_coreg = df1[df1.flirt_coreg==1] ; df_coreg.index = range(len(df_coreg))
#df_nocoreg = df1[df1.flirt_coreg==0]; df_nocoreg.index = range(len(df_nocoreg))
#df_nocoreg.columns = ['noShift_' + k for k in  df_nocoreg.columns]
#df_coreg = pd.concat([df_coreg, df_nocoreg], sort=True, axis=1); del(df_nocoreg)


#add max amplitude and select data
dq_list, fitpar_list = [], []
for ii, fp_path in enumerate(df_coreg.fp.values):
    fitpar = np.loadtxt(fp_path)
    amplitude_max = fitpar.max(axis=1) - fitpar.min(axis=1)
    trans = npl.norm(fitpar[:3,:], axis=0); angle = npl.norm(fitpar[3:6,:], axis=0)
    for i in range(6):
        cname = f'amp_max{i}';        df_coreg.loc[ii, cname] = amplitude_max[i]
        #cname = f'zero_shift{i}';        df_coreg.loc[ii, cname] = fitpar[i,fitpar.shape[1]//2]
        #cname = f'mean_shift{i}';        df_coreg.loc[ii, cname] = np.mean(fitpar[i,:])
        cname = f'center_Disp_{i}';        df_coreg.loc[ii, cname] = fitpar[i,fitpar.shape[1]//2]
        cname = f'mean_Disp_{i}';        df_coreg.loc[ii, cname] = np.mean(fitpar[i,:])

    #dd = distance_matrix(fitpar[:3,:].T, fitpar[:3,:].T)
    trans_diff = fitpar.T[:,None,:3] - fitpar.T[None,:,:3]  #numpy broadcating rule !
    dd = np.linalg.norm(trans_diff, axis=2)
    ddrot = np.linalg.norm(fitpar.T[:,None,3:] - fitpar.T[None,:,3:] , axis=-1)
    df_coreg.loc[ii, 'trans_max'] = dd.max(); df_coreg.loc[ii, 'rot_max'] = ddrot.max();
    df_coreg.loc[ii, 'amp_max_max'] = amplitude_max.max() #not very usefule (underestimation)
    df_coreg.loc[ii, 'amp_max_trans'] = amplitude_max[:3].max() #not very usefule (underestimation)
    df_coreg.loc[ii, 'amp_max_rot'] = amplitude_max[3:].max() #not very usefule (underestimation)
    #store max trans and max rot
    i,j = np.unravel_index(dd.argmax(), dd.shape)
    fitpar_tmax = (fitpar[:,i] - fitpar[:,j])
    df_coreg.loc[ii, 'max_translation'] = str( fitpar_tmax.tolist())
    ir,jr = np.unravel_index(ddrot.argmax(), ddrot.shape)
    fitpar_rmax = fitpar[:,ir] - fitpar[:,jr]
    df_coreg.loc[ii, 'max_rotation'] = str(fitpar_rmax)

    P = np.hstack([fitpar_tmax, [1, 1, 1, 0, 0, 0]])
    aff = spm_matrix(P, order=1)
    dq = DualQuaternion.from_homogeneous_matrix(aff)
    fitpar_list.append(fitpar_tmax)
    dq_list.append(dq)
    if not (i==ir) and (j==jr):
        print(f'suj {ii} diff max trans and rot')
    else:
        P = np.hstack([fitpar_rmax, [1, 1, 1, 0, 0, 0]])
        aff = spm_matrix(P, order=1)
        dq = DualQuaternion.from_homogeneous_matrix(aff)
        fitpar_list.append(fitpar_rmax)
        dq_list.append(dq)

isel= ( (df_coreg.trans_max>2) & (df_coreg.trans_max<10) ) | ( (df_coreg.rot_max>2) & (df_coreg.rot_max<10))
dfsub = df_coreg[isel]
dfsub.to_csv('cati_fitpar_max_2_10.csv')


plt.figure()
plt.hist(df_coreg.trans_max[df_coreg.trans_max<10],bins=100)
PP = np.array(fitpar_list)
Pmax = np.max(PP,axis=1)
PP = PP[Pmax<10,:] # filter out 79 suj
fig ,ax = plt.subplots(nrows=2,ncols=3)
legend = ['Tx','Ty','Tz','Rx','Ry','Rz']
for i in range(6):
    ii,jj = np.unravel_index(i, [2,3])
    aa = ax[ii,jj]; aa.hist(PP[:,i], bins=250); aa.set_xlim([-5,5])
    aa.title.set_text(legend[i])

dq_arr = np.array(dq_list) ; dq_arr = dq_arr[Pmax<10]
nbpts = len(dq_arr)
thetas = np.zeros(nbpts); disp = np.zeros(nbpts); axiscrew = np.zeros((3,nbpts))
line_distance = np.zeros(nbpts); origin_pts = np.zeros((3,nbpts))
trans = np.zeros(nbpts);angle_sum = np.zeros(nbpts); one_sub_dq_list=[]
for nbt, dq in enumerate(dq_arr):
    l, m, tt, dd = dq.screw()
    thetas[nbt] = np.rad2deg(tt); disp[nbt] = abs(dd)
    if npl.norm(l)<1e-10:
        origin_pts[:,nbt] = [0, 0, 0]
    else:
        origin_pts[:,nbt] = np.cross(l,m)
    axiscrew[:,nbt] = l;   line_distance[nbt] = npl.norm(m)
    P = PP[nbt,:]
    trans[nbt] = npl.norm(P[:3])
    angle_sum[nbt] = npl.norm(np.abs(P[3:6])) #too bad in maths : euler norm = teta ! but sometimes aa small shift

isel = (line_distance<300) & (trans <10) & (thetas<10); nbbins=150

fig ,ax = plt.subplots(nrows=5,ncols=3)
aa=ax[0,0]; aa.hist(axiscrew[0,isel],bins=nbbins); aa.set_ylabel('orient X');aa.axvline(axiscrew[0,isel].mean(), color='k', linestyle='dashed', linewidth=2)
aa=ax[0,1]; aa.hist(axiscrew[1,isel],bins=nbbins); aa.set_ylabel('orient Y');aa.axvline(axiscrew[0,isel].mean(), color='k', linestyle='dashed', linewidth=2)
aa=ax[0,2]; aa.hist(axiscrew[2,isel],bins=nbbins); aa.set_ylabel('orient Z');aa.axvline(axiscrew[0,isel].mean(), color='k', linestyle='dashed', linewidth=2)
aa=ax[1,0]; aa.hist(origin_pts[0,isel2],bins=nbbins); aa.set_ylabel('origin_pts X'); aa.axvline(origin_pts[0,isel2].mean(), color='k', linestyle='dashed', linewidth=2); #aa.set_xlim([-200,200])
aa=ax[1,1]; aa.hist(origin_pts[1,isel2],bins=nbbins); aa.set_ylabel('origin_pts Y'); aa.axvline(origin_pts[1,isel2].mean(), color='k', linestyle='dashed', linewidth=2); #aa.set_xlim([-200,200])
aa=ax[1,2]; aa.hist(origin_pts[2,isel2],bins=nbbins); aa.set_ylabel('origin_pts Z'); aa.axvline(origin_pts[2,isel2].mean(), color='k', linestyle='dashed', linewidth=2); #aa.set_xlim([-200,200])
aa=ax[2,0]; aa.hist(thetas[isel], bins=nbbins); aa.set_ylabel('theta'); aa.axvline(thetas.mean(), color='k', linestyle='dashed', linewidth=2)
aa=ax[2,1]; aa.hist(trans[isel], bins=nbbins); aa.set_ylabel('trans_norm'); aa.axvline(trans.mean(), color='k', linestyle='dashed', linewidth=2)
aa=ax[2,2]; aa.hist(disp[isel], bins=nbbins); aa.set_ylabel('disp'); aa.axvline(disp.mean(), color='k', linestyle='dashed', linewidth=2)
#fig, ax = plt.subplots(nrows=3,ncols=3)
aa=ax[3,0]; aa.hist(PP[isel,0],bins=nbbins); aa.set_ylabel('TX');aa.axvline(PP[isel,0].mean(), color='k', linestyle='dashed', linewidth=2); #aa.set_xlim([-4,4])
aa=ax[3,1]; aa.hist(PP[isel,1],bins=nbbins); aa.set_ylabel('TY');aa.axvline(PP[isel,1].mean(), color='k', linestyle='dashed', linewidth=2); #aa.set_xlim([-4,4])
aa=ax[3,2]; aa.hist(PP[isel,2],bins=nbbins); aa.set_ylabel('TZ');aa.axvline(PP[isel,2].mean(), color='k', linestyle='dashed', linewidth=2); #aa.set_xlim([-4,4])
aa=ax[4,0]; aa.hist(PP[isel,3],bins=nbbins); aa.set_ylabel('RX');aa.axvline(PP[isel,3].mean(), color='k', linestyle='dashed', linewidth=2); #aa.set_xlim([-4,4])
aa=ax[4,1]; aa.hist(PP[isel,4],bins=nbbins); aa.set_ylabel('RY');aa.axvline(PP[isel,4].mean(), color='k', linestyle='dashed', linewidth=2); #aa.set_xlim([-4,4])
aa=ax[4,2]; aa.hist(PP[isel,5],bins=nbbins); aa.set_ylabel('RZ');aa.axvline(PP[isel,5].mean(), color='k', linestyle='dashed', linewidth=2); #aa.set_xlim([-4,4])

tnorm = npl.norm(PP[:,:3],axis=1); rnorm = npl.norm(PP[:,3:],axis=1)
aa=ax[2,0]; aa.hist(rnorm[isel], bins=nbbins); aa.set_ylabel('euler_norm'); aa.axvline(rnorm.mean(), color='k', linestyle='dashed', linewidth=2)
aa=ax[2,1]; aa.hist(tnorm[isel], bins=nbbins); aa.set_ylabel('trans_norm'); aa.axvline(tnorm.mean(), color='k', linestyle='dashed', linewidth=2)
fig ,ax = plt.subplots(nrows=2,ncols=2)
aa=ax[0,0]; aa.scatter(disp, trans)
aa=ax[0,1]; aa.scatter(thetas, angle_sum)

#same with random affine
nbpts=10000
thetas = np.zeros(nbpts); disp = np.zeros(nbpts); axiscrew = np.zeros((3,nbpts))
line_distance = np.zeros(nbpts); origin_pts = np.zeros((3,nbpts))
trans = np.zeros(nbpts);trans_e = np.zeros(nbpts);angle_sum = np.zeros(nbpts); one_sub_dq_list=[]
PP = np.zeros((nbpts,6)); skip=0
for nbt in range(nbpts):
    #aff = get_random_afine(angle=(-5,5), trans=(-5,5), origine=(0,150), mode='quat')
    aff = get_random_afine(angle=(-5,5), trans=(-5,5), origine=(0,150), mode='quat2')
    dq = DualQuaternion.from_homogeneous_matrix(aff)
    res = get_info_from_dq(dq)
#    while res['line_dist'] > 500:
#        skip +=1
#        aff = get_random_afine(angle=(-5,5), trans=(-5,5), origine=(0,150), mode='euler2')
#        dq = DualQuaternion.from_homogeneous_matrix(aff)
#        res = get_info_from_dq(dq)

    thetas[nbt] = res['theta'];  disp[nbt] =  res['disp']
    origin_pts[:,nbt] = res['origin_pts']; axiscrew[:,nbt] = res['l'];   line_distance[nbt] = res['line_dist']
    trans[nbt] = npl.norm(res['trans'])
    P = spm_imatrix(aff, order=0) ; PP[nbt,:] = P[:6]
    trans_e[nbt] = npl.norm(P[:3])
    angle_sum[nbt] = npl.norm(np.abs(P[3:6])) #too bad in maths : euler norm = teta ! but sometimes aa small shift
isel = range(nbpts)
isel2 = line_distance<1000; print(f'skip {nbpts - np.sum(isel2)} jsut for orign_ptx')

#get dual_quaternion description angle disp screw axis ...
#select suj to perform motion sim on
ind_sel = (df_coreg.amp_max_max>2) | ((df_coreg.amp_max_max>2)) ; print(f'selecting {np.sum(ind_sel)}')

ind_sel = df_coreg.amp_max_max>20
if 'suj_origin' in dir(): del (suj_origin);
dq_list = []
for ii, fp_path in enumerate(df_coreg.fp.values):
    #if not ind_sel[ii]:        continue
    print(ii)
    fitpar = np.loadtxt(fp_path)
    nbpts = fitpar.shape[1]
    thetas = np.zeros(nbpts); disp = np.zeros(nbpts); axiscrew = np.zeros((3,nbpts))
    line_distance = np.zeros(nbpts); origin_pts = np.zeros((3,nbpts))
    trans = np.zeros(nbpts);angle_sum = np.zeros(nbpts); one_sub_dq_list=[]
    q_dist_qr= np.ones(nbpts);q_dist_qd= np.zeros(nbpts);
    if 'dq_pre' in dir(): del(dq_pre)
    for nbt in range(nbpts):
        P = np.hstack([fitpar[:, nbt], [1, 1, 1, 0, 0, 0]])
        trans[nbt] = npl.norm(P[:3])
        angle_sum[nbt] = npl.norm(np.abs(P[3:6])) #too bad in maths : euler norm = teta ! but sometimes aa small shift
        aff = spm_matrix(P, order=1)
        dq = DualQuaternion.from_homogeneous_matrix(aff)
        one_sub_dq_list.append(dq)
        l, m, tt, dd = dq.screw()

        thetas[nbt] = np.rad2deg(tt); disp[nbt] = abs(dd)
        if npl.norm(l)<1e-10:
            origin_pts[:,nbt] = [0, 0, 0]
        else:
            origin_pts[:,nbt] = np.cross(l,m)
        axiscrew[:,nbt] = l;   line_distance[nbt] = npl.norm(m)

        if 'dq_pre' not in dir():
            dq_pre=dq
        else:
            #lp, mp, ttp, ddp = dq.screw(rtol=1e-5)
            ttt = np.dot(dq.q_r.q, dq_pre.q_r.q) #dot product r* et r_pre
            q_dist_qr[nbt] = ttt #hmm ==1
            q_dist_qd[nbt] = (dq_pre*dq.inverse()).q_d.norm #not equal but same order as diff disp
            dq_pre = dq
    seuil_theta = 2
    if np.sum(thetas > seuil_theta)>0:
        if 'suj_origin' not in dir():
            sel_ind = thetas > seuil_theta
            suj_origin = origin_pts[:, sel_ind]
            suj_screew = axiscrew[:, sel_ind]
            ind_all_suj = np.ones_like(origin_pts[1, sel_ind])*ii
            keep_dq_list = list(np.array(one_sub_dq_list)[sel_ind])
        else:
            sel_ind = thetas > seuil_theta
            suj_origin = np.hstack((suj_origin,origin_pts[:, sel_ind]))
            suj_screew = np.hstack((suj_screew,axiscrew[:, sel_ind]))
            ind_all_suj = np.hstack((ind_all_suj, np.ones_like(origin_pts[1, sel_ind])*ii))
            keep_dq_list = list(np.array(one_sub_dq_list)[sel_ind])
        dq_list += keep_dq_list


    origin_diff1 = np.array([0]+[npl.norm(np.cross(origin_pts[:,nbt] ,origin_pts[:,nbt-1] ))/npl.norm(origin_pts[:,nbt]) for nbt in np.arange(1,nbpts)])
    origin_diff = origin_diff1 + np.array([0] + [npl.norm(np.cross(origin_pts[:,nbt] ,origin_pts[:,nbt+1] ))/npl.norm(origin_pts[:,nbt]) for nbt in np.arange(1,nbpts-1)]+[0])
    fig ,ax = plt.subplots(nrows=2,ncols=3)
    aa=ax[0,0]; aa.plot(origin_pts[1,:]); aa.set_ylabel('origin_pts Y');
    aa = ax[0, 1]; aa.plot(line_distance); aa.set_ylabel('line_dist');
    aa = ax[0, 2]; aa.plot(thetas); aa.set_ylabel('theta');aa.plot(angle_sum); aa.legend(['Theta','euler_sum']);
    aa = ax[1, 0]; aa.plot(disp); aa.set_ylabel('disp');aa.plot(trans); aa.legend(['Disp','trans_norm']);
    aa = ax[1, 1]; aa.plot(fitpar.T); aa.legend(range(6))
    aa = ax[1, 2]; aa.plot(origin_diff);


plt.figure();plt.plot(q_dist_qr); plt.plot(q_dist_qd)
plt.figure();plt.plot(disp, trans)
plt.figure(); plt.plot(angle_sum,thetas)
from mpl_toolkits import mplot3d

fig = plt.figure();ax = plt.axes(projection ='3d');plt.xlabel('x') ;plt.ylabel('y')
X, Y, Z = zip(*origin_pts.T); U,V,W = zip(*axiscrew.T*5)
ax.quiver(X, Y, Z, U, V, W)
fig = plt.figure();ax = plt.axes(projection ='3d');plt.xlabel('x') ;plt.ylabel('y')
ax.scatter(suj_origin[0,:],suj_origin[1,:],suj_origin[2,:])

df = pd.DataFrame()
dq_dict = [get_info_from_dq(dq) for dq in dq_list]
df = pd.DataFrame(dq_dict) #much faster than append in a for loop
fig ,ax = plt.subplots(nrows=2,ncols=2);
aa=ax[0,0]; aa.hist(df.trans.apply(lambda x: npl.norm(x)),bins=500); aa.set_ylabel('trans norm');aa.grid()
aa=ax[0,1]; aa.hist(df.theta,bins=500); aa.set_ylabel('theta');aa.grid()
aa=ax[1,0]; aa.hist(df.line_dist,bins=500); aa.set_ylabel('line_dist');aa.grid()
aa=ax[1,1]; aa.hist(df.disp,bins=500); aa.set_ylabel('disp');aa.grid()
fig.suptitle('40000 affine from CATI raw theta>2')

plt.plot(np.sort(df_coreg.amp_max_max));plt.grid()
df_coreg = df_coreg[ (df_coreg.amp_max_max>2) ];  df_coreg.index = range(len(df_coreg))



#explore mean disp for different transform (select from frmi)
df = pd.DataFrame()
for dq in dq_list:
    aff = dq.homogeneous_matrix()
    disp_norm = get_dist_field(aff, list(image.shape))
    mean_disp = np.mean(disp_norm)
    wmean_disp = np.sum(disp_norm*image.numpy())/np.sum(image.numpy())
    dqmetric = get_info_from_dq(dq)
    dqmetric['mean_field_dis'] = mean_disp
    dqmetric['wmean_field_dis'] = wmean_disp
    dqmetric['trans_norm'] = npl.norm(aff[:3,3])
    df = df.append(dqmetric, ignore_index=True)

fig ,ax = plt.subplots(nrows=2,ncols=3)
aa=ax[0,0]; aa.scatter(df.mean_field_dis, df.disp); aa.set_ylabel('disp');aa.grid()
aa=ax[0,1]; aa.scatter(df.mean_field_dis, df.theta); aa.set_ylabel('theta');aa.grid()
aa=ax[0,2]; aa.scatter(df.mean_field_dis, df.line_dist); aa.set_ylabel('line dist');aa.grid()
aa=ax[1,0]; aa.scatter(df.mean_field_dis, df.line_dist*np.deg2rad(df.theta)+df.disp); aa.set_ylabel('line dist*theta/100 + disp');aa.grid()
aa=ax[1,1]; aa.scatter(df.mean_field_dis, (df.line_dist+100)*np.deg2rad(df.theta)+df.disp); aa.set_ylabel('line dist');aa.grid()
aa=ax[1,2]; aa.scatter(df.mean_field_dis, (df.line_dist+150)*np.deg2rad(df.theta)+df.disp); aa.set_ylabel('line dist');aa.grid()
severity = (df.line_dist.values+50)*np.deg2rad(df.theta.values)+df.disp.values
ind = (severity<42) & (severity>40)
imin, imax = df[ind].mean_field_dis.argmin(), df[ind].mean_field_dis.argmax()
index1, index2 = df[ind].index[imin], df[ind].index[imax]
df.loc[index1,:]
cmap = sns.color_palette("coolwarm", len(df)) #
cmap = sns.cubehelix_palette(as_cmap=True)
fig=plt.figure();ppp = plt.scatter(df.mean_field_dis, severity,c=df.line_dist, cmap=cmap); fig.colorbar(ppp), plt.grid()


df=pd.read_csv('/data/romain/PVsynth/displacement/displacement_sphere_5mm_5deg.csv')
df.aff = df.aff.apply(lambda x: makeArray(x))
df.euler_fp = df.euler_fp.apply(lambda x: makeArray(x))

#Displacement testing
df = pd.DataFrame(); sphere_mask = get_sphere_mask(image)
brain_mask[brain_mask>0] = 1 #need pure mask
for i in range(1000):
    #aff = get_random_afine(angle=(-5,5), trans=(-5,5), origine=(0,150), mode='quat')
    #aff = get_random_afine(angle=(2,2), trans=(-5,5), origine=(0,150), mode='euler2')
    aff = get_random_afine(angle=(-5,5), trans=(-5,5), origine=(0,150), mode='euler2')

    disp_norm = get_dist_field(aff, list(image.shape))
    #disp_norm_small = get_dist_field(aff, [22,26,22], scale=8)
    disp_norm_mask = disp_norm * (brain_mask.numpy())
    mean_disp, max_disp, min_disp = np.mean(disp_norm), np.max(disp_norm_mask), np.min(disp_norm_mask[brain_mask>0])
    wmean_disp = np.sum(disp_norm*image.numpy())/np.sum(image.numpy())
    wmean_disp_mask = np.sum(disp_norm_mask)/np.sum(brain_mask.numpy())
    disp_norm_mask_sphere = disp_norm*sphere_mask
    wmean_disp_sphere = np.sum(disp_norm_mask_sphere)/np.sum(sphere_mask)
    max_disp_sphere, min_disp_sphere = np.max(disp_norm_mask_sphere), np.min(disp_norm_mask_sphere[sphere_mask>0])
    res = dict(mean_disp=mean_disp, wmean_disp=wmean_disp,wmean_disp_sphere=wmean_disp_sphere, aff=aff,
               wmean_disp_mask=wmean_disp_mask, max_disp=max_disp, min_disp=min_disp,
               max_disp_sphere=max_disp_sphere, min_disp_sphere=min_disp_sphere)
    fp = spm_imatrix(aff, order=0)[:6]
    res['euler_trans'] = npl.norm(fp[:3])
    res['euler_rot'] = npl.norm(fp[3:])
    res['euler_fp'] = fp
    #fp = spm_imatrix(aff, order=1)[:6]  #the translation vector change, but the norm is the same !!!
    #res['euler1_trans'] = npl.norm(fp[:3])
    #res['euler1_rot'] = npl.norm(fp[3:])

    res = dict(get_info_from_dq(DualQuaternion.from_homogeneous_matrix(aff)), **res)
    df = df.append(res, ignore_index=True)

axiscrew = np.vstack( df.l.values).T; origin_pts = np.vstack(df.origin_pts.values).T; thetas = df.theta.values
trans = df.trans.apply(lambda x: npl.norm(x)).values; disp = df.disp
PP=np.vstack(df.aff.apply(lambda x: spm_imatrix(x,order=0)[:6]).values)
isel2 = isel = range(trans.shape[0])

img_center = np.array(brain_mask.numpy().shape)//2
import scipy
center_ofm=[0,0,0] #scipy.ndimage.measurements.center_of_mass(brain_mask.numpy()) - img_center

df['fd_P'] = df.euler_fp.apply(lambda x : compute_FD_P(x) )
df['fd_J'] = df.aff.apply(lambda x : compute_FD_J(x, center_of_mass=center_ofm) )
df['fd_M'] = df.aff.apply(lambda x : compute_FD_max(x) )
err = df.fd_J - df.wmean_disp_sphere
err = df.euler_trans-df.disp
transrot = df.apply(lambda x: np.dot(x.l,x.trans/npl.norm(x.trans)), axis=1)
transrotN = df.apply(lambda x: np.dot(x.l,x.trans/npl.norm(x.trans)), axis=1)
transrot = abs(df.apply(lambda x: np.dot(x.l,x.trans), axis=1))
transrotN = df.apply(lambda x: npl.norm(np.cross(x.l,x.trans)), axis=1)
(df.euler_trans - np.sqrt(transrot**2+transrotN**2)).max() #identique
(transrot-abs(df.disp)).max()       #identique

plt.figure();plt.scatter(err,abs(transrot)/abs(df.euler_trans))
plt.figure();plt.scatter(err,abs(transrotN)/abs(transrot))


plt.figure();plt.scatter(df.disp,df.euler_trans)
x = df.max_disp; # df.wmean_disp_mask #df.wmean_disp_sphere # x = df.mean_disp
fig, axs = plt.subplots(nrows=3, ncols=3); nrow=0
for x_key in sel_key:
    x = df[x_key]
    aa=axs[nrow,0]; aa.scatter(x, df.fd_P); aa.grid(); aa.set_ylabel('fd_P'); aa.set_xlabel(x_key)
    aa=axs[nrow,1]; aa.scatter(x, df.fd_J); aa.grid(); aa.set_ylabel('fd_J'); aa.set_xlabel(x_key)
    aa=axs[nrow,2]; aa.scatter(x, df.fd_M); aa.grid(); aa.set_ylabel('fd_M'); aa.set_xlabel(x_key)
    nrow+=1

sel_key =[ 'wmean_disp_mask', 'wmean_disp_sphere', 'max_disp_sphere']
sel_key =['mean_disp', 'max_disp', 'wmean_disp_mask', 'wmean_disp_sphere', 'max_disp_sphere']
sns.pairplot(df[sel_key],kind="scatter", corner=True)

ind = (df.wmean_disp_sphere>29.5)&(df.wmean_disp_sphere<30.5)
dd = df.loc[ind,:]
aff0 = dd.iloc[dd.fd_J.argmin(),0]; dq0 = DualQuaternion.from_homogeneous_matrix(aff0)
aff1 = dd.iloc[dd.fd_J.argmax(),0]; dq1 = DualQuaternion.from_homogeneous_matrix(aff1)

orig =  np.vstack(df.l.values) # orig = np.vstack(df.origin_pts.values) orig = np.vstack(df.trans.values)
fig = plt.figure(); ax = plt.axes(projection ='3d')
ax.plot3D(orig[:,0], orig[:,1], orig[:,2],'ko');ax.set_ylabel('Oy');ax.set_xlabel('Ox');ax.set_zlabel('Oz')
plt.figure();plt.hist(df.euler_trans) #plt.hist(df.line_dist)# plt.hist(df.disp,bins=100)

plt.figure()
x = df.mean_disp
plt.scatter(x, df.trans.apply(lambda x: npl.norm(x)) ); plt.grid()
plt.scatter(x, df.theta ); plt.grid()
plt.scatter(x, (df.line_dist + 100) * np.deg2rad(df.theta) + df.trans.apply(lambda x: npl.norm(x)))


#let's do it directly
aff = spm_matrix([0,0,0,20,10,15,1,1,1,0,0,0],order=1)
aff_list=[]; #del(aff1)
angles = [25,25,25,25]; voxel_shifts = [[0,0,0], [-80,80,80], [8,-8,8], [80,-80,80]]
for angle,voxel_shift in zip(angles, voxel_shifts):
    aff = spm_matrix([0,0,0,angle/2,angle,angle,1,1,1,0,0,0],order=0)
    #voxel_shift = [8,8,8]
    T = spm_matrix([voxel_shift[0], voxel_shift[1], voxel_shift[2], 0, 0, 0, 1, 1, 1, 0, 0, 0], order=4);
    Ti = npl.inv(T)
    aff = np.matmul(T, np.matmul(aff, Ti))
    dq = DualQuaternion.from_homogeneous_matrix(aff)

    orig_pos = voxel_shift
    l = [0, 0, 1];    m = np.cross(orig_pos, l);
    theta = np.deg2rad(angle);    disp = 0;
    dq = DualQuaternion.from_screw(l, m, theta, disp)
    aff = dq.homogeneous_matrix()

    aff_list.append(aff)
    disp_norm = get_dist_field(aff, list(image.shape))
    disp_norm_small = get_dist_field(aff, [22,26,22], scale=8)

    mean_disp = np.mean(disp_norm)
    wmean_disp = np.sum(disp_norm*image.numpy())/np.sum(image.numpy())

    print(get_info_from_dq(dq))
    print(f'         mean disp is {mean_disp} weighted mean {wmean_disp}')

    do_plot=False
    if do_plot:
        [sx, sy, sz] = [32, 32, 32]
        disp_field = get_dist_field(aff,[32,32,32], return_vect_field=True)
        zslice=0
        vm = disp_field[:,:,:,zslice]
        [kx, ky, kz] = np.meshgrid(np.arange(0, sx, 1), np.arange(0, sy, 1), np.arange(0, sz, 1), indexing='ij')
        kxm, kym = kx[:,:,zslice], ky[:,:,zslice]
        plt.figure();plt.quiver(kxm, kym, vm[0,:,:],vm[1,:,:],width=0.001)

    dq = DualQuaternion.from_homogeneous_matrix(aff)
    l, m, tt, dd = dq.screw(rtol=1e-5)
    origin_pts = np.cross(l, m)
    X, Y, Z = origin_pts;    U, V, W = l*5;    ax.quiver(X, Y, Z, U, V, W)

fig = plt.figure();ax = plt.axes(projection ='3d');plt.xlabel('x') ;plt.ylabel('y')
origin_pts = np.cross(l,m)
X, Y, Z = origin_pts; U,V,W = l
ax.quiver(X, Y, Z, U, V, W)

import time
start=time.time()
fitpar=np.loadtxt(fp_path)
tmean, twImean = [], []
for nbt in range(fitpar.shape[1]):
    P = np.hstack([fitpar[:,nbt],[1,1,1,0,0,0]])
    aff = spm_matrix(P,order=1)
    disp_norm = get_dist_field(aff, list(image.shape))
    tmean.append(np.mean(disp_norm))
    twImean.append( np.sum(disp_norm * image.numpy()) / np.sum(image.numpy()) )

print(f'don in {time.time()-start} ')

ishape = (32,32,32)
disp_vec = get_dist_field(aff,list(ishape),return_vect_field=True)
zslice=16; [sx,sy,sz] = np.array(ishape)
[kx, ky, kz] = np.meshgrid(np.arange(0, sx, 1), np.arange(0, sy, 1), np.arange(0, sz, 1), indexing='ij')
kxm, kym, kzm = kx[:, :, zslice], ky[:, :, zslice], kz[:,:,zslice]

plt.figure();plt.quiver(kxm, kym, disp_vec[0,:,:,zslice],disp_vec[1,:,:,zslice],width=0.001)

fig = plt.figure();ax = plt.axes(projection ='3d');plt.xlabel('x') ;plt.ylabel('y')
ax.quiver(kxm,kym,kzm,disp_vec[0,:,:,zslice],disp_vec[1,:,:,zslice],disp_vec[2,:,:,zslice])

#get the data
param = dict();param['suj_contrast'] = 1;param['suj_noise'] = 0.01;param['suj_index'] = 0;param['suj_deform'] = 0;param['displacement_shift_strategy']=None
sdata, tmot, config_runner = select_data(fjson, param, to_canonical=False)
image = sdata.t1.data[0]

fi = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)
#fi_phase = np.fft.fftshift(np.fft.fft(image, axis=1)) #do not show corectly in ov
fi_phase = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(image), axis=1)) #phase in y why it is the case in random_motion ?

#compute with different weigth on fipar
for ii, fp_path in enumerate(df_coreg.fp.values):
    fitpar = np.loadtxt(fp_path)
    if fitpar.shape[1] != w_coef_short.shape[0] :
        #fitparn = interpolate_fitpars(fitpar,df_coreg.loc[ii,'TR']/1000)  #no because randMotion did not consider a TR
        fitpar = interpolate_fitpars(fitpar, len_output=w_coef_short.shape[0])

    #fitparinter = _interpolate_space_timing(fitpar, 0.004, 2.3,[218, 182],0)
    #fitparinter = _tile_params_to_volume_dims(fitparinter, list(image.shape))

    m_quat, mshift, m_lin = average_fitpar(fitpar, np.ones_like(w_coef_short))
    w_quat, wshift, w_lin = average_fitpar(fitpar, w_coef_short)
    w_quat_shaw, wshift_shaw, w_lin_shaw = average_fitpar(fitpar, w_coef_shaw)
    w2_quat, w2shift, w2_lin = average_fitpar(fitpar, w_coef_short**2)
    w2_quat_shaw, w2shift_shaw, w2_lin_shaw = average_fitpar(fitpar, w_coef_shaw**2)


    print(f'suj {ii}  working on {fp_path}')
    for i in range(0, 6):
        #ffi = fitparinter[i].reshape(-1)
        #rrr = np.sum(ffi * w_coef_flat) / np.sum(w_coef_flat)
        #already sifted    rcheck2 = df_coreg.loc[ii,f'm_wTF_Disp_{i}']
        #rcheck = df_nocoreg.loc[ii,f'm_wTF_Disp_{i}']
        #print(f'n={i} mean disp {i} = {rrr}  / {rcheck} after shift {rcheck2}')
        #cname = f'before_coreg_wTF_Disp_{i}';        df_coreg.loc[ii, cname] = rrr
        rrr2 = np.sum(fitpar[i]*w_coef_short) #/ np.sum(w_coef_short)
        #fsl_shift = df_coreg.loc[ii, f'shift_T{i + 1}'] if i < 3 else df_coreg.loc[ii, f'shift_R{i - 2}']
        #fsl_shift = 111#df_coreg.loc[ii, f'shift_{i}']
        #print(f'n={i} mean disp {i} (full/approx) = {rrr}  / {rrr2} after shift {rcheck2}')
        #cname = f'before_coreg_short_wTF_Disp_{i}';        df_coreg.loc[ii, cname] = rrr2

        cname = f'w_exp_Me_disp{i}';        df_coreg.loc[ii, cname] = mshift[i]
        cname = f'w_quat_Me_disp{i}';        df_coreg.loc[ii, cname] = m_quat[i]
        cname = f'w_eul_Me_disp{i}';        df_coreg.loc[ii, cname] = m_lin[i]

        cname = f'w_exp_TF_disp{i}';        df_coreg.loc[ii, cname] = wshift[i]
        cname = f'w_exp_shaw_disp{i}';        df_coreg.loc[ii, cname] = wshift_shaw[i]
        cname = f'w_quat_TF_disp{i}';        df_coreg.loc[ii, cname] = w_quat[i]
        cname = f'w_quat_shaw_disp{i}';        df_coreg.loc[ii, cname] = w_quat_shaw[i]
        cname = f'w_eul_TF_disp{i}';        df_coreg.loc[ii, cname] = w_lin[i]
        cname = f'w_eul_shaw_disp{i}';        df_coreg.loc[ii, cname] = w_lin_shaw[i]

        cname = f'w2_exp_TF_disp{i}';        df_coreg.loc[ii, cname] = w2shift[i]
        cname = f'w2_exp_shaw_disp{i}';        df_coreg.loc[ii, cname] = w2shift_shaw[i]
        cname = f'w2_quat_TF_disp{i}';        df_coreg.loc[ii, cname] = w2_quat[i]
        cname = f'w2_quat_shaw_disp{i}';        df_coreg.loc[ii, cname] = w2_quat_shaw[i]
        cname = f'w2_eul_TF_disp{i}';        df_coreg.loc[ii, cname] = w2_lin[i]
        cname = f'w2_eul_shaw_disp{i}';        df_coreg.loc[ii, cname] = w2_lin_shaw[i]


df_coreg.to_csv(out_path+'/df_coreg_fitparCATI_new_raw_sub.csv')
df_coreg.to_csv(out_path+'/df_coreg_fitpar_Sigmas_X256.csv')

df_coreg = pd.read_csv(out_path+'/df_coreg_fitparCATI_new_raw_sub.csv')
df_coreg['fp'] = df_coreg['fp'].apply(lambda x: change_root_path(x,root_path='/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cati_full/delivery_new'))
restore_key = ['shift_rot',  'shift_trans', 'zero_shift_trans', 'mean_shift_trans', 'zero_shift_rot','mean_shift_rot',
               'w2_exp_TF_trans', 'w2_exp_TF_rot' ]
for k in restore_key:
    df_coreg[k] = df_coreg[k].apply(lambda  x: parse_string_array(x))

key_disp = [k for k in df_coreg.keys() if 'isp_1' in k]; key_replace_length = 7  # computed in torchio
key_disp = [k for k in df_coreg.keys() if 'isp1' in k]; key_replace_length = 6  #if computed here
key_disp = ['shift_0']; key_replace_length = 2
key_disp = [ 'zero_shift0', 'mean_shift4']; key_replace_length = 1
for k in key_disp:
    new_key = k[:-key_replace_length] +'_trans'
    df_coreg[new_key] = df_coreg.apply(lambda s: disp_to_vect(s, k, 'trans'), axis=1)
    new_key = k[:-key_replace_length] +'_rot'
    df_coreg[new_key] = df_coreg.apply(lambda s: disp_to_vect(s, k, 'rot'),  axis=1)
    for ii in range(6):
        key_del = f'{k[:-1]}{ii}';  print(f'create {new_key}  delete {key_del}') #del(df_coreg[key_del])

#same but with vector
#ynames=['w_exp_TF_trans','w_exp_shaw_trans','w_quat_TF_trans','w_quat_shaw_trans','w_eul_TF_trans','w_eul_shaw_trans']
ynames=[ 'w_exp_TF_trans','w_exp_shaw_trans','w_eul_TF_trans','w_eul_shaw_trans']
ynames+=['w_exp_Me_trans', 'w_eul_Me_trans','w_exp_Me_rot', 'w_eul_Me_rot',]
ynames+=['w2_exp_TF_trans','w2_exp_shaw_trans','w2_eul_TF_trans','w2_eul_shaw_trans', 'zero_shift_trans', 'mean_shift_trans']
#ynames=['w_exp_TF_rot','w_exp_shaw_rot','w_quat_TF_rot','w_quat_shaw_rot','w_eul_TF_rot','w_eul_shaw_rot']
ynames+=['w_exp_Me_rot','w_exp_TF_rot','w_exp_shaw_rot','w_eul_TF_rot','w_eul_shaw_rot']
ynames+=['w2_exp_TF_rot','w2_exp_shaw_rot','w2_eul_TF_rot','w2_eul_shaw_rot', 'zero_shift_rot', 'mean_shift_rot']

#name from torchio metric
ynames = ['wTF_trans', 'wTF2_trans', 'wTFshort_trans', 'wTFshort2_trans', 'wSH_trans', 'wSH2_trans']
ynames += ['wTF_rot', 'wTF2_rot', 'wTFshort_rot', 'wTFshort2_rot', 'wSH_rot', 'wSH2_rot']
ynames = [ 'no_shift_wTF_trans', 'no_shift_wTF2_trans', 'no_shift_wTFshort_trans', 'no_shift_wTFshort2_trans', 'no_shift_wSH_trans', 'no_shift_wSH2_trans']
ynames += [ 'no_shift_wTF_rot', 'no_shift_wTF2_rot', 'no_shift_wTFshort_rot', 'no_shift_wTFshort2_rot', 'no_shift_wSH_rot', 'no_shift_wSH2_rot']
ynames += [ 'center_trans', 'mean_trans','center_rot', 'mean_rot']


e_yname,d_yname=[],[]
e_yname += [f'error_{yy}' for yy in ynames ]
d_yname += [f'{yy}' for yy in ynames ]
for yname in ynames:
    xname ='shift_rot' if  'rot' in yname else 'shift_trans';
    x = df_coreg[f'{xname}'];  y = df_coreg[f'{yname}'];
    cname = f'error_{yname}';
    df_coreg[cname] = (y).apply(lambda x: npl.norm(x)) #
    df_coreg[cname] = (y-x).apply(lambda x: npl.norm(x))

ind_sel = (df_coreg.trans_max<10) & (df_coreg.rot_max<10)
dfm = df_coreg[ind_sel].melt(id_vars=['fp'], value_vars=e_yname, var_name='shift', value_name='error')
dfm["ei"] = 0
for kk in  dfm['shift'].unique() :
    dfm.loc[dfm['shift'] == kk, 'ei'] = 'trans' if 'trans' in kk else 'rot'  #int(kk[-1])
    dfm.loc[dfm['shift'] == kk, 'shift'] = kk[6:-6] if 'trans' in kk else kk[6:-4]
    #dfm.loc[dfm['shift'] == kk, 'shift'] = kk[6:] if 'trans' in kk else kk[6:]
dfm.loc[dfm['shift'].str.contains('exp'),'interp'] = 'exp'
dfm.loc[dfm['shift'].str.contains('eul'),'interp'] = 'eul'
dfm.loc[dfm['shift'].str.contains('quat'),'interp'] = 'quat'
dfm.loc[dfm['shift'].str.contains('zero'),'interp'] = 'eul'
dfm.loc[dfm['shift'].str.contains('mean'),'interp'] = 'eul'
dfm['shift']=dfm['shift'].str.replace('_exp_','');dfm['shift']=dfm['shift'].str.replace('_eul_','')
dfm['shift']=dfm['shift'].str.replace('_quat_','');
#dfm['shift']=dfm['shift'].str.replace('zero_','');dfm['shift'] = dfm['shift'].str.replace('mean_', '')

sns.set_style("darkgrid")
sns.catplot(data=dfm,x='shift', y='error', col='ei', kind='boxen', col_wrap=2, dodge=True)

sns.catplot(data=dfm,x='shift', y='error', col='ei',hue='interp', kind='strip', col_wrap=2, dodge=True)
sns.catplot(data=dfm,x='shift', y='error', col='ei',hue='interp', kind='boxen', col_wrap=2, dodge=True)
sns.pairplot(df1[sel_key], kind="scatter", corner=True)

#concat for further substraction
def concat_col(x, col1, col2):
    return np.hstack([x[col1], x[col2]])
df_coreg.w2_eul_TF_trans = df_coreg.w2_eul_TF_trans.apply(lambda x: parse_string_array(x))
df_coreg.w2_eul_TF_rot = df_coreg.w2_eul_TF_rot.apply(lambda x: parse_string_array(x))
df_coreg.subtract_fit = df_coreg.apply(lambda x: concat_col(x,'w2_eul_TF_trans','w2_eul_TF_rot'), axis=1)

#plot some exp lin quat diff
(df_coreg.w_eul_Me_trans - df_coreg.mean_shift_trans).apply(lambda x:npl.norm(x)).max() #max is 0.0197 because fitpar interp
print(f'euler / exp Trans  {(df_coreg.w_eul_Me_trans - df_coreg.w_exp_Me_trans).apply(lambda x: npl.norm(x)).max()}')
print(f'euler / exp Rot    {(df_coreg.w_eul_Me_rot - df_coreg.w_exp_Me_rot).apply(lambda x:npl.norm(x)).max()}')
print(f'euler / quat Trans {(df_coreg.w_eul_Me_trans - df_coreg.w_quat_Me_trans).apply(lambda x:npl.norm(x)).max()}')
print(f'euler / quat Rot   {(df_coreg.w_eul_Me_rot  - df_coreg.w_quat_Me_rot).apply(lambda x: npl.norm(x)).max()}')
print(f'exp / quat Trans {(df_coreg.w_exp_Me_trans - df_coreg.w_quat_Me_trans).apply(lambda x:npl.norm(x)).max()}')
print(f'exp / quat Rot   {(df_coreg.w_exp_Me_rot  - df_coreg.w_quat_Me_rot).apply(lambda x: npl.norm(x)).max()}')
#max error for i=298
errT = (df_coreg.w2_exp_TF_trans - df_coreg.shift_trans).apply(lambda x:npl.norm(x))
errR = (df_coreg.w2_exp_TF_rot - df_coreg.shift_rot).apply(lambda x:npl.norm(x))

#explore max error on CATI fit
df_coreg['error_w_exp_shaw_trans']
np.sort(df_coreg['error_w_exp_shaw_trans'].values)[::-1][:50]
ind_sel = np.argsort(df_coreg['error_w_exp_shaw_rot'].values)[::-1][:10];
ind_sel=np.where(errT>1)[0]
do_plot = False
for ii in ind_sel:
    print(f'row {ii} loading {df_coreg.fp.values[ii]}')
    fitpar = np.loadtxt(df_coreg.fp.values[ii])
    if do_plot:
        fig=plt.figure()
        plt.plot(fitpar.T); plt.legend(['tx','ty','tz','rx','ry','rz']); plt.grid()
        center = fitpar.shape[1]//2
        ax=fig.get_axes()[0]; plt.plot([center, center], ax.get_ylim(),'k')
    perform_one_motion(df_coreg.fp.values[ii], fjson, fsl_coreg=True, return_motion=False, root_out_dir='/data/romain/PVsynth/motion_on_synth_data/fit_parmCATI_raw_saved')
#visu check 92 368 1015
fcsv = glob.glob('/data/romain/PVsynth/motion_on_synth_data/fit_parmCATI_raw_saved/*/*csv')
dfres = pd.concat(   [pd.read_csv(ff) for ff in fcsv] , ignore_index=True);
dfres['shift_trans'] = dfres.apply(lambda s: disp_to_vect(s, 'shift_0', 'trans'), axis=1)
dfres['shift_rot'] = dfres.apply(lambda s: disp_to_vect(s, 'shift_0', 'rot'), axis=1)

df3 = pd.merge(dfres, df_coreg, on='fp')
df3.sort_values('m_L1_map_x', inplace=True)

import statsmodels.api as sm
for yname in ynames:
    #plot
    fig, axs = plt.subplots(nrows=2,ncols=3, figsize=(14,8));
    max_errors=0
    for  i, ax in enumerate(axs.flatten()):
        #fsl_shift = df_coreg[ f'shift_T{i + 1}'] if i < 3 else df_coreg[ f'shift_R{i - 2}']
        fsl_shift =  df_coreg[ f'shift_{i}']
        xname ='shift_' #'before_coreg_short_wTF_Disp_'#'no_shift_wTF_Disp_' # 'w_expTF_disp' #'m_wTF2_Disp_' #
        #yname = 'w_quat_shaw_disp' #'no_shift_wTF2_Disp_' #'before_coreg_wTF_Disp_'  #
        #x = df_coreg[f'm_wTF2_Disp_{i}'] + fsl_shift
        #yname = 'no_shift_wTF2_Disp_'
        #xname = 'wTFshort_Disp_' #'w_exp_shaw_disp'
        #yname = 'wTF_Disp_' #'w_quat_shaw_disp'
        #x = fsl_shift if xname=='fsl_shift' else df_coreg[f'{xname}{i}'];
        x = df_coreg[f'{xname}{i}']
        y = df_coreg[f'{yname}{i}'];
        #y=df_coreg[f'wTF_Disp_{i}'] + fsl_shift   #identic

        #ax.scatter(x,y);ax.plot([x.min(),x.max()],[x.min(),x.max()]);
        sm.graphics.mean_diff_plot(y,x, ax=ax);plt.tight_layout()
        #ax.hist(y-x, bins=64);
        max_error = np.max(np.abs(y-x)); mean_error = np.mean(np.abs(y-x))
        max_errors = max_error if max_error>max_errors else max_errors
        corrPs, Pval = ss.pearsonr(y,x)
        print(f'cor is {corrPs} P {Pval} max error for {i} is {max_error}')
        ax.title.set_text(f'R = {corrPs:.2f} err mean {mean_error:.2f} max {max_error:.2f}')
    fig.text(0.5, 0.04, xname, ha='center')
    fig.text(0.04, 0.5, yname, va='center', rotation='vertical')
    print(f'Yname {yname} max  is {max_errors}')
    # m_wTF_Disp_ +fsl_shift ==  before_coreg_wTF_Disp_

plt.figure();plt.scatter(df_coreg.w_exp_shaw_disp5, df_coreg.w_quat_shaw_disp5)
err = df_coreg.w_exp_shaw_disp5 - df_coreg.w_quat_shaw_disp5
df_coreg.fp.Valeus[err.argmax()]
wshift_quat, wshift_exp, lin_fitpar = average_fitpar(fitpar)

#same comparison but withe trans and rot vector difference

#plot sigmas
ykeys=['m_grad_H_camb','m_grad_H_nL2','m_grad_H_corr','m_grad_H_dice','m_grad_H_kl','m_grad_H_jensen','m_grad_H_topsoe','m_grad_H_inter']
for k in ykeys:
    print(np.sum(np.isnan(df_coreg[k].values)))

ykeys=['error_w_exp_TF_rot', 'error_w_exp_shaw_rot', 'error_w2_exp_TF_rot', 'error_w2_exp_shaw_rot', 'error_zero_shift_rot', 'error_mean_shift_rot']
ykeys=['stra','m_L1_map', 'm_NCC', 'm_ssim_SSIM', 'm_grad_ratio', 'm_nRMSE', 'm_grad_nMI2', 'm_grad_EGratio','m_grad_cor_diff_ratio']
ykeys=['stra','m_L1_map', 'm_NCC', 'm_ssim_SSIM', 'm_grad_ratio', 'm_nRMSE', 'm_grad_nMI2', 'm_grad_EGratio','m_grad_cor_diff_ratio']
ykeys_noshift = ['no_shift_'+ kk[2:] for kk in ykeys]
cmap = sns.color_palette("coolwarm", len(df_coreg.amplitude.unique()))
ykey = 'm_NCC_brain' #ykeys_noshift[0]
for ykey in ykeys:
    fig = sns.relplot(data=df_coreg, x="sigma", y=ykey, hue="amplitude", legend='full', kind="line",
                      palette=cmap, col='mvt_axe', col_wrap=2)
                    #palette=cmap, col='mvt_type', col_wrap=2)

#different sigma
dfsub = df_coreg[df_coreg['sym']==1]
dfsub = df_coreg[df_coreg['no_shift']==0]

#dfsub = df_coreg[(df_coreg['sym']==0) & (df_coreg['no_shift']==0)]
cmap = sns.color_palette("coolwarm", len(df_coreg.sigma.unique()))
for ykey in ykeys:
    fig = sns.relplot(data=dfsub, x="xend", y=ykey, hue="sigma", legend='full', kind="line",
                  palette=cmap, col='amplitude', col_wrap=2, style='mvt_axe') #, style='no_shift')
    for ax in fig.axes: ax.grid()


#plot_volume index x0 for different sigma
sigmas = [2, 4,  8,  16,  32, 64, 128] #np.linspace(2,256,128).astype(int), # [2, 4,  8,  16,  32, 44, 64, 88, 128], ,
x0_min, nb_x0s = 0, 32
resolution, sym = 512, False
plt.figure();plt.grid()
for sigma in sigmas:
    if sym:
        xcenter = resolution // 2 #- sigma // 2;
        # x0s = np.floor(np.linspace(xcenter - x0_min, xcenter, nb_x0s))
        x0s = np.floor(np.linspace(x0_min, xcenter, nb_x0s))
        x0s = x0s[x0s >= sigma // 2]  # remove first point to not reduce sigma
        x0s = x0s[x0s <= (xcenter - sigma // 2)]  # remove last points to not reduce sigma because of sym
    else:
        xcenter = resolution // 2;
        x0s = np.floor(np.linspace(x0_min, xcenter, nb_x0s))
        x0s = x0s[x0s >= sigma // 2]  # remove first point to not reduce sigma
    plt.plot(np.array(x0s), range(len(x0s)))
    print(f'sigma  {sigma} nb_x {len(x0s)}')
    print(x0s)
plt.legend(sigmas)


# expand no_shift column to lines
col_no_shift, new_col_name = [], [];
col_map = dict()
for k in df1.keys():
    if 'no_shift' in k:
        col_no_shift.append(k)
        newc = 'm_' +  k[9:]
        if newc not in df1:
            newc = k[9:]
            if newc not in df1:
                print(f'error with {newc}')
        new_col_name.append(newc)
        col_map[k] = newc

dfsub2 = df1.copy() # df1[col_no_shift].copy()
dfsub1 = df1.copy()
dfsub1=dfsub1.drop(col_no_shift,axis=1)
dfsub2=dfsub2.drop(new_col_name,axis=1)
dfsub2 = dfsub2.rename(columns = col_map)
dfsub1.loc[:,'no_shift'] = 0; dfsub2.loc[:,'no_shift'] = 1;
df_coreg = pd.concat([dfsub1, dfsub2], axis=0, sort=True)
df_coreg.index = range(len(df_coreg))





def get_sujname_output_from_sigma_params(df, out_path, filename):
    for i, row in df.iterrows():
        suj_name = f'Suj_{row["subject_name"]}_I{int(row["suj_index"])}_C{int(row["suj_contrast"])}_N_{int(row["suj_noise"] * 100)}_D{int(row["suj_deform"]):d}_S{int(row["suj_seed"])}'
        amplitude_str = f'{row["amplitude"]}'
        extend=False
        if len(amplitude_str)>2:
            amplitude_str =amplitude_str[:-2] + '*'
            extend=True
        mvt_type = row["mvt_type"] if row['sym'] is False else 'Ustep' #arg todo correct right name in generation
        fp_name  = f'fp_x{int(row["x0"])}_sig{int(row["sigma"])}_Amp{amplitude_str}_M{mvt_type}_A{row["mvt_axe"]}_sym{int(row["sym"])}'
        suj_name += fp_name
        out_dir =  out_path + '/' + suj_name + '/' + filename
        if extend:
            oo = glob.glob(out_dir)
            if len(oo)!=1:
                print(f'ERROR  four {len(oo)} for {out_dir}')
            else:
                out_dir = oo[0]
        df.loc[i,'out_dir'] = out_dir
    return df

df1 = get_sujname_output_from_sigma_params(df1,out_path,'vol_motion_no_shift.nii')
mv_ax='rotY'
dfsub = df1[df1['mvt_axe'] == mv_ax]

amp_val = np.sort(dfsub.amplitude.unique())
#write output 4D for center x0
for mv_ax in df1['mvt_axe'].unique():
    dfsub = df1[df1['mvt_axe'] == mv_ax]
    amp_val = np.sort(dfsub.amplitude.unique())
    sym = 1
    for ii, amp in enumerate(amp_val):
        dd = dfsub[dfsub['amplitude']==amp]
        ddd = dd.sort_values(axis=0,by="sigma")
        out_volume = f'out_put_noshift{mv_ax}_{int(amp*100)}_sym{sym}.nii'
        print(f'writing {out_volume}')
        cmd = f'fslmerge -t {out_path}/{out_volume} '
        for vv in ddd['out_dir']:
            cmd += vv + ' '
        cmd = cmd[:-1]
        outvalue = subprocess.run(cmd.split(' '))


#write output 4D for varying x0
amp_val = np.sort(dfsub.amplitude.unique())
for mv_ax in df1['mvt_axe'].unique():
    dfsub = df1[df1['mvt_axe'] == mv_ax]
    amp_val = np.sort(dfsub.amplitude.unique())
    sym=1
    for ii, amp in enumerate(amp_val):
        dfsub_sig = dfsub[dfsub['amplitude']==amp]
        sigmas = np.sort(dfsub_sig.sigma.unique())
        for sig in sigmas:
            dd = dfsub_sig[dfsub_sig['sigma']==sig]
            ddd = dd.sort_values(axis=0,by="x0")
            #out_volume = f'out_put_noshift{mv_ax}_S{sig}_A{int(amp*100)}_sym{sym}.nii.nii'
            out_volume = f'out_put_fitpar_{mv_ax}_S{sig}_A{int(amp*100)}_sym{sym}.nii.nii'
            print(f'writing {out_volume}')
            cmd = f'fslmerge -t {out_path}/{out_volume} '
            for vv in ddd['out_dir']:
                cmd += vv + '/vol_motion.nii '
                #cmd += vv + '/vol_motion_no_shift.nii '
            cmd = cmd[:-1]
            outvalue = subprocess.run(cmd.split(' '))



one_df, smot = perform_one_motion(fp_path, fjson, fsl_coreg=False, return_motion=True)
tma = smot.history[2]

mres = ModelCSVResults(df_data=one_df, out_tmp="/tmp/rrr")
keys_unpack = ['transforms_metrics', 'm_t1'];
suffix = ['m', 'm_t1']
one_df1 = mres.normalize_dict_to_df(keys_unpack, suffix=suffix);
one_df1.m_wTF_Disp_1



#screw motion
json_file='/data/romain/PVsynth/motion_on_synth_data/test1/main.json'
df=pd.DataFrame()
param = {'amplitude': 8, 'sigma': 128, 'nb_x0s': 1, 'x0_min': 206, 'sym': False, 'mvt_type': 'Ustep',
 'mvt_axe': [6], 'cor_disp': False, 'disp_str': 'no_shift', 'suj_index': 0, 'suj_seed': 1, 'suj_contrast': 1,
 'suj_deform': False, 'suj_noise' : 0.01}
pp = SimpleNamespace(**param)

amplitude, sigma, sym, mvt_type, mvt_axe, cor_disp, disp_str, nb_x0s, x0_min =  pp.amplitude, pp.sigma, pp.sym, pp.mvt_type, pp.mvt_axe, pp.cor_disp, pp.disp_str, pp.nb_x0s, pp.x0_min
resolution, sigma, x0 = 218, int(sigma), 109
extra_info = param;
fp = corrupt_data(x0, sigma=sigma, method=mvt_type, amplitude=amplitude, mvt_axes=mvt_axe,center='none', return_all6=True, sym=sym, resolution=resolution)
testm = spm_matrix(np.hstack([fp[:,100],[1,1,1,0,0,0]]),order=0)

dq = DualQuaternion.from_homogeneous_matrix(testm)
orig_pos = [0, -80, 0] #np.array([90, 28, 90]) - np.array([90,108, 90])
#l=[1,0,0]; m = np.cross(l,orig_pos); theta = np.deg2rad(2); disp=0;
l=[0,0,1]; m = np.cross(orig_pos,l); theta = np.deg2rad(8); disp=0;  #bad this inverse origin
dq = DualQuaternion.from_screw(l, m, theta, disp)
fitpar =  np.tile(spm_imatrix(dq.homogeneous_matrix(), order=0)[:6,np.newaxis],(1,218))
cor_disp=True
smot8, df, res_fitpar, res = apply_motion(sdata, tmot, fitpar, config_runner,suj_name='rotfront_Z_2',
                                          root_out_dir='/home/romain.valabregue/tmp', param=param)

#apply motion on fp tremor
fjson = '/data/romain/PVsynth/motion_on_synth_data/test1/main.json'
param = dict();param['suj_contrast'] = 1;param['suj_noise'] = 0.01;param['suj_index'] = 0;param['suj_deform'] = 0;param['displacement_shift_strategy']=None
sdata, tmot, config_runner = select_data(fjson, param, to_canonical=False)
image = sdata.t1.data[0]; brain_mask = sdata.brain.data[0]

dout = '/data/romain/PVsynth/motion_on_synth_data/tremor/'
fp_file = glob.glob(dout+'/files_rp/rp*txt')
df=pd.DataFrame()
for fpf in fp_file:
    fp = np.loadtxt(fpf)
    tmot.nT = fp.shape[1];    tmot.simulate_displacement = False
    fout = f'{os.path.basename(fpf)[:-4]}'
    root_out_dir = dout+fout;
    if not os.path.isdir(root_out_dir): os.mkdir(root_out_dir)
    df = apply_motion(sdata, tmot, fp, config_runner,root_out_dir=root_out_dir,param=param,
                                              fsl_coreg=True,suj_name='suj')
    #sout.t1.save(dout + fout+'.nii')



#CATI fitpar
import json
ljson = json.load(open(dircati_rrr + 'summary.json'))
dircati_rrr = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/CATI_datasets/fmri/'
dfall = pd.read_csv(dircati_rrr+'/description_copie.csv')
dfall['sujname'] = dfall['Fitpars_path'].apply(get_sujname_only_from_path);
dfall['session'] = dfall['Fitpars_path'].apply(get_session_from_path);
dfall['center'] =  dfall['Fitpars_path'].apply(get_center_from_path);
dfall['cati_path'] = dfall.apply(lambda x : get_cati_path(x,ljson), axis=1)
#remove suj not found
dfall = dfall.dropna()

dfall['brain_mask'] = dfall.cati_path.apply(get_cati_brain_mask)
dfall.to_csv(dircati_rrr+'/data_all.csv')

def get_cati_path(x, ljson):
    proto = x['Protocol']
    center = x['center']
    suj = x['sujname']
    session = x['session']

    if proto in ljson:
        ljson = ljson[proto]
    else:
        print(f'missin proto {proto}'); return None

    if center in ljson:
        ljson = ljson[center]
    else:
        print(f'missin proto {proto} center {center}'); return None

    if suj in ljson:
        ljson = ljson[suj]
    else:
        print(f'missin proto {proto} center {center} suj {suj}'); return None
    if session in ljson:
        dir_path = ljson[session]
    else:
        print(f'missiong proto {proto} center {center} suj {suj} sess {session}')
        return None
    dir_path = dir_path + '/input/'
    if not os.path.isdir(dir_path):
        print(f'path {dir_path} not exist')
    return dir_path
def get_session_from_path(ff):
    name = [];
    dn = os.path.dirname(ff)
    dn = os.path.basename(dn)
    return dn
def get_sujname_only_from_path(ff):
    name = [];
    dn = os.path.dirname(ff)
    dn = os.path.dirname(dn)
    dn = os.path.basename(dn)
    return dn
def get_center_from_path(ff):
    name = [];
    dn = os.path.dirname(ff)
    dn = os.path.dirname(dn)
    dn = os.path.dirname(dn)
    dn = os.path.basename(dn)
    return dn
def get_cati_brain_mask(ff):
    vol_mask = ff+'/rsfMRI_brain.nii.gz'
    if os.path.isfile(vol_mask):
        return vol_mask
    else:
        print(f'missing {vol_mask}')
        return None
def get_sujname_from_path(ff):
    name = [];
    dn = os.path.dirname(ff)
    for k in range(3):
        name.append(os.path.basename(dn))
        dn = os.path.dirname(dn)
    return '_'.join(reversed(name))

#read from global csv (to get TR)
dfall = pd.read_csv(dircati_rrr+'/data_all.csv') #dircati+'/description_copie.csv')
#dfall['Fitpars_path'] = dfall['Fitpars_path'].apply(change_root_path)
dfall['sujname_all'] = dfall['Fitpars_path'].apply(get_sujname_from_path); #dfall['sujname_all'] = out_path+dfall['resdir']
dfall.fp = dfall.Fitpars_path

allfitpars_raw = dfall['Fitpars_path']
afffitpars_preproc = [os.path.dirname(p) + '/fitpars_preproc.txt' for p in allfitpars_raw]
dfall = dfall.sort_values(by=['Fitpars_path'])

df_coreg.to_csv(dircati_rrr+'/data_all_param.csv')

df_big = df_coreg[(df_coreg['trans_max']>10) | (df_coreg['rot_max']>10) ]
dfsub = df_coreg[(df_coreg['trans_max']<=10) & (df_coreg['rot_max']<=10) ]
plt.figure();plt.hist(dfsub.rot_max,bins=100)
plt.figure();plt.hist(dfsub.trans_max,bins=100)
df_medium = dfsub[ (dfsub['trans_max']>2) | (dfsub['rot_max']>2) ]


nbsuj = len(df_medium)
trans = np.zeros(nbsuj);angle_sum = np.zeros(nbsuj);thetas = np.zeros(nbsuj);disp = np.zeros(nbsuj);
origin_pts= np.zeros((3,nbsuj)); axiscrew= np.zeros((3,nbsuj)); line_distance= np.zeros(nbsuj); y_min = np.zeros(nbsuj)
euler_list = np.zeros((6,nbsuj))
for ii, fp in enumerate(df_medium.Fitpars_path):
    #fitpar  = np.loadtxt(fp)
    #P = np.hstack([fitpar[:, nbt], [1, 1, 1, 0, 0, 0]])
    P =  np.hstack([parse_string_array(df_medium.max_rotation.values[ii]), [1, 1, 1, 0, 0, 0]])
    euler_list[:,ii] = P[:6]
    trans[ii] = npl.norm(P[:3])
    angle_sum[ii] = npl.norm(np.abs(P[3:6]))  # too bad in maths : euler norm = teta ! but sometimes aa small shift
    aff = spm_matrix(P, order=1)
    dq = DualQuaternion.from_homogeneous_matrix(aff)
    #one_sub_dq_list.append(dq)
    l, m, tt, dd = dq.screw()

    thetas[ii] = np.rad2deg(tt);
    disp[ii] = abs(dd)
    if npl.norm(l) < 1e-10:
        origin_pts[:, ii] = [0, 0, 0]
    else:
        origin_pts[:, ii] = np.cross(l, m)
    axiscrew[:, ii] = l;
    line_distance[ii] = npl.norm(m)

    mask_path = os.path.dirname(df_medium.brain_mask.values[ii]) + '/rsfMRI_brain_mask.nii.gz'
    v=nb.load(mask_path)
    aff=v.affine;
    M=aff[:3,:3]; voxel_size = np.diagonal(np.sqrt(M @ M.T))
    data=v.get_fdata()
    mask = data>0;    mm = mask.sum(axis=2).sum(axis=0)
    #find first non zero
    kk = -1; y=0
    while y==0:
        kk+=1
        y = mm[kk]
    y_min[ii] = (kk - data.shape[1]//2 ) * voxel_size[1]

fig = plt.figure();ax = plt.axes(projection ='3d');plt.xlabel('x') ;plt.ylabel('y')
X, Y, Z = zip(*origin_pts.T); U,V,W = zip(*axiscrew.T*1)
ax.quiver(X, Y, Z, U, V, W)

fig = plt.figure();ax = plt.axes(projection ='3d');plt.xlabel('x') ;plt.ylabel('y')
ax.scatter(axiscrew[0,:],axiscrew[1,:],axiscrew[2,:])

fig ,ax = plt.subplots(nrows=2,ncols=2)
aa=ax[0,0]; aa.hist(origin_pts[0,:],bins=100); aa.set_ylabel('origin_pts X');aa.grid()
#aa=ax[0,1]; aa.hist(origin_pts[1,:],bins=100); aa.set_ylabel('origin_pts Y');aa.grid()
aa=ax[0,1]; aa.hist(origin_pts[1,:]-y_min,bins=100); aa.set_ylabel('origin_pts Y');aa.grid()
aa=ax[1,0]; aa.hist(origin_pts[2,:],bins=100); aa.set_ylabel('origin_pts Z');aa.grid()
aa=ax[1,1]; aa.hist(thetas,bins=100); aa.set_ylabel('theta');aa.grid()

fig ,ax = plt.subplots(nrows=2,ncols=2)
aa=ax[0,0]; aa.hist(axiscrew[0,:],bins=100); aa.set_ylabel('suj_screew X');aa.grid()
aa=ax[0,1]; aa.hist(axiscrew[1,:],bins=100); aa.set_ylabel('suj_screew Y');aa.grid()
aa=ax[1,0]; aa.hist(axiscrew[2,:],bins=100); aa.set_ylabel('suj_screew Z');aa.grid()
aa=ax[1,1]; aa.hist(trans,bins=100); aa.set_ylabel('trans');aa.grid()

fig ,ax = plt.subplots(nrows=2,ncols=3)
aa=ax[0,0]; aa.hist(euler_list[0,:],bins=100); aa.set_ylabel('TX');aa.grid()
aa=ax[0,1]; aa.hist(euler_list[1,:],bins=100); aa.set_ylabel('TY');aa.grid()
aa=ax[0,2]; aa.hist(euler_list[2,:],bins=100); aa.set_ylabel('TZ');aa.grid()
aa=ax[1,0]; aa.hist(euler_list[3,:],bins=100); aa.set_ylabel('RX');aa.grid()
aa=ax[1,1]; aa.hist(euler_list[4,:],bins=100); aa.set_ylabel('RY');aa.grid()
aa=ax[1,2]; aa.hist(euler_list[5,:],bins=100); aa.set_ylabel('RZ');aa.grid()

plt.figure(); plt.scatter(trans, disp); plt.grid()
plt.figure(); plt.scatter(angle_sum, thetas); plt.grid()