import torch.nn.functional as F
import torch, math, time, os
import numpy as np
import pandas as pd
from segmentation.config import Config
from read_csv_results import ModelCSVResults
from itertools import product
from types import SimpleNamespace
import torchio as tio

pi = torch.tensor(3.14159265358979323846)

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    #for instance in itertools.product(*vals):
    #    yield dict(zip(keys, instance))
    #return (dict(zip(keys, values)) for values in product(*vals))  #this yield a generator, but easier with a list -> known size
    return list(dict(zip(keys, values)) for values in product(*vals))

def apply_motion(sin, tmot, fp, df, res_fitpar, res, extra_info, config_runner,
                 displacement_shift_strategy='None', shifts=range(-15, 15, 1),
                 correct_disp=True):
    start = time.time()
    tmot.nT = fp.shape[1]
    tmot.simulate_displacement = False
    tmot.fitpars = fp
    tmot.displacement_shift_strategy = displacement_shift_strategy
    sout = tmot(sin)

    l1_loss = torch.nn.L1Loss()

    # compute and record L1 shift
    if correct_disp:
        data_ref = sin.t1.data
        data = sout.t1.data
        dimy = data.shape[2]
        l1dist = [];
        res_extra_info = extra_info.copy()
        for shift in shifts:
            if shift < 0:
                d1 = data[:, :, dimy + shift:, :]
                d2 = torch.cat([d1, data[:, :, :dimy + shift, :]], dim=2)
            else:
                d1 = data[:, :, 0:shift, :]
                d2 = torch.cat([data[:, :, shift:, :], d1], dim=2)
            l1dist.append(float(l1_loss(data_ref, d2).numpy()))
            res_extra_info['L1'], res_extra_info['vox_disp'] = l1dist[-1], shift
            res = res.append(res_extra_info, ignore_index=True, sort=False)

        disp = shifts[np.argmin(l1dist)]
        extra_info['shift'] = disp

        if disp > 0:
            print(f' redoo motion disp is {disp}')
            fp[1, :] = fp[1, :] - disp
            tmot.fitpars = fp
            sout = tmot(sin)

    batch_time = time.time() - start
    df, report_time = config_runner.record_regression_batch(df, sout, torch.zeros(1).unsqueeze(0), torch.zeros(1).unsqueeze(0),
                                                 batch_time, save=False, extra_info=extra_info)
    # record extra_info in df
    #last_line = df.shape[0] - 1
    #for k, v in extra_info.items():
    #    if k not in df:
    #        df[k] = v
    #    else:
    #        df.loc[last_line, k] = v

    # record fitpar
    res_extra_info = extra_info.copy()
    fit_pars = tmot.fitpars  # - np.tile(tmot.to_substract[..., np.newaxis],(1,200))
    dff = pd.DataFrame(fit_pars.T);
    dff.columns = ['x', 'trans_y', 'z', 'r1', 'r2', 'r3'];
    dff['nbt'] = range(0, tmot.nT)
    for k, v in res_extra_info.items():
        dff[k] = v
    res_fitpar = res_fitpar.append(dff, sort=False)

    return sout, df, res_fitpar, res

def select_data(json_file, param=None):
    result_dir= os.path.dirname(json_file) +'/rrr' #/data/romain/PVsynth/motion_on_synth_data/test1/rrr/'
    config = Config(json_file, result_dir, mode='eval', save_files=False) #since cluster read (and write) the same files, one need save_file false to avoid colusion
    config.init()
    mr = config.get_runner()

    if param is not None:
        suj_ind = param['suj_index']
        if 'suj_seed' in param:
            suj_seed = param['suj_seed']
            np.random.seed(suj_seed)
            torch.manual_seed(suj_seed)
        else:
            suj_seed=-1
        contrast = param['suj_contrast']
        suj_deform = param['suj_deform']
        noise = param['suj_noise']
    else:
        suj_ind, contrast, suj_deform, suj_noise = 0, 1, False, 0

    s1 = config.train_subjects[suj_ind]
    print(f'loading suj {s1.name} with contrast {contrast} deform is {suj_deform} sujseed {suj_seed}')

    transfo_list = config.train_transfo_list

    #same label, random motion
    tsynth = transfo_list[0]

    if contrast==1:
        tsynth.mean = [(0.6, 0.6), (0.1, 0.1), (1, 1), (0.6, 0.6), (0.6, 0.6), (0.6, 0.6), (0.6, 0.6), (0.6, 0.6), (0.6, 0.6), (0.6, 0.6),
         (0.9, 0.9), (0.6, 0.6), (1, 1), (0.2, 0.2), (0.4, 0.4), (0, 0)]
    elif contrast ==2:
        tsynth.mean = [(0.5, 0.6), (0.1, 0.2), (0.9, 1), (0.5, 0.6), (0.5, 0.6), (0.5, 0.6), (0.5, 0.6), (0.5, 0.6), (0.5, 0.6), (0.5, 0.6),
         (0.8, 0.9), (0.5, 0.6), (0.9, 1), (0.2, 0.3), (0.3, 0.4), (0, 0)]
    elif contrast ==3:
        tsynth.mean = [(0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9),
         (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0, 0)]
    else:
        raise(f'error contrast not define {contrast}')

    ssynth = tsynth(s1)
    tmot = transfo_list[1]
    if suj_deform:
        taff_ela = transfo_list[2]
        ssynth = taff_ela(ssynth)
    if suj_noise:
        tnoise = tio.RandomNoise(std = (suj_noise,suj_noise))
        ssynth = tnoise(ssynth)

    return ssynth, tmot, mr

def perform_motion_step_loop(json_file, params, out_path=None, out_name=None, resolution=512):

    mvt_axe_str_list = ['transX', 'transY', 'transZ', 'rotX', 'rotY', 'rotZ']

    nb_x0s = params[0]['nb_x0s']
    nb_sim = len(params) * nb_x0s
    print(f'performing loop of {nb_sim} iter 10s per iter is {nb_sim * 10 / 60 / 60} Hours {nb_sim * 10 / 60} mn ')
    if out_name is not None:
        print(f'save will be made in {out_path} with name {out_name}')

    df, res, res_fitpar, extra_info, i = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), dict(), 0
    for param in params:
        pp = SimpleNamespace(**param)
        #[amplitude, sigma, sym, mvt_type, mvt_axe, cor_disp, disp_str, nb_x0s, x0_min] = param
        amplitude, sigma, sym, mvt_type, mvt_axe, cor_disp, disp_str, nb_x0s, x0_min =  pp.amplitude, pp.sigma, pp.sym, pp.mvt_type, pp.mvt_axe, pp.cor_disp, pp.disp_str, pp.nb_x0s, pp.x0_min

        ssynth, tmot, mr = select_data(json_file, param)

        mvt_axe_str = ''
        for ii in mvt_axe: mvt_axe_str +=  mvt_axe_str_list[ii]

        if sym:
            xcenter = resolution // 2 - sigma // 2;
            x0s = np.floor(np.linspace(xcenter - x0_min, xcenter, nb_x0s))
        else:
            xcenter = resolution // 2;
            x0s = np.floor(np.linspace(x0_min, xcenter, nb_x0s))
            x0s = x0s[x0s>=sigma//2] #remove first point to not reduce sigma
        #x0s=[242]
        for x0 in x0s:
            start = time.time()
            sigma, x0, xend = int(sigma), int(x0), x0 + sigma // 2
            # print(f'{amplitude} {sigma} {type(sigma)}')

            fp = corrupt_data(x0, sigma=sigma, method=mvt_type, amplitude=amplitude, mvt_axes=mvt_axe,
                              center='none', return_all6=True, sym=sym, resolution=resolution)
            # plt.figure();plt.plot(fp.T)
            extra_info = param
            extra_info['xend'], extra_info['x0'] =  xend, x0
            extra_info['mvt_type'] = mvt_type + '_sym' if sym else mvt_type
            extra_info['mvt_axe'] = mvt_axe_str  #change list of int, to str, easier for csv ...
            extra_info['sujname'] = ssynth.name

            smot, df, res_fitpar, res = apply_motion(ssynth, tmot, fp, df, res_fitpar, res, extra_info, config_runner=mr,
                                                     correct_disp=cor_disp)
            # ov(smot.t1.data[0])
            # ssynth.t1.save(out_path+'/vol_orig.nii')
            # smot.t1.save(out_path+'/vol_mot_x0_256_sigma128.nii')
            i += 1
            total_time = time.time() - start
            print(f'{i} / {nb_sim} in {total_time} ')

    mres = ModelCSVResults(df_data=df, out_tmp="/tmp/rrr")
    keys_unpack = ['transforms_metrics', 'm_t1'];
    suffix = ['m', 'm_t1']
    df1 = mres.normalize_dict_to_df(keys_unpack, suffix=suffix);
    # if 'shift' in df1: #not sure good idea to correct here ... (other mean Disp should also be shifted ...
    #    df1['m_wTF_Disp_1'] = df1['m_wTF_Disp_1'] + df1['shift']  # if compute after shifting the fp

    if out_name is not None:
        if not os.path.isdir(out_path): os.mkdir(out_path)
        if res.shape[0] > 0:  # only if correct_disp is True
            res.to_csv(out_path + f'/res_shift{out_name}.csv')
        res_fitpar.to_csv(out_path + f'/res_fitpars{out_name}.csv')
        df1.to_csv(out_path + f'/res_metrics{out_name}.csv')

    return df1, res, res_fitpar

def create_motion_job(params, split_length, file, out_path, res_name):

    nb_param = len(params)
    nb_job = int(np.ceil(nb_param / split_length))

    from script.create_jobs import create_jobs

    cmd_init = '\n'.join(['python -c "', "from util_affine import perform_motion_step_loop "])
    jobs = []
    for nj in range(nb_job):
        ind_start = nj * split_length;
        ind_end = np.min([(nj + 1) * split_length, nb_param])
        print(f'{nj} {ind_start} {ind_end} ')
        print(f'param = {params[ind_start:ind_end]}')

        cmd = '\n'.join([cmd_init, f'params = {params[ind_start:ind_end]}',
                         f'out_path = \'{out_path}\'',
                         f'out_name = \'{res_name}_split{nj}\'',
                         f'json_file = \'{file}\'',
                         '_ = perform_motion_step_loop(json_file,params, out_path=out_path, out_name=out_name) "'])
        jobs.append(cmd)

    job_params = dict()
    job_params[
        'output_directory'] = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion/' + f'{res_name}_job'
    job_params['jobs'] = jobs
    job_params['job_name'] = 'motion'
    job_params['cluster_queue'] = 'bigmem,normal'
    job_params['cpus_per_task'] = 4
    job_params['mem'] = 8000
    job_params['walltime'] = '12:00:00'
    job_params['job_pack'] = 1

    create_jobs(job_params)


def corrupt_data( x0, sigma= 5, amplitude=20, method='gauss', mvt_axes=[1], center='zero', resolution=200, sym=False,
                  return_all6=False):
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
        y = np.hstack((np.zeros((1, (x0 - sigma))),
                       np.linspace(0, amplitude, 2 * sigma + 1).reshape(1, -1),
                       np.ones((1, ((resolution - x0) - sigma - 1))) * amplitude))
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
        y = y -y[resolution//2]
    if sym:
        center = y.shape[0]//2
        y = np.hstack([y[0:center], np.flip(y[0:center])])

    if return_all6:
        for xx in mvt_axes:
            fp[xx, :] = y
        y=fp

    return y

def torch_deg2rad(tensor: torch.Tensor) -> torch.Tensor:
    r"""Function that converts angles from degrees to radians.
    Args:
        tensor (torch.Tensor): Tensor of arbitrary shape.
    Returns:
        torch.Tensor: tensor with same shape as input.
    Examples::
        >>> input = 360. * torch.rand(1, 3, 3)
        >>> output = kornia.deg2rad(input)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(tensor)))

    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.

def spm_imatrix(M):
    import nibabel as ni
    import numpy.linalg as npl

    #translation euler angle  scaling shears % copy from from spm_imatrix
    [rot,[tx,ty,tz]] = ni.affines.to_matvec(M)

    # rrr    C = np.transpose(npl.cholesky(np.transpose(rot).dot(rot)))
    C = np.transpose(npl.cholesky(rot.dot(np.transpose(rot))))
    [scalx, scaly, scalz] = np.diag(C)
    if npl.det(rot)<0:  # Fix for -ve determinants
        scalx=-scalx

    C = np.dot(npl.inv(np.diag(np.diag(C))),C)
    [shearx, sheary, shearz] = [C[0,1], C[0,2], C[1,2]]

    P=np.array([0,0,0,0,0,0,scalx, scaly, scalz,shearx, sheary, shearz])
    R0=spm_matrix(P)
    R0=R0[:3,:3]
    #rrr    R1 = rot.dot(npl.inv(R0))
    R1 = (npl.inv(R0)).dot(rot)

    [rz, ry, rx]  = ni.eulerangles.mat2euler(R1)
    [rz, ry, rx] = [rz*180/np.pi, ry*180/np.pi, rx*180/np.pi] #rad to degree
    #rrr I do not know why I have to add - to make it coherent
    P = np.array([tx,ty,tz,-rx,-ry,-rz,scalx, scaly, scalz,shearx, sheary, shearz])
    return P

def spm_matrix(P,order=0):
    """
    FORMAT [A] = spm_matrix(P )
    P(0)  - x translation
    P(1)  - y translation
    P(2)  - z translation
    P(3)  - x rotation around x in degree
    P(4)  - y rotation around y in degree
    P(5)  - z rotation around z in degree
    P(6)  - x scaling
    P(7)  - y scaling
    P(8)  - z scaling
    P(9) - x affine
    P(10) - y affine
    P(11) - z affine

    order - application order of transformations. if order (the Default): T*R*Z*S if order==0 S*Z*R*T
    """
    convert_to_torch=False
    if torch.is_tensor(P):
        P = P.numpy()
        convert_to_torch=True

    [P[3], P[4], P[5]] = [P[3]*180/np.pi, P[4]*180/np.pi, P[5]*180/np.pi] #degre to radian

    T = np.array([[1,0,0,P[0]],[0,1,0,P[1]],[0,0,1,P[2]],[0,0,0,1]])
    R1 =  np.array([[1,0,0,0],
                    [0,np.cos(P[3]),np.sin(P[3]),0],#sing change compare to spm because neuro versus radio ?
                    [0,-np.sin(P[3]),np.cos(P[3]),0],
                    [0,0,0,1]])
    R2 =  np.array([[np.cos(P[4]),0,-np.sin(P[4]),0],
                    [0,1,0,0],
                    [np.sin(P[4]),0,np.cos(P[4]),0],
                    [0,0,0,1]])
    R3 =  np.array([[np.cos(P[5]),np.sin(P[5]),0,0],  #sing change compare to spm because neuro versus radio ?
                    [-np.sin(P[5]),np.cos(P[5]),0,0],
                    [0,0,1,0],
                    [0,0,0,1]])

    #R = R1.dot(R2.dot(R3))
    R = (R1.dot(R2)).dot(R3)

    Z = np.array([[P[6],0,0,0],[0,P[7],0,0],[0,0,P[8],0],[0,0,0,1]])
    S = np.array([[1,P[9],P[10],0],[0,1,P[11],0],[0,0,1,0],[0,0,0,1]])
    if order==0:
        A = S.dot(Z.dot(R.dot(T)))
    else:
        A = T.dot(R.dot(Z.dot(S)))

    if convert_to_torch:
        A = torch.from_numpy(A).float()

    return A


def get_affine_from_rot_scale_trans(rot, scale, trans, inverse_affine=False):

    if torch.is_tensor(rot):
        mr = create_rotation_matrix_3d(torch_deg2rad(rot))
        if rot.is_cuda: mr = mr.cuda()
        ms = torch.diag(scale)
        mmm = torch.mm(ms, mr)
        affine_torch = torch.zeros([3, 4])
        affine_torch[0:3, 0:3] = mmm
        affine_torch[0:3, 3] = trans
        affine_torch = affine_torch.unsqueeze(0)

    else:
        mr = create_rotation_matrix_3d(np.deg2rad(rot))
        ms = np.diag(scale)
        # image_size = np.array([ii[0].size()])
        # trans_torch = np.array(trans) / (image_size / 2)
        # affine_torch = np.hstack((ms @ mr, trans.T))
        affine_torch = np.hstack((ms @ mr, np.expand_dims(trans,0).T))

        affine_torch = affine_torch[np.newaxis,:] #one color chanel todo : duplicate if more !
        affine_torch = torch.from_numpy(affine_torch)

    if inverse_affine:
        # transform affine from a 3*4 to a 4*4 matrix
        xx = torch.zeros(4, 4);
        xx[3, 3] = 1
        xx[0:3, 0:4] = affine_torch[0]
        affine_torch_inv = xx.inverse()
        affine_torch_inv = affine_torch_inv[0:3, 0:4].unsqueeze(0)
        return affine_torch_inv, affine_torch

    return affine_torch


def target_normalize(target, in_range=None, method=1):
    if in_range is not None:
        norm_min = torch.ones(target.shape[0]) * in_range[0]
        norm_max = torch.ones(target.shape[0]) * in_range[1]

    if method == 1:  # use min max define in range to put data between 0 1
        vout = (target - norm_min) / (norm_max - norm_min)

    if method == 2:  # use min max define in range to put data between -1 1
        vout = (target - (norm_max + norm_min) / 2) * 2 / (norm_max - norm_min)

    if method == 3: # for scaling value |0 1] is map to [-inf 0] and [1 inf] to [0 inf]
        vout = torch.zeros(target.shape[0])
        for index in range(target.shape[0]):
            vout[index] = 1-1/target[index] if target[index]<1 else target[index] -1

    return vout


def target_normalize_back(vout, in_range=None, method=1):
    if in_range is not None:
        norm_min = torch.ones(vout.shape[0]) * in_range[0]
        norm_max = torch.ones(vout.shape[0]) * in_range[1]
        if vout.is_cuda:
            norm_min, norm_max = norm_min.cuda(), norm_max.cuda()

    if method == 1:  # use min max define in range to put data between 0 1
        target = vout *  (norm_max - norm_min) + norm_min

    if method == 2:  # use min max define in range to put data between -1 1
        target = vout * (norm_max - norm_min)/2 + (norm_max + norm_min)/2

    if method == 3: # for scaling value |0 1] is map to [-inf 0] and [1 inf] to [0 inf]
        target = torch.zeros(vout.shape[0])
        if vout.is_cuda: target = target.cuda()

        for index in range(vout.shape[0]):
            target[index] = -1/(vout[index]-1) if vout[index]<0 else vout[index] + 1

    return target


def get_random_affine(in_range=[-45,45,0.6,1.5],
                      range_norm=[-90, 90, 0.1, 1.9],
                      norm_method=1):

    # rot = np.random.uniform(in_range[0],in_range[1],size=3)
    # scale = np.random.uniform(in_range[2],in_range[3],size=3)
    # trans =  np.random.uniform(0,0,size=3)
    rot = torch.FloatTensor(3).uniform_(in_range[0],in_range[1])
    scale = torch.FloatTensor(3).uniform_(in_range[2],in_range[3])
    trans = torch.FloatTensor(3).uniform_(0, 0)

    if norm_method==1:  #normalize all value to 0 1 (fixed by in_range)
        rot_norm = target_normalize(rot, in_range[0:2], method=1)
        scale_norm = target_normalize(scale, in_range[2:4], method=1)

    elif norm_method==2:
        rot_norm = target_normalize(rot, range_norm[0:2], method=2)
        scale_norm = target_normalize(scale, None, method=3)

    rots = torch.cat((rot, scale))
    vout = torch.cat((rot_norm, scale_norm))

    affine_torch = get_affine_from_rot_scale_trans(rot, scale, trans)

    return affine_torch, vout, rots

def get_affine_from_predict(y_predicts, in_range, range_norm=[-90, 90, 0.1, 1.9], norm_method=1):

    if len(y_predicts.shape) > 1:
        nb_batch = y_predicts.shape[0]
    else:
        nb_batch = 1

    if len(y_predicts.shape) == 1:
        y_predict = y_predicts.unsqueeze(0) #add an extra dim to make the for loop working for single case

    batch_affine_inv, batch_affine, batch_rot = [], [], []
    for i in range(nb_batch):
        y_predict = y_predicts[i]
        rot_pred, scale_pred, trans_pred = y_predict[0:3], y_predict[3:6], torch.zeros((3))

        if norm_method == 1:  # normalize all value to 0 1 (fixed by in_range)
            rot = target_normalize_back(rot_pred, in_range[0:2], method=1)
            scale = target_normalize_back(scale_pred, in_range[2:4], method=1)
            trans = trans_pred

        elif norm_method == 2:
            rot = target_normalize_back(rot_pred, range_norm[0:2], method=2)
            scale = target_normalize_back(scale_pred, None, method=3)
            trans = trans_pred

        if rot.is_cuda: trans = trans.cuda();

        #y_values = torch.cat((rot, scale, trans)) not yet
        y_values = torch.cat((rot, scale))

        affine_torch_inv, affine_torch = get_affine_from_rot_scale_trans(rot, scale, trans, inverse_affine=True)
        # since the network learn to predict the rotation made, we then need the inverse (to correcte the detected affine)
        batch_affine_inv.append(affine_torch_inv)
        batch_affine.append(affine_torch)
        batch_rot.append(y_values.unsqueeze(0))

    return torch.cat(batch_affine_inv), torch.cat(batch_affine), torch.cat(batch_rot)

def apply_affine_to_data(data_in, mat_affine, align_corners=False):

    #transpose to have similar motion as in nibabel resample (also sitk, to check)
    x = data_in.permute(0, 1, 4, 3, 2) # if no batch it should be .unsqueeze(0)     #x = data_in.permute(0,3,2,1).unsqueeze(0)

    grid = F.affine_grid(mat_affine, x.shape, align_corners=align_corners).float()
    if x.is_cuda:  grid = grid.cuda()
    x = F.grid_sample(x, grid, align_corners=align_corners)

    x = x.permute(0, 1, 4, 3, 2) #come back in original indice order
    #xx = x[0,0].numpy().transpose(2,1,0)    #ov(xx) for viewing
    return x

def torch_transform_random_affine(data_in, in_range=[-45,45,0.6,1.5], range_norm=[-90, 90, 0.1, 1.9], align_corners=False, norm_method=1):

    mat_affine, target, rots = [], [], []
    for nb in range(data_in.shape[0]):  # nb batch
        aff_, aff_n, rot_ = get_random_affine(in_range=in_range, norm_method=norm_method, range_norm=range_norm)
        mat_affine.append(aff_)
        target.append(aff_n.unsqueeze(0))
        rots.append(rot_.unsqueeze(0))
    mat_affine = torch.cat(mat_affine)
    target = torch.cat(target).float()
    rots = torch.cat(rots).float()

    if data_in.is_cuda:
        mat_affine, target, rots = mat_affine.cuda(), target.cuda(), rots.cuda()

    x = apply_affine_to_data(data_in, mat_affine, align_corners=align_corners)

    return x, mat_affine, target, rots


def create_rotation_matrix_3d(angles):
    """
    given a list of 3 angles, create a 3x3 rotation matrix that describes rotation about the origin
    :param angles (list or numpy array) : rotation angles in 3 dimensions
    :return (numpy array) : rotation matrix 3x3
    """
    if torch.is_tensor(angles):

        mat1 = torch.tensor([[1., 0., 0.],
                             [0., torch.cos(angles[0]), torch.sin(angles[0])],
                             [0., -torch.sin(angles[0]), torch.cos(angles[0])]] )

        mat2 = torch.tensor([[torch.cos(angles[1]), 0., -torch.sin(angles[1])],
                            [0., 1., 0.],
                            [torch.sin(angles[1]), 0., torch.cos(angles[1])]] )

        mat3 = torch.tensor([[torch.cos(angles[2]), torch.sin(angles[2]), 0.],
                            [-torch.sin(angles[2]), torch.cos(angles[2]), 0.],
                            [0., 0., 1.]] )

        mat = torch.mm(torch.mm(mat1, mat2), mat3)
    else:
        mat1 = np.array([[1., 0., 0.],
                         [0., math.cos(angles[0]), math.sin(angles[0])],
                         [0., -math.sin(angles[0]), math.cos(angles[0])]],
                        dtype='float')

        mat2 = np.array([[math.cos(angles[1]), 0., -math.sin(angles[1])],
                         [0., 1., 0.],
                         [math.sin(angles[1]), 0., math.cos(angles[1])]],
                        dtype='float')

        mat3 = np.array([[math.cos(angles[2]), math.sin(angles[2]), 0.],
                         [-math.sin(angles[2]), math.cos(angles[2]), 0.],
                         [0., 0., 1.]],
                        dtype='float')

        mat = (mat1 @ mat2) @ mat3
    return mat

def apply_model_affine(trans_in, model, in_range, nb_pass=3):
    """
    direct affine prediciton with multiple pass
    """
    cum_affine_correction = torch.cat([torch.diag(torch.ones(4)).unsqueeze(0) for i in range(trans_in.shape[0])])
    trans_in_corected = trans_in
    one_line = torch.zeros([1, 4]);
    one_line[0, 3] = 1
    for nb_iter in range(nb_pass):
        #print(nb_iter)
        if nb_iter==0:
            outputs1 = model(trans_in_corected)
            outputs = outputs1
        else:
            outputs = model(trans_in_corected)
        mpred_inv, mpred, rots_pred = get_affine_from_predict(outputs, in_range=in_range)
        trans_in_corected = apply_affine_to_data(trans_in_corected, mpred_inv)

        if trans_in_corected.is_cuda is False: trans_in_corected = trans_in_corected.cuda()

        # trans_in = trans_in.cuda()
        for nb_batch in range(mpred_inv.shape[0]):
            mat1 = cum_affine_correction[nb_batch]
            mat2 = torch.cat([mpred_inv[nb_batch], one_line])
            cum_affine_correction[nb_batch] = torch.mm(mat1, mat2)

    #del(trans_in_corected)

    # compute what would have been the prediciton if one pass
    # mpred_inv = cum_affine_correction[:,0:3,:]
    # bb = apply_affine_to_data(trans_in,mpred_inv)

    if trans_in.is_cuda:
        rots_pred = rots_pred.cuda()

    for nb_batch in range(mpred_inv.shape[0]):
        mm = cum_affine_correction[nb_batch]
        P = spm_imatrix(mm.inverse().detach().cpu())
        rots_pred[nb_batch] = torch.from_numpy(P[3:9])
        #outputs1.data[nb_batch] = (rots_pred.data[nb_batch] - norm_min) / (norm_max - norm_min)

    return outputs, rots_pred, cum_affine_correction
