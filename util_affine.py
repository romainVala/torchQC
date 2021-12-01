import torch.nn.functional as F
import torch, math, time, os
import numpy as np
import pandas as pd
from segmentation.config import Config
from read_csv_results import ModelCSVResults
from itertools import product
from types import SimpleNamespace
import torchio as tio
import SimpleITK as sitk

pi = torch.tensor(3.14159265358979323846)
import numpy.linalg as npl
try:
    import quaternion as nq
except :
    print('can not import quaternion package, ... not a big deal')

from dual_quaternions import DualQuaternion
import scipy.linalg as scl

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    #for instance in itertools.product(*vals):
    #    yield dict(zip(keys, instance))
    #return (dict(zip(keys, values)) for values in product(*vals))  #this yield a generator, but easier with a list -> known size
    return list(dict(zip(keys, values)) for values in product(*vals))

def apply_motion(sdata, tmot, fp, config_runner=None, df=pd.DataFrame(), extra_info=dict(), param=dict(),
                 root_out_dir=None, suj_name='NotSet', suj_orig_name=None,
                 do_coreg='Elastix', return_motion=False, save_fitpars=False,
                 compute_displacement=False):

    if 'displacement_shift_strategy' not in param: param['displacement_shift_strategy']=None
    if 'freq_encoding_dim' not in param: param['freq_encoding_dim']=0
    ''
    # apply motion transform
    start = time.time()
    #tmot.metrics = None
    if isinstance(tmot, tio.transforms.augmentation.intensity.RandomMotionFromTimeCourse):
        tmot.nT = fp.shape[1]
        tmot.simulate_displacement = False; #tmot.oversampling_pct = 1
        tmot.displacement_shift_strategy = param['displacement_shift_strategy']
        tmot.freq_encoding_dim = param['freq_encoding_dim'] #only use in the constructor to set
        tmot.frequency_encoding_dim = param['freq_encoding_dim']
        # if tmot.displacement_shift_strategy is None: #let's force demean for the first try to be not too far
        #     fp = fp - np.mean(fp, axis=1, keepdims=True)
        #     print('first motion is done with demean')
        #     #arg, may be not a good idea, for sigma motion

        tmot.fitpars = fp

    smot = tmot(sdata)
    #update fp, since it may have change is displacement_shift_strategy is used
    for hh in smot.history:
        if isinstance(hh, tio.transforms.augmentation.intensity.random_motion_from_time_course.MotionFromTimeCourse):
            fp = hh.fitpars['t1']  #fitpar is modifie in Motion transform not in tmot
            print(f'max fp is {fp.max(axis=1)}')

    batch_time = time.time() - start;     start = time.time()
    print(f'First motion in  {batch_time} ')


    df_before_coreg = pd.DataFrame()
    df_before_coreg, report_time = config_runner.record_regression_batch(df_before_coreg, smot, torch.zeros(1).unsqueeze(0),
                                torch.zeros(1).unsqueeze(0), batch_time, save=False, extra_info=extra_info)

    clean_output = param["clean_output"] if "clean_output" in param else 0
    if do_coreg is not None:

        if root_out_dir is None: raise ('error outut path (root_out_dir) must be se ')
        if not os.path.isdir(root_out_dir): os.mkdir(root_out_dir)
        if suj_orig_name is None:
            orig_vol_name = f'vol_orig_I{param["suj_index"]}_C{param["suj_contrast"]}_N_{int(param["suj_noise"] * 100)}_D{param["suj_deform"]:d}.nii'
        else:
            orig_vol_name = f'{suj_orig_name}.nii'
        # save orig data in the upper dir
        # if not os.path.isfile(orig_vol):
        #    sdata.t1.save(orig_vol)
        # bad idea with random contrast, since it can change with seeding ... and then wrong flirt

        out_dir = root_out_dir + '/' + suj_name
        if not os.path.isdir(out_dir): os.mkdir(out_dir)
        orig_vol = f'{out_dir}/{orig_vol_name}'
        sdata.t1.save(orig_vol)

        out_vol = out_dir + '/vol_motion.nii'
        smot.t1.save(out_vol)
        out_affine = out_dir + '/coreg_affine.txt'

        if do_coreg == 'Elastix':
            out_aff, elastix_trf = ElastixRegister(sdata.t1, smot.t1)
            trans_rot = get_euler_and_trans_from_matrix(out_aff)

        else:
            import subprocess
            import numpy.linalg as npl
    
            # since (I guess motion do rotation relatif to volume center, shift the affine to have 0,0,0 at volume center)
            notuse = False
            if notuse:  # not necessary for fsl affine, (since independant of nifti header)
                aff = sdata.t1.affine
                dim = sdata.t1.data[0].shape
                aff[:, 3] = [dim[0] // 2, dim[1] // 2, dim[2] // 2, 1]  # todo !! works becaus not rotation in nifti affinne
                new_affine = npl.inv(aff)
                sdata.t1.affine = new_affine
                smot.t1.affine = new_affine
                # arg when pure rotation I get the same fsl matrix even if I do not change the header origin
    
            if np.allclose(sdata.t1.affine, smot.t1.affine) is False:
                print(f'WWWWWWWWWWhat the fuck Afine orig is {sdata.t1.affine} but after motion {smot.t1.affine} ')
                qsdf

            #out_vol_nifti_reg = out_dir + '/r_vol_motion.nii'
            # cmd = f'reg_aladin -rigOnly -ref {orig_vol} -flo {out_vol} -res {out_vol_nifti_reg} -aff {out_affine}'
            # cmd = f'flirt -dof 6 -ref {out_vol} -in {orig_vol}  -out {out_vol_nifti_reg} -omat {out_affine}'
            cmd = f'flirt -dof 6 -ref {out_vol} -in {orig_vol}  -omat {out_affine}'
            outvalue = subprocess.run(cmd.split(' '))
            if not outvalue == 0:
                print(" ** Error  in " + cmd + " satus is  " + str(outvalue))
    
            if clean_output>0:
                os.remove(out_vol)
            #### reading and transforming FSL flirt Affine to retriev trans and rot set with spm_matrix(order=0) (as used torchio motion)
            out_aff = np.loadtxt(out_affine)
            affine = smot.t1.affine
    
            # fsl flirt rotation center is first voxel (0,0,0) image corner
            shape = sdata.t1.data[0].shape;
            pixdim = [1]  #
            center = np.array(shape) // 2
            new_rotation_center = -center  # I do not fully understand, it should be +center, but well ...
            T = spm_matrix([new_rotation_center[0], new_rotation_center[1], new_rotation_center[2], 0, 0, 0, 1, 1, 1, 0, 0, 0],
                           order=4);
            Ti = npl.inv(T)
            out_aff = np.matmul(T, np.matmul(out_aff, Ti))
    
            # convert to trans so that we get the same affine but from convention R*T
            rot = out_aff.copy();
            rot[:, 3] = [0, 0, 0, 1]
            trans2 = out_aff.copy();
            trans2[:3, :3] = np.eye(3)
            trans1 = np.matmul(npl.inv(rot),
                               np.matmul(trans2, rot))  # trans1 = np.matmul(rot, np.matmul(trans2,  npl.inv(rot)))
            out_aff[:, 3] = trans1[:, 3]
    
            if npl.det(affine) > 0:  # ie canonical, then flip, (because flirt affine convention is with -x
                shape = sdata.t1.data[0].shape;
                pixdim = [1]  #
                x = (shape[0] - 1) * pixdim[0]
                flip = np.eye(4);
                flip[0, 0] = -1;
                flip[0, 3] = 0  # x  not sure why not translation shift ... ?
                out_aff = np.matmul(flip, out_aff)
                # out_aff = np.matmul(out_aff,flip)

            trans_rot = spm_imatrix(out_aff)[:6]

        np.savetxt(out_dir + '/coreg_affine_MotConv.txt', out_aff)
        for i in range(6):
            fp[i, :] = fp[i, :] - trans_rot[i]

        if isinstance(tmot, tio.transforms.augmentation.intensity.RandomMotionFromTimeCourse):
            tmot.fitpars = fp
            tmot.displacement_shift_strategy = None
        else: #MotionFromTimeCourse
            #argument are then in dict ... arg ...
            for kkey in tmot.fitpars.keys():
                tmot.fitpars[kkey] = fp
                if not isinstance(tmot.frequency_encoding_dim ,dict):
                    print('arggg')
                    tmot.frequency_encoding_dim = dict(t1=tmot.frequency_encoding_dim)

        smot_shift = tmot(sdata)
        out_vol = out_dir + '/vol_motion_no_shift.nii'
        if clean_output <2:
            smot_shift.t1.save(out_vol)

        batch_time = time.time() - start
        print(f'coreg and second motion in  {batch_time} ')

        extra_info['flirt_coreg'] = 1
        for i in range(6):
            extra_info[f'shift_{i}'] = trans_rot[i]
        df_after_coreg = pd.DataFrame()

        # extra_info["shift_T1"] = trans_rot[0]; extra_info["shift_T2"] = trans_rot[1]; extra_info["shift_T3"] = trans_rot[2]
        # extra_info["shift_R1"] = trans_rot[3]; extra_info["shift_R2"] = trans_rot[4]; extra_info["shift_R3"] = trans_rot[5]
        df_after_coreg, report_time = config_runner.record_regression_batch(df_after_coreg, smot_shift, torch.zeros(1).unsqueeze(0),
                                                                torch.zeros(1).unsqueeze(0),
                                                                batch_time, save=False, extra_info=extra_info)


        #concatenate metrics dataframe before and after coreg
        df_keep1 = df_before_coreg[['transforms_metrics']]
        mres = ModelCSVResults(df_data=df_keep1, out_tmp="/tmp/rrr")
        keys_unpack = ['transforms_metrics', 'no_shift_t1'];
        suffix = ['no_shift', 'no_shift']
        df1 = mres.normalize_dict_to_df(keys_unpack, suffix=suffix);
        df1.pop('transforms_metrics')

        mres = ModelCSVResults(df_data=df_after_coreg, out_tmp="/tmp/rrr")
        keys_unpack = ['transforms_metrics', 't1'];
        suffix = ['', 'm']
        df2 = mres.normalize_dict_to_df(keys_unpack, suffix=suffix);
        df2.pop('transforms_metrics')

        if compute_displacement:
            start = time.time()
            image, brain_mask = sdata.t1.data[0].numpy(), sdata.brain.data[0].numpy()
            df_disp = get_displacement_field_metric(image, brain_mask, fp)
            batch_time = time.time() - start
            print(f'Displacement field metrics in   {batch_time} ')

            dfall = pd.concat([df_disp, df2, df1], sort=True, axis=1);
        else:
            print('Skiping displacement field computation')
            dfall = pd.concat([df2, df1], sort=True, axis=1);

        if out_dir is not None:
            if not os.path.isdir(out_dir): os.mkdir(out_dir)
            saved_filename =  f'{out_dir}/metrics_fp_{suj_name}'
            dfall.to_pickle(saved_filename + '.gz', protocol=3)
            dfall["history"] = saved_filename + '.gz'
            dfall.to_csv(saved_filename + ".csv")

            if save_fitpars:
                np.savetxt(out_dir + '/fitpars_shift.txt',fp)
                fp_orig = fp.copy()
                for i in range(6):
                    fp_orig[i, :] = fp_orig[i, :] + trans_rot[i]
                np.savetxt(out_dir + '/fitpars_orig.txt',fp_orig)

        if return_motion:
            return df1, smot_shift
        else:
            return df1

    if return_motion:
        return df_before_coreg, smot
    else:
        return df_before_coreg

    test_direct_problem = False
    if test_direct_problem:
        fp = np.ones_like(fp) * 6;
        for iii, r in enumerate(fp):
            fp[iii, :] = (6 - iii) * 3
        # fp[:3,:]=0
        out_dir = '/home/romain.valabregue/tmp/a'

        # if nocanonical not necessary if flip matrix
        # aff_direct = spm_matrix([fp[0, 0], fp[1, 0], fp[2, 0],fp[3, 0], -fp[4, 0], -fp[5, 0], 1, 1, 1, 0, 0, 0, ], order=0)
        # if canonical
        aff_direct = spm_matrix([fp[0, 0], fp[1, 0], fp[2, 0], fp[3, 0], fp[4, 0], fp[5, 0], 1, 1, 1, 0, 0, 0, ],
                                order=0)
        nii_img0 = nb.load('/home/romain.valabregue/tmp/a/vol_orig_I0_C1_N_1_D0.nii')
        nii_img = nb.load('/home/romain.valabregue/tmp/a/vol_orig_I0_C1_N_1_D0.nii')
        out_p = '/home/romain.valabregue/tmp/a/lille_005011AAA_BL0/nii_dir.nii'
        nii_aff = nii_img.get_affine()

        # change aff_direct to have rotation expres at nifi origin
        shape = sdata.t1.data[0].shape;
        pixdim = [1];
        center = np.array(shape) // 2
        origin_vox = np.matmul(npl.inv(nii_aff), np.array([0, 0, 0, 1]))[:3]
        voxel_shift = -origin_vox + center
        T = spm_matrix([voxel_shift[0], voxel_shift[1], voxel_shift[2], 0, 0, 0, 1, 1, 1, 0, 0, 0], order=4);
        Ti = npl.inv(T)
        aff_direct = np.matmul(T, np.matmul(aff_direct, Ti))

        if npl.det(nii_aff) < 0:  # ie nocanonical, then flip, (inverse of fsl)
            flip = np.eye(4);
            flip[
                0, 0] = -1;  # #pixdim = [1]  # x = (shape[0] - 1) * pixdim[0] flip[0, 3] = 0  # x  not sure why not translation shift ... ?
            aff_direct = np.matmul(flip, aff_direct)
            nii_aff = np.matmul(flip, nii_aff)
            # nii_img0.affine[:] =  np.matmul(flip, nii_img0.affine)[:]

        # # convert to trans so that we get the same affine but from convention R*T   needed only if aff_direct is done with order>0
        # rot = aff_direct.copy();            rot[:, 3] = [0, 0, 0, 1]
        # trans2 = aff_direct.copy();         trans2[:3, :3] = np.eye(3)
        # #trans1 = np.matmul(npl.inv(rot),np.matmul(trans2, rot))  # arg here it is the inverse compare to transformaing flirt matrix
        # trans1 = np.matmul(rot, np.matmul(trans2,  npl.inv(rot)))
        # aff_direct[:, 3] = trans1[:, 3]

        nii_img.affine[:] = np.matmul(aff_direct, nii_aff)[:]
        # nii_img.affine[:] = np.matmul(npl.inv(aff_direct), nii_aff)[:]     #nii_img.affine[:] = np.matmul( nii_aff, aff_direct)[:]
        out_img = nbp.resample_from_to(nii_img, nii_img0, cval=0)
        nb.save(out_img, out_p)

        # fslpy
        from fsl.transform.affine import decompose, compose

        angle = np.deg2rad([fp[3, 0], fp[4, 0], fp[5, 0]])
        aff_direct_fsl = compose((1, 1, 1), (fp[0, 0], fp[1, 0], fp[2, 0]), angle)
        scale, trans, angles = decompose(aff_direct);
        angles = np.rad2deg(angles)
        spm_imatrix(aff_direct)[:6]
        # argg other convention for angle ... I stop here

        # image space motion
        ta = tio.Compose([tio.ToCanonical(), tio.Affine(scales=1, degrees=[3, -6, -9], translation=0)])
        ta = tio.Affine(scales=1, degrees=[-fp[3, 0], -fp[4, 0], -fp[5, 0]], translation=0)
        ta = tio.Affine(scales=1, degrees=[-fp[3, 0], fp[4, 0], fp[5, 0]], translation=0)
        smoti = ta(sdata)
        smoti.t1.save(out_dir + '/tio_affma3.nii')

        # rotation around point 50, 0, 0
        rot = spm_matrix([0, 0, 0, 10, 11, 12, 1, 1, 1, 0, 0, 0], order=4)
        T = spm_matrix([10, -18, -76, 0, 0, 0, 1, 1, 1, 0, 0, 0], order=4);
        Ti = npl.inv(T)
        new_affine = np.matmul(T, np.matmul(rot, Ti))
        # new_affine.dot([0, 80,0, 1]) is the same point, it is the rotation center
        # taking into account that motion is doing R*T (and not T*R), to get the translation :
        trans2 = new_affine.copy();
        trans2[:3, :3] = np.eye(3)
        trans1 = np.matmul(npl.inv(rot), np.matmul(trans2, rot))
        new_affine[:, 3] = trans1[:, 3]
        fp = np.zeros_like(fp);
        fp[0, :] = trans1[0, 3];
        fp[1, :] = trans1[1, 3];
        fp[2, :] = trans1[2, 3];
        fp[3, :] = 10

        # find fsl rotation center http://www.euclideanspace.com/maths/geometry/affine/aroundPoint/index.htm
        from sympy import symbols, Eq, solve

        rot = out_aff
        x, y, z = symbols('x,y,z')
        eq1 = Eq((x - rot[0, 0] * x - rot[0, 1] * y - rot[0, 2] * z), rot[0, 3])
        eq2 = Eq((y - rot[1, 0] * x - rot[1, 1] * y - rot[1, 2] * z), rot[1, 3])
        eq3 = Eq((z - rot[2, 0] * x - rot[2, 1] * y - rot[2, 2] * z), rot[2, 3])
        print(solve((eq1, eq2, eq3), (x, y, z)))
        # why is there several solutions .... (ie several fix point for the given affine)
        # rotation axis
        u = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])


def apply_motion_old_with_shift(sin, tmot, fp, df, res_fitpar, res, extra_info, config_runner,
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

def select_data(json_file, param=None, to_canonical=False):
    result_dir= os.path.dirname(json_file) +'/rrr' #/data/romain/PVsynth/motion_on_synth_data/test1/rrr/'
    config = Config(json_file, result_dir, mode='eval', save_files=False) #since cluster read (and write) the same files, one need save_file false to avoid colusion
    config.init()
    mr = config.get_runner()

    if param is not None:
        suj_ind = param['suj_index']
        if 'suj_seed' in param:
            suj_seed = param['suj_seed']
            if suj_seed is not None:
                np.random.seed(suj_seed)
                torch.manual_seed(suj_seed)
        else:
            suj_seed=-1
        contrast = param['suj_contrast']
        suj_deform = param['suj_deform']
        suj_noise = param['suj_noise']
    else:
        suj_ind, contrast, suj_deform, suj_noise = 0, 1, False, 0

    s1 = config.train_subjects[suj_ind]
    print(f'loading suj {s1.name} with contrast {contrast} deform is {suj_deform} sujseed {suj_seed} suj_noise {suj_noise}')

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
    if to_canonical:
        tcano = tio.ToCanonical(p=1)
        ssynth = tcano(ssynth)
    tmot = transfo_list[1]
    if suj_deform:
        taff_ela = transfo_list[2]
        ssynth = taff_ela(ssynth)
    if suj_noise:
        tnoise = tio.RandomNoise(std = (suj_noise,suj_noise), abs_after_noise=True)
        ssynth = tnoise(ssynth)

    return ssynth, tmot, mr

def perform_motion_step_loop(json_file, params, out_path=None, out_name=None, resolution=512):

    mvt_axe_str_list = ['transX', 'transY', 'transZ', 'rotX', 'rotY', 'rotZ','oy1','oy2']

    nb_x0s = params[0]['nb_x0s']
    nb_sim = len(params) * nb_x0s
    print(f'performing loop of {nb_sim} iter 10s per iter is {nb_sim * 10 / 60 / 60} Hours {nb_sim * 10 / 60} mn ')
    if out_name is not None:
        print(f'save will be made in {out_path} with name {out_name}')

    df, extra_info, i = pd.DataFrame(), dict(), 0
    for param in params:
        pp = SimpleNamespace(**param)
        amplitude, sigma, sym, mvt_type, mvt_axe, nb_x0s, x0_min =  pp.amplitude, pp.sigma, pp.sym, pp.mvt_type, pp.mvt_axe, pp.nb_x0s, pp.x0_min
        xend_steps = pp.xend_steps #if 'xend_steps' in pp else None

        ssynth, tmot, config_runner = select_data(json_file, param)

        mvt_axe_str = ''
        for ii in mvt_axe: mvt_axe_str +=  mvt_axe_str_list[ii]

        if sym:
            xcenter = resolution // 2 #- sigma // 2;  #to have the same
            #x0s = np.floor(np.linspace(xcenter - x0_min, xcenter, nb_x0s))
            x0s = np.floor(np.linspace(x0_min, xcenter, nb_x0s))
            x0s = x0s[x0s>=sigma//2] #remove first point to not reduce sigma
            x0s = x0s[x0s<=(xcenter-sigma//2)] #remove last points to not reduce sigma because of sym
        else:
            xcenter = resolution // 2;
            x0s = np.floor(np.linspace(x0_min, xcenter, nb_x0s))
            x0s = x0s[x0s>=sigma//2] #remove first point to not reduce sigma
        if xend_steps is not None:
            #interprete as a list of xend
            xend_steps = np.array(xend_steps)
            if sym:
                xend_steps = xend_steps[xend_steps<=xcenter]
            x0s = xend_steps - sigma//2
            x0s = x0s[x0s > sigma // 2]

        print(f'preparing {len(x0s)} with Sigma {sigma} Sym {sym} and xend : {x0s + sigma//2}')

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

            #get a unique sujname to write resultt
            suj_name = f'Suj_{ssynth.name}_I{param["suj_index"]}_C{param["suj_contrast"]}_N_{int(param["suj_noise"] * 100)}_D{param["suj_deform"]:d}_S{param["suj_seed"]}'
            if 'displacement_shift_strategy' in param:
                if param['displacement_shift_strategy'] is not None:
                    suj_name += f'Disp_{param["displacement_shift_strategy"]}'
            if 'freq_encoding_dim' in param:
                suj_name += f'_F{param["freq_encoding_dim"]}'
            fp_name  = f'fp_x{x0}_sig{sigma}_Amp{amplitude}_M{mvt_type}_A{mvt_axe_str}_sym{int(sym)}'
            if xend_steps is not None:
                fp_name = f'fp_x{xend}_sig{sigma}_Amp{amplitude}_M{mvt_type}_A{mvt_axe_str}_sym{int(sym)}'

            suj_name += fp_name
            extra_info['out_dir'] = suj_name

            _ = apply_motion(ssynth, tmot, fp, config_runner, extra_info=extra_info, param=param,
                 root_out_dir=out_path, suj_name=suj_name, do_coreg='Elastix', save_fitpars=True)

            i += 1
            total_time = time.time() - start
            print(f'{i} / {nb_sim} in {total_time} ')


    #if out_name is not None:
    #    if not os.path.isdir(out_path): os.mkdir(out_path)
    #    if res.shape[0] > 0:  # only if correct_disp is True
    #        res.to_csv(out_path + f'/res_shift{out_name}.csv')
    #    res_fitpar.to_csv(out_path + f'/res_fitpars{out_name}.csv')
    #    df1.to_csv(out_path + f'/res_metrics{out_name}.csv')

    #return df1

def create_motion_job(params, split_length, fjson, out_path, res_name, type='motion_loop',
                      mem=6000, cpus_per_task=2, walltime='12:00:00', job_pack=1,
                      jobdir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/PVsynth/job/motion_elastix/' ):

    nb_param = len(params)
    nb_job = int(np.ceil(nb_param / split_length))

    from script.create_jobs import create_jobs

    if type=='motion_loop':
        cmd_init = '\n'.join(['python -c "', "from util_affine import perform_motion_step_loop "])
        jobs = []
        for nj in range(nb_job):
            ind_start = nj * split_length;
            ind_end = np.min([(nj + 1) * split_length, nb_param])
            print(f'{nj} {ind_start} {ind_end} ')
            print(f'param = {params[ind_start:ind_end]}')

            cmd = '\n'.join([cmd_init, f'params = {params[ind_start:ind_end]}',
                             f'out_name = \'{res_name}_split{nj}\'',
                             f'json_file = \'{fjson}\'',
                             f'out_path = \'{out_path}\'',
                             '_ = perform_motion_step_loop(json_file,params, out_path=out_path, out_name=out_name) "'])
            jobs.append(cmd)
    elif type=='one_motion':
        cmd_init = '\n'.join(['python -c "', "from util_affine import perform_one_motion "])
        jobs = []
        for nj in range(nb_job):
            ind_start = nj * split_length;
            ind_end = np.min([(nj + 1) * split_length, nb_param])

            cmd = '\n'.join([cmd_init, f'fp_path = {params[ind_start:ind_end]}',
                             f'out_path = \'{out_path}\'',
                             f'json_file = \'{fjson}\'',
                             '_ = perform_one_motion(fp_path,json_file, root_out_dir=out_path) "'])
            jobs.append(cmd)
    elif type=='one_motion_simulated':
        cmd_init = '\n'.join(['python -c "', "from util_affine import perform_one_simulated_motion "])
        jobs = []
        for nj in range(nb_job):
            ind_start = nj * split_length;
            ind_end = np.min([(nj + 1) * split_length, nb_param])

            cmd = '\n'.join([cmd_init, f'params = {params[ind_start:ind_end]}',
                             f'out_path = \'{out_path}\'',
                             f'json_file = \'{fjson}\'',
                             '_ = perform_one_simulated_motion(params,json_file, root_out_dir=out_path) "'])
            jobs.append(cmd)

    job_params = dict()
    job_params[
        'output_directory'] = jobdir + f'{res_name}_job'
    job_params['jobs'] = jobs
    job_params['job_name'] = 'motion'
    job_params['cluster_queue'] = 'bigmem,normal'
    job_params['cpus_per_task'] = cpus_per_task
    job_params['mem'] = mem
    job_params['walltime'] = walltime
    job_params['job_pack'] = job_pack

    create_jobs(job_params)

def get_default_param(param=None):
    if param is None:
        param = dict()

    if not isinstance(param, list):
        params = [param]
    else:
        params = param
    for param in params:
        if 'suj_contrast' not in param: param['suj_contrast'] = 1
        if 'suj_noise' not in param: param['suj_noise'] = 0.01
        if 'suj_index' not in param: param['suj_index'] = 0
        if 'suj_deform' not in param: param['suj_deform'] = 0
        if 'displacement_shift_strategy' not in param: param['displacement_shift_strategy']=None

    return params

def perform_one_simulated_motion(params, fjson, root_out_dir=None,do_coreg='Elastix', return_motion=False):
    df = pd.DataFrame()
    params  = get_default_param(params)
    for param in params:
        # get the data
        sdata, tmot, config_runner = select_data(fjson, param, to_canonical=False)
        # load mvt fitpars

        suj_name0 = f'Suj_{sdata.name}_I{param["suj_index"]}_C{param["suj_contrast"]}_N_{int(param["suj_noise"] * 100)}_D{param["suj_deform"]:d}_S{param["suj_seed"]}'
        if 'displacement_shift_strategy' in param:
            if param['displacement_shift_strategy'] is not None:
                suj_name0 += f'_Disp_{param["displacement_shift_strategy"]}'

        for nbmot in range(param['nb_x0s']):
            if 'new_suj' in param:
                if param['new_suj']:
                    sdata, tmot, config_runner = select_data(fjson, param, to_canonical=False)

            cmd = '\n'.join(['python -c "', "from util_affine import perform_one_simulated_motion ",
                             f'params = {param}',
                             f'out_path = \'{root_out_dir}\'',
                             f'json_file = \'{fjson}\'',
                             '_ = perform_one_simulated_motion(params,json_file, root_out_dir=out_path) "'
                             ])

            tmot.preserve_center_frequency_pct = 0;
            tmot.nT = 218
            amplitude = param['amplitude']
            tmot.maxGlobalDisp, tmot.maxGlobalRot = (amplitude, amplitude), (amplitude, amplitude)
            tmot._simulate_random_trajectory()
            fp = tmot.fitpars
            
            fp_name = f'_fp_Amp{amplitude}_N{nbmot:02d}'
            suj_name = suj_name0 + fp_name

            extra_info = dict( suj_name_fp=suj_name, flirt_coreg=1)
            extra_info = dict(param, **extra_info)
            extra_info['out_dir'] = suj_name
            extra_info['job_cmd'] = cmd

            one_df = apply_motion(sdata, tmot, fp, config_runner, extra_info=extra_info, param=param,
                                  root_out_dir=root_out_dir, suj_name=suj_name,  suj_orig_name = suj_name0,
                                  do_coreg=do_coreg, save_fitpars=True)
            if len(df)==0:
                df = one_df
            else:
                df = pd.concat([df, one_df], axis=0, sort=False)

    return df

def perform_one_motion(fp_paths, fjson, param=None, root_out_dir=None,do_coreg='Elastix', return_motion=False):
    def get_sujname_from_path(ff):
        name = [];
        dn = os.path.dirname(ff)
        for k in range(3):
            name.append(os.path.basename(dn))
            dn = os.path.dirname(dn)
        return '_'.join(reversed(name))

    df = pd.DataFrame()
    param  = get_default_param(param)

    if isinstance(fp_paths, str):
        fp_paths = [fp_paths]

    for fp_path in fp_paths:
        # get the data
        sdata, tmot, config_runner = select_data(fjson, param, to_canonical=False)
        # load mvt fitpars
        fp = np.loadtxt(fp_path)

        suj_name = get_sujname_from_path(fp_path)

        extra_info = dict(fp= fp_path, suj_name_fp=suj_name, flirt_coreg=1)
        extra_info = dict(param, **extra_info)

        # apply motion transform
        if return_motion:
            one_df, smot = apply_motion(sdata, tmot, fp, config_runner, extra_info=extra_info, param=param,
                     root_out_dir=root_out_dir, suj_name=suj_name, do_coreg=do_coreg, return_motion=return_motion)
        else:
            one_df = apply_motion(sdata, tmot, fp, config_runner, extra_info=extra_info, param=param,
                                  root_out_dir=root_out_dir, suj_name=suj_name, do_coreg=do_coreg)

        if len(df)==0:
            df = one_df
        else:
            df = pd.concat([df, one_df], axis=0, sort=False)

    if return_motion:
        return df, smot
    else:
        return df


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
    elif method == 'Const':
        y = np.ones(resolution) * amplitude

    if center=='zero':
        y = y -y[resolution//2]
    if sym:
        fp_center = y.shape[0]//2
        y = np.hstack([y[0:fp_center], np.flip(y[0:fp_center])])

    if return_all6:
        if mvt_axes[0] == 6: #oy1
            orig_pos = [0, -80, 0]  # np.array([90, 28, 90]) - np.array([90,108, 90])
            l = [0, 0, 1];            m = np.cross(orig_pos, l);
            theta = np.deg2rad(amplitude);            disp = 0;
            dq = DualQuaternion.from_screw(l, m, theta, disp)
            fp = np.tile(spm_imatrix(dq.homogeneous_matrix(), order=0)[:6, np.newaxis], (1, resolution))
            fp[:,y==0] = 0
        elif mvt_axes[0] == 7: #oy2
            orig_pos = [0, 80, 0]  # np.array([90, 28, 90]) - np.array([90,108, 90])
            l = [0, 0, 1];
            m = np.cross(orig_pos, l);
            theta = np.deg2rad(amplitude);
            disp = 0;
            dq = DualQuaternion.from_screw(l, m, theta, disp)
            fp = np.tile(spm_imatrix(dq.homogeneous_matrix(), order=0)[:6, np.newaxis], (1, resolution))
            fp[:, y == 0] = 0
        else:
            for xx in mvt_axes:
                fp[xx, :] = y
            if len(mvt_axes)>1:
                if len(mvt_axes)==3:
                    max0 = np.max(fp)
                    fp = np.sqrt(fp**2 /3);
                    print(f'because of 3 ax reducing amplitude from {max0} to {np.max(fp)} of 0.86' )
                else:
                    print('WARNING pb in amplitude norm')
        y=fp

    return y

def get_displacement_field_metric(image, brain_mask, fitpar):

    #prepare weigths to check size
    img_fft = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)
    coef_TF = np.sum(abs(img_fft), axis=(0,2)) ;
    coef_shaw = np.sqrt( np.sum(abs(img_fft**2), axis=(0,2)) ) ;

    if fitpar.shape[1] != coef_TF.shape[0] :
        #just interpolate end to end. at image slowest dimention size
        fitpar = interpolate_fitpars(fitpar, len_output=coef_TF.shape[0])
        print(f'displacement_field: interp fitpar for wcoef new shape {fitpar.shape}')

    nbt = fitpar.shape[1]

    brain_mask[brain_mask > 0] = 1  # need pure mask
    mean_disp_mask = np.zeros(nbt);    wimage_disp = np.zeros(nbt)    #mean_disp = np.zeros(nbt);
    max_disp = np.zeros(nbt);    min_disp = np.zeros(nbt)

    for i in range(nbt):
        P = np.hstack((fitpar[:, i], np.array([1, 1, 1, 0, 0, 0])))
        aff = spm_matrix(P, order=0)

        disp_norm = get_dist_field(aff, list(image.shape))
        # disp_norm_small = get_dist_field(aff, [22,26,22], scale=8)
        disp_norm_mask = disp_norm * (brain_mask)
        # mean_disp[i] = np.mean(disp_norm)  #not usefull since take the all cube
        max_disp[i], min_disp[i] = np.max(disp_norm_mask), np.min(disp_norm_mask[brain_mask > 0])
        #compute the weighted displacement (weigths beeing the image intensity) but wihtin brain mask !
        wimage_disp[i] = np.sum(disp_norm_mask * image) / np.sum(image)
        mean_disp_mask[i] = np.sum(disp_norm_mask) / np.sum(brain_mask)

    #compute weighted metrics
    res = dict(
        mean_disp_mask = mean_disp_mask,
        max_disp = max_disp,
        min_disp = min_disp,
        MD_mask_mean = np.mean(mean_disp_mask),
        MD_wimg_mean = np.mean(wimage_disp),
        MD_mask_wTF  = np.sum(mean_disp_mask * coef_TF) / np.sum(coef_TF),
        MD_mask_wTF2 = np.sum(mean_disp_mask * coef_TF**2) / np.sum(coef_TF**2),
        MD_mask_wSH  = np.sum(mean_disp_mask * coef_shaw) / np.sum(coef_shaw),
        MD_mask_wSH2 = np.sum(mean_disp_mask * coef_shaw**2) / np.sum(coef_shaw**2),
        MD_wimg_wTF  = np.sum(wimage_disp * coef_TF) / np.sum(coef_TF),
        MD_wimg_wTF2 = np.sum(wimage_disp * coef_TF**2) / np.sum(coef_TF**2),
        MD_wimg_wSH  = np.sum(wimage_disp * coef_shaw) / np.sum(coef_shaw),
        MD_wimg_wSH2 = np.sum(wimage_disp * coef_shaw**2) / np.sum(coef_shaw**2),
    )

    df = pd.DataFrame();
    df = df.append(res, ignore_index=True)
    return df

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

def spm_imatrixOLD(M):
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

def spm_imatrix(M, order=1):
    def rang(x):
        return np.min([np.max([x,-1]), 1])
    import nibabel as ni
    import numpy.linalg as npl

    if order==0: #change rotation Translation order   ie: get same trans as those given by spm_matrix(order=0)
        rot = M.copy();            rot[:, 3] = [0, 0, 0, 1]
        trans2 = M.copy();        trans2[:3, :3] = np.eye(3)
        trans1 = np.matmul(npl.inv(rot), np.matmul(trans2, rot))
        M[:,3] = trans1[:,3]

    #translation euler angle  scaling shears % copy from from spm_imatrix
    [rot,[tx,ty,tz]] = ni.affines.to_matvec(M)

    C = npl.cholesky(np.transpose(rot).dot(rot)).T
    #C = np.transpose(npl.cholesky(rot.dot(np.transpose(rot))))
    [scalx, scaly, scalz] = np.diag(C)
    if npl.det(rot)<0:  # Fix for -ve determinants
        scalx=-scalx

    C = np.matmul(npl.inv(np.diag(np.diag(C))),C)
    [shearx, sheary, shearz] = [C[0,1], C[0,2], C[1,2]]

    P=np.array([0,0,0,0,0,0,scalx, scaly, scalz,shearx, sheary, shearz])
    R0=spm_matrix(P,order=4)
    R0=R0[:3,:3]
    #rrr    R1 = rot.dot(npl.inv(R0))
    R1 = np.matmul(rot, npl.inv(R0)) #(npl.inv(R0)).dot(rot)

    ry = np.arcsin(R1[0,2])
    if (abs(ry) - np.pi / 2) ** 2 < 1e-9:
        rx = 0;
        rz = np.arctan2(-rang(R1[1, 0]), rang(-R1[2, 0] / R1[0, 2]));
    else:
        c = np.cos(ry);
        rx = np.arctan2(rang(R1[1, 2] / c), rang(R1[2, 2] / c));
        rz = np.arctan2(rang(R1[0, 1] / c), rang(R1[0, 0] / c));

    #[rz, ry, rx]  = ni.eulerangles.mat2euler(R1)
    [rz, ry, rx] = [rz*180/np.pi, ry*180/np.pi, rx*180/np.pi] #rad to degree
    #rrr I do not know why I have to add - to make it coherent
    #P = np.array([tx,ty,tz,-rx,-ry,-rz,scalx, scaly, scalz,shearx, sheary, shearz])
    P = np.array([tx, ty, tz, rx, ry, rz, scalx, scaly, scalz, shearx, sheary, shearz])
    return P


def spm_matrix(P, order=1, set_ZXY=False, rotation_center=None):
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

    #[P[3], P[4], P[5]] = [P[3] * 180 / np.pi, P[4] * 180 / np.pi, P[5] * 180 / np.pi]  # degre to radian
    P[3], P[4], P[5] = P[3]*np.pi/180, P[4]*np.pi/180, P[5]*np.pi/180 #degre to radian

    T = np.array([[1,0,0,P[0]],[0,1,0,P[1]],[0,0,1,P[2]],[0,0,0,1]])
    R1 =  np.array([[1,0,0,0],
                    [0,np.cos(P[3]),np.sin(P[3]),0],
                    [0,-np.sin(P[3]),np.cos(P[3]),0],
                    [0,0,0,1]])
    R2 =  np.array([[np.cos(P[4]),0,-np.sin(P[4]),0],  #sing change compare to spm to match tio Affine
                    [0,1,0,0],
                    [np.sin(P[4]),0,np.cos(P[4]),0],
                    [0,0,0,1]])
    R3 =  np.array([[np.cos(P[5]),np.sin(P[5]),0,0],
                    [-np.sin(P[5]),np.cos(P[5]),0,0],
                    [0,0,1,0],
                    [0,0,0,1]])

    #R = R3.dot(R2.dot(R1)) #fsl convention (with a sign difference)
    if set_ZXY:
        R = (R2.dot(R1)).dot(R3)
    else:
        R = (R1.dot(R2)).dot(R3)

    Z = np.array([[P[6],0,0,0],[0,P[7],0,0],[0,0,P[8],0],[0,0,0,1]])
    S = np.array([[1,P[9],P[10],0],[0,1,P[11],0],[0,0,1,0],[0,0,0,1]])
    if order == 0:
        A = S.dot(Z.dot(R.dot(T)))
    else:
        A = T.dot(R.dot(Z.dot(S)))

    if rotation_center is not None:
        A = change_affine_rotation_center(A, rotation_center)

    return A

def change_affine_rotation_center(A, new_center):
    aff_center = np.eye(4);
    aff_center[:3, 3] = np.array(new_center)
    return np.dot(aff_center, np.dot(A, np.linalg.inv(aff_center)))


def ras_to_lps_vector(triplet: np.ndarray):
    return np.array((-1, -1, 1), dtype=float) * np.asarray(triplet)

def ras_to_lps_affine(affine):
    FLIPXY_44 = np.diag([-1, -1, 1, 1])
    affine = np.dot(affine, FLIPXY_44)
    affine = np.dot(FLIPXY_44, affine)
    return affine


def get_matrix_from_euler_and_trans(P, rot_order='yxz', rotation_center=None):
    from transforms3d.euler import euler2mat

    # default choosen as the same default as simpleitk
    rot = np.deg2rad(P[3:6])
    aff = np.eye(4)
    aff[:3,3] = P[:3]  #translation
    if rot_order=='xyz':
        aff[:3,:3]  = euler2mat(rot[0], rot[1], rot[2], axes='sxyz')
    elif rot_order=='yxz':
        aff[:3,:3] = euler2mat(rot[1], rot[0], rot[2], axes='syxz') #strange simpleitk convention of euler ... ?
    else:
        raise(f'rotation order {rot_order} not implemented')

    if rotation_center is not None:
        aff = change_affine_rotation_center(aff, rotation_center)
    return aff

def get_euler_and_trans_from_matrix(aff, rot_order = 'yxz', rotation_center=None):
    from transforms3d.euler import mat2euler

    if rot_order=='xyz':
        revers_angl = np.rad2deg(mat2euler(aff[:3, :3], axes='sxyz'))
    elif rot_order=='yxz':
        rrr = np.rad2deg(mat2euler(aff[:3, :3], axes='syxz'))
        revers_angl = np.array([rrr[1], rrr[0], rrr[2]])
    else:
        raise(f'rotation order {rot_order} not implemented')

    if rotation_center is not None:
        aff = change_affine_rotation_center(aff, - rotation_center)
    revers_tran = aff[:3,3]

    return np.hstack([revers_tran, revers_angl])


def itk_euler_to_affine(Euler_angle, Translation, v_center, degree_to_radian=True, make_RAS=False,
                    set_ZYX=False):

    if degree_to_radian:
        Euler_angle = np.deg2rad(Euler_angle)
    if make_RAS: #RAS to LPS   not very usefull here since, we invert it again of the affine at the end :
        Euler_angle = ras_to_lps_vector(Euler_angle)
        Translation = ras_to_lps_vector(Translation)
        #v_center = ras_to_lps(v_center)  suppose to be given already in lps ... !

    rigid_euler = sitk.Euler3DTransform(v_center,Euler_angle[0],Euler_angle[1],Euler_angle[2],Translation)
    if set_ZYX:
        rigid_euler.SetComputeZYX(True) #default is ZXY !!! arggg !!!!

    A1 = np.asarray(rigid_euler.GetMatrix()).reshape(3,3)
    c1 = np.asarray(rigid_euler.GetCenter())
    t1 = np.asarray(rigid_euler.GetTranslation())
    affine = np.eye(4)
    affine[:3,:3] = A1
    affine[:3,3] = t1+c1 - np.matmul( A1,c1)

    """LPS to RAS"""
    if make_RAS :
        affine = ras_to_lps_affine(affine)
    return affine

def get_affine_from_Elastixtransfo(elastixImageFilter, do_print=False):
    tp = elastixImageFilter.GetTransformParameterMap()[0]

    v_center = np.array(np.double(tp['CenterOfRotationPoint'])) #* np.array((-1, -1, 1), dtype=float)

    Euler_angle = np.array(np.double(tp['TransformParameters'][:3]))
    Translation = np.array(np.double(tp['TransformParameters'][3:]))
    affine = itk_euler_to_affine(Euler_angle, Translation, v_center,
                             degree_to_radian=False, make_RAS=False)


    affine = ras_to_lps_affine(affine)
    affine = np.linalg.inv(affine)  #because itk store matrix for ref to src

    if do_print:
        print(f"Found Rot  {np.rad2deg(Euler_angle)} ")
        print(f"Trans      {Translation}")
        print(f"vcenter    {v_center}")

    return affine

def ElastixRegister(img_src, img_ref):

    # inputs are tio Images
    img1 = img_src.data # if img_src.data.shape[0] == 1 else no4C
    img2 = img_ref.data # if img_ref.data.shape[0] == 1 else no4C

    i1, i2 = tio.io.nib_to_sitk(img1, img_src.affine), tio.io.nib_to_sitk(img2.data, img_ref.affine)
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(i2);
    elastixImageFilter.SetMovingImage(i1)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.Execute()
    return get_affine_from_Elastixtransfo(elastixImageFilter), elastixImageFilter

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


def get_screw_from_affine(affine):
    dq = DualQuaternion.from_homogeneous_matrix(affine)
    s_ax_dir, m, theta, d = dq.screw()
    theta = np.rad2deg(theta)
    return s_ax_dir, m, theta, d
def get_info_from_quat(q):
    angle = get_rotation_angle(q)
    ax = get_rotation_axis(q)
    return dict(angle=angle, ax=ax)
def get_info_from_dq(dq, verbose=False):
    l, m, tt, dd = dq.screw()#(rtol=1e-5)
    theta = np.rad2deg(tt); disp = dd
    if npl.norm(l)<1e-10:
        origin_pts = [0, 0, 0]
    else:
        #origin_pts = np.cross(m,l) # because norm of l is 1
        origin_pts = np.cross(l,m) # change to match set get and have the origine where it should
        #so with this the set is done from orig_pos with  m = np.cross(orig_pos,l);
        #from spatialmath         return np.cross(self.v, self.w) / np.dot(self.w, self.w)    # where V (3-vector) is the moment and W (3-vector) is the line direction.
        #origin_pts = np.cross(m,l)/np.dot(l,l)
    line_distance = npl.norm(m)
    # from spatialmath        return math.sqrt(np.dot(self.v, self.v) / np.dot(self.w, self.w) )
    #line_dist2 = np.sqrt(np.dot(m,m)/np.dot(l,l))
    trans = dq.translation()
    res = dict(l=l, m=m, origin_pts=origin_pts, line_dist=line_distance, disp=disp, theta=theta, trans=trans)
    if verbose:
        for k in res:
            print(f'{k}: {res[k]}')
    return res

#to and from euler with different representation using spm_matrix euler conversion
def get_affine_rot_from_euler(e_array, mode='spm'):
    if mode=='spm':
        aff_list=[]
        for r in e_array:
            aff_list.append(spm_matrix([0, 0, 0, r[0],r[1],r[2], 1, 1, 1, 0, 0, 0], order=1) )
            #aff_list.append(spm_matrix([0, 0, 0, r[0], 0,0, 1, 1, 1, 0, 0, 0], order=1))
        return aff_list
    else:
        print('waring this does not work properly, with negativ euler angles ... ')
        aff_list = []
        for r in e_array:
            r = np.deg2rad(r)
            qr = nq.from_euler_angles(r)
            aff = np.eye(4, 4);
            aff[:3, :3] = nq.as_rotation_matrix(qr)
            aff_list.append(aff)
        return aff_list

# same here : to and from euler with different representation but using quaternion euler convention
def get_modulus_euler_in_degree(euler):
    euler = np.rad2deg(euler)
    for index, e in enumerate(euler):
        if e < -180:
            euler[index] = e + 360
        if e > 180:
            euler[index] = e -360
    return euler
def get_euler_from_qr(qr,  mode='spm'):
    if mode=='spm':
        aff = np.eye(4,4)
        aff[:3,:3] = nq.as_rotation_matrix(qr)
        P = spm_imatrix(aff)
        return P[3:6]
    else:
        return get_modulus_euler_in_degree(nq.as_euler_angles(qr))
def get_euler_from_dq(dq, mode='spm'):
    if mode=='spm':
        P = spm_imatrix(dq.homogeneous_matrix())
        return P[3:6]
    else:
        qr = nq.from_rotation_matrix(dq.homogeneous_matrix())
        return get_modulus_euler_in_degree(nq.as_euler_angles(qr))
def get_euler_from_affine(aff, mode='spm'):
    if mode=='spm':
        P = spm_imatrix(aff)
        return P[3:6]
    else:
        qr = nq.from_rotation_matrix(aff)
        return get_modulus_euler_in_degree(nq.as_euler_angles(qr))

#let's explore quaternion
def get_rotation_angle2(q):
    angle = np.linalg.norm(nq.as_rotation_vector(q))
    if angle>np.pi:
        angle = angle - 2*np.pi
    return np.rad2deg(angle)
def get_rotation_angle(q):
    qa=nq.as_float_array(q)
    angle = 2*np.arctan2(np.sqrt(qa[1]**2+qa[2]**2+qa[3]**2),qa[0])
    if angle>np.pi:
        angle = angle - 2*np.pi
    return np.rad2deg(angle)
def get_rotation_axis(q):
    qa = nq.as_float_array(q)
    return qa[1:] / np.sqrt(qa[1]**2+qa[2]**2+qa[3]**2)

def interpolate_fitpars(fpars, tr_fpars=None, tr_to_interpolate=2.4, len_output=250):
    fpars_length = fpars.shape[1]
    if tr_fpars is None: #case where fitpart where give as it in random motion (the all timecourse is fitted to kspace
        xp = np.linspace(0,1,fpars_length)
        x  = np.linspace(0,1,len_output)
    else:
        xp = np.asarray(range(fpars_length))*tr_fpars
        x = np.asarray(range(len_output))*tr_to_interpolate
    interpolated_fpars = np.asarray([np.interp(x, xp, fp) for fp in fpars])
    if xp[-1]<x[-1]:
        diff = x[-1] - xp[-1]
        npt_added = diff/tr_to_interpolate
        print(f'adding {npt_added:.1f}')
    return interpolated_fpars

def geodesicL2Mean(aff_list, Rmean_estimate):

    print(f'Rmean {Rmean_estimate}')
    for i in range(10):
        diff_aff = np.zeros_like(aff_list[0])
        for aff in aff_list:
            diff_aff = diff_aff + scl.logm(np.matmul(Rmean_estimate.T, aff))
        diff_aff = diff_aff / len(aff_list)
        print(f'error in matrix Mean is {npl.norm(diff_aff)}')
        Rmean_estimate = np.matmul(Rmean_estimate, scl.expm(diff_aff).T)
        print(f'Rmean {Rmean_estimate}')

    return Rmean_estimate

#Average fitpar
def average_fitpar(fitpar, weights=None):
    aff_list = []  #can't
    Aff_mean = np.zeros((4, 4))
    if weights is None:
        weights = np.ones(fitpar.shape[1])
    #normalize weigths
    weights = weights / np.sum(weights)

    dq_mean = DualQuaternion.identity()
    lin_fitpar = np.sum(fitpar*weights, axis=1)
    for nbt in range(fitpar.shape[1]):
        P = np.hstack([fitpar[:,nbt],[1,1,1,0,0,0]])
        affine = spm_matrix(P.copy(),order=0)  #order 0 to get the affine really applid in motion (change 1 to 0 01/04/21) it is is equivalent, since at the end we go back to euleur angle and trans...
        aff_list.append(affine)
        # new_q = nq.from_rotation_matrix(affine)
        # if 'previous_q' not in dir():
        #     previous_q = new_q
        # else:
        #     _ = nq.unflip_rotors([previous_q, new_q], inplace=True)
        #     previous_q = new_q
        #q_list.append( new_q  )
        Aff_mean = Aff_mean + weights[nbt] * scl.logm(affine)
        dq = DualQuaternion.from_homogeneous_matrix(affine)
        #dq_mean = DualQuaternion.sclerp(dq_mean, dq,(weights[nbt])/(1+weights[nbt]))
        dq_meanE = weights[nbt] * dq if 'dq_meanE' not in dir() else  dq_meanE + weights[nbt] * dq

    dq_meanE.normalize()
    Aff_mean = scl.expm(Aff_mean)
    wshift_exp = spm_imatrix(Aff_mean, order=0)[:6]

    Aff_mean2 = geodesicL2Mean(aff_list, Aff_mean)
    wshift_exp2 = spm_imatrix(Aff_mean2, order=0)[:6]

    #q_mean = nq.mean_rotor_in_chordal_metric(q_list)
    #Aff_q_rot = nq.as_rotation_matrix(q_mean)

    #Aff_q_rot = scl.polar(Aff_Euc_mean)[0]
    #Aff_q = np.eye(4)
    #Aff_q[:3,:3] = Aff_q_rot
    #wshift_quat = spm_imatrix(Aff_q, order=0)[:6]
    #wshift_quat[:3] = lin_fitpar[:3]
    #wshift_quat = spm_imatrix(dq_mean.homogeneous_matrix(), order=0)[:6]
    wshift_quatE = spm_imatrix(dq_meanE.homogeneous_matrix(), order=0)[:6]

    return wshift_quatE, wshift_exp, lin_fitpar, wshift_exp2


def paw_quaternion(qr, exponent):
    rot_vector = nq.as_rotation_vector(qr)
    #theta = np.linalg.norm(rot_vector)
    theta = 2 * np.arccos(nq.as_float_array(qr)[0] )  #equivalent
    s0 = rot_vector / np.sin(theta / 2)

    quaternion_scalar = np.cos(exponent*theta/2)
    quaternion_vector = s0 * np.sin(exponent*theta/2)

    return nq.as_quat_array( np.hstack([quaternion_scalar, quaternion_vector]) )
def exp_mean_affine(aff_list, weights=None):
    if weights is None:
        weights = np.ones(len(aff_list))
    #normalize weigths
    weights = weights / np.sum(weights)

    Aff_mean = np.zeros((4, 4))
    for aff, w in zip(aff_list, weights):
        Aff_mean = Aff_mean + w*scl.logm(aff)
    Aff_mean = scl.expm(Aff_mean)
    return Aff_mean
def polar_mean_affin(aff_list, weights=None):
    if weights is None:
        weights = np.ones(len(aff_list))
    #normalize weigths
    weights = weights / np.sum(weights)
    Aff_Euc_mean = np.zeros((3, 3))
    for aff, w in zip(aff_list, weights):
        Aff_Euc_mean = Aff_Euc_mean + w * aff[:3, :3]

    Aff_mean = np.eye(4)
    Aff_mean[:3,:3] = scl.polar(Aff_Euc_mean)[0]
    return Aff_mean
def dq_euclidian_mean(qr_list, weights=None):
    if weights is None:
        weights = np.ones(len(aff_list))
    #normalize weigths
    weights = weights / np.sum(weights)

    for ii, qr in enumerate(qr_list):
        if ii==0:
            res_mean =  weights[ii]*qr
        else:
            res_mean = res_mean + weights[ii]*qr
    res_mean.normalize()
    return res_mean

def dq_slerp_mean(dq_list):
    c_num, c_deno = 1, 2
    for ii, dq in enumerate(dq_list):
        if ii==0:
            res_mean = dq
        else:
            t = 1 - c_num/c_deno
            res_mean = DualQuaternion.sclerp(res_mean, dq, t) #res_mean * c_num + dq) / c_deno
            c_num+=1; c_deno+=1
    return res_mean
def qr_slerp_mean(qr_list):
    c_num, c_deno = 1, 2
    for ii, qr in enumerate(qr_list):
        if ii==0:
            res_mean = qr
        else:
            # qr_mult = res_mean*qr
            # if nq.as_float_array(qr_mult)[0] < 0:
            #     #print("changing sign !!")
            #     res_mean = res_mean

            t = 1 - c_num/c_deno
            res_mean = nq.slerp(res_mean, qr, 0, 1, t) #res_mean * c_num + dq) / c_deno
            c_num+=1; c_deno+=1
    return res_mean
def qr_euclidian_mean(qr_list, weights=None):
    if weights is None:
        weights = np.ones(len(aff_list))
    #normalize weigths
    weights = weights / np.sum(weights)

    for ii, qr in enumerate(qr_list):
        if ii==0:
            res_mean =  weights[ii]*qr
        else:
            res_mean = res_mean + weights[ii]*qr

    return res_mean / res_mean.norm()

def my_mean(x_list):
    #just decompose the mean as a recursiv interpolation between 2 number, (to be extend to slerp interpolation)
    c_num, c_deno = 1, 2
    for ii, x in enumerate(x_list):
        if ii==0:
            res_mean = x
        else:
            t = 1 - c_num/c_deno
            res_mean = np.interp(t, [0,1], [res_mean, x])
            #res_mean = (res_mean * c_num + x) / c_deno
            c_num+=1; c_deno+=1
    return res_mean
#x = np.random.random(100)
#np.mean(x)-my_mean(x)

def random_rotation(amplitude = 360):
    """
    from_rotation_vector(rot):
    rot : (Nx3) float array
        Each vector represents the axis of the rotation, with norm
        proportional to the angle of the rotation in radians.
    """
    #get random orientation
    #V = np.random.uniform(-1,1,3) #np.random.rand(3)
    V = np.random.normal(size=3) #to get uniform orientation ...
    V = V / npl.norm(V) #warning if small norm ... ?
    theta = np.random.rand(1) * amplitude / 360 * np.pi * 2
    V = V * theta
    quat = nq.from_rotation_vector(V)
    aff = np.eye(4, 4);
    aff[:3,:3] = nq.as_rotation_matrix(quat)
    return aff

def get_random_vec(range=[-1,1], size=3, normalize=False):
    if normalize:
        #if normalize, I assume on want unifor orientation ie, uniform point on a sphere r=1
        res = np.random.normal(size=3)  # to get uniform orientation on the sphere ...
        res = res/npl.norm(res)
        return res
    res = np.random.uniform(low=range[0], high=range[1], size=size )
    return res
def get_random_afine(angle=(2,10), trans=(0,0), origine=(80,100), mode='quat'):
    if mode == 'quat':
        theta = np.deg2rad(get_random_vec(angle,1)[0])
        l = get_random_vec(normalize=True);
        #orig = get_random_vec(normalize=True) * get_random_vec(origine,1)
        #m = np.cross(orig, l);
        #this does not work because only the projection of orig in the normal plan of l, is taken
        # so add the wanted distance from origine directly to m
        orig = get_random_vec(normalize=True)
        m = np.cross(orig, l)
        m = m / npl.norm(m) * get_random_vec(origine,1)
        disp = get_random_vec(trans,1)[0];
        #print(f'dual quat with l {l} m {m}  norm {npl.norm(m)} theta {np.rad2deg(theta)} disp {disp}')
        dq = DualQuaternion.from_screw(l, m, theta, disp)
        #get_info_from_dq(dq, verbose=True)
        aff = dq.homogeneous_matrix()
    if mode == 'quat2':
        theta = np.deg2rad(get_random_vec(angle,1)[0])
        l = get_random_vec(normalize=True);
        #m = get_random_vec(normalize=True)
        #m = m  * get_random_vec(origine,1)
        m = np.zeros(3)
        m[:2] = get_random_vec(size=2)
        m[2] = -(l[0]*m[0] + l[1]*m[1] ) / l[2]
        m = m/npl.norm(m) * get_random_vec(origine,1)
        disp = get_random_vec(trans,1)[0];
        #print(f'dual quat with l {l} m {m}  norm {npl.norm(m)} theta {np.rad2deg(theta)} disp {disp}')
        dq = DualQuaternion.from_screw(l, m, theta, disp)
        #res=get_info_from_dq(dq, verbose=True)
        if abs(res['disp'])>5:
            qsdf
        aff = dq.homogeneous_matrix()
    if mode == 'euler':
        fp = np.ones(12); fp[9:]=0
        fp[3:6] = get_random_vec(angle,3)
        fp[:3] = get_random_vec(trans,3)
        aff = spm_matrix(fp, order=0)
    if mode == 'euler2':
        fp = np.ones(12); fp[9:]=0
        angle_amplitude = get_random_vec(angle,1)
        trans_amplitude = get_random_vec(trans, 1)
        fp[:3] =  get_random_vec(normalize=True) * trans_amplitude
        fp[3:6] = get_random_vec(normalize=True) * angle_amplitude
        aff = spm_matrix(fp, order=0)
    if mode == 'euler3':
        fp = np.ones(12); fp[9:]=0
        angle_amplitude = np.random.normal(loc=angle[1]/2, scale=angle[1],size=1)
        trans_amplitude = np.random.normal(loc=trans[1]/2, scale=trans[1],size=1)
        fp[:3] =  get_random_vec(normalize=True) * trans_amplitude
        fp[3:6] = get_random_vec(normalize=True) * angle_amplitude
        aff = spm_matrix(fp, order=0)
    return(aff)

def random_rotation_matrix_test(amplitude=1, randgen=None):
    """
    Bof bof, I do not understand the resulting theta we get if we restric amplitude (if not seems ok)
    Creates a random rotation matrix.
    randgen: if given, a np.random.RandomState instance used for random numbers (for reproducibility)
    # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    *  R A N D _ R O T A T I O N      Author: Jim Arvo, 1991                  *
    *                                                                         *
    *  This routine maps three values (x[0], x[1], x[2]) in the range [0,1]   *
    *  into a 3x3 rotation matrix, M.  Uniformly distributed random variables *
    *  x0, x1, and x2 create uniformly distributed random rotation matrices.  *
    *  To create small uniformly distributed "perturbations", supply          *
    *  samples in the following ranges                                        *
    *                                                                         *
    *      theta: x[0] in [ 0, d ]                                                   *
    *      phi  : x[1] in [ 0, 1 ]                                                   *
    *      z    : x[2] in [ 0, d ]                                                   *
    *                                                                         *
    * where 0 < d < 1 controls the size of the perturbation.  Any of the      *
    * random variables may be stratified (or "jittered") for a slightly more  *
    * even distribution.
    I add the amplitude (d) as argument
    """
    if randgen is None:
        randgen = np.random.RandomState()

    theta, phi, z = tuple(randgen.rand(3).tolist())
    theta = theta * 2.0*np.pi * amplitude # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0 #* amplitude # For magnitude of pole deflection.

    print(f'theta is {np.rad2deg(theta)}')

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = ( np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z)) #

    st = np.sin(theta)
    ct = np.cos(theta)
    Sx = Vx * ct - Vy * st;
    Sy = Vx * st + Vy * ct;

    # Construct the rotation matrix  ( V Transpose(V) - I ) R, which is equivalent to V S - R.                                        */
    aff = np.eye(4, 4);
    aff[0,0] = Vx * Sx - ct;
    aff[0,1] = Vx * Sy - st;
    aff[0,2] = Vx * Vz;

    aff[1,0] = Vy * Sx + st;
    aff[1,1] = Vy * Sy - ct;
    aff[1,2] = Vy * Vz;

    aff[2,0] = Vz * Sx;
    aff[2,1] = Vz * Sy;
    aff[2,2] = 1.0 - z;   # This equals Vz * Vz - 1.0

    #same ...
    #R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    #M = (np.outer(V, V) - np.eye(3)).dot(R)
    return aff

#import SimpleITK as sitk #cant understand fucking convention with itk TransformToDisplacementField
def get_sphere_mask(image, radius=80):
    mask = np.zeros_like(image)
    (sx, sy, sz) = image.shape  # (64,64,64) #
    center = [sx // 2, sy // 2, sz // 2]

    [kx, ky, kz] = np.meshgrid(np.arange(0, sx, 1), np.arange(0, sy, 1), np.arange(0, sz, 1), indexing='ij')
    [kx, ky, kz] = [kx - center[0], ky - center[1], kz - center[2]]  # to simulate rotation around center
    ijk = np.stack([kx, ky, kz])
    dist_ijk = npl.norm(ijk,axis=0)
    mask[dist_ijk<radius] = 1
    return mask

#exact displacement field on a grid
def get_dist_field(affine, img_shape, return_vect_field=False, scale=None):

    (sx, sy, sz) = img_shape  # (64,64,64) #
    center = [sx // 2, sy // 2, sz // 2]

    [kx, ky, kz] = np.meshgrid(np.arange(0, sx, 1), np.arange(0, sy, 1), np.arange(0, sz, 1), indexing='ij')
    [kx, ky, kz] = [kx - center[0], ky - center[1], kz - center[2]]  # to simulate rotation around center
    ijk = np.stack([kx, ky, kz, np.ones_like(kx)])
    ijk_flat = ijk.reshape((4, -1))
    if scale is not None:
        sc_mat = np.eye(4) * scale; sc_mat[3,3] = 1
        affine = sc_mat.dot(affine)
    xyz_flat = affine.dot(ijk_flat)  # implicit convention reference center at 0,0,0
    if scale is None:
        disp_flat = xyz_flat - ijk_flat
    else:
        disp_flat = xyz_flat - ijk_flat*scale

    disp_norm = npl.norm(disp_flat[:3, :], axis=0)
    if return_vect_field:
        return disp_flat[:3, :].reshape([3] + img_shape)
    return disp_norm.reshape(img_shape)

#displacement quantification
def compute_FD_P(fp, rmax=80):
    #https://github.com/FCP-INDI/C-PAC/blob/master/CPAC/generate_motion_statistics/generate_motion_statistics.py
    fd = np.sum(np.abs(fp[:3])) + (rmax * np.pi/180) * np.sum(np.abs(fp[3:6]))
    return fd
def compute_FD_J(aff, rmax=80, center_of_mass=np.array([0,0,0])):
    M = aff - np.eye(4)
    A = M[0:3, 0:3]
    b = M[0:3, 3]   # np.sqrt(np.dot(b.T, b))  is the norm of translation vector
    b = b + np.dot(A,center_of_mass)
    fd = np.sqrt( (rmax * rmax / 5) * np.trace(np.dot(A.T, A)) + np.dot(b.T, b) )
    return fd
def compute_FD_max(aff, rmax=80):
    #tisdall et all MRM 2012
    res = get_info_from_dq(DualQuaternion.from_homogeneous_matrix(aff))
    theta = np.deg2rad(res['theta'])
    deltaR = rmax * ( (1 - np.cos(theta))**2 + (np.sin(theta)**2) )**0.5 #maximum disp on a sphere
    fd = deltaR + npl.norm(res['trans'])
    return fd
def compute_FD_test(x):
    trans = npl.norm(x['trans'])
    rot = x['theta']
    fd = trans + 80 / np.sqrt(1) * (np.pi/180) * rot
    return fd/2

