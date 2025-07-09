import torchio as tio, numpy as np
import torch, pandas as pd, nibabel as nib

from utils_file import get_parent_path, gfile, gdir, addprefixtofilenames
from utils_labels import get_mask_external_broder, pool_remap
from utils_labels import remap_filelist, get_fastsurfer_remap, get_remap_from_csv,resample_and_smooth4D
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage import label as scipy_label
import subprocess, os
from script.create_jobs import create_jobs
import skimage

def compute_GMpv_from_freesurferSurf(dfreeS, fref_list='/network/iss/opendata/data/HCP/training39/100307/T1w/r025_T1w.nii.gz',
                                     dtpm='/data/romain/tmp/freeS', PVthr=0.5, out_prefix = 'r025_', clean_tmp=True, convex_GM=False):
    my_env = os.environ.copy()
    my_env["FREESURFER_HOME"] = "/network/lustre/iss01/apps/software/noarch/freesurfer/7.4.1"
    if not isinstance(fref_list,list): #make it a list
        fref_list = [fref_list for k in dfreeS]

    for one_suj, fref in zip(dfreeS, fref_list):
        fsurf = gfile(one_suj, '(h.pial$|h.white$)')
        fos = [f'{dtpm}/{ff}.vtk' for ff in get_parent_path(fsurf)[1]]
        #from surface to PV GM WM
        for f1,fo in zip(fsurf,fos):
            cmd = f'/network/lustre/iss01/apps/software/noarch/freesurfer/7.4.1/bin/mris_convert --to-scanner {f1} {fo}'
            print(f'Running {cmd}')
            outvalue = subprocess.run(cmd.split(' '), env=my_env)
            cmd = f'mesh2voxel -nthreads 34 {fo} {fref} {fo}.nii.gz'
            outvalue = subprocess.run(cmd.split(' '))

        f1 = os.path.join(dtpm,'lh.white.vtk.nii.gz');  f2 = os.path.join(dtpm,'rh.white.vtk.nii.gz')
        f3 = os.path.join(dtpm,'lh.pial.vtk.nii.gz');   f4 = os.path.join(dtpm,'rh.pial.vtk.nii.gz')
        if convex_GM:
            ill = [ tio.ScalarImage(f3), tio.ScalarImage(f4) ]
            foos = [ f'{dtpm}/convex_{ss}h_pial.nii.gz' for ss in ['l','r']]
            for img_tio,foo in zip(ill,foos):
                datain = img_tio.data[0].numpy()>0
                datac = skimage.morphology.convex_hull_image(datain)
                img_tio.data[0] = torch.tensor(datac)
                img_tio.save(foo)

        fo2 = f'{dtpm}/mask_wm_not_gm.nii.gz'
        #inter GM/WM is WM
        cmd = (f'mrcalc {f1} 0 -gt {f1} 1 -lt -and {f2} 0 -gt {f2} 1 -lt -and -max {f3} 0 -gt {f3} 1 -lt -and {f4} 0 -gt {f4} 1 -lt -and -max -and {fo2}')
        outvalue = subprocess.run(cmd.split(' '))
        fo3 = f'{dtpm}/largestCC_mask_wm_not_gm.nii.gz'
        cmd = (f'maskfilter -largest {fo2} connect {fo3}'); outvalue = subprocess.run(cmd.split(' '))

        igm = tio.ScalarImage([f3, f4]); iwm = tio.ScalarImage([f1, f2])
        ia = tio.ScalarImage([fo2,fo3])
        one_dic = {}
        ma_wm_not_gm = ia.data[1]>0 ; ma_gm_not_wm = (ia.data[0]>0) & ~(ma_wm_not_gm)
        one_dic['sufWM_inter_surGM'] = ma_gm_not_wm.sum().numpy()
        GM = torch.max(igm.data[0], igm.data[1]); WM = torch.max(iwm.data[0], iwm.data[1])
        GM[ma_wm_not_gm] = 0;  WM[ma_wm_not_gm] = 1;
        GM[ma_gm_not_wm] = 1;  WM[ma_gm_not_wm] = 0;

        if PVthr==0 :
            print('taking up GM')
            GM[(GM>0) & (GM<1)] = 1 ; WM[(WM>0) & (WM<1)] = 0
        elif PVthr==0.5:
            print('taking 05 PV to bin')
            GM[(GM >= 0.5) & (GM < 1)] = 1; WM[(WM >= 0.5) & (WM < 1)] = 1
            GM[(GM > 0) & (GM < 0.5)] = 0; WM[(WM > 0) & (WM < 0.5)] = 0
        elif PVthr == 1:
            print('taking down GM')
            GM[(GM>0) & (GM<1)] = 0;  WM[(WM>0) & (WM<1)] = 1  #erode GM/CSF and GM/WM interface no PV
        else :
            error_PV_not_known
        GM[WM>0] = 0;  #clean wrong GM and remove GM inside_mask
            #GM[(WM>0) & (WM<1)] = 1-WM[(WM>0) & (WM<1)]  #PV with WM

        io = tio.ScalarImage(f1)
        io.data = torch.zeros([1]+list(igm.data.shape[-3:]), dtype=torch.int)
        Sout = io.data[0]
        Sout[GM>0] = 1;     Sout[WM>0] = 2
        io.data[0] = Sout
        io.save(os.path.join(one_suj,f'{out_prefix}GM_WM.nii.gz'))

        #clean
        if clean_tmp:
            del(igm); del(iwm); del(ia); del(io)
            f = gfile(dtpm,'.*')
            for ff in f:
                cmd = f'rm -f {ff}'
                outvalue = subprocess.run(cmd.split(' '))

def parcelate_tissue(fPV, tissu_ind, fparc, tmap_parc=None):
    if len(fPV)==len(fparc) is False:
        error_length_input_list
    for f1, f2 in zip(fPV, fparc):
        ipv = tio.ScalarImage(f1); ia = tio.LabelMap(f2)


def pool_dill(data, scale=2, interp_up = 'nearest'):
    maxpool = torch.nn.MaxPool3d(scale)
    dill_data = maxpool(data.unsqueeze(0).unsqueeze(0).float())
    ups = torch.nn.Upsample(scale_factor=scale, mode=interp_up)
    return ups(dill_data)[0][0]

def con_comp_dill(data, scale=2, interp_up = 'nearest'):
    maxpool = torch.nn.MaxPool3d(scale)
    dill_data = maxpool(data.unsqueeze(0).unsqueeze(0).float())
    components, n_components = scipy_label(dill_data);
    components = torch.tensor(components)
    ups = torch.nn.Upsample(scale_factor=scale, mode=interp_up)
    upc_cc = ups(components.float()).int()[0][0]
    return upc_cc, n_components

def merge_PV_AssN_Aseg_cereb(finter, finter_firstcc, fAssN, fAssNcereb, fopv, fGML,fGMR,fWML,fWMR, fAseg, skip=True  ):
    df = pd.DataFrame()
    for ii, (f1,f2,fa,facereb,fo1) in enumerate(zip(finter, finter_firstcc, fAssN, fAssNcereb, fopv)):
        if os.path.isfile(fo1): #ii< df.shape[0]:
            if skip:
                print(f'skip because exist {fo1}')
                continue
        print(f'computing {fo1}')
        one_dic = {}
        igm = tio.ScalarImage([fGML[ii], fGMR[ii]]); iwm = tio.ScalarImage([fWML[ii], fWMR[ii]])
        ia = tio.ScalarImage([f1,f2,fa, fAseg[ii], facereb])
        io = tio.ScalarImage(f1)

        ma_wm_not_gm = ia.data[1]>0 ; ma_gm_not_wm = (ia.data[0]>0) & ~(ma_wm_not_gm)
        one_dic['sufWM_inter_surGM'] = ma_gm_not_wm.sum().numpy()

        GM = torch.max(igm.data[0], igm.data[1]); WM = torch.max(iwm.data[0], iwm.data[1])
        GM[ma_wm_not_gm] = 0;  WM[ma_wm_not_gm] = 1;
        GM[ma_gm_not_wm] = 1;  WM[ma_gm_not_wm] = 0;
        if 'PVup' in fo1:
            print('taking up GM')
            GM[(GM>0) & (GM<1)] = 1 ; WM[(WM>0) & (WM<1)] = 0
        elif 'PV05' in fo1:
            print('taking 05 PV to bin')
            GM[(GM >= 0.5) & (GM < 1)] = 1; WM[(WM >= 0.5) & (WM < 1)] = 1
            GM[(GM > 0) & (GM < 0.5)] = 0; WM[(WM > 0) & (WM < 0.5)] = 0
        elif 'PVdown' in fo1:
            print('taking down GM')
            GM[(GM>0) & (GM<1)] = 0;  WM[(WM>0) & (WM<1)] = 1  #erode GM/CSF and GM/WM interface no PV
        elif 'PV_down_up':
            print('taking CSF up and GM down GM')
            GM[(GM>0) & (GM<1)] = 0;  GM[(WM>0) & (WM<1)] = 1; WM[(WM>0) & (WM<1)] = 0;

        else:
            error_which_PV
        GM[WM>0] = 0;  #clean wrong GM and remove GM inside_mask
            #GM[(WM>0) & (WM<1)] = 1-WM[(WM>0) & (WM<1)]  #PV with WM

        Sa = ia.data[2];
        Saseg = ia.data[3]
        Sacereb = ia.data[4]
        # we want hyppo ammy from aseg, for better continuity with GM freesurfer (we take also ventricul because bordery with hyppo)
        #clear Ass hyppo and Amy -> assign to WM and get freesurfer aseg labels
        Sa[(Sa==10) | (Sa==12)] = 2
        Sa[(Saseg==6)] = 10 ; Sa[(Saseg==5)] = 12 ; Sa[(Saseg==4)] = 4 ; #because of special remap Saseg 4 5 6 vent hyp ammyg

        mask_add = torch.zeros_like(Sa)
        for l in label_to_add:
            l_name = label_name[l] ; l_repl = label_name_replace[l]
            mlabel = Sa==l
            if "d" in l_repl : #dilate
                mlabel_dill = pool_dill(mlabel)
                interL = (mlabel_dill > 0) & (GM > 0)
            else:
                interL = (mlabel>0) & (GM>0)
            vol_inter = interL.sum().numpy()
            if vol_inter>0:
                if 'B' in l_repl :
                    Sa[interL] = 0; mlabel[interL] = 0 #do not count
                elif 'W' in l_repl : #those tisse should be suronded by WM
                    WM[interL] = 1 ; GM[interL] = 0 ;
                elif l_repl == 'G':
                    GM[interL] = 0
                else:
                    qsdfTODO
            WM[mlabel] = 0

            one_dic[f'repl_{l_name}_i_GM'] = l_repl
            one_dic[f'sum_{l_name}_i_GM'] = vol_inter
            one_dic[f'psum_{l_name}_i_GM'] = vol_inter / mlabel.sum().numpy() * 100

            mask_add [ mlabel ] = 1

        if ((mask_add>0) & (GM>0)).sum():
            qsdf_shoul_no_happend

        SaWM = Sa==2; SaGM = Sa==1;  #RibGM = irib.data[0]>0
        Sa[SaWM | SaGM] = 0 #remove AssN GM and WM
        Sa[GM>0] = 1;     Sa[WM>0] = 2  #replace to the one from surface
        #Sa[SaWM & (Sa==0)] = 2 #BG voxel that was classified as WM are WM around Hipp and Amyg but take all (WM outside GM !)
        for nb_pass in [0]: # for amyg and hyppr replace empty voxel near border to WM (2)
            print(f'Pass {nb_pass}')
            if nb_pass==0:
                mask_data = SaWM & (Sa==0); replace_value = 2
            else:
                #mask_data = SaGM & (Sa==0)  #does not work (take all extra GM ...)
                mask_data = RibGM & (Sa == 0)  ; replace_value = 15

            upc_cc, n_components = con_comp_dill(mask_data)
            c=0
            for l in range(1,n_components+1):
                val_inter = Sa[upc_cc==l].unique()
                if 12 in val_inter:
                    Sa[ mask_data & (upc_cc==l)] = replace_value
                    #print(f'keeping c {l} {val_inter}')
                    c += (mask_data & (upc_cc==l)).sum().numpy()
                elif 10 in val_inter:
                    Sa[ mask_data & (upc_cc==l)] = replace_value
                    print(f'keeping c {l} {val_inter}')
                    c += (mask_data & (upc_cc == l)).sum().numpy()
                #else:
                #    print(f'Skiping c {l} {val_inter}')

            one_dic[f'repla_empy_i_pas{nb_pass}'] = c
            c=0

        ind_cereb_wm, ind_cereb_gm = (Sa==14), (Sa==5)
        ind_cereb_deepceres_wm, ind_cereb_deepceres_gm = (Sacereb==1), (Sacereb==2)
        Sa = Sa.int()
        Sa[ind_cereb_wm] = 13 # change in brainstem(13) / cerb(14). AssN cereb is overestimated,
        #todo check nb component of bs (we suppose  wm Assem is include in WM+GM of deepCERES expect for brainstem, ... like SaWM above ?
        Sa[ind_cereb_gm] = 0
        Sa[ind_cereb_deepceres_gm], Sa[ind_cereb_deepceres_wm] = 5, 14

        io.data[0] = Sa;    io.save(fo1)
        df = pd.concat([df, pd.DataFrame([one_dic])])
    return df

def get_cat12_labelmap(dcat_dir, fref, more_WM=0.1, more_CSF=0.1):
    fp = gfile(dcat_dir,'p1.*nii') + gfile(dcat_dir,'p2.*nii') + gfile(dcat_dir,'p3.*nii')
    img = tio.LabelMap(fp)
    tresam = tio.Resample(target=fref, label_interpolation='linear'); thoti = tio.OneHot(invert_transform=True)
    img =tresam(img)
    if more_WM>0:
        img.data[1][img.data[1]>more_WM] = 1
    if more_CSF>0:
        img.data[2][img.data[2]>more_CSF] = 0.9
    img = thoti(img)
    return img

wm_thr = torch.logspace(np.log10(0.1),np.log10(0.4),16); wm_thr[:16:4]=0; csf_thr = torch.logspace(np.log10(0.1),np.log10(0.4),16); csf_thr[1:16:4]=0
dcat_arg=torch.cat([wm_thr.unsqueeze(0),csf_thr.unsqueeze(0)],dim=0).numpy()

def merge_PV_AssN_Aseg_Cat12(finter, finter_firstcc, fAssN, fopv, fGML,fGMR,fWML,fWMR, fAseg, dcat, dcat_arg ):
    df = pd.DataFrame()
    for ii, (f1,f2,fa,fo1) in enumerate(zip(finter, finter_firstcc, fAssN, fopv)):
        if ii>10000:
            continue
        if os.path.isfile(fo1): #ii< df.shape[0]:
            print(f'skip because exist {fo1}')
            continue
        print(f'computing {fo1}')
        one_dic = {}
        igm = tio.ScalarImage([fGML[ii], fGMR[ii]]); iwm = tio.ScalarImage([fWML[ii], fWMR[ii]])
        ia = tio.ScalarImage([f1,f2,fa, fAseg[ii]])
        io = tio.ScalarImage(f1);

        ma_wm_not_gm = ia.data[1]>0 ; ma_gm_not_wm = (ia.data[0]>0) & ~(ma_wm_not_gm)
        one_dic['sufWM_inter_surGM'] = ma_gm_not_wm.sum().numpy()

        GM = torch.max(igm.data[0], igm.data[1]); WM = torch.max(iwm.data[0], iwm.data[1])
        GM[ma_wm_not_gm] = 0;  WM[ma_wm_not_gm] = 1;
        GM[ma_gm_not_wm] = 1;  WM[ma_gm_not_wm] = 0;
        if 'PVup' in fo1:
            print('taking up GM')
            GM[(GM>0) & (GM<1)] = 1 ; WM[(WM>0) & (WM<1)] = 0
        elif 'PV05' in fo1:
            print('taking 05 PV to bin')
            GM[(GM >= 0.5) & (GM < 1)] = 1; WM[(WM >= 0.5) & (WM < 1)] = 1
            GM[(GM > 0) & (GM < 0.5)] = 0; WM[(WM > 0) & (WM < 0.5)] = 0
        else:
            print('taking down GM')
            GM[(GM>0) & (GM<1)] = 0;  WM[(WM>0) & (WM<1)] = 1  #erode GM/CSF and GM/WM interface no PV

        GM[WM>0] = 0;  #clean wrong GM and remove GM inside_mask
            #GM[(WM>0) & (WM<1)] = 1-WM[(WM>0) & (WM<1)]  #PV with WM

        Sa = ia.data[2];
        Saseg = ia.data[3]
        # we want hyppo ammy from aseg, for better continuity with GM freesurfer (we take also ventricul because bordery with hyppo)
        #clear Ass hyppo and Amy -> assign to WM and get freesurfer aseg labels
        Sa[(Sa==10) | (Sa==12)] = 2
        Sa[(Saseg==6)] = 10 ; Sa[(Saseg==5)] = 12 ; Sa[(Saseg==4)] = 4 ;

        mask_add = torch.zeros_like(Sa)
        for l in label_to_add:
            l_name = label_name[l] ; l_repl = label_name_replace[l]
            mlabel = Sa==l
            if "d" in l_repl : #dilate
                mlabel_dill = pool_dill(mlabel)
                interL = (mlabel_dill > 0) & (GM > 0)
            else:
                interL = (mlabel>0) & (GM>0)
            vol_inter = interL.sum().numpy()
            if vol_inter>0:
                if 'B' in l_repl :
                    Sa[interL] = 0
                elif 'W' in l_repl : #those tisse should be suronded by WM
                    WM[interL] = 1 ; GM[interL] = 0 ;
                elif l_repl == 'G':
                    GM[interL] = 0
                else:
                    qsdfTODO
            WM[mlabel] = 0

            one_dic[f'repl_{l_name}_i_GM'] = l_repl
            one_dic[f'sum_{l_name}_i_GM'] = vol_inter
            one_dic[f'psum_{l_name}_i_GM'] = vol_inter / mlabel.sum().numpy() * 100

            mask_add [ mlabel ] = 1

        if ((mask_add>0) & (GM>0)).sum():
            qsdf_shoul_no_happend

        SaWM = Sa==2; SaGM = Sa==1;  #RibGM = irib.data[0]>0
        Sa[SaWM | SaGM] = 0 #remove AssN GM and WM
        Sa[GM>0] = 1;     Sa[WM>0] = 2  #replace to the one from surface
        #Sa[SaWM & (Sa==0)] = 2 #BG voxel that was classified as WM are WM around Hipp and Amyg but take all (WM outside GM !)

        for nb_pass in [0]: # skip here because no need with freesurfer labels
            print(f'Pass {nb_pass}')
            if nb_pass==0:
                mask_data = SaWM & (Sa==0); replace_value = 2
            else:
                #mask_data = SaGM & (Sa==0)  #does not work (take all extra GM ...)
                mask_data = RibGM & (Sa == 0)  ; replace_value = 15

            upc_cc, n_components = con_comp_dill(mask_data)
            c=0
            for l in range(1,n_components+1):
                val_inter = Sa[upc_cc==l].unique()
                if 12 in val_inter:
                    Sa[ mask_data & (upc_cc==l)] = replace_value
                    #print(f'keeping c {l} {val_inter}')
                    c += (mask_data & (upc_cc==l)).sum().numpy()
                elif 10 in val_inter:
                    Sa[ mask_data & (upc_cc==l)] = replace_value
                    print(f'keeping c {l} {val_inter}')
                    c += (mask_data & (upc_cc == l)).sum().numpy()
                #else:
                #    print(f'Skiping c {l} {val_inter}')

            one_dic[f'repla_empy_i_pas{nb_pass}'] = c
            c=0

        i_cat = get_cat12_labelmap(dcat[ii], fa, more_WM=dcat_arg[0,ii], more_CSF=dcat_arg[1,ii])
        i_cat.data[i_cat.data==0] = 5 ;i_cat.data[i_cat.data==1] = 14; i_cat.data[i_cat.data==2] = 3 ;  #label valus for cer_WM et cer_GM CSF
        ind_cereb = (Sa==5) | (Sa==14)
        Sa = Sa.int()
        Sa[ind_cereb] = i_cat.data[0][ind_cereb].int()

        io.data[0] = Sa;    io.save(fo1)
        df = pd.concat([df, pd.DataFrame([one_dic])])
    return df

def merge_PV_AssN(finter, finter_firstcc, fAssN, fopv, fGML,fGMR,fWML,fWMR ):
    df = pd.DataFrame()
    for ii, (f1,f2,fa,fo1) in enumerate(zip(finter, finter_firstcc, fAssN, fopv)):
        if ii< df.shape[0]:
            continue
        print(f'computing {fo1}')
        one_dic = {}
        igm = tio.ScalarImage([fGML[ii], fGMR[ii]]); iwm = tio.ScalarImage([fWML[ii], fWMR[ii]])
        ia = tio.ScalarImage([f1,f2,fa])
        io = tio.ScalarImage(f1);

        ma_wm_not_gm = ia.data[1]>0 ; ma_gm_not_wm = (ia.data[0]>0) & ~(ma_wm_not_gm)
        one_dic['sufWM_inter_surGM'] = ma_gm_not_wm.sum().numpy()

        GM = torch.max(igm.data[0], igm.data[1]); WM = torch.max(iwm.data[0], iwm.data[1])
        GM[ma_wm_not_gm] = 0;  WM[ma_wm_not_gm] = 1;
        GM[ma_gm_not_wm] = 1;  WM[ma_gm_not_wm] = 0;
        if 'PVup' in fo1:
            print('taking up GM')
            GM[(GM>0) & (GM<1)] = 1 ; WM[(WM>0) & (WM<1)] = 0
        else:
            print('taking down GM')
            GM[(GM>0) & (GM<1)] = 0;  WM[(WM>0) & (WM<1)] = 1  #erode GM/CSF and GM/WM interface no PV

        GM[WM>0] = 0;  #clean wrong GM and remove GM inside_mask
            #GM[(WM>0) & (WM<1)] = 1-WM[(WM>0) & (WM<1)]  #PV with WM

        Sa = ia.data[2];
        mask_add = torch.zeros_like(Sa)
        for l in label_to_add:
            l_name = label_name[l] ; l_repl = label_name_replace[l]
            mlabel = Sa==l
            if "d" in l_repl : #dilate
                mlabel_dill = pool_dill(mlabel)
                interL = (mlabel_dill > 0) & (GM > 0)
            else:
                interL = (mlabel>0) & (GM>0)
            vol_inter = interL.sum().numpy()
            if vol_inter>0:
                if 'B' in l_repl :
                    Sa[interL] = 0
                elif 'W' in l_repl : #those tisse should be suronded by WM
                    WM[interL] = 1 ; GM[interL] = 0 ;
                elif l_repl == 'G':
                    GM[interL] = 0
                else:
                    qsdfTODO
            WM[mlabel] = 0

            one_dic[f'repl_{l_name}_i_GM'] = l_repl
            one_dic[f'sum_{l_name}_i_GM'] = vol_inter
            one_dic[f'psum_{l_name}_i_GM'] = vol_inter / mlabel.sum().numpy() * 100

            mask_add [ mlabel ] = 1

        if ((mask_add>0) & (GM>0)).sum():
            qsdf_shoul_no_happend

        SaWM = Sa==2; SaGM = Sa==1;  #RibGM = irib.data[0]>0
        Sa[SaWM | SaGM] = 0 #remove AssN GM and WM
        Sa[GM>0] = 1;     Sa[WM>0] = 2  #replace to the one from surface

        #Sa[SaWM & (Sa==0)] = 2 #BG voxel that was classified as WM are WM around Hipp and Amyg but take all (WM outside GM !)

        for nb_pass in [0]: #[0,1]:
            print(f'Pass {nb_pass}')
            if nb_pass==0:
                mask_data = SaWM & (Sa==0); replace_value = 2
            else:
                #mask_data = SaGM & (Sa==0)  #does not work (take all extra GM ...)
                mask_data = RibGM & (Sa == 0)  ; replace_value = 15

            upc_cc, n_components = con_comp_dill(mask_data)
            c=0
            for l in range(1,n_components+1):
                val_inter = Sa[upc_cc==l].unique()
                if 12 in val_inter:
                    Sa[ mask_data & (upc_cc==l)] = replace_value
                    #print(f'keeping c {l} {val_inter}')
                    c += (mask_data & (upc_cc==l)).sum().numpy()
                elif 10 in val_inter:
                    Sa[ mask_data & (upc_cc==l)] = replace_value
                    print(f'keeping c {l} {val_inter}')
                    c += (mask_data & (upc_cc == l)).sum().numpy()
                #else:
                #    print(f'Skiping c {l} {val_inter}')

            one_dic[f'repla_empy_i_pas{nb_pass}'] = c
            c=0

        io.data[0] = Sa;    io.save(fo1)
        df = pd.concat([df, pd.DataFrame([one_dic])])

def merge_PV_headMida(fpv, fmida_inv, fo, fmida_all_inv=None, Reslice_smooth=True, tpad=None,skip=True):
    if len(fmida_inv)==len(fmida_inv) is False:
        error_length_input_list
    df = pd.DataFrame()
    for ii, (f1, fa, fo1) in enumerate(zip(fpv, fmida_inv, fo)):
        io = tio.ScalarImage(f1);
        if fmida_all_inv is not None:
            faall = fmida_all_inv[ii]

        if os.path.isfile(fo1): #ii< df.shape[0]:
            if skip:
                print(f'skip because exist {fo1}')
                continue
        print(f'computing {fo1}')
        i1 = tio.LabelMap(f1)
        if tpad is not None:
            io = tpad(io);
            i1 = tpad(i1)

        if Reslice_smooth:
            tmap_dic = {i: 1 for i in range(16)};
            tmap_dic.update({i + 16: i + 2 for i in range(13)})
            tmap_dic[0] = 0
            tmap = tio.RemapLabels(remapping=tmap_dic)  # remove not used tissue to make smoth 4D possible
            imid = remap_filelist([fa], tmap, fref=[i1], save=False, reslice_4D=True, blur4D=0.7)
            Sm = imid.data[0].to(torch.uint8) + 14  # because onehot on imid>15
            del(imid)
            if fmida_all_inv is not None:
                print('second mida file')
                Sm[Sm == 14] = 0
                # mida_all to get spinal cord
                tmap_dic = {i+1:0 for i in range(118)  } #mida 118 label but keep 13 and 15
                tmap_dic[13] = 1 ; tmap_dic[15] = 1 ;
                tmap = tio.RemapLabels(remapping=tmap_dic)  # remove not used tissue to make smoth 4D possible

                imid_all = remap_filelist([faall], tmap, fref=[i1], save=False, reslice_4D=True, blur4D=0.7)
                Sm_all = imid_all.data[0].to(torch.uint8)
                del(imid_all)

        else:
            imid = tio.LabelMap(fa)
            Sm = imid.data[0].to(torch.uint8)

        one_dic = {}

        Sa = i1.data[0].to(torch.uint8);
        # get mida Head
        head_mask = Sm > 15.5
        for lv, ln in zip([1, 2, 5], ['GM', 'WM', 'CerGM']):
            inter = (Sa == lv) & (head_mask)
            Sa[inter] = 0  # remove cerGM intersec with head  #
            one_dic[f'repla_{ln}_i_head'] = inter.sum()

        brain_mask = Sa > 0
        inter_s = head_mask & brain_mask
        vol_inter = inter_s.sum()
        if vol_inter > 0:
            one_dic[f'repla_brain_i_head'] = vol_inter
            print(f'remaning Head inter Brain {vol_inter}')
            print(f'valu from brain are {Sa[inter_s].unique()}')

        Sa[head_mask] = Sm[head_mask]  # .to(Sa.dtype)
        # mask_data = (Sm ==2) & (Sa == 0)
        # upc_cc, n_components = con_comp_dill(mask_data)
        # Sa[mask_data] = 2 #should be remaining brainstem

        Sa[(Sm > 0) & (Sa == 0)] = 3  # fille with csf

        if fmida_all_inv is not None:
            #add spinal-coord to WM
            Sa[(Sm_all==1) & (Sa==3)] = 13 #csf voxel changed to brainst
        else: #take label 13 (bs and spine) only if csf
            Sa[(Sm==13) & (Sa==3)] = 13


        io.data[0] = Sa;
        io.save(fo1)
        df = pd.concat([df, pd.DataFrame([one_dic])])
    return df

def merge_mida_ventricles():
    # (torch23) icm-cenir-le70 /network/iss/opendata/data/template/MIDA_v1.0/MIDA_v1_voxels/mida_all [521] L> ln -s /network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/UTE/ULTRABRAIN_001_012_CL/mida_v5/nwHSmanu_mida_new_std_v4_cor_csf.nii.gz mida_in_suj12.nii.gz
    sujnum=11
    rd = '/network/iss/opendata/data/template/MIDA_v1.0/MIDA_v1_voxels/mida_all/'
    tr = tio.Resample(target=rd+f'mida_in_suj{sujnum}.nii.gz')
    fmid = tio.LabelMap(rd + f'mida_in_suj{sujnum}.nii.gz')
    fo = rd + f'suj{sujnum}Ven_mida_in_suj{sujnum}.nii.gz'
    fass = tr(tio.LabelMap(rd + f'ass_suj{sujnum}.nii.gz'))
    #mide replace dGM vent hyp amm by WM and then Ass
    mask_change = (fmid.data==4)|(fmid.data==5)|(fmid.data==7)|(fmid.data==8)|(fmid.data==16 )|(fmid.data==17)| (fmid.data==116)|(fmid.data==6)
    fmid.data[mask_change] = 12#WM
    fmid.data[fass.data==4] = 6#wentricle
    mask_thal = (fass.data==6) & ~(fmid.data==21) #remove mida hypothalamus
    fmid.data[mask_thal] = 116#
    fmid.data[fass.data==7] = 17#Pal
    fmid.data[fass.data==11] = 16#Accu
    fmid.data[fass.data==8] = 8#Put
    fmid.data[fass.data==9] = 7#Cau
    fmid.data[fass.data==12] = 5
    fmid.data[fass.data==10] = 4#Accu
    fmid.save(fo)

    #then resample sur g1 +350 G 1H avec 50 CPU
    from utils_file import gfile, get_parent_path, addprefixtofilenames
    import torchio as tio, numpy as np
    from utils_labels import resample_and_smooth4D

    rd = '/network/iss/opendata/data/template/MIDA_v1.0/MIDA_v1_voxels/mida_all/'
    fis   = gfile(rd,'^crop_suj')
    frefs = addprefixtofilenames(fis,'r025_mrt_')
    fos = addprefixtofilenames(fis,'r025s05_')

    for fi,fref,fo in zip(fis,frefs,fos):
        resample_and_smooth4D(fi, fref, fout=fo, blur4D=0.5, skip_blur=None)
        #then  add csv and vessel
        ilo  = tio.LabelMap(fo)
        tr = tio.Resample(target=ilo, label_interpolation='bspline')
        ilin = tio.LabelMap(fi);        ilin.data = ilin.data==32 #CSF
        ilin.data = ilin.data.to(float);        ilt = tr(ilin)
        ilo.data[(ilt.data>0.5) & ( (ilo.data==10) | (ilo.data==2))] = 32 #replace only GM and cerGM

        ilin = tio.LabelMap(fi);        ilin.data = ilin.data==24 #arteries
        ilin.data = ilin.data.to(float);        ilt = tr(ilin)
        ilo.data[(ilt.data>0.5) ] = 24

        ilin = tio.LabelMap(fi);        ilin.data = ilin.data==25 #veine
        ilin.data = ilin.data.to(float);        ilt = tr(ilin)
        ilo.data[(ilt.data>0.5) ] = 25

        ilo.data[(ilo.data==0)] = 50 #set BG (0) due to nl warp, to BG mida (50)

        fo2 = addprefixtofilenames(fo,'csf_veine_')[0]
        ilo.save(fo2)
    #check if all label survives:
    df = pd.read_csv('/network/iss/opendata/data/template/remap/my_synth/mida_labels.csv', comment='#')
    dic_all = {dfs.value:dfs.Name for ii,dfs in df.iterrows()}
    i1 = tio.LabelMap(rd+'crop_suj11Ven_mida_in_suj11.nii.gz')
    i2 = tio.LabelMap(rd+'r025s05_crop_suj11Ven_mida_in_suj11.nii.gz')
    i3 = pool_remap(i2, pooling_size=3)[0]
    del(dic_vol)
    for k,v in dic_all.items():
        s1, s2, s3 = (i1.data==k).sum()*np.prod(i1.spacing), (i2.data==k).sum()*np.prod(i2.spacing), (i3.data==k).sum()*np.prod(i3.spacing)
        vol_r1 = (s2)/(s1)
        vol_r2 = (s3) / (s1)
        nrow = pd.DataFrame({'name':[v], 'vol_s1': [s1], 'vol_s2':[s2], 'vol_s3':[s3], 'vol_r1':[vol_r1], 'vol_r2':[vol_r2]})
        if 'dic_vol' in locals():
            dic_vol = pd.concat([dic_vol,nrow], ignore_index=True)
        else:
            dic_vol = nrow
        print(f'{vol_r1:0.2} \t{vol_r2:0.2} \t : {v}')
        if s2==0:
            print(f"MISSING {v}")

    #for both 11 et 12 missing 47 et 48 Vertebra C4 et C5


######################################################
### For ULTRABRAIN
suj = gdir('/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/UTE/','ULTRABRAIN')
dmid = gdir(suj,'mida_v5'); #suj = get_parent_path(dmid)[0]
sujname = [f'ULTRA_{s[15:18]}' for s in get_parent_path(suj)[1]]
f025 = gfile(suj,'^r025.*nii')
#suj = [suj[3]]+suj[5:] #suj = get_parent_path(dAssN)[0]; suj = suj[:3]+suj[4:5]
dcat = gdir(suj,'cat12')
fmida_inv = gfile(dmid,'^nwHSmanu_midaV4') #fmida_inv = gfile(droi,'^nwPVsk_midaV4_')


## For HCP
suj=gdir('/network/lustre/iss02/opendata/data/HCP/training39',['.*','T1w'])
suj=gdir('/network/iss/opendata/data/HCP/training39',['.*','T1w']) #09/2024 change to purestorage (new deepCeres)
#suj = suj[:16]#training set
sujname = get_parent_path(suj,2)[1]
f025 = ['/network/lustre/iss02/opendata/data/HCP/training39/100307/T1w/r025_T1w.nii.gz' for kk in suj]
dcat = gdir(suj,'cat12surf')

#for both
dfree = gdir(suj, ['freesurfer','suj$','mri']);dfreeS = gdir(suj, ['freesurfer','suj$','surf'])
dAssN = gdir(suj,'AssemblyNet');dAssNcereb = gdir(dAssN,'deepCERES')

droi = gdir(suj,'dmida2') #droi = gdir(suj,'^ROI$');
#fmida_inv = gfile(droi,'^nwBm_s')
fAssN =  gfile(dAssN, '^remap025b07_native_structures') ;
fAssNcereb = gfile(dAssNcereb, '^remap025b07_native_tissue') ;
fAseg = gfile(dfree,'^remap025b07')
#frib = gfile(suj,'^remapGM_ribbo')

#clean erreur due to bad surface (intersection) GM within the wm gm too close to WM
finter = gfile(dfreeS,'^r025_mask_wm_not_gm.nii.gz')
finter_firstcc = gfile(dfreeS,'largestCC_r025_mask_wm_not_gm.nii.gz')
#fGMpv = gfile(dfreeS,'maskPV_GM.nii.gz');
# fWMpv = gfile(dfreeS,'maskPV_WM.nii.gz')
fGML=gfile(dfreeS, 'r025_lh.pial.*nii'); fGMR=gfile(dfreeS, 'r025_rh.pial.*nii')
fWML=gfile(dfreeS, 'r025_lh.white.*nii'); fWMR=gfile(dfreeS, 'r025_rh.white.*nii')

#add
#BG:0,GM:1,WM:2,CSF:3,CSFv:4,cerGM:5,thal:6,Pal:7,Put:8,Cau:9,amyg:10,accuben:11,Hypp:12, WM_lower:13",,
label_to_add = range(4,14)
#dfmap = pd.read_csv('/network/lustre/iss02/opendata/data/template/remap/remap_vol2Brain_label_brainstem.csv')
label_name = ['BG','GM','WM','CSF','CSFv','cerGM','thal','Pal','Put','Cau','amyg','accuben','Hypp', 'WM_lower']
label_name_replace = ['N','N','N','N','Wd','G',   'Wd',  'Wd','Wd', 'Wd',  'G',   'Wd',     'G',     'W']
label_name_replace = ['N','N','N','N','Wd','B',   'Wd',  'Wd','Wd', 'Wd',  'G',   'Wd',     'G',     'W']
#cerGM inter GM should be BG (G, -> large cut in GM for Ultra) especially if deepCeres is take for cereb


######### PV + inside
fopv = [dd + '/r025_bin_PVup_GM_Ass_Aseg_cat.nii.gz' for dd in dfreeS];
df = merge_PV_AssN_Aseg_Cat12(finter, finter_firstcc, fAssN, fopv, fGML,fGMR,fWML,fWMR, fAseg, dcat, dcat_arg )
fopv = [dd + '/r025_bin_PV05_GM_Ass_Aseg_cat.nii.gz' for dd in dfreeS];
df = merge_PV_AssN_Aseg_Cat12(finter, finter_firstcc, fAssN, fopv, fGML,fGMR,fWML,fWMR, fAseg, dcat, dcat_arg )
wm_thr = wm_thr-wm_thr +0.5; csf_thr = csf_thr-csf_thr +0.5

fopv = [dd + '/r025_bin_PV05_GM_Ass_Aseg_cereb.nii.gz' for dd in dfreeS];
fopv = [dd + '/r025_bin_PV_down_up_GM_Ass_Aseg_cereb.nii.gz' for dd in dfreeS];
df = merge_PV_AssN_Aseg_cereb(finter, finter_firstcc, fAssN, fAssNcereb, fopv, fGML,fGMR,fWML,fWMR, fAseg, skip=False )

######### PV + Head mida
tp = tio.Pad(padding=(0, 0, 76, 0, 0, 0))  # GM/WMhas no padding

droi=dmid
fmida_inv = gfile(droi,'^nwHSmanu_midaV4')
fmida_inv_all = gfile(droi,'^nwHSmanu_mida_new_std_v4_cor_csf')
fpv = gfile(dfreeS,'^r025_bin_PV05_GM_Ass_Aseg_cereb.nii.gz')
fo = [dd + '/r025_bin_PV_head_mida_Aseg_cereb.nii.gz' for dd in droi]
df1 = merge_PV_headMida(fpv,fmida_inv,fmida_inv_all,fo,tpad=tp)

#rrrrrrrrrrrr mix exact same head with
ffpv = gfile(dfreeS,'^r025_bin_PV[du_].*_GM_Ass_Aseg_cereb.nii.gz')
droi = gdir(get_parent_path(ffpv,4)[0],'mida_v5')
fmida_inv_smoot  = gfile(droi,'^r025_bin_PV_head_mida_Aseg_cereb.nii.gz')
gm_name = [ss[9:19] for ss in get_parent_path(ffpv,1)[1] ]
uname = [f'U_{ss[-6:-3]}' for ss in get_parent_path(fmida_inv_smoot,3)[1]]
fo = [dd + f'/r025_{gm_name[ii]}_head_{uname[ii]}_midaV4s05.nii.gz' for ii,dd in enumerate(droi)] #fo = [dd + '/r025_bin_PV_head.nii.gz' for dd in droi]

df = merge_PV_headMida(ffpv, fmida_inv_smoot, fo, Reslice_smooth=False, skip=False)

fpv = gfile(dfreeS,'r025_bin_PVup_GM_Ass_Aseg_cat.nii.gz')
fo = [dd + '/r025_bin_PVup_head_mida_Aseg_cat.nii.gz' for dd in droi]
df2 = merge_PV_headMida(fpv,fmida_inv,fo,tpad=tp)

fmida_inv = gfile(droi[:4],'^nwHm_.*010') + gfile(droi[4:8],'^nwHm_.*011') + gfile(droi[8:12],'^nwHm_.*012') + gfile(droi[12:16],'^nwHm_.*013')
fpv = gfile(dfreeS,'r025_bin_PV_GM_Ass.nii.gz') #fpv = gfile(dfreeS,'r025_bin_PV_GM_Ass.nii.gz')
fo = [dd + '/r025_bin_PV_head_U.nii.gz' for dd in droi] #fo = [dd + '/r025_bin_PV_head.nii.gz' for dd in droi]
df3 = merge_PV_headMida(fpv,fmida_inv,fo,tpad=tp)
fpv = gfile(dfreeS,'r025_bin_PV05_GM_Ass_Aseg_cereb.nii.gz') #fpv = gfile(dfreeS,'r025_bin_PVup_GM_Ass.nii.gz') #fpv = gfile(dfreeS,'r025_bin_PV_GM_Ass.nii.gz')
fo = [dd + '/r025_bin_PVup_head_U.nii.gz' for dd in droi] #fo = [dd + '/r025_bin_PV_head.nii.gz' for dd in droi]
df4 = merge_PV_headMida(fpv,fmida_inv,fo,tpad=tp)
fpv = gfile(dfreeS,'r025_bin_PV_GM_Ass.nii.gz') #fpv = gfile(dfreeS,'r025_bin_PV_GM_Ass.nii.gz')

## For HCP tes-retest comput freesurfer GM from surface
suj = gdir('/network/iss/opendata/data/HCP/raw_data/test_retest/session1/',['.*','T1w'])
dfreeS = gdir(suj,['freesur','suj','surf'])
compute_GMpv_from_freesurferSurf(dfreeS)

# HCP space with ultrabrain mida ... not yet done
fpv = (gfile(dfreeS,'PV_dow.*cereb') + gfile(dfreeS,'PVdow.*cereb') +
       gfile(dfreeS,'PV05.*cereb')[:12] +  gfile(dfreeS,'PVup.*cereb')[-6:])
sujsel = get_parent_path(fpv,4)[0]; droi = gdir(sujsel,'dmida2')
fmida_inv = gfile(droi[:10],'^nwHm_.*006') + gfile(droi[10:20],'^nwHm_.*010') + gfile(droi[20:30],'^nwHm_.*011') + gfile(droi[30:40],'^nwHm_.*012') + gfile(droi[40:50],'^nwHm_.*013')

# HCP space with ultrabrain mida for  hcp16_U5_v51
fpv = gfile(dfreeS,'PV05.*cereb') #only PV05 for eval  #fpv = gfile(dfreeS,'cat.nii.gz') v51
fmida_inv = gfile(droi,'^nwHm_.*013')
sujsel = get_parent_path(fpv,4)[0]; droi = gdir(sujsel,'dmida2')

fmida_inv1 = gfile(droi[:6],'^nwHm_.*006') + gfile(droi[6:12],'^nwHm_.*010') + gfile(droi[12:18],'^nwHm_.*011') + gfile(droi[18:24],'^nwHm_.*012') + gfile(droi[24:],'^nwHm_.*013')
fmida_inv2 = gfile(droi[:6],'^nwHm_.*013') + gfile(droi[6:12],'^nwHm_.*012') + gfile(droi[12:18],'^nwHm_.*011') + gfile(droi[18:24],'^nwHm_.*010') + gfile(droi[24:],'^nwHm_.*006')
fmida_inv =[]
for f1,f2 in zip(fmida_inv1[::2],fmida_inv2[1::2]):
    fmida_inv.append(f1);fmida_inv.append(f2);

pvname = [ff[:16] for ff in get_parent_path(fpv)[1]]
dout, dname = get_parent_path(fmida_inv)
fo = [f'{dd}/{f1}_cereb_{f2}' for dd,f1,f2 in zip(dout,pvname,dname)]#addprefixtofilenames(fmida_inv,'r025_bin_Head_GM_Ass_Aseg_cat')
df4 = merge_PV_headMida(fpv,fmida_inv,fo)


df['sujname'] = sujname
df.to_csv('/network/lustre/iss02/opendata/data/HCP/stat/hcp_trainin39_mergePV_gmAss.csv')
df.to_csv('/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/UTE/stat/mergePVup_GM_Ass.csv')


## smooth GM hcp to ULTRA
dmid = gdir(suj,'dmida')
ffin = gfile(dmid,'^nwto_U')
ffo = addprefixtofilenames(ffin,'s')
for fi,fo in zip(ffin,ffo):
    ilt = resample_and_smooth4D(fi, None, blur4D=0.5)
    break

#reslice pad ULTRA (because bigger FOV)
fins = gfile(dmid,'^pad'); fout = addprefixtofilenames(fins,'r025_')
for fin, fout in zip(fins,fout):
    cmd = f'mrgrid {fin} regrid -voxel 0.25 -datatype uint8 {fout}'
    outvalue = subprocess.run(cmd.split(' '))

#from surface to PV GM WM
# cmd = sprintf('%s\n ltr%s.vtk ',cmd, fi, fi )    ;  freesurfer to vtk
fsurf = gfile(dfreeS, 'vtk')
fs_out = addprefixtofilenames(fsurf,'r025_')
fref = '/network/lustre/iss02/opendata/data/HCP/raw_data/nii/100307/T1w/r025_T1w.nii.gz'

for fin, fout in zip(fsurf,fs_out):
    cmd = f'mesh2voxel -nthreads 34 {fin} {fref} {fout}.nii.gz'
    outvalue = subprocess.run(cmd.split(' '))

#surface to volume
#remap and reslice
fAssNcereb = gfile(dAssNcereb, '^native_tissues_')
tmap = tio.RemapLabels({k:k for k in range(3)})
remap_filelist(fAssNcereb, tmap, prefix='remap025b07_', fref=f025, skip=True, reslice_4D=True, blur4D=0.5)

#for ribbon
frib = gfile(suj,'^ribbo')
tmap = tio.RemapLabels({0:0, 2:0, 41:0, 43:0, 44:0, 45:0, 42:1,  3:1})
remap_filelist(frib, tmap, prefix='remapGM_', skip=False)

fAssN = gfile(dAssN, '^native_structures_')
tmap = get_remap_from_csv('/network/lustre/iss02/opendata/data/template/remap/remap_vol2Brain_label_brainstem.csv')
remap_filelist(fAssN, tmap, prefix='remap025b07_', fref=f025, skip=True, reslice_4D=True, blur4D=0.7)
ffree = gfile(dfree,'^aparc.aseg')
tmap =get_fastsurfer_remap(ffree[0],'/network/lustre/iss02/opendata/data/template/remap/free_remapV2.csv',index_col_remap=3) #BG/GM/WM/CSF/Ve/Hipp(5)/Amy(6)
remap_filelist(ffree, tmap, prefix='remap025b07_', fref=f025, skip=True, reslice_4D=True, blur4D=0.7)


#flWM = gfile(dfreeS,'lh.white.nii'); frWM = gfile(dfreeS,'rh.white.nii'); flGM = gfile(dfreeS,'lh.pial.nii'); frGM = gfile(dfreeS,'rh.pial.nii')
flWM = gfile(dfreeS,'r025_lh.white'); frWM = gfile(dfreeS,'r025_rh.white'); flGM = gfile(dfreeS,'r025_lh.pial'); frGM = gfile(dfreeS,'r025_rh.pial')

#inter GM/WM is WM
for f1,f2,f3,f4,dirout in zip(flWM, frWM, flGM, frGM, dfreeS):
    cmd = (f'mrcalc {f1} 0 -gt {f1} 1 -lt -and {f2} 0 -gt {f2} 1 -lt -and -max {f3} 0 -gt {f3} 1 -lt -and {f4} 0 -gt {f4} 1 -lt -and -max -and {dirout}/r025_mask_wm_not_gm.nii.gz')
    outvalue = subprocess.run(cmd.split(' '))
    #cmd = (f'mrcalc {f1} 0 -gt {f1} 1 -lt -and {f2} 0 -gt {f2} 1 -lt -and -max {dirout}/maskPV_WM.nii.gz') #not used
    #cmd = (f'mrcalc {f3} 0 -gt {f3} 1 -lt -and {f4} 0 -gt {f4} 1 -lt -and -max {dirout}/maskPV_GM.nii.gz') #not used
    #cmd = (f'mrcalc {dirout}/maskPV_GM.nii.gz {dirout}/maskPV_WM.nii.gz -and {dirout}/mask_wm_not_gm.nii.gz')

#PV GM csf, is now
# mrcalc PV_GM.nii.gz 1 -lt PV_GM.nii.gz 0 -gt -and PV_WM.nii.gz 1 -lt PV_WM.nii.gz 0 -gt -and -neq qsdf.nii

finter = gfile(dfreeS,'^r025_mask_wm_not_gm.nii.gz')
fout = addprefixtofilenames(finter,'largestCC_')
for fi, fo in zip(finter, fout):
    cmd = (f'maskfilter -largest {fi} connect {fo}')
    outvalue = subprocess.run(cmd.split(' '))


fsurf = gfile(dfreeS, '(h.pial$|h.white$)')
#from surface to PV GM WM
for f1 in fsurf:
    print(f'mris_convert  --to-scanner {f1} {f1}.vtk')

fsurf = gfile(dfreeS, 'vtk')
fs_out = addprefixtofilenames(fsurf,'r025_')
fref025 = gfile(get_parent_path(fsurf,4)[0],'^r025')

for fin, fout, fref  in zip(fsurf,fs_out, fref025):
    cmd = f'mesh2voxel -nthreads 34 {fin} {fref} {fout}.nii.gz'
    print(cmd)
    outvalue = subprocess.run(cmd.split(' '))


### split 4D
ff=gfile(dmid,'to_U'); dmid,fname = get_parent_path(ff)
do = gdir(dmid,'nwto_4D')
dfre = gdir(get_parent_path(dmid)[0],['freesurfer','suj$','surf'])
ff = gfile(dfre,'r025_bin_PV_GM_Ass.nii.gz'); fname = get_parent_path(ff)[1]
for fin,dout,fn in zip(ff,do, fname):
    for i in range(14):
        cmd = f'mrcalc {fin} {i:02} -eq {dout}/label{i}_{fn}'
        outvalue = subprocess.run(cmd.split(' '))

thoti = tio.OneHot(invert_transform=True)
avgpool = torch.nn.AvgPool3d(kernel_size=2)
avgpool = torch.nn.AvgPool3d(kernel_size=3,ceil_mode=True) #meme taille que mrgrid

#cd '/home/romain.valabregue/datal/segment_RedNucleus/UTE/ULTRABRAIN_001_010_LO/mida_v4/'
io = tio.LabelMap('r05_bin_PVup_head.nii.gz')
dpo = avgpool(il.data.float())
io.data = dpo
io.save('ttt.nii.gz')
iobin = thoti(io)


########### #HCP to ULTRA skv5.1
sujHcp=gdir('/network/iss/opendata/data/HCP/training39',['.*','T1w'])
droiHcp = gdir(sujHcp,'dmida2')
fpvHcp = gfile(droiHcp,'nwto_U.*cereb')  ;
sujHcp = get_parent_path(fpvHcp,2)[0]

multi_fpv = [ [fpvHcp[i] for i in range(j,16,4)] for j in range(4)] #
multi_fpv = [ [fpvHcp[i] for i in range(j,30,6)] for j in range(6)] # 6 because 3*5 suj *2 PV PVup # skv5.1
multi_fpv = [ [fpvHcp[i] for i in range(j,60,12)] for j in range(12)] # 6 Ultra 15 HCP * 4 PV
dfa = pd.DataFrame()
fmida_inv_smoot = gfile(dmid,'^r025_bin_PV_head.*cereb')
dfa=pd.DataFrame()
uname = [f'U_{ss[-6:-3]}' for ss in get_parent_path(fmida_inv_smoot,3)[1]]
for ffpv in multi_fpv:
    hname = get_parent_path(ffpv,4)[1]
    gm_name = get_parent_path(ffpv,1)[1][0][25:31] #'PVup' if 'up' in ffpv[0] else 'PV'
    #fo = [dd + f'/r025_nwfrom_{hname[ii]}_{gm_name}_head_nwHSmanu_midaV4_s05_split_merge.nii.gz' for ii,dd in enumerate(droi)] #fo = [dd + '/r025_bin_PV_head.nii.gz' for dd in droi]
    fo = [dd + f'/r025_nwfrom_{hname[ii]}_{gm_name}_head_{uname[ii]}_midaV4s05.nii.gz' for ii,dd in enumerate(droi)] #fo = [dd + '/r025_bin_PV_head.nii.gz' for dd in droi]
    print(get_parent_path(fo)[1])
    df = merge_PV_headMida(ffpv,fmida_inv_smoot,fo, Reslice_smooth=False, skip=False)
    dfa = pd.concat([dfa, df])

fpv = gfile(dfreeS,'r025_bin_PVup_GM_Ass.nii.gz') #fpv = gfile(dfreeS,'r025_bin_PV_GM_Ass.nii.gz')
fo = [dd + '/r025_bin_PVup_head.nii.gz' for dd in droi] #fo = [dd + '/r025_bin_PV_head.nii.gz' for dd in droi]
df = merge_PV_headMida(fpv,fmida_inv,fo)

dfc = pd.read_csv('/network/iss/opendata/data/template/MIDA_v1.0/MIDA_v1_voxels/new_label_v5_hcp.csv')
tmap = tio.RemapLabels( {dd.synth:dd.synth_tissu for ii,dd in dfc.iterrows()}) #remap BS WMcereb to WM for synth
tmap_GT = tio.RemapLabels( {dd.synth:dd.target for ii,dd in dfc.iterrows()}) #
fpv = gfile(dmid,'^r025_bin_PV_head_mida_Aseg_cereb') #.r025.*PV
fref = gfile(suj,'^UTE.nii')

fin = gfile(dmid,'^label', list_flaten=False)
single_to_4D(fin, addprefixtofilenames(fpv, 'rUTE_4Dmrt_'), addprefixtofilenames(fpv, 'rUTE_binmrt_'), delete_single=True)

resample_to(fpv, fref, tmap=tmap_GT, prefix='rUTE_nearest_') #test
resample_mrt_remap_to_4DPV(fpv, fref, tmap=tmap_GT, prefix='rUTE_', skip=False, jobdir='/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/UTE/job/mrt_resample')
pool_remap_to_4DPV(fpv,pooling_size=3, ensure_multiple=6, tmap=tmap, prefix='r075' )

def resample_to(fpv, fref, tmap=None, prefix='rUTE_', skip=True, jobdir='', interp='nearest'   ):

    dic_map = tmap.remapping if tmap is not None else None
    fo = addprefixtofilenames(fpv,f'{prefix}_4D_')
    fobin = addprefixtofilenames(fpv,f'{prefix}_bin_')
    jobs = []
    for (f1, f2, f3, f4) in zip(fpv, fref, fo, fobin):
        if os.path.isfile(f4):  # ii< df.shape[0]:
            if skip:
                print(f'skip existing {f4}')
                continue
            else:
                print(f'no skip ERASING {f4}')
        print(f'computing {f4}')

        tr = tio.Resample(target=f2, image_interpolation=interp)
        tc = tio.Compose([tmap, tr]) if tmap is not None else tr
        print(f'transform {tc}')
        qsdf
        io = tc(tio.ScalarImage(f1))
        io.save(f4)

def resample_mrt_remap_to_4DPV(fpv, fref, tmap=None, prefix='rUTE_', skip=True, jobdir=''):

    dic_map = tmap.remapping if tmap is not None else None
    fo = addprefixtofilenames(fpv,f'{prefix}_4D_')
    fobin = addprefixtofilenames(fpv,f'{prefix}_bin_')
    jobs = []
    for (f1,f2,f3,f4) in zip(fpv, fref, fo,fobin):
        if os.path.isfile(f3):  # ii< df.shape[0]:
            if skip:
                print(f'skip existing {f3}')
                continue
            else:
                print(f'no skip ERASING {f3}')
        print(f'computing {f3}')
        if dic_map is None : #make identity from existing value
            print('identity mapping')
            i1 = tio.LabelMap(f1)
            dic_map = {k:k for k in range(i1.data.max() )}
        value_4D = np.unique([v for k,v in dic_map.items()])
        for val in value_4D:
            #get input value to remap in single val
            input_value= []
            for k,v in dic_map.items():
                if v==val:
                    input_value.append(k)
            out_label = f'label_{val:03}_{prefix}.nii'
            in_dir, f1_name = get_parent_path(f1)
            cmd = f'cd {in_dir};\n mrcalc {f1_name} {input_value[0]} -eq '
            for in_val in input_value[1:]:
                cmd = f'{cmd} {f1_name} {in_val} -eq -or'
            cmd = f'{cmd} - | mrgrid -force - regrid -template {f2} {out_label}'
            jobs.append(cmd)

        #better to do separately
        #cmd = f'mrcat label_*{prefix}.nii {f3}'
        #cmd = f'{cmd}\n rm -f label_*{prefix}.nii'
        #jobs.append(cmd)

    job_params = dict()
    job_params[
        'output_directory'] = jobdir
    job_params['jobs'] = jobs
    job_params['job_name'] = 'predict'
    job_params['walltime'] = '24:00:00'
    job_params['job_pack'] = 1
    job_params['cluster_queue'] = '-p norma,bigmem'
    job_params['cpus_per_task'] = 14
    job_params['mem'] = 60000
    create_jobs(job_params)

def single_to_4D(fin, fo4D, fobin, delete_single=True):
    thoti = tio.OneHot(invert_transform=True);
    for (f1, f2, f3) in zip(fin, fobin, fo4D):
        i1 = tio.LabelMap(f1)
        i1.save(f3)
        ibin = thoti(i1)
        ibin.save(f2)
        if delete_single:
            for ff in f1:
                outvalue = subprocess.run(f'rm -f {ff}'.split(' '))

def rescale_affine_corrected(affine, shape, zooms, new_shape=None):
    """copy (modify from nibable
    """
    shape = np.asarray(shape)
    new_shape = np.array(new_shape if new_shape is not None else shape)

    s = nib.affines.voxel_sizes(affine)
    rzs_out = affine[:3, :3] * zooms / s

    # Using xyz = A @ ijk, determine translation
    centroid = nib.affines.apply_affine(affine, (shape -1) / 2)
    t_out = centroid - rzs_out @ ((new_shape -1) / 2)
    return nib.affines.from_matvec(rzs_out, t_out)

def pool_remap_to_4DPV(fpv, pooling_size=2, ensure_multiple=None, tmap=None, skip=True, prefix='r075',
                       dirout=None):

    if isinstance(fpv, str):
        fpv = [fpv]
    fo = addprefixtofilenames(fpv, f'{prefix}_4D_')
    fobin = addprefixtofilenames(fpv, f'{prefix}_bin_')
    if dirout is not None:
        fo = [os.path.join(dirout, ff) for ff in get_parent_path(fo)[1]]
        fobin = [os.path.join(dirout, ff) for ff in get_parent_path(fobin)[1]]
        

    thoti = tio.OneHot(invert_transform=True); thot = tio.OneHot() #avgpool = torch.nn.AvgPool3d(kernel_size=2)
    avgpool = torch.nn.AvgPool3d(kernel_size=pooling_size,ceil_mode=True) #meme taille que mrgrid

    if ensure_multiple is None :
        ensure_multiple = pooling_size #this remove inprecision for nifti header (origin coordinate)
    tpad = tio.EnsureShapeMultiple(ensure_multiple)

    #remap_filelist(forig, tmap, prefix='target')
    for (f1,f3,f4) in zip(fpv,fo,fobin):
        if os.path.isfile(f4):  # ii< df.shape[0]:
            if skip:
                print(f'skip existing {f4}')
                continue
            else:
                print(f'no skip ERASING {f4}')
        print(f'computing {f3}')

        i11 = tpad(tmap(tio.LabelMap(f1))) if tmap is not None else tpad(tio.LabelMap(f1))

        # get the new affine
        orig_shape = np.array(i11.shape[1:])
        new_shape = orig_shape // pooling_size
        new_vox_size = nib.affines.voxel_sizes(i11.affine) * pooling_size
        io_affine = rescale_affine_corrected(i11.affine, orig_shape, new_vox_size, new_shape)

        i11['data'] = i11.data.int()
        label_values = i11.data.unique()
        nb_channel = len(label_values)
        io_data = torch.zeros([nb_channel] + list(new_shape))
        #i1 = thot(i11) to much memory ! so do it one by one
        for channel in range(nb_channel):
            Sin = i11.data[0]==label_values[channel]
            io_data[channel] = avgpool(Sin.float().unsqueeze(0))[0]
        del(Sin)

        iout = tio.LabelMap(tensor=io_data, affine=io_affine)
        iout.save(f3)
        iout = thoti(iout)
        iout.save(f4)


subdirname = 'hcp16_U5_v51' #'suj_skull_v5.1' 'suj_skull_v5'
dout = f'/network/lustre/iss02/opendata/data/template/MIDA_v1.0/for_training/{subdirname}/'
douts = [dout, f'/data/users/romain/template/MIDA/for_training/suj_skull_v5.1/{subdirname}/',
         f'/linkhome/rech/gencme01/urd29wg/work/data/template/MIDA_v1.0/for_training/{subdirname}/']
csv_namse=['suj.csv','suj_g1.csv','suj_jz.csv']
sujn = [ f'{dout}U_{dd[15:18]}_{ff}' for dd,ff in zip(get_parent_path(fo,3)[1],get_parent_path(fo)[1]) ] #HCP in U space
sujn = [ f'{dout}H_{dd}_{ff[:-3]}' for dd,ff in zip(get_parent_path(fo,4)[1],get_parent_path(fo)[1]) ] # U head in H space
#sujn = [ f'{dout}H_{dd}_{ff}' for dd,ff in zip(get_parent_path(fobin,4)[1],get_parent_path(fobin)[1]) ] # U head in H space

# 140 M but 20 M if gunzip ...  but 300 M or 1G !!! after copy arggg
for  f1,f2 in zip(fo,sujn):
    cmd = f'mrconvert {f1} {f2}' ;   outvalue = subprocess.run(cmd.split(' '))

import shutil
for f1,f2 in zip(fo,sujn):
    shutil.copyfile(f1,f2)

for fcsv,ddout in zip(csv_namse,douts):
    df = pd.DataFrame()
    #sujn = [f'{ddout}U_{dd[15:18]}_{ff}' for dd, ff in zip(get_parent_path(fo, 3)[1], get_parent_path(fo)[1])]
    sujn = [f'{ddout}H_{dd}_{ff[:-3]}' for dd, ff in zip(get_parent_path(fo, 4)[1], get_parent_path(fo)[1])]  # HCP in U space

    volname = [f'v{k+1}' for k in range(len(sujn))]
    df['sujname'] = volname; df['vol_synth'] = sujn
    df.sample(frac = 1).to_csv(f'{dout}/{fcsv}')

#Svas manual label
#dillat GM into CSF/WM
def dill_label_within(fin_list, label_to_dill, within_label, nb_iter=2, out_prefix = 'dill_'):
    for fin in fin_list:
        il = tio.LabelMap(fin)
        if not isinstance(within_label, list):
            within_label = [within_label]

        for nbiter in range(nb_iter):
            print(f"nbiter {nbiter}")
            mask = il.data==label_to_dill
            mask_dill = binary_dilation(mask.numpy()).astype(int)
            for nn, wlab in enumerate(within_label):
                if nn==0:
                    mask_within = (il.data == wlab).numpy()
                else:
                    mask_within = mask_within | (il.data == wlab).numpy()
            voxel_to_add = mask_dill * mask_within
            il.data[voxel_to_add>0] = label_to_dill

            if nbiter>0:
                out_prefix = f'{out_prefix}it{nbiter+1}_'
            fout = addprefixtofilenames(fin, out_prefix)[0]
            print(f'saving {fout}')
            il.save(fout)
def reinsert_vessel_dura():
    illow = tio.LabelMap(filow)
    il = tio.LabelMap(fi)
    tr = tio.Resample(target=fi)
    #vessel
    illow.data = (illow.data==17).to(int) #vascular brain
    ilt = tr(illow)
    inter = ((il.data==19) | (il.data==20) | (il.data==16)) & (ilt.data>0.5)  #do not eras skull and dura
    il.data[(ilt.data>0.5) & (inter==0)] = 17
    #durra matter
    illow = tio.LabelMap(filow)
    illow.data = (illow.data==16).to(int) #dura matter
    ilt = tr(illow)
    il.data[(ilt.data>0.5) & (il.data == 3) ] = 16 #only replace if csf

    fout = addprefixtofilenames(fi,'repair_ves_dura_')[0]
    il.save(fout)

#usample and smooth all labels
suj = gdir('/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/',['Svas_','synth_v2'])
fis = gfile(suj,'^Synt')
frefs = gfile(gdir(get_parent_path(fis,2)[0],'synth2' ),'^r025.*orig')
#frefs =[ '/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/Svas_04_2024_10_25_TEST_ANAT_rrr_SO/synth/r025mrt_nearest_Synth_Vas_04.nii.gz' for ii in fis]
#frefs[2]='/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/Svas_03_2024_09_26_TEST_ANAT_rrr_AR/synth/r025_mrt_nearest_SynthVas_03_00.nii.gz'
fout = addprefixtofilenames(fis,'r025_s05_')

for fi,fo,fref in zip(fis, fout, frefs):
    ilt = resample_and_smooth4D(fi, fref, blur4D=0.5, fout=fo, skip_blur=None)

#correct vessel and dura
dirsuj1 = '/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/Svas_04_2024_10_25_TEST_ANAT_rrr_SO/synth/'
fi = dirsuj1 + 'r025_s05_Synth_Vas_04_dill.nii.gz'
filow = dirsuj1 + 'Synth_Vas_04_dill.nii.gz'
dirsuj2 = '/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/Svas_03_2024_09_26_TEST_ANAT_rrr_AR/synth/'
fi = dirsuj2 + 'r025_s05_SynthVas_03.nii.gz'
filow = dirsuj2 + 'SynthVas_03.nii.gz'
reinsert_vessel_dura()

dirsuj = [dirsuj1,dirsuj2]
flab = gfile(dirsuj, '^repair_ves_dura')
dill_label_within(flab, label_to_dill=1, within_label=[3], nb_iter=2, out_prefix = 'dill_inCSF_')
dill_label_within(flab, label_to_dill=1, within_label=[2,3], nb_iter=2, out_prefix = 'dill_inWMCSF_')
dill_label_within(flab, label_to_dill=1, within_label=[2], nb_iter=2, out_prefix = 'dill_inWM_')

from scipy.ndimage import generate_binary_structure
st =  generate_binary_structure(3,2)
fin = gfile(dirsuj1,'^dill_inCSF_it2_rep')
il = tio.LabelMap(fin)
maskGM = (il.data == 1).numpy(); maskCSF = (il.data == 3).numpy()
maskWM = (il.data == 2).numpy()
mask_dill = binary_dilation(maskWM[0], iterations=4, structure=st).astype(int); mask_dill = mask_dill[np.newaxis,...]
dill_ext = mask_dill * maskGM #- mask.astype(int)
il.data[maskGM] = 3
il.data[dill_ext>0 ]= 1
fout = addprefixtofilenames(fin,'ST2testWMdilGM_')[0]
il.save(fout)

maskGM = (il.data == 1).numpy(); maskCSF = (il.data == 3).numpy()
mask_dill = binary_dilation(maskGM[0], iterations=2,  structure=st).astype(int);mask_dill = mask_dill[np.newaxis,...]
il.data[(mask_dill*maskCSF)>0 ]= 1
fout = addprefixtofilenames(fin,'ST2testdilGM2_')[0]
il.save(fout)

fout = addprefixtofilenames(fin,'ST2testdilGM2_')[0]
mask_dill = binary_dilation(mask_dill, iterations=2, structure=st).astype(int)
il.data[(mask_dill*maskCSF)>0 ]= 1

##################################################################################
#new skull ULTRA from CT reinsert brain and mida (DS 708)
suj = gdir('/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/Skull/',['.*','slicer2'])
sujn = get_parent_path(suj,2)[1]
sujultra = [ gdir('/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/UTE/',[sss,'mida_v5'])[0] for sss in sujn]
suju = get_parent_path(suj,1)[0]
fref = gfile(suju,'r025_PV05_head_')
fi = gfile(suj,'^ok_final.*.nii.gz$')
fos = addprefixtofilenames(fi,'r025s05_')
fos = addprefixtofilenames(fi,'r025_')

for ff, fr, fo in zip(fi, fref, fos):
    resample_and_smooth4D(ff, fr, blur4D=0, fout=fo, skip_blur=None)

fsynth_brain = gfile(sujultra,'^r025_PV',list_flaten =False)
fsynth_head = gfile(suj,'^r025s05_')
dfb = pd.read_csv('/network/iss/opendata/data/template/remap/my_synth/new_label_v5_hcp_DS702_DS704.csv')
dfh = pd.read_csv('/network/iss/opendata/data/template/remap/my_synth/skull_and_head_Ultra_v2_label.csv')
label_head = {dd.Name: dd.synth for   ii,dd in dfh.iterrows()}
label_brain_sel = {dd.Name: dd.synth for   ii,dd in dfb[1:15].iterrows()}
label_brain = {dd.Name: dd.synth for   ii,dd in dfb.iterrows()}

new_label_head, dic_map_brain = {}, {} #{k:v+14 for k,v in label_head.items()}
for (k,v) in label_brain.items():
    if v<14.5:
        dic_map_brain[v] = v
    else:
        dic_map_brain[v] = 0

ini, ii = 14,0
for (k,v) in label_head.items():
    if k=='BG':
        new_label_head[k] = 0
    elif k=='Brain':
        new_label_head[k] = 3 #missing voxel would then be assignated to csf !
    else:
        ii+=1
        new_label_head[k] = ii+ini
dic_map_head = {v1:v2 for (k1,v1),(k2,v2) in zip(label_head.items(), new_label_head.items())}
label_final = label_brain_sel.copy(); label_final.update(new_label_head); label_final.pop('Brain')

tmap_head = tio.RemapLabels(dic_map_head)
tmap_brain = tio.RemapLabels(dic_map_brain)

def mixt_head_brain(f_head, tmap_head, f_brain, tmap_brain, label_brain_in_head=3, fout=None ):
    if fout is not None:
        if os.path.isfile(fout):
            print(f"SKUPING {fout}")
            return
        else:
            print(f'computing {fout}')

    ihead = tmap_head(tio.LabelMap(f_head))
    ibrain = tmap_brain(tio.LabelMap(f_brain))
    ihead['data'] = ihead.data.type(torch.ByteTensor)
    ibrain['data'] = ibrain.data.type(torch.ByteTensor)

    brain_mask_in_head = (ihead.data == label_brain_in_head)
    brain_mask_in_brain = (ibrain.data >0)
    #replace only at the intersection of the 2 mask.
    #non intersecting region stay with their label (csf) for brain in head
    mask_replace = brain_mask_in_head & brain_mask_in_brain
    ihead.data[mask_replace] = ibrain.data[mask_replace]
    if fout is not None:
        print(f"saving {fout}")
        ihead.save(fout)

#apply for many brain
for kk, fi_head in enumerate(fsynth_head):
    for fi_brain in fsynth_brain[kk]:
        do = get_parent_path(fi_head)[0]
        fo = f'{do}/inVesSkCT_r025s05_{get_parent_path(fi_brain)[1]}'

        mixt_head_brain(fi_head, tmap_head, fi_brain, tmap_brain, label_brain_in_head=3, fout=fo )


#labels change from 708 to 710
#GM -> L and R GM
#ventricle ->   vent / vent34 / vent_choro
#hypophysis Mammillary and yeb_other