import torchio as tio, numpy as np
import torch, pandas as pd
from utils_file import get_parent_path, gfile, gdir, addprefixtofilenames
from utils_labels import get_mask_external_broder
from utils_labels import remap_filelist, get_fastsurfer_remap, get_remap_from_csv,resample_and_smooth4D
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage import label as scipy_label
import subprocess, os


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

## For HCP
suj=gdir('/network/lustre/iss02/opendata/data/HCP/training39',['.*','T1w'])
sujname = get_parent_path(suj,2)[1]
dfree = gdir(suj, ['freesurfer','suj$','mri'])
dfreeS = gdir(suj, ['freesurfer','suj$','surf'])
dAssN = gdir(suj,'AssemblyNet')
droi = gdir(suj,'^ROI$'); fmida_inv = gfile(droi,'^nwBm_s')
droi = gdir(suj,'dmida2')
dcat = gdir(suj,'cat12surf')
fAssN =  gfile(dAssN, '^remap025b07_native_structures') ;
fAseg = gfile(dfree,'^remap025b07')
f025 = ['/network/lustre/iss02/opendata/data/HCP/training39/100307/T1w/r025_T1w.nii.gz' for kk in fAssN]
#frib = gfile(suj,'^ribbo')

#clean erreur due to bad surface (intersection) GM within the wm gm too close to WM
finter = gfile(dfreeS,'^r025_mask_wm_not_gm.nii.gz')
finter_firstcc = gfile(dfreeS,'largestCC_r025_mask_wm_not_gm.nii.gz')
#fGMpv = gfile(dfreeS,'maskPV_GM.nii.gz');
# fWMpv = gfile(dfreeS,'maskPV_WM.nii.gz')
fopv = [dd + '/r025_bin_PVup_GM_Ass.nii.gz' for dd in dfreeS];
fGML=gfile(dfreeS, 'r025_lh.pial.*nii'); fGMR=gfile(dfreeS, 'r025_rh.pial.*nii')
fWML=gfile(dfreeS, 'r025_lh.white.*nii'); fWMR=gfile(dfreeS, 'r025_rh.white.*nii')

#add
#BG:0,GM:1,WM:2,CSF:3,CSFv:4,cerGM:5,thal:6,Pal:7,Put:8,Cau:9,amyg:10,accuben:11,Hypp:12, WM_lower:13",,
label_to_add = range(4,14)
#dfmap = pd.read_csv('/network/lustre/iss02/opendata/data/template/remap/remap_vol2Brain_label_brainstem.csv')
label_name = ['BG','GM','WM','CSF','CSFv','cerGM','thal','Pal','Put','Cau','amyg','accuben','Hypp', 'WM_lower']
label_name_replace = ['N','N','N','N','Wd','G',   'Wd',  'Wd','Wd', 'Wd',  'G',   'Wd',     'G',     'W']


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

def merge_PV_headMida(fpv, fmida_inv, fo, Reslice_smooth=True, tpad=None):
    df = pd.DataFrame()
    for ii, (f1, fa, fo1) in enumerate(zip(fpv, fmida_inv, fo)):
        io = tio.ScalarImage(f1);

        if os.path.isfile(fo1): #ii< df.shape[0]:
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
            Sm[Sm == 14] = 0
            # imid = resample_and_smooth4D(fa, f1, blur4D=0.6)
            # tresample = tio.Resample(target=f1)  # label map take nearrest
            # imid = tresample(tio.LabelMap(fa))
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
        io.data[0] = Sa;
        io.save(fo1)
        df = pd.concat([df, pd.DataFrame([one_dic])])
    return df

######### PV + inside
fopv = [dd + '/r025_bin_PVup_GM_Ass_Aseg_cat.nii.gz' for dd in dfreeS];
df = merge_PV_AssN_Aseg_Cat12(finter, finter_firstcc, fAssN, fopv, fGML,fGMR,fWML,fWMR, fAseg, dcat, dcat_arg )
fopv = [dd + '/r025_bin_PV05_GM_Ass_Aseg_cat.nii.gz' for dd in dfreeS];
df = merge_PV_AssN_Aseg_Cat12(finter, finter_firstcc, fAssN, fopv, fGML,fGMR,fWML,fWMR, fAseg, dcat, dcat_arg )
wm_thr = wm_thr-wm_thr +0.5; csf_thr = csf_thr-csf_thr +0.5


######### PV + Head mida
tp = tio.Pad(padding=(0, 0, 76, 0, 0, 0))  # GM/WMhas no padding

fmida_inv = gfile(droi,'^nwHSmanu_midaV4')
fpv = gfile(dfreeS,'r025_bin_PV_GM_Ass_Aseg_cat.nii.gz')
fo = [dd + '/r025_bin_PV_head_mida_Aseg_cat.nii.gz' for dd in droi]
df1 = merge_PV_headMida(fpv,fmida_inv,fo,tpad=tp)
fpv = gfile(dfreeS,'r025_bin_PVup_GM_Ass_Aseg_cat.nii.gz')
fo = [dd + '/r025_bin_PVup_head_mida_Aseg_cat.nii.gz' for dd in droi]
df2 = merge_PV_headMida(fpv,fmida_inv,fo,tpad=tp)

fmida_inv = gfile(droi[:4],'^nwHm_.*010') + gfile(droi[4:8],'^nwHm_.*011') + gfile(droi[8:12],'^nwHm_.*012') + gfile(droi[12:16],'^nwHm_.*013')
fpv = gfile(dfreeS,'r025_bin_PV_GM_Ass.nii.gz') #fpv = gfile(dfreeS,'r025_bin_PV_GM_Ass.nii.gz')
fo = [dd + '/r025_bin_PV_head_U.nii.gz' for dd in droi] #fo = [dd + '/r025_bin_PV_head.nii.gz' for dd in droi]
df3 = merge_PV_headMida(fpv,fmida_inv,fo,tpad=tp)
fpv = gfile(dfreeS,'r025_bin_PVup_GM_Ass.nii.gz') #fpv = gfile(dfreeS,'r025_bin_PV_GM_Ass.nii.gz')
fo = [dd + '/r025_bin_PVup_head_U.nii.gz' for dd in droi] #fo = [dd + '/r025_bin_PV_head.nii.gz' for dd in droi]
df4 = merge_PV_headMida(fpv,fmida_inv,fo,tpad=tp)

# HCP space with ultrabrain mida
fpv = gfile(dfreeS,'cat.nii.gz')
sujsel = get_parent_path(fpv,4)[0]; droi = gdir(sujsel,'dmida2')
fmida_inv1 = gfile(droi[:6],'^nwHm_.*006') + gfile(droi[6:12],'^nwHm_.*010') + gfile(droi[12:18],'^nwHm_.*011') + gfile(droi[18:24],'^nwHm_.*012') + gfile(droi[24:],'^nwHm_.*013')
fmida_inv2 = gfile(droi[:6],'^nwHm_.*013') + gfile(droi[6:12],'^nwHm_.*012') + gfile(droi[12:18],'^nwHm_.*011') + gfile(droi[18:24],'^nwHm_.*010') + gfile(droi[24:],'^nwHm_.*006')
fmida_inv =[]
for f1,f2 in zip(fmida_inv1[::2],fmida_inv2[1::2]):
    fmida_inv.append(f1);fmida_inv.append(f2);
pvname = [ff[:15] for ff in get_parent_path(fpv)[1]]
dout, dname = get_parent_path(fmida_inv)
fo = [f'{dd}/{f1}_{f2}' for dd,f1,f2 in zip(dout,pvname,dname)]#addprefixtofilenames(fmida_inv,'r025_bin_Head_GM_Ass_Aseg_cat')
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

remap_filelist(fAssN, tmap, prefix='remap025b07_', fref=f025, skip=True, reslice_4D=True, blur4D=0.7)


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

cd '/home/romain.valabregue/datal/segment_RedNucleus/UTE/ULTRABRAIN_001_010_LO/mida_v4/'
io = tio.LabelMap('r05_bin_PVup_head.nii.gz')
il =
dpo = avgpool(il.data.float())
io.data = dpo
io.save('ttt.nii.gz')
iobin = thoti(io)

######################################################
### For ULTRABRAIN
suj = gdir('/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/UTE/','ULTRABRAIN')
dmid = gdir(suj,'mida_v5'); suj = get_parent_path(dmid)[0]
#suj = suj[-4:]; #suj = get_parent_path(dAssN)[0];
dfreeS = gdir(suj, ['freesurfer','suj$','surf']); dAssN = gdir(suj,'AssemblyNet')
dfree = gdir(suj, ['freesurfer','suj$','mri'])
sujname = [f'ULTRA_{s[15:18]}' for s in get_parent_path(suj)[1]]
f025 = gfile(dmid,'^r025_pad.*nii') #
f025 = gfile(suj,'^r025.*nii')
dcat = gdir(suj,'cat12')
fAseg = gfile(dfree,'^remap025b07')

droi = gdir(suj,'mida_v5') #droi = gdir(suj,'mida_v4')
fAssN =  gfile(dAssN, '^remap025b07_native_structures') ;
fmida_inv = gfile(droi,'nwHSmanu_midaV4') #fmida_inv = gfile(droi,'^nwPVsk_midaV4_')

fref = gfile(dmid,'^head_brain_mask.nii.gz') #gfile(suj,'^UTE.nii')
fout=addprefixtofilenames(fref,'r075_')
for fi,fo in zip(fref,fout):
    cmd = f'mrgrid {fi} regrid -voxel 0.75 {fo}'
    outvalue = subprocess.run(cmd.split(' '))


########### #HCP to ULTRA
sujHcp=gdir('/network/lustre/iss02/opendata/data/HCP/training39',['.*','T1w'])
droiHcp = gdir(sujHcp,'dmida2')
fpvHcp = gfile(droiHcp,'nwto_U.*')  ; sujHcp = get_parent_path(fpvHcp,2)[0]

multi_fpv = [ [fpvHcp[i] for i in range(j,16,4)] for j in range(4)] #
multi_fpv = [ [fpvHcp[i] for i in range(j,30,6)] for j in range(6)] # 6 because 3*5 suj *2 PV PVup
dfa = pd.DataFrame()
fmida_inv_smoot = gfile(droi,'^r025_bin_PV_head')
dfa=pd.DataFrame()
for ffpv in multi_fpv:
    hname = get_parent_path(ffpv,4)[1]
    gm_name = 'PVup' if 'up' in ffpv[0] else 'PV'
    fo = [dd + f'/r025_nwfrom_{hname[ii]}_{gm_name}_head_nwHSmanu_midaV4_s05_split_merge.nii.gz' for ii,dd in enumerate(droi)] #fo = [dd + '/r025_bin_PV_head.nii.gz' for dd in droi]
    df = merge_PV_headMida(ffpv,fmida_inv_smoot,fo, Reslice_smooth=False)
    dfa = pd.concat([dfa, df])

fpv = gfile(dfreeS,'r025_bin_PVup_GM_Ass.nii.gz') #fpv = gfile(dfreeS,'r025_bin_PV_GM_Ass.nii.gz')
fo = [dd + '/r025_bin_PVup_head.nii.gz' for dd in droi] #fo = [dd + '/r025_bin_PV_head.nii.gz' for dd in droi]
df = merge_PV_headMida(fpv,fmida_inv,fo)


fin = gfile(droi,'^r025.*PV')
fref= gfile(get_parent_path(fin)[0],'^r075_head.*nii.gz') #'^r075_pad_head_mask.nii.gz')
fo = addprefixtofilenames(fin,'r075_4D_')
fobin = addprefixtofilenames(fin,'r075_bin_')

thoti = tio.OneHot(invert_transform=True); thot = tio.OneHot() #avgpool = torch.nn.AvgPool3d(kernel_size=2)
avgpool = torch.nn.AvgPool3d(kernel_size=3,ceil_mode=True) #meme taille que mrgrid

dfc = pd.read_csv('/network/lustre/iss02/opendata/data/template/MIDA_v1.0/MIDA_v1_voxels/new_label_v5_hcp.csv')
tmap = tio.RemapLabels( {dd.synth:dd.synth_tissu for ii,dd in dfc.iterrows()})
#remap_filelist(forig, tmap, prefix='target')
for (f1,f2,f3,f4) in zip(fin,fref,fo,fobin):
    if os.path.isfile(f4):  # ii< df.shape[0]:
        print(f'no skip because exist {f4}')
        #continue
    print(f'computing {f3}')
    i11 = tmap(tio.LabelMap(f1))
    i11.data = i11.data.int()
    i1 = thot(i11)
    io = tio.LabelMap(f2)
    Spool = avgpool(i1.data.float())
    io.data = Spool # no I want PV .to(torch.uint8)
    iout = tio.LabelMap(tensor=Spool, affine=io.affine)
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
