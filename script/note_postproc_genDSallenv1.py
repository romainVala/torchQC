
import torchio
import torchio as tio, numpy as np
import torch, pandas as pd, nibabel as nib, tempfile

from utils_file import get_parent_path, gfile, gdir, addprefixtofilenames, r_move_file, r_mkdir
from utils_labels import get_mask_external_broder
from utils_labels import remap_filelist, get_fastsurfer_remap, get_remap_from_csv,resample_and_smooth4D
from utils_labels import single_to_4D, pool_remap_to_4DPV, pool_remap
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure

from scipy.ndimage import label as scipy_label
import subprocess, os
from script.create_jobs import create_jobs
import commentjson as json
from segmentation.run_model import ArrayTensorJSONEncoder
from typing import Dict, List, Optional, Sequence, Tuple, Union

import csv
from script.export_nnunet import create_nnunet_dataset_from_nii, nnunet_train_job, nnunet_siam_pred_job, \
    make_validation_csv, create_SynthSeg_job, create_FS_job, make_validation_csv, create_FS_job, create_SynthSeg_job, \
    create_AssN_job, create_nnunet_dataset_from_nii, concat_nnunet_dataset, generate_DS_region,load_json

# -----------------------------------------------------------------------------
# Pool and Resample taken from Synthetic-mri-gen
# -----------------------------------------------------------------------------
def pool_remap(
    image: Union[tio.LabelMap, tio.ScalarImage],
    pooling_size: int = 2,
    ensure_multiple: Optional[int] = None,
    transform_map: Optional[tio.Transform] = None,
    keep_missing_label: bool = True,
    is_label: bool = True,
) -> Union[Tuple[tio.LabelMap, tio.LabelMap], tio.ScalarImage]:
    """
    Downsample (pool) a label map or scalar image, with optional one-hot remapping for labels.

    For label maps, returns both a pooled binary label map and a multi-channel label map.
    For scalar images, returns the pooled image.

    Parameters
    ----------
    image : tio.LabelMap or tio.ScalarImage
        The input image to downsample (label or scalar).

    pooling_size : int, optional
        The size of the pooling kernel (default: 2).

    ensure_multiple : int, optional
        If set, pad the image to ensure its shape is a multiple of this value (default: None).

    transform_map : tio.Transform, optional
        An optional transform to apply before pooling (default: None).

    keep_missing_label : bool, optional
        If True, keep all possible label channels up to the max label (default: True).

    is_label : bool, optional
        If True, treat the input as a label map; otherwise as a scalar image (default: True).

    Returns
    -------
    tuple of (tio.LabelMap, tio.LabelMap)
        If is_label is True: (binary_labels, multi_channel_labels)

    tio.ScalarImage
        If is_label is False: the pooled scalar image.

    """
    #thot_inv = OneHotTransform(invert_transform=True)
    thot_inv = tio.OneHot(invert_transform=True) #OneHotTransform(invert_transform=True)

    pool = torch.nn.AvgPool3d(kernel_size=pooling_size, ceil_mode=True)
    tpad = tio.EnsureShapeMultiple(ensure_multiple or pooling_size)

    # Apply optional pre-transform
    image = tpad(transform_map(image)) if transform_map else tpad(image)

    # Get the new affine
    original_shape = np.array(image.shape[1:])
    new_shape = original_shape // pooling_size
    new_voxel_size = nib.affines.voxel_sizes(image.affine) * pooling_size
    new_affine = rescale_affine_corrected(
        image.affine, original_shape, new_voxel_size, new_shape
    )

    if is_label:

        labels = image.data.squeeze(0).int()
        label_values = labels.unique()
        nb_channel = (
            int(label_values.max()) + 1 if keep_missing_label else len(label_values)
        )
        binary_map = torch.zeros([nb_channel, *new_shape])

        # For each label/channel, create a binary mask and pool it
        for channel in range(nb_channel):
            mask = labels == (channel if keep_missing_label else label_values[channel])
            binary_map[channel] = pool(mask.float().unsqueeze(0))[0]

        label_image = tio.LabelMap(tensor=binary_map, affine=new_affine)
        binary = thot_inv(label_image)

        return binary, label_image
    else:
        img = image.data.float().unsqueeze(0)
        down = pool(img)[0]
        return tio.ScalarImage(tensor=down, affine=new_affine)

def rescale_affine_corrected(
    affine: np.ndarray,
    shape: Sequence[int],
    zooms: Sequence[float],
    new_shape: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Compute a new affine for resampled image to preserve centering.
    """
    shape = np.array(shape)
    new_shape = np.array(new_shape)

    # Compute the original voxel spacing from the affine
    old_spacing = nib.affines.voxel_sizes(affine)
    # Compute the new scaling matrix for the affine
    scale = affine[:3, :3] * (zooms / old_spacing)

    # Compute the center of the old image in world coordinates
    center_old = nib.affines.apply_affine(affine, (shape - 1) / 2)
    # Compute the center of the new image in world coordinates (using new scale)
    center_new = scale @ ((new_shape - 1) / 2)
    # Compute the translation needed to keep the image centered
    trans = center_old - center_new

    # Return the new affine matrix with updated scale and translation
    return nib.affines.from_matvec(scale, trans)

def check_GenTmpDir(dirout, dir_regex='generate', nb_contrast=1,verbose=False):
    #suj = gdir(dirout,f'{dir_regex}_{prefix}')
    suj = gdir(dirout,f'{dir_regex}')
    #ff = gfile(dout1,'^L*gz')
    ind_0 = 0
    fOK, fKO, fNan = [], [], []
    for k,dirgen in enumerate(suj):
        num_vol = k + ind_0
        fL = gfile(dirgen, '^Lab.*gz')
        fS = gfile(dirgen, '^Sim.*gz')
        if len(fL)==0:
            fNan += [dirgen]
        else:
            if nb_contrast*len(fL) == len(fS):
                fOK += fL; #fallS += fS
            else:
                fKO += [dirgen]
                #print(f'    RRRRRRRR {get_parent_path(dirgen)[1]} : missing found {len(fKO)} Lab but {len(fS)} image ratio {len(fS)/len(fL)}')

    print(f'\t{get_parent_path(dirout)[1]} Total found Lab {len(fOK)} Img {len(fKO)} NaN {len(fNan)}')
    if verbose:
        print(fKO[:5])
        print(fNan[:5])

def regroupe_GenTmpDirMultiContrast(dirout, dir_regex='generate', prefix='', nb_con=1, move_type='copy'):
    #regroup all generated data in one folder  #just change the generation number in file name
    AnoName = get_parent_path(dirout)[1]
    suj = gdir(dirout,f'{dir_regex}')
    dirout = get_parent_path(dirout)[0]
    if len(prefix)==0:
        dout1, dout2 = dirout+f'/synth_bin' , dirout + f'/synth_4D'
    else:
        dout1, dout2 = dirout+f'/{prefix}_synth_bin' , dirout + f'/{prefix}_synth_4D'

    if not os.path.isdir(dout1): os.mkdir(dout1);
    if not os.path.isdir(dout2): os.mkdir(dout2);
    ff = gfile(dout1,'^L.*gz')
    num_vol = len(ff)

    for k,dirgen in enumerate(suj):
        fL = gfile(dirgen, '^Lab')
        for ffl in fL :
            dirname, flname = get_parent_path(ffl)

            f4D = dirname  + '/4DLab' + flname[3:]
            fHR = dirname  + '/HRLab' + flname[3:]
            fcsv = dirname  + '/CSV' + flname[3:-7] + '.csv'
            fnoles = dirname  + '/NoLes_lab' + flname[3:]
            fnolesLR = dirname  + '/LowResNoLes_lab' + flname[3:]
            fnamelab = dirname  + '/Names' + flname[3:-7] + '.csv'

            fS = gfile(dirgen, '^Sim.*gz')
            if not (len(fS) == nb_con):
                print(f'Skiping {get_parent_path(dirgen)[1]}')
                continue
            else:
                for num_contrast, ffS in enumerate(fS):
                    filesujname = f'{AnoName}_' + flname[11:24]
                    doutL = f'{dout1}/Lab_Suj_{num_vol:04}_{filesujname}.nii.gz'
                    doutS = f'{dout1}/Sim_Suj_{num_vol:04}_{filesujname}_C{num_contrast+1}.nii.gz'

                    if num_contrast==0:

                        doutL_firstContrast = doutL

                        r_move_file([ffl],[doutL],move_type)
                        r_move_file([ffS],[doutS],move_type)

                        r_move_file([f4D] ,[f'{dout2}/4D_Suj_{num_vol:04}_{filesujname}.nii.gz'],move_type)
                        r_move_file([fHR] ,[f'{dout2}/HRLab_Suj_{num_vol:04}_{filesujname}.nii.gz'],move_type)
                        r_move_file([fnoles],[f'{dout2}/NoLes_lab_Suj_{num_vol:04}_{filesujname}.nii.gz'],move_type)
                        r_move_file([fnolesLR],[f'{dout2}/LowResNoLes_lab_Suj_{num_vol:04}_{filesujname}.nii.gz'],move_type)
                        r_move_file([fcsv],[f'{dout2}/CSV_Suj_{num_vol:04}_{filesujname}.csv'],move_type)
                        if os.path.isfile(fnamelab): #not always here but not a big deal
                            r_move_file([fnamelab],[f'{dout2}/NameL_Suj_{num_vol:04}_{filesujname}.csv'],move_type)

                    else:
                        r_move_file([doutL_firstContrast],[doutL],'link')
                        r_move_file([ffS],[doutS],move_type)

                    num_vol += 1


def regroupe_GenTmpDir(dirout, dir_regex='generate', prefix=''):
    #regroup all generated data in one folder  #just change the generation number in file name
    suj = gdir(dirout,f'{dir_regex}_{prefix}')
    dout1, dout2 = dirout+f'/{prefix}_synth_bin' , dirout + f'/{prefix}_synth_4D'
    if not os.path.isdir(dout1): os.mkdir(dout1);
    if not os.path.isdir(dout2): os.mkdir(dout2);
    ff = gfile(dout1,'^L*gz')
    ind_0 = len(ff)
    for k,dirgen in enumerate(suj):
        num_vol = k + ind_0
        f = gfile(dirgen, '^[SL]')
        fname = get_parent_path(f)[1]
        fnew = [ f'{dout1}/{ff[:7]}{num_vol:03}{ff[10:]}' for ff in fname]
        r_move_file(f, fnew, type='move')
        
        f = gfile(dirgen, '^[4CHN]')
        fname = get_parent_path(f)[1]
        fnew = [ f'{dout2}/{ff[:9]}{num_vol:03}{ff[12:]}' for ff in fname]
        r_move_file(f, fnew, type='move')
def isnotNaN(num):
    return num == num

def get_csv_remaping(modelname):
    siam_data = os.environ.get('SIAM_DATA')
    if not siam_data:
        siam_data = '/data/romain/template/siam_label_template/'
        siam_data = '/network/iss/opendata/data/template/siam_label/v4.2'
        #raise RuntimeError('Environment variable SIAM_DATA is not set')

    if 'vasc' in modelname:
        labels_csv = os.path.join(siam_data, 'vascular', 'vascular_full_label_v4.csv')
    if 'mida' in modelname:
        labels_csv = os.path.join(siam_data, 'mida', 'mida_labels_v4_RU_SN.csv')
    if 'skull' in modelname:
        labels_csv = os.path.join(siam_data, 'skull', 'brain_and_skull_Ultra_V4.csv')
    if 'allen' in modelname:
        labels_csv = os.path.join(siam_data, 'AllenB', 'label_AllenB.csv')
    if 'big' in modelname:
        labels_csv = os.path.join(siam_data, 'bigB', 'label_BigBrain.csv')
    return labels_csv
def get_target_remaping(modelname, add_WM_ANO=False, add_ANO=False, label_val="targetRegV4",label_name="NametargetRegV4"):
    label_csv = get_csv_remaping(modelname)
    df = pd.read_csv(label_csv,comment='#')
    dic_map_target = {ll['synth']:ll[label_val] for ii,ll in df.iterrows() if isnotNaN(ll['synth'] ) }
    label_dic = {ll[label_name]:ll[label_val] for ii,ll in df.iterrows() if isnotNaN(ll['synth'] )}
    label_dic = {k: v for k, v in sorted(label_dic.items(), key=lambda item: item[1])}
    label_dic_all =  {ll['Name']:ll['synth'] for ii,ll in df.iterrows()  if isnotNaN(ll['synth'] )}
    if add_WM_ANO:
        max_synth,max_tar = max(dic_map_target.keys()), max(label_dic.values())
        for k in range(5):
            dic_map_target[max_synth+1+k] = max_tar + 1
        label_dic["AnoWM"] = max_tar + 1
    elif add_ANO:
        max_synth,max_tar = max(dic_map_target.keys()), max(label_dic.values())
        for k in range(5):
            dic_map_target[max_synth+1+k] = max_tar
        # already in the csv ....   label_dic["Anomalies"] = max_tar

    else:
        if 'lesion' in label_dic.keys():
            label_dic.pop('lesion',None)
        if 'Lesion' in label_dic.keys():
            label_dic.pop('Lesion',None)

    #print(dic_map_target, label_dic)
    #print(label_dic)
    return dic_map_target, label_dic, label_dic_all

def regroupe_missing_GenTmpDirMultiContrast(dirout, dir_regex='generate', prefix='', nb_con=1, move_type='copy'):
    #regroup all generated data in one folder  #just change the generation number in file name
    AnoName = get_parent_path(dirout)[1]
    suja = gdir(dirout,f'{dir_regex}')
    suj=[]
    for gg in suja:
        fmis = gfile(gg,'.*gz')
        if len(fmis)==3:
            suj.append(gg)

    dirout = get_parent_path(dirout)[0]
    if len(prefix)==0:
        dout1, dout2 = dirout+f'/synth_bin' , dirout + f'/synth_4D'
    else:
        dout1, dout2 = dirout+f'/{prefix}_synth_bin' , dirout + f'/{prefix}_synth_4D'

    if AnoName=='Ano1':
        num_vol=0
    elif AnoName=='Ano2':
        num_vol = len(gfile(dout2,'^NoLes.*_Ano1_'))*3

    flabok = gfile(dout2,f'^NoLes.*_{AnoName}_')
    #flabok.sort(key=os.path.getmtime, reverse=True)
    #fll = gfile(suj,'LowR')
    #fll.sort(key=os.path.getmtime, reverse=True)
    if len(flabok) == len(suj) :
        ok=1
    else:
        qsdf
    for k,dirgen in enumerate(suj):
        fnolesLR = gfile(dirgen, '^LowResNoLes')
        dirname, flname = get_parent_path(fnolesLR)
        filesujname = f'{AnoName}_' + flname[0][23:36] #[11:24]
        if k==101:
            print(f'NoLes_lab_Suj_{num_vol:04}_{filesujname}.nii.gz with \n   {fnolesLR[0]} ')
        r_move_file(fnolesLR,[f'{dout2}/LowResNoLes_lab_Suj_{num_vol:04}_{filesujname}.nii.gz'],move_type)
        num_vol+=3

rd = "/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/SynthGenerator/v4_2contrast"
rd = "/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/SynthGenerator/v4.2/gen_data"
dds = gdir(rd,'.*')
#dds = dds[:2]; dds.pop(0);dds.pop(0)
dsname = get_parent_path(dds)[1]
#dsname = ['mida', 'skull', 'vascular_suj2', 'allen', 'bigbrain']
dataset_name_nnunet = [ f'Dataset{751+k}_{ddn}_AnoV42' for k,ddn in enumerate(dsname) ]
dataset_name_nnunet[-1] = dataset_name_nnunet[-2] #regroup vascular2 into vascular
dnnunet_root = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/'

#
resample_NoLes_file = "now done in generate"
if resample_NoLes_file:
    job = []

    for dd,dn, dataset_name in zip(dds,dsname,dataset_name_nnunet):
        #regroupe_GenTmpDir(dd,prefix='A')
        din = gdir(dd,'A2_synth_bin')
        din4D = gdir(dd,'A2_synth_4D')
        fimg,flab = gfile(din,'^S.*gz'), gfile(din,'^L.*gz')
        flname = get_parent_path(flab)[1]
        flab = [f'{din4D[0]}/NoLes_lab{ff[3:]}' for ff in flname]
        #print(f'{dn} : in bin found {len(fimg)} img {len(flab)} lab')
        flabnew = addprefixtofilenames( gfile(din,'^L.*gz') , 'rP3_')
        nb_pool=3
        for ffi,ffo in zip(flab, flabnew):
            if os.path.isfile(ffo):
                print(f'Skiping exist {ffo}')
            else:
                job.append(f' python /network/iss/cenir/software/irm/toolbox_python/romain/torchQC/script/generate_pool.py -i {ffi} -o {ffo}')

                #label_bin, label_4d = pool_remap(tio.LabelMap(ffi), pooling_size=nb_pool, ensure_multiple=nb_pool*2)
                #label_bin.save(ffo)
                #print(f'saving {ffo}')
    from script.create_jobs import create_jobs
    job_params = dict()
    job_params[ 'output_directory'] = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/SynthGenerator/v4.2/job/pool'
    job_params['jobs'] = job
    job_params['job_name'] = 'predict'

    job_params['walltime'] = '1:00:00'
    job_params['job_pack'] = 1
    job_params['cluster_queue'] = '-p compute'
    job_params['cpus_per_task'] = 8
    job_params['mem'] = 64000
    create_jobs(job_params)

    #now check all is there



#new x contrast
dsname = get_parent_path(dds)[1]
for dd,dn  in zip(dds,dsname):
    #regroupe_GenTmpDir(dd,prefix='A')
    din = gdir(dd,'A2_synth_bin')
    if din==3:
        fimg,flab = gfile(din,'^S.*gz'), gfile(din,'^L.*gz')
        print(f'{dn} : in bin found {len(fimg)} img {len(flab)} lab')
    dgen = gdir(dd,'Ano[12]') #,'generate'])
    for dg in dgen:
        #ff = gfile(gdir(dg,'generate'),'^Lab.*gz')
        #regroupe_missing_GenTmpDirMultiContrast(dg, dir_regex='generate', prefix='A2', nb_con=3, move_type='move')

        if len(ff)>0:
            a=3
            print(f"DS:{dn} {get_parent_path(dg)[1]} in generate_A {len(ff)} regrouping ...")
            check_GenTmpDir(dg,nb_contrast=3,verbose=False)

            regroupe_GenTmpDirMultiContrast(dg, dir_regex='generate', prefix='A2', nb_con=3,move_type='move')

     
#regroup all DS
dout1 = "/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/SynthGenerator/v4_2contrast/allDS"
for dd,dn  in zip(dds,dsname):
    #regroupe_GenTmpDir(dd,prefix='A')
    din = gdir(dd,'A_synth_bin')
    fimg,flab = gfile(din,'^S.*gz'), gfile(din,'^L.*gz')
    print(f'{dn} : in bin found {len(fimg)} img {len(flab)} lab')
    dd,fname = get_parent_path(fimg)
    fnew = [ f'{dout1}/{dn[:4]}_{ff}' for ff in fname]
    r_move_file(fimg,fnew,'link')

#change Lab to noLab
#previous 1 contrast
check_new=False; DoLesion=True
for dd,dn, dataset_name in zip(dds,dsname,dataset_name_nnunet):
    #regroupe_GenTmpDir(dd,prefix='A')
    din = gdir(dd,'A2_synth_bin')
    din4D = gdir(dd,'A2_synth_4D')
    fimg = gfile(din,'^S.*gz')
    flabn = gfile(din4D,'LowRes')
    if DoLesion:
        flab = gfile(din,'^Lab')
    else:
        flab = []
        for ff in flabn:
            for i in range(3):
                flab.append(ff)

    if check_new:
        flname = get_parent_path(flab)[1]
        flab = [f'{din4D[0]}/NoLes_lab{ff[3:]}' for ff in flname]
        print(f'{dn} : in bin found {len(fimg)} img {len(flab)} lab')
        dgen = gdir(dd,['Ano','generate'])
        ff = gfile(dgen,'^L.*gz')
        if len(ff)>0:
            print(f"in generate_A {len(ff)} regrouping ...")
            regroupe_GenTmpDir(dd,prefix='A')
    dic_map_target, label_name, label_name_all = get_target_remaping(dn,add_ANO=True)

    print(f'Creating {dataset_name} from {din}')
    print(dic_map_target)
    print(label_name)

    create_nnunet_dataset_from_nii(fimg, flab, label_name, dataset_name, dnnunet_root, base_name='RRR',
                                   tmap_lab=tio.RemapLabels(dic_map_target), start_from='last') #859
    
#arrg if different number label / images
for k in range(875):
    ff= gfile(din,f'Lab_gen{k:03}')
    if len(ff)==0:
        print(f'missing label {k}')
    ff= gfile(din,f'Sim_gen{k:03}')
    if len(ff)==0:
        print(f'missing Sim {k}')

#concatenate
ds = gdir(dnnunet_root ,'Dataset75.*_AnoV42$')
ds.append(ds[0]); ds.pop(0) #do not start with allen because less labels in dataset.json

#make 1/3 one third
dfirst = gdir(dnnunet_root ,'Dataset750_AnoV42_nsD4')
dfirst = gdir(dnnunet_root ,'Dataset740_V42_nDS4')
fdsjson = gfile(dfirst, 'dataset.json')
dnnunetData = dnnunet_root +'/Dataset75003_AnoV42_nsD4_onethird/'
dnnunetData = dnnunet_root +'/Dataset770_mergeNoLes/'
img_path, label_path = dnnunetData + 'imagesTr/', dnnunetData + 'labelsTr/'
if not os.path.isdir(img_path): os.mkdir(img_path);
if not os.path.isdir(label_path): os.mkdir(label_path)
fi1 = gfile(gdir(dfirst,'image'),'RR.*gz')
fl1 = gfile(gdir(dfirst,'label'),'RR.*gz')
fimg,flab = [],[]
for ii, (f1,f2) in enumerate(zip(fi1,fl1)):
    if ii%3==0:
        fimg.append(f1); flab.append(f2)

curent_vol_idx = 0; base_name = 'RRR'
for ii, (fdata, flabel) in enumerate(zip(fimg, flab)):
    fout_img = f'{base_name}_{curent_vol_idx:04}_0000.nii.gz'  #only one input channel
    fout_lab = f'{base_name}_{curent_vol_idx:04}.nii.gz'  # no channels here

    os.symlink(fdata, img_path + fout_img)
    os.symlink(flabel, label_path + fout_lab)
    curent_vol_idx += 1


ds = gdir(dnnunet_root ,'(Dataset75003_AnoV42_nsD4_onethird|Dataset720_V4_nDS5)')

concat_nnunet_dataset(ds, 'Dataset760_AnoV42_nsD4_and_720') #,max_perDS=800)

#for noLesion
ds =  gdir(dnnunet_root ,'Dataset720')
fimg = gfile(gdir(ds,'image'),'RR.*gz')
flab = gfile(gdir(ds,'label'),'RR.*gz')

dic_map_target = {k:k for k in range(14)}; dic_map_target[13]=2
label_name={
        "background": 0,
        "GM": 1,
        "WM": 2,
        "CSF": 3,
        "Ventricles": 4,
        "Cereb": 5,
        "Thal": 6,
        "Striatum": 7,
        "RU": 8,
        "Dura": 9,
        "vascular": 10,
        "Skull": 11,
        "Head": 12    }

create_nnunet_dataset_from_nii(fimg, flab, label_name, 'Dataset770_mergeNoLes', dnnunet_root, base_name='RRR',
                               tmap_lab=tio.RemapLabels(dic_map_target), start_from='last')
#! nnUNetv2_plan_and_preprocess  -c 3d_fullres -d 750 -np 32
# nnUNetv2_plan_experiment -d 740 -pl nnUNetPlannerResEncXL
# ensuite verifier diff :  kompare nnUNetResEncUNetXLPlans.json ../Dataset715_MixSuj6/nnUNetResEncUNetXLPlans.json
plan_model = ['3d_fullres']  # , '3d_fullres']
plan_type = ['-p nnUNetResEncUNetXLPlans  -tr nnUNetTrainerNoDA']
# avant ... iidd = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Results/Dataset712_Vasc2suj_v3_Few/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/'
# '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Results/Dataset715_MixSuj6/nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres/'
iidd = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Results/Dataset720_V4_nDS5/nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres'
nnunet_train_job(760, jobdir_name='DS760', nbfold=6, nbcpu=14,
                 plan_model=plan_model, plan_type=plan_type, init_model_dir=iidd)
# attention avec cette option pas de --c du coup il ecrase tout si le job se relance !!!
# sur amper 14 cpu mais 24 sur gpu-cenir
# lancer que le premier job (array=1) et attendre le debut du training ... mias peut etre plus utile a partir 2.6.0
plan_model = ['3d_fullres']  # , '3d_fullres']
plan_type = ['-p nnUNetResEncUNetXLPlans  -tr nnUNetTrainerNoDA2000']
plan_type = ['-p nnUNetResEncUNetXLPlans  -tr nnUNetTrainerNoDA4000']
iidd = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Results/Dataset750_AnoV42_nsD4/nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres/'
iidd = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Results/Dataset740_V42_nDS4/nnUNetTrainerNoDA2000__nnUNetResEncUNetXLPlans__3d_fullres'
nnunet_train_job(740, jobdir_name='DS750_Conti4K', nbfold=5, nbcpu=14,
                 plan_model=plan_model, plan_type=plan_type, continue_from=iidd)

## subregion models
rd = "/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/SynthGenerator/v4.2/gen_data"
dsname = ['mida', 'skull', 'vascular', 'allen', 'bigbrain']
dsname = ['vascular', 'skull', 'mida','allen']
dnnunet_root = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/'
dataset_name_nnunet = [get_parent_path(gdir(dnnunet_root,f'Dataset74._{dd}_V42' ))[1][0] for dd in dsname]
dds = gdir(rd,'.*')
dataset_name_nnunet = ['Dataset744_vascular_V42',
 'Dataset743_skull_V42',
 'Dataset742_mida_V42',
 'Dataset741_allen_V42']

region_list_byDS = {
    'vascular' : [['Thal','Striatum','RU'],'GM','WM','Ventricles','Cereb','Head'],
    'skull' : ['Skull'],
    'mida' : ['Head'],
    'allen' : [['Thal','Striatum','RU'],'GM','WM','Ventricles']
}
region_list_byDS = {
    'vascular' : ['skull'],
    'allen' : ['GM', ['Thal','Striatum','RU']]
}
dsname.pop(0);dataset_name_nnunet.pop(0)
for dn, dataset_name in zip(dsname,dataset_name_nnunet):
    dd = gdir(rd,dn)
    din = gdir(dd,'A2_synth_bin')
    din4D = gdir(dd,'A2_synth_4D')

    fimg,flabL = gfile(din,'^S.*gz'), gfile(din4D,'^Low.*gz')
    flab=[];
    for ff in flabL:
        for i in range(3):
            flab.append(ff)

    dsnum = int(dataset_name[7:10])*10
    #dsnum=777; din=dn; fimg,flab=[],[] #to remove RRR
    print(f'#################   {dataset_name}  ############################')
    print(f'{dn} : in bin found {len(fimg)} img {len(flab)} lab')
    dic_map_target, label_name, label_name_all = get_target_remaping(dn)
    print(f'DS {dn} {dataset_name} ')#from {din} {label_name}')
    mmm = 1 if 'vascular' in dn else 5
    region_list = region_list_byDS[dn]# [['Thal','Striatum','RU'],'GM','WM','Ventricles','Cereb','Skull','Head']

    generate_DS_region(fimg, flab, label_name, dic_map_target, label_name_all,dsnum,
                       f'{dn}_V42', Fake=True,auto_crop=True,min_subregion=mmm,
                       region_list=region_list,dnnunet = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/regions42/')
    
    break
#2026_06_29 fix region bad order in alle Thal and GM because of target_ids in add_subregions_labels
df = pd.read_csv('/network/iss/opendata/data/template/siam_label/v4.2/AllenB/new_dic_bad_allen.csv')
dic_init = dict(zip(df.label,df.Name))
dic_bad = dict(zip(df.Name,df.label_bad_set))
#fix thalamus GM
ii=1
for  i in range(56,73): #Thalamus ii=1
ii=14
dic_new = {}
for i in range(79,132): #i in range(56,73): Thalamus ii=1
    print(f"{df.iloc[i,0]},{ii}")
    #print(f'\t"{dic_init[dic_bad[df.iloc[i,0]]]}": {ii},')
    dic_new[df.iloc[i,0]] = int(df.iloc[i,2])
    ii+=1
dic_new = {k: v for k, v in sorted(dic_new.items(), key=lambda item: item[1])}
for ii,kk in enumerate(dic_new.keys()):
    print(f'\t"{kk}": {ii+14},')


#find selected res
sel_reg=[]
for dn, dataset_name in zip(dsname,dataset_name_nnunet):
    dsnum = int(dataset_name[7:10])*10
    dsreg = gdir(dnnunet_root,[f"Dataset...._.*{dn}_V42"])
    regname =[   dd[12:(-(len(dn)+4))] for dd in get_parent_path(dsreg)[1]]
    regnum =[   int(dd[7:11]) for dd in get_parent_path(dsreg)[1]]
    #print(f'{dn} ')
    for rnum,rname in zip(regnum,regname):
        if not ('qsdf' == rname):
            sel_reg.append(rnum)
            print(f'    # DS{dn} region {rname}')
            print(f'nnUNetv2_plan_and_preprocess  -c 3d_fullres -d {rnum} -np 32')
            print(f' nnUNetv2_plan_experiment -d {rnum} -pl nnUNetPlannerResEncL -tr nnUNetTrainerNoDA')

#to run on jzay
dser = gdir('/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Preproc/region42','^Data')
dser = gdir('/linkhome/rech/gencme01/urd29wg/scratch/saved_sample_from_scratch/nnunet_region_V42','^Data')
regnum = [int(dd[7:11]) for dd in get_parent_path(dser)[1]]
dout = '/linkhome/rech/gencme01/urd29wg/training/nnunet/regionv42'
dout = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/job/regionv42All'
plan_model = ['3d_fullres']  # , '3d_fullres']
plan_type = ['-p nnUNetResEncUNetLPlans  -tr nnUNetTrainerNoDA']
dres_previous = "/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Results/region_jzayV42"
for ss in regnum:
    iidd = gdir(dres_previous,[str(ss),'nnUNetTrainerNoDA__nnUNetResEncUNetLPlans__3d_fullres'])
    if len(iidd)==1:
        print(f"DS {ss} found {get_parent_path(iidd)[1]}")
        iidd = iidd[0]
    else:
        print(f"NO init for DS {ss}")
        iidd = None

    #print(f' nnUNetv2_plan_experiment -d {ss} -pl nnUNetPlannerResEncL -tr nnUNetTrainerNoDA')
    nnunet_train_job(ss, jobdir_name=f'DS{ss}', nbfold=-1, nbcpu=14, dout=dout,# serveur='jzay',
                     plan_model=plan_model, plan_type=plan_type, init_model_dir=iidd)
fjson = gfile(dser,'split_finale')

#for i in {7240..7247}; do nnUNetv2_plan_and_preprocess  -c 3d_fullres -d $i -np 32; done
#for i in {7240..7247}; do ; done
dsreg,regname,regnameDS=[],[],[]
for dn in dsname:
    ss = gdir(dnnunet_root,["Preproc",f"Dataset...._.*{dn}_V4"])
    regname +=[   dd[12:(-(len(dn)+4))] for dd in get_parent_path(ss)[1]]
    regnameDS +=  ss#get_parent_path(ss)[1]
    dsreg += ss
regnum =[   int(dd[7:11]) for dd in get_parent_path(dsreg)[1]]
dic_regn = {nn:(pp,tt) for nn,pp,tt in zip(regnum,regname,regnameDS)}
for k,v in dic_regn.items():
    print(f"rsync -auxlv {v[1]} jzay:/linkhome/rech/gencme01/urd29wg/scratch/saved_sample_from_scratch/nnunet_region_V4/")
reg6=[ 'GM', 'Head', 'Ventricles', 'WM']
reg5=['Cereb',   'Striatum', 'Thal',]; reg4=['RU']
sall = []
for rn in reg5: #np.sort(np.unique(regname)):
    seld = [v[1] for v in dic_regn.values() if v[0]==rn]
    fp = gfile(seld,'nnUNetResEncUNetLPlans.json')
    dnn=get_parent_path(fp,2)[1]
    print(f'region {rn} found {len(fp)} conf {dnn}')
    dicj = get_js_conf(fp,dnn)
    sall += fp
for ss in sall:
    print(f'vglrun kompare {sall[0]} {ss}')

#Plans
dsreg = gdir(dnnunet_root,['Preproc','region_740','Data'])
dsreg = gdir('/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Preproc/region42','Data')

fp = gfile(dsreg,'nnUNetResEncUNetLPlans.json')
fref = gfile(get_parent_path(dsreg[0])[0],'nnUNetResEncUNetLPlans')[0]
fp_new = gfile(dsreg,'^new.*json')
fbck = addprefixtofilenames(fp,'orig_')
r_move_file(fp,fbck,'move');r_move_file(fp_new,fp,'move');
for ii,ff in enumerate(fp):
    replace_model(ff,fref)

fp = gfile(dsreg,'nnUNetResEncUNetLPlans.json')
f1 = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Preproc/Dataset7234_Thal_vascular_V4/nnUNetResEncUNetLPlans.json'
f2 = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Preproc/Dataset7245_Thal_allen_V4/nnUNetResEncUNetLPlans.json'
for f1 in fp:
    replace_model(f1,fref)
def get_js_conf(fp,regname):
    dicp = {d: load_json(ff) for d,ff in zip(regname,fp)}
    prname, prarch = ['batch_size','patch_size'],['n_stages','features_per_stage']
    for k, v in dicp.items():
        l3 = v['configurations']['3d_fullres']
        larch = l3['architecture']['arch_kwargs']
        print( [f'{nn} : {l3[nn]}' for nn in prname ] + [f'{nn} : {larch[nn]}' for nn in prarch ] )
    return dicp
def save_js_conf(data,fo):
    with open(fo, 'w') as f:
        json.dump(data, f,indent=4)
def replace_model(fj,fjref,fjo=None):
    if fjo is None:
        fjo = addprefixtofilenames(fj,'new_')[0]
    dic = get_js_conf([fj,fjref],['move','ref'])
    dic['move']['configurations']['3d_fullres']['batch_size'] = dic['ref']['configurations']['3d_fullres']['batch_size']
    dic['move']['configurations']['3d_fullres']['patch_size'] = dic['ref']['configurations']['3d_fullres']['patch_size']
    dic['move']['configurations']['3d_fullres']['architecture'] = dic['ref']['configurations']['3d_fullres']['architecture']
    save_js_conf(dic['move'],fjo)

#Dataset7212_Head_mida_V4   32 G


#2026_07 building no augmentation training set
din='/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/SynthGenerator/v4.2/gen_data_no_augmentation'
fimg = gfile('/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/SynthGenerator/v4.2/gen_data_no_augmentation/Sim_img','.*gz')
sujn = [ss.split('_')[1] for ss in get_parent_path(fimg)[1] ]
flab=[]
for ss in sujn:
    flab.append(gfile(din,f'{ss}.*target')[0])
label_name={
        "background": 0,
        "GM": 1,
        "WM": 2,
        "CSF": 3,
        "Ventricles": 4,
        "Cereb": 5,
        "Thal": 6,
        "Striatum": 7,
        "RU": 8,
        "Dura": 9,
        "vascular": 10,
        "Skull": 11,
        "Head": 12    } #arg c'est pas les bon target
#je refais le remap
dsname = ['vasc', 'skull', 'mida']
for dn in dsname:
    dic_map_target, label_name, label_name_all = get_target_remaping(dn, label_val=    "synth_targetRegionFew" ,label_name="NameTargetRegionFew")
    fin = gfile(din,f'{dn}.*all_region.nii.gz')
    fo =[ff.replace('all_region','target') for ff in fin]
    print(label_name)
    print(f'{dn} with {len(fin)}')
    for ff,ffo in zip(fin,fo) :
        tl = tio.RemapLabels(dic_map_target)(tio.LabelMap(ff))
        tl.save(ffo)

    break

dic_map_target={k:k for k in range(17)}

create_nnunet_dataset_from_nii(fimg, flab, label_name, 'Dataset610_3DS_noAugment', dnnunet_root, base_name='RRR',
                               tmap_lab=None, start_from='last')

iidd = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Results/Dataset715_MixSuj6/nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres/'
plan_model = ['3d_fullres']  # , '3d_fullres']
plan_type = ['-p nnUNetResEncUNetXLPlans  ']
nnunet_train_job(610, jobdir_name='DS610', nbfold=5, nbcpu=14,
                 plan_model=plan_model, plan_type=plan_type, init_model_dir=iidd)
#nnUNetv2_plan_and_preprocess  -c 3d_fullres -d 610 -np 32
#nnUNetv2_plan_experiment -d 610 -pl nnUNetPlannerResEncXL
