import torch,numpy as np,  torchio as tio
import json, os, seaborn as sns, tarfile, subprocess, pathlib, shutil
from utils_file import gfile, gdir, get_parent_path, addprefixtofilenames, remove_extension, r_move_file
from utils_labels import remap_filelist
import pandas as pd
from nibabel.viewers import OrthoSlicer3D as ov
import matplotlib.pyplot as plt
from script.create_jobs import create_jobs
from utils_labels import remap_filelist, get_fastsurfer_remap

def main_createDS_old():
    din = '/data/romain/PVsynth/saved_sample/Uhcp4_skv5.1/tio_save_mot_nii/'
    din = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/Uhcp4_skv5.2/tio_save_mot_nii_noSpMean/'
    din = 'c5.1/tio_save_mot_nii_noSpMean'

    dsuj = gdir(din, 'ep_')
    fsuj = gfile(dsuj, '^suj.*gz')
    fcsv = gfile(dsuj, '^suj.*csv')
    fin, flab = gfile(dsuj, '^t1'), gfile(dsuj, '^lab')

    dflab = pd.read_csv(
        '/network/iss/opendata/data/template/MIDA_v1.0/for_training/suj_skull_v5.1/new_label_v5_hcp.csv')

    label_dic.update({df[1]['NameTarget']: df[1]['target'] for df in dflab[1:13].iterrows()})
    label_dic['eyes'], label_dic['skull'], label_dic['head'] = 13, 14, 15  # remap in training
    thoti = tio.OneHot(invert_transform=True, copy=False)  # sinon pbr de path de jzay

def main_init_nnUNET_tree():
    dnnunet = '/data/romain/PVsynth/saved_sample/nnunet/'
    dnnunet = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/'

    dataset_name = 'Dataset702_Uhcp4_skv51_mot_noSpMean1000_stdTh045'  # 'Dataset701_Uhcp4_skv5.1_tio_save_mot_nii/',
    dataset_name = 'Dataset704_Uhcp_skv52_mot_elaBig'  # 'Dataset701_Uhcp4_skv5.1_tio_save_mot_nii/',
    dnnunetData = dnnunet + dataset_name + '/'
    dnnunetPreproc, dnnunetRes = dnnunet + 'Preproc/', dnnunet + 'Results/'
    if not os.path.isdir(dnnunetData): os.mkdir(dnnunetData);
    if not os.path.isdir(dnnunetPreproc): os.mkdir(dnnunetPreproc);
    if not os.path.isdir(dnnunetRes): os.mkdir(dnnunetRes);

    with open(dnnunet + '/nnunet_path.sh', 'w') as file:
        file.write(f'export nnUNet_raw="{dnnunet}"\n')
        file.write(f'export nnUNet_preprocessed="{dnnunetPreproc}"\n')
        file.write(f'export nnUNet_results="{dnnunetRes}"\n')

def generate_DS_region_main():
    din = ['/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/Vascular3/synth_bin']
    din = ['/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/Mida_all/synth_bin']
    fin, flab = gfile(din, 'Sim.*gz'), gfile(din, '^Lab.*gz')
    #for DS712
    df = pd.read_csv('/network/iss/opendata/data/template/remap/my_synth/brain_and_skull_and_Ass_vascular_v3_label_DS709.csv')
    label_dic = {ll['NameTargetRegionFew']: ll['synth_targetRegionFew'] for ii, ll in df.iterrows()}
    label_dic = {k: v for k, v in sorted(label_dic.items(), key=lambda item: item[1])}
    dic_map = {ll['synth_target']: ll['synth_targetRegionFew'] for ii, ll in df.iterrows()}
    label_dic_all = {ll['NameTarget']: ll['synth_target'] for ii, ll in df.iterrows()}
    DS_region_num, DS_root_name = 7120, 'Vasc2suj_v3_Few'

    df = pd.read_csv('/network/iss/opendata/data/template/remap/my_synth/mida_labels.csv',comment='#')
    label_dic = {ll['NameTargetRegionFew']: ll['synth_targetRegionFew'] for ii, ll in df.iterrows()}
    label_dic = {k: v for k, v in sorted(label_dic.items(), key=lambda item: item[1])}
    dic_map = {ll['value']: ll['synth_targetRegionFew'] for ii, ll in df.iterrows()}
    label_dic_all = {ll['Name']: ll['value'] for ii, ll in df.iterrows()}
    DS_region_num, DS_root_name = 7140, 'MidaSuj3_RFew'

    generate_DS_region(fin, flab, label_dic, dic_map, label_dic_all,DS_region_num, DS_root_name)

def generate_DS_region(fin, flab, label_dic, dic_map, label_dic_all,DS_region_num, DS_root_name,
        dnnunet = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/'):

    tmap_label = tio.RemapLabels(dic_map)
    for k,v in label_dic.items() :
        dic_region={}
        for kk,vv in label_dic_all.items():
            if dic_map[vv]==v:
                dic_region.update({kk:vv})
        #print(f'"""""{k}""""""""""')
        nb_subregion = len(dic_region)
        if nb_subregion>1:
            print(f'{k} has {nb_subregion} region')
            label_dic_region_LR, dic_map_regionLR = {'background': 0}, {}
            for kk,vv in dic_region.items():
                if (kk.startswith('R_') | kk.startswith('L_')):
                    new_name = kk[2:]
                else:
                    new_name = kk
                dic_map_regionLR.update({vv:new_name})
                label_dic_region_LR.update({new_name:vv})

            for ii, kk in enumerate(label_dic_region_LR.keys()):
                label_dic_region_LR[kk] = ii

            new_dic_map = dic_map.copy()
            for kk,vv in new_dic_map.items():
                if kk in dic_map_regionLR:
                    new_dic_map[kk] = label_dic_region_LR[dic_map_regionLR[kk]]
                else:
                    new_dic_map[kk] = 0

            print(f"after LR {len(label_dic_region_LR)} regions")
            print(label_dic_region_LR)
            #print(new_dic_map)
            tmap_label = tio.RemapLabels(new_dic_map)
            dataset_name = f'Dataset{DS_region_num}_{k}_{DS_root_name}'
            print(f"Creation {dataset_name}")
            create_nnunet_dataset_from_nii(fin, flab, label_dic_region_LR, dataset_name=dataset_name,
                                           dnnunet_root=dnnunet, tmap_lab=tmap_label,region_mask=True)
            DS_region_num+=1


def main_all():
    dnnunet = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/'

    din = ['/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/Vascular3/synth_bin']
    fin, flab = gfile(din, 'Sim.*gz'), gfile(din, '^Lab.*gz')
    df = pd.read_csv('/network/iss/opendata/data/template/remap/my_synth/brain_and_skull_and_Ass_vascular_v3_label_DS709.csv')
    label_dic = {ll['NameTargetRegionFew']: ll['synth_targetRegionFew'] for ii, ll in df.iterrows()}
    label_dic = {k: v for k, v in sorted(label_dic.items(), key=lambda item: item[1])}
    tmap_label = tio.RemapLabels({ll['synth_target']: ll['synth_targetRegionFew'] for ii, ll in df.iterrows()})
    #label_dic = {'background': 0}
    create_nnunet_dataset_from_nii(fin, flab, label_dic, dataset_name='Dataset714_Vasc2suj_v3_T12',
                                   dnnunet_root=dnnunet, tmap_lab = tmap_label)

    label_dic_all = {ll['NameTarget12']: ll['synth_target12'] for ii, ll in df.iterrows()}
    label_dic = {k: v for k, v in sorted(label_dic.items(), key=lambda item: item[1])}

    # su 2840
    # nnUNetv2_plan_and_preprocess -np 24 -d 706 --verify_dataset_integrity
    # nnUNetv2_plan_experiment -d 706 -gpu_memory_target 80 -overwrite_plans_name nnUNetPlans_80G
    # nnUNetv2_plan_experiment -d 706 -gpu_memory_target 70 -overwrite_plans_name nnUNetPlans_70G
    # nnUNetv2_plan_experiment -d 706 -pl nnUNetPlannerResEncXL

    # nnUNetv2_plan_and_preprocess  -c 3d_fullres -d 707 -gpu_memory_target 40 -overwrite_plans_name  nnUNetPlans_40G
    # -c to run only 3d_fullres

    dtest='/network/isscenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/'
    fcsv_list = gfile('/network/iss/opendata/data/template/validataion_set/' ,'csv$')
    create_nnunet_testset_from_csv(fcsv_list, dtest)

    nnunet_train_job(709, jobdir_name='trainVasv3_ResEnc', nbfold=5, nbcpu=14)
    plan_model = ['3d_fullres']  # , '3d_fullres']
    plan_type = ['-p nnUNetResEncUNetXLPlans  -tr nnUNetTrainerNoDA' ]  # -tr nnUNetTrainer_onlyMirror01 ' ] #,-p nnUNetPlannerResEncXL '-p nnUNetPlans']
    iidd='/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Results/Dataset712_Vasc2suj_v3_Few/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/'
    nnunet_train_job(715, jobdir_name='job_SujMix6_DS715', nbfold=5, nbcpu=14,
                     plan_model=plan_model, plan_type=plan_type,init_model_dir=iidd)

    indirs = gdir('/data/romain/PVsynth/saved_sample/nnunet/testing/', '.*UL')
    nnunet_make_predict_job(indirs)
    indirs = gdir(
        '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set',
        ['ULTRA_all', 'vol'])
    plan_model = ['3d_fullres', '3d_cascade_fullres', '3d_lowres', '3d_fullres', '3d_fullres', '3d_fullres']
    plan_train = ['nnUNetTrainer', 'nnUNetTrainer', 'nnUNetTrainer', 'nnUNetTrainerNoDA',
                  'nnUNetTrainerNoDeepSupervision', 'nnUNetTrainer']
    plan_type = ['nnUNetPlans', 'nnUNetPlans', 'nnUNetPlans', 'nnUNetPlans', 'nnUNetPlans', 'nnUNetResEncUNetXLPlans']
    plan_model = ['3d_fullres']  # ['3d_fullres', '3d_fullres']
    plan_train = ['nnUNetTrainer']  # ['nnUNetTrainer', 'nnUNetTrainer']
    plan_type = ['nnUNetResEncUNetXLPlans']  # ['nnUNetPlans', 'nnUNetResEncUNetXLPlans']
    nnunet_make_predict_job(indirs, plan_model, plan_train, plan_type,
                            jobdir='/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/nnunet_pred/job')
    plan_model = ['3d_fullres'];
    plan_train = ['nnUNetTrainer'];
    plan_type = ['nnUNetPlannerResEncXL_80G']  # ['nnUNetPlans_80G']
    jobdir = '/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/nnunet_pred/job/prednn'  # '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/job_pred/nnunet_Ultra'
    nnunet_make_predict_job(indirs, plan_model, plan_train, plan_type, datasetID=708, jobdir=jobdir, cmd_option='')

    model_num_list=[3, 122, 13,14,15 ]
    nnunet_siam_pred_job(indirs,model_num_list)

### CREATE from tio suj  and  filter csv augmentation
def create_nnunet_dataset_from_tio(fsuj, fcsv, base_name = 'RRR'):
    std_noise = np.array([json.loads(pd.read_csv(ff).T_Noise.values[0])['std']['t1'] for ff in fcsv])
    nb_suj=1000
    std_thr = 100 #np.percentile(std_noise, nb_suj/len(std_noise)*100)
    ind_suj = 0
    for ii, filename_tar in enumerate(fsuj):
        fout_img = img_path + f'{base_name}_{ind_suj:04}_0000.nii.gz'  #only one input channel
        fout_lab = label_path + f'{base_name}_{ind_suj:04}.nii.gz'  #no channels here
        if std_noise[ii] > std_thr:
            print(f'skip std {std_noise[ii]} {filename_tar}')
            continue
        if ( os.path.isfile((fout_lab)) ) & (  os.path.isfile((fout_img)) ):
            print(f'out {ind_suj} exist')
            ind_suj += 1
            continue

        filename_torch = get_parent_path(filename_tar)[1][:-7]
        print(f"from {filename_torch} to {fout_img} ")
        with tarfile.open(filename_tar, "r") as tar:
            suj = torch.load(tar.extractfile(filename_torch))
        sujt = thoti(suj)
        sujt.label.data = sujt.label.data.to(torch.uint8)
        sujt.t1.data = (sujt.t1.data*1000).to(torch.int16)
        sujt.t1.save(fout_img); sujt.label.save(fout_lab)
        ind_suj += 1
        if ind_suj>1000:
            break

### CREATE from nifti file (link image, OneHotInv for labels
def create_nnunet_dataset_from_nii(fimg, flab,label_dic,dataset_name, dnnunet_root,
                                   base_name = 'RRR',lab_one_hot=False, tmap_lab = None, region_mask=False,
                                   start_from=0):
    dnnunetData = os.path.join(dnnunet_root, dataset_name) + '/'
    if not os.path.isdir(dnnunetData): os.mkdir(dnnunetData)
    img_path, label_path = dnnunetData + 'imagesTr/', dnnunetData + 'labelsTr/'
    if not os.path.isdir(img_path): os.mkdir(img_path);
    if not os.path.isdir(label_path): os.mkdir(label_path)

    for ii, (fdata, flabel) in enumerate(zip(fimg, flab)):
        fout_name = f'{base_name}_{(ii+start_from):04}_0000.nii.gz'  #only one input channel
        if os.path.isfile((img_path+fout_name)): #skip if exist
            continue
        if not os.path.isfile((img_path+fout_name)):
            os.symlink(fdata, img_path + fout_name)

        fout_name = f'{base_name}_{(ii+start_from):04}.nii.gz'  #no channels here
        if lab_one_hot:
            #must transform 4d label to 3D os.symlink(flabel, label_path + fout_name)
            if not os.path.isfile((label_path+fout_name)):
                il = tio.LabelMap(flabel)
                il = thoti(il)
                il.save(label_path+fout_name)
        elif tmap_lab is not None:
            il = tmap_lab(tio.LabelMap(flabel))
            il.save(label_path+fout_name)
            if region_mask:
                fout_mask_name = f'{base_name}_{(ii+start_from):04}_0001.nii.gz'
                il['data'][il.data>0] = 1
                il.save(img_path + fout_mask_name)
        else:
            os.symlink(flabel, label_path + fout_name)


        print((ii+start_from))
    nb_suj = len(fimg)+start_from
    if region_mask:
        json_dic = {"channel_names": {"0": "T1","1":"mask"}, 'labels': label_dic, "numTraining": nb_suj, "file_ending": ".nii.gz"}
    else:
        json_dic = {"channel_names": {"0":"T1"}, 'labels':label_dic, "numTraining": nb_suj, "file_ending": ".nii.gz"}
    with open(dnnunetData+'/dataset.json', 'w') as file:
        json.dump(json_dic, file, indent=4, sort_keys=False)

### CREATE test set from file path and file_name list
def create_nnunet_testset_from_file_list(file_list, name_list, dout, datasetName):
    df = pd.DataFrame()
    df['vol_in'] = file_list
    df['sujname'] = name_list
    return create_nnunet_testset_from_csv([df], dout, [datasetName])

### CREATE Inferenc form validataion csv
def create_nnunet_testset_from_csv(fcsv_list, dout, ds_name_list=None):

    if ds_name_list is None:
        ds_name_list = remove_extension(get_parent_path(fcsv_list)[1])
    fcsv_out_list = []
    for fcsv, ds_name in zip(fcsv_list,ds_name_list):
        df = pd.read_csv(fcsv) if isinstance(fcsv, str) else fcsv; # df = df[:10]
        keys_img_list = []
        for kk in df.keys():
            if kk.startswith('vol'):
                keys_img_list.append(kk)
        keys_lab_list = []
        for kk in keys_img_list:
            print(f'[{kk}]')

        for keys_img in keys_img_list:
            dataset_dir = dout + '/' + ds_name + '/' + keys_img
            if os.path.isdir(dataset_dir):
                print(f'Skip existing dir {dataset_dir}')
                continue
            else:
                pathlib.Path(dataset_dir).mkdir(parents=True)
            dfnew = pd.DataFrame()
            for ind_suj, dfser_mmm in enumerate(df.iterrows()):
                dfser = dfser_mmm[1]
                fin = dfser[keys_img]
                if fin[-3:]=='nii': #one need to zip
                    if os.path.isfile(fin+'.gz'): #already done !
                        fin += '.gz'
                    else:
                        cmd = f'gzip {fin}';
                        outvalue = subprocess.run(cmd.split(' '))
                        fin += '.gz'

                sujname_orig = dfser['sujname']
                fname = remove_extension( get_parent_path(fin)[1] )
                #new_sujname = f'{fname}_{ind_suj:04}'
                new_sujname = f'{fname}_{sujname_orig}'
                fout = f'{dataset_dir}/{new_sujname}_0000.nii.gz'
                #print(f"{fout} from {fin}")
                if not os.path.isfile(fout):
                    os.symlink(fin, fout)
                one_dict = {'sujname': [f'{new_sujname}'], 'vol_path':[fout], 'vol_path_orig':[fin],'sujname_orig':[sujname_orig]}
                dfdic = pd.DataFrame.from_dict(one_dict)
                dfnew = pd.concat([dfnew, dfdic])

            for kk in df.keys():
                if (kk.startswith('lab')) | (kk.startswith('mask')):
                    dfnew[kk] = df[kk].values
            fcsv_outname = dout+'/'+ ds_name + '_' + keys_img + '.csv'
            fcsv_out_list.append(fcsv_outname)
            dfnew.to_csv(fcsv_outname, index=False)
    return fcsv_out_list

def nnunet_train_job(datasetNum, jobdir_name = 'trainVasReg',nbfold = 3,nbcpu = 14, nbcpreproccpu=12,    plan_model = ['3d_fullres'],
                     plan_type = [ '-p nnUNetPlans'], init_model_dir=None, dout=None, serveur='ICM'):

    jobs=[];  #40 for V100  24 for A100 ?
    #export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cmd_ini = f'export nnUNet_n_proc_DA={nbcpreproccpu}; nnUNetv2_train  {datasetNum} ' #--c does not work if model
    #cmd_ini = f'nnUNetv2_train {datasetNum} '
    for pm, pt in zip(plan_model, plan_type):
        for fold in range(nbfold):
            if init_model_dir is not None:
                dir_mod = gdir(init_model_dir,f'fold_{fold}')
                fmod = gfile(dir_mod,'checkpoint_final.pth')
                jobs.append(f'{cmd_ini} {pm} {fold} {pt} -pretrained_weights {fmod[0]}')

            else:
                jobs.append(f'{cmd_ini} {pm} {fold} {pt} ')
    if dout is None:
        dout = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/job'
        #dout = '/linkhome/rech/gencme01/urd29wg/training/nnunet/mida_RFew'
    from script.create_jobs import create_jobs
    job_params = dict()
    job_params['output_directory'] = dout + '/' + jobdir_name
    job_params['jobs'] = jobs
    job_params['job_name'] = 'nnUNet'
    if serveur=='ICM':
        job_params['cluster_queue'] = '-p gpu-ampere,gpu-cenir --gres=gpu:1 --mem 200G'
        job_params['walltime'] = '200:00:00'

    else:
        job_params['cluster_queue'] = '-C v100-32g --qos=qos_gpu-t4 --account=ezy@v100 --gres=gpu:1'
        #job_params['cluster_queue'] = '-C a100 --account=ezy@a100  --qos=qos_gpu_a100-t3 --gres=gpu:1'
        job_params['walltime'] = '20:00:00'
        #job_params['cluster_queue'] = '-p gpu-ampere,gpu-cenir --gres=gpu:1 --mem 200G'
        job_params['walltime'] = '100:00:00'

    job_params['cpus_per_task'] = nbcpu
    #job_params['mem'] = 32000
    job_params['job_pack'] = 1

    create_jobs(job_params)


def nnunet_make_predict_job(indirs,plan_model,plan_train,plan_type,
                            nbcpu=4, cmd_option=' -f 0 1 2 ', datasetID=702,Lustre=True,
                            jobdir='/data/romain/PVsynth/saved_sample/nnunet/testing/predictions'):
    jobs = [];
    if Lustre :
        source_path = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/nnunet_path.sh'
    else:
        source_path = '/data/romain/PVsynth/saved_sample/nnunet/nnunet_path.sh'
    cmd_ini = f' source {source_path};' + f'export nnUNet_n_proc_DA={nbcpu}; nnUNetv2_predict '

    testset_name = get_parent_path(indirs)[1]
    for ii, indir in enumerate(indirs):
        ds_name = testset_name[ii]
        for pm, pty, ptr in zip(plan_model, plan_type, plan_train):
            model_name = f'{pm}_{ptr}_{pty}_DS{datasetID}'
            #out_path = os.path.join(result_dir,ds_name, model_name)
            out_path = os.path.join(indir,model_name)
            jobs.append( f'{cmd_ini} {cmd_option} --continue_prediction -i {indir} -o {out_path} -d {datasetID} -p {pty} -tr {ptr} -c {pm} ' )

    from script.create_jobs import create_jobs
    job_params = dict()
    job_params['output_directory'] = jobdir
    job_params['jobs'] = jobs
    job_params['job_name'] = 'nnUNet'
    job_params['cluster_queue'] = '-p medium --mem 16G '

    job_params['cpus_per_task'] = nbcpu
    # job_params['mem'] = 32000
    job_params['walltime'] = '20:00:00'
    job_params['job_pack'] = 1

    create_jobs(job_params)

    #option  --disable_progress_bar to disable progress bar. Recommended for HPC environments
            # --save_probabilities
    #dataset 708 (nb Classe 39) lancé en cpu : 33 G but runtime is crasy ! 3H par volume !

def nnunet_siam_pred_job(indirs,model_num_list=[1],
                            nbcpu=4,Lustre=True,device='gpu',
                            jobdir='/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/job_pred/siam_pred'):
    ddd=['/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/ultracortex/vol_T1std',
 '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/ULTRA_all/vol_ct',
 '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/ULTRA_all/vol_flair',
 '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/ULTRA_all/vol_inv2',
 '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/ULTRA_all/vol_uni',
 '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/ULTRA_all/vol_ute',
 '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/HCP_test_retest_07mm_suj82/vol_T1_07',
 '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/HCP_test_retest_07mm_suj82/vol_T2_07',
 '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/MICCAIstd_testset_suj20/vol_T1',
 '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/DBB_sel18/vol_T1',
 '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/dHcp_old075/vol_T2',
 '/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/nnunet_pred/Vascular/vol_DixSeg',
 '/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/nnunet_pred/Vascular/vol_FlowSeg',
 '/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/nnunet_pred/Vascular/vol_T1Seg',
 '/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/nnunet_pred/Vascular/vol_T2Seg']

    jobs = [];
    if Lustre :
        source_path = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/nnunet_path.sh'
    else:
        source_path = '/data/romain/PVsynth/saved_sample/nnunet/nnunet_path.sh'
    #cmd_ini = f' source {source_path};\n' + f'export nnUNet_n_proc_DA={nbcpu}; predNN '
    cmd_ini ='siam-pred '

    testset_name = get_parent_path(indirs)[1]
    for ii, indir in enumerate(indirs):
        ds_name = testset_name[ii]
        for model_num in model_num_list:
            jobs.append( f'{cmd_ini} -m {model_num} -i {indir} -o res  ' )

    from script.create_jobs import create_jobs
    job_params = dict()
    job_params['output_directory'] = jobdir
    job_params['jobs'] = jobs
    job_params['job_name'] = 'predNN'

    if device=="gpu":
        job_params['cpus_per_task'] = 24
        job_params['mem'] = 164000
        job_params['cluster_queue'] = '-p gpu-cenir'
        job_params['sbatch_args'] = '--gres=gpu:1'
    else: #cpu
        job_params['cpus_per_task'] = 8
        job_params['cluster_queue'] = '-p medium --mem 16G '
        job_params['cpus_per_task'] = nbcpu
        # job_params['mem'] = 32000
    job_params['walltime'] = '20:00:00'
    job_params['job_pack'] = 1

    create_jobs(job_params)

    #option  --disable_progress_bar to disable progress bar. Recommended for HPC environments
            # --save_probabilities
    #dataset 708 (nb Classe 39) lancé en cpu : 33 G but runtime is crasy ! 3H par volume !



def add_label_to_csv_pred(fcsv_in, fcsv_with_label):
    df_lab = pd.read_csv(fcsv_with_label)
    for ff in fcsv_in:
        df = pd.read_csv(ff)
        for key in df_lab.keys():
            if 'lab' in key:
                df[key] = df_lab[key]
        df.to_csv(ff,index=False)

#fcsv_with_label = '/network/iss/opendata/data/template/validataion_set/HCP_trainset_07mm_suj16.csv'
#fcsv_in = gfile('/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/csv_prediction','HCP_trainset_07mm')

def get_prediction_vol_list(vol_name_list , dir_pred):
    # check if pred are
    pred_path_list, model_name_list, input_type_list, dataset_name_list = [], [], [], []
    model_name = get_parent_path(dir_pred)[1]
    input_type = get_parent_path(dir_pred, 2)[1]
    for vol_n in vol_name_list:
        if "FastSurfer" in model_name:
            dd = gdir(dir_pred, [vol_n, 'mri'])
            ff = gfile(dd, '^remap')
            if len(ff) == 0:
                print(f'Doing missing remap file in FastSurfer {dir_pred}')
                fapar = gfile(dd, '^aparc')
                tmap = get_fastsurfer_remap(fapar[0], fcsv='/network/iss/opendata/data/template/remap/free_remapV2.csv',
                                            index_col_remap=4)
                ft1 = gfile(get_parent_path(fapar, 4)[0], vol_n)
                remap_filelist(fapar, tmap, fref=ft1, prefix='remapHyp_')
                ff = gfile(dd, '^remap')

        elif "SynthSeg" in model_name:
            ff = gfile(dir_pred, '^remap.*' + vol_n)
            if len(ff) == 0:
                print(f'Doing missing remap file in FastSurfer {dir_pred}')
                fapar = gfile(dir_pred, vol_n)
                tmap = get_fastsurfer_remap(fapar[0], fcsv='/network/iss/opendata/data/template/remap/free_remapV2.csv',
                                            index_col_remap=4)
                ft1 = gfile(get_parent_path(fapar, 2)[0], vol_n)
                remap_filelist(fapar, tmap, fref=ft1, prefix='remapHyp_')
                ff = gfile(dir_pred, '^remap.*' + vol_n)
        else:
            ff = gfile(dir_pred, vol_n)

        if len(ff) == 1:
            pred_path_list.append(ff[0]);
            model_name_list.append(model_name);
            dataset_name_list.append(ds_name);
            input_type_list.append(input_type)
        else:
            print(f'missing pred dir {get_parent_path(dir_pred)[1]} miss at least prediction for {vol_n}')
            continue
    return pred_path_list, model_name_list, input_type_list, dataset_name_list

def make_validation_csv_compare_vol(fcsv, outdir_csv, compare_label=False, pred_regex='.*',label_remap_csv='auto'):
    fcsv_out=[]
    df_multi, dict_one_row_multi = pd.DataFrame(), {}

    csv_pred =  fcsv[0]
    ds_name = get_parent_path(csv_pred)[1][:-4]
    print(f'DS name {ds_name}')
    df = pd.read_csv(csv_pred)
    lab_col=[]
    for kk in df.keys():
        if kk.startswith('lab'):
            lab_col.append(kk)
    df = df.drop(lab_col,axis=1)

    dir_data = get_parent_path(df.vol_path[0])[0]
    dir_preds = gdir(dir_data,pred_regex)
    vol_name_list = get_parent_path(df.vol_path.values)[1];
    vol_name_list = [vv[:-12] for vv in vol_name_list]  # name for pred : remove _0000.nii.gz from nnunet naming conv
    for dir_pred in dir_preds:
        pred_path_list, model_name_list, input_type_list, dataset_name_list = get_prediction_vol_list(vol_name_list, dir_pred)
        model_name = model_name_list[0]
        for other_vol_ind in range(1,len(fcsv)):
            other_df = pd.read_csv(fcsv[other_vol_ind])
            other_dir_data = get_parent_path(other_df.vol_path[0])[0]
            other_dir_preds = gdir(other_dir_data, model_name)
            other_vol_name_list = get_parent_path(other_df.vol_path.values)[1];
            other_vol_name_list = [vv[:-12] for vv in other_vol_name_list]
            other_pred_path_list, other_model_name_list, other_input_type_list, other_dataset_name_list = (
                get_prediction_vol_list(other_vol_name_list,other_dir_preds[0]))

            new_lab = f'lab_{other_input_type_list[0]}'
            df[new_lab] = other_pred_path_list
        if (len(pred_path_list)==len(vol_name_list)) & (len(other_pred_path_list)==len(other_vol_name_list)):
            if label_remap_csv == 'auto' :
                dir_transfo = '/network/iss/opendata/data/template/remap/my_synth/transfo'
                if ('FastSur' in model_name) | ('SynthSeg'in model_name):
                    reg_transfo = '^DSFree_remap'
                elif model_name.startswith('pred_DS'):
                    reg_transfo = f'^{model_name[5:10]}'

                #which GT depend on dataset
                if 'DBB' in ds_name:
                    reg_transfo += '.*label_DBB_GT'

                elif 'dHcp' in ds_name:
                    reg_transfo += '.*label_dHCP_GT'
                else:
                    reg_transfo += '.*label_GT'
                fff = gfile(dir_transfo, reg_transfo)
                if not (len(fff)==1):
                    misssingfilessss
                label_remap_fname = fff[0]
            else:
                label_remap_fname = label_remap_csv

            dfremap = pd.read_csv(label_remap_fname)
            dfremap_new = pd.DataFrame()
            dfremap_new[f'prediction_{new_lab}'] = dfremap['prediction_lab_mid']
            dfremap_new[f'label_names_{new_lab}'] = dfremap['label_names_lab_mid']
            dfremap_new[f'{new_lab}'] = dfremap['prediction_lab_mid']
            dfremap_new.dropna(inplace=True)
            new_label_remap_fname = f'{outdir_csv+get_parent_path(label_remap_fname)[1]}'
            dfremap_new.to_csv(new_label_remap_fname,index=False)

            df['predict_path'] = pred_path_list; df['model_name'] = model_name_list;
            df['dataset_name'] = dataset_name_list; df['input_type'] = input_type_list
            fcsv_out_name = f'{outdir_csv}/{ds_name}_{model_name}.csv'
            fcsv_out.append(fcsv_out_name)
            df.to_csv(fcsv_out_name,index=False)
            dict_one_row_multi['dataset_name'] = ds_name
            dict_one_row_multi['metrics'] = "dice volume"
            dict_one_row_multi['subject_csv_filepath'] = fcsv_out_name
            dict_one_row_multi['label_remap'] = new_label_remap_fname
            dict_one_row_multi['save_dir'] = os.path.join(outdir_csv,'results')
            df_multi = pd.concat([df_multi, pd.DataFrame(dict_one_row_multi, index=[0])])


    fname_multi = f'{outdir_csv}/multi_all.csv'
    if os.path.isfile(fname_multi):
        print(f"append to {fname_multi}")
        df_previous = pd.read_csv(fname_multi)
        df_multi = pd.concat([df_previous, df_multi])
    df_multi.to_csv(fname_multi, index=False)
    return fcsv_out


def make_validation_csv(fcsv, outdir_csv, compare_label=False, pred_regex='.*',
                        label_remap_csv='auto', metrics_name= "dice volume hausdorff"):
    fcsv_out=[]
    df_multi, dict_one_row_multi = pd.DataFrame(), {}
    for csv_pred in fcsv:
        ds_name = get_parent_path(csv_pred)[1][:-4]
        print(f'DS name {ds_name}')
        df = pd.read_csv(csv_pred)
        dir_data = get_parent_path(df.vol_path[0])[0]
        if isinstance(pred_regex,list):
            dir_preds = []
            for pp in pred_regex:
                dir_preds += gdir(dir_data,f'^{pp}$')
        else:
            dir_preds = gdir(dir_data,pred_regex)
        vol_name_list = df.sujname
        #vol_name_list = get_parent_path(df.vol_path.values)[1];
        #vol_name_list = [vv[:-12] for vv in vol_name_list]  # name for pred : remove _0000.nii.gz from nnunet naming conv
        for dir_pred in dir_preds:
            #check if pred are
            pred_path_list, model_name_list, input_type_list, dataset_name_list = [], [], [], []
            model_name = get_parent_path(dir_pred)[1]
            input_type = get_parent_path(dir_pred,2)[1]
            for vol_n in vol_name_list:
                if "FastSurfer" in model_name:
                    dd = gdir(dir_pred,[vol_n +'$', 'mri'])
                    ff = gfile(dd,'^remap')
                    if len(ff)==0:
                        print(f'Doing missing remap file in FastSurfer {dir_pred}')
                        fapar = gfile(dd,'^aparc')
                        tmap = get_fastsurfer_remap(fapar[0],fcsv='/network/iss/opendata/data/template/remap/free_remapV2.csv',index_col_remap=4)
                        ft1 = gfile(get_parent_path(fapar,4)[0],vol_n)
                        fffo = addprefixtofilenames(fapar,'rmrt_')
                        fo_final = addprefixtofilenames(fapar,'remapHyp_')
                        fffo[0] = fffo[0][:-4]+'.nii.gz'
                        fo_final[0] = fo_final[0][:-4]+'.nii.gz'
                        import subprocess
                        cmd = f'mrgrid {fapar[0]} regrid -interp nearest -template {ft1[0]} -strides {ft1[0]} {fffo[0]}'
                        outvalue = subprocess.run(cmd.split(' '))
                        #remap_filelist(fapar, tmap, fref=ft1,prefix='remapHyp_')
                        remap_filelist(fffo, tmap, fref=ft1, prefix='remapHyp_')
                        badfo = addprefixtofilenames(fffo, 'remapHyp_')
                        r_move_file(badfo,fo_final ,'move')
                        os.remove(fffo[0])
                        ff = gfile(dd, '^remap')

                elif "SynthSeg" in model_name:
                    ff = gfile(dir_pred,'^remap.*'+vol_n)
                    if len(ff)==0:
                        print(f'Doing missing remap file in FastSurfer {dir_pred}')
                        fapar = gfile(dir_pred,vol_n)
                        if 'ouhfi' in model_name:
                            tmap = get_fastsurfer_remap(fapar[0],index_col_in=1,index_col_remap=3,fcsv='/network/iss/opendata/data/template/remap/my_synth/label_gouhfi.csv')
                            #dfgouhfi = pd.read_csv(rd + 'label_gouhfi.csv')
                            #tmap = {dd.synth: dd.target for ii, dd in dfgouhfi.iterrows()}
                        else:
                            tmap = get_fastsurfer_remap(fapar[0],fcsv='/network/iss/opendata/data/template/remap/free_remapV2.csv',index_col_remap=4)
                        ft1 = gfile(get_parent_path(fapar,2)[0],vol_n)
                        print(f'found T1 {ft1} for {fapar}')
                        # remap_filelist(fapar, tmap, fref=ft1,prefix='remapHyp_',reslice_with_mrgrid=True)
                        remap_filelist(fapar, tmap, fref=ft1,prefix='remapHyp_',reslice_4D=True)
                        ff = gfile(dir_pred, '^remap.*' + vol_n)
                else:
                    ff = gfile(dir_pred,vol_n)

                if len(ff) == 1:
                    pred_path_list.append(ff[0]); model_name_list.append(model_name);
                    dataset_name_list.append(ds_name); input_type_list.append(input_type)
                else:
                    print(f'missing pred dir {get_parent_path(dir_pred)[1]} miss at least prediction for {vol_n}')
                    continue
            if len(pred_path_list)==len(vol_name_list):
                if label_remap_csv == 'auto' :
                    dir_transfo = '/network/iss/opendata/data/template/remap/my_synth/transfo'
                    if ('FastSur' in model_name) | ('SynthSeg'in model_name):
                        reg_transfo = '^DSFree_remap'
                    elif model_name.startswith('pred_DS'):
                        reg_transfo = f'^{model_name[5:10]}'

                    #which GT depend on dataset
                    if 'DBB' in ds_name:
                        reg_transfo += '.*label_DBB_GT'

                    elif 'dHcp' in ds_name:
                        reg_transfo += '.*label_dHCP_GT'
                    else:
                        reg_transfo += '.*label_GT_Head'
                        #reg_transfo += '.*label_GT.csv'
                    fff = gfile(dir_transfo, reg_transfo)
                    if not (len(fff)==1):
                        misssingfilessss
                    label_remap_fname = fff[0]
                else:
                    label_remap_fname = label_remap_csv

                df['predict_path'] = pred_path_list; df['model_name'] = model_name_list;
                df['dataset_name'] = dataset_name_list; df['input_type'] = input_type_list
                fcsv_out_name = f'{outdir_csv}/{ds_name}_{model_name}.csv'
                fcsv_out.append(fcsv_out_name)
                df.to_csv(fcsv_out_name,index=False)
                dict_one_row_multi['dataset_name'] = ds_name
                dict_one_row_multi['metrics'] = metrics_name
                dict_one_row_multi['subject_csv_filepath'] = fcsv_out_name
                dict_one_row_multi['label_remap'] = label_remap_fname
                dict_one_row_multi['save_dir'] = os.path.join(outdir_csv,'results')
                df_multi = pd.concat([df_multi, pd.DataFrame(dict_one_row_multi, index=[0])])


        if compare_label:
            lab_key=[]
            for ll in df.keys():
                if ('lab' in ll) :
                    lab_key.append(ll)
            print(f'Label_keys are {lab_key}')
            for kk in range(len(lab_key) - 1) :
                label_name = lab_key[kk]
                df['predict_path'] = df[label_name]; df['model_name'] = label_name;
                df['dataset_name'] = ds_name; df['input_type'] = 'label'
                df.pop(label_name)
                fcsv_out_name = f'{outdir_csv}/{ds_name}_{label_name}.csv'
                fcsv_out.append(fcsv_out_name)
                df.to_csv(fcsv_out_name,index=False)

    fname_multi = f'{outdir_csv}/multi_all.csv'
    if os.path.isfile(fname_multi):
        print(f"append to {fname_multi}")
        df_previous = pd.read_csv(fname_multi)
        df_multi = pd.concat([df_previous, df_multi])
    df_multi.to_csv(fname_multi, index=False)
    return fcsv_out

def main_make_DS():
    dtest = gdir('/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/','testing_set')
    fcsv_list = gfile(dtest,'ULTRA_all.*csv$')
    outdir_csv = gdir(dtest,'csv_validationULTRA')[0]
    make_validation_csv(fcsv_list, outdir_csv, pred_regex='708')

    for fcsv in fcsv_list:
        df = pd.read_csv(fcsv)
        dir_suj = get_parent_path(df.vol_path_orig)[0]
        #flabel = gfile(gdir(dir_suj, 'mida_v5'), '^rUTE_binmrt_r025_bin_PV_head_mida_Aseg_cereb')
        flabel = gfile(gdir(dir_suj, 'mida_v5'), '^crop_rUTE_binmrt_r025_bin_PV_head_mida_Aseg_cereb')
        if len(flabel) == len(df):
            df['lab_Assn'] = flabel
            df.to_csv(fcsv)
        else:
            qsdfqsdf

    from utils_labels import remap_filelist
    vo = gfile('/home/romain.valabregue//lll/datal/PVsynth/training_saved_sample/nnunet/testing_set/DBB_sel18/vol_T1/SynthSeg/','.*gz')
    tmap = get_fastsurfer_remap(vo[0],fcsv ='/network/iss/opendata/data/template/remap/free_remapV2.csv', index_col_remap=4)
    remap_filelist(vo,tmap, prefix='remapHyp_',)


### predic my model on nnunet testset (same arc)
def create_predict_job(model_name, model_weigh, vol_in_list, jobdir, pred_option='-vs 0.75',
                       replace_link=True, device='gpu', job_pack=10):
    cmd_ini = 'python /network/iss/cenir/software/irm/toolbox_python/romain/torchQC/segmentation//predict.py '
    if device=='cpu':
        pred_option += ' -d cpu '
    jobs = [];
    vol_dir_list, vol_name_list = get_parent_path(vol_in_list);
    vol_name_list = [vv[:-12] for vv in vol_name_list]  #name for pred : remove _0000.nii.gz from nnunet naming conv
    for m_name, m_weigh in zip(model_name, model_weigh):
        model_json = os.path.join(get_parent_path(m_weigh)[0],'model.json')
        if not os.path.isfile(model_json):
            raise(f'can not find {model_json}')

        for vol_in, vol_name, vol_dir in zip(vol_in_list, vol_name_list, vol_dir_list) :
            if replace_link:
                vol_link = os.readlink(vol_in)
                if vol_link.startswith('/network/iss/'):
                    vol_in = '/network/lustre/iss02' + vol_link[12:]

            pred_path = os.path.join(vol_dir, m_name)
            if not os.path.isdir(pred_path):
                os.mkdir(pred_path)
            fout = os.path.join(pred_path, vol_name)
            fres = addprefixtofilenames(fout,'bin_')[0] + '.nii.gz'
            if os.path.isfile(fres):
                print(f'skiping res exist {fres}')
            else:
                jobs.append( f'{cmd_ini} -m {m_weigh} -mj {model_json} -v {vol_in} -f {fout} {pred_option}' )

    job_params = dict()
    job_params['output_directory'] = jobdir
    job_params['jobs'] = jobs
    job_params['job_name'] = 'predict'
    job_params['walltime'] = '24:00:00'
    job_params['job_pack'] = job_pack
    if device=="gpu":
        job_params['cpus_per_task'] = 12
        job_params['mem'] = 64000
        job_params['cluster_queue'] = '-p gpu-cenir,gpu-ampere'
        job_params['sbatch_args'] = '--gres=gpu:1'
    else: #cpu
        job_params['cluster_queue'] = '-p norma,bigmem'
        job_params['cpus_per_task'] = 14
        job_params['mem'] = 60000

    create_jobs(job_params)


def main_predict() :
    fcsv = gfile(dtest,'.*csv')
    df = pd.concat([pd.read_csv(ff) for ff in fcsv])
    vol_in = df.vol_path.values

    model_name = ['e3_reduce_ep160', 'e3_ep110']
    model_weigh = ['/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training/HCP_training39/Uhcp4_skul_v5/e3nnUnet/fromdir_reduce/res_from_continue/model_ep40_it1280_loss11.0000.pth.tar',
                   '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training/HCP_training39/Uhcp4_skul_v5/e3nnUnet/fromdir/res5.1_from_mot_sc2/model_ep110_it5120_loss11.0000.pth.tar']
    model_name = ['Uhcp4_skv5.1_jzfd_ep150', 'hcp16_v51__jzfd_ep102']
    model_weigh = ['/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/jzay/training/RES_mida/Uhcp4_skul_v5/skv5.1/probaHead256_fromdir/noSpMean/res_bs4_m6_P192_gpu2cpu40_fromep90/model_ep60_it854_loss11.0000.pth.tar',
                   '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/jzay/training/RES_mida/Uhcp4_skul_v5/hcp16_U5_v51/fromdir/noSpMean/res_fromUhcp4skv5_bs3_P192_gpu1cpu40/model_ep102_it683_loss11.0000.pth.tar']
    jobdir = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/job_pred/my_synth_fromdir'
    create_predict_job(model_name, model_weigh, vol_in, jobdir, pred_option='-vs 0.75 ', device='gpu')

    ### some high res data
    suj_name = ['cher7T_05_orig','cher7T_05_dbdn','yv98_05','yv98_025','BigBrain_03','HighRes_035_cut_midle','HumaneP_03']
    f=['/network/iss/cenir/analyse/irm/users/romain.valabregue/DTI_cenirdev/2024_10_02_cherazade7T/T1_mprage_0.50_ns_up_orig.nii.gz',
    '/network/iss/cenir/analyse/irm/users/romain.valabregue/DTI_cenirdev/2024_10_02_cherazade7T/T1_mprage_0.50_ns_up_db_dn.nii.gz',
    '/network/iss/opendata/data/template/human_phantom/Segment/T1_05.nii.gz',
    '/network/iss/opendata/data/template/human_phantom/ds003563/derivatives/sub-yv98/T1w/averages/sub-yv98_ses-3512+3555+3589+3637+3681_offline_reconstruction_denoised-BM4D-manual_T1w_biasCorrected.nii.gz',
    '/network/iss/opendata/data/template/human_phantom/BigBrainMR/r03_BigBrainMR_T1weighted_100um.nii.gz',
    '/network/iss/opendata/data/template/HighRes/src/src/sub-01_ses-T1/MP2RAGE/sub-01_ses-T1_run-01_dir-AP_MP2RAGE_uni.nii.gz',
    '/network/iss/opendata/data/template/human_phantom/7TMRI_exvivo1mm/ds002179/derivatives/sub-EXC004/processed_data/r03_synt_FLASH.nii.gz']
    dout = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set'
    create_nnunet_testset_from_file_list(f, suj_name, dout,'HighRes')

    ### paleobrain
    fute = gfile(gdir('/data/romain/PaleoBrain', ['SUJE','UTE']),'^v.*nii')
    fute = gfile(gdir('/data/romain/PaleoBrain', ['ujet','UTE']),'^v.*nii')
    for ff in fute:
        cmd = f'gzip {ff}'
        outvalue = subprocess.run(cmd.split(' '))

    fname = remove_extension(get_parent_path(fute)[1])
    dout = '/data/romain/PaleoBrain/nnunet_testset_s14'

    #run predict
    #nnUNetv2_predict -chk checkpoint_best.pth --save_probabilities -i /data/romain/PaleoBrain/nnunet_testset/ -o /data/romain/PaleoBrain/nnunet_testset/predict/ResEncXL_fullres  -d 702 -tr nnUNetTrainer -p nnUNetResEncUNetXLPlans -c 3d_fullres -f 0 1 2

    #create skull mask
    fpred = gfile(gdir(dout,['predict','ResEncXL_fullres']),'.*nii.gz')
    from segmentation.utils import get_largest_connected_component
    for fin in fpred:
        il = tio.LabelMap(fin)
        imask = torch.zeros_like(il.data)
        imask[il.data==14] = 1
        tmp_mask = get_largest_connected_component(imask)
        il.data = imask * tmp_mask
        il.save(addprefixtofilenames(str(il.path),'skull_')[0])

def create_SynthSeg_job(fcsv_list, option='--robust --cpu --threads=8',jobdir = "/data/romain/PVsynth/saved_sample/nnunet/testing/predictions/jobs/predict_SS", replace_link=False):
    cmd_ini = f"module load FreeSurfer/7.4.1\nmri_synthseg"
    if isinstance(fcsv_list,str):
        fcsv_list = [fcsv_list]

    jobs = []
    for fcsv in fcsv_list:
        df = pd.read_csv(fcsv)
        outdir = os.path.join(get_parent_path(df.vol_path)[0][0], 'SynthSeg')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        vol_in = df.vol_path.values
        volnames = get_parent_path(vol_in)[1]
        for volpath, volname in zip(vol_in, volnames):
            if replace_link:
                vol_link = os.readlink(volpath)
                if vol_link.startswith('/network/iss/'):
                    volpath = '/network/lustre/iss02' + vol_link[12:]

            print(f'{volname}')
            jobs.append( f"{cmd_ini} --i {volpath} --o {outdir}/{volname} {option}" )

    job_params = dict()
    job_params[
        'output_directory'] = jobdir
    job_params['jobs'] = jobs
    job_params['job_name'] = 'predSynthSeg'
    job_params['walltime'] = '24:00:00'
    job_params['job_pack'] = 1
    job_params['cluster_queue'] = '-p medium'
    job_params['cpus_per_task'] = 4
    job_params['mem'] = 16000

    create_jobs(job_params)

def create_hdbet_job(fcsv, option='--save_bet_mask --no_bet_image ',jobdir = "/data/romain/PVsynth/saved_sample/nnunet/testing/predictions/jobs/predict_HD"):
    cmd_ini = f"hd-bet"
    df = pd.read_csv(fcsv)
    outdir = os.path.join(get_parent_path(df.vol_path)[0][0], 'hdbet')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    jobs = []
    vol_in = df.vol_path.values
    volnames = get_parent_path(vol_in)[1]
    for volpath, volname in zip(vol_in, volnames):
        print(f'{volname}')
        jobs.append( f"{cmd_ini} -i {volpath} --o {outdir}/{volname} {option}" )

    job_params = dict()
    job_params[
        'output_directory'] = jobdir
    job_params['jobs'] = jobs
    job_params['job_name'] = 'predictFS'
    job_params['walltime'] = '24:00:00'
    job_params['job_pack'] = 1
    job_params['cluster_queue'] = '-p norma,bigmem'
    job_params['cpus_per_task'] = 14
    job_params['mem'] = 60000

    create_jobs(job_params)


def create_FS_job(fcsv_list, option='--vox_size min  --seg_only --no_cereb --no_hypothal  --parallel --3T ', prefix_out = 'FastSurfer',
                  jobdir = "/data/romain/PVsynth/saved_sample/nnunet/testing/predictions/jobs/predict_FS", device='cpu'):

    if isinstance(fcsv_list,str):
        fcsv_list = [fcsv_list]

    jobs = []
    for fcsv in fcsv_list:
        if os.path.isfile(fcsv) & (fcsv[-3:]=='.gz') :
            sujn = get_parent_path(fcsv)[1][:-7]
            df = pd.DataFrame({'vol_path' : fcsv, 'sujname' : sujn} ,index= [0])
        else:
            df = pd.read_csv(fcsv)
        outdir = os.path.join(get_parent_path(df.vol_path)[0][0], prefix_out)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        cmd_ini = "singularity exec   --network=bridge  --nv --no-home -B /network/iss/cenir/analyse/irm/users/:/network/iss/cenir/analyse/irm/users/ -B /network/iss/opendata/data/:/network/iss/opendata/data/"
        cmd_ini = f"{cmd_ini} -B {get_parent_path(df.vol_path)[0][0]}:/data"
        cmd_ini = f"{cmd_ini} -B {outdir}:/output -B /network/iss/apps/software/scit/freesurfer/7.4.1/:/fs_license "
        cmd_ini = f"{cmd_ini} /network/iss/cenir/software/irm/singularity/fastsurfer-gpu.sif /fastsurfer/run_fastsurfer.sh"
        cmd_ini = f"{cmd_ini}  {option} --sd /output"
        cmd_ini = f"{cmd_ini}  --fs_license /fs_license/license.txt"

        for volname, sujname in zip(get_parent_path(df.vol_path)[1], df.sujname):
            print(f'{sujname} {volname}')
            jobs.append( f"{cmd_ini}   --sid {sujname} --t1 /data/{volname} " )

    job_params = dict()
    job_params[
        'output_directory'] = jobdir
    job_params['jobs'] = jobs
    job_params['job_name'] = 'predictFS'
    job_params['walltime'] = '24:00:00'
    job_params['job_pack'] = 1
    if device=="gpu":
        job_params['cpus_per_task'] = 12
        job_params['mem'] = 64000
        job_params['cluster_queue'] = '-p gpu-cenir,gpu-ampere'
        job_params['sbatch_args'] = '--gres=gpu:1'
    else: #cpu
        job_params['cluster_queue'] = '-p medium'
        job_params['cpus_per_task'] = 4
        job_params['mem'] = 16000

    create_jobs(job_params)

def create_AssN_job(fcsv, option=' ',
                  jobdir = "/data/romain/PVsynth/saved_sample/nnunet/testing/predictions/jobs/predict_AssN", device='cpu'):
    df = pd.read_csv(fcsv)
    inputdir = os.path.join(get_parent_path(df.vol_path)[0][0], 'AssN')
    if not os.path.isdir(inputdir):
        os.mkdir(inputdir)

    jobs=[]; ii=0
    for volname, sujname in zip(get_parent_path(df.vol_path)[1], df.sujname):
        print(f'{sujname} {volname}')
        outdir = os.path.join(inputdir, sujname)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        fout = os.path.join(outdir,volname)
        if not os.path.isfile(fout):
            #os.symlink(f'../../{volname}', fout)
            shutil.copyfile(df.vol_path.values[ii], fout)
        ii+=1
        cmd_ini = f"export od={outdir}\n"
        cmd_ini = f"{cmd_ini} singularity run -B /network/iss/cenir/analyse/irm/users/:/network/iss/cenir/analyse/irm/users/ "
        cmd_ini = f"{cmd_ini} -B $od:/data  -B $od:/tmp -B $od:/data_out "
        cmd_ini = f"{cmd_ini} /network/iss/cenir/software/irm/singularity/assemblynet_1.0.0.sif  /data /data_out "
        jobs.append(cmd_ini)

    job_params = dict()
    job_params[
        'output_directory'] = jobdir
    job_params['jobs'] = jobs
    job_params['job_name'] = 'predict'
    job_params['walltime'] = '24:00:00'
    job_params['job_pack'] = 1
    if device=="gpu":
        job_params['cpus_per_task'] = 12
        job_params['mem'] = 64000
        job_params['cluster_queue'] = '-p gpu-cenir,gpu-ampere'
        job_params['sbatch_args'] = '--gres=gpu:1'
    else: #cpu
        job_params['cluster_queue'] = '-p medium'
        job_params['cpus_per_task'] = 4
        job_params['mem'] = 16000

    create_jobs(job_params)


# jobdir = '/data/romain/PVsynth/saved_sample/nnunet/testing/predictions/jobs/predict_FS2'
# create_FS_job(fcsv,option='--vox_size min --parallel --3T ', prefix_out='FastSurferAll', jobdir=jobdir)

def get_nnunet_proba(fprob_list, fnii_list, label_list, prefix_list):

    for (fprob, fnii) in zip(fprob_list, fnii_list):
        arr = np.load(fprob)

        for lab, prefix in zip (label_list, prefix_list):
            fo = addprefixtofilenames(fnii,prefix)[0]
            if os.path.isfile(fo):
                print(f'Skipin exist {fo}')
            else:
                data = arr["probabilities"][lab]  # csf
                il = tio.LabelMap(fnii)
                il['data'] = torch.tensor(data.transpose((2, 1, 0))).unsqueeze(0)
                print(f'Saving {fo}')
                il.save(fo)

def main_get_proba() :
    indir = '/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/nnunet_pred/Vascular/vol_T1/3d_fullres_nnUNetTrainer_nnUNetResEncUNetXLPlans'

    fprob = gfile(indir,'013.*npz')
    fnii  = gfile(indir,'013.*nii.gz')
    label_list=[17,18]; prefix_list=['dura_','vessel_']
    get_nnunet_proba(fprob, fnii, label_list, prefix_list)


def create_Ines_eval_jobs(fcsv_list, dir_out, metric = 'dice volume',
                          mask=None, pred_remap=None): #todo make it a function
    #eval Ines
    root_py = '/data/romain/toolbox_python/romain/ines/brain-mri-multi-segmentation/pipelines/'
    cmdi1 = f'python {root_py}evaluation.py  '
    cmdi2 = f'sujname /data/romain/toolbox_python/romain/ines/brain-mri-multi-segmentation/config1/./csv_3d_partial/labels_remap.csv NameTarget target -f -a  -x {dir_out} '
    #not sure about -a option
    cmdi2 = f'sujname /data/romain/toolbox_python/romain/ines/brain-mri-multi-segmentation/config1/./csv_3d_partial/labels_remap.csv NameTarget target -f  -x {dir_out} '
    cmd = []
    for ff in fcsv_list:
        df = pd.read_csv(ff)
        col_lab = []
        for kk in df.keys():
            if 'lab_' in kk:
                col_lab.append(kk)
        for lab in col_lab:
            option = f' --metrics {metric} '
            if mask is not None:
                option = f' {option} --mask {mask} '
            if pred_remap is not None:
                option = f' {option} --pred_remap {pred_remap} '
            cmd.append(f'{cmdi1} {ff} predict_path {lab} {cmdi2} -i {df.input_type.values[0]} {option}')

    job_params = dict()
    job_params['output_directory'] = dir_out + '/job_eval'
    job_params['jobs'] = cmd
    job_params['cluster_queue'] = '-p normal,bigmem'
    job_params['cpus_per_task'] = 4
    job_params['walltime'] = '24:00:00'
    job_params['job_pack'] = 1
    job_params['mem'] = 60000
    job_params['job_name'] = 'eval'

    create_jobs(job_params)

def main_ines():
    outdir_csv = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/csv_validationULTRA/AGAIN'
    fcsv_list = gfile(outdir_csv, '.*csv')
    dir_out = outdir_csv + '/results/'
    ftransfo = '/network/iss/opendata/data/template/remap/my_synth/transfo/map_DS710_to_DS704.csv'
    create_Ines_eval_jobs(fcsv_list, dir_out, mask='mask_top', pred_remap=ftransfo )

    #remap pred DS_706 to GT new v5
    rdDS = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/ULTRA_all'
    suj = gdir(rdDS,['.*','XLPlans_DS706']); suj.pop(3)
    fpred = gfile(suj,'.*gz')
    dfl = pd.read_csv('/network/iss/opendata/data/template/remap/my_synth/Svas_03_label_DS706.csv')
    tmap = tio.RemapLabels( {dd.synth_target:dd.synth_target_v5 for ii,dd in dfl.iterrows()}) #

    remap_filelist(fpred, tmap, prefix='remap_target_v5_')

def remap_dataset_from_csv():
    #conversion from different DS
    def dic_to_csv(dic_remap,fout, key_name='col1', val_name = 'col2' ):
        df=pd.DataFrame()
        df[key_name] = dic_remap.keys()
        df[val_name] = dic_remap.values()
        df.to_csv(fout)

    def make_label_remap(df8, df8_name='synth_target', df8_val_remap='Map_to_label_GT_Name',ds_name = 'DS708',
                         csv_GT_name='label_GT'):
        rd = '/network/iss/opendata/data/template/remap/my_synth/'
        rd_transfo = gdir(rd, 'transfo')[0] + '/'
        dflab = pd.read_csv(rd + f'{csv_GT_name}.csv')
        dfmap = pd.DataFrame()
        dfmap['prediction_val'] = np.sort(df8[df8_name].unique())
        for k in dflab:
            if k.startswith('lab_'): # in k:
                if f'remap_{k}' in dflab: #meaning we want also to remap GT
                    colGT = f'remap_{k}'
                else:
                    colGT = k
                print(k)
                dic_lab = {ll['Name']: ll[colGT] for ii, ll in dflab.iterrows()}

                dic_lab_inv={}
                for ii,ll in dflab.iterrows():
                    if ll[k] not in dic_lab_inv: #first occurence of 0 will be BG and not missing labels
                        dic_lab_inv.update({ll[colGT] : ll['Name']})

                #dic_lab_inv = {ll[k] : ll['Name'] for ii, ll in dflab.iterrows()}
                print(dic_lab)
                dic_remap_8_to_lab = {ll[df8_name]: dic_lab[ll[df8_val_remap]] for ii, ll in df8.iterrows()}
                #added sorted for 710 ... because implicit (continuous increasing values for "targe label" mapping
                dic_remap_8_to_lab = {k: v for k, v in sorted(dic_remap_8_to_lab.items(), key=lambda item: item[0])}
                dfmap[f'prediction_{k}'] =  [v for k,v in dic_remap_8_to_lab.items()]
                dfmap[f'label_names_{k}'] =  [dic_lab_inv[v] for k,v in dic_remap_8_to_lab.items()]

                if f'remap_{k}' in dflab: #meaning we want also to remap GT
                    dfmap[f'{k}'] = dflab[f'remap_{k}'].dropna().astype(int)
                    #  dfmap[f'{k}'] = dflab[f'remap_{k}'].astype(int)
                    dfmap[f'{k}'] = dfmap[f'{k}'].astype('Int64')
        dfmap.to_csv(f'{rd_transfo}/{ds_name}_remap_to_{csv_GT_name}.csv', index=False)

    df12 = pd.read_csv(rd+'region_Few_DS712.csv')
    make_label_remap(df12, df8_name='synth', df8_val_remap='Map_to_label_GT_Name', ds_name='DS712')
    make_label_remap(df12, df8_name='synth', df8_val_remap='Map_to_label_GT_Head_Name', ds_name='DS712',csv_GT_name='label_GT_Head')
    make_label_remap(df12, df8_name='synth', df8_val_remap='Map_to_label_dHCP_GT_Name', ds_name='DS712',csv_GT_name='label_dHCP_GT')

    df8 = pd.read_csv(rd+'brain_and_skull_and_head_Ultra_v2_label_DS708.csv');    df8.drop(47, inplace=True) #Tumor add
    make_label_remap(df8)
    make_label_remap(df8, df8_name='synth_target', df8_val_remap='Map_to_label_GT_Head_Name', ds_name='DS708',csv_GT_name='label_GT_Head')
    make_label_remap(df8, df8_name='synth_target', df8_val_remap='Map_to_label_dHCP_GT_Name', ds_name='DS708',csv_GT_name='label_dHCP_GT')

    df9 = pd.read_csv(rd+'brain_and_skull_and_Ass_vascular_v3_label_DS709.csv')
    make_label_remap(df9, df8_name='synth_target', df8_val_remap='Map_to_label_GT_Name', ds_name='DS709')
    make_label_remap(df9, df8_name='synth_targetRegion', df8_val_remap='Map_to_label_GT_Name', ds_name='DS710')
    make_label_remap(df9, df8_name='synth_target', df8_val_remap='Map_to_label_dHCP_GT_Name', ds_name='DS709',csv_GT_name='label_dHCP_GT')
    make_label_remap(df9, df8_name='synth_targetRegion', df8_val_remap='Map_to_label_dHCP_GT_Name', ds_name='DS710',csv_GT_name='label_dHCP_GT')

    df6 = pd.read_csv(rd+'Svas_04_label_DS706.csv')
    make_label_remap(df6, df8_name='synth_target', df8_val_remap='Map_to_label_GT_Name', ds_name='DS706')
    make_label_remap(df6, df8_name='synth_target', df8_val_remap='Map_to_label_dHCP_GT_Name', ds_name='DS706',csv_GT_name='label_dHCP_GT')

    df4 = pd.read_csv(rd+'new_label_v5_hcp_DS702_DS704.csv')
    make_label_remap(df4, df8_name='target', df8_val_remap='Map_to_label_GT_Name', ds_name='DS704')
    make_label_remap(df4, df8_name='target', df8_val_remap='Map_to_label_dHCP_GT_Name', ds_name='DS704',csv_GT_name='label_dHCP_GT')

    dfFree = pd.read_csv(rd+'label_remapHyp_Free.csv')
    make_label_remap(dfFree, df8_name='target', df8_val_remap='Map_to_label_dHCP_GT_Name', ds_name='DSFree',csv_GT_name='label_dHCP_GT_nohead')
    make_label_remap(dfFree, df8_name='target', df8_val_remap='Map_to_label_GT_Name', ds_name='DSFree',csv_GT_name='label_GT_nohead')
    # make_label_remap(dfFree, df8_name='target', df8_val_remap='Map_to_label_GT_Head_Name', ds_name='DS712',csv_GT_name='label_GT_Head')
    make_label_remap(dfFree, df8_name='target', df8_val_remap='Map_to_label_GT_Name', ds_name='DSFree',csv_GT_name='label_GT_Head')

    dfgouhfi = pd.read_csv(rd+'label_gouhfi.csv')
    tmap = {dd.synth: dd.target for ii, dd in dfgouhfi.iterrows()}
    #same as FastS make_label_remap(dfgouhfi, df8_name='synth', df8_val_remap='Map_to_label_GT_Name', ds_name='DSGouhfi',csv_GT_name='label_GT_nohead')
    #df = pd.read_csv(rd + '/transfo/DSGouhfi_remap_to_label_GT_nohead.csv');


    df4 = pd.read_csv(rd+'new_label_v5_hcp_DS702_DS704.csv')
    make_label_remap(df4, df8_name='target', df8_val_remap='Map_to_label_DBB_GT_Name', ds_name='DS704',csv_GT_name='label_DBB_GT')
    df6 = pd.read_csv(rd+'Svas_04_label_DS706.csv')
    make_label_remap(df6, df8_name='synth_target', df8_val_remap='Map_to_label_DBB_GT_Name', ds_name='DS706',csv_GT_name='label_DBB_GT')
    df8 = pd.read_csv(rd+'brain_and_skull_and_head_Ultra_v2_label_DS708.csv');    df8.drop(47, inplace=True) #Tumor add
    make_label_remap(df8, df8_name='synth_target', df8_val_remap='Map_to_label_DBB_GT_Name', ds_name='DS708',csv_GT_name='label_DBB_GT')
    df9 = pd.read_csv(rd+'brain_and_skull_and_Ass_vascular_v3_label_DS709.csv')
    make_label_remap(df9, df8_name='synth_target', df8_val_remap='Map_to_label_DBB_GT_Name', ds_name='DS709',csv_GT_name='label_DBB_GT')
    make_label_remap(df9, df8_name='synth_targetRegion', df8_val_remap='Map_to_label_DBB_GT_Name', ds_name='DS710',csv_GT_name='label_DBB_GT')
    df12 = pd.read_csv(rd+'region_Few_DS712.csv')
    make_label_remap(df12, df8_name='synth', df8_val_remap='Map_to_label_DBB_GT_Name', ds_name='DS712',csv_GT_name='label_DBB_GT')
    dfFree = pd.read_csv(rd+'label_remapHyp_Free.csv')
    make_label_remap(dfFree, df8_name='target', df8_val_remap='Map_to_label_DBB_GT_Name', ds_name='DSFree',csv_GT_name='label_DBB_GT')

    #Slicer_color_maps from freesufer
    rg,dic_FS = read_freesurfer_colorlut()
    flab = rd+'brain_and_skull_and_head_Ultra_v2_label_DS708.csv'
    flab_out = rd+'slicer_def/FS_brain_and_skull_and_head_Ultra_v2_label_DS708.ctlb'
    df8 = pd.read_csv(flab);    df8.drop(47, inplace=True) #Tumor add
    with open(flab_out, "w") as f:
        for ii, cc in enumerate(df8.Map_to_FreeColor):
            f.write(f'{ii} {cc} {dic_FS[cc][0][0]} {dic_FS[cc][0][1]} {dic_FS[cc][0][2]} 255\n')
    dic_lab_synth={}
    for ii,ff in df8.iterrows():
        if ff.synth_tissu not in dic_lab_synth:
            dic_lab_synth[ff.synth_tissu] = ff.Map_to_FreeColor
    with open(flab_out, "w") as f:
        for ii,(kk, vv) in enumerate(dic_lab_synth.items()):
            f.write(f'{ii} {vv} {dic_FS[vv][0][0]} {dic_FS[vv][0][1]} {dic_FS[vv][0][2]} 255\n')
