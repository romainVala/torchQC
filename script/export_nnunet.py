import torch,numpy as np,  torchio as tio
import json, os, seaborn as sns, tarfile, subprocess, pathlib
from utils_file import gfile, gdir, get_parent_path, addprefixtofilenames, remove_extension
import pandas as pd
from nibabel.viewers import OrthoSlicer3D as ov
import matplotlib.pyplot as plt
from script.create_jobs import create_jobs

din = '/data/romain/PVsynth/saved_sample/Uhcp4_skv5.1/tio_save_mot_nii/'
din = '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/Uhcp4_skv5.1/tio_save_mot_nii_noSpMean'

dsuj = gdir(din,'ep')
fsuj = gfile(dsuj,'^suj.*gz')
fcsv = gfile(dsuj,'^suj.*csv')
fin, flab = gfile(dsuj,'^t1'), gfile(dsuj,'^lab')


dflab = pd.read_csv('/network/lustre/iss02/opendata/data/template/MIDA_v1.0/for_training/suj_skull_v5.1/new_label_v5_hcp.csv')
label_dic = {'background' : 0}
label_dic.update({ df[1]['NameTarget'] : df[1]['target'] for df in dflab[1:13].iterrows() })
label_dic['eyes'], label_dic['skull'],label_dic['head'] = 13, 14, 15 #remap in training
thoti = tio.OneHot(invert_transform=True, copy=False) #sinon pbr de path de jzay


dnnunet = '/data/romain/PVsynth/saved_sample/nnunet/'
dnnunet = '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/'
base_name =  'RRR'
dataset_name = 'Dataset702_Uhcp4_skv51_mot_noSpMean1000_stdTh045'  #'Dataset701_Uhcp4_skv5.1_tio_save_mot_nii/',

dnnunetData = dnnunet + dataset_name +'/'
dnnunetPreproc, dnnunetRes, img_path, label_path = dnnunet + 'Preproc/', dnnunet + 'Results/',dnnunetData + 'imagesTr/',dnnunetData + 'labelsTr/'
if not os.path.isdir(dnnunetData): os.mkdir(dnnunetData);
if not os.path.isdir(dnnunetPreproc): os.mkdir(dnnunetPreproc);
if not os.path.isdir(dnnunetRes): os.mkdir(dnnunetRes);
if not os.path.isdir(img_path): os.mkdir(img_path);
if not os.path.isdir(label_path): os.mkdir(label_path)

with open(dnnunet+'/nnunet_path.sh', 'w') as file:
    file.write(f'export nnUNet_raw="{dnnunet}"\n')
    file.write(f'export nnUNet_preprocessed="{dnnunetPreproc}"\n')
    file.write(f'export nnUNet_results="{dnnunetRes}"\n')


### CREATE from tio suj  and  filter csv augmentation
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
for ii, (fdata, flabel) in enumerate(zip(fin, flab)):
    fout_name = f'{base_name}_{ii:04}_0000.nii.gz'  #only one input channel
    if not os.path.isfile((img_path+fout_name)):
        os.symlink(fdata, img_path + fout_name)

    #must transform 4d label to 3D os.symlink(flabel, label_path + fout_name)
    fout_name = f'{base_name}_{ii:04}.nii.gz'  #no channels here
    if not os.path.isfile((label_path+fout_name)):
        il = tio.LabelMap(flabel)
        il = thoti(il)
        il.save(label_path+fout_name)

    print(ii)
nb_suj = 1000 #len(fin)
json_dic = {"channel_names": {"0":"T1"}, 'labels':label_dic, "numTraining": nb_suj, "file_ending": ".nii.gz"}
with open(dnnunetData+'/dataset.json', 'w') as file:
    json.dump(json_dic, file, indent=4, sort_keys=False)

### CREATE test set from file path and file_name list
def create_nnunet_testset_from_file_list(file_list, name_list, dout, max_suj=100000, ext='.nii.gz'):
    for ind_suj, (fin, fname) in enumerate(zip(file_list, name_list)):
        fout = f'{dout}/{fname}_0000{ext}'
        if not os.path.isfile(fout):
            os.symlink(fin, fout)
        if ind_suj>=max_suj:
            break



### CREATE Inferenc form validataion csv
def create_nnunet_testset_from_csv(fcsv_list, dout):
    ds_name_list = remove_extension(get_parent_path(fcsv_list)[1])

    for fcsv, ds_name in zip(fcsv_list,ds_name_list):
        df = pd.read_csv(fcsv); # df = df[:10]
        keys_img_list = []
        for kk in df.keys():
            if kk.startswith('vol'):
                keys_img_list.append(kk)
        keys_lab_list = []

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
                if kk.startswith('lab'):
                    dfnew[kk] = df[kk].values
            dfnew.to_csv(dout+'/'+ ds_name + '_' + keys_img + '.csv', index=False)

dtest='/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/'
fcsv_list = gfile('/network/iss/opendata/data/template/validataion_set/' ,'csv$')
create_nnunet_testset_from_csv(fcsv_list, dtest)


def nnunet_make_predict_job(indirs,plan_model,plan_train,plan_type,
                            nbcpu=12, cmd_option=' -f 0 1 2 ', datasetID=702,
                            jobdir='/data/romain/PVsynth/saved_sample/nnunet/testing/predictions'):
    jobs = [];
    source_path = '/data/romain/PVsynth/saved_sample/nnunet/nnunet_path.sh'
    cmd_ini = f' source {source_path};' + f'export nnUNet_n_proc_DA={nbcpu}; nnUNetv2_predict '

    testset_name = get_parent_path(indirs)[1]
    for ii, indir in enumerate(indirs):
        ds_name = testset_name[ii]
        for pm, pty, ptr in zip(plan_model, plan_type, plan_train):
            model_name = f'{pm}_{ptr}_{pty}'
            #out_path = os.path.join(result_dir,ds_name, model_name)
            out_path = os.path.join(indir,model_name)
            jobs.append( f'{cmd_ini} {cmd_option} -i {indir} -o {out_path} -d 702 -p {pty} -tr {ptr} -c {pm} ' )

    dout = os.path.join(jobdir,'jobs')
    from script.create_jobs import create_jobs
    job_params = dict()
    job_params['output_directory'] = dout + '/predict'
    job_params['jobs'] = jobs
    job_params['job_name'] = 'nnUNet'
    job_params['cluster_queue'] = ''
    job_params['cpus_per_task'] = nbcpu
    # job_params['mem'] = 32000
    job_params['walltime'] = '20:00:00'
    job_params['job_pack'] = 1

    create_jobs(job_params)


indirs = gdir('/data/romain/PVsynth/saved_sample/nnunet/testing/','.*UL')
nnunet_make_predict_job(indirs)
indirs = gdir('/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set',
              ['.*','vol'])
plan_model = ['3d_fullres', '3d_cascade_fullres', '3d_lowres', '3d_fullres', '3d_fullres', '3d_fullres']
plan_train = ['nnUNetTrainer', 'nnUNetTrainer', 'nnUNetTrainer', 'nnUNetTrainerNoDA', 'nnUNetTrainerNoDeepSupervision', 'nnUNetTrainer']
plan_type = ['nnUNetPlans', 'nnUNetPlans', 'nnUNetPlans', 'nnUNetPlans', 'nnUNetPlans', 'nnUNetResEncUNetXLPlans']
nnunet_make_predict_job(indirs,plan_model, plan_train, plan_type)


def make_validation_csv(fcsv, outdir_csv):
    for csv_pred in fcsv:
        ds_name = get_parent_path(csv_pred)[1][:-4]
        df = pd.read_csv(csv_pred)
        dir_data = get_parent_path(df.vol_path[0])[0]
        dir_preds = gdir(dir_data,'.*')
        vol_name_list = get_parent_path(df.vol_path.values)[1];
        vol_name_list = [vv[:-12] for vv in vol_name_list]  # name for pred : remove _0000.nii.gz from nnunet naming conv
        for dir_pred in dir_preds:
            #check if pred are
            pred_path_list, model_name_list, input_type_list, dataset_name_list = [], [], [], []
            model_name = get_parent_path(dir_pred)[1]
            input_type = get_parent_path(dir_pred,2)[1]
            for vol_n in vol_name_list:
                ff = gfile(dir_pred,vol_n)
                if len(ff) == 1:
                    pred_path_list.append(ff[0]); model_name_list.append(model_name);
                    dataset_name_list.append(ds_name); input_type_list.append(input_type)
                else:
                    print(f'missing pred dir {get_parent_path(dir_pred)[1]} miss at least prediction for {vol_n}')
                    break
            if len(pred_path_list)==len(vol_name_list):
                df['predict_path'] = pred_path_list; df['model_name'] = model_name_list;
                df['dataset_name'] = dataset_name_list; df['input_type'] = input_type_list
                df.to_csv(f'{outdir_csv}/{ds_name}_{model_name}.csv',index=False)

fcsv_list = gfile(dtest ,'csv$')
outdir_csv = gdir(dtest,'csv_validation')[0]
make_validation_csv(fcsv, outdir_csv)


def nnunet_train_job():
    nbfold = 3; jobs=[]; nbcpu = 40 #40 for V100  24 for A100 ?
    cmd_ini = f'export nnUNet_n_proc_DA={nbcpu}; /linkhome/rech/gencme01/urd29wg/.local/bin/nnUNetv2_train 702 '
    plan_model = ['3d_lowres', '3d_fullres', '3d_fullres']
    plan_model = ['3d_fullres', '3d_fullres']
    plan_type = [ '', '-p nnUNetResEncUNetXLPlans']
    for pm, pt in zip(plan_model, plan_type):
        for fold in range(nbfold):
            jobs.append( f'{cmd_ini} {pm} {fold} {pt}' )

    dout = '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/job'
    from script.create_jobs import create_jobs
    job_params = dict()
    job_params[
        'output_directory'] = dout+'/train2'
    job_params['jobs'] = jobs
    job_params['job_name'] = 'nnUNet'
    job_params['cluster_queue'] = '-C v100-32g --qos=qos_gpu-t3 --account=ezy@v100 --gres=gpu:1'
    job_params['cpus_per_task'] = nbcpu
    #job_params['mem'] = 32000
    job_params['walltime'] = '20:00:00'
    job_params['job_pack'] = 1

    create_jobs(job_params)


### predic my model on nnunet testset (same arc)
def create_predict_job(model_name, model_weigh, vol_in_list, jobdir, pred_option='-vs 0.75',
                       replace_link=True, device='gpu', job_pack=10):
    cmd_ini = 'python /network/lustre/iss02/cenir/software/irm/toolbox_python/romain/torchQC/segmentation//predict.py '
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
    job_params[
        'output_directory'] = jobdir
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


fcsv = gfile(dtest,'.*csv')
df = pd.concat([pd.read_csv(ff) for ff in fcsv])
vol_in = df.vol_path.values

model_name = ['e3_reduce_ep160', 'e3_ep110']
model_weigh = ['/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/training/HCP_training39/Uhcp4_skul_v5/e3nnUnet/fromdir_reduce/res_from_continue/model_ep40_it1280_loss11.0000.pth.tar',
               '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/training/HCP_training39/Uhcp4_skul_v5/e3nnUnet/fromdir/res5.1_from_mot_sc2/model_ep110_it5120_loss11.0000.pth.tar']
model_name = ['Uhcp4_skv5.1_jzfd_ep150', 'hcp16_v51__jzfd_ep102']
model_weigh = ['/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/jzay/training/RES_mida/Uhcp4_skul_v5/skv5.1/probaHead256_fromdir/noSpMean/res_bs4_m6_P192_gpu2cpu40_fromep90/model_ep60_it854_loss11.0000.pth.tar',
               '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/jzay/training/RES_mida/Uhcp4_skul_v5/hcp16_U5_v51/fromdir/noSpMean/res_fromUhcp4skv5_bs3_P192_gpu1cpu40/model_ep102_it683_loss11.0000.pth.tar']
jobdir = '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/job_pred/my_synth_fromdir'
create_predict_job(model_name, model_weigh, vol_in, jobdir, pred_option='-vs 0.75 ', device='gpu')



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
