import torchio
import torchio as tio, numpy as np
import torch, pandas as pd, nibabel as nib, tempfile

from utils_file import get_parent_path, gfile, gdir, addprefixtofilenames, r_move_file
from script.export_nnunet import create_nnunet_dataset_from_nii, nnunet_train_job
import subprocess, os
from script.create_jobs import create_jobs
import commentjson as json
from segmentation.run_model import ArrayTensorJSONEncoder

import csv, argparse

#function to parse the json parameter and construct the corresponding torchio transform

def record_history(info, sample, idx=None):
    is_batch = not isinstance(sample, torchio.Subject)
    order = []
    history = sample.get('history') if is_batch else sample.history
    transforms_metrics = sample.get("transforms_metrics") if is_batch else sample.transforms_metrics
    if history is None or len(history) == 0:
        return
    relevant_history = history[idx] if is_batch else history
    info["history"] = relevant_history

    relevant_metrics = transforms_metrics[idx] if is_batch else transforms_metrics

    if len(relevant_metrics) == 1 and isinstance(relevant_metrics[0], list):
        relevant_metrics = relevant_metrics[0]
    info["transforms_metrics"] = relevant_metrics
    if len(relevant_history)==1 and isinstance(relevant_history[0], list):
        relevant_history = relevant_history[0] #because ListOf transfo to batch make list of list ...

    for hist in relevant_history:
        if isinstance(hist, dict) :
            histo_name = hist['name']
            for key, val in hist.items():
                if callable(val):
                    hist[key] = str(val)
            str_hist = str( hist )
        else:
            histo_name = hist.name #arg bad idea to mixt transfo and dict
            str_hist = dict()
            for name in  hist.args_names :
                val = getattr(hist, name)
                if callable(val):
                    val = str(val)
                str_hist[name] = val
#               str_hist = {name: str() if isinstance(getattr(hist, name),funtion) else getattr(hist, name) for name in hist.args_names}
        #instead of using str(hist) wich is not correct as a python eval, make a dict of input_param
        if f'T_{histo_name}' in info:
            histo_name = f'{histo_name}_2'
        info[f'T_{histo_name}'] = json.dumps(str_hist, cls=ArrayTensorJSONEncoder)
        order.append(histo_name)

    info['transfo_order'] = '_'.join(order)
def get_transfo_short_name(in_str):
    ll = in_str.split('_')
    name = ''
    for sub in ll:
        name += sub[:3]
    return name

def get_transform_from_json(json_file):
    with open(json_file) as f:
        transfo_st = json.load(f)

    return parse_transform(transfo_st['train_transforms'])

def parse_transform(t):
    if isinstance(t, list):
        transfo_list = [parse_transform(tt) for tt in t]
        return tio.Compose(transfo_list)

    attributes = t.get('attributes') or {}
    print(f'attr {attributes}')

    t_class = getattr(tio.transforms, t['name'])
    return t_class(**attributes)

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


def generate_one_suj(label_file, label_csv, fout, transfo_list):
    print(f'computing synth image {fout}')
    foimg, folab_bin, folab_4D = (addprefixtofilenames(fout,'Sim_')[0], addprefixtofilenames(fout,'Lab_')[0], addprefixtofilenames(fout,'4DLab_')[0])

    df = pd.read_csv(label_csv)
    dic_lab = {ll['NameTarget']:ll['synth_target'] for ii,ll in df.iterrows()}
    dic_map_synth = {ll['synth_target']:ll['synth_tissu'] for ii,ll in df.iterrows()}
    #dic_map_target = {ll['synth']:ll['synth_target'] for ii,ll in df.iterrows()}
    dic_map_target = {40:40, 41:40, 42:40 } #all tumor (possible) subclass will be segmented as one label (40)
    max_ind_synth = dic_map_synth[max(dic_map_synth, key=dic_map_synth.get)] #get the last index of synth data
    dic_map_synth.update({40:(max_ind_synth+1), 41:(max_ind_synth+2), 42:(max_ind_synth+3)})

    tmapSynth, tmapTarget = tio.RemapLabels(dic_map_synth), tio.RemapLabels(dic_map_target)

    #add three tumor type for synth

    ilabel = tio.LabelMap(label_file)
    suj = tio.Subject(label=ilabel) #, label_target = i_lab_target )
    suj_synth = tmapSynth(suj)

    i_lab_target = tio.LabelMap(tensor=suj.label.data, affine=suj.label.affine)
    i_lab_target = tmapTarget(i_lab_target)

    sujt = transfo_list[0](suj_synth)  #randomLabel2Image + bias + motion + noise

    suj_label_target = tio.LabelMap(tensor=i_lab_target.data, affine=suj.label.affine)
    thoti = tio.OneHot(invert_transform=True)
    suj_label_target = thoti(suj_label_target)

    out_prefix=''
    info1 = {}
    record_history(info1,sujt)
    transfo_name = get_transfo_short_name(info1['transfo_order'])
    folab_bin= folab_bin[:-7] + out_prefix + transfo_name + '.nii.gz'
    foimg = foimg[:-7] + out_prefix + transfo_name + '.nii.gz'

    suj_label_target.save(folab_bin)

    sujt.t1.save(foimg)
    focsv =  foimg[:-7] + '.csv'
    info1.pop('history')
    with open(focsv,'w') as f:
        w = csv.writer(f)
        w.writerows(info1.items())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_array', type=str,
                        help='integer')
    args = parser.parse_args()

    #Synth generation v3 from 4D label with brast tumor inserted (by ines)
    num_array = int(args.num_array)
    print(f'input arg is {num_array} ')

    dir_label = '/network/iss/cenir/analyse/irm/users/ines.khemir/gen_tumor_v1/random_tumor_Ulra3suj_SkVessel'
    fins = gfile(dir_label, '^4D.*gz')
    print(f'taking only file array from {10 * (num_array-1)} to {10*(num_array)}')
    fins = fins[10 * (num_array-1) : 10*(num_array)]
    print(f'{len(fins)}')

    dirout = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/Ulra3suj_SkVessel_tumorBrast/'

    label_csv = '/network/iss/opendata/data/template/remap/my_synth/brain_and_skull_and_Ass_vascular_v3_label.csv'
    label_csv = '/network/iss/opendata/data/template/remap/my_synth/brain_and_skull_and_head_Ultra_v2_label.csv'

    transfo_list = [ get_transform_from_json(dirout + 'transform.json')]

    tmp_name = get_parent_path(tempfile.TemporaryDirectory().name)[1]
    output_dir = dirout + f'/generate_{tmp_name}/'
    os.mkdir(output_dir)
    nb_example_per_subject = 1
    add_lab = ['cereb', 'hip', 'yeb', 'gm']

    for ngen in range(nb_example_per_subject):
        print(f'Pass {ngen+1}')
        for k, label_file  in enumerate(fins ) :
            sujname = f'Suj_{k:02}'
            fout = output_dir + f'gen{ngen+100:03}_{sujname}.nii.gz'
            generate_one_suj(label_file, label_csv, fout, transfo_list)

    # sbatch -p medium,gpu-cenir --array=1-91 --cpus-per-task=8 -t 1-00:00:00  --mem 40000 --job-name=rrr -o vas_rrr-%A_%a.log -e vas_rrr-%A_%a.err run_vas.sh
    # with  cat run_vas.sh
    # #!/bin/bash
    # python /network/iss/cenir/software/irm/toolbox_python/romain/torchQC/script/generate_synth_with_tumor.py -n ${SLURM_ARRAY_TASK_ID}

def regroupe_GenTmpDir(dirout):
    #regroup all generated data in one folder  #just change the generation number in file name
    suj = gdir(dirout,'gene')
    dout1, dout2 = dirout+'synth_bin' , dirout + 'synth_4D'
    if not os.path.isdir(dout1): os.mkdir(dout1);
    if not os.path.isdir(dout2): os.mkdir(dout2);

    for k,dirgen in enumerate(suj):
        f = gfile(dirgen, '^[SL]')
        fname = get_parent_path(f)[1]
        fnew = [ f'{dout1}/{ff[:7]}{k:03}{ff[10:]}' for ff in fname]
        r_move_file(f, fnew, type='move')

        f = gfile(dirgen, '^[4d]')
        fname = get_parent_path(f)[1]
        fnew = [ f'{dout2}/{ff[:9]}{k:03}{ff[12:]}' for ff in fname]
        r_move_file(f, fnew, type='move')

def main_scrip_line_after_generation():

    dirout = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/Ulra3suj_SkVessel_tumorBrast/'
    regroupe_GenTmpDir(dirout)

    #create nnunet DS with sym link
    dataset_name, dnnunet_root = 'Dataset718_Ultra_SkulVasc40_TumorBrast', '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/'

    dout1 = dirout + 'synth_bin'
    fimg, flab = gfile(dout1,'^Sim.*gz'), gfile(dout1,'^Lab.*gz')
    #also add nnunet training data without tumor
    dirout_notumor = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/Ulra3suj_SkVessel/synth_bin/'
    fimg += gfile(dirout_notumor, '^Sim.*gz')
    flab += gfile(dirout_notumor, '^Lab.*gz')

    df = pd.read_csv('/network/iss/opendata/data/template/remap/my_synth/brain_and_skull_and_head_Ultra_v2_label.csv')
    label_dic = {ll['NameTarget']:ll['synth_target'] for ii,ll in df.iterrows()}

    create_nnunet_dataset_from_nii(fimg, flab,label_dic,dataset_name, dnnunet_root, base_name = 'RRR',lab_one_hot=False)

    # nnUNetv2_plan_and_preprocess  -c 3d_fullres -d 718 --verify_dataset_integrity # very long with 902*2 volumes
    # nnUNetv2_plan_experiment -d 718 -pl nnUNetPlannerResEncL
    # nnUNetv2_plan_experiment -d 718 -gpu_memory_target 25 -overwrite_plans_name nnUNetPlans_25G

    plan_model = ['3d_fullres','3d_fullres']  # , '3d_fullres']
    plan_type = ['-p nnUNetResEncUNetLPlans', '-p nnUNetPlans_50G' ]  # ' ] #,-p nnUNetPlannerResEncXL '-p nnUNetPlans']
    nnunet_train_job(710, jobdir_name='trainVasv3_region', nbfold=3, nbcpu=14,
                     plan_model=plan_model, plan_type=plan_type)

