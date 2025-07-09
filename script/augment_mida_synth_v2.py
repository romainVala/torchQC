import torchio
import torchio as tio, numpy as np
import torch, pandas as pd, nibabel as nib, tempfile

from utils_file import get_parent_path, gfile, gdir, addprefixtofilenames, r_move_file
from utils_labels import get_mask_external_broder
from utils_labels import remap_filelist, get_fastsurfer_remap, get_remap_from_csv,resample_and_smooth4D
from utils_labels import single_to_4D, pool_remap_to_4DPV, pool_remap
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure

from scipy.ndimage import label as scipy_label
import subprocess, os
from script.create_jobs import create_jobs
import commentjson as json
from segmentation.run_model import ArrayTensorJSONEncoder

import csv

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


#dillat GM into CSF/WM
def dill_label_within(il, label_to_dill, within_label, nb_iter=2, out_prefix =None):
    if isinstance(il,str):
        fin = il
        il = tio.LabelMap(il)
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

    if out_prefix is not None:
        if nbiter>0:
            out_prefix = f'{out_prefix}it{nbiter+1}_'
        fout = addprefixtofilenames(fin, out_prefix)[0]
        print(f'saving {fout}')
        il.save(fout)
    else:
        return il

def add_some_dill(ilab, dic_map):
    def dill_dura(ilab):
        rd_choice = torch.randint(0,4,(1,)).numpy()[0]
        if rd_choice>=1 :
            return ilab, ''

        nb_iter = torch.randint(1,4,(1,)).numpy()[0]
        if nb_iter<2:
            out_prefix = f'_dDura_inGmCsfSku_it{nb_iter}'
            label_to_dill = dic_map['Dura']
            within_label = [ dic_map['Skull Inner Table'], dic_map['CSF'], dic_map['GM'] ]
        else:
            rd_side = torch.randint(1,4,(1,)).numpy()[0]
            if rd_side==1:
                label_to_dill = dic_map['Dura']
                within_label = [ dic_map['CSF'], dic_map['GM']]
                out_prefix = f'_dDura_inGmCsf_it{nb_iter}'
            else:
                label_to_dill = dic_map['Skull Inner Table']
                within_label = [dic_map['Skull_Diploe']]
                ilab = dill_label_within(ilab, label_to_dill=label_to_dill, within_label=within_label, nb_iter=nb_iter-1)

                label_to_dill = dic_map['Dura']
                within_label = [dic_map['Skull Inner Table']]
                out_prefix = f'_dSku_inDip_dDura_inSku_it{nb_iter}'

        ilab = dill_label_within(ilab, label_to_dill=label_to_dill, within_label=within_label, nb_iter=nb_iter)
        return ilab,out_prefix

    def dill_WM(ilab):
        rd_choice = torch.randint(0,4,(1,)).numpy()[0]
        if rd_choice>1 :
            return ilab, ''

        nb_iter = torch.randint(1,5,(1,)).numpy()[0]
        out_prefix = f'_dWM_inDGMHipltr_it{nb_iter}'
        label_to_dill = dic_map['WM']
        within_label = [dic_map['Thal'], dic_map['Pal'], dic_map['Put'], dic_map['Caud'],
                        dic_map['Accu'], dic_map['Hippo'], dic_map['Amyg'], ]
        ilab = dill_label_within(ilab, label_to_dill=label_to_dill, within_label=within_label, nb_iter=nb_iter)
        return ilab,out_prefix

    def dill_GM(ilab):
        rd_choice = torch.randint(0,6,(1,)).numpy()[0]
        nb_iter = torch.randint(1,5,(1,)).numpy()[0]
        if rd_choice==0:
            out_prefix = '_dGM_inCSF'
            label_to_dill = dic_map['GM']; within_label = [dic_map['CSF']]
        elif rd_choice==1:
            out_prefix = '_dGM_inWMCSF'
            label_to_dill = dic_map['GM']; within_label=[dic_map['WM'], dic_map['CSF']]
        elif rd_choice == 2:
            out_prefix = '_dGM_inWM'
            label_to_dill = dic_map['GM']; within_label=[dic_map['WM']]
        elif (rd_choice == 3) | (rd_choice == 4):
            st = generate_binary_structure(3, 2)
            nb_iter = torch.randint(6, 14, (1,)).numpy()[0]
            maskGM = (ilab.data == dic_map['GM']).numpy();
            maskWM = (ilab.data == dic_map['WM']).numpy()
            mask_dill = binary_dilation(maskWM[0], iterations=nb_iter, structure=st).astype(int);
            mask_dill = mask_dill[np.newaxis, ...]
            mask_dill[maskWM] = 0
            ilab.data[maskGM] = dic_map['CSF']

            GM_dill = binary_dilation(maskGM, iterations=2).astype(int)
            dill_ext = mask_dill * GM_dill  # - mask.astype(int)
            ilab.data[dill_ext > 0] = dic_map['GM']
            out_prefix = f'_dWMit{nb_iter}_asGM'
            return ilab, out_prefix
        else:
            return ilab, ''

        if nb_iter>2:
            within_label.append(dic_map['Arteries']);within_label.append(dic_map['Veins']);
            within_label.append(dic_map['Dura'])
            out_prefix = f'{out_prefix}Vas_Dura_it{nb_iter}'
        else:
            out_prefix = f'{out_prefix}_it{nb_iter}'
        ilab = dill_label_within(ilab, label_to_dill=label_to_dill, within_label=within_label, nb_iter=nb_iter)
        return ilab,out_prefix

    ilab,out_prefix = dill_GM(ilab)
    ilab,out_prefix2 = dill_WM(ilab)
    out_prefix = out_prefix + '_' + out_prefix2
    return ilab, out_prefix


def generate_one_suj(label_file, label_csv, fout, transfo_list, DO_Erode=True):
    print(f'computing synth image {fout}')
    foimg, folab_bin, folab_4D = (addprefixtofilenames(fout,'Sim_')[0], addprefixtofilenames(fout,'Lab_')[0], addprefixtofilenames(fout,'4DLab_')[0])

    if isinstance(label_csv, str):
        df = pd.read_csv(label_csv)
        dic_lab = {ll['Name']:ll['synth'] for ii,ll in df.iterrows()}
        dic_map_synth = {ll['synth']:ll['synth_tissu'] for ii,ll in df.iterrows()}
        dic_map_target = {ll['synth']:ll['synth_target'] for ii,ll in df.iterrows()}
    else:
        dic_lab, dic_map_synth, dic_map_target = label_csv
    ilabel = tio.LabelMap(label_file)
    if DO_Erode:
        ilabel, out_prefix = add_some_dill(ilabel, dic_lab)
    else:
        out_prefix = ''
    suj = tio.Subject(label=ilabel) #, label_target = i_lab_target )
    suj_aff = transfo_list[0](suj) #only spatial transfo
    tmapSynth, tmapTarget = tio.RemapLabels(dic_map_synth), tio.RemapLabels(dic_map_target)
    suj_synth = tmapSynth(suj_aff)
    print(f'input max s{suj_aff.label.data.max()} ')
    i_lab_target = tio.LabelMap(tensor=suj_aff.label.data, affine=suj_aff.label.affine)
    i_lab_target = tmapTarget(i_lab_target)

    sujt = transfo_list[1](suj_synth)  #randomLabel2Image
    print(f'After r2i {sujt.t1.data.shape}  ')

    suj_label_bin, suj_label_4D = pool_remap(i_lab_target, pooling_size=3, ensure_multiple=6, tmap=None, islabel=True)

    img_t1 = pool_remap(sujt.t1, pooling_size=3, ensure_multiple=6, tmap=None, islabel=False)
    img_t1 = tio.ScalarImage(tensor=img_t1.data, affine=img_t1.affine)
    sujt_down = tio.Subject(t1=img_t1)

    print(f'suj sujt_down is {sujt_down.t1} data {sujt_down.t1.data.shape}')
    sujt_down = transfo_list[2](sujt_down)  #Intensities transfo need a suj with key t1 for motion metrics .... bof ....

    info1, info2 = {}, {}
    record_history(info1,sujt)
    record_history(info2,sujt_down)
    transfo_name = get_transfo_short_name(info1['transfo_order']) + get_transfo_short_name(info2['transfo_order'])
    folab_bin, folab_4D = folab_bin[:-7] + out_prefix + transfo_name + '.nii.gz', folab_4D[:-7] + out_prefix + transfo_name + '.nii.gz'
    foimg = foimg[:-7] + out_prefix + transfo_name + '.nii.gz'
    suj_label_bin.save(folab_bin);
    suj_label_4D.save(folab_4D);
    sujt_down.t1.save(foimg)
    focsv =  foimg[:-7] + '.csv'
    info1.pop('history')
    info2.pop('history')
    with open(focsv,'w') as f:
        w = csv.writer(f)
        w.writerows(info1.items())
        w.writerows(info2.items())

#Synth generation
#dirvas = '/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/'
#sujdir = gdir(dirvas,['(AR$|SO$)','synth2'])
#dirout = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/Vasular1/'
#fins = gfile(sujdir,'^r025_s05')

# 2025/05  change from vascular synth v2 (name for vessel)

sujdir = ['/network/iss/opendata/data/template/MIDA_v1.0/MIDA_v1_voxels/mida_all/']
dirout = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/MidaS1_all/'
label_csv = '/network/iss/opendata/data/template/remap/my_synth/mida_labels.csv'

df = pd.read_csv(label_csv, comment='#')
dic_lab = {ll['Name']: ll['value'] for ii, ll in df.iterrows()}
dic_map_synth = {ll['value']: ll['synth_tissue_29'] for ii, ll in df.iterrows()}
dic_map_target = {ll['value']: ll['value'] for ii, ll in df.iterrows()}
label_csv = (dic_lab, dic_map_synth, dic_map_target)

transfo_list = [ get_transform_from_json(dirout + 'transform_Spatial.json') ,
                 get_transform_from_json(dirout + 'transform_lab2img.json'),
                 get_transform_from_json(dirout + 'transform_2.json')]
DO_Erode=False
if DO_Erode:
    fins = gfile(sujdir,'^csf_veine_r025s05')
else:
    fins = gfile(sujdir,'^csf_veine_r025s05_mida')

tmp_name = get_parent_path(tempfile.TemporaryDirectory().name)[1]
output_dir = dirout + f'/generate_{tmp_name}/'
os.mkdir(output_dir)
nb_example_per_subject = 9

for ngen in range(nb_example_per_subject):
    print(f'Pass {ngen+1}')
    for k, label_file  in enumerate(fins ) :
        sujname = f'Suj_{k:02}'
        fout = output_dir + f'gen{ngen+100:03}_{sujname}.nii.gz'
        generate_one_suj(label_file, label_csv, fout, transfo_list, DO_Erode=DO_Erode)


TEST = False
if TEST:
    from script.export_nnunet import create_nnunet_dataset_from_nii

    #regroup all generated data in one folder  #just change the generation number in file name
    suj = gdir(dirout,'gene')
    dout1, dout2 = gdir(dirout,'synth_bin')[0], gdir(dirout,'synth_4D')[0]
    gen_ind = 0
    for k,dirgen in enumerate(suj):
        for ii in range(9): #because 3 gen per subject
            f = gfile(dirgen, f'^[SL].*gen10{ii}')
            if len(f)==9:
                continue
            fname = get_parent_path(f)[1]
            fnew = [ f'{dout1}/{ff[:7]}{gen_ind:03}{ff[10:]}' for ff in fname]
            r_move_file(f, fnew, type='move')

            f = gfile(dirgen, f'^[4d].*gen10{ii}')
            fname = get_parent_path(f)[1]
            fnew = [ f'{dout2}/{ff[:9]}{gen_ind:03}{ff[12:]}' for ff in fname]
            #print(fnew)
            r_move_file(f, fnew, type='move')
            gen_ind+=1

    fimg, flab = gfile(dout1,'Sim.*gz'), gfile(dout1,'^Lab.*gz')
    df = pd.read_csv('/network/iss/opendata/data/template/remap/my_synth/mida_labels.csv',comment='#')

    ## create with remap to manu reg
    dic_map = { ll['value']:ll['synth_targetRegionFew'] for ii,ll in df.iterrows()}
    label_dic = {ll['NameTargetRegionFew']:ll['synth_targetRegionFew'] for ii,ll in df.iterrows()}
    label_dic = {k: v for k, v in sorted(label_dic.items(), key=lambda item: item[1])}
    tmap = tio.RemapLabels(dic_map)
    dataset_name, dnnunet_root = 'Dataset714_MidaSuj3_RFew', '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/'
    dataset_name, dnnunet_root = 'Dataset713_MidaSuj1', '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/'

    create_nnunet_dataset_from_nii(fimg, flab,label_dic,dataset_name, dnnunet_root, base_name = 'RRR',
                                   lab_one_hot=False, tmap_lab = tmap)
    # nnUNetv2_plan_and_preprocess  -c 3d_fullres -d 714 --verify_dataset_integrity
    # nnUNetv2_plan_experiment -d 714 -pl nnUNetPlannerResEncXL

    #remap DS708 to same region as 712
    df = pd.read_csv('/network/iss/opendata/data/template/remap/my_synth/brain_and_skull_and_head_Ultra_v2_label_DS708.csv')
    dic_map = { ll['synth_target']:ll['synth_region_few'] for ii,ll in df.dropna().iterrows()}
    label_dic = {ll['Name_region_few']:ll['synth_region_few'] for ii,ll in df.dropna().iterrows()}
    label_dic = {k: v for k, v in sorted(label_dic.items(), key=lambda item: item[1])}
    tmap = tio.RemapLabels(dic_map)

    ds708 = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Dataset708_Ultra_SkulVasc40/'
    ds712 = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Dataset712_Vasc2suj_v3_Few/'
    ds714 = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/Dataset714_MidaSuj3_RFew//'
    fimg, flab = gfile(ds708+'imagesTr','.*gz'), gfile(ds708+'labelsTr','.*gz')
    dataset_name, dnnunet_root = 'Dataset333_tmp', '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/'
    create_nnunet_dataset_from_nii(fimg, flab,label_dic,dataset_name, dnnunet_root, base_name = 'RRR',
                                   lab_one_hot=False, tmap_lab = tmap)
    # mv Dataset333_tmp/labelsTr Dataset708_Ultra_SkulVasc40/labelsTr_few
    f1,l1 = gfile(ds708+'imagesTr','.*gz'), gfile(ds708+'labelsTr_few','.*gz')
    f2,l2 = gfile(ds712+'imagesTr','.*gz'), gfile(ds712+'labelsTr','.*gz')
    f3,l3 = gfile(ds714+'imagesTr','.*gz'), gfile(ds714+'labelsTr','.*gz')
    fimg , flab = f1[:800] + f2[:800] + f3[:800], l1[:800] + l2[:800] + l3[:800]
    dataset_name, dnnunet_root = 'Dataset715_MixSuj6', '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/'
    create_nnunet_dataset_from_nii(fimg, flab,label_dic,dataset_name, dnnunet_root, base_name = 'RRR')

