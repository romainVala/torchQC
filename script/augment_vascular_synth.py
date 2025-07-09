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
    def dill_WM(ilab):
        rd_choice = torch.randint(0,4,(1,)).numpy()[0]
        if rd_choice>1 :
            return ilab, ''

        nb_iter = torch.randint(1,5,(1,)).numpy()[0]
        out_prefix = f'_dWM_inDGMHipltr_it{nb_iter}'
        label_to_dill = dic_map['WM']
        within_label = [dic_map['Thal'], dic_map['Pal'], dic_map['Put'], dic_map['Caud'],
                        dic_map['Accu'], dic_map['Hyp'], dic_map['Amyg'], ]
        #v2                         dic_map['Accu'], dic_map['Hippo'], dic_map['Amyg'], ]
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
            within_label.append(dic_map['vascular_brain']); within_label.append(dic_map['Dura'])
            #v2 within_label.append(dic_map['Vas_brain']); within_label.append(dic_map['Dura'])

            out_prefix = f'{out_prefix}Vas_Dura_it{nb_iter}'
        else:
            out_prefix = f'{out_prefix}_it{nb_iter}'
        print(f' ADDING some DILL prefix {out_prefix}')
        ilab = dill_label_within(ilab, label_to_dill=label_to_dill, within_label=within_label, nb_iter=nb_iter)
        return ilab,out_prefix

    ilab,out_prefix = dill_GM(ilab)
    ilab,out_prefix2 = dill_WM(ilab)
    out_prefix = out_prefix + '_' + out_prefix2
    return ilab, out_prefix

def reinsert_vessel_dura(ilhigh, filow, fref, flab, fout=None ):
    df = pd.read_csv(flab)
    dic_lab = {ll['Name']:ll['Value'] for ii,ll in df.iterrows()}

    tr = tio.Resample(target=fref)
    #vessel
    illow = tio.LabelMap(filow)
    illow['data'] = (illow.data == dic_lab['Vas_brain'] ).to(int) #vascular brain
    ilt = tr(illow)
    #inter = ((ilhigh.data==dic_lab['Skull']) | (ilhigh.data== dic_lab['Skull_dipl']) |
    #         (ilhigh.data==dic_lab['Dura'])) & (ilt.data>0.5)  #do not eras skull and dura
    #ilhigh.data[(ilt.data>0.5) & (inter==0)] = dic_lab['Vas_brain']
    #more restriction reinsert only pixel in csf
    ilhigh.data[(ilt.data > 0.5) & (ilhigh.data == dic_lab['CSF'])] = dic_lab['Vas_brain']
    #durra matter
    illow = tio.LabelMap(filow)
    illow['data'] = (illow.data==dic_lab['Dura']).to(int) #dura matter
    ilt = tr(illow)
    ilhigh.data[(ilt.data>0.5) & (ilhigh.data == dic_lab['CSF']) ] = dic_lab['Dura'] #only replace if csf

    if fout is not None:
        ilhigh.save(fout)

    return ilhigh

def generate_one_suj(label_file, label_csv, fout, transfo_list):
    print(f'computing synth image {fout}')
    foimg, folab_bin, folab_4D = (addprefixtofilenames(fout,'Sim_')[0], addprefixtofilenames(fout,'Lab_')[0], addprefixtofilenames(fout,'4DLab_')[0])

    df = pd.read_csv(label_csv)
    dic_lab = {ll['Name']:ll['synth'] for ii,ll in df.iterrows()}
    dic_map_synth = {ll['synth']:ll['synth_tissu'] for ii,ll in df.iterrows()}
    dic_map_target = {ll['synth']:ll['synth_target'] for ii,ll in df.iterrows()}
    ilabel = tio.LabelMap(label_file)

    ilabel, out_prefix = add_some_dill(ilabel, dic_lab)
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

def add_Ass_label(i_synth, name_collumn = 'Name_continuous', label_collumn='label_continuous',
                  replace_collumn = 'Replace_continuous'):
    dirlab = '/network/iss/opendata/data/template/remap/my_synth'
    dfsk = pd.read_csv( gfile(dirlab,'Svas_synth_v3.csv')[0] )
    dic_synth  = {ff.Name : ff.synth for ii,ff in dfsk.iterrows()}

    add_lab = ['cereb','hip','yeb','gm']
    lab_replaced = ['lGM','rGM', 'Cereb', 'WMCereb', 'Hyp','Thal', 'yeb_other']

    new_dic = {}
    for ii,ff in dfsk.iterrows():
        if ff.Name not in lab_replaced:
            new_dic[ff.Name] = ff.synth


    label_file = str(i_synth.path) #fins[0]
    dir_ass = gdir(get_parent_path(label_file,2)[0], 'Ass')
    #i_synth = tio.LabelMap(label_file)

    ind_start = dic_synth[max(dic_synth, key=dic_synth.get)] +1

    for lab in add_lab:
        lab_add = gfile(dir_ass, f'^r025s05_enlarge.*{lab}')[0]
        df_add = pd.read_csv( gfile(dirlab, f'^label.*{lab}')[0]  )
        dic_add = {ff.get(name_collumn) : ff.get(label_collumn) for ii,ff in df_add.iterrows()} #csv collumn label_LR
        dic_remap = {}
        for ii, (kk,vv) in enumerate(dic_add.items()):
            new_value = ind_start + ii
            dic_remap[vv] = new_value
            dic_add[kk] = new_value

        tmap = tio.RemapLabels(dic_remap)
        i_add = tmap( tio.LabelMap(lab_add) )

        list_replace_lab = df_add[replace_collumn].unique() #csv collumn Replace
        if "GM" in list_replace_lab : #should be lGM and rGM  keep FIX
            list_replace_lab = ['lGM','rGM']

        mask_add = i_synth.data==dic_synth[list_replace_lab[0]]
        for ii in range(1,len(list_replace_lab)):
            mask_add = mask_add | ( i_synth.data==dic_synth[list_replace_lab[ii]] )

        if i_add.data[mask_add].all() == 0:
            print(f'non assign value for {lab}')
            argggg

        i_synth.data[mask_add] = i_add.data[mask_add]
        print(f'replaced label {list_replace_lab} with  {dic_add}')
        new_dic.update(dic_add)
        ind_start = new_dic[max(new_dic, key=new_dic.get)] +1


    #continuous labels

    dic_remap = {}
    for ii, (kk,vv) in enumerate(new_dic.items()):
        dic_remap[vv] = ii
        new_dic[kk] = ii

    tmap = tio.RemapLabels(dic_remap)
    i_synth = tmap(i_synth)
    return i_synth


def generate_and_insert_one_suj(label_file, label_csv, fout, transfo_list):
    print(f'computing synth image {fout}')
    foimg, folab_bin, folab_4D = (addprefixtofilenames(fout,'Sim_')[0], addprefixtofilenames(fout,'Lab_')[0], addprefixtofilenames(fout,'4DLab_')[0])

    df = pd.read_csv(label_csv)
    #dic_lab = {ll['Name']:ll['synth'] for ii,ll in df.iterrows()}
    dic_map_synth = {ll['synth']:ll['synth_tissu'] for ii,ll in df.iterrows()}
    dic_map_target = {ll['synth']:ll['synth_target'] for ii,ll in df.iterrows()}
    ilabel = tio.LabelMap(label_file)
    #tttt tmap for GM
    dirlab = '/network/iss/opendata/data/template/remap/my_synth'
    dfsk = pd.read_csv( gfile(dirlab,'Svas_synth_v3.csv')[0] )

    dic_lab_before_add = {ff.Name : ff.synth for ii,ff in dfsk.iterrows()}
    tmap = tio.RemapLabels({dic_lab_before_add['lGM']:1,dic_lab_before_add['rGM']:1 })
    ilabel = tmap(ilabel)  #concatenate left and right GM
    dic_lab_before_add['GM'] = 1

    ilabel, out_prefix = add_some_dill(ilabel, dic_lab_before_add)
    ilabel = add_Ass_label(ilabel)  #extend label with ass and yeb


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
    ###NO4D .hmmm   suj_label_4D.save(folab_4D);
    sujt_down.t1.save(foimg)
    focsv =  foimg[:-7] + '.csv'
    info1.pop('history')
    info2.pop('history')
    with open(focsv,'w') as f:
        w = csv.writer(f)
        w.writerows(info1.items())
        w.writerows(info2.items())

#Synth generation v3
dirvas = '/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/'
sujdir = gdir(dirvas,['(AR$|SO$)','synth_v3'])
sujdir = gdir(dirvas,['(AR$)','synth_v3'])
dirout = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/Vascular3/'

label_csv = '/network/iss/opendata/data/template/remap/my_synth/brain_and_skull_and_Ass_vascular_v3_label_DS709.csv'

transfo_list = [ get_transform_from_json(dirout + 'transform_Spatial.json') ,
                 get_transform_from_json(dirout + 'transform_lab2img.json'),
                 get_transform_from_json(dirout + 'transform_2.json')]
fins = gfile(sujdir,'^r025_s05')

tmp_name = get_parent_path(tempfile.TemporaryDirectory().name)[1]
output_dir = dirout + f'/generate_{tmp_name}/'
os.mkdir(output_dir)
nb_example_per_subject = 10
add_lab = ['cereb', 'hip', 'yeb', 'gm']

for ngen in range(nb_example_per_subject):
    print(f'Pass {ngen+1}')
    for k, label_file  in enumerate(fins ) :
        sujname = f'Suj_{k:02}'
        dir_ass = gdir(get_parent_path(label_file, 2)[0], 'Ass')
        for lab in add_lab:
            lab_add = gfile(dir_ass, f'^r025s05_enlarge.*{lab}')[0]

        fout = output_dir + f'gen{ngen+100:03}_{sujname}.nii.gz'
        generate_and_insert_one_suj(label_file, label_csv, fout, transfo_list)


TEST = False
if TEST:
    #regroup all generated data in one folder  #just change the generation number in file name
    suj = gdir(dirout,'gene')
    dout1, dout2 = gdir(dirout,'synth_bin')[0], gdir(dirout,'synth_4D')[0]
    for k,dirgen in enumerate(suj):
        f = gfile(dirgen, '^[SL]')
        fname = get_parent_path(f)[1]
        fnew = [ f'{dout1}/{ff[:7]}{k:03}{ff[10:]}' for ff in fname]
        r_move_file(f, fnew, type='move')

        f = gfile(dirgen, '^[4d]')
        fname = get_parent_path(f)[1]
        fnew = [ f'{dout2}/{ff[:9]}{k:03}{ff[12:]}' for ff in fname]
        r_move_file(f, fnew, type='move')

    fimg, flab = gfile(dout1,'Sim.*gz'), gfile(dout1,'^Lab.*gz')
    df = pd.read_csv('/network/iss/opendata/data/template/remap/my_synth/brain_and_skull_and_Ass_vascular_v3_label.csv') #brain_and_skull_and_head_Ultra_v2_label.csv')
    label_dic = {ll['NameTarget']:ll['synth_target'] for ii,ll in df.iterrows()}
    dataset_name, dnnunet_root = 'Dataset709_Vasc2suj_v3', '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/'
    create_nnunet_dataset_from_nii(fimg, flab,label_dic,dataset_name, dnnunet_root, base_name = 'RRR',lab_one_hot=False)

    ## create with remap to manu reg
    dic_map = { ll['synth_target']:ll['synth_targetRegion'] for ii,ll in df.iterrows()}
    label_dic = {ll['NameTargetRegion']:ll['synth_targetRegion'] for ii,ll in df.iterrows()}
    label_dic = {k: v for k, v in sorted(label_dic.items(), key=lambda item: item[1])}
    tmap = tio.RemapLabels(dic_map)
    dataset_name, dnnunet_root = 'Dataset710_Vasc2suj_v3_Region', '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/'
    create_nnunet_dataset_from_nii(fimg, flab,label_dic,dataset_name, dnnunet_root, base_name = 'RRR',
                                   lab_one_hot=False, tmap_lab = tmap)


    # usample and smooth all labels
    dirvas = '/network/iss/cenir/analyse/irm/users/romain.valabregue/segment_RedNucleus/vascular_pc3D/preproc/'
    suj = gdir(dirvas, ['(AR$|SO$)', 'synth2'])
    fins = gfile(suj, '^SynthVa')
    frefs = gfile(get_parent_path(fins)[0], '^r025mrt')
    flabels = gfile(get_parent_path(fins)[0], '^label.*csv')
    fout = addprefixtofilenames(fins, 'r025_s05_rVesDur')

    # upsample to 0.25 and correct vessel and dura
    for fi, flab, fo, fref in zip(fins, flabels, fout, frefs):
        # break
        if os.path.isfile(fo):
            print(f'skiping file exist {fo}')
        else:
            print(f'Computing {fo}')
            ilt = resample_and_smooth4D(fi, fref, blur4D=0.5, skip_blur=None)
            ilhigh = reinsert_vessel_dura(ilt, fi, fref, flab, fout=fo)
            ilhigh.save(fo)

    dirsuj = [dirsuj1, dirsuj2]
    flab = gfile(dirsuj, '^repair_ves_dura')
    dill_label_within(flab, label_to_dill=1, within_label=[3], nb_iter=2, out_prefix='dill_inCSF_')
    dill_label_within(flab, label_to_dill=1, within_label=[2, 3], nb_iter=2, out_prefix='dill_inWMCSF_')
    dill_label_within(flab, label_to_dill=1, within_label=[2], nb_iter=2, out_prefix='dill_inWM_')

    fin = gfile(dirsuj1, '^dill_inCSF_it2_rep')
    il = tio.LabelMap(fin)

    from scipy.ndimage import generate_binary_structure
    st = generate_binary_structure(3, 2)
    maskGM = (il.data == 1).numpy();
    maskCSF = (il.data == 3).numpy()
    maskWM = (il.data == 2).numpy()
    mask_dill = binary_dilation(maskWM[0], iterations=4, structure=st).astype(int);
    mask_dill = mask_dill[np.newaxis, ...]
    dill_ext = mask_dill * maskGM  # - mask.astype(int)
    il.data[maskGM] = 3
    il.data[dill_ext > 0] = 1
    fout = addprefixtofilenames(fin, 'ST2testWMdilGM_')[0]
    il.save(fout)



