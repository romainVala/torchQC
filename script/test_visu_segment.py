import os, glob
from segmentation.config import Config
from segmentation.run_model import RunModel
from plot_dataset import PlotDataset
import matplotlib.pyplot as plt
plt.interactive(True)
from nibabel.viewers import OrthoSlicer3D as ov
import torchio as tio
import numpy as np
from nibabel.viewers import OrthoSlicer3D as ov
from scipy.ndimage import label as scipy_label
from scipy.ndimage.morphology import binary_dilation
import torch
fjson = '/data/romain/PVsynth/training/NN_regres_motion_New_train_random_synth/save_to_dir/main.json'
fjson = '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/ext_tool/eval_synth_seg_onSNclean/eval-1mm-SynthSeg_t1_GM_3_noise_01/split_main_0.json'
fjson = '/data/romain/PVsynth/jzay/training/baby/bin_dseg9_feta_bg_T1T2_mP192/main.json'
fjson = '/data/romain/PVsynth/jzay/training/baby/bin_dseg9_feta_bg_shape/main.json'

result_dir= 'results_cluster' #os.path.dirname(fjson) +'/test_script'
config = Config(fjson, result_dir, mode='eval', save_files=False) #since cluster read (and write) the same files, one need save_file false to avoid colusion
config.init()
viz_set = config.train_set #val_set
viz_set = config.val_set
np.random.seed(12)
torch.manual_seed(12)
suj = viz_set[0]

for i in range(20):
    suj = viz_set[i]
    suj.t1.save(f'dat_n{i}.nii.gz')
    #suj.label.data[suj.label.data<0.01] = 0
    suj.label.save(f'lab_n{i}.nii.gz')
    #suj.label_synth.save(f'labsynth_n{i}.nii.gz')


#read saved sample and select those with clean WM (and make a link)
fjson = '/linkhome/rech/gencme01/urd29wg/training/baby/pve_15suj/scaled_from_dir/bgEM_wm_all/main.json'
result_dir= 'results_cluster' #os.path.dirname(fjson) +'/test_script'
config = Config(fjson, result_dir, mode='eval', save_files=False) #since cluster read (and write) the same files, one need save_file false to avoid colusion
config.init()
trainset_list = config.train_set

for i in range(27):
    trainset_list.pop(0)

#first de 0 a 30  second de 20 a 29  third de 14 a 19
for nb_epoch in range(0,len(trainset_list)):
    print(nb_epoch)

    viz_set = trainset_list[nb_epoch]
    nb_epoch+=28
    print(nb_epoch)

    for ind_suj, suj in enumerate(viz_set):
            print(f'epoch {nb_epoch} suj {ind_suj} ')
            dd=suj.label_synth.data[suj.label.data==3]
            #if dd.max()==dd.min():
            if len(dd[dd>3])==0:
                print('make link')
                do = f"select_WMclena_epo{nb_epoch:03}/"
                #if ind_suj==0:  #bug if skip first
                if not( os.path.isdir(do) ):
                    os.mkdir(do)

                suj_tar = viz_set._subjects[ind_suj]
                filename = os.path.basename(suj_tar)
                filename_dir = os.path.basename(os.path.dirname(suj_tar))
                os.symlink(f'../{filename_dir}/{filename}', do+filename)
                #os.symlink(suj_tar, do+filename)
            else:
                print('skiping')

tc = tio.CropOrPad(target_shape=[192,192,192])
tc = tio.CropOrPad(target_shape=[224,304,304])
cd '/data/romain/PVsynth/jzay/training/baby/bin_dseg9_feta_bg_T1T2_mP192'
suj = tc(suj)
suj.t1.save('t1s.nii')
suj.t1w.save('t1w.nii')
suj.t2w.save('t2w.nii')

model, device = config.load_model(config.model_structure)
model.eval()
sofm = torch.nn.Softmax()

data = tc(suj.t1w.data).unsqueeze(0).to('cuda')
with torch.no_grad():
    pred = sofm(model(data))

suj.label.data = pred[0].cpu().detach().numpy()
suj.label.save('predT1.nii')


data=torch.rand([16,16,16])
img = tio.ScalarImage(tensor=data.unsqueeze(0))
tre = tio.Resize(suj.t1.shape[1:])
imgr = tre(img)

tl = config.train_loader
i=0
for bl in tl:
    if i>0:
        break
    i+=1


#second pas remove GM pv from denomalize roi (center and CC) relabel WM
for kk, suj in enumerate(viz_set):
    #keep only biggest component of GM AND add hyppo + Amyg to GM
    pvgm = suj.label.data[1] ; pvwm = suj.label.data[0] ; pvcsf = suj.label.data[2]
    pvcc = suj.extra.data[0]
    ind_remove = (pvcc>0.5) & (pvgm > 0.01)  #0.5 because fsl reslice

    pvwm[ind_remove] = pvwm[ind_remove] + pvgm[ind_remove]
    pvgm[ind_remove]=0;

    maskgm = pvgm * (pvgm>0.25)
    components, n_components = scipy_label(maskgm, None)
    n_count = np.bincount(components.flat)[1:]
    remove_component = torch.zeros_like(pvgm)

    if (np.argmax(n_count)==0) & ( len(n_count)> 1 ):
        for i in range(2,n_components+1):
            remove_component[components==i] = 1
            remove_component = remove_component > 0
            pvwm[remove_component] = pvwm[remove_component] + pvgm[remove_component]
            pvgm[remove_component] = 0;
    else:
        if len(n_count)<1:
            raise ('should not happen')

    if (len(n_count)==1) & (ind_remove.sum()<10) :
        print(f'skiping {kk}')
    else:
        label_dir =  os.path.dirname( suj.label.path[0] )
        suj.t1aa.data[0] = pvgm ; suj.t1aa.save(label_dir + '/GM_allc.nii.gz')
        suj.t1aa.data[0] = pvwm ; suj.t1aa.save(label_dir + '/WM.nii.gz')


    all = suj.label.data
    all[0] = pvwm
    all[1] = pvgm

    sum_min = all.sum(axis=0).min(); sum_max = all.sum(axis=0).max()
    print(f'{sum_min}  {sum_max}')
    if sum_min<0.9 or sum_max>1.1:
        raise('what the fuck')
    if kk%20:
        print(kk)

#first pass with /data/romain/PVsynth/example_data_json/data_synth0.json
for kk, suj in enumerate(viz_set):
    #keep only biggest component of GM AND add hyppo + Amyg to GM
    pvgm = suj.label.data[1] + suj.label.data[6] + suj.label.data[8]; #gm + amyg + hippo
    pvwm = suj.label.data[0] ; pvcsf = suj.label.data[2]
    maskgm = pvgm * (pvgm>0.25)
    components, n_components = scipy_label(maskgm, None)
    bigest_comp = components == np.argmax(np.bincount(components.flat)[1:]) + 1 if n_components > 0 else maskgm.copy()
    maskgm_clean = bigest_comp*maskgm.numpy()
    maskgm_dill = binary_dilation(maskgm_clean)
    new_gm = pvgm * maskgm_dill

    ind_remove = ( (pvgm -new_gm) > 0 )
    replace_by_wm  = ind_remove & (pvwm >= pvcsf)
    replace_by_csf = ind_remove & (pvwm < pvcsf)

    pvwm[replace_by_wm] = pvwm[replace_by_wm]  + pvgm[replace_by_wm]
    pvcsf[replace_by_csf] = pvcsf[replace_by_csf]  + pvgm[replace_by_csf]

    #merge brainsteam and WM an cerebWM
    pvbs = suj.label.data[12];    pvsn = suj.label.data[3];    pvred = suj.label.data[4];
    pvtha = suj.label.data[11];pvput = suj.label.data[10];
    pvcerebwm =  suj.label.data[14]; pvcerebgm =  suj.label.data[13]

    pvbsdil = torch.tensor(binary_dilation(pvbs * (pvbs > 0.001), iterations=2))
    pvcerebwm_dill = torch.tensor(binary_dilation(pvcerebwm * (pvcerebwm > 0.001), iterations=2))
    pvwmdil = torch.tensor(binary_dilation(pvwm * (pvwm > 0.001), iterations=2))

    intersect1 = pvbsdil & pvwmdil & (pvtha<0.001) & (new_gm<0.001) & (pvcerebgm<0.001) #& (pvput<0.001)
    intersect2 = pvcerebwm_dill & pvwmdil & (new_gm<0.001) & (pvcerebgm<0.001) #& (pvput<0.001)
    maskinter = torch.zeros_like(pvgm)
    maskinter[intersect1] = 1
    maskinter[intersect2] = 1

    new_wm = pvwm + pvbs + maskinter + pvsn + pvred + pvcerebwm
    new_wm[new_wm>1] = 1
    new_wm = new_wm - pvsn - pvred

    #reasign add voxel to
    all = suj.label.data
    keep_tissu = ((torch.arange(all.shape[0]) != 6) & (torch.arange(all.shape[0]) != 8) &
                  (torch.arange(all.shape[0]) != 12) & (torch.arange(all.shape[0]) != 14))  # remove hy amyg brainsteam wm_cereb
    all = all[keep_tissu]
    all[0] = new_wm
    all[1] = new_gm
    all[2] = pvcsf

    maskinter = torch.zeros_like(pvgm)
    maskinter[all.sum(axis=0)>1.0001] = 1
    pvcsf[maskinter>0] = 1-new_wm[maskinter>0]

    all[2] = pvcsf

    #save results
    label_dir =  os.path.dirname( suj.label.path[0] )
    suj.t1aa.data[0] = new_gm ; suj.t1aa.save(label_dir + '/GM_allc.nii.gz')
    suj.t1aa.data[0] = new_wm ; suj.t1aa.save(label_dir + '/WM.nii.gz')
    suj.t1aa.data[0] = pvcsf ; suj.t1aa.save(label_dir + '/CSF.nii.gz')


    maskinter = torch.zeros_like(pvgm)
    maskinter[all.sum(axis=0)>1.0001] = 1

    sum_min = all.sum(axis=0).min(); sum_max = all.sum(axis=0).max()
    print(f'{sum_min}  {sum_max}')
    if sum_min<0.9 or sum_max>1.1:
        raise('what the fuck')
    if kk%20:
        print(kk)






suj.t1aa.data[0] = torch.tensor(new_gm)







viz_set = config.val_set
viz_arg = config.viz_structure['kwargs']
viz_arg['image_key_name'] = config.image_key_name
viz_arg['label_key_name'] = config.label_key_name

PlotDataset(viz_set, **viz_arg)



mr = config.get_runner()

s1 = config.val_subjects
s1[0].t1.path
s1[0].name

suj = next(iter(viz_set))



suj = config.val_subjects[0]  #get the subject but no transfo apply
suj  = next(iter(config.val_set)) #to get the transfo applied

ds, ts = config.config_structure['data_json'],config.config_structure['transfo_json']
ts['val_transforms'][1]['attributes']['percentiles']=[0,95]
config.update_data_load(ds,ts )



## test mida
fjson='/home/romain.valabregue/datal/PVsynth/jzay/training/RES1mm_prob/pve_synth_mod3_P128_mida/maintest.json'
result_dir= 'results_cluster' #os.path.dirname(fjson) +'/test_script'
config = Config(fjson, result_dir, mode='eval', save_files=False) #since cluster read (and write) the same files, one need save_file false to avoid colusion
config.init()

viz_set = config.train_set
np.random.seed(1)
torch.manual_seed(1)

suj = viz_set[0]

#mida to std
suj = tio.Subject({"t1": tio.ScalarImage('/data/romain/template/MIDA_v1.0/MIDA_v1_voxels/mida_merge.nii')})
ta = tio.ToCanonical()
sujt = ta(suj);
aff = np.eye(4)*0.5; aff[-1]=1; aff[:3,3]=[-80, -100, -100]
sujt.t1.affine=aff
sujt.t1.save('mida_merge_std.nii')

ta = tio.Crop((1,0,0,0,151,0))
sujtc = ta(sujt)
sujtc.t1.save('crop_mida_merge_std.nii')


suj.t1.save('t1_orig.nii')
suj.label.save('lab_orig.nii')
tr = tio.Resample(target='/data/romain/template/MIDA_v1.0/MIDA_v1_voxels/r1mm_crop_mida_merge_std.nii')
sujT = tr(suj)
sujT.t1.save(f'rt1_orig{i}.nii')
sujT.label.save(f'rlab_orig.nii')

for i in range(5):
    suj = viz_set[0]
    suj.t1.save(f'data_rdLabel_{i}.nii')
    suj.label.save(f'label_rdLabel_{i}.nii')

#affine
taff = tio.RandomAffine(scales=0.1, degrees=20, translation=10)
tela = tio.RandomElasticDeformation()
tr = tio.Resample(target='/data/romain/template/MIDA_v1.0/MIDA_v1_voxels/r1mm_crop_mida_merge_std.nii')
tall = tio.Compose( [taff, tela, tr]); transfo_name = 'AffElaRes'
tall = tio.Compose( [tr, taff, tela]); transfo_name = 'ResAffEla'
torch.manual_seed(1)
for i in range(1):
    sujT = tall(suj)
    sujT.t1.save(f'data_{transfo_name}_{i}.nii')
    sujT.label.save(f'label_{transfo_name}_{i}.nii')

hist = sujT.history
tall = tio.Compose([hist[3],hist[1],hist[2]]);  transfo_name = 'ResAffEla'

######## explore label sampler
fjson = '/data/romain/PVsynth/jzay/training/baby/bin_dseg9_feta_bg_midaMotion/main.json'
result_dir= 'rrr' #os.path.dirname(fjson) +'/test_script'
config = Config(fjson, result_dir, mode='eval', save_files=False) #since cluster read (and write) the same files, one need save_file false to avoid colusion
config.init()
viz_set = config.train_set #val_set
suj = viz_set[0]
all_patch = torch.zeros_like(suj.t1.data)
tloader = config.train_loader
for i, sample in enumerate(tloader):
    print(f'{i} suj_name = {sample["name"]}')

    patch_select = torch.zeros_like(suj.t1.data)
    location = [ int(k) for k in sample['location'][0] ]
    patch_select[0,location[0]:location[3],location[1]:location[4],location[2]:location[5]] = 1;
    all_patch += patch_select

suj.t1.save('t1.nii')
suj.label.save('l.nii')

suj.t1.data = all_patch
suj.t1.save('patchRand.nii')



#for reading data from disk
import torchio, glob
from torch.utils.data import DataLoader

def get_queue_loader_from_arg(arg_dict):
    #add for data set with a lot of files, (previously list of dataloader ... with load_from_dir)
    sample_files = eval(arg_dict['subjects'])
    train_set =  torchio.SubjectsDataset(sample_files,
                                      load_from_dir=arg_dict['load_from_dir'],
                                      transform=arg_dict['transform'],
                                      add_to_load=arg_dict['add_to_load'],
                                      add_to_load_regexp=arg_dict['add_to_load_regexp'])

    train_queue = torchio.Queue(train_set,  start_background=True,
                                **arg_dict['queue_param'] )

    dataloader = DataLoader(train_queue, self.batch_size, num_workers=0,
                                   collate_fn=arg_dict['collate_fn'])

    return dataloader

def get_dataset_from_arg(arg_dict, batch_size=1):
    #add for data set with a lot of files, (previously list of dataloader ... with load_from_dir)
    sample_files = eval(arg_dict['subjects'])
    train_set =  torchio.SubjectsDataset(sample_files,
                                      load_from_dir=arg_dict['load_from_dir'],
                                      transform=arg_dict['transform'],
                                      add_to_load=arg_dict['add_to_load'],
                                      add_to_load_regexp=arg_dict['add_to_load_regexp'])

    #dataloader = DataLoader(train_set, batch_size, num_workers=0)
                                   #collate_fn=arg_dict['collate_fn'])

    return train_set #dataloader

viz_set = config.train_set #val_set
param = viz_set[0]
ds = get_dataset_from_arg(param)
suj = ds[0]

sample_files = glob.glob('/data/romain/training_saved_sample/baby_15_suj/no_scaling/drawEM_15suj_bgmida_tSTD_mot05/tio_save_mida_wmclean_tSTD_mot05strong_BG0_15suj/ep_000/*gz')
#let make it directly
sample_files = glob.glob('/data/romain/training_saved_sample/speciesV3/tio_save_specieV3_clean_largeScale/ep_000/*dup0*gz')

DS = tio.SubjectsDataset(sample_files,load_from_dir=True, transform=None,add_to_load=None,add_to_load_regexp=None)
#suj = DS[0]
for suj in DS:
    suj.t1.save(f't1_{suj.name}.nii.gz')
    suj.label.save(f'lab_{suj.name}.nii.gz')
