
import torch,numpy as np,  torchio as tio
from utils_metrics import get_tio_data_loader, predic_segmentation, load_model, computes_all_metric
from timeit import default_timer as timer
import json, os, seaborn as sns
from utils_file import gfile, gdir, get_parent_path, addprefixtofilenames, r_move_file
import pandas as pd
from nibabel.viewers import OrthoSlicer3D as ov


sns.set_style("darkgrid")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

droot = '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/species/'
dout_pred = droot + 'pred/'; dout_data = droot + 'data/'; dout_label = droot + 'label/' ; dmod = droot + 'model/'
dout_res = droot + 'results/'
#remap predictions
def get_remap_from_csv(fin):
    df = pd.read_csv(fin, comment='#')
    #dic_map= { r[0]:r[1]  for i,r in df.iterrows()}
    dic_map={}
    for i,r in df.iterrows():
        if r[1]==r[1]: #remove nan
            dic_map[r[0]] = r[1]
    #remap_keys = set(dic_map.keys())
    return tio.RemapLabels(remapping=dic_map)

def remap_filelist(fin, tmap, prefix='remap_'):
    fout = addprefixtofilenames(fin, prefix)
    for fi, fo in zip(fin, fout):
        if os.path.isfile(fo):
            print(f'SKIP {fo}')
        else:
            il = tio.LabelMap(fi)
            ilt = tmap(il)
            ilt.save(fo)

def compute_metric_from_list(f1_list,f2_list,sujname_list, labels_name, selected_label=None, **kwargs):
    df = pd.DataFrame()
    thot = tio.OneHot(); tc = tio.ToCanonical() #hcp labels are with -1 but pred with +1

    for f1,f2, sujname  in zip(f1_list, f2_list, sujname_list):

        i1 = tc(tio.LabelMap(f1));      i2 = tc(tio.LabelMap(f2))
        #guess from target
        if selected_label is None:
            sel_labels = i2.data.unique().numpy().astype(int)#[1:]
            sel_name = labels_name[sel_labels.astype(int)]
        else:
            sel_labels = selected_label; sel_name = labels_name

        i1 = thot(i1);              i2 = thot(i2);

        prediction = i1.data.unsqueeze(0)
        target = i2.data.unsqueeze(0)

        df_one = computes_all_metric(prediction, target, sel_name, selected_label=sel_labels,**kwargs)
        df_one['sujname'] = sujname
        df_one['volume_pred'] = f1; df_one['volume_targ'] = f2

        df = pd.concat([df, df_one], ignore_index=True)

    return df


dfdata = pd.read_csv(droot + 'test_set3.csv')
df = dfdata.iloc[:8,:] #only bcat
### copy data label
fin = df.data.values; dout = dout_data
fin = df.label.values[-40:]; dout = dout_label
fo = []; second_extention_list = ['.nii', '.pt']
for ff,sujn in zip(fin,df.sujname[-40:]):
     rr1, ext1 = os.path.splitext(ff)
     rr2, ext2 = os.path.splitext(rr1)
     if ext2 in second_extention_list:
         ext = ext2 + ext1
     else:
         ext = ext1
     fo.append(dout + sujn + ext)
r_move_file(fin,fo)

### remap label,
# dhcp is needed)
ff = gfile(dout_label, '^dhc')
remap_filelist(ff,tmap_dHCP,prefix='re') #sans prefix cela remplace le fichier du lien :::!
#for i in dhcp*; do echo $i;  rm -f $i;  mv re$i $i; done
# ahcp 15 (nerve -> bg 16 vessel -> WM)
ff = gfile(dout_label,'ahcp')
tmap = tio.RemapLabels(remapping={15:0,16:2})
remap_filelist(ff,tmap,prefix='re') #sans prefix cela remplace le fichier du lien :::!
#for i in ahcp*; do echo $i;  rm -f $i;  mv re$i $i; done


### predict all with predict.py
model = gfile(gdir(dmod,'.*'),'tar')
model_name = get_parent_path(model,2)[1]
fcmd = open('cmd2.bash', 'w')

cmd_ini = "python /network/lustre/iss02/cenir/software/irm/toolbox_python/romain/torchQC/segmentation/predict.py -bc 1 2 " # -d cpu"
#cmd_ini = "python /data/romain/toolbox_python/romain/torchQC/segmentation/predict.py -bc 1 2 " # -d cpu"
for finnot, sujname in zip(df.data, df.sujname):
    fin = gfile(dout_data,sujname)[0]
    for mod,mod_nam in zip(model,model_name):
        dir_out = dout_pred + mod_nam
        fout = f'pred_{sujname}'
        if os.path.isfile(dir_out + '/bin_' + fout+'.nii.gz'):
            print(f'skip {dir_out} {fout}')
        else:
            fcmd.write(f'mkdir -p {dir_out} \n cd {dir_out}\n')
            fcmd.write(f'{cmd_ini} -v {fin} -m {mod} -f {fout} \n')
fcmd.close()
#remap
tmap_SN_clean = get_remap_from_csv('/network/lustre/iss02/opendata/data/template/remap/remap_model_SNclean_ep_40.csv')
tmap_dHCP = get_remap_from_csv('/network/lustre/iss02/opendata/data/template/remap/remap_model_dHCP.csv')
tmap_mmac = get_remap_from_csv('/network/lustre/iss02/opendata/data/template/remap/remap_model_mmac.csv')
tmap_spe_wmall = get_remap_from_csv('/network/lustre/iss02/opendata/data/template/remap/remap_model_species_wmall.csv')
tmap_list = [tmap_SN_clean, tmap_dHCP, tmap_mmac, None, tmap_spe_wmall, None]
model_pred = gdir(dout_pred,'.*')
for tmap,mod_dir in zip(tmap_list, model_pred):
    f = gfile(mod_dir, '^bin_pred')
    if tmap is not None:
        remap_filelist(f,tmap)
fi = gfile(model_pred[3],'^bin*'); fo = addprefixtofilenames(fi, 'remap_'); r_move_file(fi,fo)

#remap to brainmask
dic_map = {k:1 for k in range(14)}; dic_map[0] = 0 ;dic_map[3] = 0 ; dic_map[13] = 0 ; tmap = tio.RemapLabels(dic_map)
flab = gfile(dout_label,'^[abdm].*') ; remap_filelist(flab,tmap,'lab_brain/')
#idem pour les pred
for mod_dir in model_pred :
    flab = gfile(mod_dir, '^remap_bin_pred')
    print(len(flab))
    remap_filelist(flab,tmap,'lab_brain/')

### compute metric
df_label = pd.read_csv('/network/lustre/iss02/opendata/data/template/remap/speciesV3_label.csv')
labels_name = df_label.Name.values[:14]
sujn = dfdata.sujname[8:]
ftarg = [ gfile(dout_label,ss)[0] for ss in sujn.values]
ftarg = [ gfile(gdir(dout_label,'lab_brain'),ss)[0] for ss in sujn.values]
dfall = pd.DataFrame()
for dpred in gdir(dout_pred,'.*'):
    modeln = get_parent_path(dpred)[1]
    print(modeln)

    #fpred = [gfile(dpred,'^remap.*'+ ss)[0] for ss in sujn.values]
    fpred = [gfile(gdir(dpred,'lab_brain'),'^remap.*'+ ss)[0] for ss in sujn.values]

    df = compute_metric_from_list(fpred, ftarg,sujn, labels_name, volume_metric=True)
    df['model'] = modeln
    dfall = pd.concat([df,dfall])

dfall.to_csv(dout_res+'/res_brain_specieCC_dhcpScla_mmacCrop_cerebrum.csv')

### plot result
df = pd.read_csv(dout_res+'res_brain_specieCC_dhcpScla_mmacCrop_cerebrum.csv')

df = pd.read_csv(dout_res+'/res_brain_specieCC_dhcpScla_mmacCrop.csv')
df = pd.read_csv(dout_res+'/res_all_tissue_86_suj_specieCC_dhcpScla_mmacCrop.csv')
df2 = pd.read_csv(dout_res+'/res_all_tissue_86_suj_specieCC.csv')
df['group'] = 0
df.loc[df.sujname.str.find('mmac')==0,'group']='rhesus macaque'
df.loc[df.sujname.str.find('bca')==0,'group']='new species'
df.loc[df.sujname.str.find('dhcp_S0')==0,'group']='dhcp Young'
df.loc[df.sujname.str.find('dhcp_S8')==0,'group']='dhcp Old'
df.loc[df.sujname.str.find('ahcp')==0,'group']='adult hcp'

#small cheet, dHCP better perf at original resolution
df.loc[(df.group=='dhcp Old') & (df.model=="dHCP_bgmida_mot05_ep240"),'dice_GM'] = df2.loc[(df2.group=='dhcp Old') & (df2.model=="dHCP_bgmida_mot05_ep240"),'dice_GM']

df.model = df.model.str.replace('speciesV3_jzAff_fromep80_ep46','all_species')
df.model = df.model.str.replace('macac_5suj_wmclean','rhesus macaque')
df.model = df.model.str.replace('dHCP_bgmida_mot05_ep240','human newborn')
df.model = df.model.str.replace('adult_SN_clean_mot40','human adult')
model_order = ['all_species', 'rhesus macaque', 'human newborn', 'human adult']
df['test set'] = df["group"] ; df["model trained on"] = df["model"]
fig = sns.catplot(data=df, y='dice_GM', x='test set',hue='model trained on',  kind='boxen', hue_order=model_order) #,col_wrap=3)
ax = fig.axes[0][0]
ax.set_ylabel('Dice Brain Mask',fontsize='xx-large') #'Dice'   'Average Surface dist' 'Volume Ratio'
yy = ax.get_yticklabels(); ax.set_yticklabels(yy,fontsize='xx-large' )
xx = ax.get_xticklabels()
ax.set_xticklabels(xx, rotation=0,fontsize='xx-large' );ax.set_xlabel('Test Set',fontsize='xx-large')
sns.move_legend(fig,"lower right",bbox_to_anchor=(.5, 0.8),fontsize='xx-large',frameon=True, shadow=True, title=None)


#labels_name = df_label.Name.values[np.r_[1:4,6:13]] ; sel_label = df_label.targ.values[np.r_[1:4,6:13]] ;

suj =gdir('/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/session1/',['\d.*','T1w'])
f1 = gfile(suj,'remap_aparc.aseg.nii.gz')
f2 = gfile(gdir(suj,'ROI$'), 'bin_PV')
sujname = get_parent_path(f1,3)[1]

df = compute_metric_from_list(f1, f2, sujname, labels_name, sel_label)

df.to_csv('test_retest_aseg_versus_PV.csv')




#sanity check
def dice_ines(inputs, targets, smooth: float = 1e-7):
    # TODOC
    # TOTYPE
    dim = None if inputs.shape[1] == 1 else [2, 3, 4]

    intersection = (inputs * targets).sum(dim=dim)
    union = inputs.sum(dim=dim) + targets.sum(dim=dim)

    dices = torch.mean((2.0 * intersection + smooth) / (union + smooth), dim=0)

    return dices

dice_coeff = dice_ines(inputs=inputs, targets=targets, smooth=self.smooth).mean()
f11  ="/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/session2/200614/T1w/ROI_PVE_1mm/bin_PV_labels_V1.nii.gz"
f22 = "/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/session2/200614/T1w/ROI_PVE_1mm/remap_aseg_free07.nii.gz"
f1  = '/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/session2/103818/T1w/ROI_PVE_1mm/bin_PV_labels_V1.nii.gz'
f2 = '/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/session2/103818/T1w/ROI_PVE_1mm/remap_aseg_free07.nii.gz'

i1 = tio.LabelMap(f1);i2 = tio.LabelMap(f2);i1 = thot(i1);i2 = thot(i2);
i11 = tio.LabelMap(f11); i22 = tio.LabelMap(f22); i11 = thot(i11); i22 = thot(i22);

pred1 = i1.data.unsqueeze(0); tar1 = i2.data.unsqueeze(0) ; pred11 = i11.data.unsqueeze(0); tar11 = i22.data.unsqueeze(0)
pred = torch.cat([pred1,pred11]);targ = torch.cat([tar1,tar11])
df_one = computes_all_metric(pred, targ, labels_name, selected_label=selected_label)
prediction = pred[:, selected_label, ...]
target = targ[:, selected_label, ...]
dice_ines(prediction,target)

def mykdir(dir):
    if not os.path.isdir(dir) :
        os.makedirs(dir)


### select data
### select dhcp_data
dfhcp = pd.read_csv('/network/iss/opendata/data/baby/all_seesion_info_order_all.csv')
dfs = dfhcp[~dfhcp.PV.isna()]; dfs.index = range(len(dfs)); dfs = dfs.iloc[np.r_[:20,703-20:703],:]; dfs.index = range(len(dfs));
sujname = [ f'dhcp_S{d.sujnum:03}_{d.sujname}_T2' for dd,d in dfs.iterrows()]
df=pd.DataFrame()
df['sujname'] = sujname; df['label'] = dfs['vol_label']; df['data']=dfs.vol_T2 ;

## again in 2025 mai
dout_nn = '/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/'
#young
dfs = dfhcp[~dfhcp.PV.isna()]; dfs.index = range(len(dfs));
dfs1 = dfs.iloc[np.r_[:20],:]; dfs1.index = range(len(dfs1));
sujname = [ f'dhcp_S{d.sujnum:03}_{d.sujname}_T2' for dd,d in dfs1.iterrows()]
dout1 = os.path.join(dout_nn,'dHcp_young075_scale')
DSname = 'dHcp_young075_scale_volT2_GT'


#old
dfs1 = dfs.iloc[np.r_[703-20:703],:]; dfs1.index = range(len(dfs1));
sujname = [ f'dhcp_S{d.sujnum:03}_{d.sujname}_T2' for dd,d in dfs1.iterrows()]
dout1 = os.path.join(dout_nn,'dHcp_old075')
DSname = 'dHcp_old075_volT2_GT'
dout_imgT2 = os.path.join(dout1,'vol_T2');dout_imgT1 = os.path.join(dout1,'vol_T1');dout_lab = os.path.join(dout1,'label_T2')
mykdir(dout_imgT2); mykdir(dout_imgT1); mykdir(dout_lab)
Hvol,new_res, scale_res = [], [], True
for ii,dfrow in dfs1.iterrows():
    if scale_res:
        il = tio.ScalarImage(dfrow.vol_label)
        Hvol.append((il.data>0).sum()*0.5**3/1000)
        newres = (1200/((il.data>0).sum()*0.5**3/1000))**(1/3)*0.5 #taking adult brain with 1200 cm^3
        new_res.append(newres)
        #for old new resolution should be around 0.6, but I still stay at 0.75 to avoid PV
    else:
        newres = 0.75

    il = tio.ScalarImage(dfrow.vol_T2)
    new_aff = il.affine; new_aff[0,0]=newres;new_aff[1,1]=newres;new_aff[2,2]=newres;
    io = tio.ScalarImage(tensor=il.data, affine=new_aff); io.save(f'{dout_imgT2}/{sujname[ii]}.nii.gz')
    il = tio.ScalarImage(dfrow.vol_T1)
    new_aff = il.affine; new_aff[0,0]=newres;new_aff[1,1]=newres;new_aff[2,2]=newres;
    io = tio.ScalarImage(tensor=il.data, affine=new_aff); io.save(f'{dout_imgT1}/{sujname[ii][:-1]}1.nii.gz')
    il = tio.ScalarImage(dfrow.binPV)
    new_aff = il.affine; new_aff[0,0]=newres;new_aff[1,1]=newres;new_aff[2,2]=newres;
    io = tio.ScalarImage(tensor=il.data, affine=new_aff); io.save(f'{dout_lab}/binPV_{sujname[ii][:-1]}1.nii.gz')
    il = tio.ScalarImage(dfrow.vol_label)
    new_aff = il.affine; new_aff[0,0]=newres;new_aff[1,1]=newres;new_aff[2,2]=newres;
    io = tio.ScalarImage(tensor=il.data, affine=new_aff); io.save(f'{dout_lab}/labFree_{sujname[ii][:-1]}1.nii.gz')

#make csv for old results
sujdirbin = gdir('/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/baby/Article/bin15suj/from0/eval_T2_model_15suj_bgMida_tSDT_wmEM_mot_ep240',
              '|'.join([f'{ss}_ses-{ssid}' for ss,ssid in zip(dfs1.suj,dfs1.session_id) ]))
sujdirpv = gdir('/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/baby/Article/pve/eval_T2_model_noscale_pv_15suj_bgMida_tSDT_wmEM_mot_ep240',
              '|'.join([f'{ss}_ses-{ssid}' for ss,ssid in zip(dfs1.suj,dfs1.session_id) ]))
sujdirorig = get_parent_path(dfs1.vol_T1)[0]
dfpred = pd.DataFrame()
dfpred['sujname'] = sujname
dfpred['vol_path'] = gfile(sujdirbin,'data.nii')
dfpred['predict_path'] = gfile(sujdirbin,'bin_prediction.nii.gz')
dfpred['lab_free'] = gfile(sujdirbin,'bin_label.nii.gz') #gfile(sujdirorig,'^sub.*drawem9_dseg.nii.gz') #bad padding
dfpred['lab_binPV'] = gfile(sujdirpv,'^bin_label.nii.gz')

dfpred['model_name'],dfpred['dataset_name'],dfpred['input_type'] = 'wmEM_mot_ep240' ,DSname,'vol_T2'
dfpred.to_csv(dout1 + '/previous_dHcp_pred_bin_wmEM_mot_ep240.csv',index=False)

dfpred = pd.DataFrame()
dfpred['sujname'] = sujname
dfpred['vol_path'] = gfile(sujdirpv,'data.nii')
dfpred['predict_path'] = gfile(sujdirpv,'bin_prediction.nii.gz')
dfpred['lab_free'] = gfile(sujdirbin,'bin_label.nii.gz') #gfile(sujdirorig,'^sub.*drawem9_dseg.nii.gz') #bad padding
dfpred['lab_binPV'] = gfile(sujdirpv,'^bin_label.nii.gz')

dfpred['model_name'],dfpred['dataset_name'],dfpred['input_type'] = 'pve_wmEM_mot' ,DSname,'vol_T2'
dfpred.to_csv(dout1 + '/previous_dHcp_pred_pve_wmEM_mot_ep240.csv',index=False)

#get volume_estimation from hcp
dfh=pd.read_csv('/network/iss/cenir/analyse/irm/users/romain.valabregue/PVsynth/training_saved_sample/nnunet/testing_set/csv_validationHCP_MICCAI/HCP_trainset_07mm_suj16_vol_T1_07_pred_DS708_5nnResXL_res.csv')

Hvol=[]
for ii,dfrow in dfh.iterrows():
    il = tio.LabelMap(dfrow.lab_Free)
    Hvol.append((il.data>0).sum())
vol_hcp_mean = 1200; # torch.tensor(Hvol).sum()/16*0.7**3 /1000 # 1215.9288 cm^3

#write GT csv (for pred eval)
dfp=pd.DataFrame()
dfp['sujname']= sujname
dfp['vol_path'] = gfile(dout_imgT2,'.*gz')
dfp['vol_free'] = gfile(dout_lab,'Free.*gz')
dfp['vol_binPV'] = gfile(dout_lab,'binPV.*gz')
dfp.to_csv(dout1+'/dHcp_young075_scale_volT2_GT.csv',index=False)



### adult hcp
dt1= gdir('/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/session2',['\d.*','T1w','T1_1mm'])
droi= gdir('/network/lustre/iss02/opendata/data/HCP/raw_data/test_retest/session2',['\d.*','T1w','ROI_PVE_1mm'])
ft1 = gfile(dt1,'^T1w_1mm.nii$'); flab = gfile(droi, '^remap')
sujname = get_parent_path(droi,3)[1]
sujname = [ f'ahcp_S{i}_retest2_{s}_T1_1mm' for i,s in enumerate(sujname)]
dfone = pd.DataFrame(); dfone['sujname'] = sujname; dfone['label'] = flab; dfone['data']= ft1 ;
df = pd.concat([dfone,df])
### macaque Marseille
fT1 = gfile('/network/lustre/iss02/opendata/data/template/primate/macac/derivatives/crop_T1w_masked/','s*')
flab = gfile('/network/lustre/iss02/opendata/data/template/primate/macac/derivatives/crop_label/','^rema*')
fT1 = gfile('/network/lustre/iss02/opendata/data/template/primate/macac/derivatives/crop_T1w_masked/','^crop.*s*')
flab = gfile('/network/lustre/iss02/opendata/data/template/primate/macac/derivatives/crop_label/','^crop.*rema*')
#fo = addprefixtofilenames(flab,'r042_')
#for f1,f2 in zip(flab,fo):
#    print(f'mrgrid {f1} regrid -voxel 0.42 -force -interp nearest {f2}')

#lab_remap = {1:3,2:1, 3:2,4:2, 5:13}; tmap = tio.RemapLabels(lab_remap); fout = addprefixtofilenames(flab,'remap_')
#for ff,ffo in zip(flab, fout):
#    il = tio.LabelMap(ff); ilt = tmap(il); ilt.save(ffo)
#flab = gfile('/network/lustre/iss02/opendata/data/template/primate/macac/derivatives/crop_label/','remap_s*')

sujname = [f'mmac_S{i+39}' for i in range(5)]
dfone = pd.DataFrame(); dfone['sujname'] = sujname; dfone['label'] = flab; dfone['data']= fT1 ;
df = pd.concat([dfone,df])

### brain cata
fT1 = gfile('/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/braincata/QC/mask/data','gz')
flab = gfile('/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/braincata/QC/mask/cerebrum','gz')
fT1 = gfile('/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/braincata/QC/mask/data_crop','gz')
flab = gfile('/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/braincata/QC/mask/cerebrum_crop','gz')

sujname = [ f'bcat_{os.path.basename(ff)[:-7]}' for ff in fT1]

fT1.append('/network/lustre/iss02/opendata/data/template/primate/Johnson_etal_Equine_BrainAtlas_and_Templates_v2/crop_r08template_pub.nii.gz')
flab.append('/network/lustre/iss02/opendata/data/template/primate/Johnson_etal_Equine_BrainAtlas_and_Templates_v2/crop_r08mask_gm_wm.nii.gz')
sujname.append('bcat_Equine')
fT1.append('/network/lustre/iss02/opendata/data/template/primate/Johnson_etal_2019_Canine_Atlas/r038Canine_population_template.nii.gz')
flab.append('/network/lustre/iss02/opendata/data/template/primate/Johnson_etal_2019_Canine_Atlas/r038mask.nii.gz')
sujname.append('bcat_Canine')
dfone = pd.DataFrame(); dfone['sujname'] = sujname; dfone['label'] = flab; dfone['data']= fT1 ;
df = pd.concat([dfone,df])
df.to_csv('/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/species/test_set3.csv')

#crop
fout=addprefixtofilenames(fT1,'crop_')
fout=addprefixtofilenames(flab,'crop_')
for fm, fo , fref in zip(flab,fout,fT1):
    print(f'mrgrid {fm} regrid -template {fref} -stride {fref} {fo}')
    #print(f'mrgrid {fm} crop -mask {fm} -uniform -10 {fo}')

#rescal dhcp old
f = gfile(dout_data, '^dhcp_S8')
f = gfile(dout_label, '^dhcp_S8')
fout = addprefixtofilenames(f,'r08_')
for fi, fo in zip(f,fout):
    if os.path.isfile(fo):
        a=1
    else:
        print(f'mrgrid {fi} regrid -voxel 0.8 -interp nearest {fo}')

#for i in dhcp_S8*; do  rm -f $i;  mv r08_$i $i; done

#crop mmac
fda = gfile(dout_data, '^mmac')
fla = gfile(dout_label, '^mmac')
fma = gfile(gdir(dout_label,'lab_brain'),'^mmac')
fout1 = addprefixtofilenames(fda,'ccrop_')
fout2 = addprefixtofilenames(fla,'ccrop_')
for fd,fl,fo1,fo2,fm in zip(fda,fla,fout1, fout2,fma):
    print(f'mrgrid -force  {fd} crop -mask {fm} -uniform -10 {fo1}')
    print(f'mrgrid  -force {fl} crop -mask {fm} -uniform -10 {fo2}')

for i in mmac*; do  rm -f $i;   mv ccrop_$i $i; done


#remap sele data
dic_map = {1:9, 7:1 , 9:7 }
ff = gfile('/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/species/figure/sel_data','^bin')
tmap = tio.RemapLabels(dic_map)
fo = remap_filelist(ff,tmap)