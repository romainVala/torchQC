import torchio as tio
from utils_file import get_parent_path, gfile, gdir, addprefixtofilenames
import torch, os, pandas as pd
suj = gdir('/network/lustre/iss02/opendata/data/baby/devHCP/rel3_dhcp_anat_pipeline',['.*','ses','ana'])
ff = gfile(suj,'^sub.*[1234567890]_T1w.*nii')
suj = get_parent_path(ff)[0]

#suj = gdir('/data/romain/baby/training_suj',['.*','ses','ana'])
thot = tio.OneHot()
thotinv = tio.OneHot(invert_transform=True)
ind_sel=[0, 4, 5, 6, 7, 8, 9 ]  #new after training remove also hypocamp and background voxels

for ind_suj, onesuj in enumerate(suj):
    if os.path.isfile(onesuj+'/binPVlabel.nii.gz'):
        print(f'SKIP binPVlabel exist for {onesuj}')
        continue

    fpv_gm = gfile(onesuj,'pial.*pv')
    fpv_wm = gfile(onesuj,'wm.*pv')
    if ( len(fpv_wm)==0 ) | ( len(fpv_gm)==0):
        print(f'SKIP FAILLURE TOBLERONE FOR {onesuj}')
        continue

    fseg = gfile(onesuj, '^nw.*drawem.*thin')
    fseg = gfile(onesuj, '^sub.*-drawem9.*nii')

    tsuj = tio.Subject({'gml':tio.ScalarImage(fpv_gm[0]),'gmr':tio.ScalarImage(fpv_gm[1]),
                        'wml':tio.ScalarImage(fpv_wm[0]),'wmr':tio.ScalarImage(fpv_wm[1]),
                        'seg': tio.LabelMap(fseg[0])})

    tsuj = thot(tsuj)
    
    WMdata = tsuj.wml.data + tsuj.wmr.data
    GMfull = tsuj.gml.data + tsuj.gmr.data 
    dseg= tsuj.seg.data.type(torch.float32)

    GMdata = GMfull- WMdata
    GMdata[GMdata<0] = 0 #do not know why there are negative value from toblerone ...
    WMdata[WMdata>1] = 1
    GMfull[GMfull>1] = 1

    #remove dseg intersection with WM
    din =   dseg[[4],...] + dseg[[5],...] + dseg[[6],...]+ dseg[[7],...]+ dseg[[8],...] + dseg[[9],...]
    WMdata[din>0] = 0
    if dseg.shape[0]>11:
        skullWM = dseg[[11], ...] * WMdata
        skullWM[skullWM>0] = 0
        dseg[11] = skullWM[0]

    #remove dseg intersection with CSF
    CSF = 1 - GMfull
    dout = dseg[10:].sum(axis=0).unsqueeze(0)
    dout = dout + dseg[[0],...] + dseg[[4],...] + dseg[[9],...]  + dseg[[8],...] + dseg[[7],...] + dseg[[6],...]

    CSF[dout>0] = 0
    # csf_within_wm = dseg[[1],...] * WMdata
    # CSF[csf_within_wm > 0] = 1  # bof car si bad label near GM
    csf_within_wm = CSF * WMdata
    CSF[csf_within_wm>0] = 1 - (WMdata[csf_within_wm>0] + GMdata[csf_within_wm>0])

    mask_ext = (dseg[10:,...].sum(axis=0)).unsqueeze(0)
    isel = mask_ext>0
    GMdata[isel] = 0; WMdata[isel] = 0; CSF[isel] = 0;

    #remove dseg intersection with GM
    for ii in ind_sel:
        dseg_lab = dseg[[ii],...]
        vox_sel = ( (GMdata>0.01) | (WMdata>0.01) | (CSF>0.01) ) & (dseg_lab>0)
        if vox_sel.sum()>0:
            print(f'GM partial volume with {ii} for {vox_sel.sum()} voxels')
            dseg_lab[vox_sel] = 1 - ( GMdata[vox_sel] + WMdata[vox_sel] + CSF[vox_sel] )
            dseg[[ii],...] = dseg_lab

    dseg[2,...] = GMdata;    dseg[1,...] = CSF;    dseg[3,...] = WMdata
    tsuj.add_image(tio.ScalarImage(fpv_wm[1]),'GM');    tsuj.add_image(tio.ScalarImage(fpv_wm[1]),'WM')
    tsuj.add_image(tio.ScalarImage(fpv_wm[1]),'CSF')
    
    tsuj.GM.data  = GMdata;    tsuj.WM.data  = WMdata;    tsuj.CSF.data = CSF;    tsuj.seg.data = dseg
    
    tsuj.GM.save(onesuj+'/GM.nii.gz');    tsuj.WM.save(onesuj+'/WM.nii.gz');    tsuj.CSF.save(onesuj+'/CSF.nii.gz')
    tsuj.seg.save(onesuj+'/PVlabel.nii.gz')
    tsuj = thotinv(tsuj)
    tsuj.seg.save(onesuj+'/binPVlabel.nii.gz')
    print(f'LABEL PV sum min {dseg.sum(axis=0).min()} max {dseg.sum(axis=0).max()}')
    if dseg.sum(axis=0).max()>1.1:
        print('ERROR pv sum')
    print(f'DONE {onesuj}  {ind_suj} / {len(suj)}')

    
nw drawem9

    index	name
1	CSF
2	Cortical gray matter
3	White matter
4	Background
5	Ventricles
6	Cerebellum
7	Deep Gray Matter
8	Brainstem
9	Hippocampi and Amygdala

BG,0
Air,10
dura_brain,11
eyes,12
muco,13
muscle,14
nerve,15
skin,16
skull,17
vessel,18

import torchio as tio
from utils_file import get_parent_path, gfile, gdir
import torch
import os


suj = gdir('/data/romain/baby/training_suj',['.*','ses','ana'])
f4D = gfile(suj,'scaleI1_iWMPVlabel.nii.gz')
fdseg = gfile(suj,'^sub.*desc-drawem9_d')
thot = tio.OneHot(invert_transform=True)
tmap = tio.RemapLabels(remapping={"0": 0,  "1": 1, "2": 2,  "3": 3, "4": 4,  "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
				    "10": 10,"11": 11,"12": 12,"13": 13,"14": 14,"15": 15,"16": 16,"17": 17,"18": 18, "19": 3,    "20": 3})
transfo = tio.Compose([tmap, thot])
for ii, onesuj in enumerate(suj):
    print(ii)
    tsuj = tio.Subject({'pv':tio.LabelMap(f4D[ii])})

    tsuj = transfo(tsuj)
    fn = '/scaleI1_labelPV_' + os.path.basename(fdseg[ii])
    tsuj.pv.data = tsuj.pv.data.type(torch.int32) # = torch.int16(tsuj.pv.data)

    tsuj.pv.save(suj[ii]+fn)


#make target label for em_synth_label
suj = gdir('/data/romain/baby/training_suj_synthseg/res_em','^[345]')
suj=gdir('/data/romain/baby/training_suj_synthseg/res_em',['^[345]','^[23]'])

flab = gfile(suj,'^s')
fout = addprefixtofilenames(flab,'target_')

for fin, fo in zip(flab, fout):

    i = tio.LabelMap(fin)
    i.data[i.data>9] = 4;
    i.save(fo)
    print(f'wrotten {fo}')

fem = gfile(suj,'^scale')
fta = gfile(suj,'^target')

#only once, then mv by hand in target dir
restarget = '/data/romain/baby/training_suj_synthseg/target9/'
fname = get_parent_path(flab)[1]
fta = addprefixtofilenames(fname, restarget)
df=pd.DataFrame();  df['emBG_label'] = flab; df['target_label9'] = fta
df['sujname'] = [f'suj{i}' for i in range(90)]
df.to_csv('rrr')

