import torchio as tio
from utils_file import get_parent_path, gfile, gdir
import torch

suj = gdir('/data/romain/baby/training_suj',['.*','ses','ana'])
fsurf = gfile(suj,'pv')
thot = tio.OneHot()
ind_sel=[4, 5, 6, 7, 8]

for onesuj in suj:
    fpv_gm = gfile(onesuj,'pial.*pv')
    fpv_wm = gfile(onesuj,'wm.*pv')
    fseg = gfile(onesuj, '^nw.*drawem.*thin')

    tsuj = tio.Subject({'gml':tio.ScalarImage(fpv_gm[0]),'gmr':tio.ScalarImage(fpv_gm[1]),
                        'wml':tio.ScalarImage(fpv_wm[0]),'wmr':tio.ScalarImage(fpv_wm[1]),
                        'seg': tio.LabelMap(fseg[0])})
    tsuj = thot(tsuj)
    
    WMdata = tsuj.wml.data + tsuj.wmr.data
    GMfull = tsuj.gml.data + tsuj.gmr.data 
    dseg= tsuj.seg.data

    GMdata = GMfull- WMdata
    GMdata[GMdata<0] = 0 #do not know why there are negative value from toblerone ...
    WMdata[WMdata>1] = 1
    GMfull[GMfull>1] = 1

    #remove dseg intersection with WM
    din =   dseg[[4],...] + dseg[[5],...] + dseg[[6],...]+ dseg[[7],...]+ dseg[[8],...] + dseg[[9],...]
    WMdata[din>0] = 0
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
    print(f'DONE {onesuj} ')
    print(f'LABEL PV sum min {dseg.sum(axis=0).min()} max {dseg.sum(axis=0).max()}')

    
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

