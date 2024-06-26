import pandas as pd, os
from utils_file import addprefixtofilenames
import torchio as tio
from scipy.ndimage.morphology import binary_erosion, binary_dilation
import skimage.measure as ski
import torch, numpy as np
from collections import Counter

def get_image_as_numpy(mask):
    return_tensor = return_image = False
    if isinstance(mask,tio.Image):
        mask_image = mask.__copy__()  #if .copy() return a dict hmmm ...
        mask = mask.data
        return_image=True
    if torch.is_tensor(mask):
        mask = mask.numpy()
        return_tensor = True
    return mask, return_image, return_tensor

def get_mask_external_broder(mask_in):

    mask, return_image, return_tensor = get_image_as_numpy(mask_in)

    mask_bin = np.zeros_like(mask)
    mask_bin[mask>0] = 1
    border = binary_dilation(mask_bin).astype(int) - mask_bin

    if return_tensor:
        border = torch.tensor(border)
    if return_image:
        mask_in['data'] = border
        return mask_in

    return border

def get_mask_neighbor(mask_in, volume_label,label_csv):
    df=pd.read_csv(label_csv)
    label_dict = {dff[1].synth: dff[1].Name for dff in df.iterrows()}
    border = get_mask_external_broder(mask_in)

    border, return_tensor, return_image = get_image_as_numpy(border)
    labels, return_tensor, return_image = get_image_as_numpy(volume_label)
    border = border[0]; labels = labels[0] #only 3D
    neighbor_values = labels[border>0]
    counter_obj = Counter(neighbor_values).most_common()
    counter_dict = {c[0]: c[1] for c in counter_obj}
    tot_vol = sum([v for v in counter_dict.values()])
    for l_val, l_count in counter_dict.items():
        #print(f'label {label_dict[l_val]} {l_val} {l_count/tot_vol*100:.2} % {l_count}')
        print(f'L {label_dict[l_val]} \t {l_count/tot_vol*100:.2}  =({l_count} vox)')
        mask_bin = np.zeros_like(border)
        mask_bin[(border>0) & (labels==l_val)] = 1
        nb_comp = ski.label(mask_bin,connectivity=3)
        properties = ['extent', 'solidity', 'area','area_convex','area_bbox', 'area_filled', 'euler_number','perimeter',
                      'eccentricity','orientation', 'centroid', 'bbox']
        df_con = pd.DataFrame(ski.regionprops_table(mask_bin, properties=properties))


def get_remap_from_csv(fin, index_col_in=0, index_col_remap=1):
    df = pd.read_csv(fin, comment='#')
    #dic_map= { r[0]:r[1]  for i,r in df.iterrows()}
    dic_map={}
    for i,r in df.iterrows():
        if r[index_col_remap]==r[index_col_remap]: #remove nan
            dic_map[r[index_col_in]] = r[index_col_remap]
    #remap_keys = set(dic_map.keys())
    return tio.RemapLabels(remapping=dic_map)



def get_fastsurfer_remap(faparc, fcsv = '/data/romain/template/free_remap.csv',index_col_in=0, index_col_remap=1):

    tmap = get_remap_from_csv(fcsv, index_col_remap=index_col_remap, index_col_in=index_col_in)
    dic_map = tmap.remapping
    remap_keys = set(dic_map.keys())
    # complet gray matter labels
    il = tio.LabelMap(faparc)
    lu=il.data.unique().numpy().astype(int)
    for ii in lu:
        if ii not in remap_keys:
            if ii < 1000:
                print(f'AAAAAAA no value for {ii} AAAAAAAAA')
            else:
                dic_map[ii] = 1  # gray matter
    tmap = tio.RemapLabels(remapping=dic_map)
    return tmap

def check_remap(il, dic_map):

    lu=il.data.unique().numpy().astype(int)
    remap_keys = set(dic_map.keys())
    print(f'check remap for {il.path}')
    for ii in lu:
        if ii not in remap_keys:
            print(f'WARNING no value for {ii}      AAAAAAAAAAAAAA')


def resample_and_smooth4D(fin,fref, blur4D=0.5):

    ilt = tio.LabelMap(fin)

    thot = tio.OneHot();
    thoti = tio.OneHot(invert_transform=True)
    if blur4D > 0:
        ts = tio.Blur(std=blur4D)

    tresample = tio.Resample(target=fref, image_interpolation='bspline')

    ilr = thot(ilt)
    ilr['data'] = ilr.data.float()
    for k in range(ilr.data.shape[0]):
        ilk = tio.ScalarImage(tensor=ilr.data[k].unsqueeze(0), affine=ilr.affine)
        if fref is not None:
            ilk = tresample(ilk)
        if blur4D > 0:
            ilk = ts(ilk)

        if k == 0:
            data_out = torch.zeros((ilr.data.shape[0],) + ilk.shape[1:])
        data_out[k] = ilk['data'][0]
    ilr.data = data_out;
    ilr.affine = ilk.affine
    ilt = thoti(ilr)
    ilt['data'] = ilt.data.to(torch.uint8)
    return ilt


def remap_filelist(fin, tmap, prefix='remap_', fref=None, skip=True, reslice_4D=False, blur4D=0.5, save=True):
    # fref must be a list of same size

    if isinstance(tmap, str):
        if tmap == 'fastsurfer':
            tmap = get_fastsurfer_remap(fin[0])
        else:
            raise(f' unknow {tmap} for tmap string')

    fout = addprefixtofilenames(fin, prefix)

    for index, (fi, fo) in enumerate(zip(fin, fout)):
        if fo[-4:] == '.mgz' :
            fo = fo[:-4] + '.nii.gz'
        if os.path.isfile(fo):
            if skip:
                print(f'SKIP {fo} (file exist)')
                continue
            else:
                print(f'Erasing existing remap file {fo}')

        il = tio.LabelMap(fi)
        dic_map = tmap.remapping
        check_remap(il, dic_map)
        ilt = tmap(il)
        if fref:
            if reslice_4D:
                thot = tio.OneHot(); thoti = tio.OneHot(invert_transform=True)
                if blur4D>0:
                    ts = tio.Blur(std=blur4D)
                tresample = tio.Resample(target=fref[index], image_interpolation='bspline')

                ilr = thot(ilt)
                ilr['data'] = ilr.data.float()
                for k in range(ilr.data.shape[0]):
                    ilk = tio.ScalarImage(tensor=ilr.data[k].unsqueeze(0), affine=ilr.affine)
                    if blur4D>0:
                        iltk = ts(tresample(ilk))
                    else:
                        iltk = tresample(ilk)

                    if k==0:
                        data_out = torch.zeros( (ilr.data.shape[0],)+ iltk.shape[1:] )
                    data_out[k] = iltk['data'][0]
                ilr.data = data_out; ilr.affine = iltk.affine
                ilt = thoti(ilr)
                ilt['data'] = ilt.data.to(torch.uint8)

            else:
                tresample = tio.Resample(target=fref[index]) #label map take nearrest
                ilt = tresample(ilt)
        if save:
            ilt.save(fo)
        else:
            return ilt


from scipy.ndimage import label as scipy_label
import numpy as np

def get_largest_connected_component(mask, structure=None):
    """Function to get the largest connected component for a given input.
    :param mask: a 2d or 3d label map of boolean type.
    :param structure: numpy array defining the connectivity.
    """
    components, n_components = scipy_label(mask, structure)
    return components == np.argmax(np.bincount(components.flat)[1:]) + 1 if n_components > 0 else mask.copy()


