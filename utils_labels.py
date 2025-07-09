import pandas as pd, os
from utils_file import addprefixtofilenames
import torchio as tio
from scipy.ndimage.morphology import binary_erosion, binary_dilation
import skimage.measure as ski
import torch, numpy as np
from collections import Counter
import subprocess
from utils_file import get_parent_path, addprefixtofilenames
import nibabel as nib
import re
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

def pool_dill(data, scale=2, interp_up = 'nearest'):
    maxpool = torch.nn.MaxPool3d(scale)
    dill_data = maxpool(data.unsqueeze(0).unsqueeze(0).float())
    ups = torch.nn.Upsample(scale_factor=scale, mode=interp_up)
    return ups(dill_data)[0][0]

def con_comp_dill(data, scale=2, interp_up = 'nearest'):
    maxpool = torch.nn.MaxPool3d(scale)
    dill_data = maxpool(data.unsqueeze(0).unsqueeze(0).float())
    components, n_components = scipy_label(dill_data);
    components = torch.tensor(components)
    ups = torch.nn.Upsample(scale_factor=scale, mode=interp_up)
    upc_cc = ups(components.float()).int()[0][0]
    return upc_cc, n_components


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

def get_remapping(fin, tmap_index_col=None,lab_name=None ):
    if ~(os.path.isfile(fin)):
        match fin:
            case 'assn':
                fin = '/network/iss/opendata/data/template/remap/remap_vol2Brain_label.csv'


    df = pd.read_csv(fin, comment='#')
    #dic_map= { r[0]:r[1]  for i,r in df.iterrows()}
    if tmap_index_col is not None:
        dic_map={}
        for i,r in df.iterrows():
            if r[tmap_index_col[1]]==r[tmap_index_col[1]]: #remove nan
                dic_map[r[tmap_index_col[0]]] = r[tmap_index_col[1]]
        #remap_keys = set(dic_map.keys())
        tmap = tio.RemapLabels(remapping=dic_map)
        return tmap

    if lab_name is not None:
        dic_lab = {ll[lab_name[0]]: ll[lab_name[1]]  for ii, ll in df.iterrows() }
        return dic_lab



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


def resample_and_smooth4D(fin,fref, blur4D=0.5, fout=None, skip_blur=None):
    if fout is not None:
        if os.path.isfile(fout):
            print(f"SKUPING {fout}")
            return
        else:
            print(f'computing {fout}')
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
            DoBlu = True
            if skip_blur is not None:
                if k in skip_blur:
                    print(f'no bluring for label {k}')
                    DoBlu = False
            if DoBlu:
                ilk = ts(ilk)

        if k == 0:
            data_out = torch.zeros((ilr.data.shape[0],) + ilk.shape[1:])
        data_out[k] = ilk['data'][0]
    ilr.data = data_out;
    ilr.affine = ilk.affine
    ilt = thoti(ilr)
    ilt['data'] = ilt.data.to(torch.uint8)

    if fout is not None:
        ilt.save(fout)
        return

    return ilt


def remap_filelist(fin, tmap, prefix='remap_', fref=None, skip=True, reslice_4D=False, blur4D=0.5,
                   save=True, reduce_BG=0 , reslice_with_mrgrid=False):
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
        if fref:
            if reslice_4D:
                thot = tio.OneHot(); thoti = tio.OneHot(invert_transform=True)
                if blur4D>0:
                    ts = tio.Blur(std=blur4D)
                tresample = tio.Resample(target=fref[index], image_interpolation='bspline')

                ilt = tmap(il)
                ilr = thot(ilt)
                ilr['data'] = ilr.data.float()
                for k in range(ilr.data.shape[0]):
                    ilk = tio.ScalarImage(tensor=ilr.data[k].unsqueeze(0), affine=ilr.affine)
                    if blur4D>0:
                        iltk = ts(tresample(ilk))
                    else:
                        iltk = tresample(ilk)

                    if k==0: #'data_out' not in locals() do not work if list ...
                        data_out = torch.zeros( (ilr.data.shape[0],)+ iltk.shape[1:] )
                    data_out[k] = iltk['data'][0]
                    if k==0 and reduce_BG>0:
                        data_out[k] *= reduce_BG

                ilr.data = data_out; ilr.affine = iltk.affine
                ilt = thoti(ilr)
                ilt['data'] = ilt.data.to(torch.uint8)

            else:
                if reslice_with_mrgrid:
                    fffo = addprefixtofilenames(fo, 'rmrt_')
                    import subprocess
                    cmd = f'mrgrid {fi} regrid -interp nearest -template {fref[index]} -strides {fref[index]} {fffo[0]}'
                    outvalue = subprocess.run(cmd.split(' '))
                    ilt = tmap(tio.LabelMap(fffo[0]))
                    os.remove(fffo[0])

                else:
                    ilt = tmap(il)
                    tresample = tio.Resample(target=fref[index]) #label map take nearrest
                    ilt = tresample(ilt)
        else:
            ilt = tmap(il)

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


def single_to_4D(fin, fo4D, fobin, delete_single=True):
    thoti = tio.OneHot(invert_transform=True);
    for (f1, f2, f3) in zip(fin, fobin, fo4D):
        i1 = tio.LabelMap(f1)
        i1.save(f3)
        ibin = thoti(i1)
        ibin.save(f2)
        if delete_single:
            for ff in f1:
                outvalue = subprocess.run(f'rm -f {ff}'.split(' '))

def rescale_affine_corrected(affine, shape, zooms, new_shape=None):
    """copy (modify from nibable
    """
    shape = np.asarray(shape)
    new_shape = np.array(new_shape if new_shape is not None else shape)

    s = nib.affines.voxel_sizes(affine)
    rzs_out = affine[:3, :3] * zooms / s

    # Using xyz = A @ ijk, determine translation
    centroid = nib.affines.apply_affine(affine, (shape -1) / 2)
    t_out = centroid - rzs_out @ ((new_shape -1) / 2)
    return nib.affines.from_matvec(rzs_out, t_out)

def pool_remap_to_4DPV(fpv, pooling_size=2, ensure_multiple=None, tmap=None, skip=True, prefix='r075',
                       dirout=None):
    if isinstance(fpv, str):
        fpv = [fpv]
    fo = addprefixtofilenames(fpv, f'{prefix}_4D_')
    fobin = addprefixtofilenames(fpv, f'{prefix}_bin_')
    if dirout is not None:
        fo = [os.path.join(dirout, ff) for ff in get_parent_path(fo)[1]]
        fobin = [os.path.join(dirout, ff) for ff in get_parent_path(fobin)[1]]

    thoti = tio.OneHot(invert_transform=True);
    thot = tio.OneHot()  # avgpool = torch.nn.AvgPool3d(kernel_size=2)
    avgpool = torch.nn.AvgPool3d(kernel_size=pooling_size, ceil_mode=True)  # meme taille que mrgrid

    if ensure_multiple is None:
        ensure_multiple = pooling_size  # this remove inprecision for nifti header (origin coordinate)
    tpad = tio.EnsureShapeMultiple(ensure_multiple)

    # remap_filelist(forig, tmap, prefix='target')
    for (f1, f3, f4) in zip(fpv, fo, fobin):
        if os.path.isfile(f4):  # ii< df.shape[0]:
            if skip:
                print(f'skip existing {f4}')
                continue
            else:
                print(f'no skip ERASING {f4}')
        print(f'computing {f3}')

        i11 = tpad(tmap(tio.LabelMap(f1))) if tmap is not None else tpad(tio.LabelMap(f1))

        # get the new affine
        orig_shape = np.array(i11.shape[1:])
        new_shape = orig_shape // pooling_size
        new_vox_size = nib.affines.voxel_sizes(i11.affine) * pooling_size
        io_affine = rescale_affine_corrected(i11.affine, orig_shape, new_vox_size, new_shape)

        i11['data'] = i11.data.int()
        label_values = i11.data.unique()
        nb_channel = len(label_values)
        io_data = torch.zeros([nb_channel] + list(new_shape))
        # i1 = thot(i11) to much memory ! so do it one by one
        for channel in range(nb_channel):
            Sin = i11.data[0] == label_values[channel]
            io_data[channel] = avgpool(Sin.float().unsqueeze(0))[0]
        del (Sin)

        iout = tio.LabelMap(tensor=io_data, affine=io_affine)
        iout.save(f3)
        iout = thoti(iout)
        iout.save(f4)

def pool_remap(il_in, pooling_size=2, ensure_multiple=None, tmap=None,keep_missing_label=True, islabel=True):

    thoti = tio.OneHot(invert_transform=True);
    avgpool = torch.nn.AvgPool3d(kernel_size=pooling_size, ceil_mode=True)  # meme taille que mrgrid

    if ensure_multiple is None:
        ensure_multiple = pooling_size  # this remove inprecision for nifti header (origin coordinate)
    tpad = tio.EnsureShapeMultiple(ensure_multiple)

    i11 = tpad(tmap(il_in)) if tmap is not None else tpad(il_in)

    # get the new affine
    orig_shape = np.array(i11.shape[1:])
    new_shape = orig_shape // pooling_size
    new_vox_size = nib.affines.voxel_sizes(i11.affine) * pooling_size
    io_affine = rescale_affine_corrected(i11.affine, orig_shape, new_vox_size, new_shape)
    if islabel:
        i11['data'] = i11.data.int()
        label_values = i11.data.unique()
        if keep_missing_label:
            nb_channel = int(label_values.max()) + 1
        else:
            nb_channel = len(label_values)
        io_data = torch.zeros([nb_channel] + list(new_shape))
        # i1 = thot(i11) to much memory ! so do it one by one
        for channel in range(nb_channel):
            if keep_missing_label:
                Sin = i11.data[0] == channel
            else:
                Sin = i11.data[0] == label_values[channel]
            io_data[channel] = avgpool(Sin.float().unsqueeze(0))[0]
        del (Sin)
        iout = tio.LabelMap(tensor=io_data, affine=io_affine)
        iout_bin = thoti(iout)
        return iout_bin, iout
    else: #image just one pooling
        io_data = avgpool(i11.data.float().unsqueeze(0))[0]
        iout = tio.LabelMap(tensor=io_data, affine=io_affine) #induce strange bug i11
        # iout.affine = io_affine
    return iout

def read_freesurfer_colorlut(fsc=None):
    if fsc is None:
        fsc = '/network/iss/opendata/data/template/remap/FreeSurferColorLUT_v8.txt'

    rgb = np.empty((0, 4), dtype=np.int64)
    label_names = {}

    with open(fsc, 'r') as f:
      raw_lut = f.readlines()

    # read and process line by line

    pattern = re.compile(r'\d{1,5}[ ]+[a-zA-Z-_0-9*.]+[ ]+\d{1,3}[ ]+\d{1,3}[ ]+\d{1,3}[ ]+\d{1,3}')
    for line in raw_lut:
      if pattern.match(line):
        s = line.rstrip().split(' ')
        s = list(filter(None, s))
        rgb = np.append(rgb, np.array([[int(s[0]), int(s[2]), int(s[3]), int(s[4])]]), axis=0)
        #label_names[int(s[0])] = s[1]
        label_names[s[1]] = np.array([[ int(s[2]), int(s[3]), int(s[4])]])
    return rgb, label_names


def create_mask(fin,diclab):
    dirout = get_parent_path(fin)[0]
    for fi, fo in zip(fin,dirout):
        il = tio.LabelMap(fi)
        for k,v in diclab.items():
            tout = torch.zeros_like(il.data)
            tout[il.data==v] = 1
            io = tio.LabelMap(tensor=tout,affine=il.affine)
            fout = f'/m_{k}.nii.gz'
            print(f'creating {fout}')
            io.save(fo+fout)


