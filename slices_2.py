#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import nibabel.processing as npi
import matplotlib.cm as cm
import os
from PIL import Image, ImageDraw, ImageFont
import torchio

# from scipy.ndimage import affine_transform
# from PIL import Image

# %%
def reslice_im(im, fref, acoreg, type_pos):
    """
    Takes as an input an image and a template, and return the resample of the first input
    in the sp
    ace linked to the template, with the following rates:
        *percentage of output space in the intersection
        *percentage of input space in the intersection
    """
    print('resampling img')
    #if type_pos != "mm":
    if type_pos == "mm_mni":
        print('changing the image affine ')
        imgaff = acoreg.dot(im.affine)
        im.affine[:] = imgaff[:]
    out_img = npi.resample_from_to(im, fref, cval=-1)
    useful_rate = round(np.sum(out_img.get_fdata() != -1) / np.prod(fref.shape), 3)
    used_rate = round(np.sum(out_img.get_fdata() != -1) / np.prod(im.shape), 3)
    return out_img,  useful_rate, used_rate

def reslice_mask(mask, fref, acoreg, type_pos):
    out_mask = None
    if mask is not None:
        if type_pos == "mm_mni":
            maskaff = acoreg.dot(mask.affine)
            mask.affine[:] = maskaff[:]
        out_mask_temp = npi.resample_from_to(mask, fref, cval=0)
        out_mask = nb.Nifti1Image(np.round(out_mask_temp.get_fdata(), decimals = 0), affine=out_mask_temp.affine) #rrr ???
    return out_mask

# %%

def get_slices(im, mask, acoreg, view, type_pos, pos, mask_cut_pix=-1): # Ajouter im_reslice?
    """
    Takes as inputs the image, the associated mask (can be None), the acoreg,
    the view ("sagittal - coronal - axial), the type of the given position
    (in index (%) or in mm or in mm in the mni words), the position of the wanted slice,
    if we cut a subbox around the mask (mask_cut_pix < 0 if not, otherwise
    a box surounding the mask with a margin of the value of the parametr is extracted),
    and return the extracted slice (recut aroud the mask or not) and the whole slice of mask
    """
    list_type = ["vox", "voxmm", "mm", "mm_mni"]
    list_view = ["sag", "cor", "ax"]
    if type_pos not in list_type:
        raise ValueError("Type_pos input not recognized among the accepted type_pos inputs")
    if view not in list_view:
        raise ValueError("View input not recognized among the accepted view inputs")
    if acoreg is None and type_pos == "mm_mni":
        raise AssertionError("Trying to use mm_mni view with no acoreg")
    
    #header, image = im.header, im.get_fdata()
    header, image = im.header, im.dataobj

    max_dim = header.get_data_shape()[list_view.index(view)]
    if mask is not None:
        mask_im = mask.get_fdata()
    
    if type_pos == "vox":
        if pos < 0 or pos >= 1:
            raise ValueError("The vox type expects a percentage 0 < pos < 1")
        pos_var = int(pos * max_dim)

    else:
        if type_pos == "voxmm":
            # imgaff = acoreg.dot(im.affine)
            mat_affine = np.linalg.inv(acoreg.dot(im.affine))
        else :
            mat_affine = np.linalg.inv(im.affine)

        pos_vect = np.zeros((4,))
        pos_vect[-1] = 1
        pos_vect[list_view.index(view)] = pos
        axis = np.dot(mat_affine, pos_vect).astype("int")
        pos_var = axis[list_view.index(view)]
        if (pos_var) >= max_dim or pos_var <0:
            raise ValueError("The value given is out of the possible values of the image: %d mm correspond to slice %d "%(pos,pos_var))
    
    if view == "sag":
        matrix_slice = image[pos_var, :, :]
    elif view == "cor":
        matrix_slice = image[:, pos_var, :]
    else:
        matrix_slice = image[:, :, pos_var]
    
    if mask is not None:
        if view == "sag":
            mask_slice = mask_im[pos_var, :, :].astype(bool)
        elif view == "cor":
            mask_slice = mask_im[:, pos_var, :].astype(bool)
        else:
            mask_slice = mask_im[:, :, pos_var].astype(bool)
    else:
        mask_slice = None
    
    #print('matrix slice {}'.format(matrix_slice.shape))

    # if plot:
    #    plt.imshow(matrix_slice.T, origin = "lower", cmap = "hot",\
    # vmin = scale_values_im[0], vmax = scale_values_im[1])
    # plt.imshow(template_matrix_slice.T, origin = "lower", alpha = 0.4, cmap = "Greys_r",\
    #              vmin = scale_values_fref[0], vmax = scale_values_fref[1])
    if mask_cut_pix >= 0:
        col_info = np.sum(mask_slice, axis=0)  # 0 a indice j => mask = False sur tte colonne j
        row_info = np.sum(mask_slice, axis=1)  # 0 a indice j => mask = False sur tte ligne 
        col_info = list(col_info>0)
        row_info = list(row_info>0)
        l_h = row_info.index(True)
        l_b = len(row_info) - 1 - (row_info[::-1]).index(True)
        l_g = col_info.index(True)
        l_d = len(col_info) - 1 - (col_info[::-1]).index(True)
        return matrix_slice[max(l_h - mask_cut_pix, 0):min(l_b + mask_cut_pix+1, matrix_slice.shape[0]),
               max(l_g - mask_cut_pix, 0):min(l_d + mask_cut_pix+1, matrix_slice.shape[1])], mask_slice
    return matrix_slice, mask_slice


def plot_view(im, mask, fref, acoreg, slices_infos, mask_info, display_order=None,
              colormap=cm.Greys_r, colormap_noise=cm.hot, percentile_values=[0,99],
              plot_single=True, figure_path=None, dpi=50, plt_ioff=False):

    if len(slices_infos) != len(mask_info):
        if len(mask_info)==1: #just duplicate
            mask_info = [ mask_info[0] for i in range(0,len(slices_infos))]
        else:
            raise AssertionError("The 2 lists about the slices must have the same size")

    use_reslice_mm = (np.array([slices_infos[k][1] for k in range(len(slices_infos))]) == "mm").any()
    use_reslice_mni = (np.array([slices_infos[k][1] for k in range(len(slices_infos))]) == "mm_mni").any()

    display_order = display_order[::-1]

    if use_reslice_mm:
        im_resliced_mm, _, _ = reslice_im(im, fref, acoreg, type_pos="mm")
        mask_resliced_mm = reslice_mask(mask, fref, acoreg, type_pos="mm")

    if use_reslice_mni:
        im_resliced_mni, _, _ = reslice_im(im, fref, acoreg, type_pos="mm_mni")
        mask_resliced_mni = reslice_mask(mask, fref, acoreg, type_pos="mm_mni")

    list_matrix = []

    for j, item2 in enumerate(slices_infos):
        view, type_pos, pos = item2
        scaling, mask_cut_pix = mask_info[j]
        if mask is None and (scaling != "whole" or mask_cut_pix >= 0):
            print("For a empty mask, the scaling must be whole and we must not do the cut!")
            scaling = "whole"
            mask_cut_pix = -1
        if type_pos == "vox" or type_pos == "voxmm":
            temp_im = im
            temp_mask = mask
        elif type_pos == "mm":
            temp_im = im_resliced_mm
            temp_mask = mask_resliced_mm
        else:
            temp_im = im_resliced_mni
            temp_mask = mask_resliced_mni
        if scaling not in ["whole", "mask", "mask_font"]:
            raise ValueError("Scaling arguent not correct for value {}".format(j))
        v_min, v_max, v_min_f, v_max_f = scaling_func(temp_im, temp_mask, scaling, percentile_values)
        matrix_slice, mask_slice = get_slices(temp_im, temp_mask, acoreg, view,
                                              type_pos, pos, mask_cut_pix)

        list_matrix.append((matrix_slice, mask_slice, v_min, v_max, v_min_f, v_max_f))

    abs_max = np.max([m[0].shape[1] for m in list_matrix])
    ord_max = np.max([m[0].shape[0] for m in list_matrix])

    matrix_fig = np.zeros((display_order[1] * abs_max, display_order[0] * ord_max, 4)).astype(np.uint8)
    curseur_abs = 0
    curseur_ord = 0
    for j, item in enumerate(list_matrix):
        matrix_slice, mask_slice, v_min, v_max, v_min_f, v_max_f = item
        if mask_info[j][0] != "mask_font":
            temp = (np.uint8(255 * (colormap(matrix_slice.T / (v_max - v_min))).astype(np.float64)))

        else:
            temp = (np.uint8(255 *
                             (colormap_noise(
                                 (matrix_slice.T / (v_max_f - v_min_f) * np.logical_not(mask_slice).T).astype(
                                     np.float64))
                              + (colormap((matrix_slice / (v_max - v_min) * mask_slice).T).astype(np.float64)))))

        matrix_fig[curseur_abs: curseur_abs + temp.shape[0], curseur_ord:curseur_ord + temp.shape[1]] = np.flipud(temp)

        if (j + 1) % display_order[1] == 0:
            curseur_abs = 0
            curseur_ord += ord_max
        else:
            curseur_abs += abs_max

    if plot_single:
        imgsize = (matrix_fig.shape[1] / dpi, matrix_fig.shape[0] / dpi)
        fig, axs = plt.subplots(1, 1, figsize=imgsize, dpi=dpi)
        plt.subplots_adjust(wspace=0, hspace=0, left=0.0, right=1.0, bottom=0.0, top=1.0)
        axs.imshow(matrix_fig)  # , origin = "lower")
        # for k in range(display_order[1]):
        #    plt.plot((k*display_order[1], 0), (y1, y2), 'r-')
        axs.axis("off")
        fig.savefig( figure_path , facecolor="w", bbox_inches='tight')
        print("Saving figure {}".format(figure_path))
        if plt_ioff:
            plt.close(fig)

    return matrix_fig


def plot_montages(image_list, montage_shape, fig_path=None, dpi=80):

    if len(montage_shape) != 2:
        raise Exception('montage shape must be list or tuple of length 2 (rows, cols)')

    abs_max = np.max([m.shape[0] for m in image_list])
    ord_max = np.max([m.shape[1] for m in image_list])
    image_shape_max = [abs_max, ord_max]

    channel = 1
    if image_list[0].ndim>2:
        channel = image_list[0].shape[2]

    #print('image_max shape{}'.format(image_shape_max))

    image_montages = []
    montage_image = np.zeros(shape=(image_shape_max[0] * (montage_shape[0]), image_shape_max[1] * montage_shape[1], channel), dtype=np.uint8)
    #print('motage shape {}'.format(montage_image.shape))

    cursor_pos = [0, 0]
    start_new_img = False
    for img in image_list:

        if type(img).__module__ != np.__name__:
            raise Exception('input of type {} is not a valid numpy array'.format(type(img)))
        start_new_img = False
        #print('new img {} cursor {}'.format(img.shape, cursor_pos))

        montage_image[cursor_pos[0]:cursor_pos[0] + img.shape[0], cursor_pos[1]:cursor_pos[1] + img.shape[1], : ] = img
        cursor_pos[0] += image_shape_max[0]  # increment cursor x position
        if cursor_pos[0] >= montage_shape[0] * image_shape_max[0]:
            cursor_pos[1] += image_shape_max[1]  # increment cursor y position
            cursor_pos[0] = 0
            if cursor_pos[1] >= montage_shape[1] * image_shape_max[1]:
                cursor_pos = [0, 0]
                image_montages.append(montage_image)
                # reset black canvas
                montage_image[cursor_pos[0]:cursor_pos[0] + img.shape[0], cursor_pos[1]:cursor_pos[1] + img.shape[1], :] = img

                start_new_img = True
    if start_new_img is False:
        image_montages.append(montage_image)  # add unfinished montage

    for ii, img in enumerate(image_montages):
        imgsize = (img.shape[1] / dpi, img.shape[0] / dpi)
        fig, axs = plt.subplots(1, 1, figsize=imgsize, dpi=dpi)
        plt.subplots_adjust(wspace=0, hspace=0, left=0.0, right=1.0, bottom=0.0, top=1.0)
        axs.imshow(img)  # , origin = "lower")
        axs.axis("off")
        if fig_path is not None:
            ff = fig_path +"_{}.png".format(ii)
            fig.savefig(ff, facecolor="w", bbox_inches='tight')
            print("Saving {}".format(ff))


def scaling_func(im, mask, scaling, percentile_values):
    """
    This function takes as input the image (3D), the corresponding mask (can be None), the type of scaling and the saling values.
    This last input is an array-like of 2 elements coorresponding to the parameter of np.percentile
    for v_min and v_max.
    The output of this function is the couple of values used for the scaling of the image, taken on the whole image or on the mask,
    and can also gives the couple to use on the mask and the one to use in the font.
    """
    if scaling == "whole":
        im_values = im.get_fdata()
    else :
        im_values = im.get_fdata()[mask.get_fdata().astype(bool)]
    
    v_min = np.percentile(im_values, percentile_values[0])
    v_max = np.percentile(im_values, percentile_values[1])
    if scaling == "mask_font":
        font_values = im.get_fdata()[np.logical_not(mask.get_fdata().astype(bool))]
        v_min_f = np.percentile(font_values, percentile_values[0])
        v_max_f = np.percentile(font_values, percentile_values[1])
        return v_min, v_max, v_min_f, v_max_f
    return v_min, v_max, -1, -1            
            
def get_acoreg(acoreg):

    if isinstance(acoreg, str):
        acoreg = np.loadtxt(acoreg, delimiter=' ')
        acoreg = np.linalg.inv(acoreg)
    elif (isinstance(acoreg, np.ndarray)):
        acoreg = acoreg
    else:
        # raise TypeError("acoreg of incorrect type for case {}".format(i))
        print('no acoreg')
    return acoreg

def my_get_image(im, ras=True):

    if im is None :
        return

    if isinstance(im, str):
        im = nb.load(im)
    elif (isinstance(im, nb.nifti1.Nifti1Image)):
        im = im
    else:
        raise TypeError("im of incorrect type for case {}".format(im))

    if ras and nb.aff2axcodes(im.affine) != ('R', 'A', 'S'):
        print('changing image affine to canonical because {}... '.format(nb.aff2axcodes(im.affine)))
        im = nb.as_closest_canonical(im)
    return im


def do_figures_from_file(l_in, slices_infos=None, mask_info=None, fref = None, display_order=None, ras = True, colormap = cm.Greys_r,
                     colormap_noise=cm.hot, percentile_values = [0,99], plot_single=True, out_dir=None,
                     montage_shape=None, dpi=50, plt_ioff=False):

    if plt_ioff: plt.ioff();

    dir_fig = '{}/figures/'.format(out_dir)
    if not os.path.exists(dir_fig):  os.makedirs(dir_fig)

    fref = my_get_image(fref)

    matrix_all=[]
    for i, item in enumerate(l_in):
        im, mask, acoreg = item

        im = my_get_image(im, ras=ras)
        mask = my_get_image(mask, ras=ras)
        acoreg = get_acoreg(acoreg)

        fig_path = dir_fig + "/fig_" + str(i) + ".png"
        matrix_fig = plot_view(im, mask, fref, acoreg, slices_infos, mask_info, display_order=display_order,
                               colormap=colormap,colormap_noise=colormap_noise, percentile_values=percentile_values,
                               figure_path=fig_path, plot_single=plot_single, dpi=dpi)

        if montage_shape is not None:
            matrix_all.append(matrix_fig)


    if montage_shape is not None:
        fig_path = dir_fig + "/m_fig_"
        plot_montages(matrix_all, montage_shape, fig_path=fig_path, dpi=dpi)

    #return matrix_all
    #build_montages(matrix_all)

def get_nibabel_from_sample_dict(img_dict):

    data = img_dict['data']
    if data.ndim==5:
        image = data[0][0].numpy()
        affine = img_dict['affine'][0]
    elif data.ndim==4:
        image = data[0].numpy()
        affine = img_dict['affine']

    nii = nb.Nifti1Image(image, affine)
    return nii


def do_figure_from_dataset(td, select_indices= None, name_fig=None, mask_key=None,
                           slices_infos=None, mask_info=None, fref = None, display_order=None, ras=True,
                           colormap = cm.Greys_r, colormap_noise=cm.hot, percentile_values = [0,99],
                           plot_single=True, out_dir=None, montage_shape=None, dpi=50, plt_ioff=True, montage_basename='m_fig'
                           ):

    if plt_ioff: plt.ioff();

    if select_indices is None:
        select_indices = range(0, len(td)) #plot all

    dir_fig = '{}/fig/'.format(out_dir)
    if not os.path.exists(dir_fig):  os.makedirs(dir_fig)

    fref = my_get_image(fref)

    matrix_all=[]
    for ind, ind_sel in enumerate(select_indices.tolist()):

        s = td[ind_sel]

        im = get_nibabel_from_sample_dict(s['image'])
        mask = None
        if mask_key is not None:
            if mask_key in s:
                mask = get_nibabel_from_sample_dict(s[mask_key])
            else:
                print('WARNING no mask for {}'.format(ind_sel))

        acoreg = None

        if name_fig is not None:
            fig_path = dir_fig + "/" + name_fig[ind] + ".png"
        else:
            fig_path = dir_fig + "/fig_" + str(ind) + ".png"

        matrix_fig = plot_view(im, mask, fref, acoreg, slices_infos, mask_info, display_order=display_order,
                               colormap=colormap,colormap_noise=colormap_noise, percentile_values=percentile_values,
                               figure_path=fig_path, plot_single=plot_single, dpi=dpi, plt_ioff = plt_ioff)

        if montage_shape is not None:
            matrix_all.append(matrix_fig)


    if montage_shape is not None:
        fig_path = dir_fig + "/{}".format(montage_basename)
        plot_montages(matrix_all, montage_shape, fig_path=fig_path, dpi=dpi)

    #return matrix_all
    #build_montages(matrix_all)


# %%
from time import time

if __name__ == "__main__":

    import pandas as pd
    import utils_file as uf
    from slices_2 import *

    if 0==3:
        ds = pd.read_csv('/home/romain.valabregue/datal/QCcnn/res/res_cat12seg_18999.csv')
        rootdir = '/network/lustre/iss01/scratch/CENIR/users/romain.valabregue/dicom/nifti_proc'

        ind_sel = np.random.randint(0,ds.shape[0],2)
        din = ds.iloc[ind_sel,1] #[ds.iloc[ii,1] for ii in ind_sel]
        din_list = din.tolist()
        din_list = ["/network/lustre/iss01/"+s for s in din_list]
        fin = uf.gfile(din_list ,'^s.*nii.gz')
        print(din)
        faff = uf.gfile(din_list, '^aff.*txt')
        fmask = uf.gfile(din_list, '^niw_Mean')
        #fmask = [None for i in range(0,3,1)]
        l_view = [("sag", "vox", 0.5), ("sag", "voxmm", -32), ("sag", "mm", -32), ("sag", "mm_mni", -32),
                  ("ax", "vox", 0.5), ("ax", "voxmm", -43), ("ax", "mm", -43), ("ax", "mm_mni", -43),
                  ("cor", "vox", 0.5), ("cor", "voxmm", 54), ("cor", "mm", 54), ("cor", "mm_mni", 54), ]
        display_order = np.array([4, 3])  # row and column of the montage
        fref = nb.load(
            '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/dicom/mni/tpl_mni_aff/mean_rmni1Kcrop.nii.gz')
        fref = nb.load('/home/romain.valabregue/datal/HCPdata/suj_100307/T1w_1mm.nii.gz')
        mask_info = [("mask", -1) for i in range(0, 12, 1)]


    d='/home/romain/QCcnn/mask_mvt_train_cati_T1/'
    fin = uf.gfile(d,'s_S07_3DT1.nii')
    faff = [None]
    fmask = [None] #uf.gfile(d,'niw_di')
    fref = None

    l_view = [("sag", "vox", 0.4), ("cor", "vox", 0.6),
              ("ax", "vox", 0.5), ]
    display_order = np.array([1, 3]) # row and column of the montage
    mask_info = [("mask_font", -1) for i in range(0,3,1)] #overlay with colormap jet in the background
    mask_info = [("mask", 1) for i in range(0, 3, 1)]  # cut around the mask (at the slice level)
    mask_info = [("mask", -1) for i in range(0, 3, 1)]  # min max within the mask
    mask_info = [("mask", 1) ]  # min max within the mask

    l_in = uf.concatenate_list([fin, fmask, faff])
    l_in.append(l_in[0])
    l_in.append(l_in[0])
    l_in.append(l_in[0])
    l_in.append(l_in[0])
    l_in.append(l_in[0])
    l_in.append(l_in[0])


    t0 = time()
    fig = do_figures_from_file(l_in, slices_infos=l_view, mask_info=mask_info, display_order=display_order,
                           fref=fref,out_dir=d, plot_single=True, montage_shape=(2,3), plt_ioff=False )

    print("Il a fallu {} secondes".format(np.round(time()-t0, 2)))


