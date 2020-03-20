#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import nibabel.processing as npi
import matplotlib.cm as cm
import os


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
    
    print(type_pos != "mm")
    #if type_pos != "mm":
    if type_pos == "mm_mni":
        print('changing the image affine before resamping')
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
 #   if fref is None and type_pos == "mm":
 #       raise  AssertionError("Trying to use mm view with no fref")
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

        
    
# %%
def scaling_func(im, mask, scaling, scaling_values):
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
    
    v_min = np.percentile(im_values, scaling_values[0])
    v_max = np.percentile(im_values, scaling_values[1])
    if scaling == "mask_font":
        font_values = im.get_fdata()[np.logical_not(mask.get_fdata().astype(bool))]
        v_min_f = np.percentile(font_values, scaling_values[0])
        v_max_f = np.percentile(font_values, scaling_values[1])
        return v_min, v_max, v_min_f, v_max_f
    return v_min, v_max, -1, -1            
            


# %%

def generate_figures(l_in, slices_infos=None, mask_info=None, fref = None, display_order=None, ras = True, colormap = cm.Greys_r,
                     colormap_noise=cm.hot, scaling_values = [0,100], figsize = (20,20)):
    """
    Takes as inputs:
        A list of tuples (the image, the associated mask, the associated acoreg)
        A list of tuples (view, type_pos, pos)
        A list giving the info about the scaling and the mask cut for each slice (if no mask_cut, put it to -1)
        The template reference for the mm case
        An array giving information about the disposition of the images
        A parameter saying if we convert all the images to RAS format (True by default)
        The colormap used for the different kinfd of scalings
        The percentages used for the scaling
        A size parameter
        
    And produces for each image as input an image (in png) composed of the differents views wanted (produced in the folder 
    computed_images) and a histogram (in the file histogram) giving the repartition of the values on the mask 
    """
    display_order = display_order[::-1]
    if not os.path.exists("histogramme"):
        os.makedirs("histogramme")
    if not os.path.exists("computed_images"):
        os.makedirs("computed_images")
    if isinstance(fref, str):
        fref = nb.load(fref)
    elif (isinstance(fref, nb.nifti1.Nifti1Image) or fref is None):
        fref = fref
    else:
        raise TypeError("fref of incorrect type")
    
    if len(slices_infos) != len(mask_info):
        raise AssertionError("The 2 lists about the slices must have the same size")
    
    if ras and fref is not None and nb.aff2axcodes(fref.affine) != ('R', 'A', 'S'):
        print('changing reference affine to canonical ... ')
        fref = nb.as_closest_canonical(fref)
        
    use_reslice_mm = (np.array([slices_infos[k][1] for k in range(len(slices_infos))]) == "mm").any()
    use_reslice_mni = (np.array([slices_infos[k][1] for k in range(len(slices_infos))]) == "mm_mni").any()

    print(l_in)
    for i, item in enumerate(l_in):
        im, mask, acoreg = item
        
        if isinstance(im, str):
            im = nb.load(im)
        elif (isinstance(im, nb.nifti1.Nifti1Image)):
            im = im
        else:
            raise TypeError("im of incorrect type for case {}".format(i))
        
        if ras and nb.aff2axcodes(im.affine) != ('R', 'A', 'S'):
            print('changing image affine to canonical ... ')
            im = nb.as_closest_canonical(im)
        
        if isinstance(mask, str):
            mask = nb.load(mask)
        elif (isinstance(mask, nb.nifti1.Nifti1Image) or mask is None):
            mask = mask
        else:
            raise TypeError("mask of incorrect type for case {}".format(i))
        
        if ras and mask is not None and nb.aff2axcodes(mask.affine) != ('R', 'A', 'S'):
            mask = nb.as_closest_canonical(mask)

        ##rrr should had a test to check dimention of mask and images is matching

        if isinstance(acoreg, str):
            acoreg = np.loadtxt(acoreg,delimiter=' ')
            acoreg = np.linalg.inv(acoreg)
        elif (isinstance(acoreg, np.ndarray)):
            acoreg = acoreg
        else:
            raise TypeError("acoreg of incorrect type for case {}".format(i))
        
        if use_reslice_mm:
            im_resliced_mm, _, _ = reslice_im(im, fref, acoreg, type_pos="mm")
            mask_resliced_mm = reslice_mask(mask, fref, acoreg, type_pos="mm")
        
        if use_reslice_mni:
            im_resliced_mni, _, _ = reslice_im(im, fref, acoreg, type_pos="mm_mni")
            mask_resliced_mni = reslice_mask(mask, fref, acoreg, type_pos="mm_mni")
        
        if mask is not None:
            plt.hist(im.get_fdata()[mask.get_fdata().astype(bool)].ravel(), bins = 200)
            plt.savefig("histogramme/histogramme_"+str(i)+".png")
        else :
            plt.hist(im.get_fdata().ravel(), bins = 200)
            plt.savefig("histogramme/histo_all_"+str(i)+".png")

        list_matrix = []
        
        for j, item2 in enumerate(slices_infos):
            view, type_pos, pos = item2
            scaling, mask_cut_pix = mask_info[j]
            if mask is None and (scaling != "whole" or mask_cut_pix >=0):
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
            v_min, v_max, v_min_f, v_max_f = scaling_func(temp_im, temp_mask, scaling, scaling_values)
            matrix_slice, mask_slice = get_slices(temp_im, temp_mask, acoreg, view, type_pos, pos, mask_cut_pix)
            list_matrix.append((matrix_slice, mask_slice,v_min, v_max, v_min_f, v_max_f))
            
        #tab_abs = np.array([[list_matrix[i*display_order[1]+k][0].shape[1] for k in range(display_order[1]) ] for i in range(display_order[0])])
        #tab_ord = np.array([[list_matrix[i*display_order[1]+k][0].shape[0] for k in range(display_order[1]) ] for i in range(display_order[0])])
        #abs_max= np.max(np.sum(tab_abs, axis = 1))
        #ord_max = np.max(np.sum(tab_ord, axis = 0))
        
        abs_max= np.max([m[0].shape[1] for m in list_matrix])
        ord_max= np.max([m[0].shape[0] for m in list_matrix])

        figure = np.zeros((display_order[1]*abs_max, display_order[0]*ord_max, 4)).astype(np.uint8)
        curseur_abs = 0
        curseur_ord=0
        for j, item in enumerate(list_matrix):
            matrix_slice, mask_slice,v_min, v_max, v_min_f, v_max_f = item
            if mask_info[j][0] != "mask_font":
                temp = (np.uint8(255*(colormap(matrix_slice.T/(v_max - v_min))).astype(np.float64)))
                figure[curseur_abs:curseur_abs+temp.shape[0],
                      curseur_ord:curseur_ord+temp.shape[1]]= np.flipud(temp)
                      
            else:
                temp = (np.uint8(255*
                     (colormap_noise((matrix_slice.T/(v_max_f - v_min_f) * np.logical_not(mask_slice).T).astype(np.float64))
                     +(colormap((matrix_slice/(v_max - v_min) * mask_slice).T).astype(np.float64)))))
                figure[curseur_abs: curseur_abs+temp.shape[0],
                      curseur_ord:curseur_ord+temp.shape[1]] = np.flipud(temp)
            if (j+1)%display_order[1] == 0:
                curseur_abs = 0
                curseur_ord += ord_max
            else:
                curseur_abs += abs_max
                     
        #figure=np.transpose(figure, (0,1,2))
        #fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0, left=0.0, right=1.0, bottom=0.0, top=1.0)
        fig, axs = plt.subplots(1,1,figsize = (40, 40))
        axs.imshow(figure)#, origin = "lower")
        #for k in range(display_order[1]):
        #    plt.plot((k*display_order[1], 0), (y1, y2), 'k-')
        axs.axis("off")
        fig.savefig("computed_images/fig_"+str(i)+".png",   facecolor ="k", bbox_inches='tight')
        print("Done for figure "+str(i))

    #return figure


# %%
from time import time

if __name__ == "__main__":

    #mask_1 = nb.load("/home/dimitri.hamzaoui/Documents/cat12/nr_mask_brain_erode_dilate.nii.gz")
    #fref = nb.load('./data_QC/mni/tpl_mni_aff/mean_rmni1Kcrop.nii.gz')
    #acoreg_inv = np.loadtxt("./Documents/cat12/aff_nr_ms_S10_t1mpr_SAG_NSel_S176_to_Mean_S50_all.txt",delimiter=' ')
    #acoreg = np.linalg.inv(acoreg_inv)
    
    #l_in = [(test_1 , None, acoreg)]
    #l_view = [("sag", "mm", 0), ("sag", "mm_mni", 0)]
    #mask_info = [("whole", -1), ("whole", -1), ("whole", -1)]
    #display_order = np.array([1, 2])
    #t0 = time()
    #fig = generate_figures(l_in, slices_infos=l_view, mask_info=mask_info, display_order=display_order, fref = fref, figsize = (10, 10))
    #print("Il a fallu {} secondes".format(np.round(time()-t0, 2)))
    #plt.imshow(fig, origin = "lower")

    #rrr
    import pandas as pd
    import utils_file as uf
    import importlib
    from slices_2 import *
    importlib.reload(uf)

    ds = pd.read_csv('/home/romain.valabregue/datal/QCcnn/res/res_cat12seg_18999.csv')
    rootdir = '/network/lustre/iss01/scratch/CENIR/users/romain.valabregue/dicom/nifti_proc'

    ind_sel = np.random.randint(0,ds.shape[0],2)
    din = ds.iloc[ind_sel,1] #[ds.iloc[ii,1] for ii in ind_sel]
    din_list = din.tolist()
    din_list = ["/network/lustre/iss01/"+s for s in din_list]
    fin = uf.gfile(din_list ,'^s.*nii.gz')
    print(din)
    faff = uf.gfile(din_list,'^aff.*txt')
    l_view=[("sag", "mm", 0) for i in range(0,3,1)]
    l_view = [("sag", "vox", 0.5),("sag", "mm", 0),("sag", "mm_mni", 0),
              ("ax", "vox", 0.5), ("ax", "mm", 0), ("ax", "mm_mni", 0),
              ("cor", "vox", 0.5), ("cor", "mm", 0), ("cor", "mm_mni", 0),]

    l_view = [("sag", "vox", 0.5), ("sag", "voxmm", -32), ("sag", "mm", -32), ("sag", "mm_mni", -32),
              ("ax", "vox", 0.5),  ("ax", "voxmm", -43),  ("ax", "mm", -43),  ("ax", "mm_mni", -43),
              ("cor", "vox", 0.5), ("cor", "voxmm", 54),  ("cor", "mm", 54),  ("cor", "mm_mni", 54),]

    mask_info = [("whole", -1) for i in range(0,12,1)]
    display_order =  np.array([4, 3]) # row and column of the montage

    fref=nb.load('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/dicom/mni/tpl_mni_aff/mean_rmni1Kcrop.nii.gz')
    fref = nb.load('/home/romain.valabregue/datal/HCPdata/suj_100307/T1w_1mm.nii.gz')

    fmask = [None for i in range(0,3,1)]
    l_in = uf.concatenate_list([fin,fmask,faff])
    print(l_in)
    t0 = time()
    fig = generate_figures(l_in, slices_infos=l_view, mask_info=mask_info, display_order=display_order, fref = fref, figsize = (10, 10))
    print("Il a fallu {} secondes".format(np.round(time()-t0, 2)))

    l_view = [("sag", "vox", 0.4), ("ax", "vox", 0.4), ("cor", "vox", 0.5), ("ax", "vox", 0.6)]
    mask_info = [("whole", -1) for i in range(0,4,1)]

    display_order = np.array([2,2])