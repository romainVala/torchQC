"""The purpose is to create the Yeb mask for all the subjects """

import numpy as np
import nibabel as nb
import os
import pandas as pd
import shutil
from tqdm import tqdm
from nibabel.processing import resample_from_to

from scipy.ndimage.morphology import  binary_erosion, binary_dilation

# ----- main function ------
do_erode = True
do_import = True
nb_iterations=2

def main():
    path_hcp = '/network/lustre/iss02/opendata/data/HCP/raw_data/nii'
    path_processed = '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/datai/connectomeV3'
    path_csv = "/data/romain/HCP_yeb/training_label.csv"
    path_data = "/data/romain/HCP_yeb/data/" #"data_yeb/"
    path_data = "/data/romain/HCP_yeb/data_erode2/" #"data_yeb/"
    path_table = "/data/romain/HCP_yeb/yeb_label.csv"
    path_label = "/data/romain/HCP_yeb/yeb_new_label.csv"

    # ----- we select the subjects ----
    subjects = find_subjects(path_hcp, path_processed)
    subjects.remove('390645')
    subjects.remove('499566')
    if do_import:
        #-------- copy of the masks -------
        os.mkdir(path_data)
        create_folder(path_data, path_hcp, path_processed, path_table)

        #------------processing------------
        print('Creation WM')
        creation_WM(path_data, path_hcp, path_csv)
    print('Preprocessing')
    process(path_data, path_table, path_hcp)
    print('Label Creation')
    modification_label_map(path_data, path_label)


# ---- Function necessary to create the folders and move files -----


def find_subjects(path_hcp, path_processed):
    """find all the subjects we can use for the training"""
    sub_hcp = os.listdir(path_hcp)
    sub_processed = os.listdir(path_processed)
    subjects = [value for value in sub_hcp if value in sub_processed if value.isdigit()]
    return subjects


def create_folder(path_data, path_hcp, path_processed, path_table):
    """create our data folder and copy all the necessary files for process inside"""

    subjects = find_subjects(path_hcp, path_processed)
    subjects.remove('499566')
    subjects.remove('390645')

    print("Creation of all the subject folders")
    create_subject_folders(subjects, path_data)

    print("The copy of all labels will start")
    for i in tqdm(range(len(subjects))):
        copy_labels(path_hcp, path_processed, subjects[i], path_table, path_data)

    print("All copies have succeed")


def create_subject_folders(subjects, path_data):
    """create a folder for each subjects"""
    for subject in subjects:
        os.mkdir(path_data + subject)


def copy_labels(path_hcp, path_processed, subject, path_table, path_data):
    """ copy the masks necessary in the subject folder """
    table = pd.read_csv(path_table, sep=',')
    for i in range(1, len(table)):
        if table.at[i, 'path'] == '/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/suj/T1w/ROI_PVE_1mm':
            files = table.at[i, 'file'].split(' + ')
            for file in files:
                shutil.copy(path_hcp + '/' + subject + '/T1w/ROI_PVE_1mm/' + file, path_data + subject + '/' + file)

        else:
            files = table.at[i, 'file'].split(' + ')
            for file in files:
                shutil.copy(path_processed + '/' + subject + '/NGC/ROIs_espace_natif_T1/' + file,
                            path_data + subject + '/' + file)


# ------- reconstruction WM --------


def creation_WM(path_data, path_hcp, path_csv):
    """ add the WM to all the subjects """
    subjects = os.listdir(path_data)

    print(creation_WM)
    for i in tqdm(range(len(subjects))):
        reconstruction_WM(subjects[i], path_hcp, path_data, path_csv)


def reconstruction_WM(subject, path_hcp, path_data, path_csv):
    """Sum the mask of all nuclei with the mask of WM for one subject"""
    csv = pd.read_csv(path_csv)
    matter = nb.load(path_hcp + '/' + subject + '/T1w/ROI_PVE_1mm/' + 'WM.nii.gz')
    data = matter.get_fdata()
    affine = matter.affine

    for i in range(1, len(csv)):
        if csv.at[i,'nucleus'] == 'yes':
            print(i)
            data += nb.load(path_hcp + '/' + subject + '/T1w/ROI_PVE_1mm/' + csv.at[i, 'file']).get_fdata()

    nb.save(nb.Nifti1Image(data, affine), path_data + subject + '/' + 'WM.nii.gz')


def test_reconstruction(subject, processed_label, path_hcp):
    """test the result obtained before"""
    mask_WM = reconstruction_WM(subject, processed_label, path_hcp)
    for i in range(1, 3):
        mask_WM += nb.load(path_hcp + '/' + subject + '/T1w/ROI_PVE_1mm/' + processed_label.at[i, 'file']).get_fdata()
    for j in range(10, 14):
        mask_WM += nb.load(path_hcp + '/' + subject + '/T1w/ROI_PVE_1mm/' + processed_label.at[j, 'file']).get_fdata()
    test = nb.Nifti1Image(mask_WM, np.eye(4))
    nb.save(test, "test.nii.gz")


# ------- processing and function necessary to run it   -------

def process(path_data, path_final_table, path_hcp):
    # subject creation
    subjects = os.listdir(path_data)

    #sided_label = ['STRCD', 'STRPU', 'STRAC', 'THAL', 'THRPT', 'EGP', 'IGP', 'STN', 'SN', 'RU']
    sided_label = ['STRCD', 'STRPU', 'STRAC', 'THAL_THRPT', 'EGP_IGP' , 'STN', 'SN', 'RU']

    for i in tqdm(range(len(subjects))):

        sum_mask(subjects[i], path_data, ['IGP_RH.nii.gz', 'EGP_RH.nii.gz'], 'EGP_IGP_RH.nii.gz')
        sum_mask(subjects[i], path_data, ['IGP_LH.nii.gz', 'EGP_LH.nii.gz'], 'EGP_IGP_LH.nii.gz')
        sum_mask(subjects[i], path_data, ['THAL_LH.nii.gz', 'THRPT_LH.nii.gz'], 'THAL_THRPT_LH.nii.gz')
        sum_mask(subjects[i], path_data, ['THAL_RH.nii.gz', 'THRPT_RH.nii.gz'], 'THAL_THRPT_RH.nii.gz')

        #-------- merge left right -------
        for label in sided_label:
            #print(f'mergin {label}')
            merge_LH_RH(subjects[i], label, path_data, erode=do_erode)

        #-------  merge of different labels --------
        overlap_gestion('STRAC.nii.gz', 'STRCD.nii.gz', path_data, subjects[i])
        #sum_mask(subjects[i], path_data, ['IGP.nii.gz', 'EGP.nii.gz'], 'EGP_IGP.nii.gz')
        #sum_mask(subjects[i], path_data, ['THAL.nii.gz', 'THRPT.nii.gz'], 'THAL_THRPT.nii.gz')

        table = pd.read_csv(path_final_table).values[:, 4]
        resolution_labels = ['WM.nii.gz', 'GM_allc.nii.gz', 'CSF.nii.gz', 'cereb_GM.nii.gz', 'skin.nii.gz',
                             'skull.nii.gz', 'background.nii.gz']

        target_img = nb.load(path_data + subjects[i] + '/WM.nii.gz')
        max_value = target_img.get_fdata().max()

        #-------- resample images to get the same resolution -------
        for label in table:
            if label not in resolution_labels:
                im = nb.load(path_data + subjects[i] + '/' + label)
                im = resample_from_to(im, target_img, mode='nearest')
                imdata = im.get_fdata()
                imdata = (imdata > (max_value/2)).astype('float64')  #romain add, binarize (max value on yeb is 256
                #imdata = binary_erosion(imdata) #(erosion before resampling)
                new_im = nb.Nifti1Image(imdata, im.affine, im.header)
                nb.save(new_im, path_data + subjects[i] + '/' + label)

        # -------add nuclei to WM------------
        nuclei = [label for label in table if label not in resolution_labels]

        for nucleus in nuclei:
            for label in resolution_labels:
                overlap_gestion(nucleus, label, path_data, subjects[i])


def merge_LH_RH(subject, label, path_data, erode=False):
    """ merge label_LH.nii.gz with label_RH.nii.gz"""
    return sum_mask(subject, path_data, [label + "_LH.nii.gz", label + "_RH.nii.gz"], label + ".nii.gz", erode=erode)


def sum_mask(subject, path_data, list_mask, file_name, erode=False):
    """"we need to sum some mask to have the good labels"""
    im = nb.load(path_data + subject + '/' + list_mask[0])
    data = im.get_fdata()
    #data = data/ data.max() #rrr make it binary  arg som intersection have value above 1
    data = (data>0).astype('float64')
    if erode:
        data = binary_erosion(data, iterations=nb_iterations).astype('float64')

    affine = im.affine

    for i in range(1, len(list_mask)):
        data_add = nb.load(path_data + subject + '/' + list_mask[i]).get_fdata()
        data_add = (data_add > 0).astype('float64')#rrr make it binary

        if erode:
            data_add = binary_erosion(data_add, iterations=nb_iterations).astype('float64')
            print(f'eroding {subject} before sum { list_mask[0]} and { list_mask[i]}')

        data += data_add

    data = (data > 0).astype('float64')
    nb.save(nb.Nifti1Image(data, affine), path_data + subject + '/' + file_name)


def overlap_gestion(label_prior, label, path_data, subject):
    """correct the two mask to deal with overlap issues"""
    im_lab = nb.load(path_data + subject + '/' + label)
    affine = im_lab.affine

    data_lab = im_lab.get_fdata() - nb.load(path_data + subject + '/' + label_prior).get_fdata()
    nb_neg = np.sum(data_lab<0)
    nb_neg_to_reset = np.sum(im_lab.get_fdata()[data_lab < 0])
    if  nb_neg_to_reset>0.01 :
        print(f'removing {nb_neg}  pint { label_prior} from  { label}')
        data_lab[data_lab < 0] = 0
        nb.save(nb.Nifti1Image(data_lab, affine), path_data + subject + '/' + label)
        if 'skull' in label:
            qsdf

def overlap_nuclei(path_table, subjects, path_data):
    """remove all nuclei from WM GM and CSF"""
    glob_label = pd.read_csv(path_table, sep=",").values[:, 3:]
    print("glob label, ", glob_label[:, 0])
    labels = []
    labels_prior = []
    for i in range(len(glob_label)):
        if glob_label[i, 0] == '/network/lustre/dtlake01/opendata/data/HCP/raw_data/nii/suj/T1w/ROI_PVE_1mm':
            labels.append(glob_label[i, 1])
        else:
            labels_prior.append(glob_label[i, 1])
    print(len(labels))
    print(len(labels_prior))
    for subject in subjects:
        for label in labels:
            for label_prior in labels_prior:
                overlap_gestion(label_prior, label, path_data, subject)
    print("finish")


# ----- creation of the label map for SynthSeg training ------

def modification_label_map(path_data, path_csv):
    """ transform labels into Billot label map"""
    labels = pd.read_csv(path_csv, sep=',')
    subjects = os.listdir(path_data)

    for i in tqdm(range(len(subjects))):
        im_label = []
        # looking for the files

        for j in range(labels.shape[0]):
            im = nb.load(path_data + subjects[i] + '/' + labels.at[j, 'file'])
            im_label.append(im.get_fdata())

        affine = im.affine
        im_label = np.array(im_label)

        # determination of the label maps
        data_label = np.argmax(im_label, axis=0)

        for j in range(labels.shape[0]):
            data_label[data_label == j] = labels.at[j, 'label']

        img = nb.Nifti1Image(data_label, affine)
        nb.save(img, path_data + subjects[i] + "/" + "label_Billot.nii.gz")


# --------- SHELL -----------
main()
