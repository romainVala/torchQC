import operator
import numpy as np
# import keras_preprocessing.image.affine_transformations as at
import scipy.ndimage as ndi
import subprocess
import nibabel as nb
import os, sys
from time import sleep, time
import sklearn.metrics as metrics
import pandas as pd

operator_dict = {
    '==': operator.eq,
    '>': operator.gt,
    '<': operator.lt,
    "|": operator.or_,
    "&": operator.and_
}


def apply_conditions_on_dataset(dataset, conditions, min_index=None, max_index=None):
    """
    Conditions of the form ((intermediate_op,) var, op, values).
    Takes one or several conditions (each condition must be in a 3-tuple or a 4-tuple, each block of conditions in a list)
    and a dataframe, and returns the part of the dataframe respecting the conditions.
    """
    if max_index is not None or min_index is not None:
        if max_index is None:
            dataset = dataset.iloc[min_index:]
        elif min_index is None:
            dataset = dataset.iloc[:max_index]
        else:
            dataset = dataset.iloc[min_index:max_index]
    for p_c in conditions:
        if type(p_c) == list:
            if len(p_c[0]) == 3:
                snap = apply_conditions_on_dataset(dataset, p_c)
            else:
                temp_var = p_c[0][0]
                p_c[0] = p_c[0][1:]
                temp_snap = apply_conditions_on_dataset(dataset, p_c)
                snap = operator_dict[temp_var](snap.copy(), temp_snap)
        elif len(p_c) == 3:
            snap = operator_dict[p_c[1]](dataset.copy()[p_c[0]], p_c[2])
        elif len(p_c) == 4:
            snap = operator_dict[p_c[0]]((snap.copy()), (operator_dict[p_c[2]](dataset.copy()[p_c[1]], p_c[3])))
    return snap


def quadriview_old(nifti_image, slice_sag, slice_orth,
                   slice_cor_1, slice_cor_2):
    """
    Takes as input a 3D image and the 4 positions of slices, and returns the fabricated image (sag - cor/ ax1 - ax2) and return 
    """
    view_1 = nifti_image[slice_sag, :, :]
    view_2 = nifti_image[:, slice_orth, :]
    view_3 = nifti_image[:, :, slice_cor_1]
    view_4 = nifti_image[:, :, slice_cor_2]

    pad_lign = max(view_1.shape[0] + view_2.shape[0], view_3.shape[0] + view_4.shape[0])
    pad_col = max(view_1.shape[1] + view_3.shape[1], view_2.shape[1] + view_4.shape[1])
    pad = np.zeros((pad_lign, pad_col), dtype=np.float64)

    pad[:view_1.shape[0], :view_1.shape[1]] = view_1
    pad[-view_2.shape[0]:, :view_2.shape[1]] = view_2
    pad[:view_3.shape[0], -view_3.shape[1]:] = view_3
    pad[-view_4.shape[0]:, -view_4.shape[1]:] = view_4

    return pad


def quadriview(nifti_title, prefix, slices_array=None):
    """
    Takes as input a 3D image and the 4 positions of slices, and returns the fabricated image (sag - cor/ ax1 - ax2) and return 
    """
    # img = np.load(prefix + nifti_title)
    img = nb.load(prefix + nifti_title)
    to = time()
    if slices_array is None:
        slice_sag = np.random.randint(0.2 * img.shape[0], 0.8 * img.shape[0])
        slice_orth = np.random.randint(0.2 * img.shape[1], 0.8 * img.shape[1])
        slice_cor_1 = np.random.randint(0.3 * img.shape[2], 0.5 * img.shape[2])
        slice_cor_2 = np.random.randint(0.5 * img.shape[2], 0.7 * img.shape[2])
    else:
        slice_sag, slice_orth, slice_cor_1, slice_cor_2 = slices_array
    nifti_image = img.dataobj
    view_1 = nifti_image[slice_sag, :, :]
    view_2 = nifti_image[:, slice_orth, :]
    view_3 = nifti_image[:, :, slice_cor_1]
    view_4 = nifti_image[:, :, slice_cor_2]

    pad_lign = max(view_1.shape[0] + view_2.shape[0], view_3.shape[0] + view_4.shape[0])
    pad_col = max(view_1.shape[1] + view_3.shape[1], view_2.shape[1] + view_4.shape[1])
    pad = np.zeros((pad_lign, pad_col), dtype=np.float64)
    pad[:view_1.shape[0], :view_1.shape[1]] = view_1
    pad[-view_2.shape[0]:, :view_2.shape[1]] = view_2
    pad[:view_3.shape[0], -view_3.shape[1]:] = view_3
    pad[-view_4.shape[0]:, -view_4.shape[1]:] = view_4
    return pad


def quadriview_tep(args):
    nifti_title, prefix, slices_array = args
    return quadriview(nifti_title, prefix, slices_array)


def quadriview_V2(nifti_image, slice_sag, slice_orth,
                  slice_cor_1, slice_cor_2):
    """
    Takes as input a 3D image and the 4 positions of slices, and returns the fabricated image (sag - cor/ ax1 - ax2) and return 
    """

    view_1 = nifti_image[slice_sag, :, :]
    view_2 = nifti_image[:, slice_orth, :]
    view_3 = nifti_image[:, :, slice_cor_1]
    view_4 = nifti_image[:, :, slice_cor_2]

    max_dim = np.max(nifti_image.shape)
    min_dim = np.min(nifti_image.shape)

    view_1 = np.pad(view_1, ((0, 0), (max_dim - min_dim, 0)), mode="constant")
    view_2 = np.pad(view_2, ((max_dim - min_dim, 0), (max_dim - min_dim, 0)), mode="constant")
    view_3 = np.pad(view_3, ((max_dim - min_dim, 0), (0, 0)), mode="constant")
    view_4 = np.pad(view_4, ((max_dim - min_dim, 0), (0, 0)), mode="constant")

    pad = np.block([[view_1, view_2], [view_3, view_4]])

    return pad


def take_slice(img_3D, view):
    """
   Takes as input a 3D image and the wanted view, and returns a slice with the wanted view taken in a random point.
   """
    input_type = isinstance(img_3D, np.ndarray)
    if input_type:
        img_3D = [img_3D]
    img_shape = img_3D[0].shape
    if view == "sag":
        slice_pos = np.random.randint(int(0.2 * img_shape[0]), int(0.8 * img_shape[0]))
        imgs_2D = [imgg_3D[slice_pos, :, :] for imgg_3D in img_3D]
    elif view == "cor":
        slice_pos = np.random.randint(int(0.2 * img_shape[1]), int(0.8 * img_shape[1]))
        imgs_2D = [imgg_3D[:, slice_pos, :] for imgg_3D in img_3D]
    else:
        slice_pos = np.random.randint(int(0.2 * img_shape[2]), int(0.8 * img_shape[2]))
        imgs_2D = [imgg_3D[:, :, slice_pos] for imgg_3D in img_3D]
    # img_2D = np.expand_dims(img_2D, 2)
    if input_type:
        return imgs_2D[0]
    return imgs_2D


#
# def transfo_imgs(image, args, mode): # Types of transformations and range inspired by Sujit 2019
#    """
#    Takes as input an image and transforms it (currently rotation and translation available)
#    """  
#    img = image
#    if len(img.shape) == 3:         
#        if np.random.rand(1)[0] < args[0]:
#            angle = 10
#            img = at.random_rotation(img, angle, row_axis=0, col_axis=1, channel_axis=2)
#        if np.random.rand(1)[0] < args[1]:
#            axs_0 = 21
#            axs_1 = 6
#            img = at.random_shift(img, axs_0, axs_1, row_axis=0, col_axis=1, channel_axis=2) 
#    elif len(img.shape) == 4:
#        img2 = np.zeros( (*img.shape[:-1], 0))
#        if np.random.rand(1)[0] < args[0]:
#            angle = 10
#            axes = tuple(np.random.choice(range(3), 2))
#            for k in range(img.shape[3]):
#                img2 = np.concatenate((img2, ndi.rotate(img[k], angle, axes=axes, reshape=False)), axis = 3)
#        img = img2
#        if np.random.rand(1)[0] < args[1]:
#            axs_0 = np.random.randint(0, 21)
#            axs_1 = np.random.randint(-5, 6)
#            axs_2 = np.random.randint(-5, 5)
#            img = shift(img, [axs_0, axs_1, axs_2])
#    return img

def normalization_func(img):
    """
    Return an image brought back between 0 and 1
    """
    vmin, vmax = img.min(), img.max()
    if vmin != vmax:
        im = (img - vmin) / (vmax - vmin)
    else:
        im = np.ones(img.shape)
    return im


def normalization_mask(img, mask):
    """
    Return an image with the foreground normalized between 0 and 1; 
    and with the background normalized between 0 and 1
    """
    zone1 = img[mask != 0]
    zone2 = img[mask == 0]
    zone1 = (zone1 - zone1.min()) / (zone1.max() - zone1.min())
    zone2 = (zone2 - zone2.min()) / (zone2.max() - zone2.min())
    imge = img.copy()
    imge[mask != 0] = zone1
    imge[mask == 0] = zone2
    return imge


def normalization_brain(img, mask):
    """
    Return an image with the foreground normalized between 0 and 1; 
    and with the background puts to 0
    """
    zone1 = img[mask != 0]
    imge = img.copy()
    imge[mask != 0] = (zone1 - zone1.min()) / (zone1.max() - zone1.min())
    imge[mask == 0] = 0
    return imge


# def normalization_fsl(img, ID, prefix, metadata, nbb, idw):
#    file_path = prefix + metadata.iloc[ID].img_file
#    temp1 = [pos for pos, char in enumerate(metadata.iloc[ID].img_file) if char == "/"][-1]
#    temp2 = [pos for pos, char in enumerate(metadata.iloc[ID].img_file) if metadata.iloc[ID].img_file[pos:pos+4]==".nii"][-1]   
#    name = (metadata.iloc[ID].img_file)[temp1+1:temp2]+"_id_"+str(nbb)+"_idw_"+str(idw)
#    p = subprocess.Popen(['bet',file_path, prefix+name+".nii.gz"])
#    p.wait()
#    mask = nb.load(prefix+name+".nii.gz").get_fdata()
#    os.remove(prefix+name+".nii.gz")
#    imge = normalization_mask(img, mask)
#    return imge

def reslice_to_ref(fin, fref, faff):
    """
    :param fin: input image to reslice (either full path or NiftiImage
    :param fref: full path to image defining space to reslice in
    :param faff: optional fullpath to a 4*4 affine matrix to apply to fin before the reslice (check with niftireg affine.txt)

    :return: numpy array of the resliced images
    """
    import nibabel.processing as nbp

    if isinstance(fin, str):
        fin = nb.load(fin)

    if faff:
        acoreg = np.loadtxt(faff, delimiter=' ')
        acoreg = np.linalg.inv(acoreg)

        imgaff = acoreg.dot(fin.affine)
        fin.affine[:] = imgaff[:]

    out_img = nbp.resample_from_to(fin, fref, cval=-1)

    fout = out_img.get_fdata()
    return fout


def crop_around_mask(fin, fmask, out_shape):
    """
    :param fin: input image (string Nifiti1Image or numpy array)
    :param fmask: mask to definde the fov of interest to be center on (string Nifiti1Image or numpy array)
    :param out_shape: tuple defining the wanted shape of the output
    :return: numpy array of the croped image
    """

    if isinstance(fmask, str):
        fmask = nb.load(fmask)

    if isinstance(fmask, nb.nifti1.Nifti1Image):
        fmask = fmask.get_fdata()

    if isinstance(fin, str):
        fin = nb.load(fin)

    if isinstance(fin, nb.nifti1.Nifti1Image):
        fin = fin.get_fdata()

    # if ras and nb.aff2axcodes(im.affine) != ('R', 'A', 'S'):
    #     print('changing image affine to canonical ... ')
    #     im = nb.as_closest_canonical(im)

    out_shape = np.array(out_shape)  # shape i    s often a tuple
    in_shape = np.array(fin.shape)
    diff_shape = in_shape - out_shape

    ii = np.argwhere(fmask > 0)
    min_pos = np.min(ii, axis=0)
    max_pos = np.max(ii, axis=0)
    fov = max_pos - min_pos
    center = min_pos + np.round(fov / 2)

    # Find fin and fout index so that there is a padding with zero
    xout1, xout2, xin1, xin2 = np.ndarray((3,), int), np.ndarray((3,), int), np.ndarray((3,), int), np.ndarray((3,), int)
    for kk in range(3):

        if diff_shape[kk] < 0:  # fin is shorter than fout
            xout1[kk], xout2[kk] = int(np.ceil(-diff_shape[kk] / 2)), int(np.ceil(out_shape[kk] + diff_shape[kk] / 2))
            xin1[kk], xin2[kk] = 0, in_shape[kk]
        else:
            xout1[kk], xout2[kk] = 0, out_shape[kk]
            xin1[kk], xin2[kk] = center[kk] - int(np.floor(out_shape[kk] / 2)), center[kk] + int(np.ceil(out_shape[kk] / 2))

        # adjust fin range to be within in_shape
        shift_ind = 0
        if xin1[kk] < 0:
            shift_ind = -xin1[kk]
        if xin2[kk] > in_shape[kk]:
            shift_ind = in_shape[kk] - xin2[kk]
        xin2[kk] = xin2[kk] + shift_ind
        xin1[kk] = xin1[kk] + shift_ind

    fout = np.zeros(out_shape)
    fout[xout1[0]:xout2[0], xout1[1]:xout2[1], xout1[2]:xout2[2]] = fin[xin1[0]:xin2[0], xin1[1]:xin2[1], xin1[2]:xin2[2]]
    return fout


def print_accuracy_df_split(res, ytrue, prediction_name=None, note_thr=2,
                            test_size=0.3, kfold=100):
    import math
    from sklearn.model_selection import train_test_split

    df = []

    yind = np.arange(len(res))
    for k in np.arange(kfold):
        y1, y2 = train_test_split(yind, shuffle=True, test_size=test_size)
        rr = res.iloc[y1, :]
        yytrue = ytrue[y1]
        df.append(get_accuracy_df(rr, yytrue))

    dmean = pd.DataFrame
    for ii, dd in enumerate(df):
        if ii == 0:
            dmean = dd
        else:
            dmean = dmean + dd
    dmean = dmean / (ii + 1)

    dstd = pd.DataFrame
    for ii, dd in enumerate(df):
        if ii == 0:
            dstd = (dd - dmean) * (dd - dmean)
        else:
            dstd = dstd + ((dd - dmean) * (dd - dmean))

    dstd = dstd / (ii + 1)
    dstd = dstd.apply(np.sqrt)

    dmean = dmean.sort_v
    dmean = dmean.sort_values('rocAUC', ascending=False)
    dstd = dstd.reindex(dmean.index)


def get_accuracy_df(res, ytrue, prediction_name=None, note_thr=2):
    y_true = ytrue.copy()

    y_true[ytrue < note_thr] = 1;  # BAD image are label 1
    y_true[ytrue >= note_thr] = 0;

    nbzeros, nbones = np.sum(y_true == 0), np.sum(y_true == 1)
    print('Choising threshold %d \t %d \t 0 and \t %d \t 1 tot \t %d' % (note_thr, nbzeros, nbones, nbzeros + nbones))

    if prediction_name is None:
        prediction_name = res.columns

    sensitivitys, specificitys, roc_aucs, best_thrs, inverse_predictions, baucs, tns, fns, fps, tps = [], [], [], [], [], [], [], [], [], []
    df = pd.DataFrame([])

    for ii, rr in enumerate(prediction_name):

        y_pred_prob = res[rr].values.copy()
        if type(y_pred_prob[0]) is str:
            continue

        num_nan = np.sum(np.isnan(y_pred_prob))
        if np.any(np.isnan(y_pred_prob)):
            print('Skiping {} because of NaN'.format(rr))
            continue

        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred_prob)
        roc_auc = metrics.auc(fpr, tpr)

        # test roc_auc with invers_pred
        y_pred_prob_inv = 1 / y_pred_prob
        if np.any(np.isinf(y_pred_prob_inv)):
            roc_auc_inv = 0
        else:
            roc_auc_inv = metrics.roc_auc_score(y_true, y_pred_prob_inv)

        if roc_auc_inv > roc_auc:
            inverse_prediction = 1
            fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred_prob_inv)
            y_pred_prob = y_pred_prob_inv
            roc_auc = roc_auc_inv
        else:
            inverse_prediction = 0

        imin2 = np.argmax(tpr - fpr)
        best_thr2 = threshold[imin2]  # threshold[np.argmax(tpr/fpr)]

        y_pred = np.round(y_pred_prob, 0)

        y_pred[y_pred_prob > best_thr2] = 1
        y_pred[y_pred_prob <= best_thr2] = 0

        tn2, fp2, fn2, tp2 = metrics.confusion_matrix(y_true, y_pred).ravel()
        sensitivity2, specificity2 = (tp2) / (tp2 + fn2), (tn2) / (tn2 + fp2)
        bauc2 = metrics.balanced_accuracy_score(y_true, y_pred, adjusted=False)

        if inverse_prediction: best_thr2 = 1 / best_thr2  # to have the correct value

        datarow = pd.DataFrame.from_dict({'sens': [sensitivity2], 'spec': [specificity2],
                                          'rocAUC': roc_auc, 'thr': best_thr2, 'inv': inverse_prediction,
                                          'bauc': bauc2, 'tn': tn2, 'fn': fn2, 'fp': fp2, 'tp': tp2, }, )
        datarow.index = [rr]

        df = df.append(datarow)
    return df


def print_accuracy_df(res, ytrue, prediction_name=None, note_thr=2):
    print("Sens \t Spec \t AUC \t Thr \t Inv \t bauc \t tn \t fn \t fp \t tp ")

    df = get_accuracy_df(res, ytrue, prediction_name, note_thr)

    df = df.sort_values('rocAUC', ascending=False)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1,
                           'display.width', 400, 'display.float_format', '{:,.2f}'.format):  # more options can be specified also
        print(df)


#    print('%.2f \t%.2f \t%.2f \t%.2f  \t %d \t\t %.2f \t %-5d \t %-5d \t %-5d \t %d  r %d %s'
#          % (sensitivity2, specificity2, roc_auc, best_thr2, inverse_prediction, bauc2, tn2, fn2, fp2, tp2, ii, rr))

#    print()

def print_accuracy(res, resname, ytrue, prediction_name='ymean', inverse_prediction=False, do_plot=False, note_thr=2):
    import sklearn.metrics as metrics
    if do_plot:
        import matplotlib.pyplot as plt

    y_true = ytrue.copy()

    y_true[ytrue < note_thr] = 1  # BAD image are label 1
    y_true[ytrue >= note_thr] = 0

    nbzeros, nbones = np.sum(y_true == 0), np.sum(y_true == 1)
    print('Choising threshold %d \t %d \t 0 and \t %d \t 1 tot \t %d' % (note_thr, nbzeros, nbones, nbzeros + nbones))

    print("Sens \t Spec \t AUC \t Thr \t bauc \t tn \t fn \t fp \t tp ")

    for ii, rr in enumerate(res):

        y_pred_prob = res[ii][prediction_name].values
        if inverse_prediction: y_pred_prob = 1 / y_pred_prob

        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred_prob)
        roc_auc = metrics.auc(fpr, tpr)
        imin = np.argmin(np.abs(tpr - 1 + fpr))
        imin2 = np.argmax(tpr - fpr)
        best_thr = threshold[imin]  # threshold[np.argmax(tpr/fpr)]
        best_thr2 = threshold[imin2]  # threshold[np.argmax(tpr/fpr)]

        best_thr = 0.5
        y_pred = np.round(y_pred_prob, 0)
        y_pred[y_pred_prob > best_thr] = 1
        y_pred[y_pred_prob <= best_thr] = 0

        # y_pred = np.round(y_pred_prob,0)

        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        sensitivity, specificity = (tp) / (tp + fn), (tn) / (tn + fp)

        bauc = metrics.balanced_accuracy_score(y_true, y_pred, adjusted=False)
        #    tres = tres.append({'Sens' : sensitivity , 'Spec':specificity , 'AUC':roc_auc,
        #                 'tn':tn, 'fn':fn, 'fp':fp,'tp':tp,'resname':resname[ii]}, ignore_index=True)

        # precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred_prob)
        # auc2 = metrics.auc(recall,precision) #always 1 because too much 1
        # auc2 = metrics.recall_score(y_true, y_pred)

        # tpr, fpr, threshold = tpr[fpr>0], fpr[fpr>0], threshold[fpr>0]
        y_pred[y_pred_prob > best_thr2] = 1
        y_pred[y_pred_prob <= best_thr2] = 0

        tn2, fp2, fn2, tp2 = metrics.confusion_matrix(y_true, y_pred).ravel()
        sensitivity2, specificity2 = (tp2) / (tp2 + fn2), (tn2) / (tn2 + fp2)
        bauc2 = metrics.balanced_accuracy_score(y_true, y_pred, adjusted=False)

        if inverse_prediction:   #get back non inverted valu for printing
            best_thr2 = 1 / best_thr2
            y_pred_prob = 1 / y_pred_prob


        print('%.2f \t%.2f \t%.2f \t%.2f \t%.2f \t %-5d \t %-5d \t %-5d \t %d  r %d \t %.2f \t %.2f %s'
              % (sensitivity2, specificity2, roc_auc, best_thr2, bauc2, tn2, fn2, fp2, tp2, ii, y_pred_prob.min(), y_pred_prob.max(), resname[ii]))
        print('%.2f \t%.2f \t%.2f \t%.2f \t%.2f \t %-5d \t %-5d \t %-5d \t %d  r %d %s'
              % (sensitivity, specificity, roc_auc, best_thr, bauc, tn, fn, fp, tp, ii, resname[ii]))
        print()

        if do_plot:
            plt.figure()
            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')


def print_accuracy_all(res, resname, ytrue, prediction_name='ymean', inverse_prediction=False, do_plot=False):
    #    if 'sklearn.metrics' not in sys.modules:
    import sklearn.metrics as metrics
    if do_plot:
        #        if 'matplotlib.pyplot' not in sys.modules:
        import matplotlib.pyplot as plt

    y_true = ytrue.copy()

    notes = np.unique(np.sort(ytrue))

    for ind, note_thr in enumerate(notes):
        if ind == 0:
            continue
        y_true[ytrue < note_thr] = 1;  # BAD image are label 1
        y_true[ytrue >= note_thr] = 0;

        nbzeros, nbones = np.sum(y_true == 0), np.sum(y_true == 1)
        print('Choising threshold %d \t %d \t 0 and \t %d \t 1 tot \t %d' % (note_thr, nbzeros, nbones, nbzeros + nbones))

        print("Sens \t Spec \t AUC \t Thr \t bauc \t tn \t fn \t fp \t tp ")

        for ii, rr in enumerate(res):

            y_pred_prob = res[ii][prediction_name].values
            if inverse_prediction: y_pred_prob = 1 - y_pred_prob

            fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred_prob)
            roc_auc = metrics.auc(fpr, tpr)
            imin = np.argmin(np.abs(tpr - 1 + fpr))
            imin2 = np.argmax(tpr - fpr)
            best_thr = threshold[imin]  # threshold[np.argmax(tpr/fpr)]
            best_thr2 = threshold[imin2]  # threshold[np.argmax(tpr/fpr)]

            y_pred = np.round(y_pred_prob, 0)
            y_pred[y_pred_prob > best_thr] = 1
            y_pred[y_pred_prob <= best_thr] = 0

            # y_pred = np.round(y_pred_prob,0)

            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
            sensitivity, specificity = (tp) / (tp + fn), (tn) / (tn + fp)

            bauc = metrics.balanced_accuracy_score(y_true, y_pred, adjusted=False)
            #    tres = tres.append({'Sens' : sensitivity , 'Spec':specificity , 'AUC':roc_auc,
            #                 'tn':tn, 'fn':fn, 'fp':fp,'tp':tp,'resname':resname[ii]}, ignore_index=True)

            # precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred_prob)
            # auc2 = metrics.auc(recall,precision) #always 1 because too much 1
            # auc2 = metrics.recall_score(y_true, y_pred)

            # tpr, fpr, threshold = tpr[fpr>0], fpr[fpr>0], threshold[fpr>0]
            y_pred[y_pred_prob > best_thr2] = 1
            y_pred[y_pred_prob <= best_thr2] = 0

            tn2, fp2, fn2, tp2 = metrics.confusion_matrix(y_true, y_pred).ravel()
            sensitivity2, specificity2 = (tp2) / (tp2 + fn2), (tn2) / (tn2 + fp2)
            bauc2 = metrics.balanced_accuracy_score(y_true, y_pred, adjusted=False)

            print('%.2f \t%.2f \t%.2f \t%.2f \t%.2f \t %-5d \t %-5d \t %-5d \t %d  r %d %s'
                  % (sensitivity2, specificity2, roc_auc, best_thr2, bauc2, tn2, fn2, fp2, tp2, ii, resname[ii]))
            # print('%.2f \t%.2f \t%.2f \t%.2f \t%.2f \t %-5d \t %-5d \t %-5d \t %d  r %d %s'
            #       % (sensitivity, specificity, roc_auc, best_thr, bauc, tn, fn, fp, tp, ii, resname[ii]))
            print()

            if do_plot:
                plt.figure()
                plt.title('Receiver Operating Characteristic')
                plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
                plt.legend(loc='lower right')
                plt.plot([0, 1], [0, 1], 'r--')
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')

def remove_extension(str_in):
    return_str = False
    second_extention_list = ['.nii', '.pt']
    if isinstance(str_in, str):
        str_in = [str_in];
        return_str = True
    #remove up to 2 extension
    #res = [os.path.splitext(os.path.splitext(ss)[0])[0] for ss in str_in]
    res = []
    for ss in str_in:
        rr1, ext = os.path.splitext(ss)
        rr2, ext = os.path.splitext(rr1)
        if ext in second_extention_list:
            res.append(rr2)
        else:
            res.append(rr1)
    #res = [os.path.splitext(ss)[0] for ss in str_in]  #one shot

    return res[0] if return_str else res


from functools import reduce


def getcommonletters(strlist):
    return ''.join([x[0] for x in zip(*strlist) \
                    if reduce(lambda a, b: (a == b) and a or None, x)])


def findcommonstart(strlist):
    strlist = strlist[:]
    prev = None
    while True:
        common = getcommonletters(strlist)
        if common == prev:
            break
        strlist.append(common)
        prev = common

    return getcommonletters(strlist)


def reduce_name_list(strlist):
    strcommon = findcommonstart(strlist.copy())
    res = [i for i in range(len(strcommon)) if strcommon.startswith('_', i)]

    keep = [ss[res[-1] + 1:] for ss in strlist]

    return strcommon, keep

def remove_string_from_name_list(strlist, string_to_remove_list):

    for string_to_remove in string_to_remove_list:
        slen = len(string_to_remove)
        sout = []
        for s in strlist:
            ind_start = s.find(string_to_remove)
            if ind_start >= 0 :
                s = s[:ind_start] + s[ind_start+slen:]
            sout.append(s)
        strlist = sout.copy()
        #print('removin {}'.format(string_to_remove))
        #print(strlist)

    return sout
