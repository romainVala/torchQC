
import torch,numpy as np,  torchio as tio
import torch.nn.functional as F
from torch.utils.data import DataLoader

from timeit import default_timer as timer
import json, os

import pandas as pd
from segmentation.config import Config
from segmentation.run_model import ArrayTensorJSONEncoder
from segmentation.collate_functions import history_collate
from segmentation.utils import to_numpy

from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance #warning cpu metrics
from monai.metrics import compute_confusion_matrix_metric, get_confusion_matrix, DiceHelper, compute_generalized_dice

from skimage.measure import euler_number, label

from segmentation.losses.dice_loss import Dice
from segmentation.metrics.utils import MetricOverlay
from utils_file import get_parent_path, gfile, gdir
import subprocess

dice_instance = Dice()
metric_dice = dice_instance.all_dice_loss
#metric_dice = dice_instance.mean_dice_loss #warning argument must have 5 dim
#metric_dice = dice_instance.dice_loss  # here 4 or 5 dim (every thing is flatten)
met_overlay = MetricOverlay(metric_dice, band_width=5) #, channels=[2])


confu_met=["false discovery rate", "miss rate", "balanced accuracy", "f1 score"]
confu_met_names=['fdr', 'miss', 'bAcc', 'f1' ]


import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt

#from pykeops.torch import LazyTensor
#from scipy.ndimage.morphology import distance_transform_edt


def erode3D(mask):
    return -F.max_pool3d(-mask, (3, 3, 3), (1, 1, 1), (1, 1, 1))

def dilate3D(mask):
    return F.max_pool3d(mask, (3, 3, 3), (1, 1, 1), (1, 1, 1))

#def erode2D(mask):
#    return -F.max_pool2d(-mask, (3, 3), (1, 1), (1, 1))

#def dilate2D(mask):
#    return F.max_pool2d(mask, (3, 3), (1, 1), (1, 1))

def dilate(mask):
    if mask.dim() == 5:
        return dilate3D(mask)
    elif mask.dim() == 4:
        return dilate2D(mask)
    else:
        raise ValueError("Tensor must be 2D or 3D")

def erode(mask):
    if mask.dim() == 5:
        return erode3D(mask)
    elif mask.dim() == 4:
        return erode2D(mask)
    else:
        raise ValueError("Tensor must be 2D or 3D")

def get_edges(mask):
    if mask.dim() == 5:
        ero = erode3D(mask)
    elif mask.dim() == 4:
        ero = erode2D(mask)
    else:
        raise ValueError("Tensor must be 2D or 3D")
    contour = torch.logical_xor(ero > 0.5, mask > 0.5)
    return contour


def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    if len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)
    else:
        raise ValueError("Can only process 3D images")

def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def hard_open(img):
    return dilate(erode(img))

def soft_skel(img, max_iter=100):
    skel = F.relu(img - soft_open(img))
    iteration = 0
    with torch.no_grad():
        while True and iteration < max_iter:
            iteration += 1
            img = soft_erode(img)
            # img = erode(img)
            delta = F.relu(img - soft_open(img))
            # delta = F.relu(img - hard_open(img))
            to_add = F.relu(delta - skel * delta)
            if not to_add.sum():
                break
            skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(y_trueAll, y_predAll, iter_=100, smooth=10e-7):
    B = y_trueAll.shape[0]
    C = y_trueAll.shape[1]
    cldice_all = torch.zeros((B,C)) #[]
    for b in range(B):
        for c in range(C):
            y_pred = y_predAll[b, c, ...]
            y_true = y_trueAll[b, c, ...]
            y_pred = y_pred.unsqueeze(0).unsqueeze(0)
            y_true = y_true.unsqueeze(0).unsqueeze(0)

            skel_pred = soft_skel(y_pred, iter_)
            skel_true = soft_skel(y_true, iter_)
            tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + smooth) / (torch.sum(skel_pred) + smooth)
            tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + smooth) / (torch.sum(skel_true) + smooth)
            #cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
            cl_dice =  2.0 * (tprec * tsens) / (tprec + tsens)
            cldice_all[b,c] = cl_dice.detach().cpu() #.numpy()

    return cldice_all


def get_soft_edges(mask):
    if mask.dim() == 5:
        ero = soft_erode(mask)
    else:
        raise ValueError("Tensor must be 2D or 3D")
    contour = torch.logical_xor(ero > 0.5, mask > 0.5)
    return contour

def Average_Hausdorff_Distance(gth, pred, method="scipy", soft=True):
    B = gth.shape[0]
    C = gth.shape[1]
    gth = (gth > 0.5).float()
    pred = (pred > 0.5).float()
    if soft:
        gth_edges = get_soft_edges(gth)
        pred_edges = get_soft_edges(pred)
    else:
        gth_edges = get_edges(gth)
        pred_edges = get_edges(pred)
    ahds_losses, ahds_losses_dir = torch.zeros((B,C)), torch.zeros((B,C)) #[], []
    hds_losses, hds_losses_dir = torch.zeros((B,C)),torch.zeros((B,C)) #[], []
    for b in range(B):
        for c in range(C):
            gth_edges_bc = gth_edges[b, c, ...]
            pred_edges_bc = pred_edges[b, c, ...]
            if (not gth_edges_bc.sum()) and (not pred_edges_bc.sum()):
                gth2pred = torch.tensor(float('nan'))
                pred2gth = torch.tensor(float('nan'))
                ahd = (gth2pred + pred2gth) / 2
            elif method == "scipy":
                gth_edges_bc = gth_edges_bc.detach().cpu().numpy()
                pred_edges_bc = pred_edges_bc.detach().cpu().numpy()
                gth_edges_bc_dm = distance_transform_edt(1 - gth_edges_bc)
                pred_edges_bc_dm = distance_transform_edt(1 - pred_edges_bc)
                gth2pred = pred_edges_bc_dm[gth_edges_bc]
                pred2gth = gth_edges_bc_dm[pred_edges_bc]
                ahd1, ahd2 = gth2pred.mean(), pred2gth.mean()
                hd1, hd2   = gth2pred.max(), pred2gth.max()

                #ahd = (gth2pred + pred2gth) / 2
            elif method == "keops":
                gth_edges_bc_coordinates = torch.nonzero(gth_edges_bc)
                pred_edges_bc_coordinates = torch.nonzero(pred_edges_bc)
                X = LazyTensor(gth_edges_bc_coordinates.view(gth_edges_bc_coordinates.shape[0], 1, gth_edges_bc_coordinates.shape[1]).float())
                Y = LazyTensor(pred_edges_bc_coordinates.view(1, pred_edges_bc_coordinates.shape[0], pred_edges_bc_coordinates.shape[1]).float())
                Distance_matrix = ((X - Y)**2).sum(dim=2)**0.5
                gth2pred = Distance_matrix.min_reduction(1).mean()
                pred2gth = Distance_matrix.min_reduction(0).mean()
                ahd = (gth2pred + pred2gth) / 2
                ahd = ahd.cpu().numpy()
            #hds_losses.append(ahd)
            #ahds_losses.append(ahd1); ahds_losses_dir.append(ahd2); hds_losses.append(hd1); hds_losses_dir.append(hd2)
            ahds_losses[b,c] = ahd1; ahds_losses_dir[b,c] = ahd2; hds_losses[b,c] = hd1; hds_losses_dir[b,c] = hd2

    #return np.array(hds_losses).mean(), gth2pred, pred2gth
    return ahds_losses, ahds_losses_dir, hds_losses, hds_losses_dir

def mrview_from_df(df, col_name, condition,  bin_overlay_class=0):
    dfsub = df[df[col_name]==condition]
    for dfser in dfsub.iterrows():
        dfser = dfser[1]
        #print(f'{dfser.metric_dice_loss_GM:.2} GM dicm from {dfser.model} Predction {dfser.fpred} ')
        if "metric_dice_loss_GM" in dfser:
            dicegm = "metric_dice_loss_GM"
        elif "dice_GM" in dfser:
            dicegm = "dice_GM"
        elif "metric_dice_GM" in dfser:
            dicegm = "metric_dice_GM"
        print(f'#{dfser[dicegm]:.2} GM dicm from {dfser.model} ')

    return mrview_overlay(list(dfsub.finput.values), [dfsub.flabel.values[0]] + list(dfsub.fpred.values),
                          bin_overlay_class=bin_overlay_class)
def mrview_overlay(bg_img, overlay_list, bin_overlay_class=0):
    if not isinstance(bg_img, list):
        bg_img = [bg_img]
    if not isinstance(overlay_list, list):
        overlay_list = [overlay_list]
    col_overlay = [ '0,1,0', '1,0,0', '0,0,1', '1,1,0', '0,1,1', '1,0,1', '1,0.5,0', '0.5,1,0', '1,0,0.5', '0.5,0,1']
    if bin_overlay_class:
        mrviewopt = [
            f'-overlay.opacity 0.4 -overlay.colour {col_overlay[k]} -overlay.intensity 0,{bin_overlay_class}   ' \
            f'-overlay.threshold_min {bin_overlay_class-0.5} -overlay.threshold_max {bin_overlay_class+0.5} ' \
            f'-overlay.interpolation 0 -mode 2  -size 1300,900 ' for k in range(len(overlay_list))]
    else:
        mrviewopt = [
            f'-overlay.opacity 0.4 -overlay.colour {col_overlay[k]} -overlay.intensity 0,1   -overlay.threshold_min 0.5  -overlay.interpolation 0 -mode 2'
            for k in range(len(overlay_list)) ]

    cmd = 'vglrun mrview '
    cmd = 'mrviewv '
    for img in bg_img:
        cmd += (f' {img} ')
    for nb_over, img in enumerate(overlay_list):
        cmd += (f' -overlay.load {img} {mrviewopt[nb_over]} ')
    print(f'{cmd} ')
    return cmd
def display_res(dir_pred, bg_files, gt_files=None):

    cmd = []
    for nb_pred, one_dir_pred in enumerate(dir_pred):
        # one_dir_pred = dir_pred[0]
        all_file = gfile(one_dir_pred, 'nii')
        print(f'working on {one_dir_pred}')
        for ii, one_pred in enumerate(all_file):
            if nb_pred == 0:
                if gt_files is not None:
                    cmd.append([gt_files[ii]])
                    cmd[ii].append(one_pred)
                else:
                    cmd.append( [one_pred])
            else:
                cmd[ii].append(one_pred)
    mrview_cmd=[ mrview_overlay(bg_files[kk], cmd[kk]) for kk in range(len(cmd))]
    return mrview_cmd

def display_res2(resdir, doit=False):
    models = gdir(resdir,'.*')
    sujname = get_parent_path(gdir(models[0],'.*'))[1]
    for sujn in sujname:
        dir_pred = gdir(resdir, ['.*', sujn])
        fdata = gfile(dir_pred[0],'data')
        flabel= gfile(dir_pred[0],'label')
        fpred = gfile(dir_pred,'pred')
        cc = mrview_overlay(fdata, flabel + fpred , bin_overlay_class=2)
        if doit:
            subprocess.run( cc.split(' ') )

def binarize_5D(data, add_extra_to_class=None):
    return  met_overlay.binarize(data, add_extra_to_class=add_extra_to_class)

def computes_all_metric(prediction, target, labels_name, indata=None, selected_label=None,
                        selected_lab_mask=None, lab_mask_name=None, verbose=True, distance_metric=False,
                        euler_metric=False, volume_metric=False, confu_metric=False):

    prediction_bin = met_overlay.binarize(prediction)
    #print(f'volume is {volume_metric}')
    mask = None

    if selected_label is not None:
        if prediction.shape[1] > 1:
            prediction = prediction[:, selected_label, ...]
            prediction_bin = prediction_bin[:, selected_label, ...]
        else: #WARNING buggy select on 3d only for pred
            print('warning TODO')
            todo
            prediction_bin = prediction
            prediction_bin[prediction_bin!=selected_label[0]]=0
            prediction_bin[prediction_bin==selected_label[0]]=1


        if target.shape[1] > 1: #more than one chanel
            if selected_lab_mask is not None:
                mask = [target[:, ssi, ...] for ssi in selected_lab_mask]
                if len(mask) != len(lab_mask_name):
                    raise('wrong size for the masks ')
            target = target[:, selected_label, ...]

    start = timer()

    #dd = metric_dice(prediction_bin, target)
    # other option from monai is :
    # metric = DiceMetric(include_background=True, reduction="none", get_not_nans=False)
    # res = metric(y_pred=prediction_bin, y=target)

    dd_dice, not_nan = DiceHelper(include_background=True, softmax=False)(prediction_bin,target)
    col_name = [f'dice_{ss}' for ss in labels_name]
    df_one = pd.DataFrame([dd_dice.numpy()], columns=col_name)

    if volume_metric:
        for kk, llname in enumerate(labels_name):
            target_vol = target[:, kk, ...].sum().numpy()
            df_one[f'vol_targ_{llname}'] = target_vol
            df_one[f'vol_pred_ration{llname}'] = prediction[:, kk, ...].sum().numpy() / target_vol

    #arg todo metric_dice without batch reduction
    #dd = metric_dice(prediction, target)
    #res_dict.update( {f'softdice_{k}':float(v) for k,v in zip(labels_name, dd)} )

    if mask is not None:
        dd = dict()
        for jj, mask_name in enumerate(lab_mask_name):
            for ii, lname in enumerate(labels_name):
                nbvox = (prediction_bin[:,ii, ...] * mask[jj]).sum()
                dd[f'nb_{lname}_in_{mask_name}'] = nbvox.numpy()
        res_dict.update(dd)

    #compute euleur number
    if euler_metric:
        dd = dict()
        for ii, lname in enumerate(labels_name):
            pred_one = prediction_bin[0,ii,...].numpy()
            pred_label, num_label = label(pred_one, return_num=True, connectivity=3)
            #find the biggest component
            ind_biggest, nb_biggest = 0,0
            for kk in range(1,num_label+1):
                nbvox = np.sum(pred_label==kk)
                if nbvox> nb_biggest:
                    nb_biggest=nbvox
                    ind_biggest = kk

            pred_biggest = pred_label==ind_biggest
            dd[f'nb_isolated_{lname}'] = np.sum(pred_one) - np.sum(pred_biggest)
            dd[f'nb_isolated_ratio{lname}'] = (np.sum(pred_one) - np.sum(pred_biggest))/np.sum(pred_one)*100
            dd[f'eul_{lname}'] = euler_number(pred_biggest, connectivity=3)

        res_dict.update(dd)

    if confu_metric:
        resConfu = get_confusion_matrix(prediction_bin, target)
        for ii,mmm in enumerate(confu_met):
            dd_confu = compute_confusion_matrix_metric(mmm,resConfu)
            for kk, llname in enumerate(labels_name):
                df_one[f'{confu_met_names[ii]}_{llname}'] = dd_confu[:,kk].numpy()

            #res_dict.update({f'{confu_met_names[ii]}_{k}': float(v) for k, v in zip(labels_name, batch_confu)})

    if distance_metric:
        # prediction = prediction.cpu()
        # buggy discrete values 1 ???
        try:

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            prediction_bin = prediction_bin.to(device)
            target = target.to(device)

            dd = Average_Hausdorff_Distance(prediction_bin, target)
            for kk, llname in enumerate(labels_name):
                df_one[f'Sdis_{llname}'] = dd[0][:, kk].numpy()

            for kk, llname in enumerate(labels_name):
                df_one[f'SdisI_{llname}'] = dd[1][:, kk].numpy()

            for kk, llname in enumerate(labels_name):
                df_one[f'SdisMax_{llname}'] = dd[2][:, kk].numpy()

            for kk, llname in enumerate(labels_name):
                df_one[f'SdisIMax_{llname}'] = dd[3][:, kk].numpy()

            #res_dict.update({f'Sdis_{k}': float(v) for k, v in zip(labels_name, dd[0])})
            #res_dict.update({f'SdisI_{k}': float(v) for k, v in zip(labels_name, dd[1])})
            #res_dict.update({f'SdisMax_{k}': float(v) for k, v in zip(labels_name, dd[2])})
            #res_dict.update({f'SdisIMax_{k}': float(v) for k, v in zip(labels_name, dd[3])})

            cl_dice = soft_cldice(prediction_bin, target)
            for kk, llname in enumerate(labels_name):
                df_one[f'clDice_{llname}'] = cl_dice[:, kk].numpy()

            #res_dict.update({f'clDice_{k}': float(v) for k, v in zip(labels_name, cl_dice)})

            # alternative with monai (but cpu)
            # dd = compute_average_surface_distance(prediction_bin, target, include_background=True)
            # res_dict.update( {f'Sdis_{k}':float(v) for k,v in zip(labels_name, dd[0])} )
            # dd = compute_hausdorff_distance(prediction_bin, target, percentile=100, include_background=True)
            # res_dict.update( {f'haus_{k}':float(v) for k,v in zip(labels_name, dd[0])} )
        except:
            print('distand failed')
        if verbose:
            print(f'Computed distance metric in {timer()-start}')
            start = timer()

    if indata is not None:
        df_sig = pd.DataFrame()
        for nb_batch, (pred, targ, inda) in enumerate( zip(prediction_bin, target, indata)):
            #Signal stat for prediction
            res_dict = dict()
            for ind_lab, labn in enumerate(labels_name):
                sig = inda[pred[[ind_lab],...]>0].numpy()   #trick to avoid unsqueeze(0)
                if len(sig)>0 : #too small patch ... no data
                    data_qt = np.quantile(sig,[0.1, 0.25, 0.5, 0.75, 0.9])
                    data_mean, data_std = np.mean(sig), np.std(sig)
                    res_dict.update({f'PredSmean_{labn}': data_mean, f'PredSstd_{labn}': data_std,f'PredSquant_{labn}':data_qt})
            #Signal stat for ground truth
            for ind_lab, labn in enumerate(labels_name):
                sig = inda[targ[[ind_lab],...]>0].numpy()   #trick to avoid unsqueeze(0)
                if len(sig)>0 : #too small patch ... no data should not happend here
                    data_qt = np.quantile(sig,[0.1, 0.25, 0.5, 0.75, 0.9])
                    data_mean, data_std = np.mean(sig), np.std(sig)
                    res_dict.update({f'LabSmean_{labn}': data_mean, f'LabSstd_{labn}': data_std,f'LabSquant_{labn}':data_qt})
            #Signal stat for False Positif
            FP_mask = (pred[[ind_lab],...]>0) * (targ[[ind_lab],...]==0)
            for ind_lab, labn in enumerate(labels_name):
                sig = inda[FP_mask].numpy()
                if len(sig)>0 : #too small patch ...
                    data_qt = np.quantile(sig,[0.1, 0.25, 0.5, 0.75, 0.9])
                    data_mean, data_std = np.mean(sig), np.std(sig)
                    res_dict.update({f'FPSmean_{labn}': data_mean, f'FPSstd_{labn}': data_std,f'FPSquant_{labn}':data_qt})
            #Signal stat for False Negative
            FN_mask = (pred[[ind_lab],...]==0) * (targ[[ind_lab],...]>0)
            for ind_lab, labn in enumerate(labels_name):
                sig = inda[FN_mask].numpy()
                if len(sig)>0 : #too small patch ...
                    data_qt = np.quantile(sig,[0.1, 0.25, 0.5, 0.75, 0.9])
                    data_mean, data_std = np.mean(sig), np.std(sig)
                    res_dict.update({f'FNSmean_{labn}': data_mean, f'FNSstd_{labn}': data_std,f'FNSquant_{labn}':data_qt})

            df_sig = pd.concat([df_sig, pd.DataFrame([res_dict])])

        df_sig.index = pd.RangeIndex(256)
        df_one = pd.concat([df_one, df_sig], axis=1) #, ignore_index=True)

    if verbose:
        print(f'Computed all metric in {timer()-start}')

    return df_one

def load_model(model_path, device):
    config = Config(None, None, save_files=False)

    model_struct = {'module': 'unet', 'name': 'UNet', 'last_one': False, 'path': model_path, 'device': device}
    model_struct = config.parse_model_file(model_struct)
    print(f'Loading on model {model_path}')
    model, device = config.load_model(model_struct)
    return model.eval()

def get_tio_data_loader(fsuj_csv, tio_transform, replicate=1, get_dataset=False,
                        t1_column_name="vol_name", label_column_name="label_name", sujname_column_name="sujname"):

    subject_list = []
    for fin_path, flab_path, suj_name in zip(fsuj_csv[t1_column_name], fsuj_csv[label_column_name], fsuj_csv[sujname_column_name]):
        fin = tio.ScalarImage(fin_path)
        flab = tio.LabelMap(flab_path)
        if replicate:
            for ii in range(replicate):
                thesujname = f'{suj_name}_d{ii+1}'
                subject_list.append( tio.Subject({'t1': fin, 'label': flab, 'name': thesujname}) )
        else:
            thesujname = suj_name
            subject_list.append(tio.Subject({'t1': fin, 'label': flab, 'name': thesujname}))

    tio_ds = tio.SubjectsDataset(subject_list, transform=tio_transform)
    if get_dataset:
        return tio_ds
    else:
        print('12 numworker')
        return DataLoader(tio_ds, 1, shuffle=False,num_workers=12, collate_fn=history_collate)

def predic_segmentation(suj, model,  df, res_dict, device,  labels_name,
                        selected_label=None, out_dir=None, save_data=True, resample_back=False):

    if out_dir is not None:
        model_name, sujname = res_dict['model_name'], res_dict['sujname']
        out_dir = out_dir + '/' + model_name
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_dir = out_dir + '/' + sujname
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_name = out_dir + '/metric_'  + res_dict['sujname'] + '.csv'
        if os.path.exists(out_name):
            print(f'skiping {out_name}  exists')
            return df


    target = suj['label']['data'].float().to(device)
    data = suj['t1']['data'].float().to(device)

    if target.ndim==4:  #comes from dataset, so missing batch dim
        target = target.unsqueeze(0)
        data = data.unsqueeze(0)

    with torch.no_grad():
        prediction = model(data)
        prediction = F.softmax(prediction, dim=1)

    res_dict.update( computes_all_metric(prediction, target, labels_name, selected_label=selected_label) )
    res_dict = record_history(res_dict, suj)
    df_one = pd.DataFrame([res_dict])
    df = pd.concat([df, df_one], ignore_index=True)

    #save output in nifti files
    if out_dir is not None:
        if isinstance(suj, tio.Subject):
            sujtio = suj
            sujtio.add_image(tio.LabelMap(tensor=to_numpy(prediction[0]), affine=to_numpy(suj["t1"]['affine']) ), 'pred')
        else:
            sujtio = tio.Subject(dict(t1=tio.ScalarImage(tensor=to_numpy(suj["t1"]["data"][0]), affine=to_numpy(suj["t1"]['affine'][0])),
                                      label=tio.LabelMap(tensor=to_numpy(suj["label"]["data"][0]), affine=to_numpy(suj["t1"]['affine'][0])),
                                      pred=tio.LabelMap(tensor=to_numpy(prediction[0]), affine=to_numpy(suj["t1"]['affine'][0]))))
        if resample_back:
            tresamp = tio.Resample(target= sujtio.t1.path, label_interpolation='bspline')
            sujtio = tresamp(sujtio)
            thot = tio.OneHot()
            label_orig = thot(tio.LabelMap(sujtio.label.path))
            if False:
                res_dict.update(computes_all_metric(sujtio.pred.data.unsqueeze(0), sujtio.label.data.unsqueeze(0).contiguous(),
                                                    labels_name, selected_label=selected_label))
                #res_dict = record_history(res_dict, suj)
                df_one2 = pd.DataFrame([res_dict])
                df_one = pd.concat([df_one, df_one2], ignore_index=True)
                df = pd.concat([df, df_one2], ignore_index=True)
            res_dict.update(computes_all_metric(sujtio.pred.data.unsqueeze(0), label_orig.data.unsqueeze(0),
                                                labels_name, selected_label=selected_label))
            #res_dict = record_history(res_dict, suj)
            df_one2 = pd.DataFrame([res_dict])
            df_one = pd.concat([df_one, df_one2], ignore_index=True)
            df = pd.concat([df, df_one2], ignore_index=True)

        if save_data:
            #out_name = out_dir + '/data_' + res_dict['sujname'] + '.nii.gz'
            out_name = out_dir +  '/data.nii.gz'
            if not os.path.exists(out_name):
                sujtio.t1.save(out_name)
            if save_data>1:
                #out_name = out_dir + '/label_' + res_dict['sujname'] + '.nii.gz'
                out_name = out_dir + '/label.nii.gz'
                if not os.path.exists(out_name):
                    sujtio.label.save(out_name)
                #out_name = out_dir + '/pred_' + res_dict['sujname'] + '_M_' + res_dict['model_name'] + '.nii.gz'
                out_name = out_dir + '/prediction.nii.gz'

                sujtio.pred.save(out_name)

            else:
                if target.shape[1]>1:
                    tiohot = tio.OneHot(invert_transform=True)
                else:
                    tiohot = tio.OneHot(invert_transform=True, include='pred')

                sujtio = tiohot(sujtio)

                out_name = out_dir + '/bin_label_' + res_dict['sujname'] + '.nii.gz'
                if not os.path.exists(out_name):
                    sujtio.label.save(out_name)
                out_name = out_dir + '/bin_pred_' + res_dict['sujname'] + '_M_' + res_dict['model_name'] + '.nii.gz'
                sujtio.pred.save(out_name)
        out_name = out_dir + '/metric_'  + res_dict['sujname'] + '.csv'
        df_one.to_csv(out_name)

    return df

def record_history(info, sample, idx=0): #copy past from run_model
    is_batch = not isinstance(sample, tio.Subject)
    order = []
    history = sample.get('history') if is_batch else sample.history
    transforms_metrics = sample.get("transforms_metrics") if is_batch else sample.transforms_metrics
    if history is None or len(history) == 0:
        return
    relevant_history = history[idx] if is_batch else history
    #info["history"] = relevant_history

    relevant_metrics = transforms_metrics[idx] if is_batch else transforms_metrics

    if len(relevant_metrics) == 1 and isinstance(relevant_metrics[0], list):
        relevant_metrics = relevant_metrics[0]
    info["transforms_metrics"] = relevant_metrics
    if len(relevant_history)==1 and isinstance(relevant_history[0], list):
        relevant_history = relevant_history[0] #because ListOf transfo to batch make list of list ...

    for hist in relevant_history:
        if isinstance(hist, dict) :
            histo_name = hist['name']
            for key, val in hist.items():
                if callable(val):
                    hist[key] = str(val)
            str_hist = str( hist )
        else:
            histo_name = hist.name #arg bad idea to mixt transfo and dict
            str_hist = dict()
            for name in  hist.args_names :
                val = getattr(hist, name)
                if callable(val):
                    val = str(val)
                str_hist[name] = val
#               str_hist = {name: str() if isinstance(getattr(hist, name),funtion) else getattr(hist, name) for name in hist.args_names}
        #instead of using str(hist) wich is not correct as a python eval, make a dict of input_param
        if f'T_{histo_name}' in info:
            histo_name = f'{histo_name}_2'
        info[f'T_{histo_name}'] = json.dumps(
            str_hist, cls=ArrayTensorJSONEncoder)
        order.append(histo_name)

    info['transfo_order'] = '_'.join(order)
    return info

def get_results_dir(model_type, data_local=True):
    if data_local:
        resdir = '/data/romain/PVsynth/eval_cnn/baby/'
    else:
        resdir = '/network/lustre/iss02/cenir/analyse/irm/users/romain.valabregue/PVsynth/eval_cnn/baby'
    ress=[]

    #ress.append(resdir+'eval_T2_model_5suj_motBigAff_ep30')
    if 'eval_T2' in model_type:
        ress.append(resdir + 'eval_T2_model_fetaBgT2_hcp_ep1')
        ress.append(resdir + 'eval_T2_model_hcpT2_elanext_5suj_BigAff_ep1')
        ress.append(resdir + 'eval_T2_model_hcpT2_elanext_5suj_ep1')
        ress.append(resdir + 'eval_T2_model_hcpT2_elanext_5suj_Mote30BigAff_ep2')
    if 'eval_T1' in model_type:
        ress.append(resdir+'eval_T1_model_fetaBgT2_hcp_ep1')
        ress.append(resdir+'eval_T1_model_hcpT2_elanext_5suj_BigAff_ep1')
        ress.append(resdir+'eval_T1_model_hcpT2_elanext_5suj_ep1')
        ress.append(resdir+'eval_T1_model_hcpT2_elanext_5suj_Mote30BigAff_ep2')

    for rr in ress:
        if not os.path.exists(rr):
            print(f'WARNING model {rr} does not exist ')

    resname = [os.path.basename(pp) for pp in ress]

    return ress, resname
