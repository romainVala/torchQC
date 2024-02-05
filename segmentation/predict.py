import argparse
import torchio
import torch
import torch.nn.functional as F
from segmentation.config import Config
from segmentation.utils import to_numpy
import nibabel as nib
import os, json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--volume', type=str,
                        help='Path to nifty volume on which to make prediction')
    parser.add_argument('-m', '--model', type=str,
                        help='Path to the model used to make prediction')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Device used to run the model, default is "cuda"')
    parser.add_argument('-mm', '--model_module', type=str, default='unet',
                        help='Module of the model used to make prediction,'
                             'default is "unet"')
    parser.add_argument('-mn', '--model_name', type=str, default='UNet',
                        help='Name of the model used to make prediction,'
                             'default is "UNet"')
    
    parser.add_argument('-mj', '--model_json', type=str, default='',
                        help='path to model json file,'
                             'default is ""')
    
    parser.add_argument('-f', '--filename', type=str,
                        default='prediction.nii.gz',
                        help='Filename used to save the prediction,'
                             'default is "prediction.nii.gz"')
    parser.add_argument('-e','--exlcude_to_softmax', type=int, default=0,
                        help='nb volume to exclude (from the end) from the softmax activation')
    parser.add_argument('-c','--CropOrPad', type=str, default='None',
                        help='tuple of target dim')
    parser.add_argument('-p','--PatchSize', type=int, default=0,
                        help='patch size to process the image (default 0 -> full input size)')
    parser.add_argument('-po','--PatchOverlap', type=int, default=0,
                        help='patch overlap  ')
    parser.add_argument('-bs','--Patch_batch_size', type=int, default=4,
                        help='batch size for the patch loader ')
    parser.add_argument('-sp', '--save_4D', type=bool, default=False, help='If True, generate 4D pred ')
    parser.add_argument('-bc','--BiggestComp', type=int, default=0,nargs='+',
                        help='batch size for the patch loader ')


    args = parser.parse_args()
    nb_vol_exclude = int(args.exlcude_to_softmax)
    vol_crop_pad = eval(args.CropOrPad)
    patch_size = args.PatchSize
    patch_overlap = args.PatchOverlap
    batch_size = args.Patch_batch_size
    save_4D = args.save_4D
    save_biggest_comp = args.BiggestComp

    volume = torchio.ScalarImage(path=args.volume)
    tscale = torchio.RescaleIntensity(out_min_max=(0,1), percentiles=(0,99))
    volume = tscale(volume)
    tcan = torchio.ToCanonical()
    volume = tcan(volume)

    if vol_crop_pad:
        tpad = torchio.CropOrPad(target_shape=vol_crop_pad)
        volume = tpad(volume)
        orig_shape = 0
    else:
        orig_shape = volume.shape[-3:]
        treshape = torchio.EnsureShapeMultiple(2**5)
        volume = treshape(volume)
        tback = torchio.CropOrPad(target_shape=orig_shape)
        print(f'suj new shape is {volume.shape} orig is {orig_shape}')

    model_struct = {
        'module': args.model_module,
        'name': args.model_name,
        'last_one': False,
        'path': args.model,
        'device': args.device,
        }

    if os.path.isfile(args.model_json):
        with open(args.model_json) as f:
            model_struct =  json.load(f)
            
        model_struct['path'] = args.model
        model_struct['last_one'] = False
        model_struct['device'] = args.device


    config = Config(None, None, save_files=False)
    model_struct = config.parse_model_file(model_struct)
    model, device = config.load_model(model_struct)
    #device = args.device
    device =  model_struct['device']


    model.eval()

    if patch_size==0:
        with torch.no_grad():
            print(device)
            prediction = model(volume.data.unsqueeze(0).float().to(device))
        if nb_vol_exclude > 0:
            pp = F.softmax(prediction[0, :-nb_vol_exclude, ...].unsqueeze(0), dim=1)
            prediction[0, :-nb_vol_exclude, ...] = pp[0]
        else:
            #print('WITH SOFTMAX')
            prediction = F.softmax(prediction, dim=1)

    else:
        suj = torchio.Subject({'t1': volume})
        grid_sampler = torchio.inference.GridSampler(
            suj, patch_size, patch_overlap, padding_mode='reflect'
        )
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
        aggregator = torchio.inference.GridAggregator(grid_sampler, overlap_mode='hann')
        print(f' preparing {len(patch_loader)} patches with batch size {batch_size}')
        for ii, one_patch in enumerate(patch_loader):
            #print(f'patch {ii}')
            with torch.no_grad():
                predictions = model(one_patch['t1']['data'].float().to(device))
            predictions = F.softmax(predictions, dim=1)
            #predictions = F.log_softmax(predictions, dim=1) #JUST FOR TRAINING ?
            locations = one_patch[torchio.LOCATION]
            aggregator.add_batch(predictions.to('cpu'), locations)  #if keept on gpu I do not uderstand memory goes up

        prediction = aggregator.get_output_tensor().unsqueeze(0)
        print('model estimation done')

    prediction = to_numpy(prediction)

    if save_biggest_comp:
        print(f'Biggest comp on {save_biggest_comp}')
        from segmentation.utils import get_largest_connected_component
        import numpy as np

        #find label index
        volume_mask = prediction > 0.5 #billot use 0.25 ... why so big ?
        for i_label in save_biggest_comp:
            tmp_mask = get_largest_connected_component(volume_mask[:, i_label, ...])
            prediction[:, i_label, ...] *= tmp_mask
        #renomalize posteriors todo what if sum over proba is null after connected compo ... ?
        # if np.sum() == 0
        prediction /= np.sum(prediction, axis=1)[:,np.newaxis,...]


    #image = nib.Nifti1Image(
    #    to_numpy(prediction[0].permute(1, 2, 3, 0)),
    #    volume.affine
    #)
    suj_pred = torchio.Subject(pred=torchio.LabelMap(tensor=prediction[0], affine=volume.affine) )

    if orig_shape:
        suj_pred = tback(suj_pred)

    out_filename = args.filename
    if ~(out_filename.endswith('.nii') | out_filename.endswith('.gz')):
        out_filename += '.nii.gz'

    if save_4D:
        print(f'saving {out_filename} threshold 0.01')
        suj_pred.pred.data[suj_pred.pred.data<0.01] = 0
        suj_pred.pred.save(out_filename)
        #nib.save(image, args.filename)

    tseg_bin = torchio.OneHot(invert_transform=True)
    suj_pred = tseg_bin(suj_pred)

    outdir =  os.path.dirname(out_filename)+'/' if os.path.dirname(out_filename) else "./"
    new_out_file = outdir + 'bin_' + os.path.basename(out_filename)
    print('saving bin version')
    suj_pred.pred.save(new_out_file)

