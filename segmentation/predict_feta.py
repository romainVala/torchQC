import argparse
import torchio
import torch
import torch.nn.functional as F
from segmentation.config import Config
from segmentation.utils import to_numpy
from segmentation.utils import to_var
import nibabel as nib
import glob, os
import numpy as np
def apply_post_transforms(tensors, affine=np.eye(4), post_transforms=None ): 

    if not post_transforms:
        return tensors, affine
    if len(post_transforms) == 0:
        return tensors, affine
    # Transforms apply on TorchIO subjects and TorchIO images require
    # 4D tensors
    transformed_tensors = []
    for i, tensor in enumerate(tensors):
        subject = torchio.Subject(
            pred=torchio.LabelMap(
                tensor=to_var(tensor, 'cpu'),
                affine=affine)
        )
        transformed = post_transforms(subject)
        tensor = transformed['pred']['data']
        transformed_tensors.append(tensor)
    new_affine = transformed['pred']['affine']
    transformed_tensors = torch.stack(transformed_tensors)
    device = 'cuda' if tensors.is_cuda else 'cpu'
    return to_var(transformed_tensors, device), new_affine
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--suj', type=str,default='/input_img',
                        help='Path to feta suj dir')
    parser.add_argument('-o', '--outputDir', type=str,default='/output',
                        help='Path to feta suj dir')
        
    parser.add_argument('-m', '--model', type=str,default='/workspace/python/model/model.pth.tar',
                        help='Path to the model used to make prediction')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Device used to run the model, default is "cuda"')
    parser.add_argument('-mm', '--model_module', type=str, default='unet',
                        help='Module of the model used to make prediction,'
                             'default is "unet"')
    parser.add_argument('-mn', '--model_name', type=str, default='UNet',
                        help='Name of the model used to make prediction,'
                             'default is "UNet"')
    parser.add_argument('-f', '--filename', type=str,
                        default='_seg_result.nii.gz',
                        help='Filename used to save the prediction,'
                             'default is "[SUJ]_seg_result.nii.gz"')
    parser.add_argument('-e','--exlcude_to_softmax', type=int, default=0,
                        help='nb volume to exclude (from the end) from the softmax activation')
    parser.add_argument('-c','--CropOrPad', type=str, default='None',
                        help='tuple of target dim')

    args = parser.parse_args()
    nb_vol_exclude = int(args.exlcude_to_softmax)
    vol_crop_pad = eval(args.CropOrPad)

    sujdir = args.suj

    #input_meta_dir = '/input_meta'  # input path of the meta information, including pathology and GA information 

    outputDir = args.outputDir #'/output'   # output path
    print(f'working on suj {sujdir}')
    
    T2wImagePath = glob.glob(os.path.join(sujdir, 'anat', '*_T2w.nii.gz'))[0]

    
    volume = torchio.ScalarImage(path=T2wImagePath)
    tscale = torchio.RescaleIntensity(percentiles=(0,99))
    volume = tscale(volume)
    tstd = torchio.ToCanonical()
    volume = tstd(volume)

    if vol_crop_pad:
        tpad = torchio.CropOrPad(target_shape=vol_crop_pad)
        volume = tpad(volume)


    model_struct = {
        'module': args.model_module,
        'name': args.model_name,
        'last_one': False,
        'path': args.model,
        'device': args.device
    }
    config = Config(None, None, save_files=False)
    model_struct = config.parse_model_file(model_struct)
    model, device = config.load_model(model_struct)
    model.eval()
    print(f'model loaded on {device}')

    with torch.no_grad():
        prediction = model(volume.data.unsqueeze(0).float().to(device))

    if nb_vol_exclude>0:
        pp = F.softmax(prediction[0,:-nb_vol_exclude,...].unsqueeze(0), dim=1)
        prediction[0,:-nb_vol_exclude,...] = pp[0]
    else:
        prediction = F.softmax(prediction, dim=1)

    tfeta_label = torchio.RemapLabels(
                remapping= {
                    "0": 0,
                    "1": 1,
                    "2": 2,
                    "3": 3,
                    "4": 0, #// skin is background for labe
                    "5": 4,
                    "6": 5,
                    "7": 6,
                    "8": 7,
                    "9": 2,
                })
    tseg = torchio.OneHot(invert_transform=True)
    tresamp = torchio.Resample(target = T2wImagePath ) #because of the initial ToCanonical !
    postT = torchio.Compose([tfeta_label, tseg, tresamp])
    affine = volume.affine
    prediction,affine = apply_post_transforms(prediction, affine=affine, post_transforms = postT)

    
    image = nib.Nifti1Image(
        to_numpy(prediction[0][0]),
        affine
    )
#   image = nib.Nifti1Image(
#        to_numpy(prediction[0].permute(1, 2, 3, 0)),
#        affine
#    )

    sub = os.path.split(T2wImagePath)[1].split('_')[0]     # to split the input directory and to obtain the suject name                                      

    
    fout = os.path.join(outputDir, sub +  args.filename)
    nib.save(image,fout)

