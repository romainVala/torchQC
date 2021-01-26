import argparse
import torchio
import torch
import torch.nn.functional as F
from segmentation.config import Config
from segmentation.utils import to_numpy
import nibabel as nib


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
    parser.add_argument('-f', '--filename', type=str,
                        default='prediction.nii.gz',
                        help='Filename used to save the prediction,'
                             'default is "prediction.nii.gz"')
    parser.add_argument('-e','--exlcude_to_softmax', type=int, default=0,
                        help='nb volume to exclude (from the end) from the softmax activation')
    parser.add_argument('-c','--CropOrPad', type=str, default='None',
                        help='tuple of target dim')

    args = parser.parse_args()
    nb_vol_exclude = int(args.exlcude_to_softmax)
    vol_crop_pad = eval(args.CropOrPad)

    volume = torchio.ScalarImage(path=args.volume)
    tscale = torchio.RescaleIntensity(percentiles=(0,99))
    volume = tscale(volume)

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

    with torch.no_grad():
        prediction = model(volume.data.unsqueeze(0).float().to(device))

    if nb_vol_exclude>0:
        pp = F.softmax(prediction[0,:-nb_vol_exclude,...].unsqueeze(0), dim=1)
        prediction[0,:-nb_vol_exclude,...] = pp[0]
    else:
        prediction = F.softmax(prediction, dim=1)

    image = nib.Nifti1Image(
        to_numpy(prediction[0].permute(1, 2, 3, 0)),
        volume.affine
    )
    nib.save(image, args.filename)
