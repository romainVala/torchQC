from doit_train import do_training, get_motion_transform, get_train_and_val_csv, get_cache_dir
from torchio.transforms import CropOrPad, RandomAffine, RescaleIntensity, ApplyMask, RandomBiasField, Interpolation
from optparse import OptionParser
from utils_file import get_parent_path

def get_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

def get_cmd_select_data_option():
    usage= "usage: %prog [options] run a model on a file "

    # Parse input arguments
    parser=OptionParser(usage=usage)

    parser.add_option("-i", "--image_in", action="callback", dest="image_in", default='', callback=get_comma_separated_args,
                                type='string', help="full path to the image to test, list separate path by , ")
    parser.add_option("--sample_dir", action="store", dest="sample_dir", default='',
                                type='string', help="instead of -i specify dir of saved sample ")
    parser.add_option("--batch_size", action="store", dest="batch_size", default=2, type="int",
                      help=" default 2")
    parser.add_option("--num_workers", action="store", dest="num_workers", default=0, type="int",
                      help=" default 0")

    parser.add_option("--add_cut_mask", action="store_true", dest="add_cut_mask", default=False,
                                help="if specifie it will adda cut mask (brain) transformation default False ")
    parser.add_option("--add_affine_zoom", action="store", dest="add_affine_zoom", default=0, type="float",
                      help=">0 means we add an extra affine transform with zoom value and if define rotations default 0")
    parser.add_option("--add_affine_rot", action="store", dest="add_affine_rot", default=0, type="float",
                      help=">0 means we add an extra affine transform with rotation values and if define rotations default 0")
    parser.add_option("--add_rescal_Imax", action="store_true", dest="add_rescal_Imax", default=False,
                                help="if specifie it will add a rescale intensity transformation default False ")
    parser.add_option("--add_mask_brain", action="store_true", dest="add_mask_brain", default=False,
                                help="if specifie it will add a apply_mask (name brain) transformation default False ")
    parser.add_option("--add_elastic1", action="store_true", dest="add_elastic1", default=False,
                                help="if specifie it will add a elastic1 transformation default False ")
    parser.add_option("--add_bias", action="store_true", dest="add_bias", default=False,
                                help="if specifie it will add a bias transformation default False ")
    parser.add_option("--add_orig", action="store_true", dest="add_orig", default=False,
                                help="if specifie it will add original image to sample ")
    parser.add_option("--target", action="store", dest="target", default='ssim',
                      help=" specify the objectiv target either 'ssim' or 'random_noise' (default ssim)")


    return parser

def get_dataset_from_option(options):

    fin = options.image_in
    dir_sample = options.sample_dir
    add_affine_zoom, add_affine_rot = options.add_affine_zoom, options.add_affine_rot

    batch_size, num_workers = options.batch_size, options.num_workers


    doit = do_training('/tmp/', 'not_use', verbose=True)
    # adding transformation
    tc = []
    name_suffix = ''
    #Attention pas de _ dans le name_suffix
    if options.add_cut_mask > 0:
        target_shape, mask_key = (182, 218, 182), 'brain'
        tc = [CropOrPad(target_shape=target_shape, mask_name=mask_key), ]
        name_suffix += '_tCropBrain'

    if add_affine_rot>0 or add_affine_zoom >0:
        if add_affine_zoom==0: add_affine_zoom=1 #0 -> no affine so 1
        tc.append( RandomAffine(scales=(add_affine_zoom, add_affine_zoom), degrees=(add_affine_rot, add_affine_rot),
                                image_interpolation = Interpolation.NEAREST ) )
        name_suffix += '_tAffineS{}R{}'.format(add_affine_zoom, add_affine_rot)

    # for hcp should be before RescaleIntensity
    mask_brain = False
    if options.add_mask_brain:
        tc.append(ApplyMask(masking_method='brain'))
        name_suffix += '_tMaskBrain'
        mask_brain = True

    if options.add_rescal_Imax:
        tc.append(RescaleIntensity(percentiles=(0, 99)))
        name_suffix += '_tRescale-0-99'

    if options.add_elastic1:
        tc.append(get_motion_transform(type='elastic1'))
        name_suffix += '_tElastic1'

    if options.add_bias:
        tc.append(RandomBiasField())
        name_suffix += '_tBias'

    if len(name_suffix)==0:
        name_suffix = '_Raw'

    target = None
    if len(tc)==0: tc = None

    add_to_load, add_to_load_regexp = None, None

    if len(dir_sample) > 0:
        print('loading from {}'.format(dir_sample))
        if options.add_orig:
            add_to_load, add_to_load_regexp = 'original', 'notused'

        data_name = get_parent_path(dir_sample)[1]
        if mask_brain and 'hcp' in data_name:
            add_to_load_regexp = 'brain_T'
            if add_to_load is None:
                add_to_load = 'brain'
            else:
                add_to_load += 'brain'

        doit.set_data_loader(batch_size=batch_size, num_workers=num_workers, load_from_dir=dir_sample, transforms=tc,
                             add_to_load=add_to_load, add_to_load_regexp=add_to_load_regexp)

        name_suffix = 'On_' + data_name + name_suffix
        target = options.target #'ssim' #suppose that if from sample, it should be simulation so set target
    else :
        print('working on ')
        for ff in fin:
            print(ff)

        doit.set_data_loader_from_file_list(fin, transforms=tc,
                                            batch_size=batch_size, num_workers=num_workers,
                                            mask_key=mask_key, mask_regex='^mask')

    return doit, name_suffix, target