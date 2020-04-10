import numpy as np
import torch
from torch.nn import Conv3d, ConvTranspose3d, ReLU, LeakyReLU, MaxPool3d, Module, Sigmoid, Softmax
from torch.nn.functional import pad
import torch.nn.functional as F
from torch import nn
from util_affine import apply_affine_to_data
from collections import OrderedDict


########################################################################################################################


def check_size(x, y):
    output = y
    if x.size() != y.size():
        output = zero_pad(x, y)
    return output


def zero_pad(x, y):
    """
    Pad y to the size of x on the 3 last dimensions
    :param x: 5 dimensions tensor
    :param y: 5 dimensions tensor
    :return: padded value of x
    """
    in_shape = np.asarray(x.shape)[2:]
    out_shape = np.asarray(y.shape)[2:]
    to_pad = in_shape - out_shape
    left_pad = np.floor(to_pad / 2).astype(int)
    right_pad = np.ceil(to_pad / 2).astype(int)
    pad_tuple = tuple(np.asarray(list(zip(left_pad, right_pad))).reshape(-1)[::-1])
    output = pad(y, pad=pad_tuple, mode="constant", value=0)
    return output


########################################################################################################################

class ConvBlock3D(Module):
    """
    Helper to build a 3D convolutional block composed of a convolution + activation + pooling
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0,
                 activation=ReLU(inplace=True), pooling=MaxPool3d(kernel_size=2), same_padding=False):

        super(ConvBlock3D, self).__init__()
        self.conv = Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                           dilation=dilation, padding=padding)
        self.activation = activation
        self.pooling = pooling
        self.same_padding = same_padding

    def forward(self, x):
        output = self.conv(x)
        output = self.activation(output)
        if self.pooling:
            output = self.pooling(output)
        if self.same_padding:
            output = zero_pad(x, output)

        return output

########################################################################################################################


class ConvTransposeBlock3D(Module):
    """
    Helper to build a 3D convolutional transpose block
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 activation=ReLU(inplace=True)):
        super(ConvTransposeBlock3D, self).__init__()
        self.convt = ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, output_padding=output_padding, dilation=dilation)

        self.activation = activation

    def forward(self, x):
        output = self.convt(x)
        output = self.activation(output)
        return output

########################################################################################################################


class SmallUnet(Module):

    def __init__(self, in_channels, out_channels=1 ):
        super(SmallUnet, self).__init__()
        ############### ENCODER ##############################
        #### CONV 1 ####
        self.conv1 = ConvBlock3D(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, dilation=2,
                                 pooling=None, activation=ReLU(inplace=True), same_padding=True)

        self.pool1 = MaxPool3d(kernel_size=2)
        #### CONV 2 ####
        self.conv2 = ConvBlock3D(in_channels=16, out_channels=32, kernel_size=3, stride=1, dilation=2,
                                 pooling=None, activation=ReLU(inplace=True), same_padding=True)

        self.pool2 = MaxPool3d(kernel_size=2)
        ############### DECODER ############################
        #### UPCONV 1 ####
        self.conv3 = ConvBlock3D(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=2,
                                 pooling=None, activation=ReLU(inplace=True), same_padding=True)

        self.convt1 = ConvTransposeBlock3D(in_channels=32, out_channels=16, kernel_size=2, stride=2,
                                           activation=ReLU(inplace=True))
        #### UPCONV 2 ####
        self.conv4 = ConvBlock3D(in_channels=16 + 32, out_channels=16, kernel_size=3, stride=1, dilation=1,
                                 activation=ReLU(inplace=True), pooling=None, same_padding=True)

        self.convt2 = ConvTransposeBlock3D(in_channels=16, out_channels=4, kernel_size=2, stride=2,
                                           activation=ReLU(inplace=True))
        #### FINAL LAYER ####

        self.conv5 = ConvBlock3D(in_channels=4 + 16, out_channels=out_channels, same_padding=True, kernel_size=3,
                                 pooling=None, activation=Softmax() if out_channels > 2 else Sigmoid())

        #self.conv5.register_backward_hook(lambda z, x, y: print("Grad in {}\n Grad out:{}".format(x, y)))


    def forward(self, x):
        ############### ENCODER ##############################
        #### CONV 1 ####
        out_conv1 = self.conv1(x)
        out_pool1 = self.pool1(out_conv1)
        #### CONV 2 ####
        out_conv2 = self.conv2(out_pool1)
        out_pool2 = self.pool2(out_conv2)
        ## CLEANING
        del out_pool1
        ############### DECODER ############################
        #### UPCONV 1 ####
        out_conv3 = self.conv3(out_pool2)
        out_convt1 = self.convt1(out_conv3)
        ## CLEANING
        del out_pool2, out_conv3
        ## CONCAT 1
        out_convt1 = check_size(out_conv2, out_convt1)
        concat1 = torch.cat([out_convt1, out_conv2], dim=1)
        #### UPCONV 2 ####
        out_conv4 = self.conv4(concat1)
        out_convt2 = self.convt2(out_conv4)
        ## CLEANING
        del out_conv2, out_convt1, concat1, out_conv4
        ## CONCAT 2
        out_convt2 = check_size(out_conv1, out_convt2)
        concat2 = torch.cat([out_convt2, out_conv1], dim=1)
        #### FINAL LAYER ####
        out_conv5 = self.conv5(concat2)
        ## CLEANING
        del out_convt2, out_conv1, concat2
        return out_conv5


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class PadMaxPool3d(nn.Module):
    def __init__(self, kernel_size, stride, return_indices=False, return_pad=False):
        super(PadMaxPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool3d(kernel_size, stride, return_indices=return_indices)
        self.pad = nn.ConstantPad3d(padding=0, value=0)
        self.return_indices = return_indices
        self.return_pad = return_pad

    def set_new_return(self, return_indices=True, return_pad=True):
        self.return_indices = return_indices
        self.return_pad = return_pad
        self.pool.return_indices = return_indices

    def forward(self, f_maps):
        coords = [self.stride - f_maps.size(i + 2) % self.stride for i in range(3)]
        for i, coord in enumerate(coords):
            if coord == self.stride:
                coords[i] = 0

        self.pad.padding = (coords[2], 0, coords[1], 0, coords[0], 0)

        if self.return_indices:
            output, indices = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, indices, (coords[2], 0, coords[1], 0, coords[0], 0)
            else:
                return output, indices

        else:
            output = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, (coords[2], 0, coords[1], 0, coords[0], 0)
            else:
                return output

class ConvN_FC3(nn.Module):

    def __init__(self, dropout=0.5, n_classes=1, in_size=[182,218,182],
                 conv_block = [15, 25, 50, 50], linear_block = [50, 40],
                 output_fnc=None, batch_norm=True):

        super(ConvN_FC3, self).__init__()

        self.encoding_blocks = nn.ModuleList()
        for nb_layer in conv_block:
            if len(self.encoding_blocks ) == 0:
                nb_in = 1
                out_size = np.ceil((np.array(in_size) - 2) / 2)
            else :
                out_size = np.ceil((out_size - 2) / 2)

            if batch_norm:
                one_conv = nn.Sequential(
                    nn.Conv3d(nb_in, nb_layer, 3),
                    nn.BatchNorm3d(nb_layer),
                    nn.ReLU(),
                    PadMaxPool3d(2, 2) )
            else:
                one_conv = nn.Sequential(
                    nn.Conv3d(nb_in, nb_layer, 3),
                    nn.ReLU(),
                    PadMaxPool3d(2, 2))

            self.encoding_blocks.append(one_conv)
            nb_in = nb_layer
        self.encoding_blocks = nn.Sequential(*self.encoding_blocks)

        print('last layer out size {} * {} '.format(out_size,nb_layer))
        out_flatten = np.prod(out_size) * nb_layer
        print('size flatten {}'.format(out_flatten))

        in_size = out_flatten
        self.classifier = nn.ModuleList()
        self.classifier.append(Flatten())

        for nb_out in linear_block :
            on_lin = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(int(in_size) , nb_out),
                nn.ReLU(),)
            in_size = nb_out
            self.classifier.append(on_lin)

        self.classifier.append(nn.Linear(in_size, n_classes))
        self.classifier = nn.Sequential(*self.classifier)

        self.output_fnc = None
        if output_fnc is not None:
            if output_fnc is 'tanh':
                self.output_fnc = torch.tanh


    def forward(self, x):
        #for eb in self.encoding_blocks:  #needed if keeping the ModuleList
        #    x = eb(x)
        x = self.encoding_blocks(x)
        x = self.classifier(x)
        if self.output_fnc is not None:
            x = self.output_fnc(x)
        return x


class STNConv(nn.Module):
    def __init__(self,  in_size=[182, 218, 182], dropout=0.5 ,
                 conv_block = [8, 16, 32, 64, 128], linear_block = [50, 40], align_corners=False ):
        super(STNConv, self).__init__()

        self.encode = ConvN_FC3(in_size=in_size, dropout=dropout, n_classes=12,
                                      conv_block=conv_block, linear_block=linear_block)


        # Initialize the weights/bias with identity transformation
        self.encode.classifier[-1].weight.data.zero_()
        self.encode.classifier[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
        self.theta = None
        self.align_corners = align_corners

    # Spatial transformer network forward function
    def stn(self, x):
        theta = self.encode(x)
        theta = theta.view(-1, 3, 4)
        #print('theta {}'.format(theta.shape))

        # 1. Generates a 2d flow field, given a batch of affine matrices theta
        # 2. grid should have most values in the range of [-1, 1].
        # grid = homography_grid(theta, x.size())
        # grid = F.affine_grid(theta, x.size(), align_corners=self.align_corners)
        # x is a volumetric input
        # xout = F.grid_sample(x, grid, align_corners=self.align_corners)

        xout = apply_affine_to_data(x, theta, align_corners = self.align_corners)
        self.theta = theta #.detach().cpu()

        return xout

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x


class Conv4_FC3(nn.Module):
    """
    Classifier for a binary classification task

    Patch level architecture used on Minimal preprocessing

    This network is the implementation of this paper:
    'Multi-modality cascaded convolutional neural networks for Alzheimer's Disease diagnosis'
    """

    def __init__(self, dropout=0.5, n_classes=1, in_size=[182,218,182]):
        super(Conv4_FC3, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 15, 3),
            nn.BatchNorm3d(15),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(15, 25, 3),
            nn.BatchNorm3d(25),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(25, 50, 3),
            nn.BatchNorm3d(50),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(50, 50, 3),
            nn.BatchNorm3d(50),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
        )
        out_size = np.ceil((np.array(in_size) - 2) / 2)
        out_size = np.ceil((out_size - 2) / 2)
        out_size = np.ceil((out_size - 2) / 2)
        out_size = np.ceil((out_size - 2) / 2)
        out_flatten = np.prod(out_size) * 50

        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(int(out_flatten) , 50),
            nn.ReLU(),

            nn.Dropout(p=dropout),
            nn.Linear(50, 40),
            nn.ReLU(),

            nn.Linear(40, n_classes)
        )

        #self.flattened_shape = [-1, 50, 2, 2, 2]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

def summary(model, input_size, batch_size=-1, device="cuda"):
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            #print(class_name)
            #print(module_idx)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            #rrr added for fernando unet where forward can return a list ...
            if isinstance(input[0], list):
                input = input[1]

            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                # rrr added for fernando unet where forward can return a list ...
                if isinstance(output[0], list):
                    output = output[1]
                #rrr there may be list of list
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
                # listout = []
                # for o in output:
                #     if isinstance(o, (list, tuple)):
                #         listout.append( [ [-1] + list(oo.size())[1:] for oo in o] )
                #     else:
                #         listout.append([-1] + list(o.size())[1:] )
                # print('RRRRRRRRRRRRR')
                # print(listout)
                # summary[m_key]["output_shape"] = listout
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    #x = [torch.rand([2] + list(in_size)).type(dtype) for in_size in input_size]

    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    #print('RRdR')
    #for xx in x:
    #    print(xx.shape)
    model(*x)
    #print('AAA')

    # remove these hooks
    for h in hooks:
        h.remove()

    txt = "----------------------------------------------------------------"
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    txt += '\n{}'.format(line_new)
    txt += '\n{}'.format("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        txt += '\n{}'.format(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    txt += '\n{}'.format("================================================================")
    txt += '\n{}'.format("Total params: {0:,}".format(total_params))
    txt += '\n{}'.format("Trainable params: {0:,}".format(trainable_params))
    txt += '\n{}'.format("Non-trainable params: {0:,}".format(total_params - trainable_params))
    txt += '\n{}'.format("----------------------------------------------------------------")
    txt += '\n{}'.format("Input size (MB): %0.2f" % total_input_size)
    txt += '\n{}'.format("Forward/backward pass size (MB): %0.2f" % total_output_size)
    txt += '\n{}'.format("Params size (MB): %0.2f" % total_params_size)
    txt += '\n{}'.format("Estimated Total Size (MB): %0.2f" % total_size)
    txt += '\n{}'.format("----------------------------------------------------------------")
# return summary
    return txt

def load_existing_weights_if_exist(resdir, model, model_name='model', log=None, device='cuda', index_mod=-1):
    from utils_file import  gfile, get_parent_path

    ep_start = 0

    resume_mod = gfile(resdir, '.*pt$')
    if len(resume_mod) > 0:
        dir_mod, fn = get_parent_path(resume_mod)
        ffn = [ff[ff.find('_ep')+3:-3] for ff in fn]
        key_list = []
        for fff, fffn in zip(ffn,fn):
            if '_it' in fff:
                ind = fff.find('_it')
                ep = int(fff[0:ind])
                it = int(fff[ind+3:])
            else:
                ep = int(fff)
                it = 100000000
            key_list.append([fffn, ep, it])
        aa = np.array(sorted(key_list, key=lambda x: (x[1], x[2])))
        name_sorted, ep_sorted = aa[:,0],  aa[:,1]

        ep_start = int(ep_sorted[index_mod])
        thelast = dir_mod[0] + '/' + name_sorted[index_mod]
        log.info('RESUME model from epoch {} weight loaded from {}'.format(ep_start, thelast))

        tl = torch.load(thelast, map_location=device)
        if model_name not in tl:
            model_name = list(tl.items())[0][0]

        prefix = 'model.'
        state_dict = tl[model_name]
        aa = next(iter(state_dict))
        if prefix in aa:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[len(prefix):]  # remove 'module.' of dataparallel
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(tl[model_name])
    else:
        log.info('New training starting epoch {}'.format(ep_start))
        thelast = resdir

    log.info('Resdir is {}'.format(resdir))

    return ep_start, get_parent_path([thelast])[1][0]

"""Common image segmentation losses.
"""

import torch

from torch.nn import functional as F


def bce_loss(true, logits, pos_weight=None):
    """Computes the weighted binary cross-entropy loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, 1, H, W]. Corresponds to
            the raw output or logits of the model.
        pos_weight: a scalar representing the weight attributed
            to the positive class. This is especially useful for
            an imbalanced dataset.
    Returns:
        bce_loss: the weighted binary cross-entropy loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        logits.float(),
        true.float(),
        pos_weight=pos_weight,
    )
    return bce_loss


def ce_loss(true, logits, weights, ignore=255):
    """Computes the weighted multi-class cross-entropy loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        weight: a tensor of shape [C,]. The weights attributed
            to each class.
        ignore: the class index to ignore.
    Returns:
        ce_loss: the weighted multi-class cross-entropy loss.
    """
    ce_loss = F.cross_entropy(
        logits.float(),
        true.long(),
        ignore_index=ignore,
        weight=weights,
    )
    return ce_loss


class dice_loss(torch.nn.Module):

    def __init__(self, type=1):
        super(dice_loss, self).__init__()
        self.type = type

    def forward(self, input, target):
        if self.type == 1 :

            ll =  dice_loss_fonction(input, target)

        elif self.type == 2 :
            dice = dice_loss_fonction(input, target)
            bce_logits = F.binary_cross_entropy_with_logits(input, target) #sigmoid done in the model
            ll = dice + bce_logits

        return ll


def dice_loss_fonction( input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    dice = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

    return dice

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = (y_true * y_pred).sum()
    return (2. * intersection + smooth) / ( y_true_f.sum() + y_pred_f.sum() + smooth)


# Just the opposite of the dice coefficient to convert into a minimizable loss
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_loss_bin(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(probas, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)


def calc_loss(pred, target, loss, metrics, bce_weight=0.5):
    bce_logits = F.binary_cross_entropy_with_logits(pred, target) #sigmoid done in the model
    bce = F.binary_cross_entropy(pred, target)
    #pred = F.sigmoid(pred)
    dice = dice_loss_fonction(pred, target)

    labels = (target > 0.5).float()
    dice_binlab = dice_loss_fonction(pred, labels)
    pred_bin = (pred > 0.5).float()
    dice_binboth = dice_loss_fonction(pred_bin, labels)

    #loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['bce_lgits'] += bce_logits.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['d_binlab'] += dice_binlab.data.cpu().numpy() * target.size(0)
    metrics['d_binboth'] += dice_binboth.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss * target.size(0)
    #metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    #return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    txt = "{}: {}".format(phase, ", ".join(outputs) )
    return txt



########################################################################################################################
################################################ T E S T ###############################################################

"""
import numpy as np
from torch.nn import  MSELoss
from torch.optim import SGD

t1_tensor = torch.randn(1, 1, 182, 218, 182)
t1_seg = torch.ones(size=(1, 3, 182, 218, 182))

unet = SmallUnet(in_channels=1, out_channels=3)
optim = SGD(unet.parameters(), lr=0.003)
## TRAIN
optim.zero_grad()
res = unet(t1_tensor)
loss = MSELoss()(res, t1_seg)
loss.backward()
optim.step()
"""
