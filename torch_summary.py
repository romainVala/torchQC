"""
Utility function to generate a Keras-like summary for Pytorch models.
Adapted from https://github.com/sksq96/pytorch-summary
"""

import torch
import torch.nn as nn

from collections import OrderedDict
import numpy as np


torch_classes = [x for x in dir(torch.nn) if x[0].isupper()]


def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None, accept_classes=None):
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes, accept_classes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None, accept_classes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)
    if accept_classes is None:
        accept_classes = torch_classes

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            if class_name not in accept_classes:
                return output

            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)

            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                for o in output:
                    print('output type', type(o), m_key)
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
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
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
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
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"

    return summary_str, (total_params, trainable_params)


def print_FOV(model, input_size, device=torch.device('cuda:0'), dtypes=None, upsample_list=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)
    if upsample_list is None:
        upsample_list = ['ConvTranspose', 'Upsample']

    def hook(module, input, output):
        if res['keep_going'] and hasattr(module, 'kernel_size'):
            if any(map(lambda e: e in str(module.__class__), upsample_list)):
                res['keep_going'] = False
                return
            k = np.array(module.kernel_size)
            s = np.array(module.stride)
            d = np.array(module.dilation)
            res['r'] += (k - 1) * res['j'] * d
            res['j'] *= s

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    res = {'j': 1, 'r': 1, 'keep_going': True}
    hooks = []

    # register hooks
    model.apply(lambda module: hooks.append(module.register_forward_hook(hook)))

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print('Receptive field of the network:', res['r'])
