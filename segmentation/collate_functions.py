import re
import torch
import numpy as np
from torch._six import container_abcs, string_classes, int_classes


np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def history_collate(batch):
    """
    Adapt default_collate from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    to handle random transform history in batches.
    """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return history_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        # The only change to the original function is here:
        # if elem has attribute 'history', then a key 'history' is added to the batch
        # which value is the list of the history of the elements of the batch
        dictionary = {key: history_collate([d[key] for d in batch]) for key in elem}
        if hasattr(elem, 'history'):
            dictionary.update({
                'history': [d.history for d in batch]
            })
        return dictionary
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(history_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [history_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def history_collate_partial_metrics(batch):
    """
    Adapt default_collate from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    to handle random transform history in batches. when metrics is not present in all sample
    but not used since now we go with history
    """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return history_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        # The only change to the original function is here:
        # if elem has attribute 'history', then a key 'history' is added to the batch
        # which value is the list of the history of the elements of the batch
        batch_keys = set.union(*[set(d.keys()) for d in batch])
        dictionary = {}
        metrics_idx = []
        metrics_values = []
        for key in batch_keys:
            to_collate = []
            for idx, d in enumerate(batch):
                if key is "metrics":
                    if "metrics" in d.keys():
                        metrics_idx.append(idx)
                        metrics_values.append(d["metrics"])
                    continue
                to_collate.append(d[key])
            if key is "metrics":
                continue
            dictionary[key] = history_collate(to_collate)

        if "metrics" in batch_keys:
            res_metrics = metrics_values
            if len(metrics_idx) != len(batch):
                res_metrics = np.asarray([dict()]*4)
                res_metrics[metrics_idx] = metrics_values
            dictionary.update({"metrics": res_metrics})

        if hasattr(elem, 'history'):
            dictionary.update({
                'history': [d.history for d in batch]
            })
        return dictionary
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(history_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [history_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))