import numpy as np
from segmentation.utils import to_numpy, to_var, mean_metric


def default_t_norm(prediction, target, background=False):
    device = prediction.device
    np_prediction = to_numpy(prediction)
    np_target = to_numpy(target)
    if background:
        res = np.maximum(np_prediction - np_target, 0)
    else:
        res = np.minimum(np_prediction, np_target)
    return to_var(res, device)


def fuzzy_true_positives(prediction, target, t_norm=default_t_norm):
    return t_norm(prediction, target, background=False).sum()


def fuzzy_true_negatives(prediction, target, t_norm=default_t_norm):
    return t_norm(1 - prediction, 1 - target, background=False).sum()


def fuzzy_false_positives(prediction, target, t_norm=default_t_norm):
    return t_norm(prediction, target, background=True).sum()


def fuzzy_false_negatives(prediction, target, t_norm=default_t_norm):
    return t_norm(1 - prediction, 1 - target, background=True).sum()


def mean_fuzzy_true_positives(prediction, target, t_norm=default_t_norm):
    return mean_metric(prediction, target, lambda p, t: fuzzy_true_positives(p, t, t_norm))


def mean_fuzzy_true_negatives(prediction, target, t_norm=default_t_norm):
    return mean_metric(prediction, target, lambda p, t: fuzzy_true_negatives(p, t, t_norm))


def mean_fuzzy_false_positives(prediction, target, t_norm=default_t_norm):
    return mean_metric(prediction, target, lambda p, t: fuzzy_false_positives(p, t, t_norm))


def mean_fuzzy_false_negatives(prediction, target, t_norm=default_t_norm):
    return mean_metric(prediction, target, lambda p, t: fuzzy_false_negatives(p, t, t_norm))
