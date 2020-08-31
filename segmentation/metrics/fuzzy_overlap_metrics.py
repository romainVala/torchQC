import numpy as np
from segmentation.utils import to_numpy, to_var
from segmentation.metrics.utils import mean_metric


def minimum_t_norm(prediction, target, background=False):
    device = prediction.device
    np_prediction = to_numpy(prediction)
    np_target = to_numpy(target)
    if background:
        res = np.maximum(np_prediction - np_target, 0)
    else:
        res = np.minimum(np_prediction, np_target)
    return to_var(res, device)


def product_t_norm(prediction, target, background=False):
    return prediction * target


class FuzzyOverlapMetric:
    """
        Implements the different fuzzy overlap based metrics (TP, TN, FP, FN)
        using a given t_norm.

        Args:
            t_norm: the t_norm to use to compute the agreement between the
                volumes.
        """
    def __init__(self, t_norm=minimum_t_norm):
        self.t_norm = t_norm

    def fuzzy_true_positives_map(self, prediction, target):
        return self.t_norm(prediction, target, background=False)

    def fuzzy_true_negatives_map(self, prediction, target):
        return self.t_norm(1 - prediction, 1 - target, background=False)

    def fuzzy_false_positives_map(self, prediction, target):
        return self.t_norm(prediction, target, background=True)

    def fuzzy_false_negatives_map(self, prediction, target):
        return self.t_norm(1 - prediction, 1 - target, background=True)

    def mean_fuzzy_true_positives(self, prediction, target):
        return mean_metric(
            prediction, target, lambda p, t: self.fuzzy_true_positives_map(p, t).sum()
        )

    def mean_fuzzy_true_negatives(self, prediction, target):
        return mean_metric(
            prediction, target, lambda p, t: self.fuzzy_true_negatives_map(p, t).sum()
        )

    def mean_fuzzy_false_positives(self, prediction, target):
        return mean_metric(
            prediction, target, lambda p, t: self.fuzzy_false_positives_map(p, t).sum()
        )

    def mean_fuzzy_false_negatives(self, prediction, target):
        return mean_metric(
            prediction, target, lambda p, t: self.fuzzy_false_negatives_map(p, t).sum()
        )
