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
        Implements the different fuzzy overlap based metrics (TP, TN, FP, FN) using a given t_norm.

        Args:
            t_norm: the t_norm to use to compute the agreement between the volumes.
        """
    def __init__(self, t_norm=minimum_t_norm, mask_cut=0.99):
        self.t_norm = t_norm
        self.mask_cut = mask_cut

    def _apply_mask(self, mapping, mask):
        if mask is None:
            return mapping
        mask = mask >= self.mask_cut
        return mapping * mask

    def fuzzy_true_positives_map(self, prediction, target, mask=None):
        mapping = self.t_norm(prediction, target, background=False)
        return self._apply_mask(mapping, mask)

    def fuzzy_true_negatives_map(self, prediction, target, mask=None):
        mapping = self.t_norm(1 - prediction, 1 - target, background=False)
        return self._apply_mask(mapping, mask)

    def fuzzy_false_positives_map(self, prediction, target, mask=None):
        mapping = self.t_norm(prediction, target, background=True)
        return self._apply_mask(mapping, mask)

    def fuzzy_false_negatives_map(self, prediction, target, mask=None):
        mapping = self.t_norm(1 - prediction, 1 - target, background=True)
        return self._apply_mask(mapping, mask)

    def mean_fuzzy_true_positives(self, prediction, target, **kwargs):
        return mean_metric(prediction, target, lambda p, t, m: self.fuzzy_true_positives_map(p, t, m).sum(), **kwargs)

    def mean_fuzzy_true_negatives(self, prediction, target, **kwargs):
        return mean_metric(prediction, target, lambda p, t, m: self.fuzzy_true_negatives_map(p, t, m).sum(), **kwargs)

    def mean_fuzzy_false_positives(self, prediction, target, **kwargs):
        return mean_metric(prediction, target, lambda p, t, m: self.fuzzy_false_positives_map(p, t, m).sum(), **kwargs)

    def mean_fuzzy_false_negatives(self, prediction, target, **kwargs):
        return mean_metric(prediction, target, lambda p, t, m: self.fuzzy_false_negatives_map(p, t, m).sum(), **kwargs)
