from segmentation.metrics.utils import mean_metric


class OverlapMetric:
    """
    Implements the different overlap based metrics (TP, TN, FP, FN) on binarized volumes.

    Args:
        cut: the threshold to binarize the volumes.
    """
    def __init__(self, cut=0.5, mask_cut=0.99):
        self.cut = cut
        self.mask_cut = mask_cut

    def _apply_mask(self, mapping, mask):
        if mask is None:
            return mapping
        mask = mask > self.mask_cut
        return mapping * mask

    def true_positives_map(self, prediction, target, mask=None):
        predicted_mask = prediction > self.cut
        target_mask = target > self.cut
        mapping = (predicted_mask == target_mask) * target_mask
        return self._apply_mask(mapping, mask)

    def true_negatives_map(self, prediction, target, mask=None):
        predicted_mask = prediction > self.cut
        target_mask = target > self.cut
        mapping = (predicted_mask == target_mask) * (~ target_mask)
        return self._apply_mask(mapping, mask)

    def false_positives_map(self, prediction, target, mask=None):
        predicted_mask = prediction > self.cut
        target_mask = target > self.cut
        mapping = (predicted_mask != target_mask) * (~ target_mask)
        return self._apply_mask(mapping, mask)

    def false_negatives_map(self, prediction, target, mask=None):
        predicted_mask = prediction > self.cut
        target_mask = target > self.cut
        mapping = (predicted_mask != target_mask) * target_mask
        return self._apply_mask(mapping, mask)

    def mean_true_positives(self, prediction, target, **kwargs):
        return mean_metric(prediction, target, lambda p, t, m: self.true_positives_map(p, t, m).sum(), **kwargs)

    def mean_true_negatives(self, prediction, target, **kwargs):
        return mean_metric(prediction, target, lambda p, t, m: self.true_negatives_map(p, t, m).sum(), **kwargs)

    def mean_false_positives(self, prediction, target, **kwargs):
        return mean_metric(prediction, target, lambda p, t, m: self.false_positives_map(p, t, m).sum(), **kwargs)

    def mean_false_negatives(self, prediction, target, **kwargs):
        return mean_metric(prediction, target, lambda p, t, m: self.false_negatives_map(p, t, m).sum(), **kwargs)
