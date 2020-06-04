from segmentation.utils import mean_metric


class OverlapMetric:
    """
    Implements the different overlap based metrics (TP, TN, FP, FN) on binarized volumes.

    Args:
        cut: the threshold to binarize the volumes.
    """
    def __init__(self, cut=0.5):
        self.cut = cut

    def true_positives(self, prediction, target):
        predicted_mask = prediction > self.cut
        target_mask = target > self.cut
        return ((predicted_mask == target_mask) * target_mask).sum()

    def true_negatives(self, prediction, target):
        predicted_mask = prediction > self.cut
        target_mask = target > self.cut
        return ((predicted_mask == target_mask) * (~ target_mask)).sum()

    def false_positives(self, prediction, target):
        predicted_mask = prediction > self.cut
        target_mask = target > self.cut
        return ((predicted_mask != target_mask) * (~ target_mask)).sum()

    def false_negatives(self, prediction, target):
        predicted_mask = prediction > self.cut
        target_mask = target > self.cut
        return ((predicted_mask != target_mask) * target_mask).sum()

    def mean_true_positives(self, prediction, target):
        return mean_metric(prediction, target, self.true_positives)

    def mean_true_negatives(self, prediction, target):
        return mean_metric(prediction, target, self.true_negatives)

    def mean_false_positives(self, prediction, target):
        return mean_metric(prediction, target, self.false_positives)

    def mean_false_negatives(self, prediction, target):
        return mean_metric(prediction, target, self.false_negatives)
