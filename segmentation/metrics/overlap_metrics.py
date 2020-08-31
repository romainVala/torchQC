from segmentation.metrics.utils import mean_metric


class OverlapMetric:
    """
    Implements the different overlap based metrics (TP, TN, FP, FN) on
    binarized volumes.

    Args:
        cut: the threshold to binarize the volumes.
    """
    def __init__(self, cut=0.5):
        self.cut = cut

    def true_positives_map(self, prediction, target):
        predicted_mask = prediction > self.cut
        target_mask = target > self.cut
        return (predicted_mask == target_mask) * target_mask

    def true_negatives_map(self, prediction, target):
        predicted_mask = prediction > self.cut
        target_mask = target > self.cut
        return (predicted_mask == target_mask) * (~ target_mask)

    def false_positives_map(self, prediction, target):
        predicted_mask = prediction > self.cut
        target_mask = target > self.cut
        return (predicted_mask != target_mask) * (~ target_mask)

    def false_negatives_map(self, prediction, target):
        predicted_mask = prediction > self.cut
        target_mask = target > self.cut
        return (predicted_mask != target_mask) * target_mask

    def mean_true_positives(self, prediction, target):
        return mean_metric(
            prediction, target, lambda p, t: self.true_positives_map(p, t).sum()
        )

    def mean_true_negatives(self, prediction, target):
        return mean_metric(
            prediction, target, lambda p, t: self.true_negatives_map(p, t).sum()
        )

    def mean_false_positives(self, prediction, target):
        return mean_metric(
            prediction, target, lambda p, t: self.false_positives_map(p, t).sum()
        )

    def mean_false_negatives(self, prediction, target):
        return mean_metric(
            prediction, target, lambda p, t: self.false_negatives_map(p, t).sum()
        )
