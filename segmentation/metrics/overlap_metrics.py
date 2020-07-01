from segmentation.metrics.utils import channel_metrics, between_channel_metrics


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

    def per_channel_true_positives(self, prediction, target):
        return channel_metrics(prediction, target, self.true_positives)

    def per_channel_true_negatives(self, prediction, target):
        return channel_metrics(prediction, target, self.true_negatives)

    def per_channel_false_positives(self, prediction, target):
        return channel_metrics(prediction, target, self.false_positives)

    def per_channel_false_negatives(self, prediction, target):
        return channel_metrics(prediction, target, self.false_negatives)

    def between_channel_true_positives(self, prediction, target):
        return between_channel_metrics(prediction, target, self.true_positives)

    def between_channel_true_negatives(self, prediction, target):
        return between_channel_metrics(prediction, target, self.true_negatives)

    def between_channel_false_positives(self, prediction, target):
        return between_channel_metrics(prediction, target, self.false_positives)

    def between_channel_false_negatives(self, prediction, target):
        return between_channel_metrics(prediction, target, self.false_negatives)
