from segmentation.metrics.utils import mean_metric


class VolumeMetric:
    """
    Implements different metrics related to volumes.

    Args:
        smooth: a value used to avoid division by zero in volume ratio.
    """
    def __init__(self, smooth=0.):
        self.smooth = smooth

    def volume_ratio(self, prediction, target):
        target_volume = target.sum()
        prediction_volume = prediction.sum()
        return prediction_volume / (target_volume + self.smooth)

    def mean_volume_ratio(self, prediction, target):
        return mean_metric(prediction, target, self.volume_ratio)
