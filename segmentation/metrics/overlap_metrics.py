from segmentation.utils import mean_metric


def true_positives(prediction, target, cut=0.5):
    predicted_mask = prediction > cut
    target_mask = target > cut
    return ((predicted_mask == target_mask) * target_mask).sum()


def true_negatives(prediction, target, cut=0.5):
    predicted_mask = prediction > cut
    target_mask = target > cut
    return ((predicted_mask == target_mask) * (~ target_mask)).sum()


def false_positives(prediction, target, cut=0.5):
    predicted_mask = prediction > cut
    target_mask = target > cut
    return ((predicted_mask != target_mask) * (~ target_mask)).sum()


def false_negatives(prediction, target, cut=0.5):
    predicted_mask = prediction > cut
    target_mask = target > cut
    return ((predicted_mask != target_mask) * target_mask).sum()


def mean_true_positives(prediction, target, cut=0.5):
    return mean_metric(prediction, target, lambda p, t: true_positives(p, t, cut))


def mean_true_negatives(prediction, target, cut=0.5):
    return mean_metric(prediction, target, lambda p, t: true_negatives(p, t, cut))


def mean_false_positives(prediction, target, cut=0.5):
    return mean_metric(prediction, target, lambda p, t: false_positives(p, t, cut))


def mean_false_negatives(prediction, target, cut=0.5):
    return mean_metric(prediction, target, lambda p, t: false_negatives(p, t, cut))
