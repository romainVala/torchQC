import torch
import torch.nn.functional as F


def metric_overlay(prediction, target, metric, channels=None, mask=None,
                   mask_cut=0.99, binary=False, activation=lambda x: x):
    """ Overlay to apply computation to prediction and target before
    computing metric. """
    if (prediction.shape[1] - target.shape[1]) == 1:
        target = torch.cat([target, 1 - target.sum(dim=1, keepdim=True)], dim=1)

    prediction = activation(prediction)

    if binary:
        indices = torch.argmax(target, dim=1)
        target = F.one_hot(indices)
        target = target.permute(0, 4, 1, 2, 3).float()

    if mask is not None:
        mask = target[mask] > mask_cut
        prediction = prediction * mask
        target = target * mask

    if channels is not None:
        prediction = prediction[:, channels, ...]
        target = target[:, channels, ...]

    return metric(prediction, target)


def mean_metric(prediction, target, metric):
    """
    Compute a given metric on every channel of the volumes and average them.
    """
    channels = list(range(target.shape[1]))
    res = 0
    for channel in channels:
        res += metric(prediction[:, channel, ...], target[:, channel, ...])

    return res / len(channels)
