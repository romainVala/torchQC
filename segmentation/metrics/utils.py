import torch
import torch.nn.functional as F


def mean_metric(prediction, target, metric, channels=None, mask=None):
    """
    Compute a given metric on every channel of the volumes and average them.
    """
    if (prediction.shape[1] - target.shape[1]) == 1:
        target = torch.cat([target, 1 - target.sum(dim=1, keepdim=True)], dim=1)
    prediction = F.softmax(prediction, dim=1)

    if channels is not None:
        prediction = prediction[:, channels, ...]
        target = target[:, channels, ...]

    channels = list(range(target.shape[1]))
    res = 0
    for channel in channels:
        res += metric(prediction[:, channel, ...], target[:, channel, ...], mask)

    return res / len(channels)
