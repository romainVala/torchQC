import torch
import torch.nn.functional as F


def channel_metrics(prediction, target, metric):
    """
    Compute a given metric on every channel of the volumes.
    """
    if target.shape[1] == 1:
        target = torch.cat([target, 1 - target], dim=1)
    prediction = F.softmax(prediction, dim=1)
    channels = list(range(target.shape[1]))
    res = []
    for channel in channels:
        res.append(metric(prediction[:, channel, ...], target[:, channel, ...]))

    return torch.tensor(res)


def between_channel_metrics(prediction, target, metric, cut=0.5):
    """
    Compute a given metric on every channel using masks computed from the other channels.
    """
    if target.shape[1] == 1:
        target = torch.cat([target, 1 - target], dim=1)
    prediction = F.softmax(prediction, dim=1)
    channels = list(range(target.shape[1]))
    res = []
    for i in channels:
        for j in channels:
            if i != j:
                mask = target[:, j, ...] > cut
            else:
                mask = (target[:, j, ...] > 0) * (target[:, j, ...] < 1)
            res.append(metric(prediction[:, i, ...] * mask, target[:, i, ...] * mask))

    return torch.tensor(res)


def mean_metric(prediction, target, metric):
    """
    Compute a given metric on every channel of the volumes and average them.
    """
    if target.shape[1] == 1:
        target = torch.cat([target, 1 - target], dim=1)
    prediction = F.softmax(prediction, dim=1)
    channels = list(range(target.shape[1]))
    res = 0
    for channel in channels:
        res += metric(prediction[:, channel, ...], target[:, channel, ...])

    return res / len(channels)