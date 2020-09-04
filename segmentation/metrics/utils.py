import torch
import torch.nn.functional as F


class MetricOverlay:
    def __init__(self, metric, channels=None, mask=None, mask_cut=(0.99, 1),
                 binary=False, activation=None, binary_volumes=False):
        self.metric = metric
        self.channels = channels
        self.mask = mask
        self.mask_cut = mask_cut
        self.binary = binary
        self.activation = activation
        self.binary_volumes = binary_volumes

    def __call__(self, prediction, target):
        if (prediction.shape[1] - target.shape[1]) == 1:
            target = torch.cat([target, 1 - target.sum(dim=1, keepdim=True)],
                               dim=1)

        if self.binary_volumes:
            prediction = F.one_hot(prediction).permute(0, 4, 1, 2, 3).float()
            target = F.one_hot(target).permute(0, 4, 1, 2, 3).float()

        if self.activation is not None:
            prediction = self.activation(prediction)

        if self.binary:
            unlabeled_volume = target.sum(dim=1) < 0.5
            val, indices = torch.max(target, dim=1)
            extra_class = indices.max() + 1
            indices[val == 0] = extra_class
            indices[unlabeled_volume] = extra_class
            target = F.one_hot(indices)
            if (val == 0).sum() or unlabeled_volume.sum():
                target = target[..., :-1]
            target = target.permute(0, 4, 1, 2, 3).float()

        if self.mask is not None:
            min_cut, max_cut = self.mask_cut
            mask = (max_cut >= target[:, self.mask]) \
                * (target[:, self.mask] >= min_cut)
            prediction = prediction * mask
            target = target * mask

        if self.channels is not None:
            prediction = prediction[:, self.channels, ...]
            target = target[:, self.channels, ...]

        return self.metric(prediction, target)


def mean_metric(prediction, target, metric):
    """
    Compute a given metric on every channel of the volumes and average them.
    """
    channels = list(range(target.shape[1]))
    res = 0.
    for channel in channels:
        res += metric(prediction[:, channel, ...], target[:, channel, ...])

    return res / len(channels)
