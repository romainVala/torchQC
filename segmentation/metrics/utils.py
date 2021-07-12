import torch
import torch.nn.functional as F


class MetricOverlay:
    def __init__(self, metric, channels=None, mask=None, mask_cut=(0.99, 1),
                 binarize_target=False, activation=None, binary_volumes=False,
                 binarize_prediction=False, band_width=None, use_far_mask=False,
                 mixt_activation=0, additional_learned_param=None):
        self.metric = metric
        self.channels = channels
        self.mask = mask
        self.mask_cut = mask_cut
        self.binarize_target = binarize_target
        self.activation = activation
        self.binary_volumes = binary_volumes
        self.binarize_prediction = binarize_prediction

        if band_width is not None:
            assert band_width % 2 == 1

        self.band_width = band_width
        self.use_far_mask = use_far_mask
        self.mixt_activation = mixt_activation
        self.additional_learned_param = additional_learned_param

    @staticmethod
    def binarize(tensor):
        unlabeled_volume = tensor.sum(dim=1) < 0.5
        val, indices = torch.max(tensor, dim=1)
        extra_class = indices.max() + 1
        indices[val == 0] = extra_class
        indices[unlabeled_volume] = extra_class
        tensor = F.one_hot(indices)
        if (val == 0).sum() or unlabeled_volume.sum():
            tensor = tensor[..., :-1]
        tensor = tensor.permute(0, 4, 1, 2, 3).float()
        return tensor

    def get_band_width_kernel(self):
        dim = 3
        kernel_shape = [self.band_width for _ in range(dim)]
        kernel = torch.ones(*kernel_shape)
        distance_range = torch.arange(
            -(self.band_width // 2), self.band_width // 2 + 1
        )
        distance_grid = torch.meshgrid([distance_range for _ in range(dim)])
        distance_map = sum(
            [distance_grid[i].flatten() ** 2 for i in range(dim)]
        ).float().sqrt().reshape(*kernel_shape)
        kernel[distance_map > self.band_width // 2] = 0
        return kernel

    def get_outer_band_mask(self, tensor):
        channels = tensor.shape[1]
        kernel = self.get_band_width_kernel().to(tensor.device)
        band = torch.conv3d(
            (tensor >= self.mask_cut[0]).float(),
            kernel.expand(channels, 1, -1, -1, -1),
            padding=self.band_width // 2,
            groups=channels
        )
        mask = (band > 0).float() - (tensor >= self.mask_cut[0]).float()
        return mask

    def get_far_mask(self, tensor):
        far_mask = torch.ones_like(tensor)
        inner_mask = self.get_outer_band_mask(tensor)
        far_mask = far_mask - inner_mask - (tensor >= self.mask_cut[0]).float()
        return far_mask

    def __call__(self, prediction, target):
        if (prediction.shape[1] - target.shape[1]) == 1:
            target = torch.cat([target, 1 - target.sum(dim=1, keepdim=True)],
                               dim=1)

        if self.binary_volumes:
            prediction = F.one_hot(prediction[:, 0, ...].long()) \
                .permute(0, 4, 1, 2, 3).float()
            target = F.one_hot(target[:, 0, ...].long()) \
                .permute(0, 4, 1, 2, 3).float()

        if self.activation is not None:
            if self.mixt_activation:
                pred_seg = self.activation(prediction[:,:-self.mixt_activation,...])
                pred_reg = prediction[:,-self.mixt_activation:,...]
                prediction = [pred_seg, pred_reg]
                target_seg = target[:,:-self.mixt_activation,...]
                target_reg = target[:,-self.mixt_activation:,...]
                target = [target_seg, target_reg]
                # torch.cat( (self.activation(pred_with_act), pred_no_act), dim=1)
                #print(f'shape {pred_no_act.shape} and {pred_with_act.shape} and {prediction.shape}')
            else:
                prediction = self.activation(prediction)

        if self.binarize_target:
            target = self.binarize(target)

        if self.binarize_prediction:
            prediction = self.binarize(prediction)

        if self.band_width is not None:
            if self.use_far_mask:
                band_mask = self.get_far_mask(target)
            else:
                band_mask = self.get_outer_band_mask(target)
            prediction = prediction * band_mask
            target = target * band_mask

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


def weighted_mean_metric(prediction, target, metric):
    channels = list(range(target.shape[1]))
    res = 0.
    weights = [0.45, 0.45, 0.1]
    for channel in channels:
        res += weights[channel] * metric(prediction[:, channel, ...], target[:, channel, ...])
    return res / len(channels)
