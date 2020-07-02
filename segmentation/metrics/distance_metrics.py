import torch
from segmentation.metrics.utils import mean_metric


def _get_border(volume, cut=0.5, dim=3):
    ref = (volume > cut).float()
    border = torch.zeros_like(volume)
    spatial_shape = ref.shape[-dim:]

    for i in range(dim):
        shape = list(spatial_shape)
        shape[i] = 1
        zeros = torch.zeros(shape).to(volume.device)

        slices = [slice(1 * (i == j), spatial_shape[j] + 1 * (i == j)) for j in range(dim)]
        concat = torch.cat([ref, zeros], dim=i)[slices]
        border[(ref - concat) == 1] = 1

        slices = [slice(spatial_shape[j]) for j in range(dim)]
        concat = torch.cat([zeros, ref], dim=i)[slices]
        border[(ref - concat) == 1] = 1

    return border


class DistanceMetric:
    def __init__(self, cut=0.5, radius=5, mask_cut=0.99):
        self.cut = cut
        self.radius = radius
        self.mask_cut = mask_cut
        self.dim = 3
        self.d_max = torch.tensor(self.radius + 1.)
        self.distance_map = self._get_distance_map()
        self.distances = self.distance_map.unique()
        self.distance_kernels = self._get_distance_kernels()

    def _apply_mask(self, volume, mask):
        if mask is None:
            return volume
        mask = mask >= self.mask_cut
        return volume * mask

    def _get_distance_map(self):
        distance_range = torch.arange(-self.radius, self.radius + 1)
        distance_grid = torch.meshgrid([distance_range for _ in range(self.dim)])
        distance_map = sum([distance_grid[i].flatten() ** 2 for i in range(self.dim)]).float().sqrt()
        distance_map[distance_map > self.radius] = 0
        return distance_map

    def _get_distance_kernels(self):
        kernels = torch.zeros(len(self.distances), 1, *[2 * self.radius + 1 for _ in range(self.dim)])

        for idx in range(len(self.distances)):
            kernel = self.distance_map * (self.distance_map == self.distances[idx])
            kernels[idx] = kernel.reshape(1, *[2 * self.radius + 1 for _ in range(self.dim)])

        return kernels

    def _pairwise_distances(self, x, y):
        device = x.device

        # Compute distances to y points
        distances_to_y = torch.conv3d(
            y.float().expand(1, 1, -1, -1, -1),
            self.distance_kernels.to(device),
            padding=self.radius
        )[0]

        # Remove zero points from x
        relevant_distances = distances_to_y.permute(1, 2, 3, 0)[x.nonzero(as_tuple=True)]

        # Compute distances from convolution values
        all_distances = torch.zeros_like(relevant_distances)
        indices = relevant_distances.nonzero(as_tuple=True)
        all_distances[indices] = self.distances[indices[1]].to(device)
        all_distances[all_distances == 0] = self.d_max

        return all_distances

    def average_hausdorff_distance(self, prediction, target, mask=None):
        prediction = prediction > self.cut
        target = target > self.cut

        prediction = self._apply_mask(prediction, mask)

        prediction_mask = prediction.clone()
        prediction_mask[prediction * target] = 0
        target_mask = _get_border(target)

        if prediction_mask.sum():
            min_dist, _ = self._pairwise_distances(prediction_mask, target_mask).min(dim=1)
            first_term = min_dist.sum() / prediction.sum()
        else:
            first_term = 0.

        prediction_mask = _get_border(prediction)
        target_mask = target.clone()
        target_mask[prediction * target] = 0

        if target_mask.sum():
            min_dist, _ = self._pairwise_distances(target_mask, prediction_mask).min(dim=1)
            second_term = min_dist.sum() / target.sum()
        else:
            second_term = 0.

        return first_term + second_term

    def amount_of_far_points(self, prediction, target, mask=None):
        prediction = prediction > self.cut
        target = target > self.cut

        prediction = self._apply_mask(prediction, mask)

        prediction_mask = prediction.clone()
        prediction_mask[prediction * target] = 0
        target_mask = _get_border(target)

        if prediction_mask.sum():
            min_dist, _ = self._pairwise_distances(prediction_mask, target_mask).min(dim=1)
            return (min_dist >= self.radius).sum()
        else:
            return 0.

    def batch_average_hausdorff_distance(self, prediction, target, mask=None):
        res = 0.
        for p, t in zip(prediction, target):
            res += self.average_hausdorff_distance(p, t, mask)
        return res

    def batch_amount_of_far_points(self, prediction, target, mask=None):
        res = 0.
        for p, t in zip(prediction, target):
            res += self.amount_of_far_points(p, t, mask)
        return res

    def mean_average_hausdorff_distance(self, prediction, target, **kwargs):
        return mean_metric(prediction, target, self.batch_average_hausdorff_distance, **kwargs)

    def mean_amount_of_far_points(self, prediction, target, **kwargs):
        return mean_metric(prediction, target, self.batch_amount_of_far_points, **kwargs)
