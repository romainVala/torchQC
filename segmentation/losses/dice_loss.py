from segmentation.metrics.utils import mean_metric


class Dice:
    """
    Implements different variants of the Dice Loss.

    Args:
        cut: the threshold to binarize the volumes, default is None: volumes are not binarized.
        smooth: a value used to avoid division by zero in soft dice loss.
    """
    def __init__(self, cut=0.5, smooth=1., mask_cut=0.99, binary=False):
        self.cut = cut
        self.smooth = smooth
        self.mask_cut = mask_cut
        self.binary = binary

    def _apply_mask(self, prediction, target, mask):
        if mask is None:
            return prediction, target
        mask = mask >= self.mask_cut
        return prediction * mask, target * mask

    def dice_loss(self, prediction, target, mask=None):
        target = target.float()
        prediction = prediction.float()
        prediction, target = self._apply_mask(prediction, target, mask)
        input_flat = prediction.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (input_flat * target_flat).sum()
        return 1 - ((2. * intersection + self.smooth) /
                    (input_flat.pow(2).sum() + target_flat.pow(2).sum() + self.smooth))

    def mean_dice_loss(self, prediction, target, **kwargs):
        return mean_metric(prediction, target, self.dice_loss, binary=self.binary, **kwargs)

    def mean_binarized_dice_loss(self, prediction, target, **kwargs):
        target = (target > self.cut).float()
        return self.mean_dice_loss(prediction, target, **kwargs)
