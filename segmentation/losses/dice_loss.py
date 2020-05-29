from segmentation.utils import mean_metric


def dice_loss(prediction, target):
    smooth = 1.
    target = target.float()
    prediction = prediction.float()
    input_flat = prediction.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (input_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (input_flat.pow(2).sum() + target_flat.pow(2).sum() + smooth))


def mean_dice_loss(prediction, target):
    return mean_metric(prediction, target, dice_loss)

    return loss / len(channels)
