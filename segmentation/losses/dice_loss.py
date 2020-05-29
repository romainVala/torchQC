def dice_loss(prediction, target):
    smooth = 1.
    target = target.float()
    prediction = prediction.float()
    input_flat = prediction.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (input_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (input_flat.pow(2).sum() + target_flat.pow(2).sum() + smooth))


def mean_dice_loss(prediction, target):
    channels = list(range(target.shape[1]))
    loss = 0
    for channel in channels:
        dice = dice_loss(prediction[:, channel, ...], target[:, channel, ...])
        loss += dice

    return loss / len(channels)
