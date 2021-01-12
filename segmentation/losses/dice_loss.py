from segmentation.metrics.utils import mean_metric
import torch.nn as nn
import torch

class MultiTaskLoss(nn.Module):
    def __init__(self, tasks):
        super(MultiTaskLoss, self).__init__()
        self.tasks = nn.ModuleList(tasks)
        self.sigma = nn.Parameter(torch.ones(len(tasks)))
        self.mse = nn.MSELoss()

    def forward(self, x, targets):
       l = [self.mse(f(x), y) for y, f in zip(targets, self.tasks)]
       l = 0.5 * torch.Tensor(l) / self.sigma**2
       l = l.sum() + torch.log(self.sigma.prod())
       return l


class MultiTaskLossSegAndReg(nn.Module): #segmentation with dice and Regression with L1
    def __init__(self, task_num=2, init_weights = [1, 1]):
        super(MultiTaskLossSegAndReg, self).__init__()
        self.task_num = task_num
        #self.log_vars = nn.Parameter(torch.zeros((task_num), device='cuda'))
        # todo should be a parameter ... ?
        try:
            self.log_vars = nn.Parameter(torch.ones((task_num), device='cuda')*
                                         torch.tensor(init_weights, device='cuda'))
        except :
            self.log_vars = nn.Parameter(torch.ones((task_num), device='cpu') *
                                         torch.tensor(init_weights, device='cpu'))

    def forward(self, prediction, target):
        Diceloss = Dice()
        loss00 = Diceloss.mean_dice_loss(prediction[0], target[0])

        l1loss = nn.L1Loss()
        loss10 = l1loss(prediction[1], target[1])

        #this can lead to log_vars negativ and then a negativ los but also the other solution
        #precision0 = torch.exp(-self.log_vars[0])
        #loss0 = precision0 * loss00 + self.log_vars[0]
        #precision1 = torch.exp(-self.log_vars[1])
        #loss1 = precision1 * loss10 + self.log_vars[1]

        precision0 = 0.5 / self.log_vars[0]**2
        loss0 = precision0 * loss00 + torch.log(self.log_vars[0] )
        precision1 = 0.5 / self.log_vars[1]**2
        loss1 = precision1 * loss10 + torch.log(self.log_vars[1] )

        #return loss0 + loss1, loss00, precision0, loss10, precision1
        return loss0 + loss1, loss00, self.log_vars[0], loss10, self.log_vars[1]

class Dice:
    """
    Implements different variants of the Dice Loss.

    Args:
        cut: the threshold to binarize the volumes, default is None: volumes are
            not binarized.
        smooth: a value used to avoid division by zero in soft dice loss.
    """
    def __init__(self, cut=0.5, smooth=1., binary=False):
        self.cut = cut
        self.smooth = smooth
        self.binary = binary

    def dice_loss(self, prediction, target):
        target = target.float()
        prediction = prediction.float()
        input_flat = prediction.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (input_flat * target_flat).sum()
        return 1 - ((2. * intersection + self.smooth) /
                    (input_flat.pow(2).sum() + target_flat.pow(2).sum()
                     + self.smooth))

    def mean_dice_loss(self, prediction, target):
        return mean_metric(prediction, target, self.dice_loss)

    def mean_binarized_dice_loss(self, prediction, target):
        target = (target > self.cut).float()
        return self.mean_dice_loss(prediction, target)
