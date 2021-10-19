from segmentation.metrics.utils import mean_metric
import torch.nn as nn
import torch

class UniVarGaussianLogLkd(object):
    """
        thanks to benoit dufumier https://github.com/Duplums/bhb10k-dl-benchmark/blob/main/losses.py
        cf. Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics, Kendall, CVPR 18
        This loss can be used for classification (using soft cross-entropy, working with both hard or soft labels) and
        regression with a single variance for each voxel. The Dice loss is not used here since it is a global measure of
        similarity between 2 segmentation maps.
    """
    def __init__(self, pb: str="regression", apply_exp=True, lamb=None, **kwargs):
        """
        :param pb: "classif" or "regression"
        :param apply_exp: if True, assumes network output log(var**2)
        :param kwargs: kwargs given to PyTorch MSELoss if regression pb
        """

        self.apply_exp = apply_exp
        if pb == "classif":
            self.sup_loss = UniVarGaussianLogLkd.softXEntropy
            self.lamb = 0.5 if lamb is None else lamb
        elif pb == "regression":
            self.sup_loss = torch.nn.MSELoss(reduction="mean", **kwargs)
            self.lamb = 1 if lamb is None else lamb
        else:
            raise ValueError("Unknown pb: %s"%pb)

    @staticmethod
    def softXEntropy(input, target):
        """
        :param input: output segmentation, shape [*, C, *]
        :param target: target soft classes, shape [*, C, *]
        :return: Xentropy, shape [*, *]
        """
        logprob = torch.nn.functional.log_softmax(input, dim=1)
        return - (target * logprob).sum(dim=1)/target.sum(dim=1)

    #def __call__(self, x, sigma2, target):
    def __call__(self, x, target):
        """
        rrr change to 2 input argument to avoid changing the generic call of the loss (loss(prediction,target)
        :param x: output segmentation, shape [*, C, *]
        :param sigma2 == sigma**2 or log(sigma**2) if apply_exp is set: variance map, shape [*, *]
        :param target: true segmentation, assuming that soft-labels are available, shape [*, C, *]
        :return: log-likelihood for logistic regression (classif)/ridge regression (regression) with uncertainty
        """
        #sigam2 is supposed to be the last predicted output
        sigma2 = x[:,-1,:]
        x = x[:,:-1,::]
        #print(f'lamb is {self.lamb} shape is {x.shape}')
        if self.apply_exp:
            log_sigma2 = sigma2
            sigma2 = torch.exp(log_sigma2)
            return (1./sigma2 * self.sup_loss(x, target) + self.lamb * log_sigma2).mean()
        else:
            return (1./sigma2 * self.sup_loss(x, target) + self.lamb * torch.log(sigma2)).mean()

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
    def __init__(self, cut=0.5, smooth=1.):
        self.cut = cut
        self.smooth = smooth

    def dice_loss(self, predictions, targets):
        #warning, flating all predictions and target and computing one dice score, is not equivalent to
        # averaging dice scores obtain on each volume ...
        nb_vol = predictions.shape[0] #btach dimension
        res = 0
        for num_vol in range(0, nb_vol ):
            prediction, target = predictions[num_vol], targets[num_vol]
            target = target.float()
            prediction = prediction.float()
            input_flat = prediction.contiguous().view(-1)
            target_flat = target.contiguous().view(-1)
            intersection = (input_flat * target_flat).sum()
            one_vol_dice =  1 - ((2. * intersection + self.smooth) /
                        (input_flat.pow(2).sum() + target_flat.pow(2).sum()
                         + self.smooth))
            res += one_vol_dice
        res = res / nb_vol
        return res

    def mean_dice_loss(self, prediction, target):
        return mean_metric(prediction, target, self.dice_loss)

    def mean_binarized_dice_loss(self, prediction, target):
        target = (target > self.cut).float()
        return self.mean_dice_loss(prediction, target)

    def generalized_dice_loss(self, prediction, target):
        """
        Adapted from Sudre et al. "Generalised Dice overlap as a deep learning loss
        function for highly unbalanced segmentations"
        """
        reference_seg = target.float()
        proba_map = prediction.float()
        smooth_num = 1e-5
        smooth_den = 1e-5
        weight = 1 / (torch.sum(reference_seg, dim=[0, 2, 3, 4]) ** 2)
        intersection = torch.sum(torch.mul(reference_seg, proba_map), dim=[0, 2, 3, 4])
        sum = torch.sum(torch.add(reference_seg, proba_map), dim=[0, 2, 3, 4])
        numerator = torch.sum(torch.mul(weight, intersection))
        denominator = torch.sum(torch.mul(weight, sum))
        gdl = 1 - (2 * numerator + smooth_num) / (denominator + smooth_den)
        return gdl

    def identity_loss(self, x, y):
        return 1
