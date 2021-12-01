from segmentation.metrics.utils import mean_metric
import torch.nn as nn
import torch
from segmentation.utils import to_numpy

class UniVarGaussianLogLkd(object):
    """
        thanks to benoit dufumier https://github.com/Duplums/bhb10k-dl-benchmark/blob/main/losses.py
        cf. Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics, Kendall, CVPR 18
        This loss can be used for classification (using soft cross-entropy, working with both hard or soft labels) and
        regression with a single variance for each voxel. The Dice loss is not used here since it is a global measure of
        similarity between 2 segmentation maps.
    """
    def __init__(self, pb: str="regression", apply_exp=True, lamb=None, fake=False,
                 sigma_prediction=1, sigma_constrain=None, return_loss_dict=False, **kwargs):
        """
        :param pb: "classif" or "regression"
        :param apply_exp: if True, assumes network output log(var**2)
        :param kwargs: kwargs given to PyTorch MSELoss if regression pb
        """

        self.apply_exp = apply_exp
        self.fake = fake
        self.sigma_prediction = sigma_prediction
        self.return_loss_dict = return_loss_dict
        self.sigma_constrain = sigma_constrain

        if pb == "classif":
            self.sup_loss = UniVarGaussianLogLkd.softXEntropy
            self.lamb = 0.5 if lamb is None else lamb
        elif pb == "BCE_logit":
            self.sup_loss = torch.nn.BCEWithLogitsLoss(reduction="mean", **kwargs)
            self.lamb = 0.5 if lamb is None else lamb
        elif pb == "regression":
            self.sup_loss = torch.nn.MSELoss(reduction="none", **kwargs)
            self.lamb = 1 if lamb is None else lamb
        else:
            raise ValueError("Unknown pb: %s"%pb)

    @staticmethod
    def LogitSoftXEntropy(input, target):
        logprob = torch.log(torch.nn.functional.sigmoid(input))
        return - (target * logprob).sum(dim=1)/target.sum(dim=1)


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
        #sigam2 is supposed to be the n last predicted output (n = self.sigma_prediction
        #sigma2 = x[:, -1, :] #withou (-1:) the dimension becomes batch,volume as classif loss
        if self.sigma_constrain == 'logsigmoid':
            sigma2 = torch.nn.functional.logsigmoid( x[:, -self.sigma_prediction:, ...] )
        elif self.sigma_constrain == "softplus":
            sigma2 = torch.nn.functional.softplus( x[:, -self.sigma_prediction:, ...] )
        else:
            sigma2 = x[:, -self.sigma_prediction:, ...]
        x = x[:, :-self.sigma_prediction, ...]

        if self.fake:
            if isinstance(self.sup_loss, torch.nn.MSELoss) :
                res_loss = self.sup_loss(x, target).sum(dim=1).mean()
            else:
                res_loss = self.sup_loss(x, target).mean()
            return res_loss

        #print(f'lamb is {self.lamb} shape is {x.shape}')
        if self.apply_exp:
            if self.sigma_constrain == "softplus":  #well do not apply ex
                sigma2 = sigma2.squeeze(dim=1) + 1e-6
                log_sigma2 = torch.log(sigma2)
            else:
                sigma2 = sigma2.squeeze(dim=1) #remove channel dim if only one sigma
                log_sigma2 = sigma2
                sigma2 = torch.exp(log_sigma2) + 1e-3

            if isinstance(self.sup_loss, torch.nn.MSELoss) :
                mse_loss = self.sup_loss(x, target).sum(dim=1)
                print(f'MSE/SIGMA min {mse_loss.min():.4f} | {sigma2.min():.4f} max {mse_loss.max():.4f} | {sigma2.max():.4f} mean {mse_loss.mean():.4f} |  {sigma2.mean():.4f}')
                res_loss = ( 1./sigma2.squeeze(dim=1) * mse_loss  +
                             self.lamb * log_sigma2.squeeze(dim=1)).mean()
            else:
                #if x.isnan().any():
                #    qsdf
                the_loss = self.sup_loss(x, target)
                res_loss = (1./sigma2 * the_loss + self.lamb * log_sigma2).mean()

        else:
            the_loss = self.sup_loss(x, target)
            print(
                f'BCE/SIGMA  max {the_loss.max():.4f} | {sigma2.max():.4f} mean {the_loss.mean():.4f} |  {sigma2.mean():.4f}')
            res_loss = (1./sigma2 * self.sup_loss(x, target) + self.lamb * torch.log(sigma2)).mean()

        if (self.return_loss_dict) : #& (target.shape[0]==1) : #only need in record  batch for single iteration
            #NOT required in the main train_loop for training (with batch >1)

            shape_loss = the_loss.shape
            lThnorm = torch.linalg.norm(
                the_loss.reshape([shape_loss[0], shape_loss[1] * shape_loss[2] * shape_loss[3]]), ord=2, dim=1).mean()
            lSnorm = torch.linalg.norm(sigma2.reshape([shape_loss[0], shape_loss[1] * shape_loss[2] * shape_loss[3]]),
                                       ord=2, dim=1).mean()
            dict_loss = {
                'loss_kll_norm': to_numpy( lThnorm ),
                'loss_sigma_norm': to_numpy(lSnorm),
                'loss_kll_mean':  to_numpy(the_loss.mean()),
                'loss_sigma_mean': to_numpy(sigma2.mean()),
                'loss_kll_max':   to_numpy(the_loss.max()),
                'loss_sigma_max':  to_numpy(sigma2.max())
            }
            return res_loss, dict_loss
        else:
            return res_loss


class MultiVarGaussianLogLkd(object):
    """
        thanks to benoit dufumier https://github.com/Duplums/bhb10k-dl-benchmark/blob/main/losses.py
        cf. Multivariate Uncertainty in Deep Learning, Russell, IEEE TNLS 21
    """

    def __init__(self, pb: str="regression", no_covar=False, **kwargs):
        """
        :param pb: "classif" or "regression"
        :param no_covar: If True, assume that the covariance matrix is diagonal
        :param kwargs: kwargs given to PyTorch Cross Entropy Loss
        """
        if pb == "classif":
            raise NotImplementedError()
        elif pb == "regression":
            pass
        else:
            raise ValueError("Unknown pb: %s"%pb)
        self.no_covar = no_covar

    #def __call__(self, x, Sigma, target):
    def __call__(self, x, target):
        """
        :param x: output segmentation, shape [*, C, *]
        :param Sigma: co-variance coefficients. It can be:
            (1) If no_covar==False, shape [*, C(C+1)/2, *] organized as row-first according to tril_indices
               from torch and numpy : [rho_11, rho_12, ..., rho_1C, rho_22, rho_23,...rho_2C,... rho_CC]
               with rho_ii = exp(.) > 0 encodes the variances and rho_ij = tanh(.) encodes the correlations.
               The covariance matrix is M is s.t  M[i][j] = rho_ij * srqrt(rho_ii) * sqrt(rho_ij)
            (2) If no_covar==True, shape [*, C, *], assuming that all non-diagonal coeff are zeros. We assume it
                has the form [sigma_1**2, sigma_2**2, ..., sigma_C**2]
        :param target: true segmentation, shape [*, C, *]
        :return: log-likelihood for logistic regression with uncertainty
        """

        if isinstance(x, list):     #should happen just for regression, where sigma_prediction is used in metric (utils)
            x, Sigma = x[0], x[1]
            log_Sigma = Sigma
            Sigma = torch.exp(log_Sigma) + 1e-6
            #if Sigma.min() < 1e-6:
            #    print(f'Warning min Sigma {Sigma.min()}')

        C, ndims = x.shape[1], x.ndim

        if self.no_covar:
            # Simplified Case
            assert C == Sigma.shape[1] and Sigma.ndim == ndims,\
                "Inconsistent shape for input data and covariance: {} vs {}".format(x.shape, Sigma.shape)
            assert torch.all(Sigma > 0), "Negative values found in Sigma"
            inv_Sigma = 1./Sigma # shape [*, C, *]
            #logdet_sigma = torch.log(torch.prod(Sigma, dim=1)) # shape [*, *]
            logdet_sigma = torch.sum(log_Sigma, dim=1) # shape [*, *]
            err = (target - x) # shape [*, C, *]
            return ((err * inv_Sigma * err).sum(dim=1) + logdet_sigma.squeeze()).mean()
        else:
            # General Case
            assert (C * (C+1))//2 == Sigma.shape[1] and Sigma.ndim == ndims, \
                "Inconsistent shape for input data and covariance: {} vs {}".format(x.shape, Sigma.shape)
            # permutes the 2nd dim to last, keeping other unchanged (in v1.9, eq. to torch.moveaxis(1, -1))
            swap_channel_last = (0,) + tuple(range(2,ndims)) + (1,)
            # First, re-arrange covar matrix to have shape [*, *, C, C]
            covar_shape = (Sigma.shape[0],) + Sigma.shape[2:] + (C, C)
            tril_ind = torch.tril_indices(row=C, col=C, offset=0)
            triu_ind = torch.triu_indices(row=C, col=C, offset=0)
            Sigma_ = torch.zeros(covar_shape, device=x.device)
            Sigma_[..., tril_ind[0], tril_ind[1]] = Sigma.permute(swap_channel_last)
            Sigma_[..., triu_ind[0], triu_ind[1]] = Sigma.permute(swap_channel_last)
            # Then compute determinant and inverse of covariance matrices
            logdet_sigma = torch.logdet(Sigma_) # shape [*, *]
            inv_sigma = torch.inverse(Sigma_) # shape [*, *, C, C]
            # Finally, compute log-likehood of multivariate gaussian distribution
            err = (target - x).permute(swap_channel_last).unsqueeze(-1) # shape [*, *, C, 1]
            return ((err.transpose(-1,-2) @ inv_sigma @ err).squeeze() + logdet_sigma.squeeze()).mean()


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

class Fake_metric:
    def __init__(self):
        pass
    def apply(self,x,y):
        return 1