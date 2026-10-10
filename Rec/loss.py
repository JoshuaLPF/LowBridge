import pytorch_ssim
import torch.nn as nn
import torch
import torch.nn.functional as F

def L2Loss(model_name):
    return nn.MSELoss()


class MSE_and_SSIM_loss(nn.Module):
    def __init__(self, alpha=0.9):
        super(MSE_and_SSIM_loss, self).__init__()
        self.MSE = nn.MSELoss()
        self.SSIM = pytorch_ssim.SSIM()
        self.alpha = alpha

    def forward(self, img1, img2):
        loss = self.alpha*self.MSE(img1, img2) + (1 - self.alpha)*(1 - self.SSIM(img1, img2))
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
    def forward(self, predicted, target):
        return torch.mean(torch.abs(predicted - target))



def bce_iou_loss(pred, mask):
    size = pred.size()[2:]
    mask = F.interpolate(mask, size=size, mode='bilinear')
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def iou_loss(pred, mask):
    size = pred.size()[2:]
    mask = F.interpolate(mask, size=size, mode='bilinear')
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou.mean()


def bce2d_new(input, target, reduction=None):
    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg
    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    weights = alpha * pos + beta * neg
    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.sigmoid()
    target = target.float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice_score = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice_score.mean()
