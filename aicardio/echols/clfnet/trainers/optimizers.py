import torch
from torch import nn
import torch.nn.functional as F

def mIoU(y_pred, masks):
    '''compute mean IoU loss approximately using prediction and target'''
    y_pred, y_true = y_pred.float(), torch.round(masks)
    intersection = (y_true * y_pred).sum((2, 3))
    union = y_true.sum((2, 3)) + y_pred.sum((2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def CriticmIoU(y_pred, masks):
    '''compute mean IoU loss approximately using prediction and target'''
    y_pred, y_true = y_pred.float(), masks.float()
    intersection = (y_true * y_pred).sum((2, 3))
    union = y_true.sum((2, 3)) + y_pred.sum((2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def IoU(y_pred, masks, threshold=0.5):
    '''compute mean IoU exaclty using prediction and target'''
    y_pred, y_true = (y_pred >= threshold).float(), torch.round(masks)
    intersection = (y_true * y_pred).sum((2, 3))
    union = y_true.sum((2, 3)) + y_pred.sum((2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def CriticIoU(y_pred, masks, threshold=0.5):
    '''compute mean IoU exaclty using prediction and target'''
    y_pred, y_true = (y_pred >= threshold).float(), (masks >= threshold).float()
    intersection = (y_true * y_pred).sum((2, 3))
    union = y_true.sum((2, 3)) + y_pred.sum((2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def DICE(y_pred, masks, threshold=0.5):
    '''compute mean DICE exactly using prediction and target'''
    y_pred, y_true = (y_pred >= threshold).float(), torch.round(masks)
    intersection = (y_true * y_pred).sum((2, 3))
    union = y_true.sum((2, 3)) + y_pred.sum((2, 3))
    dice = (2*intersection + 1e-6) / (union + 1e-6)
    return dice.mean()

def soft_IoU_loss(y_pred, y_true):
    '''compute IoU loss using prediction and target'''
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return -torch.log(iou)

def weighted_dice_loss(y_pred, y_true, positive_proportion=0.1):
    '''compute weighted DICE loss using prediction and target'''
    w_1 = 1/positive_proportion**2
    w_0 = 1/(1-positive_proportion)**2
    
    y_true_f_1 = y_true
    y_pred_f_1 = y_pred
    y_true_f_0 = 1 - y_true
    y_pred_f_0 = 1 - y_pred

    intersection_0 = w_0 * (y_true_f_0 * y_pred_f_0).sum((2, 3))
    intersection_1 = w_1 * (y_true_f_1 * y_pred_f_1).sum((2, 3))
    union_0 = w_0 * (y_true_f_0.sum((2, 3)) + y_pred_f_0.sum((2, 3)))
    union_1 = w_1 * (y_true_f_1.sum((2, 3)) + y_pred_f_1.sum((2, 3)))
    dice = 2 * (intersection_0 + intersection_1) / (union_0 + union_1)
    return - torch.log(dice).mean()

def hdloss_l2(ypred, ytrue, hd, alpha=2.0):
    '''compute L2 Haussdoff loss using prediction and target'''
    return (((ypred-ytrue)**2)*((1-hd)**alpha)).mean()

def hdloss(ypred, ytrue, hd, alpha=1.0):
    '''compute BCE Haussdoff loss using prediction and target'''
    bce = -(ytrue*torch.log(ypred+1e-6) + (1-ytrue)*torch.log(1-ypred+1e-6))
    return (bce*((1-hd)**alpha)).mean()

def weighted_binary_cross_entropy(y_pred, y_true, weights=None):
    '''compute weighted BCE loss using prediction and target'''
    if weights is not None:
        loss = weights[1] * (y_true * torch.log(y_pred + 1e-6)) + \
               weights[0] * ((1 - y_true) * torch.log(1 - y_pred + 1e-6))
    else:
        loss = y_true * torch.log(y_pred + 1e-6) + (1 - y_true) * torch.log(1 - y_pred + 1e-6)
    return torch.neg(torch.mean(loss))

def get_segmentation_optimizer(model, max_iter, base_lr=0.001, power=0.9):
    '''generate SGD optimizer and scheduler'''
    # used by: Deeplab, PSPNet, SegNet
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, 
                                momentum=0.9, weight_decay=0.0)
    lr_update = lambda iter: (1 - iter/max_iter)**power
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_update)
    return optimizer, scheduler

def get_captioning_optimizer(model, base_lr=5e-4):
    '''generate Adam optimizer and StepLR scheduler'''
    # used by: Neural Baby Talk, Grounded Video Captioning
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    return optimizer, scheduler

def get_quick_optimizer(model, max_iter, base_lr=0.001, power=0.9):
    '''generate Adam optimizer and StepLR scheduler'''
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    lr_update = lambda iter: (1 - iter/max_iter)**power
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_update)
    return optimizer, scheduler

def get_quick_optimizer_multi(models, max_iter, base_lr=0.001, power=0.9):
    '''generate Adam optimizer and StepLR scheduler'''
    optimizer = torch.optim.Adam(sum([list(m.parameters()) for m in models], []), lr=base_lr)
    lr_update = lambda iter: (1 - iter/max_iter)**power
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_update)
    return optimizer, scheduler

if __name__ == "__main__":
    print(help(mIoU))