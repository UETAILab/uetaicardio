import torch
import torch.nn.functional as F


def focal_loss(y_pred, labels, gamma=2.0, class_weights=[1.22996795, 1.49852976, 1.21532547, 0.05617683]):
    y_pred = F.softmax(y_pred, dim=1)
    focal_loss_terms = -(1 - y_pred)**gamma * torch.log(y_pred)
    loss = 0.0 
    for i in range(y_pred.shape[1]):
        loss += class_weights[i] * torch.sum(focal_loss_terms[:, i] * (labels == i))
    return loss
