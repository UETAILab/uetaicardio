import importlib
import argparse
import os
import glob
import math

from easydict import EasyDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from segmentation.trainers.train_single_frame import Trainer, init_weights
from segmentation.trainers.datasets import HDSingleFrameDataset, torch_hub_normalize
from segmentation.trainers.optimizers import (
    get_segmentation_optimizer, get_quick_optimizer,
    IoU, mIoU, soft_IoU_loss, hdloss,
    weighted_binary_cross_entropy,
)


class TrainerAuxLoss(Trainer):
    '''A Trainer for segmentation model with auxiliary layer'''
    def __init__(self, datasets, train_config):
        super().__init__(datasets, train_config)

    def get_loss(self, ypred, masks, batch):
        '''compute loss function = weighted BCE + soft IoU + Hausdorff distance loss'''
        if type(ypred) != tuple:
            y_pred = torch.sigmoid(ypred.type(torch.float))
            soft_iou_loss = soft_IoU_loss(y_pred, masks)
            wce_loss = weighted_binary_cross_entropy(y_pred, masks, weights=[0.1, 0.9])
            hd = batch['hd'].to(y_pred.device)
            hd_loss = hdloss(y_pred, masks, hd)
        else: # ypred is a tuple (aux3, aux2, out)
            aux_logits_3, aux_logits_2, logits = ypred
            y_pred = torch.sigmoid(logits.type(torch.float))
            aux_y_pred_2 = torch.sigmoid(aux_logits_2.type(torch.float))
            aux_y_pred_3 = torch.sigmoid(aux_logits_3.type(torch.float))
            
            soft_iou_loss = soft_IoU_loss(y_pred, masks) \
                          + soft_IoU_loss(aux_y_pred_2, masks) \
                          + soft_IoU_loss(aux_y_pred_3, masks)
            wce_loss = weighted_binary_cross_entropy(y_pred, masks, weights=[0.1, 0.9]) \
                     + weighted_binary_cross_entropy(aux_y_pred_2, masks, weights=[0.1, 0.9]) \
                     + weighted_binary_cross_entropy(aux_y_pred_3, masks, weights=[0.1, 0.9])
            hd = batch['hd'].to(y_pred.device)
            hd_loss = hdloss(y_pred, masks, hd)\
                    + hdloss(aux_y_pred_2, masks, hd)\
                    + hdloss(aux_y_pred_3, masks, hd)
            
        loss = wce_loss + soft_iou_loss + hd_loss
        return loss

    def get_iou(self, ypred, masks):
        '''compute iou in case ypred is a tuple or not'''
        if type(ypred) != tuple:
            y_pred = torch.sigmoid(ypred.type(torch.float))
        else:
            aux_logits_3, aux_logits_2, logits = ypred
            y_pred = torch.sigmoid(logits.type(torch.float))
        return IoU(y_pred, masks)

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_name", type=str, default="segmentation", help="module / package name")
    parser.add_argument("--model_class", type=str, required=True, help="model class name, eg., MobileNetv2_DeepLabv3")
    parser.add_argument("--trainset", type=str, required=True, help="train_dev dataset folder")
    parser.add_argument("--testset", type=str, required=True, help="test dataset folder")
    parser.add_argument("--log_dir", type=str, default="tmp/log", help="log folder")
    parser.add_argument("--model_prefix", type=str, default="tmp/m2d3", help="model output path prefix")
    parser.add_argument("--image_size", type=int, default=256, help="input tensor size")
    parser.add_argument("--epochs", type=int, default=1, help="number of train epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="train batch size")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--device", type=str, default="cuda:0", help="number of cpu threads to use during batch generation")
    parser.add_argument("--base_lr", type=float, default=0.001, help="train learning rate")
    parser.add_argument("--power", type=float, default=0.9, help="train scheduler power")
    parser.add_argument("--momentum", type=float, default=0.9, help="train momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="train weight decay factor")
    parser.add_argument("--init_weight", action="store_true", help="init with random weight (xavier)")
    parser.add_argument("--torchhub_process", action="store_true", help="normalize samples for torch.hub pretrained models")
    opt = parser.parse_args()
    print(opt)
    return opt

def get_model(opt):
    module = importlib.import_module(opt.module_name)
    model = getattr(module, opt.model_class)().to(opt.device)
    if opt.init_weight:
        print("init with random weight (xavier)")
        model.apply(init_weights)
    return model

def main():
    opt = get_options()
    model = get_model(opt)
    # set additional process function
    additional_process = torch_hub_normalize if opt.torchhub_process else None
    
    # load datasets
    data_config = EasyDict(dict(
        trainset=opt.trainset,
        testset=opt.testset,
        image_size=opt.image_size,
        additional_process=additional_process
    ))
    datasets = HDSingleFrameDataset.get_datasets(data_config)
    datasets.train.summary()
    datasets.test.summary()
    
    # set training configuration
    max_iter = opt.epochs * math.ceil(len(datasets.train) / opt.batch_size)
    optimizer, scheduler = get_quick_optimizer(model, max_iter, base_lr=opt.base_lr, power=opt.power)

    train_config = EasyDict(dict(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        n_cpu=opt.n_cpu,
        device=opt.device,
        log_dir=opt.log_dir,
        model_path=opt.model_prefix,
    ))

    # carry out training
    trainer = TrainerAuxLoss(datasets, train_config)
    trainer.train()

if __name__ == "__main__":
    # cd ~/pipeline
    # export PYTHONPATH=$(pwd)
    main()
