import importlib
import argparse
import os
import glob
import math

import torch
from easydict import EasyDict
import numpy as np
import tqdm

from torch.utils.data import DataLoader
from segmentation.trainers.optimizers import get_segmentation_optimizer, IoU, mIoU
from segmentation.trainers.datasets import SingleFrameDataset
from segmentation.trainers.utils import Logger, bcolors, init_weights


class Trainer:
    '''A trainer for segmentation model'''
    def __init__(self, datasets, train_config):
        '''initialize the Trainer with config
        
        train_config
          .model      = the model to train,
          .optimizer  = the optimizer,
          .scheduler  = the scheduler,
          .epochs     = number of epoch,
          .batch_size = train batch size,
          .n_cpu      = number of data workers,
          .device     = compute device (cpu, cuda, cuda:0, cuda:1),
          .log_dir    = tensorboard log dir,
          .model_path = model output prefix,
        '''
        
        self.train_config = train_config
        self.datasets = datasets
        self.device = self.train_config.device
        print("trainset", self.datasets.train.config)
        print("testset ", self.datasets.test.config)

    def train(self):
        '''carry out training for a number of epochs, save checkpoints'''
        self.model = self.train_config.model
        self.optimizer = self.train_config.optimizer
        self.scheduler = self.train_config.scheduler
        self.logger = Logger(self.train_config.log_dir)

        self.best_criteria, self.best_epoch = float('-inf'), -1
        self.last_criteria, self.last_epoch = float('-inf'), -1
        for epoch in range(self.train_config.epochs):
            self.train_eval = self.__train_one_epoch(epoch)
            self.test_eval = self.__test_one_epoch(epoch)
            
            self.__save_model(epoch)

    def get_loss(self, ypred, ytrue, batch):
        '''compute loss tensor from prediction and target and additional batch information, to be override by other Trainer'''
        loss = torch.nn.BCEWithLogitsLoss()
        loss = loss(ypred, ytrue) - torch.log( mIoU(torch.sigmoid(ypred), ytrue) )
        return loss

    def get_iou(self, ypred, ytrue):
        '''compute iou metric from prediction and target, to be override by other Trainer'''
        return IoU(torch.sigmoid(ypred), ytrue)
    
    def __save_model(self, epoch):
        '''save a checkpoint of model, check for best validation and delete old files'''
        # the data to saved
        save_obj = dict(
            model=self.model.state_dict(),
            epoch=epoch,
            val_loss=self.test_eval.loss,
            val_iou=self.test_eval.iou,
            train_loss=self.train_eval.loss,
            train_iou=self.train_eval.iou,
        )
        # check if there is improvement and save the best one
        val_iou = self.test_eval.iou
        if self.best_criteria < val_iou:
            path = self.__get_model_path(epoch, val_iou, best=True)
            print(bcolors.log(f"epoch {epoch} improved from {self.best_criteria:.4f} to {val_iou:.4f} file {path}"))
            torch.save(save_obj, path)
            if self.best_epoch >= 0:
                path = self.__get_model_path(self.best_epoch, self.best_criteria, best=True)
                os.remove(path)

            self.best_criteria = val_iou
            self.best_epoch = epoch
        # save the current model
        path = self.__get_model_path(epoch, val_iou)
        torch.save(save_obj, path)
        if self.last_epoch >= 0:
            path = self.__get_model_path(self.last_epoch, self.last_criteria)
            os.remove(path)
        self.last_criteria = val_iou
        self.last_epoch = epoch
        
    def __get_model_path(self, epoch, val_iou, best=False):
        '''return model path with epoch number and validation iou'''
        return os.path.join(f"{self.train_config.model_path}_{epoch:04d}_{val_iou:.4f}" + ("_best" if best else "") + ".pth")

    def __optimizer_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
    def __get_metric(self, loss, iou):
        return EasyDict(dict(loss=loss.item(), iou=iou.item()))
    
    def __train_one_epoch(self, epoch):
        '''train the model for one epoch with progress bar'''
        self.model.train()
        loader = DataLoader(self.datasets.train, batch_size=self.train_config.batch_size, shuffle=True, num_workers=self.train_config.n_cpu)
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
        metrics = []
        for step, batch in pbar:
            img, ytrue = batch['img'].to(self.device), batch['msk'].to(self.device)
            
            if len(img) < 2: # avoid BN error when len(img) = 1
                continue
            
            ypred = self.model(img)
            loss = self.get_loss(ypred, ytrue, batch)
            self.__optimizer_step(loss)

            iou = self.get_iou(ypred, ytrue)
            metrics.append(self.__get_metric(loss, iou))

            evaluation = self.__get_mean(metrics)
            pbar.set_description(f"train epoch {epoch} step {step} loss {evaluation.loss:.4f} iou {evaluation.iou:.4f}")
        
        self.logger.list_of_scalars_summary([ ("train_loss", evaluation.loss), ("train_iou", evaluation.iou) ], epoch)
        return evaluation

    def __test_one_epoch(self, epoch):
        '''test the model for one epoch with progress bar'''
        self.model.eval()
        loader = DataLoader(self.datasets.test, batch_size=self.train_config.batch_size, shuffle=False, num_workers=self.train_config.n_cpu)
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
        metrics = []
        for step, batch in pbar:
            img, ytrue = batch['img'].to(self.device), batch['msk'].to(self.device)

            with torch.no_grad():
                ypred = self.model(img)
                loss = self.get_loss(ypred, ytrue, batch)
                iou = self.get_iou(ypred, ytrue)
                metrics.append(self.__get_metric(loss, iou))

            evaluation = self.__get_mean(metrics)
            pbar.set_description(f"test  epoch {epoch} step {step} loss {evaluation.loss:.4f} iou {evaluation.iou:.4f}")
            # break

        self.logger.list_of_scalars_summary([ ("val_loss", evaluation.loss), ("val_iou", evaluation.iou) ], epoch)
        return evaluation

    def __get_mean(self, metrics):
        '''compute the average of metrics in a list of EasyDict(...) '''
        return EasyDict({key : np.mean([x[key] for x in metrics]) for key in ['loss', 'iou']})

if __name__ == "__main__":
    # cd ~/pipeline
    # export PYTHONPATH=$(pwd)
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
    opt = parser.parse_args()
    print(opt)

    module_name = opt.module_name
    model_class = opt.model_class

    module = importlib.import_module(module_name)
    model = getattr(module, model_class)().to(opt.device)
    model.apply(init_weights)
    

    data_config = EasyDict(dict(
        trainset=opt.trainset,
        testset=opt.testset,
        image_size=opt.image_size
    ))
    datasets = SingleFrameDataset.get_datasets(data_config)
    datasets.train.summary()
    datasets.test.summary()
    
#     optimizer = torch.optim.SGD(
#         model.parameters(), nesterov=True,
#         lr=opt.base_lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    max_iter = opt.epochs * math.ceil(len(datasets.train) / opt.batch_size)
    optimizer, scheduler = get_segmentation_optimizer(model, max_iter, base_lr=opt.base_lr, power=opt.power)

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
    trainer = Trainer(datasets, train_config)
    trainer.train()

