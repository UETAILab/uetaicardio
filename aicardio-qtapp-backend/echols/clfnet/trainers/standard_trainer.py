#!/data.local/giangh/envs/pipeline/bin/python
import os
import tqdm
from easydict import EasyDict
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

from classification.datasets.standard_dataset import StandardDataset
from classification.trainers import optimizers
from classification.trainers.utils import Logger, bcolors, init_weights


class StandardTrainer:
    def __init__(self, datasets, config):
        r"""Initialize the Trainer with config

        config
            .model (nn.Module): Model to train
            .optimizer (optim.Optimizer): Optimizer
            .scheduler (optim.lr_scheduler._LRScheduler): Scheduler
            .epochs (int): Number of epochs
            .batch_size (int): Batch size for training
            .n_cpu (int): Number of data workers
            .device (str): Device for training
            .log_dir (str): Tensorboard log dir
            .model_path (str): Model output prefix
        """
        self.config = config
        self.datasets = datasets
        self.device = self.config.device
        print("trainset", self.datasets.train.config)
        print("testset", self.datasets.test.config)

    def train(self):
        self.model = self.config.model
        self.optimizer = self.config.optimizer
        self.scheduler = self.config.scheduler
        self.logger = Logger(self.config.log_dir)

        self.best_criteria, self.best_epoch = float("-inf"), -1
        self.last_criteria, self.last_epoch = float("-inf"), -1
        for epoch in range(self.config.epochs):
            self.train_eval = self.__train_one_epoch(epoch)
            self.test_eval = self.__test_one_epoch(epoch)
            self.__save_model(epoch)

    def __train_one_epoch(self, epoch):
        self.model.train()
        loader = DataLoader(
            self.datasets.train, batch_size=self.config.batch_size,
            shuffle=True, num_workers=self.config.n_cpu
        )
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader))

        metrics = []
        for step, batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            if len(images) < 2:
                continue
            y_pred = self.model(images)
            loss = self.__compute_loss(y_pred, labels)
            self.__optimizer_step(loss)

            metrics.append(self.__compute_metrics(y_pred, labels))
            avg_meter = self.__update_avg_meter(metrics)
            desc = f"train epoch {epoch} step {step} " \
                 + " ".join([f"{key} {avg_meter[key]:.4f}" for key in avg_meter])
            pbar.set_description(desc)
        self.logger.list_of_scalars_summary(
            [(key, avg_meter[key]) for key in avg_meter], epoch
        )
        return avg_meter
    
    def __test_one_epoch(self, epoch):
        self.model.eval()
        loader = DataLoader(self.datasets.test, 
                            batch_size=self.config.batch_size,
                            shuffle=True, 
                            num_workers=self.config.n_cpu)
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader))

        metrics = []
        for step, batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            with torch.no_grad():
                y_pred = self.model(images)
                loss = self.__compute_loss(y_pred, labels)
                metrics.append(self.__compute_metrics(y_pred, labels))
            avg_meter = self.__update_avg_meter(metrics)
            desc = f"test epoch {epoch} step {step} " \
                 + " ".join([f"{key} {avg_meter[key]:.4f}" for key in avg_meter])
            pbar.set_description(desc)
        self.logger.list_of_scalars_summary(
            [(key, avg_meter[key]) for key in avg_meter], epoch
        )
        return avg_meter

    def __compute_loss(self, y_pred, labels):
        loss = F.cross_entropy(y_pred, labels)
        return loss

    def __optimizer_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def __compute_metrics(self, y_pred, labels):
        y_pred = torch.argmax(y_pred, dim=1)
        n_corrects = float(torch.sum(y_pred == labels).item())
        n_total = float(y_pred.numel())
        return EasyDict(dict(n_corrects=n_corrects, n_total=n_total))

    def __update_avg_meter(self, metric_values):
        avg_meter = {}
        n_corrects = np.sum([x["n_corrects"] for x in metric_values])
        n_total = np.sum([x["n_total"] for x in metric_values])
        avg_meter["acc"] = n_corrects / n_total
        return EasyDict(avg_meter)

    def __save_model(self, epoch):                                              
        '''save a checkpoint of model, check for best validation and delete old files'''
        # the data to saved                                                     
        save_obj = dict(                                                        
            model=self.model.state_dict(),                                      
            epoch=epoch,                                                        
            val_acc=self.test_eval.acc, 
            train_acc=self.train_eval.acc,
        )
        # check if there is improvement and save the best one                   
        val_acc = self.test_eval.acc
        if self.best_criteria < val_acc:                                        
            path = self.__get_model_path(epoch, val_acc, best=True)             
            print(bcolors.log(f"epoch {epoch} improved from {self.best_criteria:.4f} to {val_acc:.4f} file {path}"))
            torch.save(save_obj, path)                                          
            if self.best_epoch >= 0:
                path = self.__get_model_path(
                    self.best_epoch, self.best_criteria, best=True
                )
                os.remove(path)                                                 
            
            self.best_criteria = val_acc
            self.best_epoch = epoch                                             
        # save the current model                                                
        path = self.__get_model_path(epoch, val_acc)
        torch.save(save_obj, path)                                              
        if self.last_epoch >= 0:                                                
            path = self.__get_model_path(self.last_epoch, self.last_criteria)   
            os.remove(path)                                                     
        self.last_criteria = val_acc
        self.last_epoch = epoch 

    def __get_model_path(self, epoch, val_iou, best=False):
        '''return model path with epoch number and validation iou'''
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        return os.path.join(f"{self.config.model_path}_{epoch:04d}_{val_iou:.4f}" \
            + ("_best" if best else "") + ".pth")


if __name__ == "__main__":
    r"""
    model, optimizer, scheduler, epochs, batch_size, n_cpu, device, log_dir, model_path
    """
    args = EasyDict(dict(
        epochs=100,
        batch_size=50,
        n_cpu=16,
        device="cuda:0",
        log_dir="log",
        model_prefix="/data.local/giangh/pipeline/classification/ckpt/mobilenetv2"
    ))
    model = models.mobilenet_v2(num_classes=4).to(args.device)
    optimizer, scheduler = optimizers.get_quick_optimizer(model, 1270*args.epochs, 0.001)

    datasets = EasyDict(
        train=StandardDataset(EasyDict(
            root="/data.local/giangh/pipeline/data/classification/train",
            augment=None, preprocess=lambda x: cv2.resize(x, (256, 256))/255.0
        )),
        test=StandardDataset(EasyDict(
            root="/data.local/giangh/pipeline/data/classification/test",
            augment=None, preprocess=lambda x: cv2.resize(x, (256, 256))/255.0
        ))
    )
    train_config = EasyDict(dict(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_cpu=args.n_cpu,
        device=args.device,
        log_dir=args.log_dir,
        model_path=args.model_prefix
    ))
    trainer = StandardTrainer(datasets, train_config)
    trainer.train()
