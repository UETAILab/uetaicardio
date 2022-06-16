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

from classification import MultiframeClassifier
from classification.datasets.multiframe_dataset import MultiframeDataset
from classification.trainers import losses
from classification.trainers import optimizers
from classification.trainers.utils import Logger, bcolors, init_weights


class MultiframeTrainer:
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
        self.labels = ["2C", "3C", "4C", "none"]
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
            frames = batch["frames"].to(self.device)
            labels = batch["label"].to(self.device)

            if len(frames) < 2:
                continue
            y_pred = self.model(frames)
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
            frames = batch["frames"].to(self.device)
            labels = batch["label"].to(self.device)
            with torch.no_grad():
                y_pred = self.model(frames)
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
        loss = losses.focal_loss(y_pred, labels)
        return loss

    def __optimizer_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def __compute_metrics(self, y_pred, labels):
        y_pred = torch.argmax(y_pred, dim=1)
        is_correct = y_pred == labels
        n_corrects = float(torch.sum(is_correct).item())
        n_total = float(y_pred.numel())

        tps, tns, fps, fns = [], [], [], []
        for _class in range(4):
            tp = ((labels == _class) * (y_pred == _class)).float().detach().cpu().numpy()
            tn = ((labels != _class) * (y_pred != _class)).float().detach().cpu().numpy()
            fp = ((labels == _class) * (y_pred != _class)).float().detach().cpu().numpy()
            fn = ((labels != _class) * (y_pred == _class)).float().detach().cpu().numpy()
            tps.append(np.sum(tp))
            tns.append(np.sum(tn))
            fps.append(np.sum(fp))
            fns.append(np.sum(fn))
        return EasyDict(dict(n_corrects=n_corrects, n_total=n_total,
                             tps=tps, tns=tns, fps=fps, fns=fns))

    def __update_avg_meter(self, metric_values):
        avg_meter = {}
        n_corrects = np.sum([x["n_corrects"] for x in metric_values])
        n_total = np.sum([x["n_total"] for x in metric_values])
        tps = np.sum([x["tps"] for x in metric_values], axis=0)
        tns = np.sum([x["tns"] for x in metric_values], axis=0)
        fps = np.sum([x["fps"] for x in metric_values], axis=0)
        fns = np.sum([x["fns"] for x in metric_values], axis=0)

        avg_meter["acc"] = n_corrects / n_total
        avg_meter.update({
            f"precision_{label}": tp / (tp + fp + 1e-7)
            for tp, fp, label in zip(tps, fps, self.labels)
        })
        avg_meter.update({
            f"recall_{label}": tp / (tp + fn + 1e-7)
            for tp, fn, label in zip(tps, fns, self.labels)
        })
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


def preprocess(frames):
    frames = [cv2.resize(frame, (256, 256))/255.0
              for frame in frames]
    return frames


if __name__ == "__main__":
    r"""
    model, optimizer, scheduler, epochs, batch_size, n_cpu, device, log_dir, model_path
    """
    args = EasyDict(dict(
        feature_extractor_weights="/data.local/data/models/chamber_classification/mobilenetv2_0049_0.9507_best.pth",
        epochs=100,
        batch_size=16,
        n_cpu=16,
        device="cuda:0",
        log_dir="log",
        model_prefix="/data.local/giangh/pipeline/classification/ckpt/multiframe_mobilenetv2_no_softmax"
    ))
    model = MultiframeClassifier(1280, 4, args.feature_extractor_weights).to(args.device)
    optimizer, scheduler = optimizers.get_quick_optimizer(model, 1270*args.epochs, 0.001)
    

    datasets = EasyDict(
        train=MultiframeDataset(EasyDict(
            root="/data.local/giangh/pipeline/data/classification/multi_frame/train",
            window_size=40, augment=None, preprocess=preprocess, is_training=True
        )),
        test=MultiframeDataset(EasyDict(
            root="/data.local/giangh/pipeline/data/classification/multi_frame/test",
            window_size=40, augment=None, preprocess=preprocess, is_training=False
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
    trainer = MultiframeTrainer(datasets, train_config)
    trainer.train()
