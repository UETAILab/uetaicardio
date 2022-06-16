import importlib
import argparse
import os
import glob
import math

import torch
from easydict import EasyDict
import numpy as np
import cv2
import tqdm
#import tensorflow as tf

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from segmentation.trainers.optimizers import get_quick_optimizer, get_quick_optimizer_multi, IoU, mIoU, CriticIoU, CriticmIoU
from segmentation.trainers.actor_critic_dataset import ActorCriticDataset
from segmentation.trainers.datasets import HDSingleFrameDataset
from segmentation.trainers.utils import Logger, init_weights, bcolors


class CriticTrainer:
    '''A trainer for segmentation model'''
    def __init__(self, datasets, actor_datasets, train_config):
        '''initialize the Trainer with config
        
        train_config
          .actor_model        = the actor model to use,
          .critic_model       = the critic model to train
          .optimizer          = the optimizer,
          .scheduler          = the scheduler,
          .critic_optimizer   = optimizer for critic model,
          .critic_scheduler   = scheduler for critic model,
          .epochs             = number of epoch,
          .batch_size         = train batch size,
          .n_cpu              = number of data workers,
          .device             = compute device (cpu, cuda, cuda:0, cuda:1),
          .log_dir            = tensorboard log dir,
          .model_path         = model output prefix,
          .n_future_neighbors = number of future frames in a critic batch
          .critic_epochs      = cycle (number of epochs) for a critic pass
        '''
        
        self.train_config = train_config
        assert self.train_config.n_future_neighbors % 2 == 0
        self.datasets = datasets
        self.actor_datasets = actor_datasets
        self.device = self.train_config.device
        print("trainset", self.datasets.train.config)
        print("testset ", self.datasets.test.config)

    def train(self):
        '''carry out training for a number of epochs, save checkpoints'''
        self.actor_model = self.train_config.actor_model
        self.critic_model = self.train_config.critic_model
        self.optimizer = self.train_config.optimizer
        self.scheduler = self.train_config.scheduler
        self.critic_optimizer = self.train_config.critic_optimizer
        self.critic_scheduler = self.train_config.critic_scheduler
        self.logger = Logger(self.train_config.log_dir)

        self.best_criteria, self.best_epoch = float('-inf'), -1
        self.last_criteria, self.last_epoch = float('-inf'), -1
        for epoch in range(self.train_config.epochs):
            self.train_eval = self.__train_one_epoch(epoch)
            self.test_eval = self.__test_one_epoch(epoch)
            
            self.__save_model(epoch)

    def get_critic_loss(self, ypred, ytrue, batch):
        '''compute loss tensor from prediction and target and additional batch information, to be override by other Trainer'''
        loss = torch.nn.BCEWithLogitsLoss()
        loss = loss(ypred, ytrue) - torch.log( CriticmIoU(torch.sigmoid(ypred), ytrue) )
        return loss

    def get_actor_loss(self, ypred, ytrue, batch):
        '''compute loss tensor from prediction and target and additional batch information, to be override by other Trainer'''
        loss = torch.nn.BCEWithLogitsLoss()
        loss = loss(ypred, ytrue) - torch.log( CriticmIoU(torch.sigmoid(ypred), ytrue) )
        return loss

    def get_iou(self, ypred, ytrue):
        '''compute iou metric from prediction and target, to be override by other Trainer'''
        return CriticIoU(torch.sigmoid(ypred), ytrue)
    
    def __save_model(self, epoch):
        '''save a checkpoint of model, check for best validation and delete old files'''
        # the data to saved
        save_obj = dict(
            model=self.actor_model.state_dict(),
            critic_model=self.critic_model.state_dict(),
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

    def __critic_optimizer_step(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        self.critic_scheduler.step()
        
    def __actor_optimizer_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
    def __get_metric(self, loss, iou):
        return EasyDict(dict(loss=loss.item(), iou=iou.item()))
    
    def __train_one_epoch(self, epoch): ################### MODIFY LATER
        '''train the model for one epoch with progress bar'''
        critic_training = (epoch % self.train_config.critic_epochs == 0) and epoch > 0
        self.actor_model.train()
        self.critic_model.train() if critic_training else self.critic_model.eval()
        loader = DataLoader(self.datasets.train, batch_size=self.train_config.batch_size, shuffle=True, num_workers=self.train_config.n_cpu, drop_last=True)
        actor_loader = iter(DataLoader(self.actor_datasets.train, batch_size=self.train_config.batch_size, shuffle=True, num_workers=self.train_config.n_cpu, drop_last=True))
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
        metrics = []
        actor_metrics = []
        for step, batch in pbar:
            batch_imgs = batch["imgs"].to(self.device)
            batch_msks = batch["msks"].to(self.device)
            batch_msk_availability = batch["msk_availability"].to(self.device)
            
            ### compute critic loss
            if critic_training:
                # run actor model
#                 batch_pred_msks = [torch.sigmoid(self.actor_model(imgs[[0,self.train_config.n_future_neighbors//2,-1]])[-1]) for imgs in batch_imgs]
#                 batch_middle_pred_msks = torch.cat([pred_msks[1][None, ...] for pred_msks in batch_pred_msks], axis=0)
                with torch.no_grad():
                    batch_pred_msks = [torch.sigmoid(self.actor_model(imgs[[0, -1]])[-1]) for imgs in batch_imgs]
                batch_middle_pred_msks = torch.cat([torch.sigmoid(self.actor_model(imgs[[self.train_config.n_future_neighbors//2]])[-1]) for imgs in batch_imgs], dim=0)
                
                # run critic model
                batch_left_neighbor_pred_msks = torch.cat([pred_msks[:1, 0, ...][None, ...] for pred_msks in batch_pred_msks], axis=0)
                batch_right_neighbor_pred_msks = torch.cat([pred_msks[-1:, 0, ...][None, ...] for pred_msks in batch_pred_msks], axis=0)
                batch_neighbor_pred_msks = torch.cat([batch_left_neighbor_pred_msks, batch_right_neighbor_pred_msks], axis=1)
                critic_outputs = self.critic_model(batch_neighbor_pred_msks)
                
                ypred = critic_outputs
                ytrue = batch_middle_pred_msks
                loss = self.get_critic_loss(ypred, ytrue, batch)
                iou = self.get_iou(ypred, ytrue)
                metrics.append(self.__get_metric(loss, iou))
                self.__critic_optimizer_step(loss)

                if step % 10 == 0:
                    ypred = torch.sigmoid(ypred)
                    for i, (critic_pred, actor_pred) in enumerate(zip(ypred, batch_pred_msks)):
                        critic_pred = critic_pred.detach().cpu().numpy().squeeze(0)
                        critic_pred = np.repeat(np.uint8(critic_pred[..., None] * 255), 3, axis=-1)
                        critic_pred[..., :2] = 0 # RED for critic

                        actor_pred = actor_pred.detach().cpu().numpy().squeeze(1)
                        actor_pred = np.repeat(np.uint8(actor_pred[..., None] * 255), 3, axis=-1)
                        actor_pred[..., 1:] = 0  # BLUE for actor, PURPLE for intersection
                        actor_pred = list(actor_pred)

                        actor_pred[1] = cv2.addWeighted(actor_pred[1], 0.5, critic_pred, 0.5, 0)
                        actor_pred = np.concatenate(actor_pred, axis=1)
                        cv2.imwrite(f"tmp/test-{i}.jpg", actor_pred)

            ### compute actor loss
            actor_batch = next(actor_loader, None)
            while actor_batch is None: # check if actor dataloader has no more data, initialize a new round
                actor_loader = iter(DataLoader(self.actor_datasets.train, batch_size=self.train_config.batch_size, shuffle=True, num_workers=self.train_config.n_cpu))
                actor_batch = next(actor_loader, None)
                
            actor_img = actor_batch["img"].to(self.device)
            actor_msk = actor_batch["msk"].to(self.device)
            
            actor_output = self.actor_model(actor_img)
            actor_loss = self.get_actor_loss(actor_output, actor_msk, actor_batch)
            actor_iou = self.get_iou(actor_output, actor_msk)
            actor_metrics.append(self.__get_metric(actor_loss, actor_iou))
            self.__actor_optimizer_step(actor_loss)
            
            desc = f"train epoch {epoch} step {step}"
            if len(metrics) > 0:
                evaluation = self.__get_mean(metrics)
                desc = desc + f" loss {evaluation.loss:.4f} critic_iou {evaluation.iou:.4f}"
            if len(actor_metrics) > 0:
                actor_evaluation = self.__get_mean(actor_metrics)
                desc = desc + f" actor_iou {actor_evaluation.iou:.4f}"                
            pbar.set_description(desc)
            
#             break
        self.logger.list_of_scalars_summary([ ("train_loss", actor_evaluation.loss), ("train_iou", actor_evaluation.iou) ], epoch)
        
        return actor_evaluation

    def __test_one_epoch(self, epoch):
        '''test the model for one epoch with progress bar'''
        self.critic_model.eval()
        self.actor_model.eval()
        loader = DataLoader(self.actor_datasets.test, batch_size=self.train_config.batch_size, shuffle=False, num_workers=self.train_config.n_cpu)
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
        actor_metrics = []
        for step, actor_batch in pbar:
            actor_img = actor_batch["img"].to(self.device)
            actor_msk = actor_batch["msk"].to(self.device)

            with torch.no_grad():
                actor_output = self.actor_model(actor_img)
                actor_loss = self.get_actor_loss(actor_output, actor_msk, actor_batch)
                actor_iou = self.get_iou(actor_output, actor_msk)
                actor_metrics.append(self.__get_metric(actor_loss, actor_iou))

            evaluation = self.__get_mean(actor_metrics)
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
    parser.add_argument("--actor_module_name", type=str, default="segmentation", help="actor module / package name")
    parser.add_argument("--actor_model_class", type=str, required=True, help="actor model class name, eg., MobileNetv2_DeepLabv3")
    parser.add_argument("--actor_model_weights", type=str, default=None, help="actor model weights, i.e. .pth file")
    parser.add_argument("--critic_module_name", type=str, default="segmentation", help="module / package name")
    parser.add_argument("--critic_model_class", type=str, required=True, help="model class name, eg., MobileNetv2_DeepLabv3")
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
    parser.add_argument("--n_future_neighbors", type=int, default=4, help="# of future frames for training critic model")
    parser.add_argument("--critic_epochs", type=int, default=3, help="critic training cycle")
    opt = parser.parse_args()
    print(opt)

    actor_module = importlib.import_module(opt.actor_module_name)
    actor_model = getattr(actor_module, opt.actor_model_class)().to(opt.device)
    if opt.actor_model_weights is None:
        actor_model.apply(init_weights)
    else:
        ckpt_data = torch.load(opt.actor_model_weights)
        actor_model.load_state_dict(ckpt_data["model"])

    critic_module = importlib.import_module(opt.critic_module_name)
    critic_model = getattr(critic_module, opt.critic_model_class)().to(opt.device)
    critic_model.apply(init_weights)

    data_config = EasyDict(dict(
        trainset=opt.trainset,
        testset=opt.testset,
        image_size=opt.image_size,
        n_future_neighbors=opt.n_future_neighbors,
        additional_process=None
    ))
    datasets = ActorCriticDataset.get_datasets(data_config)
    actor_datasets = HDSingleFrameDataset.get_datasets(data_config)

    max_iter = opt.epochs * math.ceil(len(datasets.train) / opt.batch_size)
    optimizer, scheduler = get_quick_optimizer(actor_model, max_iter, base_lr=opt.base_lr, power=opt.power)
    critic_optimizer, critic_scheduler = get_quick_optimizer_multi([critic_model, actor_model], max_iter, base_lr=opt.base_lr, power=opt.power)

    train_config = EasyDict(dict(
        actor_model=actor_model,
        critic_model=critic_model,
        optimizer=optimizer,
        scheduler=scheduler,
        critic_optimizer=critic_optimizer,
        critic_scheduler=critic_scheduler,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        n_cpu=opt.n_cpu,
        device=opt.device,
        log_dir=opt.log_dir,
        model_path=opt.model_prefix,
        n_future_neighbors=opt.n_future_neighbors,
        critic_epochs=opt.critic_epochs
    ))
    trainer = CriticTrainer(datasets, actor_datasets, train_config)
    trainer.train()

