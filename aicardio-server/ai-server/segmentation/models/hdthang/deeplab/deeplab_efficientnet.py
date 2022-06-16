# https://github.com/jfzhang95/pytorch-deeplab-xception

import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from torch.utils import data

from aicardio.data.config import get_config_values, get_compute_device
from aicardio.data.csv_dataset import CSVDataset
from aicardio.data.metrics import Metric
from aicardio.models.hdthang.deeplab.aspp import build_aspp
from aicardio.models.hdthang.deeplab.backbone import build_backbone, efficientnet
from aicardio.models.hdthang.deeplab.decoder import build_decoder
from aicardio.models.hdthang.deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Params():
    def __init__(self):
        # network structure parameters
        self.model = 'DeepLab'
        self.framework = 'torch'
        self.input_shape = [3, 256, 256]
        self.model_path = 'aicardio/runs/torch_models/DeepLab_MobileNet'
        self.s = [2, 1, 2, 2, 2, 1, 1]  # stride of each conv stage
        self.t = [1, 1, 6, 6, 6, 6, 6]  # expansion factor t
        self.n = [1, 1, 2, 3, 4, 3, 3]  # number of repeat time
        # output channel of each conv stage
        self.c = [32, 16, 24, 32, 64, 96, 160]
        self.output_stride = 16
        self.multi_grid = (1, 2, 4)
        self.aspp = (6, 12, 18)
        self.down_sample_rate = 32  # classic down sample rate
        self.num_class = 1
        self.backbone = 'mobilenet'
        self.sync_bn = True
        self.freeze_bn = False


class DeepLab(nn.Module):
    def __init__(self, config=Params()):
        super(DeepLab, self).__init__()
        self.config = config
        if self.config.backbone == 'drn':
            self.config.output_stride = 8
        if self.config.sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(self.config.backbone, self.config.output_stride, BatchNorm)
        self.aspp = build_aspp(self.config.backbone, self.config.output_stride, BatchNorm)
        self.decoder = build_decoder(self.config.num_class, self.config.backbone, BatchNorm)

        if self.config.freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return torch.sigmoid(x)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    # MUST implement: convert H,W,3 image (np.uint8) to 1,3,H,W tensor (float32)
    def image_to_tensor(self, img, add_N=True):
        w, h = self.config.input_shape[2], self.config.input_shape[1]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        if add_N:
            img = img[None, ...]
            return torch.tensor(img).permute(0, 3, 1, 2).type(torch.float32)
        else:
            return torch.tensor(img).permute(2, 0, 1).type(torch.float32)

    # MUST implement: convert H,W,1 image (np.uint8) to 1,H,W tensor (float32)
    def mask_image_to_tensor(self, mask):
        w, h = self.config.input_shape[2], self.config.input_shape[1]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = mask[..., None].astype(np.float32) / 255.0
        return torch.tensor(mask).permute(2, 0, 1).type(torch.float32)

    # MUST implement: convert N,1,H,W output tensor (float32) to N,H,W,1 mask image (np.uint8)
    def mask_tensor_to_image(self, ts, original_input=None):
        ts = ts.permute(0, 2, 3, 1)
        mask = torch.round(ts * 255.0).cpu().numpy().astype(np.uint8)
        if original_input is not None:
            mask = [self.resize_mask_to_original_size(
                m, o) for m, o in zip(mask, original_input)]
        return mask

        # MUST implement: complete training code using data from train_csv and test_csv

    def train_model(self, train_csv, test_csv, continue_training=False):
        print(f"Start training model using data {train_csv}")
        train_dataset = CSVDataset(train_csv, model=self)
        test_dataset = CSVDataset(test_csv, model=self, augment=False)
        print(f"train {len(train_dataset)} test {len(test_dataset)}")

        device = get_compute_device()

        train_config = get_config_values(
            self.config,
            {'max_epochs': 30, 'batch_size': 16,
             'learning_rate': 5e-4, 'weight_decay': 1e-4}
        )
        print('Training config', train_config)
        params = {'batch_size': train_config.batch_size,
                  'shuffle': True, 'num_workers': 6}

        train_generator = data.DataLoader(train_dataset, **params)
        test_generator = data.DataLoader(test_dataset, **params)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)

        start_epoch, best_ave = 0, None
        if continue_training: start_epoch, best_ave = self.load_model_to_continue()
        self.to(device)
        print(f"Start training from epoch {start_epoch}")
        for epoch in range(start_epoch, start_epoch + train_config.max_epochs):
            train_metric = self.train_one_epoch(
                epoch, train_generator, optimizer, device)
            test_metric = self.test_one_epoch(epoch, test_generator, device)
            test_ave = test_metric.average()
            best_ave = self.save_if_better(
                epoch, test_ave, best_ave, train_metric.average())

    # MUST implement: load a model's weights from config.model_path
    def load_model(self, model_path=None):
        if model_path is None:
            model_path = f"{self.config.model_path}.best.pth"
        print("Loading model from", model_path)
        self.load_state_dict(torch.load(model_path))

        # load previous training log (epoch, best_ave.iou) and model

    def load_model_to_continue(self):
        model_path = self.config.model_path
        log_path = f"{model_path}.log"
        model_path = f"{model_path}.pth"
        print(f"Try to continue training from log {log_path}")
        if not os.path.isfile(model_path) or not os.path.isfile(log_path):
            return 0, None
        df = pd.read_csv(log_path, skiprows=1)
        start_epoch = df['epoch'].max() + 1
        best_ave = EasyDict({"iou": df['val_iou'].max()})
        self.load_model(model_path)
        print(f"Loaded previous training log best IoU = {best_ave.iou} and model from {model_path}")
        return start_epoch + 1, best_ave

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)

    def save_if_better(self, epoch, new_ave, best_ave, train_ave):
        model_path = self.config.model_path
        keys = ['epoch'] + list(train_ave.keys()) + \
               ['val_' + k for k in new_ave.keys()] + ['best_model']
        if best_ave is None:
            log_path = f"{model_path}.log"
            with open(log_path, 'w') as f:
                print(f"Save training log to {log_path}")
                print(self.config, file=f)
                print(",".join(keys), file=f)

        with open(f"{model_path}.log", 'a') as f:
            print(f"{epoch},", file=f, end='')
            [print(f"{train_ave[k]:.4f},", file=f, end='')
             for k in train_ave.keys()]
            [print(f"{new_ave[k]:.4f},", file=f, end='')
             for k in new_ave.keys()]
            print(f"{best_ave is None or best_ave.iou < new_ave.iou}", file=f)

        self.save_model(f"{model_path}.pth")
        if best_ave is None or best_ave.iou < new_ave.iou:
            print(f"new best IoU {new_ave.iou:.4f}")
            self.save_model(f"{model_path}.best.pth")
            return new_ave
        else:
            return best_ave

    def bce_dice_loss(self, ypred, ytrue, epsilon=1e-6):
        bce_loss = nn.BCELoss()(ypred, ytrue)
        dice = (2 * torch.sum(ypred * ytrue) + epsilon) / \
               (torch.sum(ypred) + torch.sum(ytrue) + epsilon)
        dice_loss = -torch.log(dice)
        return bce_loss + dice_loss

    def segmentation_metrics(self, ypred, ytrue, epsilon=1e-6):
        ypred = torch.round(ypred)
        ytrue = torch.round(ytrue)
        intersection = torch.sum(ypred * ytrue)
        sum_y = torch.sum(ypred) + torch.sum(ytrue)
        dice = (2 * intersection + epsilon) / (sum_y + epsilon)
        iou = (intersection + epsilon) / (sum_y - intersection + epsilon)
        return dict(dice=dice.item(), iou=iou.item())

    def train_one_epoch(self, epoch, generator, optimizer, device):
        self.train()
        metric = Metric()
        for step, (batch, labels) in enumerate(generator):
            optimizer.zero_grad()
            ypred = self(batch.to(device))
            ytrue = labels.to(device)
            loss = self.bce_dice_loss(ypred, ytrue)
            metrics = {"loss": loss.item(), **
            (self.segmentation_metrics(ypred, ytrue))}
            loss.backward()
            optimizer.step()

            metric.accumulate(metrics, len(batch))
            print(
                f"train epoch {epoch} step {step} loss {loss.item():.4f} ave {metric.pretty_average()}", end='\r')
        print()
        return metric

    def test_one_epoch(self, epoch, generator, device):
        self.eval()
        metric = Metric()
        with torch.no_grad():
            for step, (batch, labels) in enumerate(generator):
                ypred = self(batch.to(device))
                ytrue = labels.to(device)
                loss = self.bce_dice_loss(ypred, ytrue)
                metrics = {"loss": loss.item(), **
                (self.segmentation_metrics(ypred, ytrue))}

                metric.accumulate(metrics, len(batch))
                print(
                    f"test  epoch {epoch} step {step} loss {loss.item():.4f} ave {metric.pretty_average()}",
                    end='\r')
        print()
        return metric

    def resize_mask_to_original_size(self, mask, original):
        h, w = original.shape[:2]
        mask = cv2.resize(
            mask, (w, h), interpolation=cv2.INTER_NEAREST)[..., None]
        return mask


def summary(model):
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total param: {params}\ntrainable param: {trainable_params}")


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    summary(model)
    model.eval()
    input = torch.rand(1, 3, 256, 256)
    output = model(input)
    print(output.size())

    # export USE_GPU=cuda:1
    # python -m aicardio.tests.test_torch_model --do_train --module aicardio.models.hdthang.deeplab --model DeepLab --config aicardio/models/hdthang/deeplab/deeplab.json --train aicardio/data-dicom/2019-10-30_2C/train_dev/train.csv --test aicardio/data-dicom/2019-10-30_2C/test/test.csv
    # python -m aicardio.tests.test_torch_model --do_train --module aicardio.models.hdthang.deeplab --model DeepLab --config aicardio/models/hdthang/deeplab/deeplab.json --train aicardio/data-dicom/2019-10-30_2C/train_dev/train.csv --test aicardio/data-dicom/2019-10-30_2C/test/test.csv
