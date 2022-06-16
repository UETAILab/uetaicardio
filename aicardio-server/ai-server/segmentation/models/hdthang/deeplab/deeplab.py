# https://github.com/jfzhang95/pytorch-deeplab-xception

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation.models.hdthang.deeplab.aspp import build_aspp
from segmentation.models.hdthang.deeplab.backbone import build_backbone
from segmentation.models.hdthang.deeplab.decoder import build_decoder
from segmentation.models.hdthang.deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Params():
    def __init__(self):
        # network structure parameters
        self.model = 'DeepLab'
        self.framework = 'torch'
        self.input_shape = [3, 256, 256]
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

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    
    