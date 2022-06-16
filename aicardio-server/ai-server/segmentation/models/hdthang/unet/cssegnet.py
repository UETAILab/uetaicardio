
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from torchvision import models


class Params():
    def __init__(self):
        # network structure parameters
        self.model = 'CSSegNet'
        self.framework = 'torch'
        self.input_shape = [3, 256, 256]
        self.model_path = 'aicardio/runs/torch_models/CSSegNet'       
        self.num_class = 1
        
 
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, stride, padding, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


        
class _DSPPB(nn.Module):
    def __init__(self, channel):
        super(_DSPPB, self).__init__()
        
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=2, dilation=2)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=3)
        
        self.conv = nn.Conv2d(5*channel, channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        #x = x1 + x2 + x3 + x4 + x5
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    

                
class CSSegNet(nn.Module):
    def __init__(self, config=Params()):
        super(CSSegNet, self).__init__()
        self.config = config
                
        # Entry Flow
        self.conv1 = nn.Conv2d(self.config.input_shape[0], 16, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(self.config.input_shape[0], 16, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(self.config.input_shape[0], 16, kernel_size=3, dilation=3, padding=3)
        
        self.entry_flow_1 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True)
        )
        
        self.entry_flow_2 = nn.Sequential(
            depthwise_separable_conv(48, 96, 3, 1, 1),
            nn.ReLU(True),            
            depthwise_separable_conv(96, 96, 3, 1, 1),
            nn.ReLU(True), 
            depthwise_separable_conv(96, 96, 3, 2, 1)
        )
        
        self.entry_flow_2_residual = nn.Conv2d(48, 96, kernel_size=1, stride=2, padding=0)
        
        self.entry_flow_3 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(96, 192, 3, 1, 1),
            nn.ReLU(True),
            depthwise_separable_conv(192, 192, 3, 1, 1),
            nn.ReLU(True),
            depthwise_separable_conv(192, 192, 3, 2, 1)
        )
        
        self.entry_flow_3_residual = nn.Conv2d(96, 192, kernel_size=1, stride=2, padding=0)
        
        self.entry_flow_4 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(192, 768, 3, 1, 1),
            nn.ReLU(True),
            depthwise_separable_conv(768, 768, 3, 1, 1),
            nn.ReLU(True),
            depthwise_separable_conv(768, 768, 3, 2, 1)
        )
        
        self.entry_flow_4_residual = nn.Conv2d(192, 768, kernel_size=1, stride=2, padding=0)
        
        # Middle Flow
        self.middle_flow = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(768, 768, 3, 1, 1),
            
            nn.ReLU(True),
            depthwise_separable_conv(768, 768, 3, 1, 1),
            
            nn.ReLU(True),
            depthwise_separable_conv(768, 768, 3, 1, 1)
        )
        
        # Exit Flow
        self.exit_flow_1 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(768, 768, 3, 1, 1),
            
            nn.ReLU(True),
            depthwise_separable_conv(768, 768, 3, 1, 1),
            
            nn.ReLU(True),
            depthwise_separable_conv(768, 768, 3, 2, 1)
        )
        self.exit_flow_1_residual = nn.Conv2d(768, 768, kernel_size=1, stride=2, padding=0)
        
        self.exit_flow_2 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(768, 768, 3, 1, 1),
            nn.ReLU(True),
            
            depthwise_separable_conv(768, 768, 3, 1, 1),
            nn.ReLU(True)
        )
        
        self.dilated_block4 = _DSPPB(768)
        self.dilated_block3 = _DSPPB(768)
        self.dilated_block2 = _DSPPB(192)
        self.dilated_block1 = _DSPPB(96)
        self.dilated_block0 = _DSPPB(48)
        
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(5424, self.config.num_class, 1)

    def forward(self, x):
        
        x1=self.conv1(x)
        x2=self.conv2(x)        
        x3=self.conv3(x)        
        x=torch.cat([x1, x2, x3], dim=1)
        
        entry_out1 = self.entry_flow_1(x)
        stage0=entry_out1
        
        
        entry_out2 = self.entry_flow_2(entry_out1) + self.entry_flow_2_residual(entry_out1)
        stage1=entry_out2
        
        entry_out3 = self.entry_flow_3(entry_out2) + self.entry_flow_3_residual(entry_out2)
        stage2=entry_out3
        
        entry_out = self.entry_flow_4(entry_out3) + self.entry_flow_4_residual(entry_out3)
        
        middle_out = self.middle_flow(entry_out) + entry_out
        stage3=middle_out

        exit_out1 = self.exit_flow_1(middle_out) + self.exit_flow_1_residual(middle_out)
        exit_out2 = self.exit_flow_2(exit_out1)
        stage4=exit_out2
        
        x = self.dilated_block4(stage4)
        
        x = self.upsample1(x)  
       
        y = self.dilated_block3(stage3)  
        
        x = torch.cat([y, x], dim=1)
        
        x = self.upsample1(x)   
        
        y = self.dilated_block2(stage2)        
        x6 = torch.cat([y, x], dim=1)
        
        x = self.upsample1(x6)        
        y = self.dilated_block1(stage1)        
        x7 = torch.cat([y, x], dim=1)
        
        x = self.upsample1(x7)        
        y = self.dilated_block0(stage0)        
        x8 = torch.cat([y, x], dim=1)
        
        x9=self.upsample2(x6) 
        x10=self.upsample1(x7) 
        x=torch.cat([x8, x9, x10], dim=1)
        
        out = self.conv_last(x)        
        return out
        
    