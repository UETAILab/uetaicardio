
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Params():
    def __init__(self):
        # network structure parameters
        self.model = 'Unet2'
        self.framework = 'torch'
        self.input_shape = [3, 256, 256]
        self.num_class = 1
        self.sync_bn = True
        self.freeze_bn = False

def double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )


class Unet(nn.Module):
    def __init__(self, config=Params()):
        super(Unet, self).__init__()
        self.config = config
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, self.config.num_class, 1)

    def forward(self, input):
        conv1 = self.dconv_down1(input)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
    
def dilated_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

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
    
                
class Unet2(nn.Module):
    def __init__(self, config=Params()):
        super(Unet2, self).__init__()
        self.config = config
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.dilated_block3 = _DSPPB(512)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dilated_block2 = _DSPPB(256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dilated_block1 = _DSPPB(128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, self.config.num_class, 1)

    def forward(self, input):
        conv1 = self.dconv_down1(input)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)
        x = self.dilated_block3(x)
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)   
        x = self.dilated_block2(x)
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)  
        y = self.dilated_block1(x)
        x = torch.cat([y, conv1], dim=1)   
        
        x = self.dconv_up1(x)        
        out = self.conv_last(x)
        
        return out
