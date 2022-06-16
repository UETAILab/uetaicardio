import torch
from torch import nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

class SimpleSkipConnection(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, upper_x, lower_x):
        return torch.cat([upper_x, lower_x], dim=1)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels, reduction)
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, reduction=16):
        super(SEBottleneck, self).__init__()

        width = out_channels // self.expansion
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=1, # change
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(out_channels, reduction)
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, n_layers=1):
        super().__init__()
        layers = []
        layers.append(SEBottleneck(in_channels, out_channels, dilation=dilation))
        for _ in range(1, n_layers):
            layers.append(SEBottleneck(out_channels, out_channels, dilation=dilation))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, n_layers=1):
        super().__init__()
        layers = []
        layers.append(SEBottleneck(in_channels, out_channels, dilation=dilation))
        for _ in range(1, n_layers):
            layers.append(SEBottleneck(out_channels, out_channels, dilation=dilation))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

class UNet(nn.Module):
    def __init__(self, n_class=1, base_rate=8):
        super().__init__()
        width_scale, depth_scale = 1, 3
        aspp_rates = [base_rate, base_rate*2, base_rate*3]

        self.conv_down1 = SEBasicBlock(in_channels=3, 
                                       out_channels=int(64 * width_scale))
        self.conv_down2 = EncoderBlock(in_channels=int(64 * width_scale), 
                                       out_channels=int(128 * width_scale),
                                       n_layers=1)
        self.conv_down3 = EncoderBlock(in_channels=int(128 * width_scale), 
                                       out_channels=int(256 * width_scale), 
                                       n_layers=1)
        self.conv_down4 = EncoderBlock(in_channels=int(256 * width_scale), 
                                       out_channels=int(512 * width_scale), 
                                       n_layers=1)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.skip3 = SimpleSkipConnection()
        self.conv_up3 = DecoderBlock(in_channels=int(256 * width_scale) + int(512 * width_scale), 
                                     out_channels=int(256 * width_scale), 
                                     dilation=base_rate, n_layers=int(depth_scale))
        self.aux3 = nn.Conv2d(int(256 * width_scale), n_class, 1)
        
        self.skip2 = SimpleSkipConnection()
        self.conv_up2 = DecoderBlock(in_channels=int(128 * width_scale) + int(256 * width_scale), 
                                     out_channels=int(128 * width_scale),
                                     dilation=base_rate, n_layers=int(depth_scale))
        self.aux2 = nn.Conv2d(int(128 * width_scale), n_class, 1)
        
        self.skip1 = SimpleSkipConnection()
        self.conv_up1 = DecoderBlock(in_channels=int(128 * width_scale) + int(64 * width_scale), 
                                     out_channels=int(64 * width_scale),
                                     dilation=base_rate, n_layers=int(depth_scale))
        self.conv_last = nn.Conv2d(int(64 * width_scale), n_class, 1)
    
    def forward(self, x):
        # encoder
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)
        x = self.conv_down4(x)
        
        # decoder
        x = self.upsample(x)
        x = self.skip3(x, conv3)
        x = self.conv_up3(x)
#         aux3 = F.interpolate(self.aux3(x), scale_factor=4, mode="bilinear", align_corners=True)
        
        x = self.upsample(x)
        x = self.skip2(x, conv2)
        x = self.conv_up2(x)
#         aux2 = F.interpolate(self.aux2(x), scale_factor=2, mode="bilinear", align_corners=True)
        
        x = self.upsample(x)
        x = self.skip1(x, conv1)
        x = self.conv_up1(x)
        out = self.conv_last(x)
        
#         if self.training:
#             return aux3, aux2, out
#         else:
#             return out
        return out