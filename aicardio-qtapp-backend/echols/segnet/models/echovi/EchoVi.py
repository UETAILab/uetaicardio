import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
 

class EchoViNet(nn.Module):
    def __init__(self, n_class):
        r"""EchoViNet for Echocardiographic video segmentation.
       
        Args:
            n_class (int): Number of output classes.
        """
        super().__init__()
        self.backbone = ResNet3D([3, 4, 23, 3])
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.conv_up4 = Bottleneck3D(1024+512, 512//4, downsample=nn.Sequential(
            c3d1x1(1024+512, 512),
            nn.BatchNorm3d(512)
        ))
        self.conv_aux4 = nn.Conv3d(512, n_class, 1)
        self.conv_up3 = Bottleneck3D(512+256, 256//4, downsample=nn.Sequential(
            c3d1x1(512+256, 256),
            nn.BatchNorm3d(256)
        ))
        self.conv_aux3 = nn.Conv3d(256, n_class, 1)
        self.conv_up2 = Bottleneck3D(256+64, 64//4, downsample=nn.Sequential(
            c3d1x1(256+64, 64),
            nn.BatchNorm3d(64)
        ))
        self.conv_aux2 = nn.Conv3d(64, n_class, 1)
        self.conv_up1 = Bottleneck3D(64+32, 32//4, downsample=nn.Sequential(
            c3d1x1(64+32, 32),
            nn.BatchNorm3d(32)
        ))
        self.conv_last = nn.Conv3d(32, n_class, 1)
   
    def forward(self, x):
        r"""Forward propagation.
 
        Args:
            x (torch.tensor): torch tensor of shape (N, C, D, H, W).
        Returns:
            torch.tensor: output tensor of shape (N, n_class, D, H, W).
        """
        x0, x1, x2, x3, x4 = self.backbone(x)
        
        x = self.upsample(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up4(x)
        aux4 = self.conv_aux4(x)
        aux4 = F.interpolate(aux4, scale_factor=8, mode="trilinear")
       
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)
        aux3 = self.conv_aux3(x)
        aux3 = F.interpolate(aux3, scale_factor=4, mode="trilinear")
       
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up2(x)
        aux2 = self.conv_aux2(x)
        aux2 = F.interpolate(aux2, scale_factor=2, mode="trilinear")
       
        x = self.upsample(x)
        x = torch.cat([x, x0], dim=1)
        x = self.conv_up1(x)
        x = self.conv_last(x)

        if self.training:
            return aux4, aux3, aux2, x
        else:
            return x
 
 
def c3d3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    r"""3D 3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)
 
def c3d1x1(in_planes, out_planes, stride=1):
    r"""3D 1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)
 
class Bottleneck3D(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        r"""3D Residual Bottleneck module"""
        super(Bottleneck3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = c3d1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = c3d3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = c3d1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
   
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
       
        out += identity
        out = self.relu(out)
 
        return out
 
class ResNet3D(nn.Module):
    def __init__(self, layers, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
 
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
       
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
       
        self.conv0 = nn.Conv3d(1, self.inplanes//2, kernel_size=3, padding=1,
                               bias=False)
        self.bn0 = norm_layer(self.inplanes//2)
       
        self.conv1 = nn.Conv3d(self.inplanes//2, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
       
        self.layer1 = self.__make_layer(Bottleneck3D, 64, layers[0])
        self.layer2 = self.__make_layer(Bottleneck3D, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self.__make_layer(Bottleneck3D, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
 
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3D):
                    nn.init.constant_(m.bn3.weight, 0)
 
    def __make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                c3d1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
       
        x = self.conv1(x0)
        x = self.bn1(x)
        x = self.relu(x)
       
        x = self.maxpool(x1)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        return x0, x1, x2, x3, x4


if __name__ == "__main__":
    model = EchoViNet(10).cuda(1)
#     torchsummary.summary(model, (3, 32, 256, 256), device="cpu", batch_size=2)
#     model.eval()
   
    x = torch.randn(1, 3, 32, 256, 256).cuda(1)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Number of params: {sum(p.numel() for p in model.parameters())}")