import torch
import torch.nn as nn

import segmentation.models.mobilenetV2_deeplabV3.layers as layers
from segmentation.models.mobilenetV2_deeplabV3.params import Params

# create model
class MobileNetv2_DeepLabv3(nn.Module):
    """
    A Convolutional Neural Network with MobileNet v2 backbone and DeepLab v3 head
    """

    """######################"""
    """# Model Construction #"""
    """######################"""

    # def __init__(self, params, datasets):
    def __init__(self, params=Params(), in_channels=3):
        super(MobileNetv2_DeepLabv3, self).__init__()
        self.params = params

        # build network
        block = []

        # conv layer 1
        block.append(nn.Sequential(nn.Conv2d(in_channels, self.params.c[0], 3, stride=self.params.s[0], padding=1, bias=False),
                                   nn.BatchNorm2d(self.params.c[0]),
                                   # nn.Dropout2d(self.params.dropout_prob, inplace=True),
                                   nn.ReLU6()))

        # conv layer 2-7
        for i in range(6):
            block.extend(layers.get_inverted_residual_block_arr(self.params.c[i], self.params.c[i+1],
                                                                t=self.params.t[i+1], s=self.params.s[i+1],
                                                                n=self.params.n[i+1]))

        # dilated conv layer 1-4
        # first dilation=rate, follows dilation=multi_grid*rate
        rate = self.params.down_sample_rate // self.params.output_stride
        block.append(layers.InvertedResidual(self.params.c[6], self.params.c[6],
                                             t=self.params.t[6], s=1, dilation=rate))
        for i in range(3):
            block.append(layers.InvertedResidual(self.params.c[6], self.params.c[6],
                                                 t=self.params.t[6], s=1, dilation=rate*self.params.multi_grid[i]))

        # ASPP layer
        block.append(layers.ASPP_plus(self.params))

        # final conv layer
        block.append(nn.Conv2d(256, self.params.num_class, 1))

        # bilinear upsample
        block.append(nn.Upsample(scale_factor=self.params.output_stride, mode='bilinear', align_corners=False))

        self.network = nn.Sequential(*block)
        
    def forward(self, x):
        return self.network(x)

class CriticMobileNetv2_DeepLabv3(MobileNetv2_DeepLabv3):
    def __init__(self):
        super(CriticMobileNetv2_DeepLabv3, self).__init__(in_channels=2)

""" TEST """
if __name__ == '__main__':
    import numpy as np
    params = Params()
    net = MobileNetv2_DeepLabv3(params)
    net.eval()
    x = torch.FloatTensor(np.zeros((1, 3, 256, 256)))

    device = "cuda"

    net.to(device)
    x = x.to(device)
    y = net(x)
    print("output", y.shape, y.device, type(y))