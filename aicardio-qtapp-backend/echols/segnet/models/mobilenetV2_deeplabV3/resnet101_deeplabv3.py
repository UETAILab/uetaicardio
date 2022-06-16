import torch
import torch.nn as nn
import numpy as np
import inspect
import torchvision

class Resnet101DeeplabV3(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super(Resnet101DeeplabV3, self).__init__()
        self.pretrained = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=pretrained)
        deeplab_head_layers = list(self.pretrained.classifier.children())
        self.deeplab_head = nn.Sequential(
            *(deeplab_head_layers[:-1]),
            nn.Conv2d(256, num_classes, 1),
        )
        self.pretrained.classifier = self.deeplab_head
#         self.pretrained.backbone.require_grad = False

    def forward(self, x):
        out = self.pretrained(x)['out']
        return out
    
if __name__ == "__main__":
    model = Resnet101DeeplabV3().cuda()
    model.eval()
    x = torch.from_numpy(np.zeros((1,3,1024,768), dtype='float32')).cuda()
    out = model(x)
    print(type(out), out.shape)
    
#     print(torch.hub.help('pytorch/vision:v0.5.0', 'deeplabv3_resnet101'))
#     print(dir(model))
    print(dir(model.pretrained))
    print(type(model.pretrained))
#     print(dir(model.pretrained.classifier))
    
#     sourceDLHEAD = inspect.getsource(torchvision.models.segmentation.deeplabv3.DeepLabHead)
#     print(sourceDLHEAD)

#     [ print(layer) for layer in model.deeplab_head.children() ]

    print( inspect.getsource(torchvision.models.segmentation.deeplabv3.DeepLabV3) )