#!/data.local/giangh/envs/pipeline/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def MobilenetV2():
    return models.mobilenet_v2(num_classes=4)


if __name__ == "__main__":
    model = MobilenetV2().to("cuda:0")
    checkpoint_data = torch.load("/data.local/data/models/chamber_classification/mobilenetv2_0049_0.9507_best.pth", map_location="cuda:0")
    model.load_state_dict(checkpoint_data["model"])

    x = torch.rand(2, 3, 256, 256).to("cuda:0").float()
    pred = model(x)
    print(pred.shape)
