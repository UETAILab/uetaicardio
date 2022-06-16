#!/data.local/giangh/envs/pipeline/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultiframeClassifier(nn.Module):
    def __init__(self, feature_dim, n_class, feature_extractor_weights=None):
        r"""

        Args:
            feature_dim (int): Number of features of feature vectors
            n_class (int): Number of classes
            feature_extractor_weights (str): Path to feature extractor weights
        """
        super().__init__()
        self.features = self.__get_feature_extractor(feature_extractor_weights)
        self.lstm = nn.LSTM(feature_dim, 16, 2, bidirectional=True)
        self.fc = nn.Linear(32, n_class)

        self.__freeze_feature_extractor()

    def __get_feature_extractor(self, feature_extractor_weights):
        feature_extractor = models.mobilenet_v2(num_classes=4)
        if feature_extractor_weights is not None:
            checkpoint_data = torch.load(feature_extractor_weights)
            feature_extractor.load_state_dict(checkpoint_data["model"])
        feature_extractor = feature_extractor.features
        return feature_extractor
    
    def __freeze_feature_extractor(self):
        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        r"""

        Args:
            x (torch.tensor): Torch tensor of shape (n, c_in, d, h, w)
        """
        features = []
        for i in range(x.shape[2]):
            feature = self.features(x[:, :, i, :, :])
            feature = F.adaptive_avg_pool2d(feature, 1)
            feature = feature.reshape(feature.shape[0], -1)
            features.append(feature[None, ...])
        features = torch.cat(features, dim=0)
        out, (hidden, cell) = self.lstm(features)
        out = self.fc(out[-1])
        return out


if __name__ == "__main__":
    r"""
    Softmax backbone:
        train epoch 6 step 83 acc 0.7381 precision_2C 0.7593 precision_3C 0.9118 precision_4C 0.8986 precision_none 0.7228 recall_2C 0.2204 recall_3C 0.2650 rec
        test epoch 5 step 205 acc 0.7452 precision_2C 0.7787 precision_3C 0.8710 precision_4C 0.8437 precision_none 0.7356 recall_2C 0.1806 recall_3C 0.2341 recall_4C 0.4615 recall_none 0.9922

    No-softmax backbone:
        train epoch 4 step 820 acc 0.8190 precision_2C 0.6937 precision_3C 0.8385 precision_4C 0.8566 precision_none 0.8223 recall_2C 0.3588 recall_3C 0.3734 re
        test epoch 4 step 205 acc 0.7908 precision_2C 0.6639 precision_3C 0.8280 precision_4C 0.9766 precision_none 0.7868 recall_2C 0.4154 recall_3C 0.3484 rec
    """
    feature_extractor_weights = "/data.local/data/models/chamber_classification/mobilenetv2_0049_0.9507_best.pth"
    device = "cuda:0"

    model = MultiframeClassifier(4, 4, feature_extractor_weights).to(device)

    x = torch.rand(2, 3, 10, 256, 256).to(device)
    y = model(x)
    print(y.shape)
