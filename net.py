import torch.nn
import torchvision.models
from torch import nn


# 基于 resnet 分类器

class ResNet(nn.Module):
    def __init__(self, num_class):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        # self.resnet.fc = nn.Sequential(
        #     nn.Linear(self.resnet.fc.in_features, num_class),
        #     nn.Dropout(0.4),
        # )
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_class),
            nn.Dropout(0.45)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x
