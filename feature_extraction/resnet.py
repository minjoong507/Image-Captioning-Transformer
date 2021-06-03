import torchvision.models as models
import torch.nn as nn
import torch


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        resnet = models.resnet152(pretrained=True)  # load pretrained model
        resnet_modules = list(resnet.children())[:-1]   # except the last fc layer
        self.ResNet = nn.Sequential(*resnet_modules)

    def forward(self, x):
        x = self.ResNet(x)
        x = x.view(x.size(0), -1)

        return x
