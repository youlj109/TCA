import torch.nn as nn
from torchvision import models
import torchvision
import torch
import timm  #load ViT or MLP-mixer

res_dict = {"resnet18": models.resnet18, 
            "resnet34": models.resnet34, 
            "resnet50": models.resnet50,
            "resnet101": models.resnet101, 
            "resnet152": models.resnet152, 
            "resnext50": models.resnext50_32x4d, 
            "resnext101": models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

print('resnet18:',res_dict["resnet18"](pretrained=True).fc.in_features)
print('resnet50:',res_dict["resnet50"](pretrained=True).fc.in_features)