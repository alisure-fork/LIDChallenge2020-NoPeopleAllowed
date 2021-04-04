import math
import torch
import random
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from alisuretool.Tools import Tools
from torch.utils.data import DataLoader, Dataset
from deep_labv3plus_pytorch.network.modeling import deeplabv3plus_resnet50


class ConvBlock(nn.Module):

    def __init__(self, cin, cout, stride=1, has_relu=True, has_bn=True):
        super().__init__()
        self.has_relu = has_relu
        self.has_bn = has_bn

        self.conv = nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)
        if self.has_bn:
            self.bn = nn.BatchNorm2d(cout)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)
        pass

    def forward(self, x):
        out = self.conv(x)
        if self.has_bn:
            out = self.bn(out)
        if self.has_relu:
            out = self.relu(out)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class CAMNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = models.vgg16_bn(pretrained=True)

        # self.head_conv = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512), ConvBlock(512, 512))
        self.head_conv = nn.Sequential(ConvBlock(512, 512))
        self.head_linear = nn.Linear(512, self.num_classes)
        pass

    def forward(self, x):
        features = self.backbone.features(x)
        features = self.head_conv(features)
        features = F.adaptive_avg_pool2d(features, output_size=(1, 1)).view((features.size()[0], -1))

        features = self.head_linear(features)
        return features

    def forward_map(self, x):
        features = self.backbone.features(x)
        features = self.head_conv(features)
        out_features = features
        features = F.adaptive_avg_pool2d(features, output_size=(1, 1)).view((features.size()[0], -1))

        features = self.head_linear(features)
        logits = torch.sigmoid(features)
        return logits, out_features

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class DeepLabV3Plus(nn.Module):

    def __init__(self, num_classes, output_stride=8):
        super().__init__()
        self.model = deeplabv3plus_resnet50(num_classes=num_classes,
                                            output_stride=output_stride, pretrained_backbone=True)
        self.set_bn_momentum(self.model.backbone, momentum=0.01)
        pass

    def forward(self, x):
        out = self.model(x)
        return out

    def get_params_groups(self):
        return list(self.model.backbone.parameters()), list(self.model.classifier.parameters())

    @staticmethod
    def set_bn_momentum(model, momentum=0.1):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = momentum
            pass
        pass

    pass
