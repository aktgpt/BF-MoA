from collections import OrderedDict
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from .resnet_utils import conv1x1, get_kwargs


class ResNetMIL(nn.Module):
    def __init__(self, model_name, in_channels=3, num_classes=1000):
        super(ResNetMIL, self).__init__()

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1

        kwargs = get_kwargs(model_name)
        block = kwargs["block"]
        layers = kwargs["layers"]
        self.groups = kwargs["groups"] if "groups" in kwargs else 1
        self.base_width = kwargs["width_per_group"] if "width_per_group" in kwargs else 64

        conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        bn1 = self._norm_layer(self.inplanes)
        relu = nn.ReLU(inplace=False)  # inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layer1 = self._make_layer(block, 64, layers[0])
        layer2 = self._make_layer(block, 128, layers[1], stride=2)
        layer3 = self._make_layer(block, 256, layers[2], stride=2)
        layer4 = self._make_layer(block, 512, layers[3], stride=2)

        avgpool = nn.AdaptiveAvgPool2d((1, 1)) 

        self.feature_extractor_1 = nn.Sequential(
            conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool
        )
        self.L = 1024
        self.D = 256
        self.K = 1

        self.feature_extractor_2 = nn.Sequential(nn.Linear(self.inplanes, self.L), nn.ReLU())
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)

        self.softmax = nn.Softmax(dim=1)

        self.fc = nn.Linear(self.L * self.K, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        b, n, c, w, h = x.shape
        x = x.view(b * n, c, w, h)

        x = self.feature_extractor_1(x).flatten(1)
        x = self.feature_extractor_2(x).view(b, n, -1)

        atten_v = self.attention_V(x)
        atten_u = self.attention_U(x)
        atten_weights = self.softmax(self.attention_weights(atten_v * atten_u))  # .transpose(1, 2)
        
        features = torch.bmm(atten_weights.transpose(1, 2), x).view(b, -1)
        x = self.fc(features)

        return x, features
