import random
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from .utils.resnet_utils import conv1x1, get_kwargs


class ResNetMSFC(nn.Module):
    def __init__(self, model_name, in_channels=3, num_classes=1000, upsample_ratio=[0.25, 0.5, 1]):
        super(ResNetMSFC, self).__init__()

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1

        kwargs = get_kwargs(model_name)
        block = kwargs["block"]
        layers = kwargs["layers"]
        self.groups = kwargs["groups"] if "groups" in kwargs else 1
        self.base_width = kwargs["width_per_group"] if "width_per_group" in kwargs else 64

        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ms_layer_total = 0

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.upsample1 = nn.Upsample(
            scale_factor=upsample_ratio[0], mode="bilinear", align_corners=False
        )
        ms_layer_total += self.inplanes

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.upsample2 = nn.Upsample(
            scale_factor=upsample_ratio[1], mode="bilinear", align_corners=False
        )
        ms_layer_total += self.inplanes

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.upsample3 = nn.Upsample(
            scale_factor=upsample_ratio[2], mode="bilinear", align_corners=False
        )
        ms_layer_total += self.inplanes

        self.msconv_layer = nn.Sequential(
            nn.Conv2d(ms_layer_total, self.inplanes, 1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
        )
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        ms1_in = self.upsample1(x)

        x = self.layer2(x)
        ms2_in = self.upsample2(x)

        x = self.layer3(x)
        ms3_in = self.upsample3(x)

        ax = self.msconv_layer(torch.cat([ms1_in, ms2_in, ms3_in], dim=1))
        ax = self.layer4(ax)

        ax = self.avgpool(ax)
        ax = torch.flatten(ax, 1)
        x = self.fc(ax)

        return x, ax


class ResNetMSFCDrop(nn.Module):
    def __init__(self, model_name, num_classes=1000, upsample_ratio=[0.5, 1, 2]):
        super(ResNetMSFCDrop, self).__init__()

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1

        kwargs = get_kwargs(model_name)
        block = kwargs["block"]
        layers = kwargs["layers"]
        self.groups = kwargs["groups"] if "groups" in kwargs else 1
        self.base_width = kwargs["width_per_group"] if "width_per_group" in kwargs else 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ms_layer_total = 0

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.upsample1 = nn.Upsample(
            scale_factor=upsample_ratio[0], mode="bilinear", align_corners=False
        )
        ms_layer_total += self.inplanes

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.upsample2 = nn.Upsample(
            scale_factor=upsample_ratio[1], mode="bilinear", align_corners=False
        )
        ms_layer_total += self.inplanes

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.upsample3 = nn.Upsample(
            scale_factor=upsample_ratio[2], mode="bilinear", align_corners=False
        )
        ms_layer_total += self.inplanes

        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

        self.msconv_layer = nn.Sequential(
            nn.Conv2d(ms_layer_total, self.inplanes, 1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
        )
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        ms1_in = self.upsample1(x)

        x = self.layer2(x)
        ms2_in = self.upsample2(x)

        x = self.layer3(x)
        ms3_in = self.upsample3(x)

        p = random.uniform(0, 1)
        if p > 0.5:
            ax = torch.cat([ms1_in, ms2_in, self.dropout3(ms3_in)], dim=1)
        else:
            ax = torch.cat([ms1_in, self.dropout2(ms2_in), ms3_in], dim=1)
        # ax = torch.cat([ms1_in, self.dropout2(ms2_in), self.dropout3(ms3_in)], dim=1)

        ax = self.msconv_layer(ax)
        ax = self.layer4(ax)

        x = self.avgpool(ax)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x  # , ax


class ResNetMSFCCorr(nn.Module):
    def __init__(self, model_name, num_classes=1000, upsample_ratio=[0.5, 1, 2]):
        super(ResNetMSFCCorr, self).__init__()

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1

        kwargs = get_kwargs(model_name)
        block = kwargs["block"]
        layers = kwargs["layers"]
        self.groups = kwargs["groups"] if "groups" in kwargs else 1
        self.base_width = kwargs["width_per_group"] if "width_per_group" in kwargs else 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ms_layer_total = 0

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.upsample1 = nn.Upsample(
            scale_factor=upsample_ratio[0], mode="bilinear", align_corners=False
        )
        ms_layer_total += self.inplanes

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.upsample2 = nn.Upsample(
            scale_factor=upsample_ratio[1], mode="bilinear", align_corners=False
        )
        ms_layer_total += self.inplanes

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.upsample3 = nn.Upsample(
            scale_factor=upsample_ratio[2], mode="bilinear", align_corners=False
        )
        ms_layer_total += self.inplanes

        self.msconv_layer = nn.Sequential(
            nn.Conv2d(ms_layer_total, self.inplanes, 1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
        )
        self.features = self.inplanes
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        ms1_in = self.upsample1(x)

        x = self.layer2(x)
        ms2_in = self.upsample2(x)

        x = self.layer3(x)
        ms3_in = self.upsample3(x)

        ax = self.msconv_layer(torch.cat([ms1_in, ms2_in, ms3_in], dim=1))
        x = self.layer4(ax)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, ax
