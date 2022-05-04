import numpy as np
import scipy.stats as st
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from collections import OrderedDict

from .utils.resnet_utils import get_kwargs, Bottleneck, BasicBlock, conv1x1


class MSABNResNet(nn.Module):
    def __init__(self, model_name, in_channels=3, num_classes=1000):
        super(MSABNResNet, self).__init__()

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
        ms_layer_total += self.inplanes

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        ms_layer_total += self.inplanes

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        ms_layer_total += self.inplanes

        self.attn_layer = nn.Sequential(
            nn.Conv2d(ms_layer_total, self.inplanes, 1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            self._make_layer(block, 512, layers[3], stride=1, attn_block=True),
            nn.BatchNorm2d(self.inplanes * 2),
            nn.Conv2d(self.inplanes * 2, num_classes, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
        )

        self.attn_conv = nn.Sequential(
            nn.Conv2d(num_classes, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.attn_cam = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
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

    def _make_layer(self, block, planes, blocks, stride=1, attn_block=False):
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

        new_inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(new_inplanes, planes, groups=self.groups, base_width=self.base_width)
            )
        if not attn_block:
            self.inplanes = new_inplanes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)

        x2 = self.layer2(x1)

        x3 = self.layer3(x2)

        ax = self.attn_layer(
            torch.cat([F.interpolate(x1, x2.shape[2:]), x2, F.interpolate(x3, x2.shape[2:])], dim=1)
        )
        attn = self.attn_conv(ax)
        cam_op = self.attn_cam(ax)

        x = x + (
            x
            * F.interpolate(
                attn,
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        )

        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        op = self.fc(x)

        return op, cam_op, attn
