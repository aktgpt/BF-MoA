from torch import nn


def get_kwargs(model_name):
    if model_name == "resnet18":
        return {"layers": [2, 2, 2, 2], "block": BasicBlock}
    elif model_name == "resnet34":
        return {"layers": [3, 4, 6, 3], "block": BasicBlock}
    elif model_name == "resnet50":
        return {"layers": [3, 4, 6, 3], "block": Bottleneck}
    elif model_name == "resnet101":
        return {"layers": [3, 4, 23, 3], "block": Bottleneck}
    elif model_name == "resnet152":
        return {"layers": [3, 8, 36, 3], "block": Bottleneck}
    elif model_name == "resnext50_32x4d":
        return {
            "layers": [3, 4, 6, 3],
            "block": Bottleneck,
            "groups": 32,
            "width_per_group": 4,
        }
    elif model_name == "resnext101_32x8d":
        return {
            "layers": [3, 4, 23, 3],
            "block": Bottleneck,
            "groups": 32,
            "width_per_group": 8,
        }
    elif model_name == "wide_resnet50_2":
        return {
            "layers": [3, 4, 6, 3],
            "block": Bottleneck,
            "width_per_group": 64 * 2,
        }
    elif model_name == "wide_resnet101_2":
        return {
            "layers": [3, 4, 23, 3],
            "block": Bottleneck,
            "width_per_group": 64 * 2,
        }


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
    ):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

 
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
    ):
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
