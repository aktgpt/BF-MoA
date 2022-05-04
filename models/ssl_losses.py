from collections import OrderedDict
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torchvision import models

from .utils.resnet_utils import conv1x1, get_kwargs


def Projector(mlp, embedding):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BTLoss(nn.Module):
    def __init__(self, batch_size, num_features=2048, lambd=0.0051):
        super(BTLoss, self).__init__()
        self.batch_size = batch_size
        self.num_features = num_features
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(num_features, affine=False)

    def forward(self, z1, z2):
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        print(on_diag.item(), off_diag.item())
        loss = on_diag + self.lambd * off_diag
        return loss


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


class VICRegLoss(nn.Module):
    def __init__(
        self, batch_size, num_features=2048, sim_coeff=1.0, std_coeff=1.0, cov_coeff=0.001
    ):
        super(VICRegLoss, self).__init__()
        self.batch_size = batch_size
        self.num_features = num_features
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y):
        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(
            cov_y
        ).pow_(2).sum().div(self.num_features)

        loss = (
            (self.sim_coeff * repr_loss) + (self.std_coeff * std_loss) + (self.cov_coeff * cov_loss)
        )
        print(repr_loss.item(), std_loss.item(), cov_loss.item())
        return loss


class VICReg(nn.Module):
    def __init__(
        self, backbone1, backbone2, sim_coeff=1.0, std_coeff=1.0, cov_coeff=0.04, batch_size=256
    ):
        super(VICReg, self).__init__()

        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.num_features = backbone1.inplanes
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.batch_size = batch_size
        self.projector1 = Projector("4096-4096-4096", backbone1.inplanes)
        self.projector2 = Projector("4096-4096-4096", backbone2.inplanes)

    def forward(self, x, y):
        pred_x, x = self.backbone1(x)
        pred_y, y = self.backbone2(y)
        x = self.projector1(x)
        y = self.projector2(y)

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(
            cov_y
        ).pow_(2).sum().div(self.num_features)

        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return pred_x, pred_y, loss
