import math
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision
from models import ssl_losses
from models.ssl_losses import Projector, VICRegLoss
from numpy.lib.shape_base import apply_along_axis
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch import nn, optim
from torch.multiprocessing import Lock
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from utils.metrics import get_metrics
from utils.multiprocess_utils import cleanup, setup

from .dist_sampler import DistributedWeightedSampler


class SSLVSDistTrainer:
    def __init__(self, config, save_folder):
        self.config = config
        self.epochs = config["epochs"]

        self.save_folder = save_folder
        # self.save_interval = config["save_interval"]
        self.image_folder = os.path.join(self.save_folder, "train_images")

        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        self.lr = config["init_lr"]
        self.wd = config["wd"]
        self.milestones = config["milestones"]

    def train(self, train_dataset, valid_dataloader, model):
        world_size = torch.cuda.device_count()
        mp.spawn(
            self.main_worker,
            args=(world_size, train_dataset, valid_dataloader, model),
            nprocs=world_size,
            join=True,
        )

    def main_worker(self, rank, world_size, train_dataset, valid_dataloader, models):
        setup(rank, world_size)

        model_f = torch.nn.SyncBatchNorm.convert_sync_batchnorm(models[0])
        model_bf = torch.nn.SyncBatchNorm.convert_sync_batchnorm(models[1])

        projector_f = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            Projector("8192-8192-8192", 2048)
        )
        projector_bf = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            Projector("8192-8192-8192", 2048)
        )

        batch_size = int(self.config["batch_size"] / world_size)
        # model = VICReg(models[0], models[1], batch_size)
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        torch.cuda.set_device(rank)

        self.model_f = DDP(model_f.to(rank), device_ids=[rank])
        self.model_bf = DDP(model_bf.to(rank), device_ids=[rank])
        self.projector_f = DDP(projector_f.to(rank), device_ids=[rank])
        self.projector_bf = DDP(projector_bf.to(rank), device_ids=[rank])

        optimizer = optim.SGD(
            list(self.model_f.parameters())
            + list(self.projector_f.parameters())
            + list(self.model_bf.parameters())
            + list(self.projector_bf.parameters()),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.wd,
        )
        # train_scheduler = optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=self.milestones, gamma=0.1, verbose=True
        # )

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=int((32 + world_size - 1) / world_size),
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            sampler=train_sampler,
            worker_init_fn=worker_init_fn,
        )

        criterions = [
            {"loss": nn.CrossEntropyLoss(), "weight": 0},
            {"loss": nn.CrossEntropyLoss(), "weight": 0},
            {"loss": VICRegLoss(batch_size, num_features=4096), "weight": 1},
        ]

        all_train_losses_log = [[] for i in range(len(criterions))]
        all_valid_losses_log = [[] for i in range(len(criterions))]
        all_accuracies_f = []
        all_accuracies_bf = []
        best_accuracy_f = -np.inf
        best_accuracy_bf = -np.inf

        for epoch in range(1, self.epochs + 1):
            train_sampler.set_epoch(epoch)
            if epoch > 250:
                criterions = [
                    {"loss": nn.CrossEntropyLoss(), "weight": epoch / 500},
                    {"loss": nn.CrossEntropyLoss(), "weight": epoch / 500},
                    {"loss": VICRegLoss(batch_size, num_features=4096), "weight": 1},
                ]
            train_losses = self._train_epoch(train_dataloader, optimizer, criterions, epoch)
            # train_scheduler.step()
            if rank == 0:
                for i in range(len(all_train_losses_log)):
                    all_train_losses_log[i].extend(train_losses[i])

                df_train = pd.DataFrame(all_train_losses_log).transpose()
                df_train.columns = [
                    x["loss"].__class__.__name__ + f"_{str(i+1)}" for i, x in enumerate(criterions)
                ]
                self.plot_losses(df_train)
                df_train.to_csv(os.path.join(self.save_folder, "losses_train.csv"))

            if (epoch > 250) and (epoch % 10) == 0:
                valid_losses, epoch_accuracy_f, epoch_accuracy_bf = self._valid_epoch(
                    valid_dataloader, criterions, epoch
                )
                if rank == 0:
                    all_accuracies_f.append(epoch_accuracy_f)
                    df_acc = pd.DataFrame({"accuracies": all_accuracies_f})
                    df_acc.to_csv(os.path.join(self.save_folder, "acc_valid_f.csv"))

                    all_accuracies_bf.append(epoch_accuracy_bf)
                    df_acc = pd.DataFrame({"accuracies": all_accuracies_bf})
                    df_acc.to_csv(os.path.join(self.save_folder, "acc_valid_bf.csv"))

                    for i in range(len(all_valid_losses_log)):
                        all_valid_losses_log[i].extend(valid_losses[i])
                    df_valid = pd.DataFrame(all_valid_losses_log).transpose()
                    df_valid.columns = [
                        x["loss"].__class__.__name__ + f"_{str(i+1)}"
                        for i, x in enumerate(criterions)
                    ]
                    df_valid.to_csv(os.path.join(self.save_folder, "losses_valid.csv"))
                    self.plot_losses(df_valid, train=False)

                    if epoch_accuracy_f > best_accuracy_f:
                        best_accuracy_f = epoch_accuracy_f
                        torch.save(
                            {
                                "model_state_dict": self.model_f.state_dict(),
                                "epoch": epoch,
                                "epoch_accuracy": epoch_accuracy_f,
                            },
                            os.path.join(self.save_folder, f"model_best_accuracy_f.pth"),
                        )
                    if epoch_accuracy_bf > best_accuracy_bf:
                        best_accuracy_bf = epoch_accuracy_bf
                        torch.save(
                            {
                                "model_state_dict": self.model_bf.state_dict(),
                                "epoch": epoch,
                                "epoch_accuracy": epoch_accuracy_bf,
                            },
                            os.path.join(self.save_folder, f"model_best_accuracy_bf.pth"),
                        )
        cleanup()
        return self.save_folder

    def plot_losses(self, df, train=True):
        idx = np.arange(0, len(df), 1)
        for i in list(df):
            df[i + "_1"] = df[i].rolling(50).mean()
            sns.lineplot(x=idx, y=i + "_1", data=df, legend="brief", label=str(i))
            # plt.ylim((0, 2))
        if train:
            plt.savefig(os.path.join(self.save_folder, f"train_losses.png"), dpi=300)
        else:
            plt.savefig(os.path.join(self.save_folder, f"valid_losses.png"), dpi=300)
        plt.close()

    def _train_epoch(self, dataloader, optimizer, criterions, epoch):
        self.model_f.train()
        self.model_bf.train()
        self.projector_f.train()
        self.projector_bf.train()

        total_loss = 0.0
        total_losses = [0.0 for i in range(len(criterions))]

        running_total_loss = 0.0
        running_losses = [0.0 for i in range(len(criterions))]

        loss_log = [[] for i in range(len(criterions))]

        for batch_idx, sample in enumerate(dataloader):
            f_in = sample[0].cuda().to(non_blocking=True)
            bf_in = sample[1].cuda().to(non_blocking=True)
            target = sample[2].cuda().to(non_blocking=True)

            lr = self.adjust_learning_rate(
                optimizer, dataloader, batch_idx + epoch * len(dataloader)
            )
            print(lr)
            optimizer.zero_grad()

            f_op, f_feat = self.model_f(f_in)
            bf_op, bf_feat = self.model_bf(bf_in)

            losses = []
            loss = 0
            for i, criterion in enumerate(criterions):
                if i == 0:
                    loss_class = criterion["loss"](f_op, target)
                elif i == 1:
                    loss_class = criterion["loss"](bf_op, target)
                elif i == 2:
                    loss_class = criterion["loss"](
                        self.projector_f(F.adaptive_avg_pool2d(f_feat, (1, 1)).flatten(1)),
                        self.projector_bf(F.adaptive_avg_pool2d(bf_feat, (1, 1)).flatten(1)),
                    )
                running_losses[i] += loss_class.item()
                total_losses[i] += loss_class.item()
                loss_log[i].append(criterion["weight"] * loss_class.item())
                losses.append(loss_class.item())
                loss += criterion["weight"] * loss_class

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            running_total_loss += loss.item()

        return loss_log

    def _valid_epoch(self, dataloader, criterions, epoch):
        self.model_f.eval()
        self.model_bf.eval()
        self.projector_f.eval()
        self.projector_bf.eval()

        total_loss = 0.0

        outputs_f = []
        outputs_bf = []
        targets = []

        loss_log = [[] for i in range(len(criterions))]
        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                f_in = sample[0].cuda().to(non_blocking=True)
                bf_in = sample[1].cuda().to(non_blocking=True)
                target = sample[2].cuda().to(non_blocking=True)

                f_op, _ = self.model_f(f_in)
                bf_op, _ = self.model_bf(bf_in)

                _, preds_f = torch.max(f_op, 1)
                _, preds_bf = torch.max(bf_op, 1)

                outputs_f.append(preds_f.detach())
                outputs_bf.append(preds_bf.detach())
                targets.append(target.detach())

                losses = []
                loss = 0
                for i, criterion in enumerate(criterions):
                    if i == 0:
                        loss_class = criterion["loss"](f_op, target)
                        loss_log[i].append(loss_class.item())
                        losses.append(loss_class.item())
                        loss += criterion["weight"] * loss_class
                    elif i == 1:
                        loss_class = criterion["loss"](bf_op, target)
                        loss_log[i].append(loss_class.item())
                        losses.append(loss_class.item())
                        loss += criterion["weight"] * loss_class

                total_loss += loss.item()

        outputs_f = torch.cat(outputs_f).cpu().numpy()
        outputs_bf = torch.cat(outputs_bf).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

        accuracy_f = balanced_accuracy_score(targets, outputs_f)
        accuracy_bf = balanced_accuracy_score(targets, outputs_bf)
        return loss_log, accuracy_f, accuracy_bf

    # def adjust_learning_rate(self, optimizer, epoch):
    #     lr = self.lr * (0.1 ** (epoch // self.lr_epochs))
    #     # print(f"Learning rate: {lr}")
    #     for param_group in optimizer.param_groups:
    #         param_group["lr"] = lr

    def adjust_learning_rate(self, optimizer, loader, step):
        max_steps = self.epochs * len(loader)
        warmup_steps = 10 * len(loader)
        base_lr = self.lr
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            step -= warmup_steps
            max_steps -= warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1 - q)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr


def worker_init_fn(worker_id):
    base_seed = int(torch.randint(2 ** 32, (1,)).item())
    lib_seed = (base_seed + worker_id) % (2 ** 32)
    os.environ["PYTHONHASHSEED"] = str(lib_seed)
    random.seed(lib_seed)
    np.random.seed(lib_seed)
