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
from models.ssl_losses import BTLoss, Projector, VICRegLoss
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


class SSLDistTrainer:
    def __init__(self, config, save_folder):
        self.config = config
        self.epochs = config["epochs"]
        self.supervised_epochs = 1

        self.save_folder = save_folder
        self.image_folder = os.path.join(self.save_folder, "train_images")

        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        self.lr = config["init_lr"]
        self.wd = config["wd"]

    def train(self, train_dataset, valid_dataloader, model):
        world_size = torch.cuda.device_count()
        mp.spawn(
            self.main_worker,
            args=(world_size, train_dataset, valid_dataloader, model),
            nprocs=world_size,
            join=True,
        )

    def main_worker(self, rank, world_size, train_dataset, valid_dataloader, model):
        setup(rank, world_size)

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        projector = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Projector("8196-8196-8196", 2048))

        batch_size = int(self.config["batch_size"] / world_size)
        self.batch_size = self.config["batch_size"]

        torch.cuda.set_device(rank)

        self.model = DDP(model.to(rank), device_ids=[rank])
        self.projector = DDP(projector.to(rank), device_ids=[rank])

        optimizer = optim.SGD(
            list(self.model.parameters()) + list(self.projector.parameters()),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.wd,
        )

        self.head = nn.Linear(2048, 10)
        self.head.weight.data.normal_(mean=0.0, std=0.01)
        self.head.bias.data.zero_()
        supervised_model = nn.Sequential(self.model, self.head)
        self.supervised_model = DDP(supervised_model.to(rank), device_ids=[rank])

        optimizer_supervised = optim.SGD(
            self.head.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.wd,
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=int((32 + world_size - 1) / world_size),
            prefetch_factor=8,
            persistent_workers=True,
            pin_memory=True,
            sampler=train_sampler,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )

        criterion_ssl = BTLoss(batch_size, num_features=8196).cuda()
        criterion_supervised = nn.CrossEntropyLoss()

        all_ssl_losses_log = []
        all_supervised_losses_log = []
        all_valid_losses_log = []
        all_accuracies = []
        best_accuracy = -np.inf

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(1, self.epochs + 1):
            # if epoch > 100:
            #     criterion_ssl = VICRegLoss(batch_size, num_features=8196, cov_coeff=1e-6 * epoch)
            train_sampler.set_epoch(epoch)
            train_losses = self._train_epoch(
                train_dataloader, optimizer, criterion_ssl, epoch, rank
            )
            if rank == 0:
                all_ssl_losses_log.extend(train_losses)
                df_train = pd.DataFrame({criterion_ssl.__class__.__name__: all_ssl_losses_log})

                self.plot_losses(df_train)
                df_train.to_csv(os.path.join(self.save_folder, "losses_ssl_train.csv"))
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(self.save_folder, f"ssl_model.pth"),
                )

            if (epoch > 1000) and (epoch % 20) == 0:
                for epoch_supervised in range(1, self.supervised_epochs + 1):
                    train_losses = self._supervised_train(
                        train_dataloader,
                        optimizer_supervised,
                        criterion_supervised,
                        epoch_supervised,
                        rank,
                    )
                    if rank == 0:
                        all_supervised_losses_log.extend(train_losses)
                        df_train = pd.DataFrame(
                            {criterion_supervised.__class__.__name__: all_supervised_losses_log}
                        )  # .transpose()
                        # df_train.columns = [criterion_supervised.__class__.__name__]
                        self.plot_losses(df_train, name="train_supervised_losses.png")
                        df_train.to_csv(
                            os.path.join(self.save_folder, "losses_supervised_train.csv")
                        )

                valid_losses, epoch_accuracy = self._valid_epoch(
                    valid_dataloader, criterion_supervised, epoch
                )
                if rank == 0:
                    all_accuracies.append(epoch_accuracy)
                    df_acc = pd.DataFrame({"accuracies": all_accuracies})
                    df_acc.to_csv(os.path.join(self.save_folder, "acc_valid.csv"))

                    all_valid_losses_log.extend(valid_losses)

                    df_valid = pd.DataFrame(
                        {criterion_supervised.__class__.__name__: all_valid_losses_log}
                    )  # .transpose()
                    # df_valid.columns = [criterion_supervised.__class__.__name__]
                    self.plot_losses(df_valid, name="valid_losses.png")
                    df_valid.to_csv(os.path.join(self.save_folder, "losses_valid.csv"))
                    if epoch_accuracy > best_accuracy:
                        best_accuracy = epoch_accuracy
                        torch.save(
                            {
                                "model_state_dict": self.supervised_model.state_dict(),
                                "epoch": epoch,
                                "epoch_accuracy": epoch_accuracy,
                            },
                            os.path.join(self.save_folder, f"model_best_accuracy.pth"),
                        )
        cleanup()
        return self.supervised_model

    def plot_losses(self, df, name="train_losses.png"):
        idx = np.arange(0, len(df), 1)
        for i in list(df):
            df[i + "_1"] = df[i].rolling(50).mean()
            sns.lineplot(x=idx, y=i + "_1", data=df, legend="brief", label=str(i))
            # plt.ylim((0, 2))

        plt.savefig(os.path.join(self.save_folder, name), dpi=300)
        # else:
        #     plt.savefig(os.path.join(self.save_folder, f"valid_losses.png"), dpi=300)
        plt.close()

    def _train_epoch(self, dataloader, optimizer, criterion, epoch, rank):
        self.model.train()
        self.projector.train()

        loss_log = []

        for batch_idx, sample in enumerate(dataloader):
            input1 = sample[0].cuda(rank).to(non_blocking=True)
            input2 = sample[1].cuda(rank).to(non_blocking=True)

            lr = adjust_learning_rate(
                optimizer,
                dataloader,
                batch_idx + epoch * len(dataloader),
                self.lr,
                self.batch_size,
                self.epochs,
            )
            print(lr)
            for param in self.model.parameters():
                param.grad = None

            proj1 = self.projector(self.model(input1))
            proj2 = self.projector(self.model(input2))

            # print(proj1.shape, torch.max(proj1).item(), torch.max(proj2).item())
            loss = criterion(proj1, proj2)

            loss_log.append(loss.item())

            loss.backward()
            optimizer.step()

        return loss_log

    def _supervised_train(self, dataloader, optimizer, criterion, epoch, rank):
        self.model.eval()
        self.head.train()

        loss_log = []

        for batch_idx, sample in enumerate(dataloader):
            input = sample[0].cuda(rank).to(non_blocking=True)
            target = sample[2].cuda(rank).to(non_blocking=True)

            lr = adjust_learning_rate(
                optimizer,
                dataloader,
                batch_idx + epoch * len(dataloader),
                self.lr,
                self.batch_size,
                self.supervised_epochs,
            )

            optimizer.zero_grad()

            op = self.supervised_model(input)

            loss = criterion(op, target)
            loss_log.append(loss.item())

            loss.backward()
            optimizer.step()

        return loss_log

    def _valid_epoch(self, dataloader, criterion, epoch):
        self.supervised_model.eval()

        outputs = []
        targets = []

        loss_log = []
        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                input = sample[0].cuda().to(non_blocking=True)
                target = sample[1].cuda().to(non_blocking=True)

                op = self.supervised_model(input)

                _, preds = torch.max(op, 1)

                outputs.append(preds.detach())
                targets.append(target.detach())

                loss = criterion(op, target)
                loss_log.append(loss.item())

        outputs = torch.cat(outputs).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

        accuracy = balanced_accuracy_score(targets, outputs)
        return loss_log, accuracy


def adjust_learning_rate(optimizer, loader, step, base_lr, batch_size, total_epochs):
    max_steps = total_epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = base_lr * batch_size / 256

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
