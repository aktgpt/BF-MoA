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


class DualAttnDistTrainer:
    def __init__(self, config, save_folder):
        self.config = config
        self.epochs = config["epochs"]

        self.save_folder = save_folder
        self.image_folder_f = os.path.join(self.save_folder, "train_images_f")
        self.image_folder_bf = os.path.join(self.save_folder, "train_images_bf")

        if not os.path.exists(self.image_folder_f):
            os.makedirs(self.image_folder_f)
        if not os.path.exists(self.image_folder_bf):
            os.makedirs(self.image_folder_bf)

        self.lr = config["init_lr"]
        self.wd = config["wd"]
        self.milestones = config["milestones"]

    def train(self, train_dataset, valid_dataset, model):
        world_size = torch.cuda.device_count()
        mp.spawn(
            self.main_worker,
            args=(world_size, train_dataset, valid_dataset, model),
            nprocs=world_size,
            join=True,
        )

    def main_worker(self, rank, world_size, train_dataset, valid_dataset, models):
        setup(rank, world_size)

        model_f = torch.nn.SyncBatchNorm.convert_sync_batchnorm(models[0])
        model_bf = torch.nn.SyncBatchNorm.convert_sync_batchnorm(models[1])

        batch_size = int(self.config["batch_size"] / world_size)

        torch.cuda.set_device(rank)

        self.model_f = DDP(model_f.to(rank), device_ids=[rank])
        self.model_bf = DDP(model_bf.to(rank), device_ids=[rank])

        optimizer = optim.SGD(
            list(self.model_f.parameters()) + list(self.model_bf.parameters()),
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
            num_workers=int((16 + world_size - 1) / world_size),
            prefetch_factor=8,
            persistent_workers=True,
            pin_memory=True,
            sampler=train_sampler,
            worker_init_fn=worker_init_fn,
        )

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=int((8 + world_size - 1) / world_size),
            prefetch_factor=8,
            persistent_workers=True,
            pin_memory=True,
        )

        criterions = [
            {"loss": nn.CrossEntropyLoss(), "weight": 1},
            {"loss": nn.CrossEntropyLoss(), "weight": 1}, 
            {"loss": nn.CrossEntropyLoss(), "weight": 1},
            {"loss": nn.CrossEntropyLoss(), "weight": 1},
            {"loss": nn.L1Loss(), "weight": 5},
        ]

        all_train_losses_log = [[] for i in range(len(criterions))]
        all_valid_losses_log = [[] for i in range(len(criterions))]
        all_accuracies_f = []
        all_accuracies_bf = []
        best_accuracy_f = -np.inf
        best_accuracy_bf = -np.inf

        for epoch in range(1, self.epochs + 1):
            train_sampler.set_epoch(epoch)

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

            if epoch % 5 == 0:
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
                    self.plot_losses(df_valid, train=False)
                    df_valid.to_csv(os.path.join(self.save_folder, "losses_valid.csv"))

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
        df.clip(lower=0, upper=5, inplace=True)
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

            for param in self.model_f.parameters():
                param.grad = None
            for param in self.model_bf.parameters():
                param.grad = None

            op_f, cam_op_f, attn_f = self.model_f(f_in)
            op_bf, cam_op_bf, attn_bf = self.model_bf(bf_in)

            losses = []
            loss = 0
            for i, criterion in enumerate(criterions):
                if i == 0:
                    loss_class = criterion["loss"](op_f, target)
                elif i == 1:
                    loss_class = criterion["loss"](cam_op_f, target)
                elif i == 2:
                    loss_class = criterion["loss"](op_bf, target)
                elif i == 3:
                    loss_class = criterion["loss"](cam_op_bf, target)
                elif i == 4:
                    loss_class = criterion["loss"](attn_bf, attn_f.detach())
                running_losses[i] += loss_class.item()
                total_losses[i] += loss_class.item()
                loss_log[i].append(criterion["weight"] * loss_class.item())
                losses.append(loss_class.item())
                loss += criterion["weight"] * loss_class

            nn.utils.clip_grad_norm_(self.model_f.parameters(), 1)
            nn.utils.clip_grad_norm_(self.model_bf.parameters(), 1)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            running_total_loss += loss.item()

        return loss_log

    def _valid_epoch(self, dataloader, criterions, epoch):
        self.model_f.eval()
        self.model_bf.eval()

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

                op_f, cam_op_f, attn_f = self.model_f(f_in)
                op_bf, cam_op_bf, attn_bf = self.model_bf(bf_in)

                if batch_idx == 0:
                    save_attn(epoch, f_in, attn_f, self.image_folder_f)
                    save_attn(epoch, bf_in, attn_bf, self.image_folder_bf)

                _, preds_f = torch.max(op_f, 1)
                _, preds_bf = torch.max(op_bf, 1)

                outputs_f.append(preds_f.detach())
                outputs_bf.append(preds_bf.detach())
                targets.append(target.detach())

                losses = []
                loss = 0
                for i, criterion in enumerate(criterions):
                    if i == 0:
                        loss_class = criterion["loss"](op_f, target)
                    elif i == 1:
                        loss_class = criterion["loss"](cam_op_f, target)
                    elif i == 2:
                        loss_class = criterion["loss"](op_bf, target)
                    elif i == 3:
                        loss_class = criterion["loss"](cam_op_bf, target)
                    elif i == 4:
                        loss_class = criterion["loss"](attn_bf, attn_f)
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


def save_attn(epoch_idx, input, attn, save_folder):
    shape = input.shape[2:]
    input = make_grid(input, normalize=True).permute(1, 2, 0).cpu().numpy()[:, :, :3]
    cv2.imwrite(
        os.path.join(save_folder, f"{epoch_idx}_input.png"),
        cv2.cvtColor((input * 255).astype("uint8"), cv2.COLOR_BGR2RGB),
    )

    attn = (
        make_grid(F.interpolate(attn, shape, mode="bilinear", align_corners=False))
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    cm = plt.get_cmap("jet")
    attn = cm(attn[:, :, 0])[:, :, :3]
    cv2.imwrite(
        os.path.join(save_folder, f"{epoch_idx}_attn.png"),
        cv2.cvtColor((attn * 255).astype("uint8"), cv2.COLOR_BGR2RGB),
    )
    a = (
        0.5
        * (
            cv2.cvtColor(
                cv2.cvtColor(
                    (input * 255).astype("uint8"),
                    cv2.COLOR_RGB2GRAY,
                ),
                cv2.COLOR_GRAY2RGB,
            )
            / 255
        )
        + 0.5 * attn
    )
    cv2.imwrite(
        os.path.join(save_folder, f"{epoch_idx}_merged.png"),
        cv2.cvtColor((a * 255).astype("uint8"), cv2.COLOR_BGR2RGB),
    )


def worker_init_fn(worker_id):
    base_seed = int(torch.randint(2 ** 32, (1,)).item())
    lib_seed = (base_seed + worker_id) % (2 ** 32)
    os.environ["PYTHONHASHSEED"] = str(lib_seed)
    random.seed(lib_seed)
    np.random.seed(lib_seed)
