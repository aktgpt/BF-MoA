import os
from pickletools import optimize
import random
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision
from numpy.lib.shape_base import apply_along_axis
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from torch import nn, optim
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.multiprocessing import Lock
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from utils.metrics import get_metrics
from utils.multiprocess_utils import cleanup, setup


class BaseDistTrainer:
    def __init__(self, config, save_folder):
        self.config = config
        self.epochs = config["epochs"]
        try:
            self.multilabel = config["multilabel"]
        except:
            self.multilabel = False
        try:
            self.balanced = config["balanced"]
        except:
            self.balanced = False

        self.save_folder = save_folder

        try:
            self.checkpoint = torch.load(os.path.join(save_folder, config["ckpt_path"]))
        except:
            self.checkpoint = None

        self.image_folder = os.path.join(self.save_folder, "train_images")

        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        self.lr = config["init_lr"]
        self.wd = config["wd"]

    def train(self, train_dataset, valid_dataloader, model):
        world_size = torch.cuda.device_count()

        print(
            "We have available ",
            torch.cuda.device_count(),
            "GPUs! and using ",
            world_size,
            " GPUs",
        )

        mp.spawn(
            self.main_worker,
            args=(world_size, train_dataset, valid_dataloader, model),
            nprocs=world_size,
            join=True,
        )

    def main_worker(self, rank, world_size, train_dataset, valid_dataloader, model):
        setup(rank, world_size)

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        torch.cuda.set_device(rank)
        model = model.to(rank)

        self.model = DDP(model, device_ids=[rank])  # , find_unused_parameters=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        if self.balanced:
            _, counts = np.unique(train_dataset.labels, return_counts=True)
            weights = torch.tensor(np.sum(counts) / counts).float().cuda()
            print("Class Weights: ", weights)
        else:
            weights = None

        if self.multilabel:
            criterions = [
                {"loss": nn.MultiLabelSoftMarginLoss().cuda(rank), "weight": 1}
            ]
        else:
            criterions = [
                {"loss": nn.CrossEntropyLoss(weight=weights).cuda(rank), "weight": 1}
            ]

        batch_size = int(self.config["batch_size"] / world_size)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=int((16 + world_size - 1) / world_size),
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            sampler=train_sampler,
            worker_init_fn=worker_init_fn,
        )

        all_train_losses_log = [[] for i in range(len(criterions))]
        all_valid_losses_log = [[] for i in range(len(criterions))]
        best_accuracy = -np.inf
        all_accuracies = []

        if self.checkpoint:
            optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
            start_epoch = self.checkpoint["epoch"]
            print(f"Loading checkpoint from {start_epoch}")
        else:
            start_epoch = 1

        for epoch in range(start_epoch, self.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_losses = self._train_epoch(
                train_dataloader, optimizer, criterions, epoch, rank
            )
            # train_scheduler.step()
            if rank == 0:
                for i in range(len(all_train_losses_log)):
                    all_train_losses_log[i].extend(train_losses[i])

                    df_train = pd.DataFrame(all_train_losses_log).transpose()
                    df_train.columns = [
                        x["loss"].__class__.__name__ for x in criterions
                    ]
                    if epoch % 5 == 0:
                        self.plot_losses(df_train)
                    df_train.to_csv(
                        os.path.join(
                            self.save_folder,
                            "ckpt_losses_train.csv"
                            if self.checkpoint
                            else "losses_train.csv",
                        )
                    )
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "epoch": epoch,
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        os.path.join(self.save_folder, f"model_ckpt.pth"),
                    )
            if epoch % 5 == 0:
                valid_losses, epoch_accuracy = self._valid_epoch(
                    valid_dataloader, criterions, epoch, rank
                )
                if rank == 0:
                    all_accuracies.append(epoch_accuracy)
                    df_acc = pd.DataFrame({"accuracies": all_accuracies})
                    df_acc.to_csv(
                        os.path.join(
                            self.save_folder,
                            "ckpt_acc_valid.csv"
                            if self.checkpoint
                            else "acc_valid.csv",
                        )
                    )

                    for i in range(len(all_valid_losses_log)):
                        all_valid_losses_log[i].extend(valid_losses[i])

                    df_valid = pd.DataFrame(all_valid_losses_log).transpose()
                    df_valid.columns = [
                        x["loss"].__class__.__name__ for x in criterions
                    ]
                    self.plot_losses(df_valid, train=False)
                    df_valid.to_csv(
                        os.path.join(
                            self.save_folder,
                            "ckpt_losses_valid.csv"
                            if self.checkpoint
                            else "losses_valid.csv",
                        )
                    )
                    if epoch_accuracy > best_accuracy:
                        best_accuracy = epoch_accuracy
                        # if rank % world_size == 0:
                        torch.save(
                            {
                                "model_state_dict": self.model.state_dict(),
                                "epoch": epoch,
                                "epoch_accuracy": epoch_accuracy,
                            },
                            os.path.join(self.save_folder, f"model_best_accuracy.pth"),
                        )
        cleanup()
        return self.save_folder

    def plot_losses(self, df, train=True):
        df.clip(lower=0, upper=5, inplace=True)
        idx = np.arange(0, len(df), 1)
        for i in list(df):
            df[i + "_1"] = df[i].rolling(100).mean()
            sns.lineplot(x=idx, y=i + "_1", data=df, legend="brief", label=str(i))
        if train:
            plt.savefig(
                os.path.join(
                    self.save_folder,
                    "ckpt_train_losses.png" if self.checkpoint else "train_losses.png",
                ),
                dpi=300,
            )
        else:
            plt.savefig(
                os.path.join(
                    self.save_folder,
                    "ckpt_valid_losses.png" if self.checkpoint else "valid_losses.png",
                ),
                dpi=300,
            )
        plt.close()

    def _train_epoch(self, dataloader, optimizer, criterions, epoch, rank):
        self.model.train()

        total_loss = 0.0
        total_losses = [0.0 for i in range(len(criterions))]

        running_total_loss = 0.0
        running_losses = [0.0 for i in range(len(criterions))]

        loss_log = [[] for i in range(len(criterions))]

        for batch_idx, sample in enumerate(dataloader):
            input = sample[0].cuda(rank).to(non_blocking=True)
            target = sample[1].cuda(rank).to(non_blocking=True)

            lr = self.adjust_learning_rate(
                optimizer, dataloader, batch_idx + epoch * len(dataloader)
            )
            print(lr)

            for param in self.model.parameters():
                param.grad = None

            output, _ = self.model(input)
            losses = []
            loss = 0
            for i, criterion in enumerate(criterions):
                loss_class = criterion["loss"](output, target)
                running_losses[i] += loss_class.item()
                total_losses[i] += loss_class.item()
                loss_log[i].append(loss_class.item())
                losses.append(loss_class.item())
                loss += criterion["weight"] * loss_class

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            running_total_loss += loss.item()

        return loss_log

    def _valid_epoch(self, dataloader, criterions, epoch, rank):
        self.model.eval()

        outputs = []
        targets = []

        loss_log = [[] for i in range(len(criterions))]
        with torch.no_grad():
            for _, sample in enumerate(dataloader):
                input = sample[0].cuda(rank).to(non_blocking=True)
                target = sample[1].cuda(rank).to(non_blocking=True)

                output, _ = self.model(input)

                if self.multilabel:
                    preds = (torch.sigmoid(output) > 0.5).float()
                else:
                    _, preds = torch.max(output, 1)
                outputs.append(preds.detach())
                targets.append(target.detach())

                losses = []
                loss = 0
                for i, criterion in enumerate(criterions):
                    loss_class = criterion["loss"](output, target)
                    loss_log[i].append(loss_class.item())
                    losses.append(loss_class.item())
                    loss += criterion["weight"] * loss_class

        outputs = torch.cat(outputs).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

        if self.multilabel:
            accuracies, f1s, mean_accuracy, mean_f1 = get_metrics(outputs, targets)
            return loss_log, mean_accuracy
        else:
            if self.balanced:
                accuracy = balanced_accuracy_score(targets, outputs)
            else:
                accuracy = accuracy_score(targets, outputs)

            return loss_log, accuracy

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
