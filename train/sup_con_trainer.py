import math
import os
import random
from pickletools import optimize

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision
import umap
from models.ssl_losses import Projector
from models.sup_con_loss import SupConLoss
from numpy.lib.shape_base import apply_along_axis
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch import nn, optim
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.multiprocessing import Lock
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from utils.metrics import get_metrics
from utils.multiprocess_utils import cleanup, setup

from .dist_sampler import DistributedWeightedSampler

moas = [
    "Aurora kinase inhibitor",
    "tubulin polymerization inhibitor",
    "JAK inhibitor",
    "protein synthesis inhibitor",
    "HDAC inhibitor",
    "topoisomerase inhibitor",
    "PARP inhibitor",
    "ATPase inhibitor",
    "retinoid receptor agonist",
    "HSP inhibitor",
]


class SupConDistTrainer:
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
        projector = nn.Sequential(
            nn.Linear(2048, 2048), nn.ReLU(inplace=False), nn.Linear(2048, 256)
        )  # torch.nn.SyncBatchNorm.convert_sync_batchnorm(Projector("2048-256", 2048))

        torch.cuda.set_device(rank)
        model = model.to(rank)
        projector = projector.to(rank)

        self.model = DDP(model, device_ids=[rank])
        self.projector = DDP(projector.to(rank), device_ids=[rank])

        optimizer = optim.SGD(
            list(self.model.parameters()) + list(self.projector.parameters()),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.wd,
        )

        criterions = [{"loss": SupConLoss().cuda(rank), "weight": 1}]

        batch_size = int(self.config["batch_size"] / world_size)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42
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

        all_train_losses_log = [[] for i in range(len(criterions))]

        if self.checkpoint:
            optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
            start_epoch = self.checkpoint["epoch"]
        else:
            start_epoch = 1

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(start_epoch, self.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_losses = self._train_epoch(train_dataloader, optimizer, criterions, epoch, rank)

            if rank == 0:
                for i in range(len(all_train_losses_log)):
                    all_train_losses_log[i].extend(train_losses[i])

                    df_train = pd.DataFrame(all_train_losses_log).transpose()
                    df_train.columns = [x["loss"].__class__.__name__ for x in criterions]
                    self.plot_losses(df_train)
                    df_train.to_csv(
                        os.path.join(
                            self.save_folder,
                            "ckpt_losses_train.csv" if self.checkpoint else "losses_train.csv",
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
            if epoch % 1 == 0:
                self._valid_epoch(train_dataloader, criterions, epoch, rank)

        cleanup()
        return self.save_folder

    def plot_losses(self, df, train=True):
        # df.clip(lower=0, upper=5, inplace=True)
        idx = np.arange(0, len(df), 1)
        for i in list(df):
            df[i + "_1"] = df[i].rolling(50).mean()
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
        self.projector.train()

        total_loss = 0.0
        total_losses = [0.0 for i in range(len(criterions))]

        running_total_loss = 0.0
        running_losses = [0.0 for i in range(len(criterions))]

        loss_log = [[] for i in range(len(criterions))]

        for batch_idx, sample in enumerate(dataloader):
            input1 = sample[0].cuda(rank).to(non_blocking=True)
            input2 = sample[1].cuda(rank).to(non_blocking=True)
            target = sample[3]

            uniques, counts = np.unique(list(target), return_counts=True)
            target = (
                torch.tensor([np.where(comp == uniques)[0][0] for comp in target])
                .cuda(rank)
                .to(non_blocking=True)
            )

            lr = self.adjust_learning_rate(
                optimizer, dataloader, batch_idx + epoch * len(dataloader)
            )
            print(counts, target)
            print(lr)

            for param in self.model.parameters():
                param.grad = None

            feat1 = self.projector(self.model(input1))
            feat2 = self.projector(self.model(input2))

            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)

            feat = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)

            losses = []
            loss = 0
            for i, criterion in enumerate(criterions):
                loss_class = criterion["loss"](feat, target)
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

        targets = []
        feature_data = []
        with torch.no_grad():
            for _, sample in enumerate(dataloader):
                input = sample[0].cuda(rank).to(non_blocking=True)
                target = sample[2].cuda(rank).to(non_blocking=True)

                feature_map = self.model(input)

                feature_data.append(feature_map)
                targets.append(target.detach())

        feature_data = torch.cat(feature_data).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

        sample_moas = []
        for target in targets:
            sample_moas.append(moas[target])

        df = pd.DataFrame(
            {
                "moa": sample_moas,
                "fv": list(feature_data),
            }
        )
        trans = umap.UMAP(random_state=42, n_components=2).fit(feature_data)
        df["x"] = trans.embedding_[:, 0]
        df["y"] = trans.embedding_[:, 1]
        fig, ax = plt.subplots(1, figsize=(16, 12))
        sns.scatterplot(
            x="x",
            y="y",
            hue="moa",
            palette="deep",
            legend="brief",
            s=100,
            alpha=0.9,
            data=df,
        )
        plt.savefig(os.path.join(self.save_folder, f"embed_moa_{epoch}.png"), dpi=500)
        plt.close()

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
