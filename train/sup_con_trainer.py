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
from models.sup_con_loss import SupConLoss
from numpy.lib.shape_base import apply_along_axis
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch import nn, optim
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.multiprocessing import Lock
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from utils.metrics import get_metrics
from utils.multiprocess_utils import cleanup, setup

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
    "dmso",
]
moas = np.sort(moas)


def make_weights_for_balanced_classes(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    weight_per_class = np.sum(counts.astype(float)) / counts.astype(float)
    weights = [0] * len(labels)
    for i, val in enumerate(labels):
        weights[i] = weight_per_class[np.where(unique_labels == val)[0][0]]
    return weights


class MyDataParallel(nn.DataParallel):
    def __init__(self, module, my_methods=["forward"], device_ids=None, output_device=None, dim=0):
        self._mymethods = my_methods
        super().__init__(module, device_ids, output_device, dim)

    def __getattr__(self, name):
        if name in self._mymethods:
            return getattr(self.module, name)
        else:
            return super().__getattr__(name)


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
        self.supervised_lr = 1e-3
        self.wd = config["wd"]
        self.lr_schedule = config["lr_schedule"]

    def train(self, train_dataset, valid_dataloader, model):
        world_size = torch.cuda.device_count()

        self.model = MyDataParallel(
            model.cuda(),
            my_methods=["forward_features", "fc", "layer4"],
            device_ids=[torch.device(f"cuda:{i}") for i in range(world_size)],
        )

        self.projection_head = MyDataParallel(
            nn.Sequential(
                nn.Linear(model.inplanes, model.inplanes),
                nn.ReLU(),
                nn.Linear(model.inplanes, 512),
            ).cuda(),
            device_ids=[torch.device(f"cuda:{i}") for i in range(world_size)],
        )

        optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.projection_head.parameters()), lr=self.lr
        )
        optimizer_cls = optim.AdamW(
            [
                {"params": self.model.fc.parameters(), "lr": self.supervised_lr},
                # {"params": self.model.layer4.parameters(), "lr": self.supervised_lr * 0.01},
            ]
        )

        criterions = [{"loss": SupConLoss(temperature=0.5).cuda(), "weight": 1}]
        criterions_cls = [{"loss": nn.CrossEntropyLoss().cuda(), "weight": 1}]

        batch_size = int(self.config["batch_size"])

        weights = make_weights_for_balanced_classes(
            train_dataset.labels
            if self.config["label_type"] == "moa"
            else train_dataset.comp_labels
        )
        print("Sample Weights: ", weights)
        sampler = WeightedRandomSampler(weights, len(weights))
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=int((16 + world_size - 1) / world_size),
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )

        all_train_losses_log = [[] for i in range(len(criterions))]

        if self.checkpoint:
            start_epoch = self.checkpoint["epoch"] + 1

            train_df = pd.read_csv(os.path.join(self.save_folder, "losses_train.csv"))
            all_train_losses_log = [
                train_df[x["loss"].__class__.__name__].to_list() for x in criterions
            ]
            train_cls_df = pd.read_csv(os.path.join(self.save_folder, "losses_cls_train.csv"))
            all_train_losses_log = [
                train_cls_df[x["loss"].__class__.__name__].to_list() for x in criterions_cls
            ]

            valid_df = pd.read_csv(os.path.join(self.save_folder, "losses_valid.csv"))
            all_valid_losses_log = [
                valid_df[x["loss"].__class__.__name__].to_list() for x in criterions
            ]
            best_accuracy = np.max(all_accuracies)
            all_accuracies = pd.read_csv(os.path.join(self.save_folder, "acc_valid.csv"))[
                "accuracies"
            ].tolist()

            optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
            print(f"Loading checkpoint from {start_epoch}")
        else:
            start_epoch = 1
            all_train_losses_log = [[] for i in range(len(criterions))]
            all_train_cls_losses_log = [[] for i in range(len(criterions))]

            all_valid_losses_log = [[] for i in range(len(criterions))]
            best_accuracy = -np.inf
            all_accuracies = []

        for epoch in range(start_epoch, self.epochs + 1):
            train_losses = self._train_epoch(train_dataloader, optimizer, criterions, epoch)
            for i in range(len(all_train_losses_log)):
                all_train_losses_log[i].extend(train_losses[i])

            df_train = pd.DataFrame(all_train_losses_log).transpose()
            df_train.columns = [x["loss"].__class__.__name__ for x in criterions]
            if epoch % 5 == 0:
                self.plot_losses(df_train)
            df_train.to_csv(os.path.join(self.save_folder, "losses_train.csv"))
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(self.save_folder, f"model_ckpt.pth"),
            )
            if epoch % 10 == 0:
                train_losses = self._train_supervised_epoch(
                    train_dataloader, optimizer_cls, criterions_cls, epoch
                )
                for i in range(len(all_train_cls_losses_log)):
                    all_train_cls_losses_log[i].extend(train_losses[i])

                train_cls_df = pd.DataFrame(all_train_cls_losses_log).transpose()
                train_cls_df.columns = [x["loss"].__class__.__name__ for x in criterions_cls]
                self.plot_losses(train_cls_df, save_name="train_cls")
                train_cls_df.to_csv(os.path.join(self.save_folder, "losses_cls_train.csv"))

                valid_losses, epoch_accuracy = self._valid_epoch(
                    valid_dataloader, criterions_cls, epoch
                )
                all_accuracies.append(epoch_accuracy)
                df_acc = pd.DataFrame({"accuracies": all_accuracies})
                df_acc.to_csv(os.path.join(self.save_folder, "acc_valid.csv"))

                for i in range(len(all_valid_losses_log)):
                    all_valid_losses_log[i].extend(valid_losses[i])

                df_valid = pd.DataFrame(all_valid_losses_log).transpose()
                df_valid.columns = [x["loss"].__class__.__name__ for x in criterions]
                self.plot_losses(df_valid, save_name="valid")
                df_valid.to_csv(os.path.join(self.save_folder, "losses_valid.csv"))
                if epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
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

    def plot_losses(self, df, save_name="train"):
        df.clip(lower=0, upper=5, inplace=True)
        idx = np.arange(0, len(df), 1)
        for i in list(df):
            df[i + "_1"] = df[i].rolling(100).mean()
            sns.lineplot(x=idx, y=i + "_1", data=df, legend="brief", label=str(i))
        # if save_name == "train":
        plt.savefig(
            os.path.join(self.save_folder, save_name + "_losses.png"),
            dpi=300,
        )
        # elif:
        #     plt.savefig(
        #         os.path.join(self.save_folder, "train_losses.png"),
        #         dpi=300,
        #     )
        # else:
        #     plt.savefig(
        #         os.path.join(self.save_folder, "valid_losses.png"),
        #         dpi=300,
        #     )
        plt.close()

    def _train_epoch(self, dataloader, optimizer, criterions, epoch):
        self.model.train()

        total_loss = 0.0
        total_losses = [0.0 for i in range(len(criterions))]

        running_total_loss = 0.0
        running_losses = [0.0 for i in range(len(criterions))]

        loss_log = [[] for i in range(len(criterions))]

        for batch_idx, sample in enumerate(dataloader):
            x_i = sample[0].cuda().to(non_blocking=True)
            x_j = sample[1].cuda().to(non_blocking=True)
            if self.config["label_type"] == "moa":
                labels = sample[2].cuda().to(non_blocking=True)
            else:
                labels = (
                    torch.tensor(
                        pd.DataFrame({"comp": sample[5]}).astype("category").comp.cat.codes.values
                    )
                    .cuda()
                    .to(non_blocking=True)
                )

            if self.lr_schedule:
                lr = self.adjust_learning_rate(
                    optimizer, dataloader, batch_idx + epoch * len(dataloader)
                )
                print(epoch, batch_idx, lr)

            for param in self.model.parameters():
                param.grad = None

            z_i = self.projection_head(self.model.forward_features(x_i))
            z_j = self.projection_head(self.model.forward_features(x_j))

            losses = []
            loss = 0
            for i, criterion in enumerate(criterions):
                loss_class = criterion["loss"](z_i, z_j, labels=labels)
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

    def _train_supervised_epoch(self, dataloader, optimizer, criterions, epoch):
        self.model.train()

        total_loss = 0.0
        total_losses = [0.0 for i in range(len(criterions))]

        running_total_loss = 0.0
        running_losses = [0.0 for i in range(len(criterions))]

        loss_log = [[] for i in range(len(criterions))]

        for batch_idx, sample in enumerate(dataloader):
            input = sample[0].cuda().to(non_blocking=True)
            target = sample[2].cuda().to(non_blocking=True)

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

    def _valid_epoch(self, dataloader, criterions, epoch):
        self.model.eval()

        outputs = []
        targets = []
        feature_data = []
        compounds = []

        loss_log = [[] for i in range(len(criterions))]
        with torch.no_grad():
            for _, sample in enumerate(dataloader):
                input = sample[0].cuda().to(non_blocking=True)
                target = sample[1].cuda().to(non_blocking=True)
                compound = sample[4]

                output, feature_map = self.model(input)

                if self.multilabel:
                    preds = (torch.sigmoid(output) > 0.5).float()
                else:
                    _, preds = torch.max(output, 1)
                outputs.append(preds.detach())
                targets.append(target.detach())
                feature_data.append(feature_map)
                compounds.extend(compound)

                losses = []
                loss = 0
                for i, criterion in enumerate(criterions):
                    loss_class = criterion["loss"](output, target)
                    loss_log[i].append(loss_class.item())
                    losses.append(loss_class.item())
                    loss += criterion["weight"] * loss_class

        outputs = torch.cat(outputs).cpu().numpy()
        feature_data = torch.cat(feature_data).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

        sample_moas = []
        for target in targets:
            sample_moas.append(moas[target])

        df = pd.DataFrame({"moa": sample_moas, "fv": list(feature_data), "compound": compounds})
        trans = umap.UMAP(random_state=42, n_components=2).fit(feature_data)
        df["x"] = trans.embedding_[:, 0]
        df["y"] = trans.embedding_[:, 1]
        fig, ax = plt.subplots(1, figsize=(16, 12))
        sns.scatterplot(
            x="x",
            y="y",
            hue="moa",
            palette=sns.color_palette("Paired", 11),
            legend="brief",
            s=100,
            alpha=0.9,
            data=df,
        )
        plt.savefig(os.path.join(self.image_folder, f"embed_moa_{epoch}.png"), dpi=500)
        plt.close()
        fig, ax = plt.subplots(1, figsize=(16, 12))
        sns.scatterplot(
            x="x",
            y="y",
            hue="moa",
            style="compound",
            palette=sns.color_palette("Paired", 11),
            legend="brief",
            s=100,
            alpha=0.9,
            data=df,
        )
        plt.savefig(os.path.join(self.image_folder, f"embed_moa_comp_{epoch}.png"), dpi=500)
        plt.close()

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
    base_seed = int(torch.randint(2**32, (1,)).item())
    lib_seed = (base_seed + worker_id) % (2**32)
    os.environ["PYTHONHASHSEED"] = str(lib_seed)
    random.seed(lib_seed)
    np.random.seed(lib_seed)


# def _plot_epoch(self, dataloader, criterions, epoch):
#         self.model.eval()

#         targets = []
#         feature_data = []
#         compounds = []
#         with torch.no_grad():
#             for _, sample in enumerate(dataloader):
#                 input = sample[0].cuda().to(non_blocking=True)
#                 target = sample[2].cuda().to(non_blocking=True)
#                 compound = sample[3]

#                 _, feature_map = self.model(input)

#                 feature_data.append(feature_map)
#                 targets.append(target.detach())
#                 compounds.extend(compound)

#         feature_data = torch.cat(feature_data).cpu().numpy()
#         targets = torch.cat(targets).cpu().numpy()

#         sample_moas = []
#         for target in targets:
#             sample_moas.append(moas[target])

#         df = pd.DataFrame({"moa": sample_moas, "fv": list(feature_data), "compound": compounds})
#         trans = umap.UMAP(random_state=42, n_components=2).fit(feature_data)
#         df["x"] = trans.embedding_[:, 0]
#         df["y"] = trans.embedding_[:, 1]
#         fig, ax = plt.subplots(1, figsize=(16, 12))
#         sns.scatterplot(
#             x="x",
#             y="y",
#             hue="moa",
#             palette=sns.color_palette("Paired", 11),
#             legend="brief",
#             s=100,
#             alpha=0.9,
#             data=df,
#         )
#         plt.savefig(os.path.join(self.image_folder, f"embed_moa_{epoch}.png"), dpi=500)
#         plt.close()
#         fig, ax = plt.subplots(1, figsize=(16, 12))
#         sns.scatterplot(
#             x="x",
#             y="y",
#             hue="moa",
#             style="compound",
#             palette=sns.color_palette("Paired", 11),
#             legend="brief",
#             s=100,
#             alpha=0.9,
#             data=df,
#         )
#         plt.savefig(os.path.join(self.image_folder, f"embed_moa_comp_{epoch}.png"), dpi=500)
#         plt.close()
