import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import utils.colormaps as cmaps
from numpy.lib.shape_base import apply_along_axis
from PIL import Image
from torch import nn, optim
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from utils.metrics import get_metrics
from torch.multiprocessing import Lock
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.decomposition import NMF, PCA
import umap

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

site_conversion = pd.DataFrame(
    {"bf_sites": ["s1", "s2", "s3", "s4", "s5"], "f_sites": ["s2", "s4", "s5", "s6", "s8"]}
)


class BaseTester:
    def __init__(self, config, save_folder):
        self.config = config
        try:
            self.multilabel = config["multilabel"]
        except:
            self.multilabel = False

        try:
            self.scale = config["scale"]
        except:
            self.scale = None

        self.save_folder = save_folder
        self.image_folder = os.path.join(
            self.save_folder,
            f"test_images_{self.scale}x{self.scale}" if self.scale else "test_images",
        )
        self.plot_folder = os.path.join(self.save_folder, "plots")

        self.nmf_folder = os.path.join(
            self.save_folder,
            f"nmf_images_{self.scale}x{self.scale}" if self.scale else "nmf_images",
        )

        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
        if not os.path.exists(self.nmf_folder):
            os.makedirs(self.nmf_folder)
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

        self.model_checkpoint = torch.load(os.path.join(save_folder, config["model_path"]))

    def test(self, test_dataloader, model):
        world_size = torch.cuda.device_count()
        if world_size > 1:
            self.model = nn.DataParallel(model.cuda())
        else:
            self.model = nn.DataParallel(model.cuda())  # model.cuda()

        self.model.load_state_dict(
            self.model_checkpoint["model_state_dict"],
        )
        best_train_epoch = self.model_checkpoint["epoch"]
        best_train_accuracy = self.model_checkpoint["epoch_accuracy"]
        print(f"Loading model with Epoch:{best_train_epoch},Accuracy:{best_train_accuracy}")

        accuracy = self._test_epoch(test_dataloader)

        if self.multilabel:
            df_acc = pd.DataFrame({"f1": [accuracy]})
            df_acc.to_csv(
                os.path.join(
                    self.save_folder, f"f1_test_{self.scale}.csv" if self.scale else "f1_test.csv"
                )
            )
        else:
            df_acc = pd.DataFrame({"accuracies": [accuracy]})
            df_acc.to_csv(
                os.path.join(
                    self.save_folder, f"acc_test_{self.scale}.csv" if self.scale else "acc_test.csv"
                )
            )

    def _test_epoch(self, dataloader):
        self.model.eval()
        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        total_loss_desc = tqdm(total=0, position=2, bar_format="{desc}")

        outputs = []
        targets = []
        plates = []
        sites = []
        compounds = []
        wells = []

        feature_data = []

        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                input = sample[0].cuda().to(non_blocking=True)
                target = sample[1].cuda().to(non_blocking=True)
                plate = sample[2]
                site = sample[3]
                compound = sample[4]
                well = sample[5]

                output, feature_map = self.model(input)

                # if batch_idx % 10 == 0:
                #     self.make_nmf(input, feature_map, batch_idx)

                if self.multilabel:
                    preds = (torch.sigmoid(output) > 0.5).float()
                else:
                    _, preds = torch.max(output, 1)

                # feat_corr_mat.append(get_feat_corr(feature_map))
                # spatial_corr_mat.append(get_spatial_corr(feature_map))
                # feature_data.append(F.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1).detach())

                feature_data.append(feature_map)

                outputs.append(preds.detach())
                targets.append(target.detach())

                plates.extend(plate)
                sites.extend(site)
                compounds.extend(compound)
                wells.extend(well)

                outer.update(1)

        outputs = torch.cat(outputs).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

        feature_data = torch.cat(feature_data).cpu().numpy()
        # self.save_correlation(feat_corr_mat, spatial_corr_mat)

        sample_moas = []
        for target in targets:
            sample_moas.append(moas[target])

        df = pd.DataFrame(
            {
                "plate": plates,
                "well": wells,
                "site": sites,
                "compound": compounds,
                "moa": sample_moas,
                "fv": list(feature_data),
            }
        )
        df.to_pickle(os.path.join(self.save_folder, "analysis"))
        self.plot_umaps(df, feature_data)
        # self.plot_umap(targets, feature_data)

        if self.multilabel:
            accuracies, f1s, mean_accuracy, mean_f1 = get_metrics(outputs, targets)
            total_loss_desc.set_description_str(f"Test Results Total Mean F1: {mean_f1:.5f}")
            return mean_f1.item()
        else:
            f1_macro = f1_score(targets, outputs, average="macro")
            pd.DataFrame([f1_macro]).to_csv(
                os.path.join(
                    self.save_folder,
                    f"f1_macro{self.scale}.csv" if self.scale else "f1_macro.csv",
                )
            )
            f1_micro = f1_score(targets, outputs, average="micro")
            pd.DataFrame([f1_micro]).to_csv(
                os.path.join(
                    self.save_folder,
                    f"f1_micro{self.scale}.csv" if self.scale else "f1_micro.csv",
                )
            )
            confusion_mat = confusion_matrix(targets, outputs)
            pd.DataFrame(confusion_mat).to_csv(
                os.path.join(
                    self.save_folder,
                    f"confusion_matrix_{self.scale}.csv" if self.scale else "confusion_matrix.csv",
                )
            )
            precision = precision_score(targets, outputs, average="macro")
            pd.DataFrame([precision]).to_csv(
                os.path.join(
                    self.save_folder,
                    f"precision_{self.scale}.csv" if self.scale else "precision.csv",
                )
            )
            recall = recall_score(targets, outputs, average="macro")
            pd.DataFrame([recall]).to_csv(
                os.path.join(
                    self.save_folder, f"recall_{self.scale}.csv" if self.scale else "recall.csv"
                )
            )
            accuracy = balanced_accuracy_score(targets, outputs)
            pd.DataFrame([accuracy]).to_csv(
                os.path.join(
                    self.save_folder,
                    f"avg_accuracy_{self.scale}.csv" if self.scale else "avg_accuracy.csv",
                )
            )
            accuracy = accuracy_score(targets, outputs)
            total_loss_desc.set_description_str(f"Test Results Total Accuracy: {accuracy:.5f}")
            return accuracy

    def make_nmf(self, input, feature_map, batch_idx):
        _, _, w_im, h_im = input.shape
        images = (
            make_grid(input, normalize=True, nrow=8, padding=0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )[:, :, :3]
        b, c, w, h = feature_map.shape
        n_rows = b // 8
        nmf_images = [[] for i in range(n_rows)]
        for i in range(b):
            a = feature_map.cpu().numpy()[i].reshape(c, w * h).transpose(1, 0)
            a1 = NMF(n_components=3, random_state=0).fit_transform(a).reshape(w, h, 3)
            a1[:, :, 0] = (a1[:, :, 0] - np.min(a1[:, :, 0])) / (
                np.max(a1[:, :, 0]) - np.min(a1[:, :, 0])
            )
            a1[:, :, 1] = (a1[:, :, 1] - np.min(a1[:, :, 1])) / (
                np.max(a1[:, :, 1]) - np.min(a1[:, :, 1])
            )
            a1[:, :, 2] = (a1[:, :, 2] - np.min(a1[:, :, 2])) / (
                np.max(a1[:, :, 2]) - np.min(a1[:, :, 2])
            )

            if i % 8 == 0:
                nmf_images[i // 8] = cv2.resize(a1, (w_im, h_im))
            else:
                nmf_images[i // 8] = np.hstack([nmf_images[i // 8], cv2.resize(a1, (w_im, h_im))])
        nmf_images = np.concatenate(nmf_images)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(images)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].imshow(nmf_images)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.savefig(
            os.path.join(self.nmf_folder, f"{batch_idx}_nmf_images.png"),
            dpi=500,
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close()

    def save_correlation(self, feat_corr_mat, spatial_corr_mat):
        feat_corr_mat = torch.cat(feat_corr_mat).mean(0).abs().cpu().numpy()
        cv2.imwrite(
            os.path.join(self.save_folder, "feat_correlation_matrix.png"),
            (feat_corr_mat * 255).astype("uint8"),
        )

        pd.DataFrame(feat_corr_mat).to_csv(
            os.path.join(self.save_folder, "feat_correlation_matrix.csv")
        )
        spatial_corr_mat = torch.cat(spatial_corr_mat).mean(0).abs().cpu().numpy()
        cv2.imwrite(
            os.path.join(self.save_folder, "spatial_correlation_matrix.png"),
            (spatial_corr_mat * 255).astype("uint8"),
        )

        pd.DataFrame(spatial_corr_mat).to_csv(
            os.path.join(self.save_folder, "spatial_correlation_matrix.csv")
        )

    def plot_umap(self, targets, feature_data):
        trans = umap.UMAP(random_state=42, n_components=2).fit(feature_data)
        fig, ax = plt.subplots(1, figsize=(14, 10))
        scatter = sns.scatterplot(
            trans.embedding_[:, 0],
            trans.embedding_[:, 1],
            hue=targets,
            palette="deep",
            # s=10,
            # c=targets,
            # alpha=0.8,
            # cmap="hsv",
        )
        # legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
        # ax.add_artist(legend1)
        # plt.colorbar()
        # plt.title("Training Set Embeddings")
        plt.savefig(self.save_folder + f"/test_embeddings.png")
        plt.close()

    def plot_umaps(self, df, feature_data):
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
        plt.savefig(os.path.join(self.plot_folder, "test_embed_moa.png"), dpi=500)
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
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        plt.savefig(os.path.join(self.plot_folder, "test_embed_moa_compound.png"), dpi=500)
        plt.close()
        fig, ax = plt.subplots(1, figsize=(16, 12))
        sns.scatterplot(
            x="x",
            y="y",
            hue="moa",
            style="well",
            palette=sns.color_palette("Paired", 11),
            legend=False,
            s=100,
            alpha=0.9,
            data=df,
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        plt.savefig(os.path.join(self.plot_folder, "test_embed_moa_well.png"), dpi=500)
        plt.close()
        fig, ax = plt.subplots(1, figsize=(16, 12))
        sns.scatterplot(
            x="x",
            y="y",
            hue="moa",
            style="plate",
            palette=sns.color_palette("Paired", 11),
            legend=False,
            s=100,
            alpha=0.9,
            data=df,
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        plt.savefig(os.path.join(self.plot_folder, "test_embed_moa_plate.png"), dpi=500)
        plt.close()
        for moa in moas:
            df_moa = df[df["moa"] == moa]
            fig, ax = plt.subplots(1, figsize=(16, 12))
            scatter = sns.scatterplot(
                x="x",
                y="y",
                hue="compound",
                style="well",
                # palette=sns.color_palette("Paired", 11),
                legend=False,
                s=100,
                alpha=0.9,
                data=df_moa,
            )
            plt.xlim(list(xlim))
            plt.ylim(list(ylim))
            plt.savefig(os.path.join(self.plot_folder, f"{moa}_well_compound.png"), dpi=500)
            plt.close()
            fig, ax = plt.subplots(1, figsize=(16, 12))
            scatter = sns.scatterplot(
                x="x",
                y="y",
                hue="compound",
                style="site",
                # palette=sns.color_palette("Paired", 11),
                legend="brief",
                s=100,
                alpha=0.9,
                data=df_moa,
            )
            plt.xlim(list(xlim))
            plt.ylim(list(ylim))
            plt.savefig(os.path.join(self.plot_folder, f"{moa}_sites_compound.png"), dpi=500)
            plt.close()
            fig, ax = plt.subplots(1, figsize=(16, 12))
            scatter = sns.scatterplot(
                x="x",
                y="y",
                hue="compound",
                style="plate",
                # palette=sns.color_palette("Paired", 11),
                legend="brief",
                s=100,
                alpha=0.9,
                data=df_moa,
            )
            plt.xlim(list(xlim))
            plt.ylim(list(ylim))
            plt.savefig(os.path.join(self.plot_folder, f"{moa}_plate_compound.png"), dpi=500)
            plt.close()


def make_cam(x, epsilon=1e-5):
    # relu(x) = max(x, 0)
    b, c, h, w = x.size()
    x = F.relu(x)
    flat_x = x.view(b, c, (h * w))
    max_value = flat_x.max(axis=-1)[0].view((b, c, 1, 1))

    return F.relu(x - epsilon) / (max_value + epsilon)


def get_feat_corr(x):
    b, c, w, h = x.shape
    x = x.permute(0, 2, 3, 1).reshape(-1, c)
    x = (x - x.mean(0)) / x.std(0)

    corr = (x.T @ x) / (b * w * h)

    return corr.unsqueeze(0)


def get_spatial_corr(x):
    x = x.flatten(2, 3)
    b, _, d = x.shape
    x = x / (x.norm(dim=1, p=2).unsqueeze(1) + 1e-5)
    corr = torch.bmm(x.permute(0, 2, 1), x)
    return corr
