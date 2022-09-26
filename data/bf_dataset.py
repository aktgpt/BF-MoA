import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy import ndimage, signal
from skimage import filters


def dmso_normalization(im, dmso_mean, dmso_std):
    return (im - dmso_mean) / dmso_std


def dmso_difference(im, dmso_mean, dmso_std):
    return im - dmso_mean


def get_normalization_stats(df, row, normalization_type):
    plate = row.plate
    well = row.well
    compound = row.compound
    # print(plate, well)
    if normalization_type == "well":
        dmso_mean = df.loc[
            (df["plate"] == plate) & (df["well"] == well),
            ["mean_C1", "mean_C2", "mean_C3", "mean_C4", "mean_C5", "mean_C6"],
        ].values
        dmso_std = df.loc[
            (df["plate"] == plate) & (df["well"] == well),
            ["std_C1", "std_C2", "std_C3", "std_C4", "std_C5", "std_C6"],
        ].values
    elif normalization_type == "dmso":
        dmso_mean = df.loc[
            (df["plate"] == plate) & (df["compound"] == "dmso") & (df["well"] == "all"),
            ["mean_C1", "mean_C2", "mean_C3", "mean_C4", "mean_C5", "mean_C6"],
        ].values
        dmso_std = df.loc[
            (df["plate"] == plate) & (df["compound"] == "dmso") & (df["well"] == "all"),
            ["std_C1", "std_C2", "std_C3", "std_C4", "std_C5", "std_C6"],
        ].values
    elif normalization_type == "comp":
        dmso_mean = df.loc[
            (df["plate"] == plate) & (df["compound"] == compound) & (df["well"] == "all"),
            ["mean_C1", "mean_C2", "mean_C3", "mean_C4", "mean_C5", "mean_C6"],
        ].values
        dmso_std = df.loc[
            (df["plate"] == plate) & (df["compound"] == compound) & (df["well"] == "all"),
            ["std_C1", "std_C2", "std_C3", "std_C4", "std_C5", "std_C6"],
        ].values
    elif normalization_type == "old":
        dmso_mean = np.array([df[plate]["C" + str(i)]["m"] for i in range(1, 7)])
        dmso_std = np.array([df[plate]["C" + str(i)]["std"] for i in range(1, 7)])
    return dmso_mean, dmso_std


def get_normalization_stats_old(df, row):
    dmso_mean = np.array([df[row.plate]["C" + str(i)]["m"] for i in range(1, 7)])
    dmso_std = np.array([df[row.plate]["C" + str(i)]["std"] for i in range(1, 7)])
    return dmso_mean, dmso_std


def gkern(sigma=3):
    """Returns a 2D Gaussian kernel array."""
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(4.0 * sd + 0.5)
    # Since we are calling correlate, not convolve, revert the kernel
    gkern1d = ndimage.filters._gaussian_kernel1d(sigma, 0, lw)[::-1]
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


class BFDataset(Dataset):
    def __init__(
        self,
        root,
        csv_file,
        normalize="dmso",  # False, "well", "dmso"
        dmso_stats_path=None,
        moas=None,
        geo_transform=None,
        colour_transform=None,
        bg_correct=False,
    ):
        self.root = root
        self.normalize = normalize

        self.geo_transform = geo_transform
        self.colour_transform = colour_transform

        self.df = pd.read_csv(csv_file, dtype=str) if isinstance(csv_file, str) else csv_file

        if self.normalize:
            self.dmso_stats_df = (
                pd.read_csv(dmso_stats_path, header=[0, 1], index_col=0)
                if self.normalize == "old"
                else pd.read_csv(dmso_stats_path)
            )

        if moas:
            self.moas = np.sort(moas)
            self.df = self.df[self.df["moa"].isin(self.moas)].reset_index(drop=True)
        else:
            self.moas = np.sort(self.df.moa.unique())

        self.df["site"] = self.df["C1"].apply(lambda x: x.split("_")[2])
        self.df["label"] = self.df["moa"].apply(lambda x: np.where(x == self.moas)[0].item())

        self.labels = np.array(self.df["moa"])

        self.bg_correct = bg_correct
        # if self.bg_correct:
        #     gaussian_kernel = gkern(sigma=50)
        #     self.gaussian_kernel_sum = np.sum(gaussian_kernel)
        #     self.gaussian_kernel = gaussian_kernel[..., np.newaxis].repeat(6, -1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.bg_correct:
            image = np.load(
                self.root + row.path + "/" + os.path.splitext(row["C5"])[0] + "_bg_corrected.npy"
            )
            # bg = np.load(self.root + row.path + "/" + os.path.splitext(row["C5"])[0] + "_bg.npy")
            # image = image - bg
            min_c = np.min(image, axis=(0, 1))
            ind = np.where(min_c < 0)
            image[:, :, ind] = image[:, :, ind] - min_c[ind]
            # image = np.float32(np.clip(image, 0, 65535))
        else:
            image = np.load(self.root + row.path + "/" + os.path.splitext(row["C5"])[0] + ".npy")

        # mean = np.mean(image, axis=(0, 1))
        # std = np.std(image, axis=(0, 1))
        # image = np.float32(image)

        if self.geo_transform:
            augmented = self.geo_transform(image=image)
            image = augmented["image"]

        if self.colour_transform:
            augmented = self.colour_transform(image=image)
            image = augmented["image"]

        if self.normalize:
            if self.bg_correct:
                image[:, :, ind] = image[:, :, ind] + min_c[ind]
            else:
                mean, std = get_normalization_stats(self.dmso_stats_df, row, self.normalize)
                image = dmso_normalization(image, mean, std)

        # Transpose to CNN shape
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)

        target = row.label
        plate = row.plate
        site = row.site
        compound = row.compound
        well = row.well

        return image, target, plate, site, compound, well


class BFSupConDataset(Dataset):
    def __init__(
        self,
        root,
        csv_file,
        normalize="dmso",  # False, "well", "dmso"
        dmso_stats_path=None,
        moas=None,
        geo_transform=None,
        colour_transform=None,
        label_type="moa",  # "comp"
    ):
        self.root = root
        self.normalize = normalize

        self.geo_transform = geo_transform
        self.colour_transform = colour_transform

        self.df = pd.read_csv(csv_file, dtype=str)

        if self.normalize:
            self.dmso_stats_df = (
                pd.read_csv(dmso_stats_path, header=[0, 1], index_col=0)
                if self.normalize == "old"
                else pd.read_csv(dmso_stats_path)
            )

        if moas:
            self.moas = np.sort(moas)
            self.df = self.df[self.df["moa"].isin(self.moas)].reset_index(drop=True)
        else:
            self.moas = np.sort(self.df.moa.unique())

        self.df["site"] = self.df["C1"].apply(lambda x: x.split("_")[2])

        self.df["label"] = self.df["moa"].apply(lambda x: np.where(x == self.moas)[0].item())

        self.label_type = label_type
        self.labels = np.array(
            self.df["label"]
            if label_type == "moa"
            else self.df["compound"].apply(
                lambda x: np.where(x == np.sort(self.df.compound.unique()))[0].item()
            )
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = np.load(self.root + row.path + "/" + os.path.splitext(row["C5"])[0] + ".npy")

        if self.geo_transform:
            augmented = self.geo_transform(image=image)
            xi = augmented["image"]
            augmented = self.geo_transform(image=image)
            xj = augmented["image"]

        if self.colour_transform: 
            augmented = self.colour_transform(image=xi)
            xi = augmented["image"]
            augmented = self.colour_transform(image=xj)
            xj = augmented["image"]

        if self.normalize:
            mean, std = get_normalization_stats(self.dmso_stats_df, row, self.normalize)
            xi = dmso_normalization(xi, mean, std)
            xj = dmso_normalization(xj, mean, std)

        # Transpose to CNN shape
        xi = torch.tensor(xi.transpose(2, 0, 1), dtype=torch.float32)
        xj = torch.tensor(xj.transpose(2, 0, 1), dtype=torch.float32)

        target = row.label
        plate = row.plate
        site = row.site
        compound = row.compound
        well = row.well

        return xi, xj, target, plate, site, compound, well


class BFMILDataset(Dataset):
    def __init__(
        self,
        root,
        csv_file,
        normalize="dmso",  # False, "well", "dmso"
        dmso_stats_path=None,
        moas=None,
        geo_transform=None,
        colour_transform=None,
        bg_correct=False,
    ):
        self.root = root
        self.normalize = normalize

        self.geo_transform = geo_transform
        self.colour_transform = colour_transform

        self.df = pd.read_csv(csv_file, dtype=str) if isinstance(csv_file, str) else csv_file

        if self.normalize:
            self.dmso_stats_df = (
                pd.read_csv(dmso_stats_path, header=[0, 1], index_col=0)
                if self.normalize == "old"
                else pd.read_csv(dmso_stats_path)
            )

        if moas:
            self.moas = np.sort(moas)
            self.df = self.df[self.df["moa"].isin(self.moas)].reset_index(drop=True)
        else:
            self.moas = np.sort(self.df.moa.unique())

        self.df["site"] = self.df["C1"].apply(lambda x: x.split("_")[2])
        self.df["label"] = self.df["moa"].apply(lambda x: np.where(x == self.moas)[0].item())

        self.labels = np.array(self.df["moa"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = np.load(self.root + row.path + "/" + os.path.splitext(row["C5"])[0] + ".npy")

        # mean = np.mean(image, axis=(0, 1))
        # std = np.std(image, axis=(0, 1))
        # image = np.float32(image)

        if self.geo_transform:
            augmented = self.geo_transform(image=image)
            image = augmented["image"]

        if self.colour_transform:
            augmented = self.colour_transform(image=image)
            image = augmented["image"]

        if self.normalize:
            mean, std = get_normalization_stats(self.dmso_stats_df, row, self.normalize)
            image = dmso_normalization(image, mean, std)

        # Transpose to CNN shape
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)

        x = (
            image.unfold(1, 512, 512)
            .unfold(2, 512, 512)
            .reshape(6, 16, 512, 512)
            .permute(1, 0, 2, 3)
        )
        idx = torch.randperm(x.size(0))
        x = x[idx, :]

        target = row.label
        plate = row.plate
        site = row.site
        compound = row.compound
        well = row.well

        return x, target, plate, site, compound, well
