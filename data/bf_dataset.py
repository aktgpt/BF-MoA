import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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

        self.labels = np.array(self.df["label"])

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
