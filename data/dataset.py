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


def get_normalization_stats(df, row, normalization_type, modality):
    plate = row.plate
    well = row.well
    compound = row.compound
    site = row.site
    mean_cols = (
        ["mean_" + str(i) for i in range(1, 7)]
        if modality == "bf"
        else ["mean_" + str(i) for i in range(1, 6)]
    )
    std_cols = (
        ["std_" + str(i) for i in range(1, 7)]
        if modality == "bf"
        else ["std_" + str(i) for i in range(1, 6)]
    )
    if normalization_type == "dmso":
        dmso_mean = df.loc[
            (df["plate"] == plate)
            & (df["compound"] == "dmso")
            & (df["well"] == "all")
            & (df["site"] == "all"),
            mean_cols,
        ].values
        dmso_std = df.loc[
            (df["plate"] == plate)
            & (df["compound"] == "dmso")
            & (df["well"] == "all")
            & (df["site"] == "all"),
            std_cols,
        ].values

    elif normalization_type == "comp":
        dmso_mean = df.loc[
            (df["plate"] == plate)
            & (df["compound"] == compound)
            & (df["well"] == "all")
            & (df["site"] == "all"),
            mean_cols,
        ].values
        dmso_std = df.loc[
            (df["plate"] == plate)
            & (df["compound"] == compound)
            & (df["well"] == "all")
            & (df["site"] == "all"),
            std_cols,
        ].values
    elif normalization_type == "well":
        dmso_mean = df.loc[
            (df["plate"] == plate)
            & (df["compound"] == compound)
            & (df["well"] == well)
            & (df["site"] == "all"),
            mean_cols,
        ].values
        dmso_std = df.loc[
            (df["plate"] == plate)
            & (df["compound"] == compound)
            & (df["well"] == well)
            & (df["site"] == "all"),
            std_cols,
        ].values
    elif normalization_type == "site":
        dmso_mean = df.loc[
            (df["plate"] == plate)
            & (df["compound"] == compound)
            & (df["well"] == well)
            & (df["site"] == site),
            mean_cols,
        ].values
        dmso_std = df.loc[
            (df["plate"] == plate)
            & (df["compound"] == compound)
            & (df["well"] == well)
            & (df["site"] == site),
            std_cols,
        ].values
    return dmso_mean, dmso_std


def get_normalization_stats_old(df, row):
    dmso_mean = np.array([df[row.plate]["C" + str(i)]["m"] for i in range(1, 7)])
    dmso_std = np.array([df[row.plate]["C" + str(i)]["std"] for i in range(1, 7)])
    return dmso_mean, dmso_std


class MOADataset(Dataset):
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
        modality="bf",  # "fl"
    ):
        self.root = root
        self.normalize = normalize

        self.geo_transform = geo_transform
        self.colour_transform = colour_transform

        self.df = pd.read_csv(csv_file, dtype=str) if isinstance(csv_file, str) else csv_file

        if self.normalize:
            self.dmso_stats_df = pd.read_csv(dmso_stats_path)

        if moas:
            self.moas = np.sort(moas)
            self.df = self.df[self.df["moa"].isin(self.moas)].reset_index(drop=True)
        else:
            self.moas = np.sort(self.df.moa.unique())

        self.df["label"] = self.df["moa"].apply(lambda x: np.where(x == self.moas)[0].item())
        self.labels = np.array(self.df["moa"])

        self.bg_correct = bg_correct
        self.modality = modality

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.bg_correct:
            image = np.load(self.root + os.path.splitext(row.path)[0] + "_bg_corrected.npy")
            min_c = np.min(image, axis=(0, 1))
            ind = np.where(min_c < 0)
            image[:, :, ind] = image[:, :, ind] - min_c[ind]
        else:
            image = np.load(self.root + row.path)

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
                mean, std = get_normalization_stats(
                    self.dmso_stats_df, row, self.normalize, self.modality
                )
                image = dmso_normalization(image, mean, std)

        # Transpose to CNN shape
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)

        target = row.label
        plate = row.plate
        site = row.site
        compound = row.compound
        well = row.well

        return image, target, plate, site, compound, well
