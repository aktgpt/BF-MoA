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


def get_normalization_stats(df, row, normalization_type, modality, mean_mode="mean"):
    plate = row.plate
    well = row.well
    compound = row.compound
    site = row.site

    mean_col_name = "mean_C" if mean_mode == "mean" else "median_C"
    std_col_name = "std_C" if mean_mode == "mean" else "mad_C"

    mean_cols = (
        [mean_col_name + str(i) for i in range(1, 7)]
        if modality == "bf"
        else [mean_col_name + str(i) for i in range(1, 6)]
    )
    std_cols = (
        [std_col_name + str(i) for i in range(1, 7)]
        if modality == "bf"
        else [std_col_name + str(i) for i in range(1, 6)]
    )

    if normalization_type == "plate":
        dmso_mean = df.loc[
            (df["plate"] == plate)
            & (df["compound"] == "all")
            & (df["well"] == "all")
            & (df["site"] == "all"),
            mean_cols,
        ].values
        dmso_std = df.loc[
            (df["plate"] == plate)
            & (df["compound"] == "all")
            & (df["well"] == "all")
            & (df["site"] == "all"),
            std_cols,
        ].values
    elif normalization_type == "dmso":
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
    elif normalization_type == "dmsosite":
        dmso_mean = df.loc[
            (df["plate"] == plate)
            & (df["compound"] == "dmso")
            & (df["well"] == "all")
            & (df["site"] == "all"),
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
        normalize="dmso",  # False, "well", "dmso", "image", "plate"
        dmso_stats_path=None,
        moas=None,
        geo_transform=None,
        colour_transform=None,
        bg_correct=False,
        modality="bf",  # "fl"
        mean_mode="mean",
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
        self.mean_mode = mean_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.bg_correct:
            image = np.load(self.root + os.path.splitext(row.path)[0] + "_bg_corrected.npy")
            min_c = np.min(image, axis=(0, 1))
            image = image - min_c
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
                image = image + min_c
            mean, std = get_normalization_stats(
                self.dmso_stats_df, row, self.normalize, self.modality, self.mean_mode
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


class MOASupConDataset(Dataset):
    def __init__(
        self,
        root,
        csv_file,
        normalize="dmso",  # False, "well", "dmso", "image", "plate"
        dmso_stats_path=None,
        moas=None,
        geo_transform=None,
        colour_transform=None,
        bg_correct=False,
        modality="bf",  # "fl"
        mean_mode="mean",
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
        self.comp_labels = np.array(
            self.df["compound"].apply(
                lambda x: np.where(x == np.sort(self.df.compound.unique()))[0].item()
            )
        )

        self.bg_correct = bg_correct
        self.modality = modality
        self.mean_mode = mean_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.bg_correct:
            image = np.load(self.root + os.path.splitext(row.path)[0] + "_bg_corrected.npy")
            min_c = np.min(image, axis=(0, 1))
            image = image - min_c
        else:
            image = np.load(self.root + row.path)

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
            if self.bg_correct:
                xi = xi + min_c
                xj = xj + min_c
            mean, std = get_normalization_stats(
                self.dmso_stats_df, row, self.normalize, self.modality, self.mean_mode
            )
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
