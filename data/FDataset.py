import glob
import os
import sys

import albumentations as album
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def dmso_normalization(im, dmso_mean, dmso_std):
    im_norm = (im.astype("float32") - dmso_mean) / dmso_std  #
    return im_norm


site_conversion = pd.DataFrame(
    {
        "bf_sites": ["s1", "s2", "s3", "s4", "s5"],
        "f_sites": ["s2", "s4", "s5", "s6", "s8"],
    }
)


class FNPChAugDataset(Dataset):
    def __init__(
        self,
        root,
        csv_file,
        bf_csv_file=None,
        channels=[0, 1, 2, 3, 4],
        to_one_hot=False,
        dmso_normalize=False,
        dmso_stats_path=None,
        subset_of_moas=True,
        moas=None,
        geo_transform=None,
        colour_transform=None,
    ):
        self.root = root
        self.csv_file = csv_file
        self.channels = channels

        self.bf_csv_file = bf_csv_file
        if bf_csv_file:
            self.bf_df = pd.read_csv(bf_csv_file)
        else:
            self.bf_df = None

        self.dmso_normalize = dmso_normalize
        self.dmso_stats_path = dmso_stats_path
        self.geo_transform = geo_transform
        self.colour_transform = colour_transform
        self.to_one_hot = to_one_hot
        self.subset_of_moas = subset_of_moas

        self.df = pd.read_csv(csv_file)

        if self.dmso_normalize:
            self.dmso_stats_df = pd.read_csv(
                dmso_stats_path, header=[0, 1], index_col=0
            )

        if self.subset_of_moas:
            self.moas = np.sort(moas)
            self.df = self.df[self.df["moa"].isin(self.moas)].reset_index(drop=True)
        else:
            self.moas = np.sort(self.df.moa.unique())
        self.labels = np.array(self.df["moa"])

    def __len__(self):
        return len(self.bf_df) if self.bf_csv_file else len(self.df)

    def __getitem__(self, idx):
        if self.bf_csv_file:
            bf_row = self.bf_df.iloc[idx]
            f_site = site_conversion["f_sites"][
                np.where(site_conversion["bf_sites"] == bf_row["site"])[0][0]
            ]
            row = self.df[
                (self.df.plate == bf_row.plate)
                & (self.df.well == bf_row.well)
                & (self.df.site == f_site)
            ].iloc[0]
        else:
            row = self.df.iloc[idx]

        im = np.load(self.root + row.path)
        target = np.where(row["moa"] == self.moas)[0].item()

        plate = row.plate
        site = row.site
        compound = row.compound
        well = row.well

        if self.to_one_hot:
            one_hot = np.zeros(len(self.labels))
            one_hot[target] = 1
            target = one_hot

        if self.geo_transform:
            augmented = self.geo_transform(image=im)
            im = augmented["image"]

        if self.colour_transform:
            augmented = self.colour_transform(image=im)
            im = augmented["image"]

        im = im[:, :, self.channels]
        if len(im.shape) == 2:
            im = im[..., np.newaxis]
        # Transpose to CNN shape
        im = im.transpose(2, 0, 1).astype("float32")

        return im, target, plate, site, compound, well


class FDataset(Dataset):
    def __init__(
        self,
        root,
        csv_file,
        bf_csv_file=None,
        channels=[0, 1, 2, 3, 4],
        to_one_hot=False,
        dmso_normalize=True,
        dmso_stats_path=None,
        subset_of_moas=True,
        moas=None,
        geo_transform=None,
        colour_transform=None,
    ):
        self.root = root
        self.csv_file = csv_file
        self.channels = channels

        self.bf_csv_file = bf_csv_file
        if bf_csv_file:
            self.bf_df = pd.read_csv(bf_csv_file)
            self.bf_df = self.bf_df["C1"].apply(lambda x: x.split("_")[2])
        else:
            self.bf_df = None

        self.dmso_normalize = dmso_normalize
        self.dmso_stats_path = dmso_stats_path
        self.geo_transform = geo_transform
        self.colour_transform = colour_transform
        self.to_one_hot = to_one_hot
        self.subset_of_moas = subset_of_moas

        self.df = pd.read_csv(csv_file)
        self.df["site"] = self.df["C1"].apply(lambda x: x.split("_")[2])

        if self.dmso_normalize:
            self.dmso_stats_df = pd.read_csv(
                dmso_stats_path, header=[0, 1], index_col=0
            )

        if self.subset_of_moas:
            self.moas = np.sort(moas)
            self.df = self.df[self.df["moa"].isin(self.moas)].reset_index(drop=True)
        else:
            self.moas = np.sort(self.df.moa.unique())
        self.labels = np.array(self.df["moa"])

    def __len__(self):
        return len(self.bf_df) if self.bf_csv_file else len(self.df)

    def __getitem__(self, idx):
        if self.bf_csv_file:
            bf_row = self.bf_df.iloc[idx]
            f_site = site_conversion["f_sites"][
                np.where(site_conversion["bf_sites"] == bf_row["site"])[0][0]
            ]
            row = self.df[
                (self.df.plate == bf_row.plate)
                & (self.df.well == bf_row.well)
                & (self.df.site == f_site)
            ].iloc[0]
        else:
            row = self.df.iloc[idx]

        site = row.site

        im = []
        for i in range(1, 6):
            local_im = cv2.imread(self.root + row.path + "/" + row["C" + str(i)], -1)
            assert site == row["C" + str(i)].split("_")[2]
            if self.dmso_normalize:
                dmso_mean = self.dmso_stats_df[row.plate]["C" + str(i)]["m"]
                dmso_std = self.dmso_stats_df[row.plate]["C" + str(i)]["std"]
                local_im = dmso_normalization(local_im, dmso_mean, dmso_std)
            im.append(local_im)
        im = np.array(im).transpose(1, 2, 0).astype("float32")

        target = np.where(row["moa"] == self.moas)[0].item()
        plate = row.plate
        compound = row.compound
        well = row.well

        if self.to_one_hot:
            one_hot = np.zeros(len(self.labels))
            one_hot[target] = 1
            target = one_hot

        if self.geo_transform:
            augmented = self.geo_transform(image=im)
            im = augmented["image"]

        if self.colour_transform:
            augmented = self.colour_transform(image=im)
            im = augmented["image"]

        im = im[:, :, self.channels]
        if len(im.shape) == 2:
            im = im[..., np.newaxis]
        # Transpose to CNN shape
        im = im.transpose(2, 0, 1).astype("float32")

        return im, target, plate, site, compound, well


class FNPDataset(Dataset):
    def __init__(
        self,
        root,
        csv_file,
        to_one_hot=False,
        dmso_normalize=True,
        dmso_stats_path=None,
        subset_of_moas=True,
        moas=None,
        transform=None,
    ):
        self.root = root
        self.csv_file = csv_file
        self.dmso_normalize = dmso_normalize
        self.dmso_stats_path = dmso_stats_path
        self.transform = transform
        self.to_one_hot = to_one_hot
        self.subset_of_moas = subset_of_moas

        self.df = pd.read_csv(csv_file)

        if self.dmso_normalize:
            self.dmso_stats_df = pd.read_csv(
                dmso_stats_path, header=[0, 1], index_col=0
            )

        if self.subset_of_moas:
            self.moas = np.sort(moas)
            self.df = self.df[self.df["moa"].isin(self.moas)].reset_index(drop=True)
        else:
            self.moas = np.sort(self.df.moa.unique())
        self.labels = np.array(self.df["moa"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        im = np.load(self.root + row.path)
        # im = []
        # for i in range(1, 6):
        #     local_im = cv2.imread(self.root + row.path + "/" + row["C" + str(i)], -1)

        #     if self.dmso_normalize:
        #         dmso_mean = self.dmso_stats_df[row.plate]["C" + str(i)]["m"]
        #         dmso_std = self.dmso_stats_df[row.plate]["C" + str(i)]["std"]
        #         local_im = dmso_normalization(local_im, dmso_mean, dmso_std)

        #     im.append(local_im)
        # im = np.array(im).transpose(1, 2, 0).astype("float32")

        target = np.where(row["moa"] == self.moas)[0].item()

        if self.to_one_hot:
            one_hot = np.zeros(len(self.labels))
            one_hot[target] = 1
            target = one_hot

        if self.transform:
            augmented = self.transform(image=im)
            im = augmented["image"]

        # Transpose to CNN shape
        im = im.transpose(2, 0, 1).astype("float32")

        return im, target


# class FDataset(Dataset):
#     def __init__(
#         self,
#         root,
#         csv_file,
#         to_one_hot=False,
#         dmso_normalize=True,
#         dmso_stats_path=None,
#         subset_of_moas=True,
#         moas=None,
#         transform=None,
#     ):
#         self.root = root
#         self.csv_file = csv_file
#         self.dmso_normalize = dmso_normalize
#         self.dmso_stats_path = dmso_stats_path
#         self.transform = transform
#         self.to_one_hot = to_one_hot
#         self.subset_of_moas = subset_of_moas

#         self.df = pd.read_csv(csv_file)

#         if self.dmso_normalize:
#             self.dmso_stats_df = pd.read_csv(dmso_stats_path, header=[0, 1], index_col=0)

#         if self.subset_of_moas:
#             self.moas = np.sort(moas)
#             self.df = self.df[self.df["moa"].isin(self.moas)].reset_index(drop=True)
#         else:
#             self.moas = np.sort(self.df.moa.unique())

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):

#         row = self.df.iloc[idx]

#         im = []
#         for i in range(1, 6):
#             local_im = cv2.imread(self.root + row.path + "/" + row["C" + str(i)], -1)

#             if self.dmso_normalize:
#                 dmso_mean = self.dmso_stats_df[row.plate]["C" + str(i)]["m"]
#                 dmso_std = self.dmso_stats_df[row.plate]["C" + str(i)]["std"]
#                 local_im = dmso_normalization(local_im, dmso_mean, dmso_std)

#             im.append(local_im)
#         im = np.array(im).transpose(1, 2, 0).astype("float32")

#         target = np.where(row["moa"] == self.moas)[0].item()

#         if self.to_one_hot:
#             one_hot = np.zeros(len(self.labels))
#             one_hot[target] = 1
#             target = one_hot

#         if self.transform:
#             augmented = self.transform(image=im)
#             im = augmented["image"]

#         # Transpose to CNN shape
#         im = im.transpose(2, 0, 1).astype("float32")

#         return im, target
