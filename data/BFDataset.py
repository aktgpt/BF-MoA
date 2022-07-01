from torch.utils.data import Dataset
import numpy as np
import glob
import pandas as pd
import albumentations as album
import cv2


def dmso_normalization(im, dmso_mean, dmso_std):
    im_norm = (im.astype("float") - dmso_mean) / dmso_std
    return im_norm


class BFNPChAugDataset(Dataset):
    def __init__(
        self,
        root,
        csv_file,
        channels=[0, 1, 2, 3, 4, 5],
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

        self.dmso_normalize = dmso_normalize
        self.dmso_stats_path = dmso_stats_path
        self.geo_transform = geo_transform
        self.colour_transform = colour_transform
        self.to_one_hot = to_one_hot
        self.subset_of_moas = subset_of_moas

        self.df = pd.read_csv(csv_file)

        if self.dmso_normalize:
            self.dmso_stats_df = pd.read_csv(dmso_stats_path, header=[0, 1], index_col=0)

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
        # Transpose to CNN shape
        im = im.transpose(2, 0, 1).astype("float32")

        return im, target, plate, site, compound, well


class BFNPChAugSSLDataset(Dataset):
    def __init__(
        self,
        root,
        csv_file,
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

        self.dmso_normalize = dmso_normalize
        self.dmso_stats_path = dmso_stats_path
        self.geo_transform = geo_transform
        self.colour_transform = colour_transform
        self.to_one_hot = to_one_hot
        self.subset_of_moas = subset_of_moas

        self.df = pd.read_csv(csv_file)

        if self.dmso_normalize:
            self.dmso_stats_df = pd.read_csv(dmso_stats_path, header=[0, 1], index_col=0)

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

        target = np.where(row["moa"] == self.moas)[0].item()

        compound = row.compound

        if self.geo_transform:
            augmented = self.geo_transform(image=im)
            im1 = augmented["image"]

            augmented = self.geo_transform(image=im)
            im2 = augmented["image"]

        if self.colour_transform:
            augmented = self.colour_transform(image=im1)
            im1 = augmented["image"]

            augmented = self.colour_transform(image=im2)
            im2 = augmented["image"]

        # Transpose to CNN shape
        im1 = im1.transpose(2, 0, 1).astype("float32")
        im2 = im2.transpose(2, 0, 1).astype("float32")

        return im1, im2, target, compound


class BNPFDataset(Dataset):
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
            self.dmso_stats_df = pd.read_csv(dmso_stats_path, header=[0, 1], index_col=0)

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
        # for i in range(1, 7):
        #     local_im = cv2.imread(row.path + "/" + row["C" + str(i)], -1)

        #     if self.dmso_normalize:
        #         dmso_mean = self.dmso_stats_df[row.plate]["C" + str(i)]["m"]
        #         dmso_std = self.dmso_stats_df[row.plate]["C" + str(i)]["std"]
        #         local_im = dmso_normalization(local_im, dmso_mean, dmso_std)

        #     im.append(local_im)
        # im = np.array(im).transpose(1, 2, 0).astype("float")

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


class BFDataset(Dataset):
    def __init__(
        self,
        csv_file,
        to_one_hot=False,
        dmso_normalize=True,
        dmso_stats_path=None,
        subset_of_moas=True,
        moas=None,
        transform=None,
    ):

        self.csv_file = csv_file
        self.dmso_normalize = dmso_normalize
        self.dmso_stats_path = dmso_stats_path
        self.transform = transform
        self.to_one_hot = to_one_hot
        self.subset_of_moas = subset_of_moas

        self.df = pd.read_csv(csv_file)

        if self.dmso_normalize:
            self.dmso_stats_df = pd.read_csv(dmso_stats_path, header=[0, 1], index_col=0)

        if self.subset_of_moas:
            self.moas = np.sort(moas)
            self.df = self.df[self.df["moa"].isin(self.moas)].reset_index(drop=True)
        else:
            self.moas = np.sort(self.df.moa.unique())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        im = []
        for i in range(1, 7):
            local_im = cv2.imread(row.path + "/" + row["C" + str(i)], -1)

            if self.dmso_normalize:
                dmso_mean = self.dmso_stats_df[row.plate]["C" + str(i)]["m"]
                dmso_std = self.dmso_stats_df[row.plate]["C" + str(i)]["std"]
                local_im = dmso_normalization(local_im, dmso_mean, dmso_std)

            im.append(local_im)
        im = np.array(im).transpose(1, 2, 0).astype("float")

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
