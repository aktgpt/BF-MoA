import glob

import albumentations as aug
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


def dmso_normalization(im, dmso_mean, dmso_std):
    im_norm = (im - dmso_mean) / dmso_std
    return im_norm


site_conversion = pd.DataFrame(
    {"bf_sites": ["s1", "s2", "s3", "s4", "s5"], "f_sites": ["s2", "s4", "s5", "s6", "s8"]}
)


class VSNPSSLChAugDataset(Dataset):
    def __init__(
        self,
        root,
        bf_csv_file,
        f_csv_file,
        shift_csv_file,
        to_one_hot=False,
        dmso_normalize=False,
        dmso_stats_path_bf=None,
        dmso_stats_path_f=None,
        subset_of_moas=True,
        moas=None,
        geo_transform=None,
        colour_transform=None,
    ):
        self.root = root
        self.bf_csv_file = bf_csv_file
        self.f_csv_file = f_csv_file
        self.shift_csv_file = shift_csv_file

        self.dmso_normalize = dmso_normalize
        self.dmso_stats_path_bf = dmso_stats_path_bf
        self.dmso_stats_path_f = dmso_stats_path_f
        self.geo_transform = geo_transform
        self.colour_transform = colour_transform
        self.to_one_hot = to_one_hot
        self.subset_of_moas = subset_of_moas

        self.bfdf = pd.read_csv(bf_csv_file)
        self.fdf = pd.read_csv(f_csv_file)
        self.shift_df = pd.read_csv(shift_csv_file)

        if self.dmso_normalize:
            self.bf_dmso_stats_df = pd.read_csv(dmso_stats_path_bf, header=[0, 1], index_col=0)
            self.f_dmso_stats_df = pd.read_csv(dmso_stats_path_f, header=[0, 1], index_col=0)

        if self.subset_of_moas:
            self.moas = np.sort(moas)
            self.bfdf = self.bfdf[self.bfdf["moa"].isin(self.moas)].reset_index(drop=True)  # [:30]
            self.fdf = self.fdf[self.fdf["moa"].isin(self.moas)].reset_index(drop=True)
        else:
            self.moas = np.sort(self.df.moa.unique())

    def __len__(self):
        return len(self.bfdf)

    def __getitem__(self, idx):
        bf_row = self.bfdf.iloc[idx]

        bf_site = bf_row["site"]

        # shift_row = self.shift_df[
        #     (self.shift_df.plate == bf_row.plate)
        #     & (self.shift_df.well == bf_row.well)
        #     & (self.shift_df.bf_site == bf_site)
        # ].iloc[0]

        f_site = site_conversion["f_sites"][np.where(site_conversion["bf_sites"] == bf_site)[0][0]]

        f_row = self.fdf[
            (self.fdf.plate == bf_row.plate)
            & (self.fdf.well == bf_row.well)
            & (self.fdf.site == f_site)
        ].iloc[0]

        bf_im = np.load(self.root + bf_row.path)
        # bf_im = bf_im[
        #     shift_row.bf_ref_a : shift_row.bf_ref_b, shift_row.bf_ref_c : shift_row.bf_ref_d
        # ]

        f_im = np.load(self.root + f_row.path)
        # f_im = f_im[shift_row.f_ref_a : shift_row.f_ref_b, shift_row.f_ref_c : shift_row.f_ref_d]

        target = np.where(bf_row["moa"] == self.moas)[0].item()
        target_f = np.where(f_row["moa"] == self.moas)[0].item()
        assert target == target_f

        plate = bf_row["plate"]
        plate_f = f_row["plate"]
        assert plate == plate_f

        site = bf_row["site"]
        # site_f = f_row["site"]
        # assert site == site_f

        compound = bf_row["compound"]
        compound_f = f_row["compound"]
        assert compound == compound_f

        well = bf_row["well"]
        well_f = f_row["well"]
        assert well == well_f

        if self.to_one_hot:
            one_hot = np.zeros(len(self.labels))
            one_hot[target] = 1
            target = one_hot

        if self.geo_transform:
            augmented = self.geo_transform(image=bf_im)
            bf_im = augmented["image"]

            augmented = self.geo_transform(image=f_im)
            f_im = augmented["image"]

        if self.colour_transform:
            augmented = self.colour_transform(image=bf_im)
            bf_im = augmented["image"]

            augmented = self.colour_transform(image=f_im)
            f_im = augmented["image"]

            # bf_list = []
            # for i in range(bf_im.shape[-1]):
            #     augmented = self.colour_transform(image=bf_im[:, :, i])
            #     im = augmented["image"]
            #     bf_list.append(im[..., np.newaxis])
            # bf_im = np.concatenate(bf_list, axis=-1)

            # f_list = []
            # for i in range(f_im.shape[-1]):
            #     augmented = self.colour_transform(image=f_im[:, :, i])
            #     im = augmented["image"]
            #     f_list.append(im[..., np.newaxis])
            # f_im = np.concatenate(f_list, axis=-1)

        # Transpose to CNN shape
        bf_im = torch.tensor(bf_im.transpose(2, 0, 1))
        f_im = torch.tensor(f_im.transpose(2, 0, 1))

        return f_im, bf_im, target, plate, site, compound, well


class VSNPMTLChAugDataset(Dataset):
    def __init__(
        self,
        root,
        bf_csv_file,
        f_csv_file,
        shift_csv_file,
        to_one_hot=False,
        dmso_normalize=False,
        dmso_stats_path_bf=None,
        dmso_stats_path_f=None,
        subset_of_moas=True,
        moas=None,
        geo_transform=None,
        colour_transform=None,
    ):
        self.root = root
        self.bf_csv_file = bf_csv_file
        self.f_csv_file = f_csv_file
        self.shift_csv_file = shift_csv_file

        self.dmso_normalize = dmso_normalize
        self.dmso_stats_path_bf = dmso_stats_path_bf
        self.dmso_stats_path_f = dmso_stats_path_f
        self.geo_transform = geo_transform
        self.colour_transform = colour_transform
        self.to_one_hot = to_one_hot
        self.subset_of_moas = subset_of_moas

        self.bfdf = pd.read_csv(bf_csv_file)
        self.fdf = pd.read_csv(f_csv_file)
        self.shift_df = pd.read_csv(shift_csv_file)

        if self.dmso_normalize:
            self.bf_dmso_stats_df = pd.read_csv(dmso_stats_path_bf, header=[0, 1], index_col=0)
            self.f_dmso_stats_df = pd.read_csv(dmso_stats_path_f, header=[0, 1], index_col=0)

        if self.subset_of_moas:
            self.moas = np.sort(moas)
            self.bfdf = self.bfdf[self.bfdf["moa"].isin(self.moas)].reset_index(drop=True)  # [:30]
            self.fdf = self.fdf[self.fdf["moa"].isin(self.moas)].reset_index(drop=True)
        else:
            self.moas = np.sort(self.df.moa.unique())

    def __len__(self):
        return len(self.bfdf)

    def __getitem__(self, idx):
        bf_row = self.bfdf.iloc[idx]

        bf_site = bf_row["site"]

        shift_row = self.shift_df[
            (self.shift_df.plate == bf_row.plate)
            & (self.shift_df.well == bf_row.well)
            & (self.shift_df.bf_site == bf_site)
        ].iloc[0]

        f_site = site_conversion["f_sites"][np.where(site_conversion["bf_sites"] == bf_site)[0][0]]

        f_row = self.fdf[
            (self.fdf.plate == bf_row.plate)
            & (self.fdf.well == bf_row.well)
            & (self.fdf.site == f_site)
        ].iloc[0]

        bf_im = np.load(self.root + bf_row.path)
        bf_im = bf_im[
            shift_row.bf_ref_a : shift_row.bf_ref_b, shift_row.bf_ref_c : shift_row.bf_ref_d
        ]

        f_im = np.load(self.root + f_row.path)
        f_im = f_im[shift_row.f_ref_a : shift_row.f_ref_b, shift_row.f_ref_c : shift_row.f_ref_d]

        target = np.where(bf_row["moa"] == self.moas)[0].item()
        target_f = np.where(f_row["moa"] == self.moas)[0].item()
        assert target == target_f

        if self.to_one_hot:
            one_hot = np.zeros(len(self.labels))
            one_hot[target] = 1
            target = one_hot

        if self.geo_transform:
            augmented = self.geo_transform(image=bf_im, mask=f_im)
            bf_im = augmented["image"]
            f_im = augmented["mask"]

        if self.colour_transform:
            augmented = self.colour_transform(image=bf_im)
            bf_im = augmented["image"]

            augmented = self.colour_transform(image=f_im)
            f_im = augmented["image"]

            # bf_list = []
            # for i in range(bf_im.shape[-1]):
            #     augmented = self.colour_transform(image=bf_im[:, :, i])
            #     im = augmented["image"]
            #     bf_list.append(im[..., np.newaxis])
            # bf_im = np.concatenate(bf_list, axis=-1)

            # f_list = []
            # for i in range(f_im.shape[-1]):
            #     augmented = self.colour_transform(image=f_im[:, :, i])
            #     im = augmented["image"]
            #     f_list.append(im[..., np.newaxis])
            # f_im = np.concatenate(f_list, axis=-1)

        # Transpose to CNN shape
        bf_im = bf_im.transpose(2, 0, 1)
        f_im = f_im.transpose(2, 0, 1)

        return f_im, bf_im, target


class VSNPMTLDataset(Dataset):
    def __init__(
        self,
        root,
        bf_csv_file,
        f_csv_file,
        shift_csv_file,
        to_one_hot=False,
        dmso_normalize=False,
        dmso_stats_path_bf=None,
        dmso_stats_path_f=None,
        subset_of_moas=True,
        moas=None,
        transform=None,
    ):
        self.root = root
        self.bf_csv_file = bf_csv_file
        self.f_csv_file = f_csv_file
        self.shift_csv_file = shift_csv_file

        self.dmso_normalize = dmso_normalize
        self.dmso_stats_path_bf = dmso_stats_path_bf
        self.dmso_stats_path_f = dmso_stats_path_f
        self.transform = transform
        self.to_one_hot = to_one_hot
        self.subset_of_moas = subset_of_moas

        self.bfdf = pd.read_csv(bf_csv_file)
        self.fdf = pd.read_csv(f_csv_file)
        self.shift_df = pd.read_csv(shift_csv_file)

        if self.dmso_normalize:
            self.bf_dmso_stats_df = pd.read_csv(dmso_stats_path_bf, header=[0, 1], index_col=0)
            self.f_dmso_stats_df = pd.read_csv(dmso_stats_path_f, header=[0, 1], index_col=0)

        if self.subset_of_moas:
            self.moas = np.sort(moas)
            self.bfdf = self.bfdf[self.bfdf["moa"].isin(self.moas)].reset_index(drop=True)  # [:30]
            self.fdf = self.fdf[self.fdf["moa"].isin(self.moas)].reset_index(drop=True)
        else:
            self.moas = np.sort(self.df.moa.unique())

    def __len__(self):
        return len(self.bfdf)

    def __getitem__(self, idx):
        bf_row = self.bfdf.iloc[idx]

        bf_site = bf_row["site"]

        shift_row = self.shift_df[
            (self.shift_df.plate == bf_row.plate)
            & (self.shift_df.well == bf_row.well)
            & (self.shift_df.bf_site == bf_site)
        ].iloc[0]

        f_site = site_conversion["f_sites"][np.where(site_conversion["bf_sites"] == bf_site)[0][0]]

        f_row = self.fdf[
            (self.fdf.plate == bf_row.plate)
            & (self.fdf.well == bf_row.well)
            & (self.fdf.site == f_site)
        ].iloc[0]

        bf_im = np.load(self.root + bf_row.path)
        bf_im = bf_im[
            shift_row.bf_ref_a : shift_row.bf_ref_b, shift_row.bf_ref_c : shift_row.bf_ref_d
        ]

        f_im = np.load(self.root + f_row.path)
        f_im = f_im[shift_row.f_ref_a : shift_row.f_ref_b, shift_row.f_ref_c : shift_row.f_ref_d]

        target = np.where(bf_row["moa"] == self.moas)[0].item()

        target_f = np.where(f_row["moa"] == self.moas)[0].item()
        assert target == target_f

        if self.to_one_hot:
            one_hot = np.zeros(len(self.labels))
            one_hot[target] = 1
            target = one_hot

        if self.transform:
            augmented = self.transform(image=bf_im, mask=f_im)
            bf_im = augmented["image"]
            f_im = augmented["mask"]

        # Transpose to CNN shape
        bf_im = bf_im.transpose(2, 0, 1)
        f_im = f_im.transpose(2, 0, 1)

        return f_im, bf_im, target


class VSNPDataset(Dataset):
    def __init__(
        self,
        root,
        bf_csv_file,
        f_csv_file,
        shift_csv_file,
        to_one_hot=False,
        dmso_normalize=False,
        dmso_stats_path_bf=None,
        dmso_stats_path_f=None,
        subset_of_moas=True,
        moas=None,
        transform_f=None,
        transform_bf=None,
    ):
        self.root = root
        self.bf_csv_file = bf_csv_file
        self.f_csv_file = f_csv_file
        self.shift_csv_file = shift_csv_file

        self.dmso_normalize = dmso_normalize
        self.dmso_stats_path_bf = dmso_stats_path_bf
        self.dmso_stats_path_f = dmso_stats_path_f
        self.transform_f = transform_f
        self.transform_bf = transform_bf
        self.to_one_hot = to_one_hot
        self.subset_of_moas = subset_of_moas

        self.bfdf = pd.read_csv(bf_csv_file)
        self.fdf = pd.read_csv(f_csv_file)
        self.shift_df = pd.read_csv(shift_csv_file)

        if self.dmso_normalize:
            self.bf_dmso_stats_df = pd.read_csv(dmso_stats_path_bf, header=[0, 1], index_col=0)
            self.f_dmso_stats_df = pd.read_csv(dmso_stats_path_f, header=[0, 1], index_col=0)

        if self.subset_of_moas:
            self.moas = np.sort(moas)
            self.bfdf = self.bfdf[self.bfdf["moa"].isin(self.moas)].reset_index(drop=True)  # [:30]
            self.fdf = self.fdf[self.fdf["moa"].isin(self.moas)].reset_index(drop=True)
        else:
            self.moas = np.sort(self.df.moa.unique())

    def __len__(self):
        return len(self.bfdf)

    def __getitem__(self, idx):
        bf_row = self.bfdf.iloc[idx]

        bf_site = bf_row["site"]
        f_site = site_conversion["f_sites"][np.where(site_conversion["bf_sites"] == bf_site)[0][0]]

        f_row = self.fdf[
            (self.fdf.plate == bf_row.plate)
            & (self.fdf.well == bf_row.well)
            & (self.fdf.site == f_site)
        ].iloc[0]

        bf_im = np.load(self.root + bf_row.path)

        f_im = np.load(self.root + f_row.path)

        target = np.where(bf_row["moa"] == self.moas)[0].item()

        target_f = np.where(f_row["moa"] == self.moas)[0].item()

        assert target == target_f

        if self.to_one_hot:
            one_hot = np.zeros(len(self.labels))
            one_hot[target] = 1
            target = one_hot

        if self.transform_f:
            augmented = self.transform_f(image=f_im)
            f_im = augmented["image"]
        if self.transform_bf:
            augmented = self.transform_bf(image=bf_im)
            bf_im = augmented["image"]

        # Transpose to CNN shape
        bf_im = bf_im.transpose(2, 0, 1)
        f_im = f_im.transpose(2, 0, 1)

        return f_im, bf_im, target


class VSDataset(Dataset):
    def __init__(
        self,
        bf_csv_file,
        f_csv_file,
        shift_csv_file,
        to_one_hot=False,
        dmso_normalize=True,
        dmso_stats_path_bf=None,
        dmso_stats_path_f=None,
        subset_of_moas=True,
        moas=None,
        transform=None,
    ):

        self.bf_csv_file = bf_csv_file
        self.f_csv_file = f_csv_file
        self.shift_csv_file = shift_csv_file

        self.dmso_normalize = dmso_normalize
        self.dmso_stats_path_bf = dmso_stats_path_bf
        self.dmso_stats_path_f = dmso_stats_path_f
        self.transform = transform
        self.to_one_hot = to_one_hot
        self.subset_of_moas = subset_of_moas

        self.bfdf = pd.read_csv(bf_csv_file)
        self.fdf = pd.read_csv(f_csv_file)
        self.shift_df = pd.read_csv(shift_csv_file)

        if self.dmso_normalize:
            self.bf_dmso_stats_df = pd.read_csv(dmso_stats_path_bf, header=[0, 1], index_col=0)
            self.f_dmso_stats_df = pd.read_csv(dmso_stats_path_f, header=[0, 1], index_col=0)

        if self.subset_of_moas:
            self.moas = np.sort(moas)
            self.bfdf = self.bfdf[self.bfdf["moa"].isin(self.moas)].reset_index(drop=True)  # [:30]
            self.fdf = self.fdf[self.fdf["moa"].isin(self.moas)].reset_index(drop=True)
        else:
            self.moas = np.sort(self.df.moa.unique())

    def __len__(self):
        return len(self.bfdf)

    def __getitem__(self, idx):

        bf_row = self.bfdf.iloc[idx]
        bf_site = bf_row.nuclei.split("_")[-2]

        shift_rows = self.shift_df[
            (self.shift_df.plate == bf_row.plate) & (self.shift_df.well == bf_row.well)
        ]
        shift_row = shift_rows[shift_rows.bf_site == bf_site].iloc[0]

        f_rows = self.fdf[(self.fdf.plate == bf_row.plate) & (self.fdf.well == bf_row.well)]
        f_row = f_rows[f_rows.C1.str.contains(shift_row.f_site)].iloc[0]

        # -------------------------Bright Field Image--------------------------

        bf_im = []
        for i in range(1, 7):
            local_im = cv2.imread(bf_row.path + "/" + bf_row["C" + str(i)], -1)

            if self.dmso_normalize:
                dmso_mean = self.bf_dmso_stats_df[bf_row.plate]["C" + str(i)]["m"]
                dmso_std = self.bf_dmso_stats_df[bf_row.plate]["C" + str(i)]["std"]
                local_im = dmso_normalization(local_im, dmso_mean, dmso_std)

            ## Crop image
            local_im = local_im[
                shift_row.bf_ref_a : shift_row.bf_ref_b, shift_row.bf_ref_c : shift_row.bf_ref_d
            ]

            bf_im.append(local_im)
        bf_im = np.array(bf_im).transpose(1, 2, 0).astype("float")

        # ------------------------Fluorescent Image---------------------------

        f_im = []
        for i in range(1, 6):
            local_im = cv2.imread(f_row.path + "/" + f_row["C" + str(i)], -1)
            if self.dmso_normalize:
                dmso_mean = self.f_dmso_stats_df[f_row.plate]["C" + str(i)]["m"]
                dmso_std = self.f_dmso_stats_df[f_row.plate]["C" + str(i)]["std"]
                local_im = dmso_normalization(local_im, dmso_mean, dmso_std)

            ## Crop image
            local_im = local_im[
                shift_row.f_ref_a : shift_row.f_ref_b, shift_row.f_ref_c : shift_row.f_ref_d
            ]

            f_im.append(local_im)
        f_im = np.array(f_im).transpose(1, 2, 0).astype("float")

        target = np.where(bf_row["moa"] == self.moas)[0].item()

        if self.to_one_hot:
            one_hot = np.zeros(len(self.labels))
            one_hot[target] = 1
            target = one_hot

        if self.transform:
            augmented = self.transform(image=bf_im, mask=f_im)
            bf_im = augmented["image"]
            f_im = augmented["mask"]

        # Transpose to CNN shape
        bf_im = bf_im.transpose(2, 0, 1)
        f_im = f_im.transpose(2, 0, 1)

        return bf_im, f_im, target
