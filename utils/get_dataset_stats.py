import glob
import os

import cv2
import numpy as np
import pandas as pd
from torch import std_mean
from tqdm import tqdm
from scipy.stats import median_abs_deviation
import random

image_path = "/proj/haste_berzelius/datasets/specs"


bf_all_dataset_df = pd.concat(
    [
        pd.read_csv("stats/new_train_val_test_splits/bf_subset_train_split1.csv"),
        pd.read_csv("stats/new_train_val_test_splits/bf_subset_val_split1.csv"),
        pd.read_csv("stats/new_train_val_test_splits/bf_subset_test_split1.csv"),
    ],
    axis=0,
).reset_index(drop=True)
bf_all_dataset_df.to_csv(
    "stats/new_train_val_test_splits/bf_all_dataset.csv", index=False
)

fl_all_dataset_df = pd.concat(
    [
        pd.read_csv("stats/new_train_val_test_splits/fl_subset_train_split1.csv"),
        pd.read_csv("stats/new_train_val_test_splits/fl_subset_val_split1.csv"),
        pd.read_csv("stats/new_train_val_test_splits/fl_subset_test_split1.csv"),
    ],
    axis=0,
    ignore_index=True,
).reset_index(drop=True)
fl_all_dataset_df.to_csv("stats/train_val_test_splits/fl_all_dataset.csv", index=False)


def get_compound_plate_stats(df, compound, plate):
    df_plate = df[df.compound == compound]
    df_plate = df_plate[df_plate.plate == plate]
    return df_plate


def get_dataset_stats(image_path, all_dataset_df, mode, mean_mode="mean"):
    unique_plates = np.unique(all_dataset_df.plate)
    plates = []
    compounds = []
    means = [[] for _ in range(6 if mode == "bf" else 5)]
    stds = [[] for _ in range(6 if mode == "bf" else 5)]

    medians = [[] for _ in range(6 if mode == "bf" else 5)]
    mads = [[] for _ in range(6 if mode == "bf" else 5)]

    for plate in tqdm(unique_plates):
        df_plate = all_dataset_df[all_dataset_df.plate == plate].reset_index(drop=True)
        unique_plate_compounds = np.unique(df_plate.compound)
        plate_images = []

        for compound in unique_plate_compounds:
            df_plate_compound = df_plate[df_plate.compound == compound].reset_index(
                drop=True
            )
            images = []
            # images_npy = []
            for id, row in df_plate_compound.iterrows():
                image = []
                for c in range(1, 7 if mode == "bf" else 6):
                    ch_image = cv2.imread(
                        image_path + row.path + "/" + row["C" + str(c)], -1
                    )
                    image.append(ch_image)
                image = np.array(image).transpose(1, 2, 0)  # .astype("float32")
                image_npy = np.load(
                    image_path
                    + row.path
                    + "/"
                    + os.path.splitext(row["C5"])[0]
                    + ".npy"
                )
                assert (
                    np.sum(image == image_npy) / image.size == 1
                ), "Images are not equal"
                # images_npy.append(image_npy[np.newaxis, ...])
                images.append(image[np.newaxis, ...])
                # if random.random() < 0.7:
                plate_images.append(image[np.newaxis, ...])
            images = np.concatenate(images, axis=0)  # .astype("float64")
            # images_npy = np.concatenate(images_npy, axis=0).astype("float64")
            # print(np.sum(np.equal(images, images_npy)) / images.size)
            print(images.shape[0])

            mean = np.mean(images, axis=(0, 1, 2))
            std = np.std(images, axis=(0, 1, 2))

            median = np.median(images, axis=(0, 1, 2))
            mad = median_abs_deviation(images, axis=(0, 1, 2), nan_policy="omit")

            # if mean_mode == "mean":
            #     mean = np.mean(images, axis=(0, 1, 2))
            #     std = np.std(images, axis=(0, 1, 2))
            # elif mean_mode == "median":
            #     mean = np.median(images, axis=(0, 1, 2))
            #     std = median_abs_deviation(images, axis=(0, 1, 2), nan_policy="omit")
            for c in range(6 if mode == "bf" else 5):
                means[c].append(mean[c])
                stds[c].append(std[c])
                medians[c].append(median[c])
                mads[c].append(mad[c])
            plates.append(row.plate)
            compounds.append(row.compound)

            print(
                f"plate: {row.plate}, compound: {row.compound},\n mean: {mean},\n std: {std},\n median: {median},\n mad: {mad}"
            )

        plate_images = np.concatenate(plate_images, axis=0)#.astype("float64")
        print(plate_images.shape[0])
        plate_mean = np.mean(plate_images, axis=(0, 1, 2))
        plate_std = np.std(plate_images, axis=(0, 1, 2))

        plate_median = np.median(plate_images, axis=(0, 1, 2))
        plate_mad = median_abs_deviation(
            plate_images, axis=(0, 1, 2), nan_policy="omit"
        )

        for c in range(6 if mode == "bf" else 5):
            means[c].append(plate_mean[c])
            stds[c].append(plate_std[c])
            medians[c].append(plate_median[c])
            mads[c].append(plate_mad[c])
        plates.append(row.plate)
        compounds.append("plate")

        print(
            f"plate: {row.plate}, compound: all,\n mean: {plate_mean},\n std: {plate_std},\n median: {plate_median},\n mad: {plate_mad}"
        )

    df_dict = {
        "plate": plates,
        "compound": compounds,
    }
    for i in range(len(means)):
        df_dict["mean_C" + str(i + 1)] = means[i]
        df_dict["std_C" + str(i + 1)] = stds[i]
        df_dict["median_C" + str(i + 1)] = medians[i]
        df_dict["mad_C" + str(i + 1)] = mads[i]
    df = pd.DataFrame(df_dict)
    df.to_csv(
        f"stats/final_stats/bf_normalization_stats_tif.csv"
        if mode == "bf"
        else f"stats/final_stats/fl_normalization_stats_all.csv",
        index=False,
    )


get_dataset_stats(image_path, bf_all_dataset_df, mode="bf")
# get_dataset_stats(image_path, fl_all_dataset_df, mode="fl")


# x = 1


# def dmso_normalization(im, dmso_mean, dmso_std):
#     im_norm = (im.astype("float") - dmso_mean) / dmso_std
#     return im_norm


# def make_numpy_df(image_path, df, dmso_stats_df, mode, save_path):
#     columns = ["plate", "well", "site", "compound", "path", "moa"]
#     new_df = pd.DataFrame(columns=columns)

#     for i in tqdm.tqdm(range(len(df))):
#         row = df.iloc[i]
#         site = row.C1.split("_")[2]

#         path = row.path + "/" + os.path.splitext(row["C" + str(5)])[0] + ".npy"

#         if not os.path.isfile(image_path + path):
#             im = []
#             for c in range(1, 7 if mode == "bf" else 6):
#                 local_im = cv2.imread(image_path + row.path + "/" + row["C" + str(c)], -1)
#                 dmso_mean = dmso_stats_df[row.plate]["C" + str(c)]["m"]
#                 dmso_std = dmso_stats_df[row.plate]["C" + str(c)]["std"]
#                 local_im = dmso_normalization(local_im, dmso_mean, dmso_std)
#                 im.append(local_im)

#             im = np.array(im).transpose(1, 2, 0).astype("float32")

#             np.save(image_path + path, im)

#         new_row = {
#             "plate": row.plate,
#             "well": row.well,
#             "compound": row.compound,
#             "path": path,
#             "site": site,
#             "moa": row.moa,
#         }
#         new_df = new_df.append(new_row, ignore_index=True)

#     new_df.to_csv(save_path, index=False)


# split_file_path = "stats/"
# split_files = glob.glob(f"{split_file_path}*")

# split_files = [f for f in split_files if "numpy" not in f]
# split_files = [f for f in split_files if "shift" not in f]
# split_files = [f for f in split_files if os.path.isfile(f)]
# split_files = [f for f in split_files if "stats" not in os.path.basename(f)]


# for split_file in split_files:
#     filename = os.path.basename(split_file)
#     print(split_file)
#     save_path = os.path.join(
#         split_file_path, os.path.splitext(filename)[0] + "_numpy" + os.path.splitext(filename)[1]
#     )
#     df = pd.read_csv(split_file)
#     if "bf" in filename:
#         mode = "bf"
#         dmso_stats_df = pd.read_csv("stats/bf_dmso_stats.csv", header=[0, 1], index_col=0)
#     elif "fl" in filename:
#         mode = "fl"
#         dmso_stats_df = pd.read_csv("stats/dmso_stats.csv", header=[0, 1], index_col=0)
#     else:
#         print("modality not recognized")
#     make_numpy_df(image_path, df, dmso_stats_df, mode, save_path)
#     print(save_path)

# # df = pd.read_csv("stats/bf_subset_val.csv", sep=",")
# # df = pd.read_csv("stats/f_subset_train.csv", sep=",")

# # df = pd.concat([train, trainid], ignore_index=True, sort=False)

# # dmso_stats_path = "stats/bf_dmso_stats.csv"
# # dmso_stats_df = pd.read_csv(dmso_stats_path, header=[0, 1], index_col=0)
