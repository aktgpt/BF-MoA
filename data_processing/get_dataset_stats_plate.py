import glob
import os

import cv2
import numpy as np
import pandas as pd
from torch import std_mean
from tqdm import tqdm
from scipy.stats import median_abs_deviation
import random
import string

image_path = "/proj/haste_berzelius/datasets/specs"


def get_dataset_stats(image_path, all_dataset_df, mode):
    unique_plates = np.unique(all_dataset_df.plate)
    plates = []
    compounds = []
    wells = []
    sites = []
    means = [[] for _ in range(6 if mode == "bf" else 5)]
    stds = [[] for _ in range(6 if mode == "bf" else 5)]

    medians = [[] for _ in range(6 if mode == "bf" else 5)]
    mads = [[] for _ in range(6 if mode == "bf" else 5)]

    for plate in tqdm(unique_plates):
        df_plate = all_dataset_df[all_dataset_df.plate == plate].reset_index(drop=True)
        print("Number of samples:", len(df_plate))
        p_sum = np.zeros(6 if mode == "bf" else 5, dtype=np.float128)
        p_sum_sq = np.zeros(6 if mode == "bf" else 5, dtype=np.float128)
        n_pixels = 0

        for index, row in tqdm(df_plate.iterrows()):
            image = np.load(image_path + row.path).astype(np.float64)
            p_sum += image.sum(axis=(0, 1))
            p_sum_sq += (image**2).sum(axis=(0, 1))
            n_pixels += image.shape[0] * image.shape[1]

        plate_mean = p_sum / n_pixels
        plate_std = np.sqrt(np.abs((p_sum_sq / n_pixels) - plate_mean**2))

        for c in range(6 if mode == "bf" else 5):
            means[c].append(plate_mean[c])
            stds[c].append(plate_std[c])
            medians[c].append(plate_mean[c])
            mads[c].append(plate_std[c])

        plates.append(row.plate)
        compounds.append("all")
        wells.append("all")
        sites.append("all")

        print(
            f"plate: {row.plate}, compound: all, well: all,\n mean: {plate_mean},\n std: {plate_std},\n median: {plate_mean},\n mad: {plate_std}"
        )

    df_dict = {
        "plate": plates,
        "compound": compounds,
        "well": wells,
        "site": sites,
    }
    for i in range(len(means)):
        df_dict["mean_C" + str(i + 1)] = means[i]
        df_dict["std_C" + str(i + 1)] = stds[i]
        df_dict["median_C" + str(i + 1)] = medians[i]
        df_dict["mad_C" + str(i + 1)] = mads[i]
    df = pd.DataFrame(df_dict)
    return df


# bf_all_dataset_df = pd.read_csv("stats/grit_based/bf_grit_based_numpy.csv")
# stats_df = get_dataset_stats(image_path, bf_all_dataset_df, mode="bf")
# stats_df.to_csv(f"stats/grit_based/bf_grit_based_stats.csv", index=False)

# fl_all_dataset_df = pd.read_csv("stats/grit_based/fl_grit_based_numpy.csv")
# stats_df = get_dataset_stats(image_path, fl_all_dataset_df, mode="fl")
# stats_df.to_csv(f"stats/grit_based/fl_grit_based_stats.csv", index=False)

bf_all_dataset_df = pd.read_csv("stats/non_grit_based/bf_non_grit_based_numpy.csv")
plate_stats_df = get_dataset_stats(image_path, bf_all_dataset_df, mode="bf")
stats_df = pd.read_csv("stats/non_grit_based/bf_non_grit_based_stats.csv")
all_stats_df = pd.concat([stats_df, plate_stats_df], axis=0).reset_index(drop=True)
all_stats_df.to_csv(f"stats/non_grit_based/bf_non_grit_based_stats.csv", index=False)

# fl_all_dataset_df = pd.read_csv("stats/non_grit_based/fl_non_grit_based_numpy.csv")
# stats_df = get_dataset_stats(image_path, fl_all_dataset_df, mode="fl")
# stats_df.to_csv(f"stats/non_grit_based/fl_non_grit_based_stats.csv", index=False)
