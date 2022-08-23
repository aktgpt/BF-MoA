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
fl_all_dataset_df.to_csv("stats/new_train_val_test_splits/fl_all_dataset.csv", index=False)


def get_dataset_stats(image_path, all_dataset_df, mode, mean_mode="mean"):
    all_dataset_df["site"] = all_dataset_df["C1"].apply(lambda x: x.split("_")[2])
    unique_plates = np.unique(all_dataset_df.plate)
    plates = []
    compounds = []
    wells = []
    means = [[] for _ in range(6 if mode == "bf" else 5)]
    stds = [[] for _ in range(6 if mode == "bf" else 5)]

    medians = [[] for _ in range(6 if mode == "bf" else 5)]
    mads = [[] for _ in range(6 if mode == "bf" else 5)]

    for plate in tqdm(unique_plates):
        df_plate = all_dataset_df[all_dataset_df.plate == plate].reset_index(drop=True)
        unique_plate_compounds = np.unique(df_plate.compound)

        for compound in unique_plate_compounds:
            df_plate_compound = df_plate[df_plate.compound == compound].reset_index(
                drop=True
            )
            unique_plate_compounds_wells = np.unique(df_plate_compound.well)

            compound_images = []
            for well in unique_plate_compounds_wells:
                df_plate_compound_well = df_plate_compound[
                    df_plate_compound.well == well
                ].reset_index(drop=True)

                well_images = []
                for index, row in df_plate_compound_well.iterrows():
                    image = []
                    for c in range(1, 7 if mode == "bf" else 6):
                        ch_image = cv2.imread(
                            image_path + row.path + "/" + row["C" + str(c)], -1
                        )
                        image.append(ch_image)
                    image = np.array(image).transpose(1, 2, 0)
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

                    well_images.append(image[np.newaxis, ...])
                    compound_images.append(image[np.newaxis, ...])

                well_images = np.concatenate(well_images, axis=0)
                print(well_images.shape)
                mean = np.mean(well_images, axis=(0, 1, 2))
                std = np.std(well_images, axis=(0, 1, 2))

                median = np.median(well_images, axis=(0, 1, 2))
                mad = median_abs_deviation(
                    well_images, axis=(0, 1, 2), nan_policy="omit"
                )

                for c in range(6 if mode == "bf" else 5):
                    means[c].append(mean[c])
                    stds[c].append(std[c])
                    medians[c].append(median[c])
                    mads[c].append(mad[c])
                plates.append(row.plate)
                compounds.append(row.compound)
                wells.append(row.well)

                print(
                    f"plate: {row.plate}, compound: {row.compound}, well: {row.well},\n mean: {mean},\n std: {std},\n median: {median},\n mad: {mad}"
                )

            compound_images = np.concatenate(compound_images, axis=0)
            print(compound_images.shape)
            compound_mean = np.mean(compound_images, axis=(0, 1, 2))
            compound_std = np.std(compound_images, axis=(0, 1, 2))

            compound_median = np.median(compound_images, axis=(0, 1, 2))
            compound_mad = median_abs_deviation(
                compound_images, axis=(0, 1, 2), nan_policy="omit"
            )

            for c in range(6 if mode == "bf" else 5):
                means[c].append(compound_mean[c])
                stds[c].append(compound_std[c])
                medians[c].append(compound_median[c])
                mads[c].append(compound_mad[c])
            plates.append(row.plate)
            compounds.append(row.compound)
            wells.append("all")

            print(
                f"plate: {row.plate}, compound: {row.compound}, well: all,\n mean: {compound_mean},\n std: {compound_std},\n median: {compound_median},\n mad: {compound_mad}"
            )

    df_dict = {
        "plate": plates,
        "compound": compounds,
        "well": wells,
    }
    for i in range(len(means)):
        df_dict["mean_C" + str(i + 1)] = means[i]
        df_dict["std_C" + str(i + 1)] = stds[i]
        df_dict["median_C" + str(i + 1)] = medians[i]
        df_dict["mad_C" + str(i + 1)] = mads[i]
    df = pd.DataFrame(df_dict)
    df.to_csv(
        f"stats/final_stats/bf_stats_tif.csv"
        if mode == "bf"
        else f"stats/final_stats/fl_stats_tif.csv",
        index=False,
    )


# get_dataset_stats(image_path, bf_all_dataset_df, mode="bf")
get_dataset_stats(image_path, fl_all_dataset_df, mode="fl")
