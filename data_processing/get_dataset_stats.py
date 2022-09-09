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
        unique_plate_compounds = np.unique(df_plate.compound)

        for compound in unique_plate_compounds:
            df_plate_compound = df_plate[df_plate.compound == compound].reset_index(drop=True)
            unique_plate_compounds_wells = np.unique(df_plate_compound.well)

            compound_images = []
            for well in unique_plate_compounds_wells:
                df_plate_compound_well = df_plate_compound[
                    df_plate_compound.well == well
                ].reset_index(drop=True)

                well_images = []
                for index, row in df_plate_compound_well.iterrows():
                    image = np.load(image_path + row.path)

                    # site mean
                    site_mean = np.mean(image, axis=(0, 1))
                    site_std = np.std(image, axis=(0, 1))
                    site_median = np.median(image, axis=(0, 1))
                    site_mad = median_abs_deviation(image, axis=(0, 1))
                    for c in range(6 if mode == "bf" else 5):
                        means[c].append(site_mean[c])
                        stds[c].append(site_std[c])
                        medians[c].append(site_median[c])
                        mads[c].append(site_mad[c])
                    plates.append(row.plate)
                    compounds.append(row.compound)
                    wells.append(row.well)
                    sites.append(row.site)

                    well_images.append(image[np.newaxis, ...])
                    compound_images.append(image[np.newaxis, ...])

                well_images = np.concatenate(well_images, axis=0)
                print(well_images.shape)

                # well mean
                well_mean = np.mean(well_images, axis=(0, 1, 2))
                well_std = np.std(well_images, axis=(0, 1, 2))
                well_median = np.median(well_images, axis=(0, 1, 2))
                well_mad = median_abs_deviation(well_images, axis=(0, 1, 2), nan_policy="omit")
                for c in range(6 if mode == "bf" else 5):
                    means[c].append(well_mean[c])
                    stds[c].append(well_std[c])
                    medians[c].append(well_median[c])
                    mads[c].append(well_mad[c])
                plates.append(row.plate)
                compounds.append(row.compound)
                wells.append(row.well)
                sites.append("all")

                print(
                    f"plate: {row.plate}, compound: {row.compound}, well: {row.well},\n mean: {well_mean},\n std: {well_std},\n median: {well_median},\n mad: {well_mad}"
                )

            compound_images = np.concatenate(compound_images, axis=0)
            print(compound_images.shape)

            # compound mean
            compound_mean = np.mean(compound_images, axis=(0, 1, 2))
            compound_std = np.std(compound_images, axis=(0, 1, 2))
            compound_median = np.median(compound_images, axis=(0, 1, 2))
            compound_mad = median_abs_deviation(compound_images, axis=(0, 1, 2), nan_policy="omit")
            for c in range(6 if mode == "bf" else 5):
                means[c].append(compound_mean[c])
                stds[c].append(compound_std[c])
                medians[c].append(compound_median[c])
                mads[c].append(compound_mad[c])
            plates.append(row.plate)
            compounds.append(row.compound)
            wells.append("all")
            sites.append("all")

            print(
                f"plate: {row.plate}, compound: {row.compound}, well: all,\n mean: {compound_mean},\n std: {compound_std},\n median: {compound_median},\n mad: {compound_mad}"
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

fl_all_dataset_df = pd.read_csv("stats/grit_based/fl_grit_based_numpy.csv")
stats_df = get_dataset_stats(image_path, fl_all_dataset_df, mode="fl")
stats_df.to_csv(f"stats/grit_based/fl_grit_based_stats.csv", index=False)
# get_dataset_stats(image_path, fl_all_dataset_df, mode="fl")
