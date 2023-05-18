import glob
import multiprocessing as mp
import os
import random
import string

import cv2
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from torch import std_mean
from tqdm import tqdm

image_path = "/proj/haste_berzelius/datasets/specs"
# image_path = "/proj/haste_berzelius/datasets/AG-48h-P3-L2"

# df_grit_based = pd.read_csv("stats/grit_based/bf_grit_based_stats_bgcorrect.csv")
# df_grit_based = df_grit_based[df_grit_based.compound == "dmso"]

# df_non_grit_based = pd.read_csv("stats/non_grit_based/bf_non_grit_based_stats_bgcorrect.csv")
# df_non_grit_based = pd.concat([df_non_grit_based, df_grit_based], axis=0)
# df_non_grit_based = df_non_grit_based.drop_duplicates(
#     subset=["plate", "compound", "well", "site"], keep="first"
# )
# df_non_grit_based.to_csv("stats/non_grit_based/bf_non_grit_based_stats_bgcorrect.csv", index=False)


def get_dataset_stats_mp(all_dataset_df, mode, kernel_size=None):
    with mp.Pool(8) as pool:  # use 3 processes
        # break up dataframe into smaller daraframes of N_ROWS rows each
        unique_plates = np.unique(all_dataset_df["plate"])
        plate_dfs = []
        for plate in unique_plates:
            plate_dfs.append(all_dataset_df[all_dataset_df.plate == plate].reset_index(drop=True))

        # unique_plates = np.unique(all_dataset_df["compound"])
        # plate_dfs = []
        # for plate in unique_plates:
        #     plate_dfs.append(
        #         all_dataset_df[all_dataset_df.compound == plate].reset_index(drop=True)
        #     )

        n = len(plate_dfs)
        results = []
        for i in range(n):
            results.append(
                pool.apply_async(get_plate_stats, args=(plate_dfs[-i], mode, kernel_size))
            )

        new_dfs = [result.get() for result in results]
        # reassemble final dataframe:
        df = pd.concat(new_dfs, ignore_index=True)
        return df


def get_plate_stats(df_plate, mode, kernel_size):
    plates = []
    compounds = []
    wells = []
    sites = []
    means = [[] for _ in range(6 if mode == "bf" else 5)]
    stds = [[] for _ in range(6 if mode == "bf" else 5)]

    medians = [[] for _ in range(6 if mode == "bf" else 5)]
    mads = [[] for _ in range(6 if mode == "bf" else 5)]

    unique_plate_compounds = np.unique(df_plate.compound)

    p_sum = np.zeros(2 if mode == "bf" else 5, dtype=np.float128)
    p_sum_sq = np.zeros(2 if mode == "bf" else 5, dtype=np.float128)
    n_pixels = 0

    for compound in unique_plate_compounds:
        df_plate_compound = df_plate[df_plate.compound == compound].reset_index(drop=True)
        unique_plate_compounds_wells = np.unique(df_plate_compound.well)

        compound_images = []
        for well in unique_plate_compounds_wells:
            df_plate_compound_well = df_plate_compound[df_plate_compound.well == well].reset_index(
                drop=True
            )

            well_images = []
            for index, row in df_plate_compound_well.iterrows():
                # image = np.load(image_path + row.path)
                image = np.load(
                    image_path
                    + os.path.splitext(row.path)[0]
                    + f"_bg_corrected_{kernel_size}_pc.npy"
                )

                ## plate mean
                p_sum += image.sum(axis=(0, 1))
                p_sum_sq += (image**2).sum(axis=(0, 1))
                n_pixels += image.shape[0] * image.shape[1]

                # site mean
                site_mean = np.mean(image, axis=(0, 1))
                site_std = np.std(image, axis=(0, 1))
                site_median = np.median(image, axis=(0, 1))
                site_mad = median_abs_deviation(image, axis=(0, 1))
                for c in range(2 if mode == "bf" else 5):
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
            # print(well_images.shape)

            # well mean
            well_mean = np.mean(well_images, axis=(0, 1, 2))
            well_std = np.std(well_images, axis=(0, 1, 2))
            well_median = np.median(well_images, axis=(0, 1, 2))
            well_mad = median_abs_deviation(well_images, axis=(0, 1, 2), nan_policy="omit")
            for c in range(2 if mode == "bf" else 5):
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
        # print(compound_images.shape)

        # compound mean
        compound_mean = np.mean(compound_images, axis=(0, 1, 2))
        compound_std = np.std(compound_images, axis=(0, 1, 2))
        compound_median = np.median(compound_images, axis=(0, 1, 2))
        compound_mad = median_abs_deviation(compound_images, axis=(0, 1, 2), nan_policy="omit")
        for c in range(2 if mode == "bf" else 5):
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

    # plate mean
    plate_mean = p_sum / n_pixels
    plate_std = np.sqrt(np.abs((p_sum_sq / n_pixels) - plate_mean**2))
    for c in range(2 if mode == "bf" else 5):
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


# bf_all_dataset_df = pd.read_csv("stats/non_grit_based/bf_non_grit_based_numpy.csv")
# # for kernel_size in [101, 201, 301, 401]:
# kernel_size = 101
# stats_df = get_dataset_stats_mp(bf_all_dataset_df, mode="bf", kernel_size=kernel_size)
# stats_df.to_csv(
#     f"stats/non_grit_based/bf_non_grit_based_stats_bgcorrect_{kernel_size}_tv.csv", index=False
# )

# bf_all_dataset_df = pd.read_csv("stats/new_data_splits/bf_numpy_main.csv")
# kernel_size = 101
# stats_df = get_dataset_stats_mp(bf_all_dataset_df, mode="bf", kernel_size=kernel_size)
# stats_df.to_csv(
#     f"stats/new_data_splits/bf_new_data_splits_stats_bg_corrected_{kernel_size}.csv", index=False
# )

fl_all_dataset_df = pd.read_csv("stats/non_grit_based/fl_non_grit_based_numpy.csv")
kernel_size = 101
stats_df = get_dataset_stats_mp(fl_all_dataset_df, mode="fl", kernel_size=kernel_size)
stats_df.to_csv(
    f"stats/new_data_splits/fl_new_data_splits_stats_bg_corrected_{kernel_size}_pc.csv", index=False
)


# fl_all_dataset_df = pd.read_csv("stats/new_data_splits/fl_numpy_main.csv")
# kernel_size = None
# stats_df = get_dataset_stats_mp(fl_all_dataset_df, mode="fl", kernel_size=kernel_size)
# stats_df.to_csv(f"stats/new_data_splits/fl_new_data_splits_stats.csv", index=False)

# bf_all_dataset_df = pd.read_csv("stats/grit_based/bf_grit_based_numpy.csv")
# stats_df = get_dataset_stats_mp(image_path, bf_all_dataset_df, mode="bf")
# stats_df.to_csv(f"stats/grit_based/bf_grit_based_stats_bgcorrect.csv", index=False)

# fl_all_dataset_df = pd.read_csv("stats/grit_based/fl_grit_based_numpy.csv")
# stats_df = get_dataset_stats(image_path, fl_all_dataset_df, mode="fl")
# stats_df.to_csv(f"stats/grit_based/fl_grit_based_stats.csv", index=False)


# plate_stats_df = get_plate_level_stats(image_path, bf_all_dataset_df, mode="bf")
# stats_df = pd.concat([stats_df, plate_stats_df]).reset_index(drop=True)

# fl_all_dataset_df = pd.read_csv("stats/non_grit_based/fl_non_grit_based_numpy.csv")
# stats_df = get_dataset_stats(image_path, fl_all_dataset_df, mode="fl")
# stats_df.to_csv(f"stats/non_grit_based/fl_non_grit_based_stats.csv", index=False)


# bf_all_dataset_df = pd.read_csv("stats/grit_based/bf_grit_based_numpy.csv")
# stats_df = get_dataset_stats(image_path, bf_all_dataset_df, mode="bf")
# stats_df.to_csv(f"stats/grit_based/bf_grit_based_stats.csv", index=False)

# fl_all_dataset_df = pd.read_csv("stats/grit_based/fl_grit_based_numpy.csv")
# stats_df = get_dataset_stats(image_path, fl_all_dataset_df, mode="fl")
# stats_df.to_csv(f"stats/grit_based/fl_grit_based_stats.csv", index=False)


# fl_all_dataset_df = pd.read_csv("stats/non_grit_based/fl_non_grit_based_numpy.csv")
# stats_df = get_dataset_stats(image_path, fl_all_dataset_df, mode="fl")
# stats_df.to_csv(f"stats/non_grit_based/fl_non_grit_based_stats.csv", index=False)


# def get_plate_level_stats(image_path, all_dataset_df, mode):
#     unique_plates = np.unique(all_dataset_df.plate)
#     plates = []
#     compounds = []
#     wells = []
#     sites = []
#     means = [[] for _ in range(6 if mode == "bf" else 5)]
#     stds = [[] for _ in range(6 if mode == "bf" else 5)]

#     medians = [[] for _ in range(6 if mode == "bf" else 5)]
#     mads = [[] for _ in range(6 if mode == "bf" else 5)]

#     for plate in tqdm(unique_plates):
#         df_plate = all_dataset_df[all_dataset_df.plate == plate].reset_index(drop=True)
#         print("Number of samples:", len(df_plate))
#         p_sum = np.zeros(6 if mode == "bf" else 5, dtype=np.float128)
#         p_sum_sq = np.zeros(6 if mode == "bf" else 5, dtype=np.float128)
#         n_pixels = 0

#         for index, row in tqdm(df_plate.iterrows()):
#             # image = np.load(image_path + row.path).astype(np.float64)
#             image = np.load(image_path + os.path.splitext(row.path)[0] + "_bg_corrected.npy")

#             p_sum += image.sum(axis=(0, 1))
#             p_sum_sq += (image**2).sum(axis=(0, 1))
#             n_pixels += image.shape[0] * image.shape[1]

#         plate_mean = p_sum / n_pixels
#         plate_std = np.sqrt(np.abs((p_sum_sq / n_pixels) - plate_mean**2))

#         for c in range(6 if mode == "bf" else 5):
#             means[c].append(plate_mean[c])
#             stds[c].append(plate_std[c])
#             medians[c].append(plate_mean[c])
#             mads[c].append(plate_std[c])

#         plates.append(row.plate)
#         compounds.append("all")
#         wells.append("all")
#         sites.append("all")

#         print(
#             f"plate: {row.plate}, compound: all, well: all,\n mean: {plate_mean},\n std: {plate_std},\n median: {plate_mean},\n mad: {plate_std}"
#         )

#     df_dict = {
#         "plate": plates,
#         "compound": compounds,
#         "well": wells,
#         "site": sites,
#     }
#     for i in range(len(means)):
#         df_dict["mean_C" + str(i + 1)] = means[i]
#         df_dict["std_C" + str(i + 1)] = stds[i]
#         df_dict["median_C" + str(i + 1)] = medians[i]
#         df_dict["mad_C" + str(i + 1)] = mads[i]
#     df = pd.DataFrame(df_dict)
#     return df
