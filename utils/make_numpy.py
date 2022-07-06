import glob
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import tqdm


def dmso_normalization(im, dmso_mean, dmso_std):
    im_norm = (im.astype("float") - dmso_mean) / dmso_std
    return im_norm


def make_numpy_df(image_path, df, dmso_stats_df, mode, save_path):
    columns = ["plate", "well", "site", "compound", "path", "moa"]
    new_df = pd.DataFrame(columns=columns)

    # total_unchanged_files = tqdm.tqdm(total=0, position=1, bar_format="{desc}")
    # check_files_correct = 0
    # check_files_total = 0
    for i in tqdm.tqdm(range(len(df))):
        row = df.iloc[i]

        plate = row.plate
        well = row.well
        path = row.path + "/" + os.path.splitext(row["C" + str(5)])[0] + ".npy"

        site = row.C1.split("_")[2]
        site = row.C1.split("_")[2]
        if mode == "bf":
            assert any(site == x for x in ["s1", "s2", "s3", "s4", "s5"])
        if mode == "fl":
            assert any(
                site == x
                for x in ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
            )

        # if random.uniform(0, 1) < checkfile_ratio:
        im = []
        for c in range(1, 7 if mode == "bf" else 6):
            row_channel_path = row["C" + str(c)]
            row_plate = row_channel_path.split("-")[3]
            assert plate == row_plate
            row_well = row_channel_path.split("_")[1]
            assert well == row_well

            # if not os.path.isfile(image_path + path):
            local_im = cv2.imread(image_path + row.path + "/" + row["C" + str(c)], -1)
            # dmso_mean = dmso_stats_df[row.plate]["C" + str(c)]["m"]
            # dmso_std = dmso_stats_df[row.plate]["C" + str(c)]["std"]
            # local_im = dmso_normalization(local_im, dmso_mean, dmso_std)
            im.append(local_im)
        im = np.array(im).transpose(1, 2, 0).astype("float32")
        np.save(image_path + path, im)

        # try:
        #     im_present = np.load(image_path + path)
        #     if (im == im_present).all():
        #         check_files_correct += 1
        #     else:
        #         np.save(image_path + path, im)
        # except:
        #     np.save(image_path + path, im)

        # check_files_total += 1

        # total_unchanged_files.set_description_str(
        #     f" {check_files_correct}/{check_files_total}"
        # )
        # if not os.path.isfile(image_path + path):
        #     im = np.array(im).transpose(1, 2, 0).astype("float32")
        #     np.save(image_path + path, im)

        new_row = {
            "plate": row.plate,
            "well": row.well,
            "compound": row.compound,
            "path": path,
            "site": site,
            "moa": row.moa,
        }
        new_df = new_df.append(new_row, ignore_index=True)

    # print(check_files_correct, check_files_total)
    new_df.to_csv(save_path, index=False)


image_path = "/proj/haste_berzelius/datasets/specs"

# dataset_file = "stats/new_stats/bf_main_11moas.csv"
# dmso_stats_df = pd.read_csv(
#     "stats/new_stats/bf_dmso_MAD_stats.csv", header=[0, 1], index_col=0
# )
# df = pd.read_csv(dataset_file)
# save_path = os.path.join(
#     "stats/new_stats",
#     os.path.splitext(os.path.basename(dataset_file))[0]
#     + "_numpy"
#     + os.path.splitext(os.path.basename(dataset_file))[1],
# )
# make_numpy_df(image_path, df, dmso_stats_df, "bf", save_path)


dataset_file = "stats/new_stats/fl_main_11moas.csv"
dmso_stats_df = pd.read_csv(
    "stats/new_stats/fl_dmso_MAD_stats.csv", header=[0, 1], index_col=0
)
df = pd.read_csv(dataset_file)
save_path = os.path.join(
    "stats/new_stats",
    os.path.splitext(os.path.basename(dataset_file))[0]
    + "_numpy"
    + os.path.splitext(os.path.basename(dataset_file))[1],
)
make_numpy_df(image_path, df, dmso_stats_df, "fl", save_path)


# split_file_path = "stats/new_train_val_test_splits/"
# splits = ["split1", "split2", "split3", "split4", "split5"]

# split_files = glob.glob(f"{split_file_path}*")

# split_files = [f for f in split_files if "numpy" not in f]
# split_files = [f for f in split_files if "all" not in f]
# split_files = [f for f in split_files if "shift" not in f]
# split_files = [f for f in split_files if os.path.isfile(f)]
# split_files = [f for f in split_files if "stats" not in os.path.basename(f)]

# modality_files = defaultdict(list)
# for f in sorted(split_files):
#     modality = os.path.basename(f).split("_")[0]
#     modality_files[modality].append(f)

# bf_split_files = defaultdict(list)
# fl_split_files = defaultdict(list)
# for split, files in modality_files.items():
#     for f in files:
#         split = os.path.basename(f).split("_")[-1].split(".")[0]
#         if "bf" in f:
#             bf_split_files[split].append(f)
#         if "fl" in f:
#             fl_split_files[split].append(f)


# checkfiles_ratios = [1.1, 0.5, 0.2, 0.1, 0.0]

# for i, (split, split_files) in enumerate(bf_split_files.items()):
#     mode = "bf"
#     dmso_stats_df = pd.read_csv(
#         "stats/new_stats/bf_dmso_MAD_stats.csv", header=[0, 1], index_col=0
#     )
#     check_file_ratio = checkfiles_ratios[i]
#     for split_file in split_files:
#         filename = os.path.basename(split_file)
#         save_path = os.path.join(
#             split_file_path,
#             os.path.splitext(filename)[0] + "_numpy" + os.path.splitext(filename)[1],
#         )
#         df = pd.read_csv(split_file)
#         make_numpy_df(image_path, df, dmso_stats_df, mode, save_path, check_file_ratio)


# for i, (split, split_files) in enumerate(fl_split_files.items()):
#     mode = "fl"
#     dmso_stats_df = pd.read_csv(
#         "stats/new_stats/fl_dmso_MAD_stats.csv", header=[0, 1], index_col=0
#     )
#     check_file_ratio = checkfiles_ratios[i]
#     for split_file in split_files:
#         filename = os.path.basename(split_file)
#         save_path = os.path.join(
#             split_file_path,
#             os.path.splitext(filename)[0] + "_numpy" + os.path.splitext(filename)[1],
#         )
#         df = pd.read_csv(split_file)
#         make_numpy_df(image_path, df, dmso_stats_df, mode, save_path, check_file_ratio)
x = 1
y = 1

# for split_file in sorted(split_files):
#     filename = os.path.basename(split_file)
#     save_path = os.path.join(
#         split_file_path,
#         os.path.splitext(filename)[0] + "_numpy" + os.path.splitext(filename)[1],
#     )
#     df = pd.read_csv(split_file)
#     print(split_file, len(df.plate.unique()), len(df.compound.unique()))
#     if "bf" in filename:
#         mode = "bf"
#         dmso_stats_df = pd.read_csv(
#             "stats/new_stats/bf_dmso_MAD_stats.csv", header=[0, 1], index_col=0
#         )
#     elif "fl" in filename:
#         mode = "fl"
#         dmso_stats_df = pd.read_csv(
#             "stats/new_stats/fl_dmso_MAD_stats.csv", header=[0, 1], index_col=0
#         )
#     else:
#         print("modality not recognized")
#     print(save_path)
#     make_numpy_df(image_path, df, dmso_stats_df, mode, save_path)
