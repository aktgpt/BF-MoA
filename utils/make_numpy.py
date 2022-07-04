import glob
import os

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

        im = []
        for c in range(1, 7 if mode == "bf" else 6):
            row_channel_path = row["C" + str(c)]
            row_plate = row_channel_path.split("-")[3]
            assert plate == row_plate
            row_well = row_channel_path.split("_")[1]
            assert well == row_well

            if not os.path.isfile(image_path + path):
                local_im = cv2.imread(
                    image_path + row.path + "/" + row["C" + str(c)], -1
                )
                dmso_mean = dmso_stats_df[row.plate]["C" + str(c)]["m"]
                dmso_std = dmso_stats_df[row.plate]["C" + str(c)]["std"]
                local_im = dmso_normalization(local_im, dmso_mean, dmso_std)
                im.append(local_im)

        if not os.path.isfile(image_path + path):
            im = np.array(im).transpose(1, 2, 0).astype("float32")
            np.save(image_path + path, im)

        new_row = {
            "plate": row.plate,
            "well": row.well,
            "compound": row.compound,
            "path": path,
            "site": site,
            "moa": row.moa,
        }
        new_df = new_df.append(new_row, ignore_index=True)

    new_df.to_csv(save_path, index=False)


image_path = "/proj/haste_berzelius/datasets/specs"

split_file_path = "stats/new_train_val_test_splits/"
split_files = glob.glob(f"{split_file_path}*")

split_files = [f for f in split_files if "numpy" not in f]
split_files = [f for f in split_files if "shift" not in f]
split_files = [f for f in split_files if os.path.isfile(f)]
split_files = [f for f in split_files if "stats" not in os.path.basename(f)]


for split_file in sorted(split_files):
    filename = os.path.basename(split_file)
    save_path = os.path.join(
        split_file_path,
        os.path.splitext(filename)[0] + "_numpy" + os.path.splitext(filename)[1],
    )
    df = pd.read_csv(split_file)
    print(split_file, len(df.plate.unique()), len(df.compound.unique()))
    if "bf" in filename:
        mode = "bf"
        dmso_stats_df = pd.read_csv(
            "stats/new_stats/bf_dmso_stats.csv", header=[0, 1], index_col=0
        )
    elif "fl" in filename: 
        mode = "fl"
        dmso_stats_df = pd.read_csv(
            "stats/new_stats/fl_dmso_stats.csv", header=[0, 1], index_col=0
        )
    else:
        print("modality not recognized")
    print(save_path)
    make_numpy_df(image_path, df, dmso_stats_df, mode, save_path)
