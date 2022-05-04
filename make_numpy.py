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
        site = row.C1.split("_")[2]

        path = row.path + "/" + os.path.splitext(row["C" + str(5)])[0] + ".npy"

        if not os.path.isfile(image_path + path):
            im = []
            for c in range(1, 7 if mode == "bf" else 6):
                local_im = cv2.imread(image_path + row.path + "/" + row["C" + str(c)], -1)
                dmso_mean = dmso_stats_df[row.plate]["C" + str(c)]["m"]
                dmso_std = dmso_stats_df[row.plate]["C" + str(c)]["std"]
                local_im = dmso_normalization(local_im, dmso_mean, dmso_std)
                im.append(local_im)

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

split_file_path = "stats/"
split_files = glob.glob(f"{split_file_path}*")

split_files = [f for f in split_files if "numpy" not in f]
split_files = [f for f in split_files if "shift" not in f]
split_files = [f for f in split_files if os.path.isfile(f)]
split_files = [f for f in split_files if "stats" not in os.path.basename(f)]


for split_file in split_files:
    filename = os.path.basename(split_file)
    print(split_file)
    save_path = os.path.join(
        split_file_path, os.path.splitext(filename)[0] + "_numpy" + os.path.splitext(filename)[1]
    )
    df = pd.read_csv(split_file)
    if "bf" in filename:
        mode = "bf"
        dmso_stats_df = pd.read_csv("stats/bf_dmso_stats.csv", header=[0, 1], index_col=0)
    elif "fl" in filename:
        mode = "fl"
        dmso_stats_df = pd.read_csv("stats/dmso_stats.csv", header=[0, 1], index_col=0)
    else:
        print("modality not recognized")
    make_numpy_df(image_path, df, dmso_stats_df, mode, save_path)
    print(save_path)

# df = pd.read_csv("stats/bf_subset_val.csv", sep=",")
# df = pd.read_csv("stats/f_subset_train.csv", sep=",")

# df = pd.concat([train, trainid], ignore_index=True, sort=False)

# dmso_stats_path = "stats/bf_dmso_stats.csv"
# dmso_stats_df = pd.read_csv(dmso_stats_path, header=[0, 1], index_col=0)
