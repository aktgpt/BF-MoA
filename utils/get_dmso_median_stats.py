import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tqdm
import os
from scipy.stats import median_abs_deviation

image_path = "/proj/haste_berzelius/datasets/specs"


# csv_path = "stats/new_stats/fl_dmso_main.csv"
# mode = "bf" if "bf" in csv_path else "fl"


def make_dmso_stats(image_path, csv_path, mode, save_path):
    channel_list = [f"C{i}" for i in range(1, 7 if mode == "bf" else 6)]
    df = pd.read_csv(csv_path)
    dmso_stats = {}
    for plate in df.plate.unique():
        dmso_stats[plate] = {}
        df_plate = df[df.plate == plate].reset_index(drop=True)
        assert len(df_plate) == 22 * 5 if mode == "bf" else 22 * 9
        for c in tqdm.tqdm(channel_list):
            im = []
            for i, row in df_plate.iterrows():
                row_channel_path = row[c]
                assert row.plate == row_channel_path.split("-")[3]
                assert row.well == row_channel_path.split("_")[1]
                channel_path = os.path.join(image_path + row.path, row_channel_path)
                channel_im = cv2.imread(channel_path, -1)
                im.append(channel_im)

            [plate][c] = {
                "m": np.median(im),
                "std": median_abs_deviation(im, axis=None, nan_policy="omit"),
            }
    reform = {
        (outerKey, innerKey): values
        for outerKey, innerDict in dmso_stats.items()
        for innerKey, values in innerDict.items()
    }
    df_test = pd.DataFrame(reform)
    df_test.to_csv(save_path)


# make_dmso_stats(
#     image_path,
#     "stats/new_stats/fl_dmso_main.csv",
#     "fl",
#     "stats/new_stats/fl_dmso_MAD_stats.csv",
# )
make_dmso_stats(
    image_path,
    "stats/new_stats/bf_dmso_main.csv",
    "bf",
    "stats/new_stats/bf_dmso_MAD_stats.csv",
)
