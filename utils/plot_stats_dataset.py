import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("stats/new_stats/fl_normalization_stats.csv")
df_all = pd.read_csv("stats/new_stats/fl_normalization_stats_all.csv")
df = pd.concat([df, df_all], axis=0).reset_index(drop=True)
save_path = "/proj/haste_berzelius/exps/specs_new_splits/dataset_exploration/fl_stats"

for plate in df.plate.unique():
    df_plate = df[df.plate == plate].reset_index(drop=True)
    columns = df_plate.columns.tolist()
    means = []
    stds = []
    medians = []
    mads = []
    compounds = []
    for i, row in df_plate.iterrows():
        mean = row["mean_C1"]
        std = row["std_C1"]
        median = row["median_C1"]
        mad = row["mad_C1"]
        means.append(mean)
        stds.append(std)
        medians.append(median)
        mads.append(mad)
        compounds.append(row["compound"])

    plt.figure(figsize=(16, 12))
    plt.errorbar(
        compounds,
        medians,
        mads,
        fmt="ok",
        lw=15,
        alpha=0.5,
        label="median",
        ecolor="blue",
    )
    plt.errorbar(
        compounds, means, stds, fmt=".k", lw=10, alpha=0.5, label="mean", ecolor="red"
    )
    plt.xticks(rotation=70)
    plt.legend()
    plt.title(plate)
    plt.savefig(f"{save_path}/{plate}.png", dpi=500)
    plt.close()

