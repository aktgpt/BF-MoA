from fileinput import close
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import string

df = pd.read_csv("stats/final_stats/all_bf_stats_tif.csv")
save_path = "/proj/haste_berzelius/exps/specs_new_splits/dataset_exploration/bf_well_stats"

alphabets = list(string.ascii_uppercase)[:16]


means = []
stds = []
plates = []
# medians = []
# mads = []
for plate in df.plate.unique():
    df_plate = df[df.plate == plate].reset_index(drop=True)
    heatmap = np.zeros((16, 24))
    unique_wells = df_plate.well.unique()
    unique_wells = unique_wells[unique_wells != "all"]
    means.append(
        df_plate[(df_plate.compound == "dmso") & (df_plate.well == "all")].mean_C1.values[0]
    )
    stds.append(df_plate[(df_plate.compound == "dmso") & (df_plate.well == "all")].std_C1.values[0])
    plates.append(plate)
    # df_plate = df_plate[df_plate.compound == "dmso"]
    for idx, row in df_plate.iterrows():
        if row.well != "all":
            compound = row.compound
            well = row.well
            idx = np.where((df_plate.well == "all") & (df_plate.compound == compound))[0][0]
            mean_comp = df_plate["mean_C1"].iloc[idx]
            std_comp = df_plate["std_C1"].iloc[idx]
            row_idx = alphabets.index(well[0])
            col_idx = int(well[1:]) - 1
            heatmap[row_idx, col_idx] =(row["mean_C1"] - mean_comp) #/ mean_comp# row["mean_C1"]
            #  row[
            #     "mean_C1"
            # ]  # (row["mean_C1"] - mean_comp) / mean_comp
    mask = np.zeros_like(heatmap)
    mask[heatmap == 0] = 1
    plt.figure(figsize=(16, 12))
    with sns.axes_style("white"):
        ax = sns.heatmap(heatmap, mask=mask, square=True)  # , center=0
    plt.savefig(f"{save_path}/{plate}_mean_heatmap.png")
    plt.close()
plt.figure(figsize=(16, 12))
plt.errorbar(
    plates,
    means,
    stds,
    fmt="ok",
    lw=15,
    alpha=0.5,
    label="mean",
    ecolor="blue",
)
plt.xticks(rotation=70)
plt.legend()
plt.savefig(f"{save_path}/all_plate_dmso_heatmap.png", dpi=500)
plt.close()
x = 1
