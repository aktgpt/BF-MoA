import pandas as pd
import seaborn as sns
import numpy as np


df = pd.read_csv("stats/bf_normalization_stats.csv")

save_path = "/proj/haste_berzelius/exps/dataset_exploration"

for plate in df.plate.unique():
    df_plate = df[df.plate == plate].reset_index(drop=True)

    x = 1

