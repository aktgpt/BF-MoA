import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from functions import *
import tqdm

dmso_csv_path = '/scratch-shared/phil/SPECS_phil/exp_stats/bf_dmso_main.csv'

df_dmso = pd.read_csv(dmso_csv_path)

dmso_stats = {}

for plate in df_dmso.plate.unique():
    dmso_stats[plate] = {}
    rows = df_dmso[df_dmso.plate == plate]
    
    for c in tqdm.tqdm(['C1','C2','C3','C4','C5', 'C6']):
        im = []
        
        for i in range(len(rows)):
            path = rows.path.iloc[i]
            path2 = rows[c].iloc[i]
            im.append(cv2.imread(path + '/' + path2, -1))
        
        dmso_stats[plate][c] = {'m': np.mean(im), 'std':np.std(im)}

reform = {(outerKey, innerKey): values for outerKey, innerDict in dmso_stats.items() for innerKey, values in innerDict.items()}

df_test = pd.DataFrame(reform)

df_test.to_csv('stats/bf_dmso_stats.csv')