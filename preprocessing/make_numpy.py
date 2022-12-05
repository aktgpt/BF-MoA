import glob
import os
import random

import cv2
import numpy as np
import pandas as pd
import tqdm

modality = 'bf' # 'bf' or 'fl'
csv_path = '~/SPECS_phil/data_tables/' # user should change root to data_tables folder
csv_save_path = csv_path + modality + '_non_grit_based_numpy.csv'
im_path = '/scratch3-shared/phil/non_grit_based_numpy_data/' # user should change to where they placed the 'tiff' files
im_save_path = im_path + modality + '/'

df = pd.read_csv(csv_path + modality + '_data.csv')

# dealing with bf and fl data having different site numberings
site_conversion = pd.DataFrame(
    {
        "bf_sites": ["s1", "s2", "s3", "s4", "s5"],
        "fl_sites": ["s2", "s4", "s5", "s6", "s8"],
    }
)

def make_numpy(df, mode, im_save_path, csv_save_path):
    # make multi-channel numpy arrays and new csv
    columns = ["plate", "well", "site", "bf_site", "compound", "path", "moa"]
    new_df = pd.DataFrame(columns=columns)

    for i in tqdm.tqdm(range(len(df))):
        row = df.iloc[i]
        plate = row.plate
        path = im_path + plate
        well = row.well
        new_path = os.path.splitext(row["C" + str(5)])[0] + ".npy"
        site = row.site
        assert site == row.C1.split("_")[2], 'site mismatch'
        if site in ['s2', 's4', 's5', 's6', 's8']:
            bf_site = site_conversion["bf_sites"][np.where(site_conversion["fl_sites"] == site)[0][0]]
            
            im = []
            for c in range(1, 7 if mode == "bf" else 6):
                row_channel_path = row["C" + str(c)]
                row_plate = row_channel_path.split("-")[3]
                assert plate == row_plate, 'plate mismatch'
                row_well = row_channel_path.split("_")[1]
                assert well == row_well, 'well mismatch'

                local_im = cv2.imread(path + "/" + row["C" + str(c)], -1)
                im.append(local_im)

            im = np.array(im).transpose(1, 2, 0).astype("uint16")
            np.save(im_save_path + new_path, im)

            new_row = {
                "plate": row.plate,
                "well": row.well,
                "site": site,
                "bf_site": bf_site,
                "compound": row.compound,
                "path": im_save_path + new_path,
                "moa": row.moa,
            }

            new_df = new_df.append(new_row, ignore_index=True)
        
    new_df.to_csv(csv_save_path, index=False)

make_numpy(df, modality, im_save_path, csv_save_path)