import numpy as np
import pandas as pd
import random

dmso_path = '/scratch-shared/phil/SPECS_phil/exp_stats/dmso_main.csv'
dmso = pd.read_csv(dmso_path)

bf_dmso_path = '/scratch-shared/phil/SPECS_phil/exp_stats/bf_dmso_main.csv'
bf_dmso = pd.read_csv(bf_dmso_path)

plates = bf_dmso.plate.unique()

for split in range(5):
    train_path = 'stats/new_train_val_test_5_splits/f_subset_train_split' + str(split + 1) + '.csv'
    val_path = 'stats/new_train_val_test_5_splits/f_subset_val_split' + str(split + 1) + '.csv'
    test_path = 'stats/new_train_val_test_5_splits/f_subset_test_split' + str(split + 1) + '.csv'
    
    bf_train_path = 'stats/new_train_val_test_5_splits/bf_subset_train_split' + str(split + 1) + '.csv'
    bf_val_path = 'stats/new_train_val_test_5_splits/bf_subset_val_split' + str(split + 1) + '.csv'
    bf_test_path = 'stats/new_train_val_test_5_splits/bf_subset_test_split' + str(split + 1) + '.csv'
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    bf_train_df = pd.read_csv(bf_train_path)
    bf_val_df = pd.read_csv(bf_val_path)
    bf_test_df = pd.read_csv(bf_test_path)
    
    train_plates = train_df.plate.unique()
    val_plates = val_df.plate.unique()
    test_plates = test_df.plate.unique()
    
    for plate in plates:
        dmso_plate = dmso[dmso['plate']==plate]
        bf_dmso_plate = bf_dmso[bf_dmso['plate']==plate]
        
        wells = dmso_plate.well.unique()
        
        # train data (five wells from each plate)
        if np.isin(plate, train_plates):
            wells_train = np.random.choice(wells, size=5, replace=False)
            
            dmso_wells = dmso_plate[dmso_plate.well.isin(wells_train)]
            dmso_wells = dmso_wells.reset_index(drop=True)
            dmso_wells['moa'] = ['dmso']*len(dmso_wells)
            dmso_wells.insert(2, 'compound', ['dmso']*len(dmso_wells))
            train_df = pd.concat([train_df, dmso_wells]).reset_index(drop=True)
            
            bf_dmso_wells = bf_dmso_plate[bf_dmso_plate.well.isin(wells_train)]
            bf_dmso_wells = bf_dmso_wells.reset_index(drop=True)
            bf_dmso_wells['moa'] = ['dmso']*len(bf_dmso_wells)
            bf_dmso_wells.insert(2, 'compound', ['dmso']*len(bf_dmso_wells))
            bf_train_df = pd.concat([bf_train_df, bf_dmso_wells]).reset_index(drop=True)
            
            wells = wells[np.isin(wells, wells_train, invert=True)]
        
        # val data (one well from each plate)
        if np.isin(plate, val_plates):
            wells_val = np.random.choice(wells, size=1, replace=False)
            
            dmso_wells = dmso_plate[dmso_plate.well.isin(wells_val)]
            dmso_wells = dmso_wells.reset_index(drop=True)
            dmso_wells['moa'] = ['dmso']*len(dmso_wells)
            dmso_wells.insert(2, 'compound', ['dmso']*len(dmso_wells))
            val_df = pd.concat([val_df, dmso_wells]).reset_index(drop=True)
            
            bf_dmso_wells = bf_dmso_plate[bf_dmso_plate.well.isin(wells_val)]
            bf_dmso_wells = bf_dmso_wells.reset_index(drop=True)
            bf_dmso_wells['moa'] = ['dmso']*len(bf_dmso_wells)
            bf_dmso_wells.insert(2, 'compound', ['dmso']*len(bf_dmso_wells))
            bf_val_df = pd.concat([bf_val_df, bf_dmso_wells]).reset_index(drop=True)
            
            wells = wells[np.isin(wells, wells_val, invert=True)]
        
        # test data (two wells from each plate)
        if np.isin(plate, test_plates):
            wells_test = np.random.choice(wells, size=2, replace=False)
            
            dmso_wells = dmso_plate[dmso_plate.well.isin(wells_test)]
            dmso_wells = dmso_wells.reset_index(drop=True)
            dmso_wells['moa'] = ['dmso']*len(dmso_wells)
            dmso_wells.insert(2, 'compound', ['dmso']*len(dmso_wells))
            test_df = pd.concat([test_df, dmso_wells]).reset_index(drop=True)
            
            bf_dmso_wells = bf_dmso_plate[bf_dmso_plate.well.isin(wells_test)]
            bf_dmso_wells = bf_dmso_wells.reset_index(drop=True)
            bf_dmso_wells['moa'] = ['dmso']*len(bf_dmso_wells)
            bf_dmso_wells.insert(2, 'compound', ['dmso']*len(bf_dmso_wells))
            bf_test_df = pd.concat([bf_test_df, bf_dmso_wells]).reset_index(drop=True)
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    bf_train_df.to_csv(bf_train_path, index=False)
    bf_val_df.to_csv(bf_val_path, index=False)
    bf_test_df.to_csv(bf_test_path, index=False)