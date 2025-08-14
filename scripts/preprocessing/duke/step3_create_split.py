from pathlib import Path 
import numpy as np 
import pandas as pd 

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

path_root = Path(r'/path/to/duke/dataset')  # Update this path to your Duke dataset location
path_root_in = path_root
path_root_out = path_root/'preprocessed_crop'

df = pd.read_excel(r"/path/to/clinical_features.xlsx", header=[0, 1, 2])  # Update this path to your clinical features file
df = df[df[df.columns[38]] != 'NC'] # check if cancer is bilateral=1, unilateral=0 or NC 
df = df[[df.columns[0], df.columns[36],  df.columns[38]]] # Only pick relevant columns: Patient ID, Tumor Side, Bilateral
df.columns = ['PatientID', 'Location', 'Bilateral']  # Simplify columns as: Patient ID, Tumor Side
dfs = []
for side in ["left", 'right']:
    dfs.append(pd.DataFrame({
        'PatientID': df["PatientID"].str.split('_').str[2],
        'UID': df["PatientID"].str.split('_').str[2] + f"_{side}",
        'Malignant':df[["Location", "Bilateral"]].apply(lambda ds: int((ds[0] == side[0].upper()) | (ds[1]==1)) , axis=1) } ) )
df = pd.concat(dfs,  ignore_index=True)

df = df.reset_index(drop=True)
splits = []
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0) # StratifiedGroupKFold
sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
for fold_i, (train_val_idx, test_idx) in enumerate(sgkf.split(df['UID'], df['Malignant'], groups=df['PatientID'])):
    df_split = df.copy()
    df_split['Fold'] = fold_i 
    df_trainval = df_split.loc[train_val_idx]
    train_idx, val_idx = list(sgkf2.split(df_trainval['UID'], df_trainval['Malignant'], groups=df_trainval['PatientID']))[0]
    train_idx, val_idx = df_trainval.iloc[train_idx].index, df_trainval.iloc[val_idx].index 
    df_split.loc[train_idx, 'Split'] = 'train' 
    df_split.loc[val_idx, 'Split'] = 'val' 
    df_split.loc[test_idx, 'Split'] = 'test' 
    splits.append(df_split)
df_splits = pd.concat(splits)

path_out = path_root_out/'splits'
path_out.mkdir(parents=True, exist_ok=True)
df_splits.to_csv(path_out/'split.csv', index=False)
