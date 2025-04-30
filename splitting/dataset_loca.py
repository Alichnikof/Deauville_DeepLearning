import os
import pandas as pd
import torch
from splitting.dataset import *  # Assumes this imports dataset_singleview and dataset_singleview_center

# Define the mapping dictionary.
loc_mapping = {
    1.0: 0,
    2.0: 1,
    5.0: 2,
    6.0: 3,
    7.0: 4,
    8.0: 5,
    9.0: 6,
    10.0: 7,
    11.0: 8,
    20.0: 9,
    25.0: 10,
    27.0: 11,
    28.0: 12,
    51.0: 13
}

# New dataset class for training that returns both targets.
class dataset_singleview_withLoca(dataset_singleview):
    def __getitem__(self, index):
        # Get image and primary target using the original method.
        img, target = super().__getitem__(index)
        # Retrieve the raw localization target from the dataframe.
        raw_target_loc = self.df.iloc[index]['target_loc']
        # Map the raw value to the consecutive class index.
        try:
            target_loc = loc_mapping[raw_target_loc]
        except KeyError:
            raise ValueError(f"Localization target {raw_target_loc} at index {index} is not in the mapping dictionary.")
        # Return a tuple of targets: (primary, localization)
        return img, (target, target_loc)

# New dataset class for evaluation (center version) that returns both targets.
class dataset_singleview_center_withLoca(dataset_singleview_center):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        raw_target_loc = self.df.iloc[index]['target_loc']
        try:
            target_loc = loc_mapping[raw_target_loc]
        except KeyError:
            raise ValueError(f"Localization target {raw_target_loc} at index {index} is not in the mapping dictionary.")
        return img, (target, target_loc)

# New helper function to create the datasets.
def get_datasets_singleview_withLoca(transform=None, norm=None, balance=False, split_index=0):
    split = 'split' + str(split_index)
    df = pd.read_csv("splitting/data_resplit.csv")
    # Compute balance weights for the primary target only.
    weight_neg_pos = [1 - (df.target == 0).sum() / len(df),
                      1 - (df.target == 1).sum() / len(df)]
    
    # Split the DataFrame into training, validation, and test sets.
    df_train = df[df[split] == 'train'].drop(df.filter(regex='split').columns, axis=1)
    train_dset = dataset_singleview_withLoca(df_train, transform=transform, norm=norm)
    trainval_dset = dataset_singleview_center_withLoca(df_train, transform=None, norm=norm)
    
    df_val = df[df[split] == 'val'].drop(df.filter(regex='split').columns, axis=1)
    val_dset = dataset_singleview_center_withLoca(df_val, transform=None, norm=norm)
    
    df_test = df[df[split] == 'test'].drop(df.filter(regex='split').columns, axis=1)
    test_dset = dataset_singleview_center_withLoca(df_test, transform=None, norm=norm)
    
    return train_dset, trainval_dset, val_dset, test_dset, weight_neg_pos



if __name__ == '__main__':
    train_dset, trainval_dset, val_dset, test_dset, weight_neg_pos = get_datasets_singleview_withLoca()
    print(f"Train dataset size: {len(train_dset)}")
    print(f"Validation dataset size: {len(val_dset)}")
    # Optionally, print the first few rows of the CSV.
    df = pd.read_csv("splitting/data_resplit.csv")
    print(df.head())
