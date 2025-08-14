"""
Fold Creator

This module provides functions for creating and validating cross-validation folds.
It includes functions for:
1. Creating k-fold cross-validation splits with train/val/test sets
2. Checking fold uniqueness to prevent data leakage

Consolidated from create_folds.py and check_fold_uniqueness.py
"""

import pandas as pd
import numpy as np
import sys
from collections import defaultdict
from pathlib import Path

def create_folds(df, n_splits=5, patient_col='record_id', random_state=42):
    """
    Create k-fold cross-validation splits with train/val/test sets.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        n_splits (int): Number of folds
        patient_col (str): Column name containing patient IDs
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame with added 'Split' and 'Fold' columns
    """
    print(f"Input samples: {len(df)}")
    
    # Get unique patients
    unique_patients = df[patient_col].unique()
    n_patients = len(unique_patients)
    print(f"Unique patients: {n_patients}")
    
    # List to store all fold dataframes
    all_folds = []
    
    # For each fold
    for fold_idx in range(n_splits):
        # Create a copy of the original dataframe for this fold
        fold_df = df.copy()
        
        # Shuffle patients
        np.random.seed(random_state + fold_idx)
        shuffled_patients = np.random.permutation(unique_patients)
        
        # Calculate split sizes
        n_test = int(len(shuffled_patients) * 0.1)
        n_val = int(len(shuffled_patients) * 0.1)
        
        # Split patients
        test_patients = shuffled_patients[:n_test]
        val_patients = shuffled_patients[n_test:n_test + n_val]
        train_patients = shuffled_patients[n_test + n_val:]
        
        # Assign splits
        fold_df.loc[fold_df[patient_col].isin(train_patients), 'Split'] = 'train'
        fold_df.loc[fold_df[patient_col].isin(val_patients), 'Split'] = 'val'
        fold_df.loc[fold_df[patient_col].isin(test_patients), 'Split'] = 'test'
        
        # Assign fold number
        fold_df['Fold'] = fold_idx
        
        # Add to list of all folds
        all_folds.append(fold_df)
        
        # Print statistics for this fold
        print(f"\nFold {fold_idx} statistics:")
        total = len(fold_df)
        for split in ['train', 'val', 'test']:
            count = len(fold_df[fold_df['Split'] == split])
            percentage = (count / total) * 100
            print(f"{split}: {count} samples ({percentage:.1f}%)")
    
    # Concatenate all folds
    final_df = pd.concat(all_folds, axis=0, ignore_index=True)
    print(f"\nOutput samples: {len(final_df)}")
    
    # Print overall statistics
    print("\nOverall Statistics:")
    total_samples = len(final_df)
    for split in ['train', 'val', 'test']:
        count = len(final_df[final_df['Split'] == split])
        percentage = (count / total_samples) * 100
        print(f"Total {split}: {count} samples ({percentage:.1f}%)")
    
    # Verify sample distribution
    print("\nVerifying split distribution in each fold:")
    for fold in range(n_splits):
        fold_data = final_df[final_df['Fold'] == fold]
        print(f"\nFold {fold}:")
        for split in ['train', 'val', 'test']:
            count = len(fold_data[fold_data['Split'] == split])
            percentage = (count / len(fold_data)) * 100
            print(f"{split}: {percentage:.1f}%")
    
    return final_df

def check_fold_uniqueness(df, id_cols=None):
    """
    Check if each item in the DataFrame appears in only one fold.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Split' and 'Fold' columns
        id_cols (list, optional): List of column names that uniquely identify an item.
                                 If None, uses ['XNAT', 'side']
        
    Returns:
        bool: True if all items appear in only one fold, False otherwise
    """
    if id_cols is None:
        id_cols = ['XNAT', 'side']
    
    # Check if required columns exist
    for col in id_cols + ['Split', 'Fold']:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame")
    
    # Dictionary to store item -> folds mapping
    item_folds = defaultdict(set)
    
    # Count total items and unique items
    total_items = len(df)
    unique_items = len(df[id_cols].drop_duplicates())
    
    # Process each row
    for _, row in df.iterrows():
        # Create a unique identifier for each item
        item_id = '_'.join(str(row[col]) for col in id_cols)
        
        # Add the fold to the set of folds for this item
        item_folds[item_id].add((row['Split'], row['Fold']))
    
    # Check if any item appears in multiple folds
    items_in_multiple_folds = {}
    for item_id, folds in item_folds.items():
        if len(folds) > 1:
            items_in_multiple_folds[item_id] = folds
    
    # Print results
    print(f"Total rows in DataFrame: {total_items}")
    print(f"Unique items ({'+'.join(id_cols)} combinations): {unique_items}")
    print(f"Items appearing in multiple split/fold combinations: {len(items_in_multiple_folds)}")
    print(f"Percentage of items in multiple folds: {len(items_in_multiple_folds)/unique_items*100:.2f}%")
    
    if items_in_multiple_folds:
        print("\nISSUE: Data leakage detected - items appear in multiple folds")
        print("\nSample of items in multiple folds (showing first 10):")
        for i, (item_id, folds) in enumerate(items_in_multiple_folds.items()):
            if i >= 10:
                break
            print(f"  {item_id}: {sorted(folds)}")
        
        # Count how many different split/fold combinations each item appears in
        fold_counts = defaultdict(int)
        for folds in items_in_multiple_folds.values():
            fold_counts[len(folds)] += 1
        
        print("\nDistribution of items by number of split/fold combinations:")
        for count, num_items in sorted(fold_counts.items()):
            print(f"  Items in {count} different split/folds: {num_items}")
        
        return False
    else:
        print("\nSUCCESS: All items appear in only one fold.")
        
        # Count items per split and fold
        split_fold_counts = defaultdict(int)
        for folds in item_folds.values():
            # There should be only one fold per item
            split_fold = list(folds)[0]
            split_fold_counts[split_fold] += 1
        
        # Print counts
        print("\nCounts per split and fold:")
        for (split, fold), count in sorted(split_fold_counts.items()):
            print(f"  {split}, fold {fold}: {count} items")
        
        return True

def create_and_validate_folds(csv_path, output_path=None, n_splits=5, patient_col='record_id', 
                             id_cols=None, random_state=42):
    """
    Create k-fold cross-validation splits and validate that there is no data leakage.
    
    Args:
        csv_path (str or Path): Path to the input CSV file
        output_path (str or Path, optional): Path to save the output CSV file. If None, uses
                                           input_path with '_with_splits' suffix
        n_splits (int): Number of folds
        patient_col (str): Column name containing patient IDs
        id_cols (list, optional): List of column names that uniquely identify an item
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame with added 'Split' and 'Fold' columns
    """
    # Read the CSV file
    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Create folds
    print("\nCreating folds...")
    df_with_splits = create_folds(df, n_splits, patient_col, random_state)
    
    # Check fold uniqueness
    print("\nChecking fold uniqueness...")
    is_unique = check_fold_uniqueness(df_with_splits, id_cols)
    
    if not is_unique:
        print("\nRECOMMENDATION:")
        print("The dataset has items appearing in multiple folds, which can lead to data leakage.")
        print("Consider the following solutions:")
        print("1. Reassign items to ensure each appears in only one fold")
        print("2. Filter the dataset to keep only one fold assignment per item")
        print("3. Group items by patient ID if that's available to ensure patient data stays in one fold")
    
    # Determine output path if not provided
    if output_path is None:
        output_path = Path(csv_path).with_stem(Path(csv_path).stem + '_with_splits')
    
    # Save the updated DataFrame
    df_with_splits.to_csv(output_path, index=False)
    print(f"\nSaved DataFrame with splits to: {output_path}")
    
    return df_with_splits

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create and validate cross-validation folds')
    parser.add_argument('csv_path', type=str, help='Path to the input CSV file')
    parser.add_argument('--output_path', type=str, help='Path to save the output CSV file')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds')
    parser.add_argument('--patient_col', type=str, default='record_id',
                        help='Column name containing patient IDs')
    parser.add_argument('--id_cols', type=str, nargs='+', help='Column names that uniquely identify an item')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--validate_only', action='store_true',
                        help='Only validate existing folds without creating new ones')
    
    args = parser.parse_args()
    
    if args.validate_only:
        # Only validate existing folds
        df = pd.read_csv(args.csv_path)
        check_fold_uniqueness(df, args.id_cols)
    else:
        # Create and validate folds
        create_and_validate_folds(
            args.csv_path,
            args.output_path,
            args.n_splits,
            args.patient_col,
            args.id_cols,
            args.random_state
        )
