"""
Duke Dataset Cleaner

This script cleans the Duke dataset by:
1. Removing folders with missing sub.nii.gz files
2. Removing extra files from the remaining folders (keeping only sub.nii.gz)

Consolidated from clean_duke.py and deepclean_duke.py
"""

import pandas as pd
from pathlib import Path
import shutil

def clean_duke_dataset(data_root, csv_path=None):
    """
    Clean the Duke dataset by removing folders with missing sub.nii.gz files
    and removing extra files from the remaining folders.
    
    Args:
        data_root (str or Path): Path to the Duke data directory
        csv_path (str or Path, optional): Path to the CSV file with UID information
    
    Returns:
        tuple: (files_deleted, folders_deleted, rows_removed) counts
    """
    data_root = Path(data_root)
    files_deleted = 0
    folders_deleted = 0
    rows_removed = 0
    
    # If CSV path is provided, check for missing sub.nii.gz and update CSV
    if csv_path:
        df = pd.read_csv(csv_path)
        rows_to_drop = []
        
        # First pass: Check for missing sub.nii.gz and remove those folders
        for index, row in df.iterrows():
            uid = row['UID']
            folder_path = data_root / f'Breast_MRI_{uid}'
            sub_file_path = folder_path / 'sub.nii.gz'
            
            # Check if sub.nii.gz exists
            if not sub_file_path.exists():
                print(f"Missing sub.nii.gz for UID: {uid}")
                rows_to_drop.append(index)
                
                # Delete the folder if it exists
                if folder_path.exists():
                    print(f"Deleting folder: {folder_path}")
                    shutil.rmtree(folder_path)
                    folders_deleted += 1
        
        # Remove rows with missing files
        if rows_to_drop:
            print(f"\nRemoving {len(rows_to_drop)} rows from CSV")
            df_clean = df.drop(rows_to_drop)
            rows_removed = len(rows_to_drop)
            
            # Save the cleaned CSV
            backup_path = Path(csv_path).parent / 'split_backup.csv'
            df.to_csv(backup_path, index=False)
            print(f"Original CSV backed up to: {backup_path}")
            
            df_clean.to_csv(csv_path, index=False)
            print(f"Cleaned CSV saved to: {csv_path}")
        else:
            print("\nNo missing files found!")
            df_clean = df
    
    # Second pass: Clean all folders by removing non-sub.nii.gz files
    print("\nCleaning folders...")
    for folder_path in data_root.iterdir():
        if folder_path.is_dir():
            print(f"\nChecking folder: {folder_path.name}")
            
            # Check if sub.nii.gz exists before cleaning
            sub_file_path = folder_path / 'sub.nii.gz'
            if not sub_file_path.exists():
                print(f"  Missing sub.nii.gz, deleting folder: {folder_path}")
                shutil.rmtree(folder_path)
                folders_deleted += 1
                continue
            
            # Remove all files except sub.nii.gz
            for file_path in folder_path.glob('*'):
                if file_path.name != 'sub.nii.gz':
                    print(f"  Deleting extra file: {file_path.name}")
                    if file_path.is_file():
                        file_path.unlink()  # Delete file
                        files_deleted += 1
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)  # Delete directory
                        folders_deleted += 1
    
    print("\nDone! Dataset cleaned.")
    print(f"Deleted {files_deleted} files and {folders_deleted} folders")
    if csv_path:
        print(f"Removed {rows_removed} rows from CSV")
    
    return files_deleted, folders_deleted, rows_removed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean Duke dataset')
    parser.add_argument('--data_root', type=str, 
                        default=r"/path/to/duke/data",  # Update this path to your Duke data directory
                        help='Path to the Duke data directory')
    parser.add_argument('--csv_path', type=str, 
                        default=r"/path/to/duke/splits/split.csv",  # Update this path to your split CSV file
                        help='Path to the CSV file with UID information')
    
    args = parser.parse_args()
    
    clean_duke_dataset(args.data_root, args.csv_path)
