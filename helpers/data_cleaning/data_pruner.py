"""
Data Pruner

This module provides functions for pruning and cleaning datasets.
It includes functions for:
1. Removing rows with invalid BIRADS scores
2. Removing duplicate rows based on specified columns
3. Pruning files that don't have corresponding entries in a CSV

Consolidated from prune.py and prune_duplicates.py
"""

import pandas as pd
import os
from pathlib import Path
from collections import Counter

def prune_by_birads(csv_path, output_path=None, processed_dir=None, min_birads=1):
    """
    Remove rows where BIRADS is less than or equal to the specified minimum value
    and optionally delete corresponding files from a processed directory.
    
    Args:
        csv_path (str or Path): Path to the CSV file
        output_path (str or Path, optional): Path to save the filtered CSV. If None, uses
                                           input_path with '_filtered' suffix
        processed_dir (str or Path, optional): Directory containing processed files to prune
        min_birads (int): Minimum BIRADS score to keep (rows with BIRADS <= min_birads are removed)
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    # Read the CSV
    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    original_count = len(df)
    
    # Check if BIRADS column exists
    if 'BIRADS' not in df.columns:
        raise ValueError("BIRADS column not found in the CSV file")
    
    # Remove rows where BIRADS <= min_birads
    df = df[df['BIRADS'] > min_birads]
    filtered_count = len(df)
    removed_count = original_count - filtered_count
    
    print(f"Removed {removed_count} rows with BIRADS <= {min_birads}")
    print(f"Remaining rows: {filtered_count}")
    
    # Delete corresponding files from processed directory if provided
    if processed_dir:
        processed_dir = Path(processed_dir)
        if not processed_dir.exists():
            print(f"Warning: Processed directory {processed_dir} does not exist")
        else:
            # Check if filename column exists
            if 'filename' not in df.columns:
                print("Warning: 'filename' column not found in the CSV file, cannot prune files")
            else:
                # Get list of files to keep
                files_to_keep = set(df['filename'].tolist())
                
                # Get all files in the processed directory
                all_files = [f for f in processed_dir.glob('*.nii.gz')]
                
                # Delete files that are not in the list of files to keep
                deleted_count = 0
                for file_path in all_files:
                    if file_path.name not in files_to_keep:
                        file_path.unlink()
                        deleted_count += 1
                        print(f"Removed file: {file_path.name}")
                
                print(f"Deleted {deleted_count} files from {processed_dir}")
    
    # Determine output path if not provided
    if output_path is None:
        output_path = Path(csv_path).with_stem(Path(csv_path).stem + '_filtered')
    
    # Save filtered CSV
    df.to_csv(output_path, index=False)
    print(f"Saved filtered CSV to: {output_path}")
    
    return df

def prune_duplicates(df, subset=None, keep='first', inplace=False):
    """
    Remove duplicate rows from a DataFrame based on specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column names to consider for identifying duplicates.
                               If None, uses all columns
        keep (str): Which duplicates to keep ('first', 'last', or False to drop all duplicates)
        inplace (bool): Whether to modify the DataFrame in place
        
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    # Count rows before removing duplicates
    rows_before = len(df)
    
    # Remove duplicates
    if inplace:
        df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        df_pruned = df
    else:
        df_pruned = df.drop_duplicates(subset=subset, keep=keep)
    
    # Count rows after removing duplicates
    rows_after = len(df_pruned)
    duplicates_removed = rows_before - rows_after
    
    print(f"Removed {duplicates_removed} duplicate rows")
    print(f"Remaining rows: {rows_after}")
    
    return df_pruned

def prune_duplicates_from_csv(csv_path, output_path=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file based on specified columns.
    
    Args:
        csv_path (str or Path): Path to the CSV file
        output_path (str or Path, optional): Path to save the pruned CSV. If None, overwrites
                                           the input file
        subset (list, optional): Column names to consider for identifying duplicates.
                               If None, uses all columns
        keep (str): Which duplicates to keep ('first', 'last', or False to drop all duplicates)
        
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    # Read the CSV file
    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Remove duplicates
    df_pruned = prune_duplicates(df, subset, keep)
    
    # Determine output path if not provided
    if output_path is None:
        output_path = csv_path
    
    # Write the pruned data back to the CSV file
    df_pruned.to_csv(output_path, index=False)
    print(f"Saved pruned CSV to: {output_path}")
    
    return df_pruned

def prune_mst_data(csv_path, mst_data_path, output_path=None):
    """
    Prune the MST BIRADS CSV to only include rows where corresponding XNAT folders exist.
    
    Args:
        csv_path (str or Path): Path to the input CSV file
        mst_data_path (str or Path): Path to the MST data folder containing XNAT folders
        output_path (str or Path, optional): Path to save the filtered CSV. If None, uses
                                           input_path with '_filtered' suffix
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows with existing folders
    """
    # Read the CSV file
    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    original_count = len(df)
    
    # Check if XNAT column exists
    if 'XNAT' not in df.columns:
        raise ValueError("XNAT column not found in the CSV file")
    
    # Check for any NaN values in XNAT column
    nan_count = df['XNAT'].isna().sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in XNAT column")
    
    # Create a function to check if folder exists for an XNAT ID
    def folder_exists(xnat_id):
        # Skip NaN values
        if pd.isna(xnat_id):
            return False
            
        # Convert to string if it's not already
        xnat_str = str(xnat_id)
        
        # Check if the XNAT ID exists directly as a folder
        folder_path = Path(mst_data_path) / xnat_str
        return folder_path.exists()
    
    # Create a mask for rows with existing folders
    print("Checking folder existence...")
    has_folder = df['XNAT'].apply(folder_exists)
    
    # Filter the DataFrame
    filtered_df = df[has_folder].copy()
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count
    
    print(f"Removed {removed_count} rows without existing folders")
    print(f"Remaining rows: {filtered_count}")
    
    # Determine output path if not provided
    if output_path is None:
        output_path = Path(csv_path).with_stem(Path(csv_path).stem + '_filtered')
    
    # Save to file if output path is provided
    filtered_df.to_csv(output_path, index=False)
    print(f"Saved filtered CSV to: {output_path}")
    
    return filtered_df

def batch_prune_duplicates(directory, pattern='*.csv', subset=None, keep='first', recursive=False):
    """
    Remove duplicate rows from all CSV files in a directory.
    
    Args:
        directory (str or Path): Directory containing CSV files
        pattern (str): Pattern to match CSV files
        subset (list, optional): Column names to consider for identifying duplicates
        keep (str): Which duplicates to keep ('first', 'last', or False to drop all duplicates)
        recursive (bool): Whether to search for CSV files recursively
        
    Returns:
        dict: Dictionary mapping CSV files to the number of duplicates removed
    """
    directory = Path(directory)
    
    # Find all CSV files
    if recursive:
        csv_files = []
        for path in directory.rglob(pattern):
            if path.is_file():
                csv_files.append(path)
    else:
        csv_files = list(directory.glob(pattern))
    
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return {}
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Process each CSV file
    results = {}
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            rows_before = len(df)
            
            # Remove duplicates
            df_pruned = prune_duplicates(df, subset, keep)
            rows_after = len(df_pruned)
            duplicates_removed = rows_before - rows_after
            
            # Save the pruned data back to the CSV file
            df_pruned.to_csv(csv_file, index=False)
            
            results[str(csv_file)] = duplicates_removed
            print(f"Removed {duplicates_removed} duplicates from {csv_file}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    # Print summary
    total_duplicates_removed = sum(results.values())
    print(f"\nTotal duplicates removed: {total_duplicates_removed}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prune and clean datasets')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    parser.add_argument('--output_path', type=str, help='Path to save the output CSV file')
    parser.add_argument('--prune_type', type=str, choices=['birads', 'duplicates', 'mst'],
                        required=True, help='Type of pruning to perform')
    parser.add_argument('--processed_dir', type=str, help='Directory containing processed files to prune')
    parser.add_argument('--min_birads', type=int, default=1,
                        help='Minimum BIRADS score to keep (for birads pruning)')
    parser.add_argument('--subset', type=str, nargs='+',
                        help='Column names to consider for identifying duplicates (for duplicates pruning)')
    parser.add_argument('--keep', type=str, choices=['first', 'last', 'false'], default='first',
                        help='Which duplicates to keep (for duplicates pruning)')
    parser.add_argument('--mst_data_path', type=str,
                        help='Path to the MST data folder containing XNAT folders (for mst pruning)')
    parser.add_argument('--batch', action='store_true',
                        help='Process all CSV files in the directory (for duplicates pruning)')
    parser.add_argument('--recursive', action='store_true',
                        help='Search for CSV files recursively (for batch processing)')
    
    args = parser.parse_args()
    
    if args.prune_type == 'birads':
        prune_by_birads(args.csv_path, args.output_path, args.processed_dir, args.min_birads)
    elif args.prune_type == 'duplicates':
        if args.batch:
            batch_prune_duplicates(
                Path(args.csv_path).parent,
                Path(args.csv_path).name,
                args.subset,
                args.keep if args.keep != 'false' else False,
                args.recursive
            )
        else:
            prune_duplicates_from_csv(
                args.csv_path,
                args.output_path,
                args.subset,
                args.keep if args.keep != 'false' else False
            )
    elif args.prune_type == 'mst':
        if not args.mst_data_path:
            parser.error("--mst_data_path is required for mst pruning")
        prune_mst_data(args.csv_path, args.mst_data_path, args.output_path)
