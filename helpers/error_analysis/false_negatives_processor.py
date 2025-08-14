"""
False Negatives Processor

This module processes false negative cases by:
1. Extracting data from CSV files
2. Adding additional information from other CSV files
3. Copying relevant XNAT folders and files
4. Updating CSV files with additional information

Consolidated from process_false_negatives.py and update_false_negatives.py
"""

import os
import csv
import shutil
import glob
import pandas as pd
from pathlib import Path

def empty_directory(directory):
    """
    Empty a directory by removing all files and subdirectories
    
    Args:
        directory (str): Path to the directory to empty
    """
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)
    print(f"Emptied directory: {directory}")

def get_birads_data(uid, birads_data, birads_csv_path="dataset_birads_with_paths.csv"):
    """
    Get XNAT, BIRADS right, BIRADS left for a given UID from dataset_birads_with_paths.csv
    
    Args:
        uid (str): The UID to look up
        birads_data (dict): Dictionary to cache results
        birads_csv_path (str): Path to the CSV file with BIRADS data
        
    Returns:
        tuple: (xnat, birads_right, birads_left)
    """
    # Check if we already have the data for this UID
    if uid in birads_data:
        return birads_data[uid]
    
    # Otherwise, search for it in the CSV file
    with open(birads_csv_path, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 5:
                continue  # Skip rows with insufficient columns
            
            record_id = row[0]
            if record_id == uid:
                xnat = row[1]
                birads_right = row[2]
                birads_left = row[3]
                birads_data[uid] = (xnat, birads_right, birads_left)
                return xnat, birads_right, birads_left
    
    # If not found, return empty values
    return "", "", ""

def get_radiological_data(uid, radiological_data, rad_csv_path="radiological_data.csv"):
    """
    Get radiological_report and radiological_evaluation for a given UID
    
    Args:
        uid (str): The UID to look up
        radiological_data (dict): Dictionary to cache results
        rad_csv_path (str): Path to the CSV file with radiological data
        
    Returns:
        tuple: (radiological_report, radiological_evaluation)
    """
    # Check if we already have the data for this UID
    if uid in radiological_data:
        return radiological_data[uid]
    
    # Otherwise, search for it in the CSV file
    with open(rad_csv_path, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 3:
                continue  # Skip rows with insufficient columns
            
            record_id = row[0]
            if record_id == uid:
                rad_report = row[1]
                rad_eval = row[2]
                radiological_data[uid] = (rad_report, rad_eval)
                return rad_report, rad_eval
    
    # If not found, return empty values
    return "", ""

def process_false_negatives(csv_path, mst_data_path=None, vdce_sparse_path=None, copy_files=True):
    """
    Process a CSV file containing false negative cases:
    1. Add XNAT, BIRADS, and radiological data
    2. Optionally copy XNAT folders and/or T1 files
    
    Args:
        csv_path (str): Path to the CSV file
        mst_data_path (str, optional): Path to the MST data folder for copying XNAT folders
        vdce_sparse_path (str, optional): Path to the vDCE sparse folder for copying T1 files
        copy_files (bool): Whether to copy files or just update the CSV
        
    Returns:
        pd.DataFrame: The processed DataFrame
    """
    # Extract the base name of the CSV file (without extension)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join("FalseNegatives", base_name)
    if copy_files:
        os.makedirs(output_dir, exist_ok=True)
        # Empty the directory if it exists
        if os.path.exists(output_dir) and os.listdir(output_dir):
            empty_directory(output_dir)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Dictionary to store XNAT, BIRADS right, BIRADS left for each Record ID
    birads_data = {}
    
    # Dictionary to store radiological_report and radiological_evaluation for each record_id
    radiological_data = {}
    
    # Process each row
    for index, row in df.iterrows():
        # Extract UID
        uid = row['UID'] if 'UID' in df.columns else None
        
        if uid:
            # Get XNAT, BIRADS right, BIRADS left
            xnat, birads_right, birads_left = get_birads_data(uid, birads_data)
            
            # Get radiological_report and radiological_evaluation
            rad_report, rad_eval = get_radiological_data(uid, radiological_data)
            
            # Update the DataFrame
            df.at[index, 'XNAT'] = xnat
            df.at[index, 'BIRADS right'] = birads_right
            df.at[index, 'BIRADS left'] = birads_left
            df.at[index, 'radiological_report'] = rad_report
            df.at[index, 'radiological_evaluation'] = rad_eval
            
            # Copy files if requested
            if copy_files and xnat:
                # Copy XNAT folder if mst_data_path is provided
                if mst_data_path:
                    source_dir = os.path.join(mst_data_path, xnat)
                    dest_dir = os.path.join(output_dir, xnat)
                    if os.path.exists(source_dir):
                        try:
                            shutil.copytree(source_dir, dest_dir)
                            print(f"Copied {source_dir} to {dest_dir}")
                        except Exception as e:
                            print(f"Error copying {source_dir} to {dest_dir}: {e}")
                    else:
                        print(f"Warning: Source directory {source_dir} does not exist")
                
                # Copy T1 files if vdce_sparse_path is provided
                if vdce_sparse_path:
                    xnat_dir = os.path.join(vdce_sparse_path, xnat)
                    if os.path.exists(xnat_dir) and os.path.isdir(xnat_dir):
                        # Look for files with the prefix "t1w_fs_sub_2_" in the XNAT directory
                        source_pattern = os.path.join(xnat_dir, f"t1w_fs_sub_2_*")
                        source_files = glob.glob(source_pattern)
                        
                        if source_files:
                            for source_file in source_files:
                                # Copy the file to the output directory
                                dest_file = os.path.join(output_dir, os.path.basename(source_file))
                                shutil.copy2(source_file, dest_file)
                                print(f"Copied {source_file} to {dest_file}")
                        else:
                            print(f"Warning: No t1w_fs_sub_2_ files found in {xnat_dir}")
                    else:
                        print(f"Warning: Directory not found for XNAT {xnat}")
    
    # Save the updated DataFrame
    df.to_csv(csv_path, index=False)
    print(f"Updated CSV file: {csv_path}")
    
    return df

def process_all_false_negatives(directory="FalseNegatives", mst_data_path=None, vdce_sparse_path=None, copy_files=True):
    """
    Process all CSV files in the specified directory
    
    Args:
        directory (str): Path to the directory containing CSV files
        mst_data_path (str, optional): Path to the MST data folder for copying XNAT folders
        vdce_sparse_path (str, optional): Path to the vDCE sparse folder for copying T1 files
        copy_files (bool): Whether to copy files or just update the CSV
        
    Returns:
        list: List of processed DataFrames
    """
    # Get all CSV files in the directory
    csv_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                if f.endswith('.csv') and os.path.isfile(os.path.join(directory, f))]
    
    if not csv_files:
        print(f"No CSV files found in the {directory} directory")
        return []
    
    print(f"Found {len(csv_files)} CSV files.")
    
    # Process each CSV file
    processed_dfs = []
    for csv_file in csv_files:
        try:
            df = process_false_negatives(csv_file, mst_data_path, vdce_sparse_path, copy_files)
            processed_dfs.append(df)
            print(f"Successfully processed {csv_file}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    return processed_dfs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process false negative cases')
    parser.add_argument('--directory', type=str, default="FalseNegatives",
                        help='Directory containing CSV files')
    parser.add_argument('--mst_data_path', type=str, 
                        default=r"/path/to/mst/data",
                        help='Path to the MST data folder')
    parser.add_argument('--vdce_sparse_path', type=str,
                        default=r"/path/to/vdce/sparse",
                        help='Path to the vDCE sparse folder')
    parser.add_argument('--no_copy', action='store_true',
                        help='Do not copy files, just update the CSV')
    
    args = parser.parse_args()
    
    process_all_false_negatives(
        args.directory, 
        args.mst_data_path, 
        args.vdce_sparse_path, 
        not args.no_copy
    )
