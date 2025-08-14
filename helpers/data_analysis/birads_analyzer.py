"""
BIRADS Analyzer

This module provides comprehensive analysis of BIRADS scores in datasets.
It includes functions for:
1. Counting BIRADS scores overall
2. Counting BIRADS scores by side (left/right)
3. Analyzing BIRADS distribution across different splits
4. Visualizing BIRADS distributions

Consolidated from analyze_birads.py, count_birads.py, and data_structure.py
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

def count_birads_overall(csv_file, output_txt_file=None):
    """
    Count BIRADS scores from the provided CSV file and optionally write results to a text file.
    Focus on individual BIRADS counts and aggregated scores.
    
    Args:
        csv_file (str): Path to the CSV file
        output_txt_file (str, optional): Path to the output text file
        
    Returns:
        dict: Dictionary containing BIRADS counts and aggregated counts
    """
    # Read the CSV file
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Check if BIRADS column exists
    if 'BIRADS' not in df.columns:
        raise ValueError("BIRADS column not found in the CSV file")
    
    # Count BIRADS scores
    birads_counts = Counter(df['BIRADS'])
    
    # Create aggregated categories
    df['BIRADS_aggregated'] = df['BIRADS'].apply(
        lambda x: 'Low' if x in [1, 2, 3] else 'High' if x in [4, 5, 6] else 'Unknown'
    )
    
    # Count aggregated categories
    agg_counts = Counter(df['BIRADS_aggregated'])
    
    # Count total in each group (1-3 and 4-6)
    low_count = sum(count for score, count in birads_counts.items() if score in [1, 2, 3])
    high_count = sum(count for score, count in birads_counts.items() if score in [4, 5, 6])
    
    # Prepare results dictionary
    results = {
        'BIRADS_1': birads_counts.get(1, 0),
        'BIRADS_2': birads_counts.get(2, 0),
        'BIRADS_3': birads_counts.get(3, 0),
        'BIRADS_4': birads_counts.get(4, 0),
        'BIRADS_5': birads_counts.get(5, 0),
        'BIRADS_6': birads_counts.get(6, 0),
        'BIRADS_1-3': low_count,
        'BIRADS_4-6': high_count,
        'Low': agg_counts.get('Low', 0),
        'High': agg_counts.get('High', 0),
        'Unknown': agg_counts.get('Unknown', 0)
    }
    
    # Write results to text file if output path is provided
    if output_txt_file:
        with open(output_txt_file, 'w') as f:
            f.write(f"BIRADS_1: {results['BIRADS_1']}\n")
            f.write(f"BIRADS_2: {results['BIRADS_2']}\n")
            f.write(f"BIRADS_3: {results['BIRADS_3']}\n")
            f.write(f"BIRADS_4: {results['BIRADS_4']}\n")
            f.write(f"BIRADS_5: {results['BIRADS_5']}\n")
            f.write(f"BIRADS_6: {results['BIRADS_6']}\n")
            f.write(f"BIRADS_1-3: {results['BIRADS_1-3']}\n")
            f.write(f"BIRADS_4-6: {results['BIRADS_4-6']}\n")
        
        print(f"Results written to: {output_txt_file}")
    
    return results

def count_birads_by_side(csv_file, fold=None):
    """
    Count the number of BIRADS 0-6 for left and right sides in the given CSV file.
    Optionally filter by a specific fold.
    
    Args:
        csv_file (str): Path to the CSV file
        fold (int, optional): The fold number to filter by
        
    Returns:
        dict: Dictionary with counts for each BIRADS score by side
    """
    # Initialize counters for each BIRADS score by side
    left_counts = defaultdict(int)
    right_counts = defaultdict(int)
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter by fold if specified
    if fold is not None:
        df = df[df['Fold'] == fold]
    
    # Process each row
    for _, row in df.iterrows():
        try:
            side = row['side']
            birads = int(row['BIRADS'])
            
            # Count BIRADS by side
            if side.lower() == 'left':
                left_counts[birads] += 1
            elif side.lower() == 'right':
                right_counts[birads] += 1
        except (ValueError, KeyError):
            # Skip rows with invalid values
            continue
    
    # Print the results
    print(f"BIRADS Counts by Side{' (Fold ' + str(fold) + ')' if fold is not None else ''}:")
    print("-" * 40)
    
    # Print header
    print(f"{'BIRADS':<10} {'Left':<10} {'Right':<10}")
    print("-" * 40)
    
    # Print counts for each BIRADS score (0-6)
    for birads in range(7):  # BIRADS 0-6
        left_count = left_counts.get(birads, 0)
        right_count = right_counts.get(birads, 0)
        print(f"{birads:<10} {left_count:<10} {right_count:<10}")
    
    # Print totals
    print("-" * 40)
    left_total = sum(left_counts.values())
    right_total = sum(right_counts.values())
    print(f"{'Total':<10} {left_total:<10} {right_total:<10}")
    
    return {'left': left_counts, 'right': right_counts}

def analyze_birads_distribution(csv_file, output_plot=None):
    """
    Analyze the distribution of BIRADS categories across different splits.
    
    Args:
        csv_file (str): Path to the CSV file
        output_plot (str, optional): Path to save the output plot
        
    Returns:
        tuple: (split_birads_dist, split_birads_pct, binary_dist, binary_pct)
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get value counts for each split and BIRADS combination
    split_birads_dist = df.groupby(['Split', 'BIRADS']).size().unstack(fill_value=0)
    split_birads_pct = df.groupby(['Split', 'BIRADS']).size().unstack(fill_value=0).div(df.groupby('Split').size(), axis=0) * 100
    
    # Create binary grouping (1-3 vs 4-5)
    df['BIRADS_Binary'] = df['BIRADS'].apply(lambda x: '1-3' if x <= 3 else '4-5')
    binary_dist = df.groupby(['Split', 'BIRADS_Binary']).size().unstack(fill_value=0)
    binary_pct = df.groupby(['Split', 'BIRADS_Binary']).size().unstack(fill_value=0).div(df.groupby('Split').size(), axis=0) * 100
    
    # Print results
    print("Detailed BIRADS distribution:")
    print("\nAbsolute counts per split:")
    print(split_birads_dist)
    print("\nPercentage distribution per split:")
    print(split_birads_pct.round(2))
    
    print("\nBinary BIRADS distribution (1-3 vs 4-5):")
    print("\nAbsolute counts per split:")
    print(binary_dist)
    print("\nPercentage distribution per split:")
    print(binary_pct.round(2))
    
    # Total counts per split
    print("\nTotal samples per split:")
    print(df['Split'].value_counts())
    
    # Create visualization if requested
    if output_plot:
        # Create two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot detailed distribution
        split_birads_dist.plot(kind='bar', ax=ax1)
        ax1.set_title('Detailed Distribution of BIRADS Categories by Split')
        ax1.set_xlabel('Split')
        ax1.set_ylabel('Count')
        ax1.legend(title='BIRADS')
        
        # Plot binary distribution
        binary_dist.plot(kind='bar', ax=ax2)
        ax2.set_title('Binary Distribution of BIRADS Categories by Split')
        ax2.set_xlabel('Split')
        ax2.set_ylabel('Count')
        ax2.legend(title='BIRADS Group')
        
        plt.tight_layout()
        plt.savefig(output_plot)
        print(f"Plot saved to: {output_plot}")
        plt.close()
    
    return split_birads_dist, split_birads_pct, binary_dist, binary_pct

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze BIRADS scores in datasets')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('--analysis_type', type=str, choices=['overall', 'by_side', 'distribution'], 
                        default='overall', help='Type of analysis to perform')
    parser.add_argument('--output_file', type=str, help='Path to the output file (text or image)')
    parser.add_argument('--fold', type=int, help='Fold number to filter by (for by_side analysis)')
    
    args = parser.parse_args()
    
    if args.analysis_type == 'overall':
        count_birads_overall(args.csv_file, args.output_file)
    elif args.analysis_type == 'by_side':
        count_birads_by_side(args.csv_file, args.fold)
    elif args.analysis_type == 'distribution':
        analyze_birads_distribution(args.csv_file, args.output_file)
