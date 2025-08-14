import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

def load_data():
    """
    Load the false negatives data for all sequences
    """
    # Load the converted CSV files
    t1_data = pd.read_csv('FalseNegatives/t1_false_negatives_converted.csv')
    dwi2_data = pd.read_csv('FalseNegatives/dwi2_false_negatives_converted.csv')
    t2_dwi2_data = pd.read_csv('FalseNegatives/t2_dwi2_false_negatives_converted.csv')
    t2_t1_data = pd.read_csv('FalseNegatives/t2_t1_false_negatives_converted.csv')
    
    # Add sequence identifier
    t1_data['Sequence'] = 'T1_sub'
    dwi2_data['Sequence'] = 'DWI_2'
    t2_dwi2_data['Sequence'] = 'T2+DWI_2'
    t2_t1_data['Sequence'] = 'T2+T1_sub'
    
    # Combine all data
    all_data = pd.concat([t1_data, dwi2_data, t2_dwi2_data, t2_t1_data], ignore_index=True)
    
    return all_data

def analyze_lesion_characteristics(data):
    """
    Analyze lesion characteristics by sequence
    """
    # Group by sequence
    grouped = data.groupby('Sequence')
    
    # Initialize results dictionary
    results = {
        'Sequence': [],
        'Count': [],
        'Mean Size (mm)': [],
        'Std Size (mm)': [],
        'Mass (%)': [],
        'NME (%)': [],
        'Foci (%)': [],
        'Other (%)': []
    }
    
    # Calculate statistics for each sequence
    for sequence, group in grouped:
        # Count of lesions
        count = len(group)
        
        # Size statistics
        mean_size = group['Diameter'].mean()
        std_size = group['Diameter'].std()
        
        # Type percentages
        type_counts = Counter(group['type'])
        mass_pct = type_counts.get('Mass', 0) / count * 100
        nme_pct = type_counts.get('NME', 0) / count * 100
        foci_pct = type_counts.get('Foci', 0) / count * 100
        other_pct = type_counts.get('Other', 0) / count * 100
        
        # Add to results
        results['Sequence'].append(sequence)
        results['Count'].append(count)
        results['Mean Size (mm)'].append(mean_size)
        results['Std Size (mm)'].append(std_size)
        results['Mass (%)'].append(mass_pct)
        results['NME (%)'].append(nme_pct)
        results['Foci (%)'].append(foci_pct)
        results['Other (%)'].append(other_pct)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def statistical_tests(data):
    """
    Perform statistical tests to compare lesion characteristics across sequences
    """
    # Group by sequence
    grouped = data.groupby('Sequence')
    
    # Extract diameters by sequence
    diameters_by_sequence = {sequence: group['Diameter'].values for sequence, group in grouped}
    
    # Perform ANOVA for diameter
    f_stat, p_value = stats.f_oneway(*diameters_by_sequence.values())
    
    print(f"ANOVA for lesion diameter across sequences: F={f_stat:.2f}, p={p_value:.4f}")
    
    # If ANOVA is significant, perform post-hoc tests
    if p_value < 0.05:
        print("\nPost-hoc tests (Tukey HSD):")
        sequences = list(diameters_by_sequence.keys())
        for i, seq1 in enumerate(sequences):
            for seq2 in sequences[i+1:]:
                t_stat, p_val = stats.ttest_ind(diameters_by_sequence[seq1], diameters_by_sequence[seq2], equal_var=False)
                print(f"{seq1} vs {seq2}: t={t_stat:.2f}, p={p_val:.4f}")
    
    # Chi-square test for lesion type distribution
    print("\nChi-square test for lesion type distribution:")
    
    # Create contingency table
    type_counts = {}
    for sequence, group in grouped:
        type_counts[sequence] = Counter(group['type'])
    
    # Convert to DataFrame for easier visualization
    contingency_df = pd.DataFrame({
        sequence: {lesion_type: type_counts[sequence].get(lesion_type, 0) 
                  for lesion_type in ['Mass', 'NME', 'Foci', 'Other']}
        for sequence in type_counts.keys()
    })
    
    print(contingency_df)
    
    # Perform chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_df)
    print(f"Chi-square test: chi2={chi2:.2f}, p={p:.4f}, dof={dof}")

def create_visualizations(data, results_df):
    """
    Create visualizations for the analysis
    """
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Boxplot of lesion sizes by sequence
    sns.boxplot(x='Sequence', y='Diameter', data=data, ax=axes[0, 0])
    axes[0, 0].set_title('Lesion Size Distribution by Sequence')
    axes[0, 0].set_ylabel('Diameter (mm)')
    axes[0, 0].set_xlabel('Sequence')
    
    # 2. Bar plot of lesion type distribution by sequence
    lesion_types = ['Mass (%)', 'NME (%)', 'Foci (%)', 'Other (%)']
    results_melted = pd.melt(results_df, id_vars=['Sequence'], value_vars=lesion_types,
                            var_name='Lesion Type', value_name='Percentage')
    
    sns.barplot(x='Sequence', y='Percentage', hue='Lesion Type', data=results_melted, ax=axes[0, 1])
    axes[0, 1].set_title('Lesion Type Distribution by Sequence')
    axes[0, 1].set_ylabel('Percentage (%)')
    axes[0, 1].set_xlabel('Sequence')
    
    # 3. Histogram of lesion sizes
    sns.histplot(data=data, x='Diameter', hue='Sequence', element='step', kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Histogram of Lesion Sizes')
    axes[1, 0].set_xlabel('Diameter (mm)')
    axes[1, 0].set_ylabel('Count')
    
    # 4. Scatter plot of lesion size vs. type by sequence
    sns.scatterplot(x='Diameter', y='type', hue='Sequence', data=data, ax=axes[1, 1])
    axes[1, 1].set_title('Lesion Size vs. Type by Sequence')
    axes[1, 1].set_xlabel('Diameter (mm)')
    axes[1, 1].set_ylabel('Lesion Type')
    
    plt.tight_layout()
    plt.savefig('lesion_characteristics_analysis.png', dpi=300)
    plt.close()

def main():
    """
    Main function to run the analysis
    """
    print("Loading data...")
    data = load_data()
    
    print("\nAnalyzing lesion characteristics...")
    results_df = analyze_lesion_characteristics(data)
    
    # Display results in format similar to the LaTeX table
    print("\nLesion Characteristics by Sequence:")
    formatted_results = results_df.copy()
    formatted_results['Mean Size (mm)'] = formatted_results.apply(
        lambda row: f"{row['Mean Size (mm)']:.0f}±{row['Std Size (mm)']:.0f}", axis=1
    )
    formatted_results = formatted_results.drop(columns=['Std Size (mm)'])
    print(formatted_results.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    
    print("\nPerforming statistical tests...")
    statistical_tests(data)
    
    print("\nCreating visualizations...")
    create_visualizations(data, results_df)
    
    print("\nAnalysis complete. Results saved to 'lesion_characteristics_analysis.png'")
    
    # Save results to CSV
    results_df.to_csv('lesion_characteristics_results.csv', index=False)
    print("Results saved to 'lesion_characteristics_results.csv'")

if __name__ == "__main__":
    main()
