"""
DeLong's Test with FDR Correction for Cross-Validation Results
Aggregates results across folds and performs statistical comparisons
"""

"""
Multi-sequence MST Statistical Analysis Script

Statistical comparison of sequence combinations using DeLong's test with FDR correction
for multi-sequence breast MRI classification.

Original MST framework by Müller-Franzes et al.
Repository: https://github.com/mueller-franzes/MST
Paper: https://arxiv.org/abs/2411.15802
Licensed under MIT License

This statistical analysis pipeline is our main contribution.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import os
from itertools import combinations
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

class DeLongFDRAnalyzer:
    """
    Performs DeLong's test with FDR correction on cross-validation results
    """
    
    def __init__(self, base_path='cross_val_results'):
        """
        Initialize analyzer
        
        Args:
            base_path: Path to cross-validation results directory
        """
        self.base_path = base_path
        self.sequence_names = {
            'dwi2': 'DWI',
            't1': 'T1',
            't2_dwi2': 'T2+DWI',
            't2_dwi2_t1': 'T2+DWI+T1',
            't2_t1': 'T2+T1'
        }
        self.aggregated_data = {}
        
    def load_fold_data(self, sequence_key):
        """
        Load and aggregate data across all folds for a sequence
        
        Args:
            sequence_key: Key for the sequence (e.g., 't1', 'dwi2')
            
        Returns:
            DataFrame with aggregated results across folds
        """
        sequence_path = os.path.join(self.base_path, sequence_key)
        
        if not os.path.exists(sequence_path):
            print(f"Warning: Path {sequence_path} does not exist")
            return None
        
        all_fold_data = []
        
        # Find all fold directories
        fold_dirs = [d for d in os.listdir(sequence_path) if d.startswith('fold_')]
        fold_dirs.sort()  # Ensure consistent ordering
        
        print(f"  Loading {len(fold_dirs)} folds for {sequence_key}")
        
        for fold_dir in fold_dirs:
            fold_path = os.path.join(sequence_path, fold_dir, 'results.csv')
            
            if os.path.exists(fold_path):
                fold_data = pd.read_csv(fold_path)
                fold_data['fold'] = fold_dir
                all_fold_data.append(fold_data)
            else:
                print(f"    Warning: {fold_path} not found")
        
        if not all_fold_data:
            print(f"  No fold data found for {sequence_key}")
            return None
        
        # Combine all folds
        combined_data = pd.concat(all_fold_data, ignore_index=True)
        
        print(f"  Loaded {len(combined_data)} samples across {len(fold_dirs)} folds")
        print(f"  Unique patients: {combined_data['UID'].nunique()}")
        print(f"  Positive cases: {combined_data['GT'].sum()}")
        print(f"  Negative cases: {len(combined_data) - combined_data['GT'].sum()}")
        
        return combined_data
    
    def load_all_sequences(self):
        """
        Load data for all available sequences
        """
        print("Loading cross-validation data...")
        
        for seq_key in self.sequence_names.keys():
            print(f"\nLoading sequence: {seq_key}")
            data = self.load_fold_data(seq_key)
            if data is not None:
                self.aggregated_data[seq_key] = data
        
        print(f"\nSuccessfully loaded {len(self.aggregated_data)} sequences")
        return len(self.aggregated_data) > 0
    
    def calculate_auc_with_ci(self, y_true, y_scores, confidence_level=0.95):
        """
        Calculate AUC with confidence interval using bootstrap
        
        Args:
            y_true: True binary labels
            y_scores: Prediction scores
            confidence_level: Confidence level for CI
            
        Returns:
            Tuple of (AUC, CI_lower, CI_upper)
        """
        n_bootstraps = 1000
        rng = np.random.RandomState(42)
        
        bootstrapped_scores = []
        
        for _ in range(n_bootstraps):
            # Bootstrap sample
            indices = rng.randint(0, len(y_scores), len(y_scores))
            if len(np.unique(y_true[indices])) < 2:
                continue
            
            score = roc_auc_score(y_true[indices], y_scores[indices])
            bootstrapped_scores.append(score)
        
        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        
        # Calculate confidence interval
        alpha = 1.0 - confidence_level
        ci_lower = np.percentile(sorted_scores, (alpha/2.0) * 100)
        ci_upper = np.percentile(sorted_scores, (1.0 - alpha/2.0) * 100)
        
        auc = roc_auc_score(y_true, y_scores)
        
        return auc, ci_lower, ci_upper
    
    def delong_test(self, y_true, y_scores1, y_scores2):
        """
        Perform DeLong's test for comparing two AUC values
        
        Args:
            y_true: True binary labels
            y_scores1: Prediction scores for model 1
            y_scores2: Prediction scores for model 2
            
        Returns:
            Tuple of (z_statistic, p_value)
        """
        # Calculate AUCs
        auc1 = roc_auc_score(y_true, y_scores1)
        auc2 = roc_auc_score(y_true, y_scores2)
        
        # DeLong's method for variance estimation
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        # Calculate structural components
        def structural_components(y_true, y_scores):
            pos_scores = y_scores[y_true == 1]
            neg_scores = y_scores[y_true == 0]
            
            # V10: For each positive case, fraction of negative cases with lower score
            v10 = np.array([np.mean(neg_scores < pos_score) + 0.5 * np.mean(neg_scores == pos_score) 
                           for pos_score in pos_scores])
            
            # V01: For each negative case, fraction of positive cases with higher score  
            v01 = np.array([np.mean(pos_scores > neg_score) + 0.5 * np.mean(pos_scores == neg_score)
                           for neg_score in neg_scores])
            
            return v10, v01
        
        v10_1, v01_1 = structural_components(y_true, y_scores1)
        v10_2, v01_2 = structural_components(y_true, y_scores2)
        
        # Calculate covariances
        s10_1 = np.var(v10_1, ddof=1) if len(v10_1) > 1 else 0
        s01_1 = np.var(v01_1, ddof=1) if len(v01_1) > 1 else 0
        s10_2 = np.var(v10_2, ddof=1) if len(v10_2) > 1 else 0
        s01_2 = np.var(v01_2, ddof=1) if len(v01_2) > 1 else 0
        
        # Covariance between the two AUCs
        s10_12 = np.cov(v10_1, v10_2)[0, 1] if len(v10_1) > 1 else 0
        s01_12 = np.cov(v01_1, v01_2)[0, 1] if len(v01_1) > 1 else 0
        
        # Variance of AUC difference
        var_auc_diff = (s10_1 / n_pos + s01_1 / n_neg + 
                       s10_2 / n_pos + s01_2 / n_neg - 
                       2 * s10_12 / n_pos - 2 * s01_12 / n_neg)
        
        if var_auc_diff <= 0:
            return 0.0, 1.0
        
        # Z-statistic
        z_stat = (auc1 - auc2) / np.sqrt(var_auc_diff)
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return z_stat, p_value
    
    def perform_delong_comparisons(self):
        """
        Perform all pairwise DeLong comparisons with FDR correction
        
        Returns:
            DataFrame with comparison results
        """
        if len(self.aggregated_data) < 2:
            print("Error: Need at least 2 sequences for comparisons")
            return pd.DataFrame()
        
        print(f"\nPerforming DeLong's tests...")
        
        # Generate all pairwise combinations
        sequence_pairs = list(combinations(self.aggregated_data.keys(), 2))
        print(f"Total comparisons: {len(sequence_pairs)}")
        
        results = []
        p_values = []
        
        for seq1_key, seq2_key in sequence_pairs:
            seq1_name = self.sequence_names[seq1_key]
            seq2_name = self.sequence_names[seq2_key]
            
            print(f"  Comparing {seq1_name} vs {seq2_name}")
            
            # Get data for both sequences
            df1 = self.aggregated_data[seq1_key]
            df2 = self.aggregated_data[seq2_key]
            
            # Find common patients (UIDs)
            common_uids = set(df1['UID']) & set(df2['UID'])
            
            if len(common_uids) == 0:
                print(f"    Warning: No common patients between {seq1_key} and {seq2_key}")
                continue
            
            # Get data for common patients
            df1_common = df1[df1['UID'].isin(common_uids)].copy()
            df2_common = df2[df2['UID'].isin(common_uids)].copy()
            
            # Sort by UID to ensure alignment
            df1_common = df1_common.sort_values('UID').reset_index(drop=True)
            df2_common = df2_common.sort_values('UID').reset_index(drop=True)
            
            # Extract labels and predictions
            y_true = df1_common['GT'].values  # Should be same for both
            y_scores1 = df1_common['NN_pred'].values
            y_scores2 = df2_common['NN_pred'].values
            
            # Calculate individual AUCs with CIs
            auc1, auc1_ci_lower, auc1_ci_upper = self.calculate_auc_with_ci(y_true, y_scores1)
            auc2, auc2_ci_lower, auc2_ci_upper = self.calculate_auc_with_ci(y_true, y_scores2)
            
            # Perform DeLong test
            try:
                z_stat, p_value = self.delong_test(y_true, y_scores1, y_scores2)
                p_values.append(p_value)
            except Exception as e:
                print(f"    Error in DeLong test: {e}")
                z_stat, p_value = np.nan, 1.0
                p_values.append(1.0)
            
            result = {
                'Sequence_1': seq1_name,
                'Sequence_2': seq2_name,
                'Sequence_1_Key': seq1_key,
                'Sequence_2_Key': seq2_key,
                'N_Common_Patients': len(common_uids),
                'N_Common_Samples': len(df1_common),
                'N_Positive': y_true.sum(),
                'N_Negative': len(y_true) - y_true.sum(),
                'AUC_1': auc1,
                'AUC_1_CI_Lower': auc1_ci_lower,
                'AUC_1_CI_Upper': auc1_ci_upper,
                'AUC_2': auc2,
                'AUC_2_CI_Lower': auc2_ci_lower,
                'AUC_2_CI_Upper': auc2_ci_upper,
                'AUC_Difference': auc1 - auc2,
                'DeLong_Z_Statistic': z_stat,
                'DeLong_P_Value': p_value
            }
            
            results.append(result)
        
        if not results:
            print("No valid comparisons could be performed")
            return pd.DataFrame()
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Apply FDR correction using Benjamini-Hochberg method
        print(f"\nApplying FDR correction using Benjamini-Hochberg method...")
        
        # Remove any NaN p-values for correction
        valid_p_values = [p for p in p_values if not np.isnan(p)]
        
        if len(valid_p_values) > 0:
            # Apply FDR correction
            rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                valid_p_values, 
                alpha=0.05, 
                method='fdr_bh'  # Benjamini-Hochberg FDR correction
            )
            
            # Add corrected p-values back to results
            p_corrected_full = []
            valid_idx = 0
            
            for p in p_values:
                if np.isnan(p):
                    p_corrected_full.append(np.nan)
                else:
                    p_corrected_full.append(p_corrected[valid_idx])
                    valid_idx += 1
            
            results_df['FDR_P_Value'] = p_corrected_full
            results_df['FDR_Significant'] = results_df['FDR_P_Value'] < 0.05
            results_df['Bonferroni_P_Value'] = np.minimum(np.array(p_values) * len(p_values), 1.0)
            results_df['Bonferroni_Significant'] = results_df['Bonferroni_P_Value'] < 0.05
        else:
            results_df['FDR_P_Value'] = np.nan
            results_df['FDR_Significant'] = False
            results_df['Bonferroni_P_Value'] = np.nan
            results_df['Bonferroni_Significant'] = False
        
        # Add significance indicators
        results_df['Significant_0.05'] = results_df['DeLong_P_Value'] < 0.05
        results_df['Significant_0.01'] = results_df['DeLong_P_Value'] < 0.01
        results_df['Significant_0.001'] = results_df['DeLong_P_Value'] < 0.001
        
        # Add method information
        results_df['Statistical_Method'] = 'DeLong'
        results_df['Multiple_Testing_Correction'] = 'Benjamini-Hochberg FDR'
        results_df['N_Comparisons'] = len(results)
        
        return results_df
    
    def generate_summary_report(self, results_df):
        """
        Generate a summary report of the analysis
        
        Args:
            results_df: DataFrame with comparison results
            
        Returns:
            String with formatted report
        """
        report = []
        report.append("=" * 80)
        report.append("DELONG'S TEST WITH FDR CORRECTION - SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Dataset summary
        report.append("DATASET SUMMARY:")
        report.append("-" * 40)
        for seq_key, data in self.aggregated_data.items():
            seq_name = self.sequence_names[seq_key]
            n_patients = data['UID'].nunique()
            n_samples = len(data)
            n_positive = data['GT'].sum()
            n_negative = len(data) - n_positive
            
            # Calculate AUC
            auc, auc_ci_lower, auc_ci_upper = self.calculate_auc_with_ci(
                data['GT'].values, data['NN_pred'].values
            )
            
            report.append(f"{seq_name:15} | Patients: {n_patients:4d} | Samples: {n_samples:4d} | "
                         f"Pos: {n_positive:3d} | Neg: {n_negative:3d} | "
                         f"AUC: {auc:.3f} (95% CI: {auc_ci_lower:.3f}-{auc_ci_upper:.3f})")
        
        report.append("")
        
        # Statistical analysis summary
        report.append("STATISTICAL ANALYSIS:")
        report.append("-" * 40)
        report.append(f"Total pairwise comparisons: {len(results_df)}")
        report.append(f"Statistical method: DeLong's test for AUC comparison")
        report.append(f"Multiple testing correction: Benjamini-Hochberg FDR")
        report.append("")
        
        # Significance summary
        n_sig_uncorrected = results_df['Significant_0.05'].sum()
        n_sig_fdr = results_df['FDR_Significant'].sum()
        n_sig_bonferroni = results_df['Bonferroni_Significant'].sum()
        
        report.append("SIGNIFICANCE SUMMARY:")
        report.append("-" * 40)
        report.append(f"Significant at p < 0.05 (uncorrected): {n_sig_uncorrected}/{len(results_df)}")
        report.append(f"Significant after FDR correction: {n_sig_fdr}/{len(results_df)}")
        report.append(f"Significant after Bonferroni correction: {n_sig_bonferroni}/{len(results_df)}")
        report.append("")
        
        # Detailed results
        report.append("DETAILED COMPARISON RESULTS:")
        report.append("-" * 80)
        report.append(f"{'Comparison':<25} {'AUC Diff':<10} {'P-value':<10} {'FDR P':<10} {'Significant':<12}")
        report.append("-" * 80)
        
        for _, row in results_df.iterrows():
            comparison = f"{row['Sequence_1']} vs {row['Sequence_2']}"
            auc_diff = row['AUC_Difference']
            p_val = row['DeLong_P_Value']
            fdr_p = row['FDR_P_Value']
            sig_status = "***" if row['FDR_Significant'] else ("*" if row['Significant_0.05'] else "ns")
            
            report.append(f"{comparison:<25} {auc_diff:>+8.3f} {p_val:>9.3f} {fdr_p:>9.3f} {sig_status:<12}")
        
        report.append("")
        report.append("Legend: *** = FDR significant, * = nominally significant, ns = not significant")
        report.append("")
        
        # Libraries used
        report.append("LIBRARIES USED FOR STATISTICAL ANALYSIS:")
        report.append("-" * 40)
        report.append("• scipy.stats - Normal distribution for DeLong's test")
        report.append("• statsmodels.stats.multitest - Benjamini-Hochberg FDR correction")
        report.append("• sklearn.metrics - ROC AUC calculation")
        report.append("• numpy - Numerical computations")
        report.append("")
        
        # Citation information
        report.append("RECOMMENDED CITATIONS:")
        report.append("-" * 40)
        report.append("• DeLong's test: DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988)")
        report.append("• FDR correction: Benjamini, Y., & Hochberg, Y. (1995)")
        report.append("• Implementation: scipy (Virtanen et al., 2020), statsmodels (Seabold & Perktold, 2010)")
        report.append("")
        
        return "\n".join(report)
    
    def run_analysis(self, output_dir='delong_fdr_results'):
        """
        Run the complete analysis pipeline
        
        Args:
            output_dir: Directory to save results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        if not self.load_all_sequences():
            print("Error: Could not load sequence data")
            return
        
        # Perform comparisons
        results_df = self.perform_delong_comparisons()
        
        if results_df.empty:
            print("Error: No valid comparisons performed")
            return
        
        # Save detailed results
        results_file = os.path.join(output_dir, 'delong_fdr_detailed_results.csv')
        results_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")
        
        # Generate and save summary report
        summary_report = self.generate_summary_report(results_df)
        report_file = os.path.join(output_dir, 'delong_fdr_summary_report.txt')
        
        with open(report_file, 'w') as f:
            f.write(summary_report)
        
        print(f"Summary report saved to: {report_file}")
        print("\n" + summary_report)
        
        return results_df

def main():
    """
    Main function to run the analysis
    """
    print("DeLong's Test with FDR Correction Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = DeLongFDRAnalyzer()
    
    # Run analysis
    results = analyzer.run_analysis()
    
    if results is not None:
        print(f"\nAnalysis completed successfully!")
        print(f"Results shape: {results.shape}")
    else:
        print("Analysis failed!")

if __name__ == "__main__":
    main()
