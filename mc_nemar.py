#!/usr/bin/env python3
"""
McNemar's Test for Model Comparison
Statistical evaluation employed McNemar's test for accuracy comparisons
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def mcnemar_test(y_true, y_pred1, y_pred2):
    """Perform McNemar's test for comparing two classifiers"""
    correct1 = (y_pred1 == y_true).astype(int)
    correct2 = (y_pred2 == y_true).astype(int)
    
    table = np.zeros((2, 2), dtype=int)
    for i in range(len(y_true)):
        table[correct1[i], correct2[i]] += 1
    
    b = table[0, 1]  # Model 1 wrong, Model 2 correct
    c = table[1, 0]  # Model 1 correct, Model 2 wrong
    
    if b + c == 0:
        chi2_stat, p_value = 0.0, 1.0
    else:
        chi2_stat = (abs(b - c) - 1)**2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
    
    return {
        'accuracy_model1': accuracy_score(y_true, y_pred1),
        'accuracy_model2': accuracy_score(y_true, y_pred2),
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'both_correct': int(table[1, 1]),
        'both_wrong': int(table[0, 0]),
        'discordant_pairs': int(b + c)
    }


def main():
    print("=" * 80)
    print("MCNEMAR'S TEST: MODEL COMPARISON")
    print("=" * 80)
    
    # Configuration
    results_dir = Path("mcnemar_results")
    results_dir.mkdir(exist_ok=True)
    
    # Data paths
    model1_path = Path("results/model1/results.csv")
    model2_path = Path("results/model2/results.csv")
    
    print(f"\nComparing models:")
    print(f"  Model 1: {model1_path}")
    print(f"  Model 2: {model2_path}\n")
    
    # Load data
    try:
        data1 = pd.read_csv(model1_path)
        data2 = pd.read_csv(model2_path)
        print(f"Loaded {len(data1)} and {len(data2)} samples")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Match common patients
    common_uids = set(data1['UID']) & set(data2['UID'])
    if len(common_uids) == 0:
        print("Error: No common patients found")
        return
    
    data1 = data1[data1['UID'].isin(common_uids)].sort_values('UID').reset_index(drop=True)
    data2 = data2[data2['UID'].isin(common_uids)].sort_values('UID').reset_index(drop=True)
    
    y_true = data1['GT'].values
    y_pred1 = (data1['NN_pred'].values >= 0.5).astype(int)
    y_pred2 = (data2['NN_pred'].values >= 0.5).astype(int)
    
    print(f"Common patients: {len(common_uids)}")
    print(f"Total samples: {len(y_true)}\n")
    
    # McNemar's test
    result = mcnemar_test(y_true, y_pred1, y_pred2)
    
    # Results
    print("RESULTS:")
    print(f"  Accuracy Model 1:     {result['accuracy_model1']:.4f}")
    print(f"  Accuracy Model 2:     {result['accuracy_model2']:.4f}")
    print(f"  Chi-square:           {result['chi2_statistic']:.4f}")
    print(f"  P-value:              {result['p_value']:.4f}")
    
    if result['p_value'] < 0.05:
        print(f"  Significance:         p < 0.05 *")
    else:
        print(f"  Significance:         ns")
    
    # Save results
    pd.DataFrame([result]).to_csv(results_dir / "mcnemar_results.csv", index=False)
    
    report = [
        "MCNEMAR'S TEST REPORT",
        "=" * 60,
        f"Samples: {len(y_true)}",
        f"Accuracy Model 1: {result['accuracy_model1']:.4f}",
        f"Accuracy Model 2: {result['accuracy_model2']:.4f}",
        f"Chi-square: {result['chi2_statistic']:.4f}",
        f"P-value: {result['p_value']:.4f}",
        f"Discordant pairs: {result['discordant_pairs']}",
        "=" * 60
    ]
    
    with open(results_dir / "mcnemar_report.txt", 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nResults saved to: {results_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
