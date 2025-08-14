import numpy as np 
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib

def auc_bootstrapping(y_true, y_score, bootstrapping=1000, drop_intermediate=False):
    tprs, aucs, thrs = [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    rand_idxs = np.random.randint(0, len(y_true), size=(bootstrapping, len(y_true))) # Note: with replacement 
    for rand_idx in rand_idxs:
        y_true_set = y_true[rand_idx]
        y_score_set = y_score[rand_idx]
        fpr, tpr, thresholds = roc_curve(y_true_set, y_score_set, drop_intermediate=drop_intermediate)
        tpr_interp = np.interp(mean_fpr, fpr, tpr) # must be interpolated to gain constant/equal fpr positions
        tprs.append(tpr_interp) 
        aucs.append(auc(fpr, tpr))
        optimal_idx = np.argmax(tpr - fpr)
        thrs.append(thresholds[optimal_idx])
    return tprs, aucs, thrs, mean_fpr

def find_sensitivity_point(fprs, tprs, thrs, target_sensitivity=0.90):
    # Find the point closest to target sensitivity
    idx = np.argmin(np.abs(tprs - target_sensitivity))
    return {
        'sensitivity': tprs[idx],
        'specificity': 1 - fprs[idx],
        'threshold': thrs[idx],
        'index': idx
    }

def find_misclassified_samples(y_true, y_score, threshold, uids=None):
    """Find misclassified samples at a given threshold."""
    y_pred = (y_score >= threshold).astype(int)
    misclassified_indices = np.where(y_true != y_pred)[0]
    false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]
    
    result = {
        'misclassified_indices': misclassified_indices,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }
    
    # If UIDs are provided, include them in the result
    if uids is not None:
        result['misclassified_uids'] = [uids[i] for i in misclassified_indices] if len(misclassified_indices) > 0 else []
        result['false_positive_uids'] = [uids[i] for i in false_positives] if len(false_positives) > 0 else []
        result['false_negative_uids'] = [uids[i] for i in false_negatives] if len(false_negatives) > 0 else []
    
    return result

def plot_roc_curve(y_true, y_score, axis, bootstrapping=1000, drop_intermediate=False, fontdict={}, name='ROC', color='b', show_wp=False, uids=None):
    # ----------- Bootstrapping ------------
    tprs, aucs, thrs, mean_fpr = auc_bootstrapping(y_true, y_score, bootstrapping, drop_intermediate)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0        
    std_tpr = np.std(tprs, axis=0, ddof=1)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    # ------ Averaged based on bootspraping ------
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs, ddof=1)
    print(f"AUC std: {std_auc:.3f}")
 
    # --------- Specific Case -------------
    fprs, tprs, thrs = roc_curve(y_true, y_score, drop_intermediate=drop_intermediate)
    auc_val = auc(fprs, tprs)
    opt_idx = np.argmax(tprs - fprs)
    opt_tpr = tprs[opt_idx]
    opt_fpr = fprs[opt_idx]

    # Find sensitivity points at 0.9, 0.95, 0.97, and 0.99
    sens_point_90 = find_sensitivity_point(fprs, tprs, thrs, target_sensitivity=0.90)
    sens_point_95 = find_sensitivity_point(fprs, tprs, thrs, target_sensitivity=0.95)
    sens_point_97 = find_sensitivity_point(fprs, tprs, thrs, target_sensitivity=0.97)
    sens_point_99 = find_sensitivity_point(fprs, tprs, thrs, target_sensitivity=0.99)
    
    # Get specificity values at these sensitivity points
    sens_90_tpr = sens_point_90['sensitivity']
    sens_90_fpr = 1 - sens_point_90['specificity']
    sens_90_threshold = sens_point_90['threshold']
    
    sens_95_tpr = sens_point_95['sensitivity']
    sens_95_fpr = 1 - sens_point_95['specificity']
    sens_95_threshold = sens_point_95['threshold']
    
    sens_97_tpr = sens_point_97['sensitivity']
    sens_97_fpr = 1 - sens_point_97['specificity']
    sens_97_threshold = sens_point_97['threshold']
    
    sens_99_tpr = sens_point_99['sensitivity']
    sens_99_fpr = 1 - sens_point_99['specificity']
    sens_99_threshold = sens_point_99['threshold']
    
    # Find misclassified samples at each threshold
    misclassified_90 = find_misclassified_samples(y_true, y_score, sens_90_threshold, uids)
    misclassified_95 = find_misclassified_samples(y_true, y_score, sens_95_threshold, uids)
    misclassified_97 = find_misclassified_samples(y_true, y_score, sens_97_threshold, uids)
    misclassified_99 = find_misclassified_samples(y_true, y_score, sens_99_threshold, uids)
    
    # Print information about misclassified samples
    print(f"\nAt {sens_90_tpr:.3f} sensitivity (threshold={sens_90_threshold:.3f}):")
    print(f"  Specificity: {1-sens_90_fpr:.3f}")
    print(f"  Misclassified samples: {len(misclassified_90['misclassified_indices'])}")
    print(f"  False positives: {len(misclassified_90['false_positives'])}")
    print(f"  False negatives: {len(misclassified_90['false_negatives'])}")
    
    print(f"\nAt {sens_95_tpr:.3f} sensitivity (threshold={sens_95_threshold:.3f}):")
    print(f"  Specificity: {1-sens_95_fpr:.3f}")
    print(f"  Misclassified samples: {len(misclassified_95['misclassified_indices'])}")
    print(f"  False positives: {len(misclassified_95['false_positives'])}")
    print(f"  False negatives: {len(misclassified_95['false_negatives'])}")
    
    print(f"\nAt {sens_97_tpr:.3f} sensitivity (threshold={sens_97_threshold:.3f}):")
    print(f"  Specificity: {1-sens_97_fpr:.3f}")
    print(f"  Misclassified samples: {len(misclassified_97['misclassified_indices'])}")
    print(f"  False positives: {len(misclassified_97['false_positives'])}")
    print(f"  False negatives: {len(misclassified_97['false_negatives'])}")
    
    print(f"\nAt {sens_99_tpr:.3f} sensitivity (threshold={sens_99_threshold:.3f}):")
    print(f"  Specificity: {1-sens_99_fpr:.3f}")
    print(f"  Misclassified samples: {len(misclassified_99['misclassified_indices'])}")
    print(f"  False positives: {len(misclassified_99['false_positives'])}")
    print(f"  False negatives: {len(misclassified_99['false_negatives'])}")
    
    # Use optimal threshold for confusion matrix
    y_scores_bin = y_score >= thrs[opt_idx] # WARNING: Must be >= not > 
    conf_matrix = confusion_matrix(y_true, y_scores_bin) # [[TN, FP], [FN, TP]]

    # Plot main ROC curve
    axis.plot(fprs, tprs, color=color, label=rf"{name} (AUC = {auc_val:.2f} $\pm$ {std_auc:.2f})",
                lw=2, alpha=.8)
    axis.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    # Plot optimal point if requested (default is now False)
    if show_wp:
        axis.hlines(y=opt_tpr, xmin=0.0, xmax=opt_fpr, color='g', linestyle='--')
        axis.vlines(x=opt_fpr, ymin=0.0, ymax=opt_tpr, color='g', linestyle='--')
        axis.plot(opt_fpr, opt_tpr, color=color, marker='o')
    
    # Plot 90% sensitivity point and lines
    axis.hlines(y=sens_90_tpr, xmin=0.0, xmax=sens_90_fpr, color='r', linestyle='--')
    axis.vlines(x=sens_90_fpr, ymin=0.0, ymax=sens_90_tpr, color='r', linestyle='--')
    axis.plot(sens_90_fpr, sens_90_tpr, 'ro', 
             label=f'Sens={sens_90_tpr:.2f}, Spec={1-sens_90_fpr:.2f}')
    
    # Plot 95% sensitivity point and lines
    axis.hlines(y=sens_95_tpr, xmin=0.0, xmax=sens_95_fpr, color='orange', linestyle='--')
    axis.vlines(x=sens_95_fpr, ymin=0.0, ymax=sens_95_tpr, color='orange', linestyle='--')
    axis.plot(sens_95_fpr, sens_95_tpr, 'o', color='orange', 
             label=f'Sens={sens_95_tpr:.2f}, Spec={1-sens_95_fpr:.2f}')
    
    # Plot 97% sensitivity point and lines
    axis.hlines(y=sens_97_tpr, xmin=0.0, xmax=sens_97_fpr, color='purple', linestyle='--')
    axis.vlines(x=sens_97_fpr, ymin=0.0, ymax=sens_97_tpr, color='purple', linestyle='--')
    axis.plot(sens_97_fpr, sens_97_tpr, 'o', color='purple', 
             label=f'Sens={sens_97_tpr:.2f}, Spec={1-sens_97_fpr:.2f}')
    
    # Plot 99% sensitivity point and lines
    axis.hlines(y=sens_99_tpr, xmin=0.0, xmax=sens_99_fpr, color='m', linestyle='--')
    axis.vlines(x=sens_99_fpr, ymin=0.0, ymax=sens_99_tpr, color='m', linestyle='--')
    axis.plot(sens_99_fpr, sens_99_tpr, 'o', color='m', 
             label=f'Sens={sens_99_tpr:.2f}, Spec={1-sens_99_fpr:.2f}')
    
    # Plot diagonal line
    axis.plot([0, 1], [0, 1], linestyle='--', color='k')
    axis.set_xlim([0.0, 1.0])
    axis.set_ylim([0.0, 1.0])
    
    axis.legend(loc='lower right')
    axis.set_xlabel('1 - Specificity', fontdict=fontdict)
    axis.set_ylabel('Sensitivity', fontdict=fontdict)
    
    # Style the plot
    axis.grid(color='#dddddd')
    axis.set_axisbelow(True)
    axis.tick_params(colors='#dddddd', which='both')
    for xtick in axis.get_xticklabels():
        xtick.set_color('k')
    for ytick in axis.get_yticklabels():
        ytick.set_color('k')
    for child in axis.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('#dddddd')
    
    # Create a dictionary with sensitivity results
    sensitivity_results = {
        '90': {
            'sensitivity': sens_90_tpr,
            'specificity': 1-sens_90_fpr,
            'threshold': sens_90_threshold,
            'misclassified': misclassified_90
        },
        '95': {
            'sensitivity': sens_95_tpr,
            'specificity': 1-sens_95_fpr,
            'threshold': sens_95_threshold,
            'misclassified': misclassified_95
        },
        '97': {
            'sensitivity': sens_97_tpr,
            'specificity': 1-sens_97_fpr,
            'threshold': sens_97_threshold,
            'misclassified': misclassified_97
        },
        '99': {
            'sensitivity': sens_99_tpr,
            'specificity': 1-sens_99_fpr,
            'threshold': sens_99_threshold,
            'misclassified': misclassified_99
        }
    }
 
    return tprs, fprs, auc_val, thrs, opt_idx, conf_matrix, sensitivity_results

def cm2acc(cm):
    # [[TN, FP], [FN, TP]] 
    tn, fp, fn, tp = cm.ravel()
    return (tn+tp)/(tn+tp+fn+fp)

def safe_div(x,y):
    if y == 0:
        return float('nan') 
    return x / y

def cm2x(cm):
    tn, fp, fn, tp = cm.ravel()
    pp = tp + fp  # predicted positive 
    pn = fn + tn  # predicted negative
    p = tp + fn   # actual positive
    n = fp + tn   # actual negative  

    ppv = safe_div(tp,pp)  # positive predictive value 
    npv = safe_div(tn,pn)  # negative predictive value 
    tpr = safe_div(tp,p)   # true positive rate (sensitivity, recall)
    tnr = safe_div(tn,n)   # true negative rate (specificity)
    # Note: other values are 1-x eg. fdr=1-ppv, for=1-npv, ....
    return ppv, npv, tpr, tnr
