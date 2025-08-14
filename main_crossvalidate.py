"""
Multi-sequence MST Cross-Validation Script

Cross-validation evaluation script with fold-based statistics for multi-sequence 
breast MRI classification.

Original MST framework by Müller-Franzes et al.
Repository: https://github.com/mueller-franzes/MST
Paper: https://arxiv.org/abs/2411.15802
Licensed under MIT License

This script is our contribution, implementing cross-validation with fold-based 
standard deviation calculation (not bootstrap CI).
"""

import argparse
from pathlib import Path
import yaml
import json
import torch
from pytorch_lightning.trainer import Trainer
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import logging
from tqdm import tqdm
import torch.nn.functional as F
# from torchvision.utils import save_image  # Not used, commented out

from mst.data.datasets.dataset_3d_duke import DUKE_Dataset3D
from mst.data.datasets.dataset_3d_lidc import LIDC_Dataset3D
from mst.data.datasets.dataset_3d_mrnet import MRNet_Dataset3D
from mst.data.datasets.dataset_3d_dwi import DWI_Dataset3D
from mst.data.datamodules import DataModule
from mst.models.dino import DinoV2ClassifierSlice
from mst.utils.roc_curve import plot_roc_curve, cm2acc, cm2x
from mst.models.utils.functions import tensor2image, tensor_cam2image, minmax_norm

# use: python scripts/main_crossvalidate.py --model_dir ./runs/t1 --output_dir ./cross_val_results

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_dataset(config, name, split, **kwargs):    
    dataset_params = config.get('dataset_params', {})
    path_img = Path(dataset_params.get('path_img', ''))
    
    if name == 'DWI':
        return DWI_Dataset3D(
            path_img=path_img,
            path_csv=dataset_params.get('path_csv'),
            sequences=dataset_params.get('sequences', ["dwi0", "dwi2", "t2"]),
            fold=dataset_params.get('fold'),
            split=split,
            image_resize=dataset_params.get('image_resize'),
            image_crop=dataset_params.get('image_crop'),
            random_center=dataset_params.get('random_center'),
            flip=dataset_params.get('flip'),
            noise=dataset_params.get('noise'),
            random_rotate=dataset_params.get('random_rotate'),
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

def get_model(config, name, **kwargs):    
    """Get model class by name."""
    if name == 'DinoV2ClassifierSlice':
        return DinoV2ClassifierSlice
    else:
        raise ValueError(f"Unknown model: {name}. Only DinoV2ClassifierSlice is supported.")

def run_pred(model, batch, save_attn=False, use_softmax=True, use_tta=False):
    """Run prediction with optional TTA and attention maps."""
    source = batch['source']
    
    # Basic prediction
    with torch.no_grad():
        pred = model(source, save_attn=save_attn)
        
    if use_softmax:
        pred = torch.softmax(pred, dim=-1)
        
    weight = None
    weight_slice = None
        
    if save_attn and hasattr(model, 'get_attention_maps'):
        # Get attention maps if available
        weight = model.get_attention_maps()
        weight = weight.mean(dim=1)
        spatial_shape = weight.shape[-2:] if isinstance(model, ResNetSliceTrans) else torch.tensor(source.shape[3:])//14
        weight = weight.view(1, 1, source.shape[2], *spatial_shape)
        
        if hasattr(model, 'get_slice_attention'):
            weight_slice = model.get_slice_attention()
            weight_slice = weight_slice.mean(dim=1)
            weight_slice = weight_slice.view(1, 1, -1, 1, 1) * torch.ones_like(source, device=weight.device)
            
    if use_tta:
        # Implement test-time augmentation
        flip_dims = [(2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)]
        for flip_dim in flip_dims:
            with torch.no_grad():
                pred_i = model(torch.flip(source, flip_dim))
                if use_softmax:
                    pred_i = torch.softmax(pred_i, dim=-1)
                pred = pred + pred_i
                
                if save_attn and weight is not None:
                    weight_i = model.get_attention_maps().mean(dim=1)
                    weight = weight + torch.flip(weight_i, flip_dim)
                    
                    if weight_slice is not None:
                        weight_slice_i = model.get_slice_attention().mean(dim=1)
                        weight_slice = weight_slice + torch.flip(weight_slice_i, flip_dim)
        
        # Average predictions
        pred = pred / (len(flip_dims) + 1)
        if save_attn and weight is not None:
            weight = weight / (len(flip_dims) + 1)
            if weight_slice is not None:
                weight_slice = weight_slice / (len(flip_dims) + 1)
    
    # Interpolate attention maps if needed
    if save_attn and weight is not None:
        weight = F.interpolate(weight, size=source.shape[2:], mode='trilinear')
        
    return pred, weight, weight_slice

def evaluate_fold(model_dir, output_dir, eval_config):
    """Evaluate a single fold."""
    try:
        # Load configuration
        config_path = model_dir / 'used_config.yaml'
        config = load_config(config_path)
        
        # Setup logging
        logger = logging.getLogger(f"fold_{model_dir.name}")
        logger.setLevel(logging.INFO)
        
        # Override image path if needed
        # Note: Update this path to point to your actual data directory
        # original_dataset_params = config.get('dataset_params', {}).copy()
        # original_dataset_params['path_img'] = "/path/to/your/processed/images"
        # config['dataset_params'] = original_dataset_params
        
        sequences = config['dataset_params'].get('sequences', [])
        sequence_name = '_'.join(sequences) if sequences else 'unknown'
        fold = config['dataset_params'].get('fold', 'unknown')
        
        print(f"Evaluating model from {model_dir}")
        print(f"Sequences: {', '.join(sequences) if sequences else 'unknown'}")
        print(f"Fold: {fold}")
        print(f"Dataset: {config['dataset']}")
        print(f"Model: {config['model']}")
        
        # Setup test dataset and dataloader
        ds_test = get_dataset(config, config['dataset'], split='test')
        dm = DataModule(
            ds_train=None,
            ds_val=None,
            ds_test=ds_test,
            batch_size=1,
            num_workers=config.get('num_workers', 24),
            pin_memory=True
        )
        
        # Load model checkpoint
        with open(model_dir / 'best_checkpoint.json', 'r') as f:
            checkpoint_info = json.load(f)
        
        checkpoint_path = model_dir / checkpoint_info['best_model_epoch']
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Initialize model
        model_class = get_model(config, config['model'])
        model = model_class.load_from_checkpoint(
            str(checkpoint_path),
            **config.get('model_params', {})
        )
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        # Setup output directory
        fold_output_dir = output_dir / sequence_name / f"fold_{fold}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        # Collect results
        results = []
        
        # Evaluation loop
        for batch in tqdm(dm.test_dataloader(), desc=f"Evaluating fold {fold}"):
            source, target = batch['source'], batch['target']
            uid = batch['uid'][0] if isinstance(batch['uid'], list) else str(batch['uid'].item())
            
            # Run prediction
            pred, weight, weight_slice = run_pred(
                model, 
                batch,
                save_attn=eval_config.get('get_attention', False),
                use_tta=eval_config.get('use_tta', False)
            )
            
            # Process predictions
            pred = pred.cpu()
            pred_binary = torch.argmax(pred, dim=1)
            pred_prob = torch.softmax(pred, dim=-1)[:, 1]
            
            # Collect results - Fixed indentation to be inside the loop
            for b in range(target.shape[0]):
                # Get the input filename from the batch if available
                input_filename = batch.get('filename', ['unknown'])[b] if isinstance(batch.get('filename', ['unknown']), list) else 'unknown'
                
                results.append({
                    'UID': uid,
                    'GT': target[b].item(),
                    'NN': pred_binary[b].item(),
                    'NN_pred': pred_prob[b].item(),
                    'filename': input_filename
                })
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        df.to_csv(fold_output_dir / 'results.csv', index=False)
        
        # Calculate metrics
        y_true = np.asarray(df['GT'])
        y_pred = np.asarray(df['NN'])
        y_pred_prob = np.asarray(df['NN_pred'])
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        n = len(df)
        
        # Calculate accuracy from confusion matrix for consistency
        acc = cm2acc(cm)
        
        # ROC curve
        fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6,6))
        fontdict = {'fontsize': 10, 'fontweight': 'bold'}
        # Pass UIDs to plot_roc_curve
        uids = df['UID'].tolist()
        tprs, fprs, auc_val, thrs, opt_idx, cm_roc, sensitivity_results = plot_roc_curve(y_true, y_pred_prob, axis, fontdict=fontdict, uids=uids)
        fig.tight_layout()
        fig.savefig(fold_output_dir / 'roc.png', dpi=300)
        plt.close(fig)
        
        # Calculate metrics directly from confusion matrix
        _, _, sens, spec = cm2x(cm)
        df_cm = pd.DataFrame(data=cm, columns=['False', 'True'], index=['False', 'True'])
        fig, axis = plt.subplots(1, 1, figsize=(4,4))
        sns.heatmap(df_cm, ax=axis, cbar=False, fmt='d', annot=True)
        #axis.set_title(f'Confusion Matrix ACC={acc:.2f}', fontdict=fontdict)
        axis.set_xlabel('Prediction', fontdict=fontdict)
        axis.set_ylabel('Ground truth', fontdict=fontdict)
        fig.tight_layout()
        fig.savefig(fold_output_dir / 'confusion_matrix.png', dpi=300)
        plt.close(fig)
        
        # Save results to text file
        with open(fold_output_dir / 'results.txt', 'w') as f:
            f.write(f"=== Results for {sequence_name} - Fold {fold} ===\n\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"AUC: {auc_val:.4f}\n")
            f.write(f"Sensitivity: {sens:.4f}\n")
            f.write(f"Specificity: {spec:.4f}\n\n")
            f.write(f"Total samples: {n}\n")
            f.write(f"True positives: {tp}\n")
            f.write(f"True negatives: {tn}\n")
            f.write(f"False positives: {fp}\n")
            f.write(f"False negatives: {fn}\n\n")
            
            # Add sensitivity results
            f.write(f"=== Sensitivity Analysis ===\n\n")
            
            # 90% sensitivity point
            sens_90 = sensitivity_results['90']
            f.write(f"At {sens_90['sensitivity']:.4f} sensitivity (threshold={sens_90['threshold']:.4f}):\n")
            f.write(f"  Specificity: {sens_90['specificity']:.4f}\n")
            f.write(f"  Misclassified samples: {len(sens_90['misclassified']['misclassified_indices'])}\n")
            f.write(f"  False positives: {len(sens_90['misclassified']['false_positives'])}\n")
            f.write(f"  False negatives: {len(sens_90['misclassified']['false_negatives'])}\n")
            
            # List misclassified UIDs if available
            if 'misclassified_uids' in sens_90['misclassified']:
                f.write("\n  Misclassified UIDs:\n")
                for uid in sens_90['misclassified']['misclassified_uids']:
                    f.write(f"    - {uid}\n")
                
                f.write("\n  False positive UIDs:\n")
                for uid in sens_90['misclassified']['false_positive_uids']:
                    f.write(f"    - {uid}\n")
                
                f.write("\n  False negative UIDs:\n")
                for i, uid in enumerate(sens_90['misclassified']['false_negative_uids']):
                    # Find the filename for this UID
                    filename = 'unknown'
                    for result in results:
                        if result['UID'] == uid:
                            filename = result.get('filename', 'unknown')
                            break
                    f.write(f"    - {uid} (File: {filename})\n")
            
            f.write("\n")
            
            # 95% sensitivity point
            sens_95 = sensitivity_results['95']
            f.write(f"At {sens_95['sensitivity']:.4f} sensitivity (threshold={sens_95['threshold']:.4f}):\n")
            f.write(f"  Specificity: {sens_95['specificity']:.4f}\n")
            f.write(f"  Misclassified samples: {len(sens_95['misclassified']['misclassified_indices'])}\n")
            f.write(f"  False positives: {len(sens_95['misclassified']['false_positives'])}\n")
            f.write(f"  False negatives: {len(sens_95['misclassified']['false_negatives'])}\n")
            
            # List misclassified UIDs if available
            if 'misclassified_uids' in sens_95['misclassified']:
                f.write("\n  Misclassified UIDs:\n")
                for uid in sens_95['misclassified']['misclassified_uids']:
                    f.write(f"    - {uid}\n")
                
                f.write("\n  False positive UIDs:\n")
                for uid in sens_95['misclassified']['false_positive_uids']:
                    f.write(f"    - {uid}\n")
                
                f.write("\n  False negative UIDs:\n")
                for i, uid in enumerate(sens_95['misclassified']['false_negative_uids']):
                    # Find the filename for this UID
                    filename = 'unknown'
                    for result in results:
                        if result['UID'] == uid:
                            filename = result.get('filename', 'unknown')
                            break
                    f.write(f"    - {uid} (File: {filename})\n")
            
            f.write("\n")
            
            # 97.5% sensitivity point
            sens_975 = sensitivity_results['975']
            f.write(f"At {sens_975['sensitivity']:.4f} sensitivity (threshold={sens_975['threshold']:.4f}):\n")
            f.write(f"  Specificity: {sens_975['specificity']:.4f}\n")
            f.write(f"  Misclassified samples: {len(sens_975['misclassified']['misclassified_indices'])}\n")
            f.write(f"  False positives: {len(sens_975['misclassified']['false_positives'])}\n")
            f.write(f"  False negatives: {len(sens_975['misclassified']['false_negatives'])}\n")
            
            # List misclassified UIDs if available
            if 'misclassified_uids' in sens_975['misclassified']:
                f.write("\n  Misclassified UIDs:\n")
                for uid in sens_975['misclassified']['misclassified_uids']:
                    f.write(f"    - {uid}\n")
                
                f.write("\n  False positive UIDs:\n")
                for uid in sens_975['misclassified']['false_positive_uids']:
                    f.write(f"    - {uid}\n")
                
                f.write("\n  False negative UIDs:\n")
                for i, uid in enumerate(sens_975['misclassified']['false_negative_uids']):
                    # Find the filename for this UID
                    filename = 'unknown'
                    for result in results:
                        if result['UID'] == uid:
                            filename = result.get('filename', 'unknown')
                            break
                    f.write(f"    - {uid} (File: {filename})\n")
        
        # Log metrics
        metrics = {
            'accuracy': acc,
            'auc': auc_val,
            'sensitivity': sens,
            'specificity': spec,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'n_total': n,
            'n_positive': np.sum(y_true),
            'fold': fold,
            'sequence': sequence_name,
            'y_true': y_true.tolist(),
            'y_pred_prob': y_pred_prob.tolist(),
            'sensitivity_results': {
                '90': {
                    'sensitivity': float(sens_90['sensitivity']),
                    'specificity': float(sens_90['specificity']),
                    'threshold': float(sens_90['threshold']),
                    'misclassified_count': len(sens_90['misclassified']['misclassified_indices']),
                    'false_positives_count': len(sens_90['misclassified']['false_positives']),
                    'false_negatives_count': len(sens_90['misclassified']['false_negatives'])
                },
                '95': {
                    'sensitivity': float(sens_95['sensitivity']),
                    'specificity': float(sens_95['specificity']),
                    'threshold': float(sens_95['threshold']),
                    'misclassified_count': len(sens_95['misclassified']['misclassified_indices']),
                    'false_positives_count': len(sens_95['misclassified']['false_positives']),
                    'false_negatives_count': len(sens_95['misclassified']['false_negatives'])
                },
                '975': {
                    'sensitivity': float(sens_975['sensitivity']),
                    'specificity': float(sens_975['specificity']),
                    'threshold': float(sens_975['threshold']),
                    'misclassified_count': len(sens_975['misclassified']['misclassified_indices']),
                    'false_positives_count': len(sens_975['misclassified']['false_positives']),
                    'false_negatives_count': len(sens_975['misclassified']['false_negatives'])
                }
            }
        }
        
        # Save metrics
        with open(fold_output_dir / 'metrics.yaml', 'w') as f:
            yaml.dump(metrics, f)
        
        return metrics
        
    except Exception as e:
        print(f"Error in evaluate_fold:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        raise

def create_aggregated_results(sequence_results, output_dir):
    """Create aggregated ROC curves and confusion matrices for all sequences."""
    
    for sequence_name, results in sequence_results.items():
        sequence_output_dir = output_dir / sequence_name
        
        # Calculate mean metrics for printing
        mean_metrics = {
            'accuracy': np.mean([r['accuracy'] for r in results]),
            'auc': np.mean([r['auc'] for r in results]),
            'sensitivity': np.mean([r['sensitivity'] for r in results]),
            'specificity': np.mean([r['specificity'] for r in results])
        }
        std_metrics = {
            'accuracy': np.std([r['accuracy'] for r in results]),
            'auc': np.std([r['auc'] for r in results]),
            'sensitivity': np.std([r['sensitivity'] for r in results]),
            'specificity': np.std([r['specificity'] for r in results])
        }
        
        # For aggregated confusion matrix and threshold-based metrics
        all_y_true = []
        all_y_pred_prob = []
        
        # Collect all results for aggregated analysis
        for r in results:
            all_y_true.extend(r['y_true'])
            all_y_pred_prob.extend(r['y_pred_prob'])
        
        all_y_true = np.array(all_y_true)
        all_y_pred_prob = np.array(all_y_pred_prob)
        
        # Calculate aggregated ROC
        fpr, tpr, thresholds = roc_curve(all_y_true, all_y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Determine optimal threshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Generate aggregated binary predictions using optimal threshold
        all_y_pred = (all_y_pred_prob >= optimal_threshold).astype(int)
        
        # Calculate aggregated confusion matrix
        aggregated_cm = confusion_matrix(all_y_true, all_y_pred)
        tn, fp, fn, tp = aggregated_cm.ravel()
        
        # Calculate accuracy using optimal threshold        
        aggregated_acc = cm2acc(aggregated_cm)
        
        # Create ROC curve with cross-validation variability
        fig, axis = plt.subplots(figsize=(6, 6))
        fontdict = {'fontsize': 10, 'fontweight': 'bold'}
        
        # Create a common base of FPR points for interpolation
        mean_fpr = np.linspace(0, 1, 100)
        
        # Store interpolated TPRs for each fold
        tprs_per_fold = []
        aucs = []
        
        # Plot individual fold ROC curves with low alpha
        for i, r in enumerate(results):
            y_true = np.array(r['y_true'])
            y_pred_prob = np.array(r['y_pred_prob'])
            
            # Calculate ROC curve for this fold
            fpr_fold, tpr_fold, _ = roc_curve(y_true, y_pred_prob)
            
            # Interpolate TPR values at the common FPR points
            tpr_interp = np.interp(mean_fpr, fpr_fold, tpr_fold)
            tpr_interp[0] = 0.0
            tpr_interp[-1] = 1.0
            tprs_per_fold.append(tpr_interp)
            
            # Calculate AUC for this fold
            auc_fold = auc(fpr_fold, tpr_fold)
            aucs.append(auc_fold)
            
            # Plot individual fold ROC curve with low opacity
            axis.plot(fpr_fold, tpr_fold, lw=1, alpha=0.3, 
                     label=f'Fold {i+1} (AUC = {auc_fold:.2f})')
        
        # Calculate mean and std of TPRs across folds
        mean_tpr = np.mean(tprs_per_fold, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs_per_fold, axis=0)
        
        # Calculate confidence interval
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        # Calculate mean and std of AUCs
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        # Plot mean ROC curve
        axis.plot(mean_fpr, mean_tpr, color='b', lw=2, 
                 label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
        
        # Plot confidence interval
        axis.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', 
                         alpha=0.2, label=r'$\pm$ 1 std. dev.')
        
        # Find optimal threshold point (Youden's J statistic)
        opt_idx = np.argmax(mean_tpr - mean_fpr)
        opt_tpr = mean_tpr[opt_idx]
        opt_fpr = mean_fpr[opt_idx]
        
        # Removed optimal point plotting as requested
        
        # Find sensitivity points at 0.9, 0.95, and 0.975
        sens_90_idx = np.argmin(np.abs(mean_tpr - 0.90))
        sens_90_tpr = mean_tpr[sens_90_idx]
        sens_90_fpr = mean_fpr[sens_90_idx]
        
        sens_95_idx = np.argmin(np.abs(mean_tpr - 0.95))
        sens_95_tpr = mean_tpr[sens_95_idx]
        sens_95_fpr = mean_fpr[sens_95_idx]
        
        sens_975_idx = np.argmin(np.abs(mean_tpr - 0.975))
        sens_975_tpr = mean_tpr[sens_975_idx]
        sens_975_fpr = mean_fpr[sens_975_idx]
        
        # Plot 90% sensitivity point with red lines
        axis.hlines(y=sens_90_tpr, xmin=0.0, xmax=sens_90_fpr, color='r', linestyle='--')
        axis.vlines(x=sens_90_fpr, ymin=0.0, ymax=sens_90_tpr, color='r', linestyle='--')
        axis.plot(sens_90_fpr, sens_90_tpr, 'ro', 
                 label=f'Sens={sens_90_tpr:.2f}, Spec={1-sens_90_fpr:.2f}')
        print(f"At {sens_90_tpr:.2f} sensitivity, specificity is {1-sens_90_fpr:.2f}")
        
        # Plot 95% sensitivity point with orange lines
        axis.hlines(y=sens_95_tpr, xmin=0.0, xmax=sens_95_fpr, color='orange', linestyle='--')
        axis.vlines(x=sens_95_fpr, ymin=0.0, ymax=sens_95_tpr, color='orange', linestyle='--')
        axis.plot(sens_95_fpr, sens_95_tpr, 'o', color='orange', 
                 label=f'Sens={sens_95_tpr:.2f}, Spec={1-sens_95_fpr:.2f}')
        print(f"At {sens_95_tpr:.2f} sensitivity, specificity is {1-sens_95_fpr:.2f}")
        
        # Plot 97.5% sensitivity point with purple lines
        axis.hlines(y=sens_975_tpr, xmin=0.0, xmax=sens_975_fpr, color='purple', linestyle='--')
        axis.vlines(x=sens_975_fpr, ymin=0.0, ymax=sens_975_tpr, color='purple', linestyle='--')
        axis.plot(sens_975_fpr, sens_975_tpr, 'o', color='purple', 
                 label=f'Sens={sens_975_tpr:.2f}, Spec={1-sens_975_fpr:.2f}')
        print(f"At {sens_975_tpr:.2f} sensitivity, specificity is {1-sens_975_fpr:.2f}")
        
        # Update roc_auc with the mean AUC
        roc_auc = mean_auc
        
        # Plot diagonal line
        axis.plot([0, 1], [0, 1], linestyle='--', color='k')
        
        # Style the plot
        axis.set_xlim([0.0, 1.0])
        axis.set_ylim([0.0, 1.0])
        axis.set_xlabel('1 - Specificity', fontdict=fontdict)
        axis.set_ylabel('Sensitivity', fontdict=fontdict)
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
        
        fig.tight_layout()
        fig.savefig(sequence_output_dir / 'aggregated_roc.png', dpi=300)
        plt.close(fig)
        
        # Create aggregated confusion matrix (similar to your per-fold code)
        acc, _, sens, spec = cm2x(aggregated_cm)
        
        df_cm = pd.DataFrame(data=aggregated_cm, columns=['False', 'True'], index=['False', 'True'])
        fig, axis = plt.subplots(figsize=(4, 4))
        sns.heatmap(df_cm, ax=axis, cbar=False, fmt='d', annot=True)
        axis.set_xlabel('Prediction', fontdict=fontdict)
        axis.set_ylabel('Ground truth', fontdict=fontdict)
        fig.tight_layout()
        fig.savefig(sequence_output_dir / 'aggregated_confusion_matrix.png', dpi=300)
        plt.close(fig)
        
        # Save aggregated results to text file
        with open(sequence_output_dir / 'aggregated_results.txt', 'w') as f:
            f.write(f"=== Aggregated Results for {sequence_name} ===\n\n")
            f.write(f"Number of folds: {len(results)}\n")
            f.write(f"Total samples: {sum(r.get('n_total', 0) for r in results)}\n")
            f.write(f"Total positive cases: {sum(r.get('n_positive', 0) for r in results)}\n\n")
            
            f.write(f"Cross-validation Metrics (Mean ± Std):\n")
            f.write(f"Accuracy: {mean_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}\n")
            f.write(f"AUC: {mean_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}\n")
            f.write(f"Sensitivity: {mean_metrics['sensitivity']:.4f} ± {std_metrics['sensitivity']:.4f}\n")
            f.write(f"Specificity: {mean_metrics['specificity']:.4f} ± {std_metrics['specificity']:.4f}\n\n")
            
            f.write(f"Aggregated Performance Metrics:\n")
            f.write(f"Aggregated Accuracy: {aggregated_acc:.4f}\n")
            f.write(f"Aggregated AUC: {roc_auc:.4f}\n")
            f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n\n")
            
            f.write(f"Aggregated Confusion Matrix:\n")
            f.write(f"True Positives: {tp}\n")
            f.write(f"True Negatives: {tn}\n")
            f.write(f"False Positives: {fp}\n")
            f.write(f"False Negatives: {fn}\n\n")
            
            # Add sensitivity analysis results
            f.write(f"=== Sensitivity Analysis ===\n\n")
            
            # Calculate mean and std of sensitivity metrics across folds
            sens_90_sensitivities = [r.get('sensitivity_results', {}).get('90', {}).get('sensitivity', 0) for r in results if 'sensitivity_results' in r]
            sens_90_specificities = [r.get('sensitivity_results', {}).get('90', {}).get('specificity', 0) for r in results if 'sensitivity_results' in r]
            
            sens_95_sensitivities = [r.get('sensitivity_results', {}).get('95', {}).get('sensitivity', 0) for r in results if 'sensitivity_results' in r]
            sens_95_specificities = [r.get('sensitivity_results', {}).get('95', {}).get('specificity', 0) for r in results if 'sensitivity_results' in r]
            
            sens_975_sensitivities = [r.get('sensitivity_results', {}).get('975', {}).get('sensitivity', 0) for r in results if 'sensitivity_results' in r]
            sens_975_specificities = [r.get('sensitivity_results', {}).get('975', {}).get('specificity', 0) for r in results if 'sensitivity_results' in r]
            
            # 90% sensitivity point
            if sens_90_sensitivities:
                mean_sens_90 = np.mean(sens_90_sensitivities)
                std_sens_90 = np.std(sens_90_sensitivities)
                mean_spec_90 = np.mean(sens_90_specificities)
                std_spec_90 = np.std(sens_90_specificities)
                
                f.write(f"At ~90% sensitivity:\n")
                f.write(f"  Mean Sensitivity: {mean_sens_90:.4f} ± {std_sens_90:.4f}\n")
                f.write(f"  Mean Specificity: {mean_spec_90:.4f} ± {std_spec_90:.4f}\n")
                f.write(f"  Aggregated Sensitivity: {sens_90_tpr:.4f}\n")
                f.write(f"  Aggregated Specificity: {1-sens_90_fpr:.4f}\n\n")
            
            # 95% sensitivity point
            if sens_95_sensitivities:
                mean_sens_95 = np.mean(sens_95_sensitivities)
                std_sens_95 = np.std(sens_95_sensitivities)
                mean_spec_95 = np.mean(sens_95_specificities)
                std_spec_95 = np.std(sens_95_specificities)
                
                f.write(f"At ~95% sensitivity:\n")
                f.write(f"  Mean Sensitivity: {mean_sens_95:.4f} ± {std_sens_95:.4f}\n")
                f.write(f"  Mean Specificity: {mean_spec_95:.4f} ± {std_spec_95:.4f}\n")
                f.write(f"  Aggregated Sensitivity: {sens_95_tpr:.4f}\n")
                f.write(f"  Aggregated Specificity: {1-sens_95_fpr:.4f}\n\n")
            
            # 97.5% sensitivity point
            if sens_975_sensitivities:
                mean_sens_975 = np.mean(sens_975_sensitivities)
                std_sens_975 = np.std(sens_975_sensitivities)
                mean_spec_975 = np.mean(sens_975_specificities)
                std_spec_975 = np.std(sens_975_specificities)
                
                f.write(f"At ~97.5% sensitivity:\n")
                f.write(f"  Mean Sensitivity: {mean_sens_975:.4f} ± {std_sens_975:.4f}\n")
                f.write(f"  Mean Specificity: {mean_spec_975:.4f} ± {std_spec_975:.4f}\n")
                f.write(f"  Aggregated Sensitivity: {sens_975_tpr:.4f}\n")
                f.write(f"  Aggregated Specificity: {1-sens_975_fpr:.4f}\n")
        
        # Save aggregated metrics to YAML
        aggregated_metrics = {
            'n_folds': len(results),
            'total_samples': sum(r.get('n_total', 0) for r in results),
            'total_positive': sum(r.get('n_positive', 0) for r in results),
            'mean_metrics': mean_metrics,
            'std_metrics': std_metrics,
            'aggregated_acc': aggregated_acc,
            'aggregated_auc': roc_auc,
            'aggregated_cm': aggregated_cm.tolist(),
            'optimal_threshold': float(optimal_threshold),
            'sensitivity_analysis': {
                '90': {
                    'aggregated_sensitivity': float(sens_90_tpr),
                    'aggregated_specificity': float(1-sens_90_fpr),
                    'mean_sensitivity': float(np.mean(sens_90_sensitivities)) if sens_90_sensitivities else 0,
                    'std_sensitivity': float(np.std(sens_90_sensitivities)) if sens_90_sensitivities else 0,
                    'mean_specificity': float(np.mean(sens_90_specificities)) if sens_90_specificities else 0,
                    'std_specificity': float(np.std(sens_90_specificities)) if sens_90_specificities else 0
                },
                '95': {
                    'aggregated_sensitivity': float(sens_95_tpr),
                    'aggregated_specificity': float(1-sens_95_fpr),
                    'mean_sensitivity': float(np.mean(sens_95_sensitivities)) if sens_95_sensitivities else 0,
                    'std_sensitivity': float(np.std(sens_95_sensitivities)) if sens_95_sensitivities else 0,
                    'mean_specificity': float(np.mean(sens_95_specificities)) if sens_95_specificities else 0,
                    'std_specificity': float(np.std(sens_95_specificities)) if sens_95_specificities else 0
                },
                '975': {
                    'aggregated_sensitivity': float(sens_975_tpr),
                    'aggregated_specificity': float(1-sens_975_fpr),
                    'mean_sensitivity': float(np.mean(sens_975_sensitivities)) if sens_975_sensitivities else 0,
                    'std_sensitivity': float(np.std(sens_975_sensitivities)) if sens_975_sensitivities else 0,
                    'mean_specificity': float(np.mean(sens_975_specificities)) if sens_975_specificities else 0,
                    'std_specificity': float(np.std(sens_975_specificities)) if sens_975_specificities else 0
                }
            }
        }
        
        with open(sequence_output_dir / 'aggregated_metrics.yaml', 'w') as f:
            yaml.dump(aggregated_metrics, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing model run folders')
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--get_attention', action='store_true')
    parser.add_argument('--get_segmentation', action='store_true')
    parser.add_argument('--use_tta', action='store_true')
    args = parser.parse_args()

    print("\n=== Starting Cross-validation Script ===")
    print(f"Current working directory: {Path.cwd()}")
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    
    print(f"\nChecking directories:")
    print(f"Model directory: {model_dir} (exists: {model_dir.exists()})")
    print(f"Output directory: {output_dir} (exists: {output_dir.exists()})")
    
    if not model_dir.exists():
        print(f"ERROR: Model directory does not exist: {model_dir}")
        return
        
    print(f"\nLooking for model runs in: {model_dir}")
    print("Requirements for valid run:")
    print("  - Must be a directory")
    print("  - Must contain 'best_checkpoint.json'")
    print("  - Must contain 'used_config.yaml'")
    
    # Find all contents of the directory
    all_contents = list(model_dir.glob('*'))
    print(f"\nAll contents in model directory ({len(all_contents)} items):")
    for item in all_contents:
        print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
    
    # Find valid run directories
    run_dirs = [d for d in model_dir.glob('*') if d.is_dir() and 
               (d / 'best_checkpoint.json').exists() and 
               (d / 'used_config.yaml').exists()]
    
    print(f"\nFound {len(run_dirs)} valid model runs:")
    for d in run_dirs:
        print(f"  - {d.name}")
        print(f"    ├── best_checkpoint.json: {(d / 'best_checkpoint.json').exists()}")
        print(f"    └── used_config.yaml: {(d / 'used_config.yaml').exists()}")

    if not run_dirs:
        print("\nERROR: No valid model runs found!")
        return

    # Dictionary to store results by sequence
    sequence_results = defaultdict(list)
    
    print("\nProcessing each run:")
    for run_dir in run_dirs:
        print(f"\n=== Processing run: {run_dir.name} ===")
        try:
            metrics = evaluate_fold(
                model_dir=run_dir,
                output_dir=output_dir,
                eval_config={
                    'get_attention': args.get_attention,
                    'get_segmentation': args.get_segmentation,
                    'use_tta': args.use_tta
                }
            )
            sequence_results[metrics['sequence']].append(metrics)
            print(f"Successfully processed {run_dir.name}")
        except Exception as e:
            print(f"Error processing {run_dir.name}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            continue

    if not sequence_results:
        print("\nERROR: No results were generated!")
        return

    print("\nCreating aggregated results:")
    create_aggregated_results(sequence_results, output_dir)

    print("\nProcessing results for each sequence:")
    for sequence_name, results in sequence_results.items():
        sequence_output_dir = output_dir / sequence_name
        
        # Calculate and print summary statistics
        mean_acc = np.mean([r['accuracy'] for r in results])
        std_acc = np.std([r['accuracy'] for r in results])
        mean_auc = np.mean([r['auc'] for r in results])
        std_auc = np.std([r['auc'] for r in results])
        mean_sens = np.mean([r['sensitivity'] for r in results])
        std_sens = np.std([r['sensitivity'] for r in results])
        mean_spec = np.mean([r['specificity'] for r in results])
        std_spec = np.std([r['specificity'] for r in results])
        
        print(f"\nResults Summary for Sequence {sequence_name}:")
        print(f"Number of folds evaluated: {len(results)}")
        print(f"Total samples: {sum(r['n_total'] for r in results)}")
        print(f"Total positive cases: {sum(r['n_positive'] for r in results)}")
        print(f"Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"AUC: {mean_auc:.3f} ± {std_auc:.3f}")
        print(f"Sensitivity: {mean_sens:.3f} ± {std_sens:.3f}")
        print(f"Specificity: {mean_spec:.3f} ± {std_spec:.3f}")

    print("\nCross-validation completed successfully!")

if __name__ == "__main__":
    main()
