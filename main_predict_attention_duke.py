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
import matplotlib.pyplot as plt
from PIL import Image
import io
from torchvision.utils import save_image

from mst.data.datasets.dataset_3d_duke import DUKE_Dataset3D
from mst.data.datasets.dataset_3d_lidc import LIDC_Dataset3D
from mst.data.datasets.dataset_3d_mrnet import MRNet_Dataset3D
from mst.data.datasets.dataset_3d_dwi import DWI_Dataset3D
from mst.data.datamodules import DataModule
from mst.models.dino import DinoV2ClassifierSlice
from mst.utils.roc_curve import plot_roc_curve, cm2acc, cm2x
from mst.models.utils.functions import minmax_norm, tensor2image, tensor_cam2image

# Function to save tensor images using matplotlib
def save_image_plt(tensor_img, filepath, normalize=False):
    """Save a tensor image to a file using matplotlib."""
    # Convert tensor to numpy array
    if isinstance(tensor_img, torch.Tensor):
        img_array = tensor_img.detach().cpu().numpy()
        # If tensor is [N, C, H, W], convert to [H, W, C] for matplotlib
        if img_array.ndim == 4:
            img_array = np.transpose(img_array[0], (1, 2, 0))
        # Normalize if needed
        if normalize:
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
    else:
        img_array = tensor_img
        if normalize and img_array.max() > 1.0:
            img_array = img_array / 255.0
    
    plt.figure(figsize=(12, 12))
    plt.imshow(img_array)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

# use: python scripts/main_predict_attention_duke.py --model_dir ./runs/t1 --output_dir ./attention_maps_duke

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
    elif name == 'DUKE':
        return DUKE_Dataset3D(
            path_img=path_img,
            path_csv=dataset_params.get('path_csv'),
            sequences=dataset_params.get('sequences', ["t1"]),
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

def generate_attention_maps(model_dir, output_dir, eval_config):
    """Generate attention maps for true positive samples."""
    try:
        # Load configuration
        config_path = model_dir / 'used_config.yaml'
        config = load_config(config_path)
        
        # Setup logging
        logger = logging.getLogger(f"attention_maps_{model_dir.name}")
        logger.setLevel(logging.INFO)
        
        # Check if debug flag is set
        debug_file_loading = config.get('debug_file_loading', False)
        
        # Use the config as is - it will be modified by the calling script
        # We're keeping the original dataset parameters and just disabling augmentations
        config['dataset_params']['flip'] = False
        config['dataset_params']['noise'] = False
        config['dataset_params']['random_rotate'] = False
        config['dataset_params']['random_center'] = False
        
        if debug_file_loading:
            print("\n=== Debug: Dataset Configuration ===")
            print(f"Dataset: {config['dataset']}")
            print(f"Image path: {config['dataset_params'].get('path_img')}")
            print(f"CSV file: {config['dataset_params'].get('path_csv')}")
            print(f"Sequences: {config['dataset_params'].get('sequences')}")
        
        sequences = config['dataset_params'].get('sequences', [])
        sequence_name = '_'.join(sequences) if sequences else 'unknown'
        fold = config['dataset_params'].get('fold', 'unknown')
        
        print(f"Generating attention maps for model from {model_dir}")
        print(f"Sequences: {', '.join(sequences) if sequences else 'unknown'}")
        print(f"Fold: {fold}")
        print(f"Dataset: {config['dataset']}")
        print(f"Model: {config['model']}")
        
        # For Duke dataset, we use all splits (train, val, test) as test set
        # since training was performed on a different private dataset
        ds_train = get_dataset(config, config['dataset'], split='train')
        ds_val = get_dataset(config, config['dataset'], split='val')
        ds_test = get_dataset(config, config['dataset'], split='test')
        
        # Create a combined dataset for evaluation
        all_samples = []
        all_samples.extend([{'data': ds_train[i], 'split': 'train'} for i in range(len(ds_train))])
        all_samples.extend([{'data': ds_val[i], 'split': 'val'} for i in range(len(ds_val))])
        all_samples.extend([{'data': ds_test[i], 'split': 'test'} for i in range(len(ds_test))])
        
        print(f"Total samples: {len(all_samples)}")
        print(f"  - Train samples: {len(ds_train)}")
        print(f"  - Validation samples: {len(ds_val)}")
        print(f"  - Test samples: {len(ds_test)}")
        
        # Create a combined dataloader
        combined_dataloader = torch.utils.data.DataLoader(
            [sample['data'] for sample in all_samples],
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
        attention_output_dir = output_dir / sequence_name / f"fold_{fold}" / "attention_maps"
        attention_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect results for all samples
        all_results = []
        
        # Counter for true positive samples
        tp_counter = 0
        max_tp_samples = eval_config.get('max_tp_samples', 1000)  # Limit the number of TP samples to process
        
        # Evaluation loop for all samples
        print(f"Generating attention maps for true positive samples")
        for i, batch in enumerate(tqdm(combined_dataloader, desc=f"Processing samples")):
            source, target = batch['source'], batch['target']
            uid = batch['uid'][0] if isinstance(batch['uid'], list) else str(batch['uid'].item())
            split = all_samples[i]['split']  # Get the split information
            
            # Run prediction
            pred, weight, weight_slice = run_pred(
                model, 
                batch,
                save_attn=True,
                use_tta=eval_config.get('use_tta', False)
            )
            
            # Process predictions
            pred = pred.cpu()
            pred_binary = torch.argmax(pred, dim=1)
            pred_prob = torch.softmax(pred, dim=-1)[:, 1]
            
            # Check if this is a true positive sample
            is_true_positive = (target.item() == 1 and pred_binary.item() == 1)
            
            # Collect results
            result = {
                'UID': uid,
                'Split': split,
                'GT': target.item(),
                'NN': pred_binary.item(),
                'NN_pred': pred_prob.item(),
                'is_true_positive': is_true_positive
            }
            all_results.append(result)
            
            # Generate attention maps for true positive samples
            if is_true_positive:
                tp_counter += 1
                
                # Create a separate folder for each sample
                sample_dir = attention_output_dir / f'sample_{uid}'
                sample_dir.mkdir(parents=True, exist_ok=True)
                
                # Process attention maps
                weight = weight.detach().cpu()
                weight_slice = weight_slice.detach().cpu()
                
                # Normalize weight_slice
                weight_slice = weight_slice / weight_slice.sum()
                
                # Clip weight to focus on high attention regions
                weight = weight.clip(*np.quantile(weight, [0.995, 0.999]))
                
                # Get the source tensor and extract individual slices
                source_np = source.detach().cpu().numpy()
                
                # Save the overall input image
                save_image(tensor2image(source), sample_dir / f'input_{uid}.png', normalize=True)
                
                # Save the overall spatial attention overlay
                save_image(tensor_cam2image(minmax_norm(source), minmax_norm(weight), alpha=0.5), 
                          sample_dir / f"overlay_{uid}.png", normalize=False)
                
                # Save the overall slice attention overlay
                save_image(tensor_cam2image(minmax_norm(source), minmax_norm(weight_slice), alpha=0.5), 
                          sample_dir / f"overlay_{uid}_slice.png", normalize=False)
                
                # Save individual slice images
                num_slices = source.shape[2]  # Get number of slices in the volume
                
                # Create subdirectory for axial slices
                axial_dir = sample_dir / 'axial'
                axial_dir.mkdir(exist_ok=True)
                
                # Extract slice attention values for visualization
                slice_attention_values = weight_slice.numpy()[0, 0, :, 0, 0]
                
                # Normalize slice attention values for better visualization
                slice_attention_normalized = (slice_attention_values - slice_attention_values.min()) / (slice_attention_values.max() - slice_attention_values.min() + 1e-8)
                
                # Create a bar chart of slice attention values
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(slice_attention_normalized)), slice_attention_normalized)
                plt.title(f'Slice Attention Distribution - Sample {uid}')
                plt.xlabel('Slice Index')
                plt.ylabel('Normalized Attention')
                plt.tight_layout()
                plt.savefig(sample_dir / f'slice_attention_distribution_{uid}.png', dpi=300)
                plt.close()
                
                # Prepare lists to store slice images for the overview
                slice_images = []
                overlay_images = []
                slice_attention_images = []
                
                # Process axial slices
                for slice_idx in range(num_slices):
                    # Extract the slice
                    slice_data = source_np[0, :, slice_idx, :, :]
                    slice_tensor = torch.from_numpy(slice_data).unsqueeze(0)
                    
                    # Extract corresponding spatial attention weight for this slice
                    weight_slice_data = weight.numpy()[0, 0, slice_idx, :, :]
                    weight_slice_tensor = torch.from_numpy(weight_slice_data).unsqueeze(0).unsqueeze(0)
                    
                    # Get the slice attention value for this slice
                    current_slice_attention = slice_attention_normalized[slice_idx]
                    
                    # Create the slice image
                    slice_img = tensor2image(slice_tensor)
                    
                    # Create the spatial attention overlay for this slice
                    slice_overlay = tensor_cam2image(
                        minmax_norm(slice_tensor), 
                        F.interpolate(weight_slice_tensor, size=slice_tensor.shape[2:], mode='bilinear'), 
                        alpha=0.5
                    )
                    
                    # Create a visualization of slice attention in a different way
                    # Instead of trying to add a border, create a new image with a colored bar
                    # indicating the attention level
                    
                    # First, create a copy of the slice image
                    if isinstance(slice_img, torch.Tensor):
                        slice_img_np = slice_img.detach().cpu().numpy()
                        if slice_img_np.ndim == 4:
                            slice_img_np = np.transpose(slice_img_np[0], (1, 2, 0))
                    else:
                        slice_img_np = slice_img
                    
                    # Normalize if needed
                    if slice_img_np.max() > 1.0:
                        slice_img_np = slice_img_np / 255.0
                    
                    # Get image dimensions
                    h, w = slice_img_np.shape[:2]
                    
                    # Create a new image with extra space for the attention indicator
                    indicator_height = max(10, int(h * 0.05))  # 5% of image height or at least 10 pixels
                    
                    # Create a new image with the indicator bar at the top
                    if len(slice_img_np.shape) == 3:  # Color image
                        slice_with_attention = np.zeros((h + indicator_height, w, 3))
                        slice_with_attention[indicator_height:, :, :] = slice_img_np
                        
                        # Add the attention indicator bar at the top
                        # Red color with intensity based on attention
                        for i in range(w):
                            # Calculate color based on position and attention
                            # Only color the portion of the bar up to the attention level
                            if i < int(w * current_slice_attention):
                                slice_with_attention[:indicator_height, i, 0] = 1.0  # Red channel
                    else:  # Grayscale image
                        # Convert to RGB
                        rgb_img = np.stack([slice_img_np] * 3, axis=-1)
                        slice_with_attention = np.zeros((h + indicator_height, w, 3))
                        slice_with_attention[indicator_height:, :, :] = rgb_img
                        
                        # Add the attention indicator bar at the top
                        for i in range(w):
                            if i < int(w * current_slice_attention):
                                slice_with_attention[:indicator_height, i, 0] = 1.0  # Red channel
                    
                    # Save individual images
                    save_image(slice_img, axial_dir / f'slice_{slice_idx:03d}.png', normalize=True)
                    save_image(slice_overlay, axial_dir / f'overlay_{slice_idx:03d}.png', normalize=False)
                    
                    # For slice_with_attention, we need to convert from numpy to tensor
                    slice_with_attention_tensor = torch.from_numpy(slice_with_attention.transpose(2, 0, 1)).float()
                    save_image(slice_with_attention_tensor, axial_dir / f'slice_attention_{slice_idx:03d}.png', normalize=False)
                    
                    # Store numpy arrays for the overview images
                    if isinstance(slice_img, torch.Tensor):
                        slice_img_np = slice_img.detach().cpu().numpy()
                        if slice_img_np.ndim == 4:
                            slice_img_np = np.transpose(slice_img_np[0], (1, 2, 0))
                    else:
                        slice_img_np = slice_img
                    
                    if isinstance(slice_overlay, torch.Tensor):
                        overlay_img_np = slice_overlay.detach().cpu().numpy()
                        if overlay_img_np.ndim == 4:
                            overlay_img_np = np.transpose(overlay_img_np[0], (1, 2, 0))
                    else:
                        overlay_img_np = slice_overlay
                    
                    slice_images.append(slice_img_np)
                    overlay_images.append(overlay_img_np)
                    slice_attention_images.append(slice_with_attention)
                
                # Create overview images by concatenating slices
                # Determine grid size (try to make it roughly square)
                grid_size = int(np.ceil(np.sqrt(num_slices)))
                rows = grid_size
                cols = grid_size
                
                # Create empty canvases for the overview images
                if len(slice_images) > 0:
                    img_height, img_width = slice_images[0].shape[:2]
                    slice_overview = np.zeros((rows * img_height, cols * img_width, 3))
                    overlay_overview = np.zeros((rows * img_height, cols * img_width, 3))
                    
                    # Place each slice in the grid
                    for i, (slice_img, overlay_img) in enumerate(zip(slice_images, overlay_images)):
                        if i >= rows * cols:
                            break
                            
                        row = i // cols
                        col = i % cols
                        
                        # Normalize images if needed
                        if slice_img.max() > 1.0:
                            slice_img = slice_img / 255.0
                        if overlay_img.max() > 1.0:
                            overlay_img = overlay_img / 255.0
                            
                        # Place in the overview
                        slice_overview[row*img_height:(row+1)*img_height, 
                                      col*img_width:(col+1)*img_width] = slice_img
                        overlay_overview[row*img_height:(row+1)*img_height, 
                                        col*img_width:(col+1)*img_width] = overlay_img
                    
                    # Save the overview images - convert numpy arrays to tensors first
                    slice_overview_tensor = torch.from_numpy(slice_overview.transpose(2, 0, 1)).float()
                    overlay_overview_tensor = torch.from_numpy(overlay_overview.transpose(2, 0, 1)).float()
                    
                    save_image(slice_overview_tensor, sample_dir / f'overview_slices_{uid}.png', normalize=False)
                    save_image(overlay_overview_tensor, sample_dir / f'overview_overlays_{uid}.png', normalize=False)
                
                print(f"Generated attention maps for TP sample {uid} ({tp_counter}/{max_tp_samples})")
                
                # Stop after processing max_tp_samples
                if tp_counter >= max_tp_samples:
                    print(f"Reached maximum number of TP samples ({max_tp_samples})")
                    break
        
        # Convert all results to DataFrame
        all_df = pd.DataFrame(all_results)
        all_df.to_csv(attention_output_dir.parent / 'attention_results.csv', index=False)
        
        # Count true positives
        tp_count = all_df[all_df['is_true_positive']].shape[0]
        print(f"Total true positive samples: {tp_count}")
        print(f"Generated attention maps for {min(tp_counter, max_tp_samples)} true positive samples")
        
        return {
            'sequence': sequence_name,
            'fold': fold,
            'tp_count': tp_count,
            'processed_count': tp_counter
        }
        
    except Exception as e:
        print(f"Error in generate_attention_maps:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing model run folders')
    parser.add_argument('--output_dir', type=str, default='./attention_maps_duke')
    parser.add_argument('--max_tp_samples', type=int, default=1000, help='Maximum number of true positive samples to process')
    parser.add_argument('--use_tta', action='store_true', help='Use test-time augmentation')
    args = parser.parse_args()

    print("\n=== Starting Duke Attention Map Generation Script ===")
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
    
    # Check if the model files are directly in the specified directory
    if (model_dir / 'best_checkpoint.json').exists() and (model_dir / 'used_config.yaml').exists():
        print(f"\nFound model files directly in {model_dir}")
        run_dirs = [model_dir]  # Use the directory itself
    else:
        # Find valid run directories - for Duke dataset, we only want fold 0
        run_dirs = [d for d in model_dir.glob('*') if d.is_dir() and 
                   (d / 'best_checkpoint.json').exists() and 
                   (d / 'used_config.yaml').exists()]
        
        # Filter for fold 0 only
        fold0_dirs = []
        for d in run_dirs:
            config_path = d / 'used_config.yaml'
            config = load_config(config_path)
            fold = config.get('dataset_params', {}).get('fold', None)
            if fold == 0:
                fold0_dirs.append(d)
                print(f"  - Found fold 0 run: {d.name}")
        
        # Use only fold 0 runs
        run_dirs = fold0_dirs
    
    print(f"\nFound {len(run_dirs)} valid model runs:")
    for d in run_dirs:
        print(f"  - {d.name}")
        print(f"    ├── best_checkpoint.json: {(d / 'best_checkpoint.json').exists()}")
        print(f"    └── used_config.yaml: {(d / 'used_config.yaml').exists()}")

    if not run_dirs:
        print("\nERROR: No valid model runs found!")
        return

    # Process each run
    results = []
    print("\nProcessing each run:")
    for run_dir in run_dirs:
        print(f"\n=== Processing run: {run_dir.name} ===")
        try:
            result = generate_attention_maps(
                model_dir=run_dir,
                output_dir=output_dir,
                eval_config={
                    'max_tp_samples': args.max_tp_samples,
                    'use_tta': args.use_tta
                }
            )
            results.append(result)
            print(f"Successfully processed {run_dir.name}")
        except Exception as e:
            print(f"Error processing {run_dir.name}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            continue

    if not results:
        print("\nERROR: No results were generated!")
        return

    # Print summary
    print("\nAttention Map Generation Summary:")
    for result in results:
        print(f"Sequence: {result['sequence']}, Fold: {result['fold']}")
        print(f"  - Total true positive samples: {result['tp_count']}")
        print(f"  - Processed samples: {result['processed_count']}")

    print("\nDuke attention map generation completed successfully!")

if __name__ == "__main__":
    main()
