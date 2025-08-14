from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import os 
import math 
import torch 
import torchio as tio
import numpy as np 
import torch.nn.functional as F
import pandas as pd 
import yaml
from torchvision.utils import save_image

from mst.data.datasets.dataset_3d_duke import DUKE_Dataset3D
from mst.data.datasets.dataset_3d_lidc import LIDC_Dataset3D
from mst.data.datasets.dataset_3d_mrnet import MRNet_Dataset3D
from mst.data.datasets.dataset_3d_dwi import DWI_Dataset3D, BilateralDWI_Dataset3D
from mst.data.datamodules import DataModule
from mst.models.dino import DinoV2ClassifierSlice
from mst.models.utils.functions import tensor2image, tensor_cam2image, minmax_norm, one_hot

def load_config():
    config_path = Path('config.yaml')
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found in current directory")        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_dataset(name, split, **kwargs):
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
            **kwargs
        )
    elif name == 'DUKE':
        return DUKE_Dataset3D(split=split, **kwargs)
    elif name == 'LIDC':
        return LIDC_Dataset3D(split=split, **kwargs)
    elif name == 'MRNet':
        return MRNet_Dataset3D(split=split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def get_model(name, **kwargs):
    if name == 'DinoV2ClassifierSlice':
        return DinoV2ClassifierSlice
    else:
        raise ValueError(f"Unknown model: {name}. Only DinoV2ClassifierSlice is supported.")


def _pred_trans(model, source, save_attn=False, use_softmax=True):
    # Run model
    if isinstance(model, ResNetSliceTrans) and save_attn:
        pred = model(source, save_attn=save_attn)
    else:
        with torch.no_grad():
            pred = model(source, save_attn=save_attn)

    if use_softmax: # Necessary to standardize the scale before TTA average 
        pred = torch.softmax(pred, dim=-1)

    if not save_attn:
        return pred, None, None 

    # Spatial attention     
    weight = model.get_attention_maps()  # [B*D, Heads, HW]
    weight = weight.mean(dim=1) # Mean of heads 
    spatial_shape = weight.shape[-2:] if isinstance(model, ResNetSliceTrans) else torch.tensor(source.shape[3:])//14 
    weight = weight.view(1, 1, source.shape[2], *spatial_shape)

    # Slice attention 
    weight_slice = model.get_slice_attention() # [B*D, Heads, 1]
    weight_slice = weight_slice.mean(dim=1) # Mean of heads 
    weight_slice = weight_slice.view(1, 1, -1, 1, 1)*torch.ones_like(source, device=weight.device)
    return pred, weight, weight_slice


def _pred_resnet(model, source, src_key_padding_mask, save_attn=False, use_softmax=True):
    # Run model
    if save_attn: # Grads required 
        pred = model(source, src_key_padding_mask=src_key_padding_mask, save_attn=True)
    else:
        with torch.no_grad():
            pred = model(source, src_key_padding_mask=src_key_padding_mask, save_attn=False)
    
    if use_softmax: # Necessary to standardize the scale before TTA average 
        pred = torch.softmax(pred, dim=-1)
      
    if not save_attn:
        return pred, None, None 

    weight = model.get_attention_maps()

    # Slice attention (dummy)
    weight_slice = torch.ones_like(source, device=weight.device)

    return pred, weight, weight_slice


def run_pred(model, batch, save_attn=False, use_softmax=True, use_tta=False):
    source, src_key_padding_mask = batch['source'], batch.get('src_key_padding_mask', None)
    pred_func = None 
    if isinstance(model, ResNetSliceTrans): 
        pred_func = _pred_trans
    elif isinstance(model, ResNet):
        pred_func = _pred_resnet
    elif isinstance(model, DinoV2ClassifierSlice):
        pred_func = _pred_trans

    pred, weight, weight_slice = pred_func(model, source, save_attn, use_softmax)    

    if use_tta:
        for flip_dim in [(2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4),]:
            pred_i, weight_i, weight_slice_i = pred_func(model, torch.flip(source, flip_dim), save_attn, use_softmax)
            pred = pred + pred_i
            if save_attn:
                weight = weight + torch.flip(weight_i, flip_dim)
                weight_slice = weight_slice + torch.flip(weight_slice_i, flip_dim)

        pred = pred / 8
        if save_attn:
            weight = weight / 8
            weight_slice = weight_slice / 8

    # Interpolate to required size 
    if save_attn:
        weight = F.interpolate(weight, size=source.shape[2:], mode='trilinear')

    return pred, weight, weight_slice 




if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Get evaluation parameters from config
    run_dir = config.get('run_dir', './runs')
    run_folder = config.get('run_folder', 'LIDC/ResNet')
    output_dir = config.get('output_dir', './')
    use_tta = config.get('use_tta', False)
    print(f"Using TTA {use_tta}")

    # Parse dataset and model from run_folder
    run_folder = Path(run_folder)
    dataset = run_folder.parent.name
    model_name = run_folder.name.split('_', 1)[0]

    # Setup paths and device
    path_run = Path(run_dir)/run_folder
    results_folder = 'results_tta' if use_tta else 'results'
    path_out = Path(output_dir)/results_folder/run_folder
    path_out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')

    # Setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO) 
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(path_out / f'{Path(__file__).name}.txt', mode='w'))

    # Load test dataset using config
    ds_test = get_dataset(name=dataset, split='test')
    
    dm = DataModule(
        ds_test=ds_test,
        batch_size=1, 
        num_workers=config.get('num_workers', 24),
        pin_memory=True,
    )

    # Initialize model using config
    model = get_model(model_name).load_best_checkpoint(path_run)
    model.to(device)
    model.eval()

    # Process all samples and generate attention maps
    for n, batch in enumerate(tqdm(dm.test_dataloader())):
        source = batch['source']
        uid = batch['uid'][0] if isinstance(batch['uid'], list) else str(batch['uid'].item())

        # Create a separate folder for each sample
        path_out_sample = path_out/'attention'/f'sample_{uid}'
        path_out_sample.mkdir(parents=True, exist_ok=True)
        
        # Run prediction with attention
        pred, weight, weight_slice = run_pred(model, batch, save_attn=True, use_softmax=use_tta, use_tta=use_tta)
        
        # Process attention weights
        weight_slice = weight_slice.detach().cpu()
        weight_slice /= weight_slice.sum()

        weight = weight.detach().cpu()
        weight = weight.clip(*np.quantile(weight, [0.995, 0.999]))

        # Save images
        save_image(tensor2image(source), path_out_sample/f'input_{uid}.png', normalize=True)
        save_image(tensor_cam2image(minmax_norm(source), minmax_norm(weight), alpha=0.5), 
                    path_out_sample/f"overlay_{uid}.png", normalize=False)
        save_image(tensor_cam2image(minmax_norm(source), minmax_norm(weight_slice), alpha=0.5), 
                    path_out_sample/f"overlay_{uid}_slice.png", normalize=False)
        
        # Save ground truth overlay if available (for LIDC dataset)
        if dataset in ['LIDC'] and 'mask' in batch:
            save_image(tensor_cam2image(minmax_norm(source), minmax_norm(batch['mask'].detach().cpu()), alpha=0.5),
                        path_out_sample/f"overlay_{uid}_gt.png", normalize=False)

        # Log the sample processing
        logger.info(f"Processed attention maps for sample {uid}")

    logger.info(f"Completed attention map generation for all samples")
    print(f"Attention maps saved to {path_out/'attention'}")
