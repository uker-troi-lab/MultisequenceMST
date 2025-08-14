"""
Multi-sequence MST Training Script

This script extends the Medical Slice Transformer (MST) framework for multi-sequence 
breast MRI classification training.

Original MST framework by Müller-Franzes et al.
Repository: https://github.com/mueller-franzes/MST
Paper: https://arxiv.org/abs/2411.15802
Licensed under MIT License

This extension focuses on multi-sequence implementation and is our main contribution.
"""

import argparse
from pathlib import Path
from datetime import datetime
import wandb 
import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import yaml
import os
from pytorch_lightning.callbacks import TQDMProgressBar

from mst.data.datasets.dataset_3d_duke import DUKE_Dataset3D
from mst.data.datasets.dataset_3d_dwi import DWI_Dataset3D, BilateralDWI_Dataset3D

from mst.data.datamodules import DataModule
from mst.models.dino import DinoV2ClassifierSlice

import warnings
warnings.filterwarnings("ignore", module="xformers")

def load_config(config_path):
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")        
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
            fold= dataset_params.get('fold'),
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
        return DUKE_Dataset3D(split=split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def get_model(config, name, **kwargs):    
    model_params = config.get('model_params', {})
        
    if name == 'DinoV2ClassifierSlice':        
        return DinoV2ClassifierSlice(
            in_ch=3,  # This stays 3 as it's what DinoV2 expects            
            out_ch=2, 
            spatial_dims=2,            
            **model_params,  # This will pass other model parameters like model_size, pretrained, etc.
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model: {name}. Only DinoV2ClassifierSlice is supported.")

def main():
    parser = argparse.ArgumentParser(description='Train MST model')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Override output directory from config')
    parser.add_argument('--disable_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--gpus', type=int, default=None,
                       help='Number of GPUs to use (0 for CPU)')
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    config = load_config(config_path)
    
    print(f"Using configuration: {config_path}")
    print(f"Dataset: {config['dataset']}")
    print(f"Model: {config['model']}")
    print(f"Sequences: {config.get('dataset_params', {}).get('sequences', 'unknown')}")
    print(f"Fold: {config.get('dataset_params', {}).get('fold', 'unknown')}")

    #------------ Settings/Defaults ----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    
    # Use output directory from args or config
    if args.output_dir:
        path_run_dir = Path(args.output_dir) / f"{config['model']}_{current_time}"
    else:
        path_run_dir = Path(config['path_root_output']) / config['dataset'] / f"{config['model']}_{current_time}_multi"
    
    path_run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the used configuration
    with open(path_run_dir / 'used_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)   
    
    # Setup accelerator
    if args.gpus is not None:
        accelerator = 'cpu' if args.gpus == 0 else 'gpu'
    else:
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    torch.set_float32_matmul_precision('high')
    
    print(f"Using accelerator: {accelerator}")
    print(f"Output directory: {path_run_dir}")

    # ------------ Load Data ----------------
    print("Loading datasets...")
    ds_train = get_dataset(config, config['dataset'], split='train')
    ds_val = get_dataset(config, config['dataset'], split='val')
    
    print(f"Training samples: {len(ds_train)}")
    print(f"Validation samples: {len(ds_val)}")
    
    samples = len(ds_train) + len(ds_val)
    batch_size = config.get('batch_size', 2)
    accumulate_grad_batches = config.get('accumulate_grad_batches', 1)
    steps_per_epoch = samples / batch_size / accumulate_grad_batches

    class_counts = ds_train.df[ds_train.LABEL].value_counts()
    class_weights = 0.5 / class_counts
    weights = ds_train.df[ds_train.LABEL].map(lambda x: class_weights[x]).values

    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_val,
        batch_size=batch_size, 
        pin_memory=True,
        weights=weights,
        num_workers=config.get('num_workers', 4),
        num_train_samples=min(len(ds_train), config.get('num_train_samples', 2000))
    )

    # ------------ Initialize Model ------------
    print("Initializing model...")
    model = get_model(config, config['model'])
    print(f"Model: {type(model).__name__}")
    
    # -------------- Training Initialization ---------------
    to_monitor = "val/AUC_ROC"
    min_max = "max"
    log_every_n_steps = config.get('log_every_n_steps', 50)
    
    # Setup logger
    if args.disable_wandb:
        logger = False
        lr_monitor = None  # Disable LR monitor when no logger
        print("Weights & Biases logging disabled")
    else:
        logger = WandbLogger(project=f'Classifier_{config["dataset"]}', name=type(model).__name__, log_model=False)
        lr_monitor = LearningRateMonitor(logging_interval='step')
    
    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0,
        patience=config.get('patience', 50),
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        monitor=to_monitor,
        save_last=True,
        save_top_k=1,
        mode=min_max,
    )
    progress_bar = TQDMProgressBar(refresh_rate=50) 
    
    # Setup callbacks
    callbacks = [checkpointing, early_stopping, progress_bar]
    if lr_monitor:
        callbacks.append(lr_monitor)
    
    trainer = Trainer(
        accelerator=accelerator,
        accumulate_grad_batches=accumulate_grad_batches,
        precision='16-mixed',
        default_root_dir=str(path_run_dir),
        callbacks=callbacks,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=log_every_n_steps,
        limit_val_batches=min(len(ds_val), 200),
        max_epochs=config.get('max_epochs', 1000),
        num_sanity_val_steps=2,
        logger=logger
    )

    # ---------------- Execute Training ----------------
    print("Starting training...")
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(path_run_dir, checkpointing.best_model_path)
    
    if not args.disable_wandb and logger:
        wandb.finish(quiet=True)
    
    print("Training completed!")
    print(f"Results saved to: {path_run_dir}")

if __name__ == "__main__":
    main()
