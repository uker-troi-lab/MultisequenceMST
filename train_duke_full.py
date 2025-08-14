import argparse
from pathlib import Path
import yaml
import json
import torch
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import wandb
import warnings
warnings.filterwarnings("ignore", module="xformers")

from mst.data.datasets.dataset_3d_duke import DUKE_Dataset3D
from mst.data.datamodules import DataModule
from mst.models.dino import DinoV2ClassifierSlice

# use: python scripts/train_duke_full.py --config configs/duke_training.yaml --output_dir ./runs/duke_full

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, output_path):
    with open(output_path, 'w') as f:
        yaml.dump(config, f)

def get_dataset(config, name, split, **kwargs):    
    dataset_params = config.get('dataset_params', {})
    path_img = Path(dataset_params.get('path_img', ''))
    
    if name == 'DUKE':
        return DUKE_Dataset3D(
            path_img=path_img,
            path_csv=dataset_params.get('path_csv'),
            sequences=dataset_params.get('sequences', ["t1"]),
            fold=dataset_params.get('fold', 0),  # Default to fold 0
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
    """Get model instance by name."""
    model_params = config.get('model_params', {})
    
    if name == 'DinoV2ClassifierSlice':
        return DinoV2ClassifierSlice(
            in_ch=3,  # This stays 3 as it's what DinoV2 expects
            out_ch=2,
            spatial_dims=2,
            **model_params,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model: {name}. Only DinoV2ClassifierSlice is supported.")

def train_model(config, output_dir):
    """Train a model on the full Duke dataset."""
    try:
        # Setup output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Override image path and dataset parameters for Duke dataset
        original_dataset_params = config.get('dataset_params', {}).copy()
        original_dataset_params['path_img'] = os.path.join(os.environ['DATA_DIR'], "DUKE/preprocessed_crop/data")
        original_dataset_params['path_csv'] = os.path.join(os.environ['DATA_DIR'], "DUKE/preprocessed_crop/splits/split.csv")
        config['dataset_params'] = original_dataset_params
        config['dataset'] = 'DUKE'  # Override dataset to use Duke
        
        # Save the configuration
        save_config(config, output_dir / 'used_config.yaml')
        
        # Setup logging
        logger = logging.getLogger("duke_training")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(output_dir / 'training.log')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info(f"Starting training with configuration: {config}")
        logger.info(f"Output directory: {output_dir}")
        
        # Get datasets
        logger.info("Loading datasets...")
        
        # For training on the full dataset, we'll use both train and validation sets for training
        # and keep a small portion for validation
        ds_train = get_dataset(config, config['dataset'], split='train')
        ds_val = get_dataset(config, config['dataset'], split='val')
        
        # Optionally, we can include the test set as well if we want to train on absolutely everything
        if config.get('use_test_set', False):
            ds_test = get_dataset(config, config['dataset'], split='test')
            logger.info(f"Including test set in training. Test samples: {len(ds_test)}")
            
            # Create a combined dataset for training
            train_samples = []
            train_samples.extend([ds_train[i] for i in range(len(ds_train))])
            train_samples.extend([ds_val[i] for i in range(len(ds_val))])
            train_samples.extend([ds_test[i] for i in range(len(ds_test))])
            
            # No validation set in this case (or keep a small random subset)
            val_samples = []
            
            logger.info(f"Training on all data. Total samples: {len(train_samples)}")
        else:
            # Use 90% of validation set for training, keep 10% for actual validation
            val_size = len(ds_val)
            val_split_idx = int(val_size * 0.9)
            
            # Create datasets
            train_samples = []
            train_samples.extend([ds_train[i] for i in range(len(ds_train))])
            train_samples.extend([ds_val[i] for i in range(val_split_idx)])
            
            val_samples = [ds_val[i] for i in range(val_split_idx, val_size)]
            
            logger.info(f"Training samples: {len(train_samples)}")
            logger.info(f"Validation samples: {len(val_samples)}")
        
        # Calculate class weights for balanced sampling
        class_counts = ds_train.df[ds_train.LABEL].value_counts()
        class_weights = 0.5 / class_counts
        weights = ds_train.df[ds_train.LABEL].map(lambda x: class_weights[x]).values
        
        # Calculate steps per epoch for logging
        samples = len(train_samples) + len(val_samples)
        batch_size = config.get('batch_size', 8)
        accumulate_grad_batches = config.get('accumulate_grad_batches', 1)
        steps_per_epoch = samples / batch_size / accumulate_grad_batches
        
        # Create data module
        data_module = DataModule(
            ds_train=train_samples,
            ds_val=val_samples,
            ds_test=val_samples,  # Use validation samples for test as well
            batch_size=batch_size,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            weights=weights,
            num_train_samples=min(len(train_samples), config.get('num_train_samples', 2000))
        )
        
        # Initialize model
        logger.info(f"Initializing {config['model']} model...")
        model = get_model(config, config['model'])
        
        # Setup callbacks
        callbacks = []
        
        # Monitor metric
        to_monitor = config.get('monitor_metric', "val/AUC_ROC")
        min_max = config.get('monitor_mode', "max")
        log_every_n_steps = config.get('log_every_n_steps', 50)
        
        # Wandb logger
        wandb_logger = WandbLogger(
            project=f'Classifier_{config["dataset"]}', 
            name=f"{config['model']}_full_training", 
            log_model=False
        )
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        # Progress bar
        progress_bar = TQDMProgressBar(refresh_rate=50)
        callbacks.append(progress_bar)
        
        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            monitor=to_monitor,
            save_last=True,
            save_top_k=1,
            mode=min_max,
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping callback
        if config.get('early_stopping', True):
            early_stop_callback = EarlyStopping(
                monitor=to_monitor,
                min_delta=0.0,
                patience=config.get('patience', 50),
                mode=min_max
            )
            callbacks.append(early_stop_callback)
        
        # Setup accelerator
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        torch.set_float32_matmul_precision('high')
        
        # Initialize trainer
        trainer = Trainer(
            accelerator=accelerator,
            accumulate_grad_batches=accumulate_grad_batches,
            precision='16-mixed',
            default_root_dir=str(output_dir),
            callbacks=callbacks,
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            log_every_n_steps=log_every_n_steps,
            limit_val_batches=min(len(val_samples) if val_samples else 1, 200),
            max_epochs=config.get('max_epochs', 1000),
            num_sanity_val_steps=2,
            logger=wandb_logger
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.fit(model, datamodule=data_module)
        
        # Save the best model path
        model.save_best_checkpoint(output_dir, checkpoint_callback.best_model_path)
        
        # Save best model info
        best_model_info = {
            'best_model_path': checkpoint_callback.best_model_path,
            'best_model_score': float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None,
            'best_model_epoch': checkpoint_callback.best_epoch if hasattr(checkpoint_callback, 'best_epoch') else None
        }
        
        with open(output_dir / 'best_model_info.json', 'w') as f:
            json.dump(best_model_info, f, indent=2)
        
        logger.info("Training completed successfully!")
        wandb.finish(quiet=True)
        return best_model_info
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='./runs/duke_full', help='Directory to save model and results')
    parser.add_argument('--use_test_set', action='store_true', help='Include test set in training')
    args = parser.parse_args()

    print("\n=== Starting Duke Full Dataset Training ===")
    print(f"Current working directory: {Path.cwd()}")
    
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    
    print(f"\nChecking paths:")
    print(f"Config file: {config_path} (exists: {config_path.exists()})")
    print(f"Output directory: {output_dir} (exists: {output_dir.exists()})")
    
    if not config_path.exists():
        print(f"ERROR: Config file does not exist: {config_path}")
        return
    
    # Load configuration
    config = load_config(config_path)
    
    # Override with command line arguments
    if args.use_test_set:
        config['use_test_set'] = True
    
    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"run_{timestamp}"
    
    # Train the model
    train_model(config, output_dir)
    
    print("\nDuke full dataset training completed successfully!")

if __name__ == "__main__":
    main()
