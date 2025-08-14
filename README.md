# Multi-sequence MST: Breast MRI Classification Pipeline

A breast MRI classification pipeline using multi-sequence imaging data with DinoV2-based transformers.

**Based on the Medical Slice Transformer (MST) by Müller-Franzes et al.**  
Original repository: https://github.com/mueller-franzes/MST  
Paper: https://arxiv.org/abs/2411.15802  
Licensed under MIT License

## 🏥 Overview

This project extends the MST framework for breast MRI classification with:
- **Multi-sequence MRI processing** (DWI, T1, T2) - our main contribution
- **Cross-validation evaluation** with fold-based statistics - our contribution  
- **Statistical analysis** with DeLong's test and FDR correction - our contribution
- Attention map generation (based on MST, heavily modified)
- Duke dataset integration (based on MST, heavily modified)

## 📚 Attribution

### Original MST Framework
- **Authors**: Müller-Franzes et al.
- **Repository**: https://github.com/mueller-franzes/MST
- **Paper**: https://arxiv.org/abs/2411.15802
- **License**: MIT License

### Components from MST (modified/extended)
- `mst/` - Core framework structure
- `mst/models/dino.py` - DinoV2 classifier (extended for multi-sequence)
- `mst/data/` - Dataset base classes (extended)
- Attention visualization approach (heavily modified)
- Duke dataset integration approach (heavily modified)

### Our Main Contributions
- **Multi-sequence implementation** - primary contribution
- **Cross-validation framework** with fold-based statistics
- **Statistical analysis pipeline** (DeLong's test with FDR correction)
- Breast MRI-specific preprocessing and evaluation

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure CUDA is available for GPU training
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Basic Usage

```bash
# 1. Train a model
python train.py --config config_1.yaml

# 2. Run cross-validation evaluation (with fold-based standard deviation)
python main_crossvalidate.py --model_dir ./runs/t1 --output_dir ./results

# 3. Generate statistical analysis
python delong_fdr_analysis.py

# 4. Create attention maps
python main_predict_attention_duke.py --model_dir ./runs/duke --output_dir ./attention_maps
```

## 📁 Project Structure

```
MulisequenceMST/
├── train.py                           # Main training script (our contribution)
├── main_crossvalidate.py              # Cross-validation evaluation (our contribution)
├── delong_fdr_analysis.py             # Statistical analysis (our contribution)
├── main_predict_attention.py          # Attention map generation (our contribution)
├── main_predict_attention_duke.py     # Duke-specific attention maps (our contribution)
├── train_duke_full.py                 # Duke dataset training (our contribution)
├── config_1.yaml                      # Example configuration (our contribution)
├── example_dataset_structure.csv      # Dataset format example (our contribution)
├── requirements.txt                   # Python dependencies
├── mst/                               # Core package (based on original MST)
│   ├── data/
│   │   ├── datasets/                  # Dataset implementations (extended from MST)
│   │   └── datamodules/               # PyTorch Lightning data modules (from MST)
│   ├── models/
│   │   ├── dino.py                    # DinoV2 classifier (modified from MST)
│   │   └── utils/                     # Model utilities (from MST)
│   └── utils/                         # General utilities (from MST)
├── scripts/preprocessing/duke/         # Duke dataset preprocessing (our contribution)
│   ├── step1_dicom2nifti.py          # DICOM to NIfTI conversion
│   ├── step2a_calc_sub.py             # Subtraction image calculation
│   ├── step2b_crop_or_pad.py          # Image standardization
│   └── step3_create_split.py          # Train/val/test splits
└── helpers/                           # Data processing utilities (our contribution)
    ├── data_analysis/                 # Data analysis tools
    ├── data_cleaning/                 # Data cleaning utilities
    ├── error_analysis/                # Error analysis tools
    ├── ml_preparation/                # ML preparation utilities
    └── attention_map_viewer.py        # GUI attention map viewer
```

## 🔧 Configuration

### Training Configuration (config_1.yaml)

```yaml
# Dataset settings
dataset: DWI
dataset_params:
  path_img: /path/to/images
  path_csv: dataset_birads_seq_splits_cleaned_with_splits.csv
  sequences: ["t2", "t1"]  # Available: dwi0, dwi2, t1, t2, t1n
  fold: 0
  image_resize: [224, 224, 38]
  image_crop: [224, 224, 38]
  flip: true
  noise: true
  random_rotate: true
  random_center: true

# Model settings (using MST framework)
model: DinoV2ClassifierSlice
model_params:
  model_size: s  # s, b, l, g
  pretrained: true
  freeze: false
  slice_fusion: transformer
  use_slice_pos_emb: true

# Training settings
batch_size: 4
accumulate_grad_batches: 1
max_epochs: 200
patience: 20
num_workers: 24
path_root_output: ./runs
```

### Dataset Format

Your CSV file should contain the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| XNAT | Patient ID | PATIENT_001 |
| side | Breast side | left/right |
| record_id | Unique record | 001_L |
| BIRADS | BI-RADS score | 1-5 |
| dwi0, dwi2, t1, t2, t1n | Sequence paths | /path/to/sequence.nii.gz |
| Split | Data split | train/val/test |
| Fold | Cross-validation fold | 0-4 |

## 🎯 Main Scripts

### 1. Training (`train.py`)

Flexible training script with multiple options:

```bash
# Basic training
python train.py

# Custom configuration
python train.py --config config.yaml

# Custom output directory
python train.py --config config.yaml --output_dir ./custom_runs

# Disable W&B logging
python train.py --disable_wandb

# Force CPU training
python train.py --gpus 0
```

**Features:**
- Automatic GPU detection and mixed precision training
- Weights & Biases integration
- Early stopping and model checkpointing
- Class-balanced sampling
- Comprehensive logging

### 2. Cross-Validation (`main_crossvalidate.py`)

Evaluate trained models across multiple folds with **fold-based standard deviation**:

```bash
python main_crossvalidate.py --model_dir ./runs/t1 --output_dir ./cross_val_results
```

**Outputs:**
- ROC curves with fold-based confidence intervals
- Confusion matrices
- Sensitivity analysis (90%, 95%, 97.5% sensitivity points)
- Per-fold and aggregated metrics
- **Standard deviation calculated across folds** (not bootstrap CI)

### 3. Statistical Analysis (`delong_fdr_analysis.py`)

Comprehensive statistical comparison of sequence combinations:

```bash
python delong_fdr_analysis.py
```

**Features:**
- DeLong's test for AUC comparison
- Benjamini-Hochberg FDR correction
- Automatic detection of cross-validation results
- Detailed statistical reports

### 4. Attention Visualization (`main_predict_attention_duke.py`)

Generate attention maps for model interpretability:

```bash
python main_predict_attention_duke.py --model_dir ./runs/duke --output_dir ./attention_maps
```

**Outputs:**
- Spatial attention overlays
- Slice attention distributions
- Individual slice visualizations
- Overview grids
- True positive sample focus

## 🏥 Duke Dataset Integration

Complete preprocessing pipeline for Duke breast cancer dataset:

```bash
# 1. Convert DICOM to NIfTI
python scripts/preprocessing/duke/step1_dicom2nifti.py

# 2. Calculate subtraction images
python scripts/preprocessing/duke/step2a_calc_sub.py

# 3. Standardize image dimensions
python scripts/preprocessing/duke/step2b_crop_or_pad.py

# 4. Create train/val/test splits
python scripts/preprocessing/duke/step3_create_split.py

# 5. Train on Duke dataset
python train_duke_full.py --config duke_config.yaml
```

## 📈 Results and Evaluation

### Cross-Validation Results (Fold-based Statistics)

The pipeline provides comprehensive evaluation metrics with **standard deviation calculated across cross-validation folds**:

- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Sensitivity/Specificity**: At multiple operating points
- **Fold-based Standard Deviation**: Variability across CV folds (not bootstrap CI)
- **Statistical Significance**: DeLong's test with FDR correction

## 🛠️ Development and Customization

### Adding New Sequences

1. Update your CSV file with new sequence columns
2. Modify the `sequences` parameter in your config file
3. Ensure your dataset class handles the new sequences

### Model Customization

The pipeline uses DinoV2ClassifierSlice (from MST framework) with configurable parameters:

```yaml
model_params:
  model_size: s        # Model size: s, b, l, g
  pretrained: true     # Use pretrained weights
  freeze: false        # Freeze backbone
  slice_fusion: transformer  # Slice fusion method
  use_slice_pos_emb: true    # Use positional embeddings
```

## 📄 License

This project is licensed under the MIT License, consistent with the original MST project.

### Original MST License
The original Medical Slice Transformer (MST) framework is licensed under the MIT License.
Copyright (c) Mueller-Franzes et al.

### Our Extensions License
Our extensions and modifications are also licensed under the MIT License.
Copyright (c) T.Nguyen, tMRI lab, University Hospital Erlangen, FAU

See the LICENSE file for full details.
