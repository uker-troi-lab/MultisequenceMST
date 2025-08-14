from pathlib import Path 
import torchio as tio 
import torch
import numpy as np 
from multiprocessing import Pool
from tqdm import tqdm
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def crop_breast_height(image, margin_top=10):
    "Crop height to 256 and try to cover breast based on intensity localization"
    # threshold = int(image.data.float().quantile(0.9))
    threshold = int(np.quantile(image.data.float(), 0.9))
    foreground = image.data>threshold
    fg_rows = foreground[0].sum(axis=(0, 2))
    top = min(max(512-int(torch.argwhere(fg_rows).max()) - margin_top, 0), 256)
    bottom = 256-top
    return  tio.Crop((0,0, bottom, top, 0, 0))


def preprocess(path_dir, path_root_in_data, path_root_out_data):
    try:
        # Check if pre.nii.gz exists
        if not (path_dir/'pre.nii.gz').exists():
            logger.warning(f"Skipping {path_dir.name}: pre.nii.gz not found")
            return {
                'status': 'skipped', 
                'patient': path_dir.name, 
                'reason': 'pre.nii.gz not found',
                'files_present': ', '.join([f.name for f in path_dir.glob('*.nii.gz')])
            }

        # -------- Settings --------------
        ref_img = tio.ScalarImage(path_dir/'pre.nii.gz')

        # Option: Static 
        target_spacing = (0.7, 0.7, 3) 
        target_shape = (512, 512, 32)
        ref_img = tio.Resample(target_spacing)(ref_img)

        transform = tio.Compose([
            tio.Resample(ref_img), # Resample to reference image to ensure that origin, direction, etc, fit
            tio.CropOrPad(target_shape, padding_mode=0),
            tio.ToCanonical(),
        ])
        crop_height = crop_breast_height(transform(ref_img))     
        split_side = {
            'right': tio.Crop((256, 0, 0, 0, 0, 0)),
            'left': tio.Crop((0, 256, 0, 0, 0, 0)),
        }
        
        for n, path_img in enumerate(path_dir.glob('*.nii.gz')):
            # Read image 
            img = tio.ScalarImage(path_img)

            # Preprocess (eg. Crop/Pad)
            img = transform(img)

            # Crop bottom and top so that height is 256 and breast is preserved  
            img = crop_height(img)

            # Split left and right side 
            for side in ['left', 'right']:
                # Create output directory 
                path_out_dir = path_root_out_data/f"{path_dir.relative_to(path_root_in_data)}_{side}"
                path_out_dir.mkdir(exist_ok=True, parents=True)

                # Crop left/right side 
                img_side = split_side[side](img)

                # Save 
                img_side.save(path_out_dir/path_img.name)

        return {
            'status': 'success', 
            'patient': path_dir.name, 
            'reason': None,
            'files_present': ', '.join([f.name for f in path_dir.glob('*.nii.gz')])
        }

    except Exception as e:
        logger.error(f"Error processing {path_dir.name}: {str(e)}")
        return {
            'status': 'error', 
            'patient': path_dir.name, 
            'reason': str(e),
            'files_present': ', '.join([f.name for f in path_dir.glob('*.nii.gz')])
        }

if __name__ == "__main__":
    path_root = Path(r'/path/to/duke/dataset')  # Update this path to your Duke dataset location
    path_root_in_data = path_root/'preprocessed/data'
    path_root_out = path_root/'preprocessed_crop'
    path_root_out_data = path_root_out/'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)
    
    path_patients = list(path_root_in_data.iterdir())
    
    # Initialize results list
    results = []
    
    # Option 1: Multi-CPU 
    with Pool() as pool:
        # Use starmap to pass multiple arguments
        process_args = [(p, path_root_in_data, path_root_out_data) for p in path_patients]
        for result in tqdm(pool.starmap(preprocess, process_args), total=len(path_patients)):
            results.append(result)

    # Create DataFrame and save to CSV
    df_results = pd.DataFrame(results)
    csv_path = path_root/'preprocessing_report.csv'
    df_results.to_csv(csv_path, index=False)

    # Print summary
    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')

    print("\nProcessing Summary:")
    print(f"Total patients: {total}")
    print(f"Successfully processed: {successful}")
    print(f"Skipped (missing pre.nii.gz): {skipped}")
    print(f"Errors: {errors}")
    print(f"\nDetailed report saved to: {csv_path}")
