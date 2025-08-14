from pathlib import Path
import logging
import torchio as tio
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)

def find_t1_file(path_patient):
    """Find T1 file with flexible naming"""
    # Skip if already processed
    if (path_patient/'T1_resampled.nii.gz').exists():
        return None
        
    for file in path_patient.glob('*.nii.gz'):
        # Convert filename to lowercase for case-insensitive matching
        filename_lower = file.name.lower()
        # Check if it's a T1 file but not already resampled
        if ('t1' in filename_lower or 't1' in filename_lower.replace(' ', '')) and 'resampled' not in filename_lower:
            return file
    return None


def process(path_patient):
    try:
        patient_id = path_patient.name
        logger.debug(f"Patient ID: {patient_id}")
        
        # Skip if already processed
        if (path_patient/'T1_resampled.nii.gz').exists():
            return {
                'patient_id': patient_id,
                'status': 'skipped',
                'missing_files': 'Already processed'
            }

        # Find T1 file
        t1_file = find_t1_file(path_patient)
        
        # Check if required files exist
        required_files = ['pre.nii.gz', 'post_1.nii.gz']
        missing_files = []
        
        for file in required_files:
            if not (path_patient/file).exists():
                missing_files.append(file)
        
        if t1_file is None:
            missing_files.append('T1.nii.gz (or similar)')
            
        if missing_files:
            return {
                'patient_id': patient_id,
                'status': 'skipped',
                'missing_files': ', '.join(missing_files)
            }

        # Compute subtraction image
        logger.debug(f"Compute and write sub to disk")
        dyn0_nii = sitk.ReadImage(str(path_patient/'pre.nii.gz'), sitk.sitkInt16)
        dyn1_nii = sitk.ReadImage(str(path_patient/'post_1.nii.gz'), sitk.sitkInt16)
        dyn0 = sitk.GetArrayFromImage(dyn0_nii)
        dyn1 = sitk.GetArrayFromImage(dyn1_nii)
        sub = dyn1-dyn0
        sub = sub-sub.min()
        sub = sub.astype(np.uint16)
        sub_nii = sitk.GetImageFromArray(sub)
        sub_nii.CopyInformation(dyn0_nii)
        sitk.WriteImage(sub_nii, str(path_patient/'sub.nii.gz'))
        
        # Compute resampled T1-weighted image
        logger.debug(f"Compute and write resampled T1 to disk")
        t1_nii = sitk.ReadImage(str(t1_file), sitk.sitkInt16)
        t1_resampled_nii = sitk.Resample(t1_nii, dyn0_nii, sitk.Transform(),
                                      sitk.sitkLinear, 0, dyn0_nii.GetPixelID())
        sitk.WriteImage(t1_resampled_nii, str(path_patient/'T1_resampled.nii.gz'))
        
        return {
            'patient_id': patient_id,
            'status': 'success',
            'missing_files': None
        }
        
    except Exception as e:
        return {
            'patient_id': patient_id,
            'status': 'error',
            'missing_files': str(e)
        }

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Path settings
    path_root = Path(r'/path/to/duke/dataset')  # Update this path to your Duke dataset location
    path_root_out = path_root/'preprocessed'
    path_root_out_data = path_root_out/'data'

    files = list(path_root_out_data.iterdir())
    
    # Process files and collect results
    results = []
    with Pool() as pool:
        for result in tqdm(pool.imap_unordered(process, files), total=len(files)):
            results.append(result)

    # Create DataFrame with results
    df_results = pd.DataFrame(results)
    
    # Save results
    df_results.to_csv(path_root/'preprocessing_report.csv', index=False)
    
    # Print summary
    total = len(df_results)
    successful = len(df_results[df_results['status'] == 'success'])
    skipped = len(df_results[df_results['status'] == 'skipped'])
    errors = len(df_results[df_results['status'] == 'error'])
    
    print("\nProcessing Summary:")
    print(f"Total patients: {total}")
    print(f"Successfully processed: {successful}")
    print(f"Skipped (missing files): {skipped}")
    print(f"Errors: {errors}")
    print(f"\nDetailed report saved to: {path_root/'preprocessing_report.csv'}")
