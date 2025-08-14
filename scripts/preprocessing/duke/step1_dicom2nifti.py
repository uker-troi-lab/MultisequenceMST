

from pathlib import Path 
import logging  
import pandas as pd 
from multiprocessing import Pool

import numpy as np 
import pydicom
import pydicom.datadict
import pydicom.dataelem
import pydicom.sequence
import pydicom.valuerep
from tqdm import tqdm
import SimpleITK as sitk 



# Logging 
# path_log_file = path_root/'preprocessing.log'
logger = logging.getLogger(__name__)
# s_handler = logging.StreamHandler(sys.stdout)
# f_handler = logging.FileHandler(path_log_file, 'w')
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     handlers=[s_handler, f_handler])



def maybe_convert(x):
    if isinstance(x, pydicom.sequence.Sequence):
        # return [maybe_convert(item) for item in x]
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.dataset.Dataset):  
        # return dataset2dict(x)
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.multival.MultiValue):
        return list(x)
    elif isinstance(x, pydicom.valuerep.PersonName):
        return str(x)
    else:
        return x 


def dataset2dict(ds, exclude=['PixelData', '']):
    return {keyword:value for key in ds.keys() 
            if ((keyword := ds[key].keyword) not in exclude)  and ((value := maybe_convert(ds[key].value)) is not None) }


def get_standardized_name(seq_name):
    """Map sequence names to standardized names (pre, post_1, T1)"""
    seq_name = seq_name.lower()
    
    # Remove any leading numbers (like "3.000000-")
    if '-' in seq_name:
        seq_name = seq_name.split('-', 1)[1]
    
    # Pattern 1: explicit pre/post naming
    if 'dyn pre' in seq_name:
        return 'pre'
    elif 'dyn 1st pass' in seq_name or 'ph1' in seq_name:
        return 'post_1'
    elif 'dyn 2nd pass' in seq_name or 'ph2' in seq_name:
        return 'post_2'
    elif 'dyn 3rd pass' in seq_name or 'ph3' in seq_name:
        return 'post_3'
    elif 'dyn 4th pass' in seq_name or 'ph4' in seq_name:
        return 'post_4'
    # Pattern 2: basic dyn sequence is pre-contrast
    elif 'ax 3d dyn' in seq_name and 'ph' not in seq_name:
        return 'pre'
    # T1 sequences
    elif any(t1_pattern in seq_name for t1_pattern in ['t1 tse', 't1-', '3d t1']):
        return 'T1'
    
    # Keep original name if no match
    return seq_name



def series2nifti(args):         
    series_info, root_in, root_out_data = args
    seq_name, path_series = series_info 
    
    # Initialize reader for this process
    reader = sitk.ImageSeriesReader()
    
    # Remove duplicate Duke-Breast-Cancer-MRI if present
    path_series = str(path_series).replace('Duke-Breast-Cancer-MRI/', '')
    path_series = root_in/Path(path_series)
    
    if not path_series.is_dir():
        logger.warning(f"Expect directory but found file: {path_series}:")
        return 
    
    try:
        # Read DICOM
        dicom_names = reader.GetGDCMSeriesFileNames(str(path_series))
        reader.SetFileNames(dicom_names) 
        img_nii = reader.Execute()

        # Read Metadata 
        ds = pydicom.dcmread(next(path_series.glob('*.dcm'), None), stop_before_pixels=True)
        metadata = dataset2dict(ds)
        
        # Create output folder 
        path_out_dir = root_out_data/path_series.parts[-3]
        path_out_dir.mkdir(exist_ok=True, parents=True)

        # Get standardized name
        std_name = get_standardized_name(seq_name)
        
        # Write 
        logger.info(f"Writing file: {std_name}.nii.gz")
        path_file = path_out_dir/f'{std_name}.nii.gz'
        sitk.WriteImage(img_nii, path_file)

        metadata['_path_file'] = str(path_file.relative_to(root_out_data))
        metadata['original_sequence_name'] = seq_name
        metadata['standardized_name'] = std_name
        return metadata

    except Exception as e:
        logger.warning(f"Error in: {path_series}")
        logger.warning(str(e))





if __name__ == "__main__":
    # Settings
    path_root = Path(r'/path/to/duke/dataset')  # Update this path to your Duke dataset location
    path_root_in = path_root
    path_root_out = path_root_in/'preprocessed'
    path_root_out_data = path_root_out/'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)

    # Init reader 
    reader = sitk.ImageSeriesReader()

    # Read the metadata CSV
    df_metadata = pd.read_csv(path_root/'metadata.csv')
    
    # Process paths similar to original implementation
    df_metadata['PatientID'] = df_metadata['File Location'].apply(
        lambda x: x.split('\\')[1]  # Gets Breast_MRI_xxx
    )
    
    df_metadata['SequenceName'] = df_metadata['File Location'].apply(
        lambda x: x.split('\\')[-1].split('-')[1]  # Gets sequence name like "ax t1 tse c"
    )
    
    # Create series info matching original format
    series = list(zip(
        df_metadata['SequenceName'],
        df_metadata['File Location']
    ))

    # Validate 
    print(f"Number Series: {len(series)} of 5034 (5034+127=5161)")

    process_args = [(s, path_root_in, path_root_out_data) for s in series]
    
    # Option 1: Multi-CPU 
    metadata_list = []
    with Pool() as pool:
        for meta in tqdm(pool.imap_unordered(series2nifti, process_args), total=len(process_args)):
            if meta is not None:
                metadata_list.append(meta)

    # Save metadata
    df = pd.DataFrame(metadata_list)
    df.to_csv(path_root_out/'metadata.csv', index=False)

    # Check export 
    num_series = len([path for path in path_root_out_data.rglob('*.nii.gz')])
    print(f"Number Series: {num_series} of 5034 (5034+127=5161)")
