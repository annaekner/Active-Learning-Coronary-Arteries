import numpy as np
import SimpleITK as sitk
from datetime import datetime

from utilities import spatial_resample_scan
from data_loader import CoronaryArteryDataLoader

def load_sample(index, config):
    # Create an instance of the data loader
    data_loader = CoronaryArteryDataLoader(config)
    
    # Load a single scan
    sample = data_loader.__getitem__(index) # or simply: data_loader[0]

    return sample

def save_segmentation(segmentation, sample, config, subset = 'train'):
    """ Save segmentation as .nii.gz file """

    # Configuration settings
    base_dir = config.base_settings.base_dir
    dataset_processed_dir = config.data_processed.dataset_dir
    train_labels_dir = config.data_processed.train_labels_dir
    test_labels_dir = config.data_processed.test_labels_dir


    # Convert segmentation array to SimpleITK image
    segmentation_nii = sitk.GetImageFromArray(segmentation.astype(int))
    
    # Retrieve original image metadata
    origin, spacing, direction = sample['origin'], sample['spacing'], sample['direction']
    segmentation_nii.SetOrigin(origin)
    segmentation_nii.SetSpacing(spacing)
    segmentation_nii.SetDirection(direction)

    # Path for saving segmentation
    file_name = f"img{sample['image_index']}.nii.gz"

    if subset == 'train':
        output_path = f"{base_dir}/{dataset_processed_dir}/{train_labels_dir}/{file_name}"
    
    elif subset == 'test':
        output_path = f"{base_dir}/{dataset_processed_dir}/{test_labels_dir}/{file_name}"

    # Write the segmentation image to file
    sitk.WriteImage(segmentation_nii, output_path)

def save_resampled_img(index, config, subset = 'train'):
    """ Save resampled image (isotropic voxel spacing) as .nii.gz file """

    # Configuration settings
    base_dir = config.base_settings.base_dir
    data_original_dir = config.data_loader.data_original_dir
    dataset_processed_dir = config.data_processed.dataset_dir
    train_images_dir = config.data_processed.train_images_dir
    test_images_dir = config.data_processed.test_images_dir
    voxel_spacing = config.data_loader.voxel_spacing
    
    # Path to the original image
    img_path = f"{base_dir}/{data_original_dir}/all/{index}.img.nii.gz"

    # Resampe the image to isotropic voxel spacing
    img_nii = sitk.ReadImage(img_path)
    img_nii = spatial_resample_scan(img_nii, voxel_spacing) # Resample image to isotropic pixel spacing

    # Path for saving resampled image
    file_name = f"img{index}_0000.nii.gz"

    if subset == 'train':
         output_path = f"{base_dir}/{dataset_processed_dir}/{train_images_dir}/{file_name}"
        
    elif subset == 'test':
        output_path = f"{base_dir}/{dataset_processed_dir}/{test_images_dir}/{file_name}"

    # Write the segmentation image to file
    sitk.WriteImage(img_nii, output_path)

# def save_centerline_mask(sample, config):
#     """ 
#     THIS DOES NOT WORK !!! 
    
#     Convert the LAD centerline indices to a 3D binary mask and save it 
#         as .nii.gz file with the same size as the original image 
#     """
#     # Load data from sample dictionary
#     img = sample['image']
#     centerline_indices = sample['centerline_indices']

#     # Initialize an empty binary mask with the same shape as the image
#     centerline_mask = np.zeros(img.shape, dtype=np.uint8)

#     # Set the centerline points in the binary mask (value of 1 for LAD centerline)
#     for idx in centerline_indices:
#         centerline_mask[tuple(idx)] = 1

#     # Convert from (x, y, z) back to (z, y, x)
#     centerline_mask = centerline_mask.transpose(2, 1, 0)

#     # Convert the binary mask to a SimpleITK image
#     centerline_nii = sitk.GetImageFromArray(centerline_mask.astype(int))

#     # Retrieve original image metadata
#     origin, spacing, direction = sample['origin'], sample['spacing'], sample['direction']
#     centerline_nii.SetOrigin(origin)
#     centerline_nii.SetSpacing(spacing)
#     centerline_nii.SetDirection(direction)

#     # Path for saving segmentation
#     file_name = f"img{sample['image_index']}_0002.nii.gz"
#     output_path = f"{config.base_settings.base_dir}/Datasets_raw/Dataset001_Heart/imagesTr/{file_name}"
    
#     # Write the segmentation image to file
#     sitk.WriteImage(centerline_nii, output_path)
