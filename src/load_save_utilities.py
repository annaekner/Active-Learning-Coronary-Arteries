import numpy as np
import SimpleITK as sitk
from datetime import datetime

from tools import spatial_resample_scan
from data_loader import CoronaryArteryDataLoader

def load_sample(index, config, subset = 'train', label_type = 'full_coronary_tree'):
    """ Load scan from the dataset into the sample dictionary """

    # Create an instance of the data loader
    data_loader = CoronaryArteryDataLoader(config, subset, label_type)
    
    # Load a single scan
    sample = data_loader.__getitem__(index) # or simply: data_loader[0]

    return sample

def load_prediction(img_index, config):
    """ Load prediction (LAD segmentation) as a numpy array """

    # Configuration settings
    base_dir = config.base_settings.base_dir
    data_predicted_dir = config.data_predicted.dataset_dir
    date = config.data_predicted.date

    # Load the prediction
    prediction_path = f"{base_dir}/{data_predicted_dir}/{date}/img{img_index}.nii.gz"

    # Convert from .nii.gz file to numpy array
    prediction_nii = sitk.ReadImage(prediction_path)
    prediction = sitk.GetArrayFromImage(prediction_nii)
    prediction = prediction.astype(np.bool)

    return prediction, prediction_nii

def save_resampled_img(img_index, config, subset = 'train'):
    """ Save resampled image (isotropic voxel spacing) as .nii.gz file """

    # Configuration settings
    base_dir = config.base_settings.base_dir
    data_original_dir = config.data_loader.data_original_dir
    dataset_processed_dir = config.data_processed.dataset_dir
    train_images_dir = config.data_processed.train_images_dir
    test_images_dir = config.data_processed.test_images_dir
    voxel_spacing = config.data_loader.voxel_spacing
    
    # Path to the original image
    img_path = f"{base_dir}/{data_original_dir}/all/{img_index}.img.nii.gz"

    # Resampe the image to isotropic voxel spacing
    img_nii = sitk.ReadImage(img_path)
    img_nii = spatial_resample_scan(img_nii, voxel_spacing) # Resample image to isotropic pixel spacing

    # Path for saving resampled image
    file_name = f"img{img_index}_0000.nii.gz"

    if subset == 'train':
         output_path = f"{base_dir}/{dataset_processed_dir}/{train_images_dir}/{file_name}"
        
    elif subset == 'test':
        output_path = f"{base_dir}/{dataset_processed_dir}/{test_images_dir}/{file_name}"

    # Write the segmentation image to file
    sitk.WriteImage(img_nii, output_path)

def save_segmentation(segmentation, sample, config, subset = 'train', output_path = None):
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

    if not output_path:
        if subset == 'train':
            output_path = f"{base_dir}/{dataset_processed_dir}/{train_labels_dir}/{file_name}"
        
        elif subset == 'test':
            output_path = f"{base_dir}/{dataset_processed_dir}/{test_labels_dir}/{file_name}"

    # Write the segmentation image to file
    sitk.WriteImage(segmentation_nii, output_path)

def save_distance_map(distance_map, sample, config, subset = 'train', output_path = None):
    """ Save distance map as .nii.gz file """

    # Configuration settings
    base_dir = config.base_settings.base_dir
    dataset_processed_dir = config.data_processed.dataset_dir
    train_images_dir = config.data_processed.train_images_dir
    test_images_dir = config.data_processed.test_images_dir

    # Convert distance map array to SimpleITK image
    distance_map_nii = sitk.GetImageFromArray(distance_map.astype(float))
    
    # Retrieve original image metadata
    origin, spacing, direction = sample['origin'], sample['spacing'], sample['direction']
    distance_map_nii.SetOrigin(origin)
    distance_map_nii.SetSpacing(spacing)
    distance_map_nii.SetDirection(direction)

    # Path for saving distance map
    file_name = f"img{sample['image_index']}_0001.nii.gz"

    if not output_path:
        if subset == 'train':
            output_path = f"{base_dir}/{dataset_processed_dir}/{train_images_dir}/{file_name}"
        
        elif subset == 'test':
            output_path = f"{base_dir}/{dataset_processed_dir}/{test_images_dir}/{file_name}"

    # Write the distance map image to file
    sitk.WriteImage(distance_map_nii, output_path)