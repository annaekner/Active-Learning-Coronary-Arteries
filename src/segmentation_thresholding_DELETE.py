import os
import hydra
import vtk
import skimage
import logging
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.ndimage import gaussian_gradient_magnitude

from data_loader import CoronaryArteryDataLoader

# Set up logging
log = logging.getLogger(__name__)

def load_sample(index, config):
    # Create an instance of the data loader
    data_loader = CoronaryArteryDataLoader(config)
    
    # Load a single scan
    sample = data_loader.__getitem__(index) # or simply: data_loader[0]

    return sample

def mask1(patch, value):
    """ Compute the gradient threshold as a percentile of the gradient magnitude """
    # Compute gradient magnitude in the patch
    sigma = 1  # Larger sigma values will smooth the image more before computing the gradient
    patch_gradient = gaussian_gradient_magnitude(patch, sigma)

    # Determine the minimum threshold based on significant gradient changes
    # Percentile value of the gradient acts as the boundary indicator 
    percentile = 90
    gradient_threshold = np.percentile(patch_gradient, percentile)

    # Create a mask where the intensity is within the desired range:
    # 1. Intensity greater than where the gradient is significant (boundary)
    # 2. Intensity less than the centerline HU value (vessel maximum HU)
    min_threshold = patch_gradient < gradient_threshold
    max_threshold = value

    mask = (patch > min_threshold) & (patch < max_threshold)
    return mask

def mask2(patch, value):
    """ Compute the gradient threshold based on the range of gradient values """
    # Compute the gradient magnitude of the patch
    sigma = 1  # Larger sigma values will smooth the image more before computing the gradient
    patch_gradient = gaussian_gradient_magnitude(patch, sigma)

    # Calculate the minimum and maximum gradient values
    min_gradient = patch_gradient.min()
    max_gradient = patch_gradient.max()

    # Calculate the range of gradient values
    gradient_range = max_gradient - min_gradient

    # Set the gradient threshold based on the strictness factor (fraction of the range)
    # Strictness is a value between 0 and 1 that controls how strict the threshold is. 
    # Lower values (closer to 0) will make the threshold less strict, allowing more voxels with small gradients to be included. 
    # Higher values (closer to 1) make it stricter, excluding more voxels with smaller gradients.
    strictness = 0.8
    gradient_threshold = min_gradient + strictness * gradient_range

    # Create a mask where the intensity is within the desired range:
    # 1. Intensity greater than where the gradient is significant (boundary)
    # 2. Intensity less than the centerline HU value (vessel maximum HU)
    min_threshold = patch_gradient < gradient_threshold
    max_threshold = value

    mask = (patch > min_threshold) & (patch < max_threshold)
    return mask

def mask3(patch, value, config):
    """ Compute the gradient threshold based on the range of gradient values """
    # Compute the gradient magnitude of the patch
    sigma = config.segmentation.sigma  # Larger sigma values will smooth the image more before computing the gradient
    patch_gradient = gaussian_gradient_magnitude(patch, sigma)

    # Calculate the minimum and maximum gradient values
    min_gradient = patch_gradient.min()
    max_gradient = patch_gradient.max()

    # Calculate the range of gradient values
    gradient_range = max_gradient - min_gradient

    # Set the gradient threshold based on the strictness factor (fraction of the range)
    # Strictness is a value between 0 and 1 that controls how strict the threshold is. 
    # Lower values (closer to 0) will exclude more voxels with large gradients (more strict boundary)
    # Higher values (closer to 1) will allow more voxels with large gradients to be included.
    strictness = config.segmentation.strictness
    gradient_threshold = min_gradient + strictness * gradient_range

    # Create a mask where the intensity is within the desired range:
    # 1. Intensity greater than where the gradient is significant (boundary)
    # 2. Intensity less than the centerline HU value (vessel maximum HU)
    intensity_mask = (patch < value) & (patch > 0)       # Vessel HU intensity range
    gradient_mask = patch_gradient < gradient_threshold  # Boundary detection based on gradients

    mask = intensity_mask & gradient_mask  # Combine both conditions

    return mask

def segment_LAD(sample, config):
    """ Segment the left anterior descending artery (LAD) using adaptive thresholding """

    # Load data from sample dictionary
    img = sample['image']
    label = sample['label']
    centerline = sample['centerline']
    centerline_indices = sample['centerline_indices']
    centerline_values = sample['centerline_values']
    img_index = sample['image_index']

    # Configuration settings
    patch_size = config.segmentation.patch_size

    img_size = img.shape                    # Image size
    num_points = len(centerline_indices)    # Number of centerline points

    # LAD segmentation steps:
    # 1. Initialize empty segmentation array with same size as image
    # 2. Loop over centerline points
    #     2.1. Extract image patch around centerline point
    #     2.2. Get HU value of centerline point to use as threshold
    #     2.3. Threshold image patch
    #     2.4. Add segmented voxels from patch to the segmentation array
    #     2.5. Repeat
    # 3. Morphological operations on segmentation array 

    # Initialize empty array for the segmentation
    segmentation = np.zeros(img_size)

    # Loop over all centerline points
    for i in range(num_points):

        coord = centerline_indices[i]  # Coordinates
        value = centerline_values[i]   # HU value

        start = coord - patch_size//2  # Start coordinates of patch
        end = start + patch_size       # End coordinates of patch

        # Extract image patch around centerline point
        patch = img[start[0] : end[0],
                    start[1] : end[1],
                    start[2] : end[2]]
        
        mask = mask3(patch, value, config)

        # Threshold image patch
        patch_segmented = np.zeros(patch.shape)
        patch_segmented[mask] = 1

        """
        # Compute adaptive threshold
        alpha = 0.5
        threshold = value - alpha * patch_gradient

        # Threshold image patch
        mask = (patch < threshold) & (patch < max_value)
        patch_segmented = np.zeros(patch.shape)
        patch_segmented[mask] = 1
        """

        """
        # Plot the gradient magnitude distribution for a specific centerline point
        if i == 100:
            patch_gradient_flat = patch_gradient.flatten()
            plt.figure(figsize=(8, 6))
            plt.hist(patch_gradient_flat, bins=50, color='b', alpha=0.7)
            plt.title(f"Gradient Magnitude Distribution (Centerline Point {i})")
            plt.xlabel("Gradient Magnitude")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()
        """

        # Add segmented voxels from patch to the segmentation array
        # Only add voxels that are segmented, without overwriting entire patch
        segmentation_patch = segmentation[start[0] : end[0],    
                                          start[1] : end[1],
                                          start[2] : end[2]]
        
        segmentation_patch[patch_segmented == 1] = 1

        segmentation[start[0] : end[0],
                     start[1] : end[1],
                     start[2] : end[2]] = segmentation_patch
        
    # Morphological closing operation (dilation followed by erosion)
    size = config.segmentation.ball_size
    segmentation = skimage.morphology.binary_closing(segmentation, footprint=skimage.morphology.ball(size))   

    # Convert from (x, y, z) back to (z, y, x)
    segmentation = segmentation.transpose(2, 1, 0)

    return segmentation

def save_segmentation(segmentation, sample, config):
    """ Save segmentation as .nii.gz file """

    # Path for saving segmentation
    file_name = f"{sample['image_index']}.img.segmentation_lad"
    time_str = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
    output_path = f"{config.base_settings.base_dir}/{config.segmentation.output_dir}/{file_name}_{time_str}.nii.gz"
    
    # Convert segmentation array to SimpleITK image
    segmentation_nii = sitk.GetImageFromArray(segmentation.astype(int))
    
    # Retrieve original image metadata
    origin, spacing, direction = sample['origin'], sample['spacing'], sample['direction']
    segmentation_nii.SetOrigin(origin)
    segmentation_nii.SetSpacing(spacing)
    segmentation_nii.SetDirection(direction)
    
    # Write the segmentation image to file
    sitk.WriteImage(segmentation_nii, output_path)

@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.3.2")
def main(config):
    # Load a sample
    index = 1
    sample = load_sample(index, config)

    # Perform LAD segmentation
    segmentation = segment_LAD(sample, config)

    # Save segmentation
    save_segmentation(segmentation, sample, config)
    
if __name__ == "__main__":
    main()