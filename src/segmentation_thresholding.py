import hydra
import vtk
import numpy as np
import SimpleITK as sitk

from datetime import datetime

from data_loader import CoronaryArteryDataLoader

def load_sample(config, index):
    # Create an instance of the data loader
    data_loader = CoronaryArteryDataLoader(config)
    
    # Load a single scan
    sample = data_loader.__getitem__(index) # or simply: data_loader[0]

    return sample

def segment_LAD(sample):
    """ Segment the left anterior descending artery (LAD) using adaptive thresholding """

    # Load data from sample dictionary
    img = sample['image']
    label = sample['label']
    centerline = sample['centerline']
    centerline_indices = sample['centerline_indices']
    centerline_values = sample['centerline_values']
    img_index = sample['image_index']
    patch_size = sample['patch_size']

    img_size = img.shape                    # Image size
    num_points = len(centerline_indices)    # Number of centerline points
    max_value = 800                         # Maximum HU value allowed for segmentation
                                            # TODO: This should be computed based on aorta segmentation, see notes

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

        # Threshold image patch
        mask = (patch > value) & (patch < max_value)
        patch_segmented = np.zeros(patch.shape)
        patch_segmented[mask] = 1

        # Add segmented voxels from patch to the segmentation array
        segmentation[start[0] : end[0],
                     start[1] : end[1],
                     start[2] : end[2]] = patch_segmented
            
    return segmentation

@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.3.2")
def main(config):
    # Load a sample
    index = 0
    sample = load_sample(config, index)

    # Perform LAD segmentation
    segmentation = segment_LAD(sample)

    # Path for saving segmentation
    file_name = f"{sample['image_index']}.img.segmentation_lad"
    time_str = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
    output_path = f"{config.base_settings.base_dir}/test/{file_name}_{time_str}.nii.gz"
    
    # Convert segmentation array to SimpleITK image
    segmentation_nii = sitk.GetImageFromArray(segmentation.astype(int))
    
    # Retrieve original image metadata
    origin, spacing, direction = sample['origin'], sample['spacing'], sample['direction']
    segmentation_nii.SetOrigin(origin)
    segmentation_nii.SetSpacing(spacing)
    segmentation_nii.SetDirection(direction)
    
    # Write the segmentation image to file
    sitk.WriteImage(segmentation_nii, output_path)
    
if __name__ == "__main__":
    main()