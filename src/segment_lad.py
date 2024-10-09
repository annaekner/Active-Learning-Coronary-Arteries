import logging
import numpy as np

from scipy.ndimage import binary_dilation
from load_save_utilities import save_segmentation

# Set up logging
log = logging.getLogger(__name__)

def extract_lad_from_full_tree(sample, config):
    """ 
    Extract the left anterior descending artery (LAD) from the full coronary tree
    """

    # Load data from sample dictionary
    label = sample['label']
    centerline_indices = sample['centerline_indices']

    # Configuration settings
    dilation_radius = config.segmentation.dilation_radius

    # Initialize an empty binary mask for the LAD
    lad_mask = np.zeros_like(label, dtype=np.bool)

    # Mark the centerline points in the binary mask 
    for idx in centerline_indices:
        lad_mask[tuple(idx)] = 1 
    
     # Dilate the centerline to approximate the volume of the LAD
    lad_dilated = binary_dilation(lad_mask, structure=np.ones((dilation_radius, dilation_radius, dilation_radius)))

    # Mask the full coronary artery tree segmentation with the dilated LAD mask
    # Element-wise multiplication between the two binary masks
    lad_segmentation = label * lad_dilated

    # Convert from (x, y, z) back to (z, y, x)
    lad_segmentation = lad_segmentation.transpose(2, 1, 0)
    
    return lad_segmentation