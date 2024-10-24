import numpy as np
import SimpleITK as sitk
import scipy.ndimage

def spatial_resample_scan(image, desired_spacing):
    """
    Resample scan to isotropic voxel spacing
    """
    current_n_vox = image.GetWidth()
    current_spacing = image.GetSpacing()
    new_n_vox_in_slice: int = int(current_n_vox * current_spacing[0] / desired_spacing)

    # voxel size in the direction of the patient
    depth_spacing = current_spacing[2]
    n_vox_depth = image.GetDepth()
    new_n_vox_depth = int(n_vox_depth * depth_spacing / desired_spacing)

    new_volume_size = [new_n_vox_in_slice, new_n_vox_in_slice, new_n_vox_depth]

    # Create new image with desired properties
    new_image = sitk.Image(new_volume_size, image.GetPixelIDValue())
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetSpacing([desired_spacing, desired_spacing, desired_spacing])
    new_image.SetDirection(image.GetDirection())

    # Make translation with no offset, since sitk.Resample needs this arg.
    translation = sitk.TranslationTransform(3)
    translation.SetOffset((0, 0, 0))

    interpolator = sitk.sitkLinear

    # Create final resampled image
    resampled_image = sitk.Resample(image, new_image, translation, interpolator)

    return resampled_image

def compute_centerline_distance_map(sample):
    """ 
    Compute the Euclidean distance transform of the centerline (i.e. the distance of each voxel to 
    the centerline), and save it as a .nii.gz file with the same dimensions as the original image.
    """
    # Load data from sample dictionary
    image = sample['image']
    centerline_indices = sample['centerline_indices']

    # Initialize an empty binary mask of the same size as the image
    distance_map = np.zeros_like(image, dtype=np.bool)

    # Set the centerline voxels to 1 in the binary mask
    for idx in centerline_indices:
        distance_map[tuple(idx)] = 1
    
    # Compute the distance transform
    distance_map = scipy.ndimage.distance_transform_edt(~distance_map)

    # Convert to float for better precision
    distance_map = distance_map.astype(np.float32)

    # Convert from (x, y, z) back to (z, y, x)
    # distance_map = distance_map.transpose(2, 1, 0)

    return distance_map

def compute_connected_components(segmentation):
    """
    Compute the connected components of given segmentation (e.g. prediction).

    Returns
        num_connected_components: Number of connected components in the segmentation.

        labeled_array: Array of same shape as the segmentation, where each voxel has a 
                       value corresponding to which connected component it belongs to,
                       either 0 = background, 1 = first connected component, 2 = second 
                       connected component, and so on. I.e. if there is only one connected 
                       component, background voxels have value 0 and connected component voxels have value 1.

        largest_cc_array: Array of same shape as the segmentation, where each voxel has a 
                          value of 1 if it belongs to the largest connected component, and 0 otherwise.
    """
    labeled_array, num_connected_components = scipy.ndimage.label(segmentation)
    largest_cc_array = (labeled_array == 1).astype(int)

    # TODO: Only consider connected components over a certain size, see Rasmus' code

    return num_connected_components, labeled_array, largest_cc_array

def compute_tp_fp_fn_tn(ground_truth, prediction):
    """ 
    Compute the number of true positives (TP), false positives (FP), 
    false negatives (FN), and true negatives (TN) between the ground truth and the prediction of the LAD.
    """
    tp = np.sum(ground_truth & prediction)   # True positive (overlap between ground truth and prediction)
    fp = np.sum(~ground_truth & prediction)  # False positive (prediction where there is no LAD)
    fn = np.sum(ground_truth & ~prediction)  # False negative (no prediction where there is LAD)
    tn = np.sum(~ground_truth & ~prediction) # True negative (no prediction where there is no LAD)

    return tp, fp, fn, tn

def compute_evaluation_metrics(ground_truth, prediction, log):
    """ 
    Compute metrics that compare the ground truth and the prediction.
    """

    # Compute true positives, false positives, false negatives, and true negatives
    tp, fp, fn, tn = compute_tp_fp_fn_tn(ground_truth, prediction)

    # DICE and IoU scores
    DICE = 2 * tp / (2 * tp + fp + fn)
    IoU = tp / (tp + fp + fn)
    
    # Connected components of prediction
    num_connected_components, labeled_array, largest_cc_array = compute_connected_components(prediction)

    # Check overlap with centerline

    # TODO: Log warning if more than one connected component
    # TODO: Log warning if low dice score
    # TODO: Log warning if low overlap with centerline

    # Print evaluation metrics
    log.info(f'--------------------------- Evaluation metrics -----------------------------')
    log.info(f'DICE: {DICE:.4f}')
    log.info(f'IoU: {IoU:.4f}')
    log.warning(f'Number of connected components: {num_connected_components}') if num_connected_components > 1 \
        else log.info(f'Number of connected components: {num_connected_components}')
    log.info(f'----------------------------------------------------------------------------')

    evaluation_metrics = {
                          'DICE': DICE, 
                          'IoU': IoU, 
                          'num_connected_components': num_connected_components}
    
    return evaluation_metrics

