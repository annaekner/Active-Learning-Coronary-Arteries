import numpy as np
import SimpleITK as sitk
import scipy.ndimage
import scipy.spatial
import skimage.morphology

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

def compute_centerline_from_prediction(prediction, prediction_nii, img_index, config):
    """
    Compute the centerline from the predicted LAD segmentation using skeletonization.
    """
    # Compute the skeleton of the prediction
    prediction_centerline = skimage.morphology.skeletonize(prediction).astype(int)

    # Get the indices of the centerline
    prediction_centerline_indices = np.argwhere(prediction_centerline)

    return prediction_centerline_indices

def compute_centerline_coverage_and_dice(path_1, path_2, config):
    """
    Compute the coverage of path 1 and path 2 and the combined path dice score.
    Paths are given as numpy arrays of the form [[z1, y1, x1], [z2, y2, x2], ...].
    """
    # Get tolerance from config
    tolerance = config.centerline_predictions.tolerance

    # Build KDTree for path_2
    tree_2 = scipy.spatial.cKDTree(path_2)

    n_points_1 = len(path_1)
    if n_points_1 < 1:
        print("No points in path_1")
        return 0, 0, 0

    # Compute coverage for path_1
    n_covered_1 = 0
    for p in path_1:
        dist, _ = tree_2.query(p)
        if dist < tolerance:
            n_covered_1 += 1

    p_covered_1 = float(n_covered_1) / float(n_points_1)

    # Build KDTree for path_1
    tree_1 = scipy.spatial.cKDTree(path_1)

    n_points_2 = len(path_2)
    if n_points_2 < 1:
        print("No points in path_2")
        return 0, 0, 0

    # Compute coverage for path_2
    n_covered_2 = 0
    for p in path_2:
        dist, _ = tree_1.query(p)
        if dist < tolerance:
            n_covered_2 += 1

    p_covered_2 = float(n_covered_2) / float(n_points_2)
    path_dice = float(n_covered_1 + n_covered_2) / float(n_points_1 + n_points_2)
    return p_covered_1, p_covered_2, path_dice


def compute_evaluation_metrics_wrtGTsegmentation(ground_truth_segmentation, prediction, prediction_nii, img_index, log, config):
    """ 
    Compute metrics that compare the ground truth LAD segmentation and the predicted LAD segmentation.
    This can only be done when the ground truth LAD segmentation is available, which is not always the case.

    Args
        ground_truth_segmentation: Ground truth LAD segmentation (binary mask, numpy array)
        prediction: Predicted LAD segmentation (binary mask, numpy array)
        prediction_nii: Predicted LAD segmentation (SimpleITK image, .nii.gz)
    """

    # Compute true positives, false positives, false negatives, and true negatives
    tp, fp, fn, tn = compute_tp_fp_fn_tn(ground_truth_segmentation, prediction)

    # DICE and IoU scores
    DICE = 2 * tp / (2 * tp + fp + fn)
    IoU = tp / (tp + fp + fn)

    # Print evaluation metrics
    log.info(f'-------------- Evaluation metrics (w.r.t GT LAD segmentation) --------------')
    log.info(f'DICE: {DICE:.4f}')
    log.info(f'IoU: {IoU:.4f}')
    log.info(f'----------------------------------------------------------------------------\n')

    evaluation_metrics = {
                          'DICE': DICE, 
                          'IoU': IoU,
                          }
    
    return evaluation_metrics

def compute_evaluation_metrics_wrtGTcenterline(ground_truth_centerline_indices, prediction_centerline_indices, prediction_segmentation, img_index, log, config):
    """ 
    Compute metrics that compare the ground truth LAD centerline and the predicted LAD segmentation.
    This can always be done, since the LAD centerline is always available.

    Args
        ground_truth_centerline_indices: Ground truth LAD centerline (numpy array of [z, y, x] index coordinates)
        prediction: Predicted LAD segmentation (binary mask, numpy array)
        prediction_nii: Predicted LAD segmentation (SimpleITK image, .nii.gz)
    """

    # Connected components of prediction
    num_connected_components, labeled_array, largest_cc_array = compute_connected_components(prediction_segmentation)

    # Get tolerance from config
    tolerance = config.centerline_predictions.tolerance

    # Check overlap with centerline
    ground_truth_centerline_indices = np.unique(ground_truth_centerline_indices, axis = 0)

    # Compute the distances from the ground truth centerline to the prediction centerline
    # p_covered_1: How well the predicted centerline covers the ground truth centerline
    # p_covered_2: How well the ground truth centerline covers the predicted centerline
    p_covered_1, p_covered_2, path_dice = compute_centerline_coverage_and_dice(ground_truth_centerline_indices, 
                                                                               prediction_centerline_indices, 
                                                                               config)
    
    # Print evaluation metrics
    log.info(f'--------------- Evaluation metrics (w.r.t GT LAD centerline) ---------------')
    log.warning(f'Number of connected components: {num_connected_components} (expected was 1)') if num_connected_components > 1 \
        else log.info(f'Number of connected components: {num_connected_components}')
    log.info(f'Number of points in predicted centerline: {len(prediction_centerline_indices)}')
    log.info(f'Number of points in ground truth centerline: {len(ground_truth_centerline_indices)}')
    log.info(f'Tolerance for centerline coverage: {tolerance} voxels')
    log.info(f'Predicted centerline coverage of the ground truth centerline: {p_covered_1:.4f}')
    log.info(f'Ground truth centerline coverage of the predicted centerline: {p_covered_2:.4f}')
    log.info(f'Combined centerline "DICE" score: {path_dice:.4f}')
    log.info(f'----------------------------------------------------------------------------\n')

    evaluation_metrics = {
                          'num_connected_components': num_connected_components}
    
    return evaluation_metrics