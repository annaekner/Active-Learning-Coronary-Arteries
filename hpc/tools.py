import re
import vtk
import glob
import scipy
import skimage
import numpy as np
import SimpleITK as sitk

def list_of_all_predictions(config, log, iteration):
    """ 
    Create a list with image indices of all predictions found
    """

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version
    dataset_name = config.dataset_settings.dataset_name
    data_predicted_dir = config.data_predicted.dir

    # Find predictions in folder
    predictions_folder_path = f"{base_dir}/{version}/{data_predicted_dir}/{dataset_name}/"
    all_predictions_filepaths = glob.glob(f'{predictions_folder_path}/*.nii.gz')

    # Get image indices of predictions
    predictions_img_indices = sorted([int(re.search(r'img(\d+)\.nii\.gz', path).group(1)) for path in all_predictions_filepaths])

    return predictions_img_indices

def load_prediction_segmentation(img_index, config, log, iteration):
    """ 
    Load the prediction (LAD segmentation) as both a numpy array and a nii.gz
    """

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version
    dataset_name = config.dataset_settings.dataset_name
    data_predicted_dir = config.data_predicted.dir      # Predicted LAD segmentations

    # Load the prediction
    prediction_path = f"{base_dir}/{version}/{data_predicted_dir}/{dataset_name}/img{img_index}.nii.gz"

    # Convert from .nii.gz file to numpy array
    prediction_nii = sitk.ReadImage(prediction_path)
    prediction = sitk.GetArrayFromImage(prediction_nii)
    prediction = prediction.astype(np.bool)

    return prediction, prediction_nii

def load_ground_truth_segmentation(img_index, config, log, iteration):
    """ 
    Load the ground truth (LAD segmentation) as both a numpy array and a nii.gz
    """

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version
    dataset_name = config.dataset_settings.dataset_name

    data_raw_dir = config.data_raw.dir                  
    test_labels_dir = config.data_raw.test_labels_dir

    # Load the ground truth 
    ground_truth_path = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{test_labels_dir}/img{img_index}.nii.gz"

    # Convert from .nii.gz file to numpy array
    ground_truth_nii = sitk.ReadImage(ground_truth_path)
    ground_truth = sitk.GetArrayFromImage(ground_truth_nii)
    ground_truth = ground_truth.astype(np.bool)

    return ground_truth, ground_truth_nii

def load_ground_truth_centerline(img_index, prediction_segmentation_nii, config, log):
    """
    Load the ground truth centerline (of the LAD)
    """

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version
    dataset_name = config.dataset_settings.dataset_name
    data_centerlines_dir = config.data_centerlines.dir  # Ground truth LAD centerlines

    # Load ground truth centerline (of left anterior descending artery)
    centerline_path = f"{base_dir}/{version}/{data_centerlines_dir}/{dataset_name}/img{img_index}_lad_centerline.vtk"
    centerline_reader = vtk.vtkPolyDataReader()
    centerline_reader.SetFileName(centerline_path)
    centerline_reader.Update()
    centerline = centerline_reader.GetOutput()

    # Extract points (physical coordinates) from the centerline
    points = centerline.GetPoints()
    num_points = points.GetNumberOfPoints()
    centerline_points = np.array([points.GetPoint(i) for i in range(num_points)])

    # Transform centerline points from physical coordinates to index coordinates
    centerline_indices = np.array([prediction_segmentation_nii.TransformPhysicalPointToIndex(point) for point in centerline_points])
    centerline_indices = centerline_indices[:, ::-1] # Convert from (x, y, z) to (z, y, x), i.e. [[z1, y1, x1], [z2, y2, x2], ...]

    return centerline_indices

def compute_centerline_from_prediction(prediction, prediction_nii, img_index, config):
    """
    Compute the centerline from the predicted LAD segmentation using skeletonization.
    """
    # Compute the skeleton of the prediction
    prediction_centerline = skimage.morphology.skeletonize(prediction).astype(int)

    # Get the indices of the centerline
    prediction_centerline_indices = np.argwhere(prediction_centerline)

    return prediction_centerline_indices

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

def compute_connected_components(segmentation, config):
    """
    Compute the connected components of given segmentation (e.g. prediction).

    Returns
        num_connected_components: Number of connected components in the segmentation.

        labeled_array: Array of same shape as the segmentation, where each voxel has a 
                       value corresponding to which connected component it belongs to,
                       either 0 = background, 1 = first connected component, 2 = second 
                       connected component, and so on. E.g. if there is only one connected 
                       component, background voxels have value 0 and connected component voxels have value 1.

        largest_cc_array: Array of same shape as the segmentation, where each voxel has a 
                          value of 1 if it belongs to the largest connected component, and 0 otherwise.
    """

    # Configuration settings
    min_size = config.connected_components.min_size

    # Compute connected components
    labeled_array, num_connected_components = scipy.ndimage.label(segmentation)
    components_sizes = np.unique(labeled_array, return_counts=True)[1] # Voxels per connected component

    # Find largest connected component
    largest_cc_index = np.argmax(components_sizes)
    largest_cc_array = (labeled_array == largest_cc_index).astype(int)

    # Remove small connected components
    if len(components_sizes) > 2:

        # Loop over connected components (excluding background, which has label 0)
        for i in range(1, num_connected_components + 1):

            # Remove connected components smaller than min_size
            if components_sizes[i] < min_size:
                labeled_array[labeled_array == i] = 0 
                num_connected_components -= 1

    return num_connected_components, labeled_array, largest_cc_array

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

def compute_entropy(img_index, config, log):
    """
    Compute uncertainty (entropy) of a given sample from the prediction probabilities.

    Low entropy means low uncertainty (the prediction is confident)
    High entropy, i.e. close to 0.5 for binary segmentation, means high uncertainty (the prediction is not confident)
    """

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version
    dataset_name = config.dataset_settings.dataset_name
    data_predicted_dir = config.data_predicted.dir

    # Load the probabilities
    npz_path = f"{base_dir}/{version}/{data_predicted_dir}/{dataset_name}/img{img_index}.npz"
    npz_file = np.load(npz_path)
    probabilities = npz_file['probabilities'] # Shape (2, z, y, x)

    # Scipy computation
    # probabilities = probabilities.transpose(1, 2, 3, 0)  # Shape (z, y, x, 2)
    # voxelwise_entropy = scipy.stats.entropy(probabilities, axis=-1)  # Compute entropy along the last axis (class axis) 

    # Manual computation 
    # voxelwise_entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-10, None)), axis = 0)
    voxelwise_entropy = -np.sum(probabilities * np.log(probabilities + 1e-20), axis=0) / np.log(2)

    # Compute image-level entropy (mean of voxel-wise entropies)
    entropy = np.mean(voxelwise_entropy)

    return float(entropy)

def compute_evaluation_metrics_wrtGTcenterline(ground_truth_centerline_indices, prediction_centerline_indices, prediction_segmentation, img_index, log, config):
    """ 
    Compute metrics that compare the ground truth LAD centerline and the predicted LAD segmentation.
    Since the LAD centerline is always available, this can be done for any data sample.

    Args
        ground_truth_centerline_indices: Ground truth LAD centerline (numpy array of [z, y, x] index coordinates)
        prediction_centerline_indices: Predicted LAD centerline (numpy array of [z, y, x] index coordinates)
        prediction_segmentation: Predicted LAD segmentation (binary mask, numpy array)
    """

    # Connected components of prediction
    num_connected_components, labeled_array, largest_cc_array = compute_connected_components(prediction_segmentation, config)

    # Configuration settings
    tolerance = config.centerline_predictions.tolerance
    min_size = config.connected_components.min_size

    # Get unique indices of the ground truth centerline
    ground_truth_centerline_indices = np.unique(ground_truth_centerline_indices, axis = 0)

    # Compute the distances from the ground truth centerline to the prediction centerline
    # p_covered_1: How well the predicted centerline covers the ground truth centerline
    # p_covered_2: How well the ground truth centerline covers the predicted centerline
    p_covered_1, p_covered_2, path_dice = compute_centerline_coverage_and_dice(ground_truth_centerline_indices, 
                                                                               prediction_centerline_indices, 
                                                                               config)
    
    # Convert number of connected components (num_cc) to weight (w): 
    # w = 1 - (num_cc * 0.1)
    weight = 1 - (num_connected_components * 0.1)

    # Multiply the combined DICE score with the weight
    # Samples with only one connected component will have high weighted scores, 
    # samples with multiple connected components will have low weighted scores, 
    weighted_score = weight * path_dice

    # Compute uncertainty (entropy)
    entropy = compute_entropy(img_index, config, log)
    
    # Log evaluation metrics
    log.info(f'--------------- Evaluation metrics (w.r.t GT LAD centerline) ---------------')
    log.info(f'Image index: {img_index}')
    log.info(f'Number of connected components: {num_connected_components}')
    log.info(f'Minimum size for connected components: {min_size} voxels')
    log.info(f'Number of points in predicted centerline: {len(prediction_centerline_indices)}')
    log.info(f'Number of points in ground truth centerline: {len(ground_truth_centerline_indices)}')
    log.info(f'Tolerance for centerline coverage: {tolerance} voxels')
    log.info(f'Predicted centerline coverage of the ground truth centerline: {p_covered_1:.4f}')
    log.info(f'Ground truth centerline coverage of the predicted centerline: {p_covered_2:.4f}')
    log.info(f'Combined centerline "DICE" score: {path_dice:.4f}')
    log.info(f'Weighted centerline "DICE" score: {weighted_score:.4f}')
    log.info(f'Entropy (uncertainty): {entropy:.8f}')
    log.info(f'----------------------------------------------------------------------------\n')

    evaluation_metrics = {
                          'num_connected_components': num_connected_components,
                          'predicted_centerline_coverage_of_ground_truth_centerline': round(p_covered_1, 4),
                          'ground_truth_centerline_coverage_of_predicted_centerline': round(p_covered_2, 4),
                          'combined_centerline_dice_score': round(path_dice, 4),
                          'weighted_centerline_dice_score': round(weighted_score, 4),
                          'entropy': round(entropy, 8),
                          }
    
    return evaluation_metrics

def compute_evaluation_metrics_wrtGTsegmentation(ground_truth_segmentation, prediction_segmentation, img_index, log, config):
    """ 
    Compute metrics that compare the ground truth LAD segmentation and the predicted LAD segmentation.
    This can only be done when the ground truth LAD segmentation is available, i.e. for the test set.

    Args
        ground_truth_segmentation: Ground truth LAD segmentation (binary mask, numpy array)
        prediction_segmentation: Predicted LAD segmentation (binary mask, numpy array)
    """

    # Compute true positives, false positives, false negatives, and true negatives
    tp, fp, fn, tn = compute_tp_fp_fn_tn(ground_truth_segmentation, prediction_segmentation)

    # DICE and IoU scores
    DICE = 2 * tp / (2 * tp + fp + fn)
    IoU = tp / (tp + fp + fn)

    # Hausdorff distance
    hausdorff_distance = skimage.metrics.hausdorff_distance(ground_truth_segmentation, prediction_segmentation)

    # Print evaluation metrics
    log.info(f'-------------- Evaluation metrics (w.r.t GT LAD segmentation) --------------')
    log.info(f'Image index: {img_index}')
    log.info(f'DICE: {DICE:.4f}')
    log.info(f'IoU: {IoU:.4f}')
    log.info(f'Hausdorff distance: {hausdorff_distance:.4f}')
    log.info(f'----------------------------------------------------------------------------\n')

    evaluation_metrics = {
                          'DICE': round(DICE, 4), 
                          'IoU': round(IoU, 4),
                          'Hausdorff_distance': round(hausdorff_distance),
                          }
    
    return evaluation_metrics

def compute_mean_of_evaluation_metrics(evaluation_metrics_test, config, log):
    """ 
    Compute the mean of the evaluation metrics for all samples in the test set
    """

    # Image indices of all test samples (as strings)
    test_img_indices_str = list(evaluation_metrics_test.keys())[1:]

    # Evaluation metrics keys from the dictionary
    evaluation_metrics_keys = list(evaluation_metrics_test[test_img_indices_str[0]].keys())

    # Prepare dictionaries
    evaluation_metrics_list = {key: [] for key in evaluation_metrics_keys}
    evaluation_metrics_mean = {f"{key}_mean": [] for key in evaluation_metrics_keys}

    for img_index in test_img_indices_str:

        # Get evaluation metrics dictionary of the image
        evaluation_metrics_img = evaluation_metrics_test[img_index]

        for key in evaluation_metrics_keys:

            # Append the metric value to the corresponding list in evaluation_metrics_list
            evaluation_metrics_list[key].append(evaluation_metrics_img[key])
    
    # Compute the mean of each evaluation metric
    for key in evaluation_metrics_keys:

        # All values of each metric
        all_values = evaluation_metrics_list[key]

        # Mean of all values
        mean_value = float(np.mean(all_values))

        # Append the mean metric value to evaluation_metrics_mean
        evaluation_metrics_mean[f"{key}_mean"] = round(mean_value, 4)

    return evaluation_metrics_list, evaluation_metrics_mean