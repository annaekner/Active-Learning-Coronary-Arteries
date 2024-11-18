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
    # iteration = config.base_settings.iteration
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
    # iteration = config.base_settings.iteration
    dataset_name = config.dataset_settings.dataset_name
    data_predicted_dir = config.data_predicted.dir      # Predicted LAD segmentations

    # Load the prediction
    prediction_path = f"{base_dir}/{version}/{data_predicted_dir}/{dataset_name}/img{img_index}.nii.gz"

    # Convert from .nii.gz file to numpy array
    prediction_nii = sitk.ReadImage(prediction_path)
    prediction = sitk.GetArrayFromImage(prediction_nii)
    prediction = prediction.astype(np.bool)

    return prediction, prediction_nii

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
    log.info(f'----------------------------------------------------------------------------\n')

    evaluation_metrics = {
                          'num_connected_components': num_connected_components,
                          'predicted_centerline_coverage_of_ground_truth_centerline': round(p_covered_1, 4),
                          'ground_truth_centerline_coverage_of_predicted_centerline': round(p_covered_2, 4),
                          'combined_centerline_dice_score': round(path_dice, 4),
                          }
    
    return evaluation_metrics