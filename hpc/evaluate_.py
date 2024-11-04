import os
import glob

from tools import list_of_all_predictions, load_prediction_segmentation, load_ground_truth_centerline, compute_centerline_from_prediction, compute_evaluation_metrics_wrtGTcenterline

def evaluate(config, log):

    # Get image indices of all predictions
    predictions_img_indices = list_of_all_predictions(config, log)

    # Nested dictionary for storing evaluation metrics of all predictions
    evaluation_metrics_all = {}

    for i, img_index in enumerate(predictions_img_indices):

        # 1. Load prediction segmentation (as numpy array and nii.gz)
        prediction_segmentation, prediction_segmentation_nii = load_prediction_segmentation(img_index, config, log)

        # 2. Load ground truth centerline
        ground_truth_centerline_indices = load_ground_truth_centerline(img_index, prediction_segmentation_nii, config, log)

        # 3. Compute centerline from the predicted LAD segmentation
        prediction_centerline_indices = compute_centerline_from_prediction(prediction_segmentation, prediction_segmentation_nii, img_index, config)
        
        # 4. Evaluate the prediction (w.r.t. the ground truth LAD centerline, which is always available)
        evaluation_metrics_centerline = compute_evaluation_metrics_wrtGTcenterline(ground_truth_centerline_indices, prediction_centerline_indices, prediction_segmentation, img_index, log, config)

        # 5. Store evaluation metrics in nested dictionary
        evaluation_metrics_all[img_index] = evaluation_metrics_centerline

    return evaluation_metrics_all