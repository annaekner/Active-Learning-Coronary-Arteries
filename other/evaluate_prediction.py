import hydra
import logging

from load_save_utilities import load_sample, load_prediction, save_prediction_centerline_to_vtk
from tools import compute_centerline_from_prediction, compute_evaluation_metrics_wrtGTsegmentation, compute_evaluation_metrics_wrtGTcenterline, select_samples_for_retraining

# Set up logging
log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.3.2")
def main(config):
    
    # TODO: These variable need to be set!
    index = 5           # Set the index (in file_list_test.txt) of the sample to load
    all_indices = True  # Set to True to evaluate all predictions, or False to evaluate a single prediction
    selection_method = 'worst'  # Method for selecting sample for retraining ('worst', 'best', 'random')

    # --------------------- Evaluate a single prediction --------------------- #
    if not all_indices:
        # 1. Load the sample with the ground truth LAD segmentation and centerline
        sample = load_sample(index, config, subset = 'test', label_type = 'LAD')

        img_index = sample['image_index']
        ground_truth_segmentation = sample['label']
        ground_truth_centerline_indices = sample['centerline_indices']

        # 2. Load the predicted LAD segmentation
        prediction_segmentation, prediction_segmentation_nii = load_prediction(img_index, config)

        assert ground_truth_segmentation.shape == prediction_segmentation.shape, 'Ground truth and prediction must have the same shape'

        # 3. Compute and save centerline from the predicted LAD segmentation
        prediction_centerline_indices = compute_centerline_from_prediction(prediction_segmentation, config)
        save_prediction_centerline_to_vtk(prediction_centerline_indices, prediction_segmentation_nii, img_index, config)

        # 4. Evaluate the prediction (w.r.t the ground truth LAD segmentation, when available)
        evaluation_metrics_segmentation = compute_evaluation_metrics_wrtGTsegmentation(ground_truth_segmentation, prediction_segmentation, prediction_segmentation_nii, img_index, log, config)

        # 5. Evaluate the prediction (w.r.t. the ground truth LAD centerline, which is always available)
        evaluation_metrics_centerline = compute_evaluation_metrics_wrtGTcenterline(ground_truth_centerline_indices, prediction_centerline_indices, prediction_segmentation, img_index, log, config)

    # --------------------- Evaluate all predictions --------------------- #
    if all_indices:
        # Set logging level to WARNING
        log.setLevel(logging.WARNING)

        # Configuration settings
        base_dir = config.base_settings.base_dir
        file_list_dir = config.data_loader.file_list_dir
        file_list_test = config.data_loader.file_list_test

        # Get number of test samples from file list
        with open(f'{base_dir}/{file_list_dir}/{file_list_test}', 'r') as f:
            num_samples = len(f.readlines())

        # Nested dictionary for storing evaluation metrics of all predictions
        evaluation_metrics_all = {}

        # Loop over all predictions
        for index in range(num_samples):

            # 1. Load the sample with the ground truth LAD segmentation and centerline
            sample = load_sample(index, config, subset = 'test', label_type = 'LAD')

            img_index = sample['image_index']
            ground_truth_centerline_indices = sample['centerline_indices']

            # 2. Load the predicted LAD segmentation
            prediction_segmentation, prediction_segmentation_nii = load_prediction(img_index, config)

            # 3. Compute centerline from the predicted LAD segmentation
            prediction_centerline_indices = compute_centerline_from_prediction(prediction_segmentation, config)

            # 4. Evaluate the prediction (w.r.t. the ground truth LAD centerline, which is always available)
            evaluation_metrics_centerline = compute_evaluation_metrics_wrtGTcenterline(ground_truth_centerline_indices, prediction_centerline_indices, prediction_segmentation, img_index, log, config)
    
            # 5. Store evaluation metrics in nested dictionary
            evaluation_metrics_all[img_index] = evaluation_metrics_centerline

        # Set logging level back to INFO
        log.setLevel(logging.INFO)

        # Select sample for retraining
        retraining = select_samples_for_retraining(evaluation_metrics_all, selection_method, log, config)

if __name__ == "__main__":
    main()