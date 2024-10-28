import hydra
import logging

from load_save_utilities import load_sample, load_prediction
from tools import compute_evaluation_metrics_wrtGTsegmentation, compute_evaluation_metrics_wrtGTcenterline

# Set up logging
log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.3.2")
def main(config):
    
    # TODO: This variable needs to be set!
    index = 0  # Set the index (in file_list_test.txt) of the sample to load

    # 1. Load the sample with the ground truth LAD segmentation and centerline
    sample = load_sample(index, config, subset = 'test', label_type = 'LAD')

    img_index = sample['image_index']
    ground_truth = sample['label']
    centerline_indices = sample['centerline_indices']

    # 2. Load the predicted LAD segmentation
    prediction, prediction_nii = load_prediction(img_index, config)

    assert ground_truth.shape == prediction.shape, 'Ground truth and prediction must have the same shape'

    # 3. Evaluate the prediction (w.r.t the ground truth LAD segmentation, when available)
    evaluation_metrics_segmentation = compute_evaluation_metrics_wrtGTsegmentation(ground_truth, prediction, prediction_nii, img_index, log, config)

    # 4. Evaluate the prediction (w.r.t. the ground truth LAD centerline, which is always available)
    evaluation_metrics_centerline = compute_evaluation_metrics_wrtGTcenterline(centerline_indices, prediction, prediction_nii, img_index, log, config)

    # TODO: Loop over all predictions and find the "worst" ones to use for re-training
    # ...
    
if __name__ == "__main__":
    main()