import hydra
import logging
import numpy as np

from load_save_utilities import load_sample, load_prediction

# Set up logging
log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.3.2")
def main(config):
    
    # TODO: This variable needs to be set!
    index = 5  # Set the index (in file_list.txt) of the sample to load

    # 1. Load the sample with the ground truth LAD segmentation and centerline
    sample = load_sample(index, config, subset = 'test', label_type = 'LAD')

    img_index = sample['image_index']
    ground_truth = sample['label']
    centerline_indices = sample['centerline_indices']
    print(f'Ground truth shape: {ground_truth.shape}')
    print(f'Ground truth values: {np.unique(ground_truth)}')

    # 2. Load the predicted LAD segmentation
    prediction = load_prediction(img_index, config)
    print(f'Prediction shape: {prediction.shape}')
    print(f'Prediction values: {np.unique(prediction)}')

    # 3. Evaluate the prediction
    # Compute DICE between the ground truth and the prediction
    

    # ...
    
if __name__ == "__main__":
    main()