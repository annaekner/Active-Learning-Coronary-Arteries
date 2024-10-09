import hydra
import logging

from data_loader import CoronaryArteryDataLoader
from segment_lad import extract_lad_from_full_tree
from load_save_utilities import load_sample, save_segmentation, save_resampled_img

# Set up logging
log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.3.2")
def main(config):
    
    # TODO: These two variables need to be set!
    index = 1           # Set the index (in the file list) of the sample to load
    subset = 'train'    # Set subset to either 'train' or 'test'

    # Load the sample
    sample = load_sample(index, config)
    img_index = sample['image_index']

    # Save the resampled image
    save_resampled_img(img_index, config, subset)

    # Extract LAD segmentation from full coronary tree
    lad_segmentation = extract_lad_from_full_tree(sample, config)

    # Save LAD segmentation
    save_segmentation(lad_segmentation, sample, config, subset)
    
if __name__ == "__main__":
    main()