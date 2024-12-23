import hydra
import logging

from load_save_utilities import get_all_images_with_centerline_data

# Set up logging
log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config_disk.yaml", version_base="1.3.2")
def main(config):
    
    # Get all images with centerline data, and save to .txt file
    get_all_images_with_centerline_data(config)
    
if __name__ == "__main__":
    main()