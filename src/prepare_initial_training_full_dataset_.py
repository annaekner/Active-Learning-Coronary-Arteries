import os
import re
import json
import numpy as np 

def prepare_initial_training_full_dataset(test_img_indices, config, log):

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version
    dataset_name = config.dataset_settings.dataset_name
    num_channels = config.base_settings.num_channels
    seed = config.base_settings.seed

    data_raw_dir = config.data_raw.dir
    train_images_dir = config.data_raw.train_images_dir
    test_images_dir = config.data_raw.test_images_dir
    train_labels_dir = config.data_raw.train_labels_dir
    test_labels_dir = config.data_raw.test_labels_dir

    data_preprocessed_dir = config.data_preprocessed.dir

    num_samples_initial_training = config.train_settings.num_samples_initial_training

    log.info(f'-------------------------- Prepare initial training -------------------------')

    # STEP 1: Prepare files for initial training round
    images_input_dir = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{train_images_dir}"
    images_output_dir = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{test_images_dir}"
    labels_input_dir = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{train_labels_dir}"
    labels_output_dir = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{test_labels_dir}"

    # List of all data sample image indices image indices of all data samples
    all_samples_file_names = os.listdir(labels_input_dir)
    all_samples_img_indices = [int(re.search(r'img(\d+)\.nii\.gz', file_name).group(1)) for file_name in all_samples_file_names]
    num_samples_total = len(all_samples_img_indices)

    log.info(f"Total number of samples in the dataset: {num_samples_total} (including the test set)")
    log.info(f"Total number of samples in the dataset: {num_samples_total - len(test_img_indices)} (excluding the test set)")

    # nnUNet_raw/dataset.json
    dataset_json_nnUNetraw_path = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/dataset.json"
    with open(dataset_json_nnUNetraw_path, "r+") as jsonFile:

        # Get the dataset.json content
        data = json.load(jsonFile)

        # Update the number of training samples
        data["numTraining"] = num_samples_total - len(test_img_indices)

        jsonFile.seek(0)
        json.dump(data, jsonFile)
        jsonFile.truncate()

    log.info(f"numTraining samples has been updated to: {num_samples_total - len(test_img_indices)} (nnUNet_raw/dataset.json)")
    
    # Move all samples from the test set from imagesTr -> imagesTs, and labelsTr -> labelsTs
    for img_index in test_img_indices:

        # Get the sample
        filename = f"img{img_index}"

        os.rename(f"{images_input_dir}/{filename}_0000.nii.gz", f"{images_output_dir}/{filename}_0000.nii.gz")
        os.rename(f"{labels_input_dir}/{filename}.nii.gz", f"{labels_output_dir}/{filename}.nii.gz")

    if num_channels == 2:
        os.rename(f"{images_input_dir}/{filename}_0001.nii.gz", f"{images_output_dir}/{filename}_0001.nii.gz")
    
    log.info(f"All samples of the test set have been moved from imagesTr -> imagesTs, and labelsTr -> labelsTs")
    log.info(f'----------------------------------------------------------------------------\n')