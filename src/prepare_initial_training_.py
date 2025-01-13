import os
import re
import json
import numpy as np 

def prepare_initial_training(test_img_indices, config, log):

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

    # # Remove image indices of test set from being eligible for initial training
    # all_samples_img_indices = [index for index in all_samples_img_indices if index not in test_img_indices]
    # num_samples_total = len(all_samples_img_indices)

    # log.info(f"Total number of samples in the dataset: {num_samples_total} (excluding the test set)")

    # ----------------------------------------------------------------------------------------------------------- #
    # NOTE: OLD METHOD (sampling them randomly, meaning they vary across experiments)
    # # Sample random image indices to be used for initial training
    # img_indices_initial_training = np.random.default_rng(seed = seed).choice(all_samples_img_indices, 
    #                                                                          size = num_samples_initial_training, 
    #                                                                          replace = False)     

    # NOTE: NEW METHOD (loading them from .txt file, meaning they are consistent across experiments)
    # Load the image indices of initial training samples from .txt file
    experiment_dir = f"{base_dir}/{version}"
    txt_filename = "initial_training_img_indices.txt"
    txt_path = f"{experiment_dir}/{txt_filename}"
    
    # Load the .txt file
    with open(txt_path, "r") as file:

        img_indices_initial_training = [int(line.strip()) for line in file]                                                                    
    # ----------------------------------------------------------------------------------------------------------- #

    # Image indices of samples to be moved (all except those used for initial training)
    samples_to_be_moved_img_indices = [x for x in all_samples_img_indices if x not in img_indices_initial_training]

    # Move all samples except initial training samples from imagesTr -> imagesTs, and labelsTr -> labelsTs
    for img_index in samples_to_be_moved_img_indices:

        # Get the sample
        filename = f"img{img_index}"

        os.rename(f"{images_input_dir}/{filename}_0000.nii.gz", f"{images_output_dir}/{filename}_0000.nii.gz")
        os.rename(f"{labels_input_dir}/{filename}.nii.gz", f"{labels_output_dir}/{filename}.nii.gz")

    if num_channels == 2:
        os.rename(f"{images_input_dir}/{filename}_0001.nii.gz", f"{images_output_dir}/{filename}_0001.nii.gz")
    
    log.info(f"Number of samples used for initial training: {num_samples_initial_training}")
    log.info(f"Image indices of samples used for initial training: {img_indices_initial_training}")
    log.info(f"All samples except from initial training samples have been moved from imagesTr -> imagesTs, and labelsTr -> labelsTs")

    # STEP 2: Update numTraining in dataset.json (both in nnUNet_preprocessed and nnUNet_raw) 
    # dataset_json_nnUNetpreprocessed_path = f"{base_dir}/{version}/{data_preprocessed_dir}/{dataset_name}/dataset.json"
    dataset_json_nnUNetraw_path = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/dataset.json"

    # # nnUNet_preprocessed/dataset.json
    # with open(dataset_json_nnUNetpreprocessed_path, "r+") as jsonFile:

    #     # Get the dataset.json content
    #     data = json.load(jsonFile)

    #     # Update the number of training samples
    #     data["numTraining"] = num_samples_initial_training

    #     jsonFile.seek(0)
    #     json.dump(data, jsonFile)
    #     jsonFile.truncate()

    # nnUNet_raw/dataset.json
    with open(dataset_json_nnUNetraw_path, "r+") as jsonFile:

        # Get the dataset.json content
        data = json.load(jsonFile)

        # Update the number of training samples
        data["numTraining"] = num_samples_initial_training

        jsonFile.seek(0)
        json.dump(data, jsonFile)
        jsonFile.truncate()

    log.info(f"numTraining samples has been updated to: {num_samples_initial_training} (nnUNet_raw/dataset.json)")
    log.info(f'----------------------------------------------------------------------------\n')