import os
import glob
import json

def move_files(retraining, config, log):

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version
    iteration = config.base_settings.iteration
    dataset_name = config.dataset_settings.dataset_name

    data_raw_dir = config.data_raw.dir
    train_images_dir = config.data_raw.train_images_dir
    test_images_dir = config.data_raw.test_images_dir
    train_labels_dir = config.data_raw.train_labels_dir
    test_labels_dir = config.data_raw.test_labels_dir

    # Input and output directory for images and labels
    images_input_dir = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{test_images_dir}"
    images_output_dir = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{train_images_dir}"
    labels_input_dir = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{test_labels_dir}"
    labels_output_dir = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{train_labels_dir}"

    dataset_json_path = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/dataset.json"
    config_yaml_path = f"../conf/config.yaml"

    # Step 1: Get image indices of retraining samples
    samples_for_retraining = retraining["samples_for_retraining"]
    num_samples_for_retraining = len(samples_for_retraining)

    # Step 2: Update dataset.json file with new number of training samples
    with open(dataset_json_path, "r+") as jsonFile:
        data = json.load(jsonFile)

        # Get the current number of training samples
        num_current_training_samples = data["numTraining"]

        # Add the number of retraining samples to the total amount
        data["numTraining"] = num_current_training_samples + num_samples_for_retraining

        jsonFile.seek(0)
        json.dump(data, jsonFile)
        jsonFile.truncate()

    for img_index in samples_for_retraining:

        # Filename
        filename = f"img{img_index}.nii.gz"

        # Step 3: Move from imagesTs --> imagesTr
        os.rename(f"{images_input_dir}/{filename}", f"{images_output_dir}/{filename}")

        # Step 4: Move from labelsTs --> labelsTr
        os.rename(f"{labels_input_dir}/{filename}", f"{labels_output_dir}/{filename}")

    # Step 6: Update config.yaml file by incrementing the iteration number
    # ...

    # Step 7: Create a new folder in data_predicted_dir for next iteration
    # ...

    # NOTE: Better idea than emptying the nnUNet_predicted folder is to have subfolders with iteration_0, iteration_1, and so on. 
    # Then I need to update the predict function to output into the corresponding folder. 
    # And then in this file, I need to:
    # -- Update config.yaml with +1 in the base_settings.iteration 
    # -- Create a new folder for the next iteration?

    # NOTE: 
    # nnUNet_raw: move from imagesTs --> imagesTr and labelsTs --> labelsTr, update dataset.json
    # nnUNet_preprocessed: will just get overwritten automatically
    # nnUNet_results: don't know if I should empty its contents before retraining, or if it will be overwritten
    # nnUNet_predictions: need to empty its contents before retraining

    log.info(f'------------------------------- Moving files -------------------------------')
    log.info(f'Number of training samples changed from {num_current_training_samples} to {num_current_training_samples + num_samples_for_retraining}')
    log.info(f'----------------------------------------------------------------------------\n')