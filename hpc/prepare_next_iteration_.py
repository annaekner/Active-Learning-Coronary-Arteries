import os
import re
import glob
import json
import yaml
import shutil

def prepare_next_iteration(retraining, config, log, iteration):

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version
    dataset_name = config.dataset_settings.dataset_name
    num_channels = config.base_settings.num_channels

    home_dir = config.home_settings.home_dir
    special_course_dir = config.home_settings.special_course_dir
    config_dir = config.home_settings.config_dir

    data_raw_dir = config.data_raw.dir
    train_images_dir = config.data_raw.train_images_dir
    test_images_dir = config.data_raw.test_images_dir
    train_labels_dir = config.data_raw.train_labels_dir
    test_labels_dir = config.data_raw.test_labels_dir

    data_predicted_dir = config.data_predicted.dir
    data_results_dir = config.data_results.dir

    data_info_dir = config.data_info.dir

    network_configuration = config.train_settings.network_configuration
    trainer = config.train_settings.trainer
    fold = config.train_settings.fold

    # Input and output directory for images and labelsx
    images_input_dir = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{test_images_dir}"
    images_output_dir = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{train_images_dir}"
    labels_input_dir = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{test_labels_dir}"
    labels_output_dir = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{train_labels_dir}"

    # Paths to .json and .yaml files 
    config_yaml_path = f"{home_dir}/{special_course_dir}/{config_dir}"
    dataset_json_nnUNetpreprocessed_path = f"{base_dir}/{version}/{data_preprocessed_dir}/{dataset_name}/dataset.json"
    dataset_json_nnUNetraw_path = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/dataset.json"

    # Path to nnUNet_results subfolder that needs to be emptied
    subfolder_name = f"{trainer}__nnUNetPlans__{network_configuration}" 
    subfolder_path = f"{base_dir}/{version}/{data_results_dir}/{dataset_name}/{subfolder_name}"

    # Step 1: Get image indices of retraining samples
    samples_for_retraining = retraining["samples_for_retraining"]
    num_samples_for_retraining = len(samples_for_retraining)

    # Step 2:  numTraining in dataset.json (both in nnUNet_preprocessed and nnUNet_raw) 
    # nnUNet_preprocessed/dataset.json
    with open(dataset_json_nnUNetpreprocessed_path, "r+") as jsonFile:

        # Get the dataset.json content
        data = json.load(jsonFile)

        # Get the current number of training samples
        num_current_training_samples = data["numTraining"]

        # Add the number of retraining samples to the total amount
        data["numTraining"] = num_current_training_samples + num_samples_for_retraining

        jsonFile.seek(0)
        json.dump(data, jsonFile)
        jsonFile.truncate()

    # nnUNet_raw/dataset.json
    with open(dataset_json_raw_path, "r+") as jsonFile:

        # Get the dataset.json content
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
        filename = f"img{img_index}"

        # Step 3: Move from imagesTs --> imagesTr
        os.rename(f"{images_input_dir}/{filename}_0000.nii.gz", f"{images_output_dir}/{filename}_0000.nii.gz")

        if num_channels == 2:
            os.rename(f"{images_input_dir}/{filename}_0001.nii.gz", f"{images_output_dir}/{filename}_0001.nii.gz")

        # # Step 4: Move from labelsTs --> labelsTr
        os.rename(f"{labels_input_dir}/{filename}.nii.gz", f"{labels_output_dir}/{filename}.nii.gz")

    # Step 5: Increment iteration number
    current_iteration = iteration
    next_iteration = current_iteration + 1

    # Step 6: Move training info files (progress.png and training_log) 
    subfolder_files = glob.glob(f"{subfolder_path}/fold_{fold}/*")
    progress_png_filename = next((re.search(r'[^/]+$', file).group() for file in subfolder_files if re.search(r'progress\.png$', file)), None)
    training_log_filename = next((re.search(r'[^/]+$', file).group() for file in subfolder_files if re.search(r'training_log.*$', file.split('/')[-1])), None)

    # Path to info folder where training info files need to be moved to
    info_output_path = f"{base_dir}/{version}/{data_info_dir}/{dataset_name}/iteration_{current_iteration}"

    os.rename(f"{subfolder_path}/fold_{fold}/{progress_png_filename}", f"{info_output_path}/{progress_png_filename}")
    os.rename(f"{subfolder_path}/fold_{fold}/{training_log_filename}", f"{info_output_path}/{training_log_filename}")

    # Step 7: Remove the nnUNet_results subfolder
    try: 
        shutil.rmtree(subfolder_path)
    except OSError as e:
        log.error(f'Error trying to remove directory: {e.filename}, {e.strerror}')
    
    # # Step 8: Create a new folder in data_info_dir for next iteration
    # new_info_iteration_dir = f"{base_dir}/{version}/{data_info_dir}/{dataset_name}/iteration_{next_iteration}"
    # os.mkdir(new_info_iteration_dir)

    # Step 9: Create a new folder in data_predicted_dir for next iteration
    new_predicted_iteration_dir = f"{base_dir}/{version}/{data_predicted_dir}/{dataset_name}/iteration_{next_iteration}"
    os.mkdir(new_predicted_iteration_dir)

    log.info(f'------------------------------- Moving files -------------------------------')
    log.info(f'Number of training samples changed from {num_current_training_samples} to {num_current_training_samples + num_samples_for_retraining} (dataset.json)')
    log.info(f'Re-training samples moved from imagesTs -> imagesTr, and labelsTs -> labelsTr')
    log.info(f'Iteration number changed from {current_iteration} to {next_iteration} (config.yaml)')
    log.info(f'Files "{progress_png_filename}" and "{training_log_filename}" have been moved to "{data_info_dir}/{dataset_name}/iteration_{current_iteration}"')
    log.info(f'Subfolder "{data_results_dir}/{dataset_name}/{subfolder_name}" has been removed')
    log.info(f'New folder "{data_predicted_dir}/iteration_{next_iteration}" created for the next iteration')
    log.info(f'New folder "{data_info_dir}/iteration_{next_iteration}" created for the next iteration')
    log.info(f'----------------------------------------------------------------------------\n')