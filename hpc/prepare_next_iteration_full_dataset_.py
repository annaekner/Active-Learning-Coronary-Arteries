import os
import re
import glob
import json
import shutil
import numpy as np

import tools

def prepare_next_iteration_full_dataset(test_img_indices, config, log, iteration):

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
    data_predicted_dir = config.data_predicted.dir
    data_results_dir = config.data_results.dir

    data_iterations_dir = config.data_iterations.dir
    iterations_results_dir = config.data_iterations.results_dir
    iterations_predictions_dir = config.data_iterations.predictions_dir

    network_configuration = config.train_settings.network_configuration
    trainer = config.train_settings.trainer
    fold = config.train_settings.fold

    num_samples_test = config.test_settings.num_samples_test

    log.info(f'-------------------------- Prepare next iteration --------------------------')
    # -------------------- STEP 3: Move files from nnUNet_predictions to /iterations/ folder -------------------- #
    # Path to the nnUNet_predictions folder
    nnUNet_predictions_folder = f"{base_dir}/{version}/{data_predicted_dir}/{dataset_name}"

    # Path to /iterations/predictions/ folder where the predictions on the test set need to be moved to
    output_path = f"{base_dir}/{version}/{data_iterations_dir}/iteration_{iteration}/{iterations_predictions_dir}"

    # Only select the three first test set samples
    selected_test_img_indices = test_img_indices[:3]

    for img_index in selected_test_img_indices:

        # Move from nnUNet_results -> /iterations/predictions/
        os.rename(f"{nnUNet_predictions_folder}/img{img_index}.nii.gz", f"{output_path}/img{img_index}.nii.gz")

    log.info(f'Predictions of images {selected_test_img_indices} from the test set have been moved to "~/iteration_{iteration}/{iterations_predictions_dir}"')

    # ------------------------------ STEP 4: Empty the nnUNet_predictions subfolder ----------------------------- #
    # Remaining files in nnUNet_predictions folder
    remaining_files = glob.glob(f'{nnUNet_predictions_folder}/*')

    for file in remaining_files:

        # Remove file
        os.remove(file)

    log.info(f'Folder "{data_predicted_dir}/{dataset_name}" has been emptied')

    # ---------------------- STEP 5: Move files from nnUNet_results to /iterations/ folder ---------------------- #
    # Path to the nnUNet_results subfolder
    trainer_name = f"{trainer}__nnUNetPlans__{network_configuration}" 
    nnUNet_results_subfolder = f"{base_dir}/{version}/{data_results_dir}/{dataset_name}/{trainer_name}"

    # Filenames of the files to be moved
    subfolder_files = [os.path.basename(x) for x in glob.glob(f"{nnUNet_results_subfolder}/fold_{fold}/*")]

    checkpoint_best_filename = next((filename for filename in subfolder_files if re.compile(r'.*_best\.pth$').match(filename)), None)
    checkpoint_final_filename = next((filename for filename in subfolder_files if re.compile(r'.*_final\.pth$').match(filename)), None)
    progress_png_filename = next((filename for filename in subfolder_files if re.compile(r'.*\.png$').match(filename)), None)
    training_log_filename = next((filename for filename in subfolder_files if re.compile(r'.*\.txt$').match(filename)), None)
    debug_json_filename = next((filename for filename in subfolder_files if re.compile(r'.*\.json$').match(filename)), None)
    
    # Path to /iterations/results/ folder where training info files need to be moved to
    output_path = f"{base_dir}/{version}/{data_iterations_dir}/iteration_{iteration}/{iterations_results_dir}"

    os.rename(f"{nnUNet_results_subfolder}/fold_{fold}/{checkpoint_best_filename}", f"{output_path}/{checkpoint_best_filename}")
    os.rename(f"{nnUNet_results_subfolder}/fold_{fold}/{checkpoint_final_filename}", f"{output_path}/{checkpoint_final_filename}")
    os.rename(f"{nnUNet_results_subfolder}/fold_{fold}/{progress_png_filename}", f"{output_path}/{progress_png_filename}")
    os.rename(f"{nnUNet_results_subfolder}/fold_{fold}/{training_log_filename}", f"{output_path}/{training_log_filename}")
    os.rename(f"{nnUNet_results_subfolder}/fold_{fold}/{debug_json_filename}", f"{output_path}/{debug_json_filename}")

    log.info(f'Files "{checkpoint_best_filename}", "{checkpoint_final_filename}", "{progress_png_filename}", "{training_log_filename}" and "{debug_json_filename}" have been moved to "~/iteration_{iteration}/{iterations_results_dir}"')

    # ------------------------------- STEP 6: Remove the nnUNet_results subfolder ------------------------------- #
    try: 
        shutil.rmtree(nnUNet_results_subfolder)
    except OSError as e:
        log.error(f'Error trying to remove directory: {e.filename}, {e.strerror}')

    log.info(f'Subfolder "{data_results_dir}/{dataset_name}/{trainer_name}" has been removed')
    log.info(f'----------------------------------------------------------------------------\n')