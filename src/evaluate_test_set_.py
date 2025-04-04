import json
import logging

import tools

def evaluate_test_set(test_img_indices, config, log, iteration):

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version
    dataset_name = config.dataset_settings.dataset_name

    num_samples_test = config.test_settings.num_samples_test

    data_raw_dir = config.data_raw.dir

    data_iterations_dir = config.data_iterations.dir
    iterations_evaluations_dir = config.data_iterations.evaluations_dir

    log.info(f'---------------------------- Evaluate test set -----------------------------')

    # Number of samples in the training set
    dataset_json_nnUNetraw_path = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/dataset.json"
    with open(dataset_json_nnUNetraw_path, "r+") as jsonFile:

        # Get the dataset.json content
        data = json.load(jsonFile)

        # Number of training samples in current iteration
        num_samples_train = data["numTraining"]

    log.info(f"Number of samples in the training set: {num_samples_train}")

    # Number of samples in the test set
    log.info(f"Number of samples in the test set: {num_samples_test}")
    # log.info(f"Image indices of samples in the test set: {test_img_indices}")

    # Nested dictionary for storing evaluation metrics of all test predictions
    evaluation_metrics_test = {}

    # Add info to the dictionary
    evaluation_metrics_test["info"] = {
                                       "num_samples_train": num_samples_train,
                                       "num_samples_test": num_samples_test,
                                       "img_indices_test": test_img_indices
                                       }
    
    # Set logging level to WARNING
    log.setLevel(logging.WARNING)

    for i, img_index in enumerate(test_img_indices):

        # 1. Load prediction segmentation (as numpy array and nii.gz)
        prediction_segmentation, prediction_segmentation_nii = tools.load_prediction_segmentation(img_index, config, log, iteration)

        # 2. Load ground truth segmentation (as numpy array and nii.gz)
        ground_truth_segmentation, ground_truth_segmentation_nii = tools.load_ground_truth_segmentation(img_index, config, log, iteration)
        
        # 3. Load ground truth centerline
        ground_truth_centerline_indices = tools.load_ground_truth_centerline(img_index, prediction_segmentation_nii, config, log)

        # 4. Compute centerline from the predicted LAD segmentation
        prediction_centerline_indices = tools.compute_centerline_from_prediction(prediction_segmentation, prediction_segmentation_nii, img_index, config)
        
        # 5. Evaluate the prediction (w.r.t. the ground truth LAD centerline,)
        evaluation_metrics_centerline = tools.compute_evaluation_metrics_wrtGTcenterline(ground_truth_centerline_indices, prediction_centerline_indices, prediction_segmentation, img_index, log, config)

        # 6. Evaluate the prediction (w.r.t the ground truth LAD segmentation)
        evaluation_metrics_segmentation = tools.compute_evaluation_metrics_wrtGTsegmentation(ground_truth_segmentation, prediction_segmentation, img_index, log, config)

        # 7. Store evaluation metrics in nested dictionary
        evaluation_metrics_test[int(img_index)] = {**evaluation_metrics_segmentation, **evaluation_metrics_centerline}

    # 8. Compute mean of all evaluation metrics, and insert into nested dictionary
    evaluation_metrics_list, evaluation_metrics_mean, evaluation_metrics_std = tools.compute_mean_std_of_evaluation_metrics(evaluation_metrics_test, config, log)
    evaluation_metrics_test["evaluations_list"] = evaluation_metrics_list
    evaluation_metrics_test["evaluations_mean"] = evaluation_metrics_mean
    evaluation_metrics_test["evaluations_std"] = evaluation_metrics_std

    # 9. Sort the images according to the weighted centerline DICE score (in ascending order)
    # sorted_evaluation_metrics = tools.sort_all_evaluation_metrics(evaluation_metrics_test, "test", config, log)
    # evaluation_metrics_test["sorted_weighted_centerline_dice_scores"] = sorted_evaluation_metrics

    # Set logging level to INFO
    log.setLevel(logging.INFO)

    # Save dictionary to .json file
    evaluation_filename = f"evaluation_test_set_iteration_{iteration}.json"
    evaluation_path = f"{base_dir}/{version}/{data_iterations_dir}/iteration_{iteration}/{iterations_evaluations_dir}/{evaluation_filename}"

    with open(evaluation_path, 'w') as file:
        json.dump(evaluation_metrics_test, file, indent = 4)

    log.info(f'Evaluations on test set saved to: "~/iteration_{iteration}/{iterations_evaluations_dir}/{evaluation_filename}"')
    log.info(f'----------------------------------------------------------------------------\n')

    return evaluation_metrics_test

