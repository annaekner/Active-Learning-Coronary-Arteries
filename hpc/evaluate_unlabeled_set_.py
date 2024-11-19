import json
import logging
import numpy as np

import tools

def evaluate_unlabeled_set(config, log, iteration):

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version
    dataset_name = config.dataset_settings.dataset_name
    seed = config.base_settings.seed
    num_samples_test = config.test_settings.num_samples_test

    data_iterations_dir = config.data_iterations.dir
    iterations_evaluations_dir = config.data_iterations.evaluations_dir

    log.info(f'------------------------- Evaluate unlabeled set ---------------------------')

    # Get image indices of all predictions
    predictions_img_indices = tools.list_of_all_predictions(config, log, iteration)

    # Sample random image indices for the test set
    test_img_indices = np.random.default_rng(seed = seed).choice(predictions_img_indices, 
                                                                 size = num_samples_test, 
                                                                 replace = False)
    
    test_img_indices = test_img_indices.tolist()
    test_img_indices = sorted(test_img_indices)

    # Set the rest of the samples to be the unlabeled set
    unlabeled_img_indices = [x for x in predictions_img_indices if x not in test_img_indices]
    num_samples_unlabeled = len(unlabeled_img_indices)

    log.info(f"Number of samples in the unlabeled set: {num_samples_unlabeled}")
    # log.info(f"Image indices of samples in the unlabeled set: {unlabeled_img_indices}")

    # Nested dictionary for storing evaluation metrics of all unlabeled predictions
    evaluation_metrics_unlabeled = {}

    # Add info to the dictionary
    evaluation_metrics_unlabeled["info"] = {
                                       "num_samples_unlabeled": num_samples_unlabeled,
                                       "img_indices_unlabeled": unlabeled_img_indices
                                       }
    
    # Set logging level to WARNING
    log.setLevel(logging.WARNING)
    
    for i, img_index in enumerate(unlabeled_img_indices):

        # 1. Load prediction segmentation (as numpy array and nii.gz)
        prediction_segmentation, prediction_segmentation_nii = tools.load_prediction_segmentation(img_index, config, log, iteration)

        # 2. Load ground truth centerline
        ground_truth_centerline_indices = tools.load_ground_truth_centerline(img_index, prediction_segmentation_nii, config, log)

        # 3. Compute centerline from the predicted LAD segmentation
        prediction_centerline_indices = tools.compute_centerline_from_prediction(prediction_segmentation, prediction_segmentation_nii, img_index, config)
        
        # 4. Evaluate the prediction (w.r.t. the ground truth LAD centerline, which is always available)
        evaluation_metrics_centerline = tools.compute_evaluation_metrics_wrtGTcenterline(ground_truth_centerline_indices, prediction_centerline_indices, prediction_segmentation, img_index, log, config)

        # 5. Store evaluation metrics in nested dictionary
        evaluation_metrics_unlabeled[int(img_index)] = evaluation_metrics_centerline

    # Set logging level to INFO
    log.setLevel(logging.INFO)

    # Save dictionary to .json file
    evaluation_filename = f"evaluation_unlabeled_set_iteration_{iteration}.json"
    evaluation_path = f"{base_dir}/{version}/{data_iterations_dir}/iteration_{iteration}/{iterations_evaluations_dir}/{evaluation_filename}"

    with open(evaluation_path, 'w') as file:
        json.dump(evaluation_metrics_unlabeled, file, indent = 4)

    log.info(f'Evaluations on unlabeled set saved to: "~/iteration_{iteration}/{iterations_evaluations_dir}/{evaluation_filename}"')
    log.info(f'----------------------------------------------------------------------------\n')

    return evaluation_metrics_unlabeled