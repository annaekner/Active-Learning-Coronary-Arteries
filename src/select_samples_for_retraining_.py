import json
import numpy as np

def select_samples_for_retraining(evaluation_metrics_unlabeled, config, log, iteration):
    """
    Select n samples for retraining based on the evaluation metrics of all predictions of unlabeled set.

    Args
        evaluation_metrics_unlabeled: Nested dictionary of evaluation metrics for the unlabeled set
    """

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version
    dataset_name = config.dataset_settings.dataset_name
    seed = config.base_settings.seed

    num_samples_per_retraining = config.retraining.num_samples_per_retraining
    selection_method = config.retraining.selection_method  # 'worst', 'best', 'uncertainty', or 'random'

    data_iterations_dir = config.data_iterations.dir
    iterations_evaluations_dir = config.data_iterations.evaluations_dir

    # All samples in the unlabeled set
    all_sample_indices_str = list(evaluation_metrics_unlabeled.keys())[1:]  # Strings, e.g. '1', '2', ...
    num_samples = len(all_sample_indices_str)

    # List of weighted DICE score for each sample
    # Weighted score is computed as: weighted_score = (1 - num_cc * 0.1) * dice
    # Samples with only one connected component will have high weighted scores, 
    # samples with multiple connected components will have low weighted scores, 
    all_weighted_scores = np.array([evaluation_metrics_unlabeled[img_index]["weighted_centerline_dice_score"] for img_index in all_sample_indices_str])

    # Sort weighted scores in descending order (from high to low), and sort sample indices accordingly
    sorted_indices_according_to_weighted_scores = np.argsort(all_weighted_scores)[::-1]

    all_weighted_scores_sorted = all_weighted_scores[sorted_indices_according_to_weighted_scores]
    all_sample_indices_sorted_according_to_weighted_scores = np.array(all_sample_indices_str)[sorted_indices_according_to_weighted_scores]

    # List of entropy (uncertainty) for each sample 
    all_entropy = np.array([evaluation_metrics_unlabeled[img_index]["entropy"] for img_index in all_sample_indices_str])

    # Sort entropy in descending order (from high to low), and sort sample indices accordingly
    sorted_indices_according_to_entropy = np.argsort(all_entropy)[::-1]

    all_entropy_sorted = all_entropy[sorted_indices_according_to_entropy]
    all_sample_indices_sorted_according_to_entropy = np.array(all_sample_indices_str)[sorted_indices_according_to_entropy]
    
    if selection_method == 'worst':

        # Select the samples with the lowest weighted scores
        samples_for_retraining = all_sample_indices_sorted_according_to_weighted_scores[-num_samples_per_retraining:]
    
    elif selection_method == 'best':

        # Select the samples with the highest weighted scores
        samples_for_retraining = all_sample_indices_sorted_according_to_weighted_scores[:num_samples_per_retraining]

    elif selection_method == 'middle':

        # Select samples between the 25th and 75th percentile of the weighted scores
        lower_bound = int(num_samples * 0.25)
        upper_bound = int(num_samples * 0.75)

        samples_for_retraining = all_sample_indices_sorted_according_to_weighted_scores[lower_bound:upper_bound]

    elif selection_method == 'uncertainty':
        
        # Select the samples with the highest entropy (uncertainty)
        samples_for_retraining = all_sample_indices_sorted_according_to_entropy[:num_samples_per_retraining]

    elif selection_method == 'random':

        # Select random samples
        samples_for_retraining = np.random.default_rng(seed = seed).choice(all_sample_indices_sorted_according_to_weighted_scores, num_samples_per_retraining, replace = False)

    # Convert from numpy array to list
    samples_for_retraining = samples_for_retraining.tolist()

    retraining = {
                'num_samples_retraining': num_samples_per_retraining,
                'img_indices_retraining': samples_for_retraining,
                'selection_method': selection_method,
                'num_connected_components': [evaluation_metrics_unlabeled[sample]["num_connected_components"] for sample in samples_for_retraining],
                'predicted_centerline_coverage_of_ground_truth_centerline': [round(evaluation_metrics_unlabeled[sample]["predicted_centerline_coverage_of_ground_truth_centerline"], 4) for sample in samples_for_retraining],
                'ground_truth_centerline_coverage_of_predicted_centerline': [round(evaluation_metrics_unlabeled[sample]["ground_truth_centerline_coverage_of_predicted_centerline"], 4) for sample in samples_for_retraining],
                'combined_centerline_dice_score': [round(evaluation_metrics_unlabeled[sample]["combined_centerline_dice_score"], 4) for sample in samples_for_retraining],
                'weighted_centerline_dice_score': [round(all_weighted_scores_sorted[np.where(all_sample_indices_sorted_according_to_weighted_scores == sample)[0][0]].tolist(), 4) for sample in samples_for_retraining],
                'entropy': [round(all_entropy_sorted[np.where(all_sample_indices_sorted_according_to_entropy == sample)[0][0]].tolist(), 8) for sample in samples_for_retraining]
                }

    # Save dictionary to .json file
    retraining_filename = f"retraining_samples_iteration_{iteration}.json"
    retraining_path = f"{base_dir}/{version}/{data_iterations_dir}/iteration_{iteration}/{iterations_evaluations_dir}/{retraining_filename}"

    with open(retraining_path, 'w') as file:
        json.dump(retraining, file, indent = 4)

    log.info(f'-------------------------------- Re-training -------------------------------')
    log.info(f'Number of samples evaluated: {num_samples}')
    log.info(f'Number of samples selected for re-training: {num_samples_per_retraining}')
    log.info(f'Selection method: "{retraining["selection_method"]}"')
    log.info(f'Image indices of samples selected for re-training: {retraining["img_indices_retraining"]}')
    log.info(f'Number of connected components: {retraining["num_connected_components"]}')
    log.info(f'Predicted centerline coverage of the ground truth centerline: {retraining["predicted_centerline_coverage_of_ground_truth_centerline"]}')
    log.info(f'Ground truth centerline coverage of the predicted centerline: {retraining["ground_truth_centerline_coverage_of_predicted_centerline"]}')
    log.info(f'Combined centerline "DICE" score: {retraining["combined_centerline_dice_score"]}')
    log.info(f'Weighted centerline "DICE" score: {retraining["weighted_centerline_dice_score"]}')
    log.info(f'Entropy (uncertainty): {retraining["entropy"]}')
    log.info(f'Retraining info saved to: "~/iteration_{iteration}/{iterations_evaluations_dir}/{retraining_filename}"')
    log.info(f'----------------------------------------------------------------------------\n')

    return retraining