import json
import hydra
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")
sns.set(font_scale=1.)

# Set up logging
log = logging.getLogger(__name__)

def get_evaluations_from_json(experiments_dir, experiment):

    directory = f"{experiments_dir}/{experiment}/iterations"

    # Lists for each metric
    DICE = []
    IoU = []
    Hausdorff_distance = []
    num_connected_components = []
    centerline_DICE = []
    weighted_centerline_DICE = []
    entropy = []

    # Standard deviation for each metric
    DICE_std = []
    IoU_std = []
    Hausdorff_distance_std = []
    num_connected_components_std = []
    centerline_DICE_std = []
    weighted_centerline_DICE_std = []
    entropy_std = []

    for iteration in range(5):

        # Paths
        evaluations_subfolder = f"{directory}/iteration_{iteration}/evaluations"
        evaluations_test_set_json = f"{evaluations_subfolder}/evaluation_test_set_iteration_{iteration}.json"

        with open(evaluations_test_set_json, "r") as jsonFile:

            # Get the .json content
            data = json.load(jsonFile)

            # Get the evaluation metrics
            evaluation_metrics = data["evaluations_mean"]

            # Append to lists
            DICE.append(evaluation_metrics["DICE_mean"])
            IoU.append(evaluation_metrics["IoU_mean"])
            Hausdorff_distance.append(evaluation_metrics["Hausdorff_distance_mean"])
            num_connected_components.append(evaluation_metrics["num_connected_components_mean"])
            centerline_DICE.append(evaluation_metrics["combined_centerline_dice_score_mean"])
            weighted_centerline_DICE.append(evaluation_metrics["weighted_centerline_dice_score_mean"])
            entropy.append(evaluation_metrics["entropy_mean"])

            DICE_std.append(evaluation_metrics["DICE_std"])
            IoU_std.append(evaluation_metrics["IoU_std"])
            Hausdorff_distance_std.append(evaluation_metrics["Hausdorff_distance_std"])
            num_connected_components_std.append(evaluation_metrics["num_connected_components_std"])
            centerline_DICE_std.append(evaluation_metrics["combined_centerline_dice_score_std"])
            weighted_centerline_DICE_std.append(evaluation_metrics["weighted_centerline_dice_score_std"])
            entropy_std.append(evaluation_metrics["entropy_std"])

        # There is only one iteration for the full dataset
        if "full_dataset" in experiment:
            break

    evaluations_experiment = {
                              "DICE": [DICE, DICE_std],
                              "IoU": [IoU, IoU_std],
                              "Hausdorff distance": [Hausdorff_distance, Hausdorff_distance_std],
                              "Number of connected components": [num_connected_components, num_connected_components_std],
                              "Centerline DICE": [centerline_DICE, centerline_DICE_std],
                              "Weighted centerline DICE": [weighted_centerline_DICE, weighted_centerline_DICE_std],
                              "Entropy": [entropy, entropy_std],
                              } 
    
    return evaluations_experiment

def plot_evaluation_metric(all_evaluations, evaluation_metric):

    # Get the evaluation metric across iterations for each selection strategy
    worst = all_evaluations["worst"][evaluation_metric][0]
    worst_std = all_evaluations["worst"][evaluation_metric][1]

    best = all_evaluations["best"][evaluation_metric][0]
    best_std = all_evaluations["best"][evaluation_metric][1]

    random = all_evaluations["random"][evaluation_metric][0]
    random_std = all_evaluations["random"][evaluation_metric][1]

    full_dataset = all_evaluations["full_dataset"][evaluation_metric][0]
    full_dataset_std = all_evaluations["full_dataset"][evaluation_metric][1]
    full_dataset = [full_dataset[0]] * 5          # Repeat the value for all iterations
    full_dataset_std = [full_dataset_std[0]] * 5  # Repeat the value for all iterations

    # Samples in training set across iterations
    num_samples_dataset = 70 - 10  # Exluding the test set
    num_samples_training = [3, 6, 9, 12, 15]
    percent_samples_training = [num_samples_training[i] / num_samples_dataset * 100 for i in range(len(num_samples_training))]

    # Combine num_samples_training and percent_samples_training into a list of strings
    xticks_labels = [f"{num_samples_training[i]} \n ({percent_samples_training[i]:.1f}%)" for i in range(len(num_samples_training))]

    plt.figure(figsize=(10, 6))

    # Plot mean of evaluation metric
    plt.plot(worst, label="Worst", marker='o', color='red')
    plt.plot(best, label="Best", marker='o', color='green')
    plt.plot(random, label="Random", marker='o', color='blue')
    plt.plot(full_dataset, label="Full dataset", linestyle = '--', color='grey')

    # Plot standard deviation of evaluation metric as shaded area
    plt.fill_between(range(5), np.array(worst) - np.array(worst_std), np.array(worst) + np.array(worst_std), color='red', alpha=0.2)
    plt.fill_between(range(5), np.array(best) - np.array(best_std), np.array(best) + np.array(best_std), color='green', alpha=0.2)
    plt.fill_between(range(5), np.array(random) - np.array(random_std), np.array(random) + np.array(random_std), color='blue', alpha=0.2)
    plt.fill_between(range(5), np.array(full_dataset) - np.array(full_dataset_std), np.array(full_dataset) + np.array(full_dataset_std), color='grey', alpha=0.2)

    plt.xticks([0, 1, 2, 3, 4], xticks_labels)
    plt.xlabel("Number(%) of samples in training set")
    plt.ylabel(f"{evaluation_metric} (mean)")
    # plt.title("Model performance throughout iterations")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Directories
    experiments_dir = r"C:/Users/annae/OneDrive - Danmarks Tekniske Universitet/Speciale/Specialkursus/experiments"

    all_experiments = [
                       "experiment_worst_70samples_v2",
                       "experiment_best_70samples_v2",
                       "experiment_random_70samples_v2",
                       "experiment_full_dataset_70samples_v2",
                       ]
    
    all_selections = [
                      "worst",
                      "best",
                      "random",
                      "full_dataset"
                      ]
    
    all_metrics = [
                    "DICE",
                    "IoU",
                    "Hausdorff distance",
                    "Number of connected components",
                    "Centerline DICE",
                    "Weighted centerline DICE",
                    "Entropy"
                ]

    # Dictionary for all evaluations
    # Keys are the selection strategy, either "worst", "best", "random" or "full_dataset"
    all_evaluations = {}
                
    for experiment, selection in zip(all_experiments, all_selections):

        # Get evaluations for the experiment   
        evaluations = get_evaluations_from_json(experiments_dir, experiment)

        # Append to dictionary
        all_evaluations[selection] = evaluations

    # Plot each evaluation metric
    for metric in all_metrics:

        plot_evaluation_metric(all_evaluations, metric)
