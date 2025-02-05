import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")
sns.set(font_scale=1.)

# Set up logging
log = logging.getLogger(__name__)

def get_evaluations_from_json(experiments_dir, experiment, num_iterations):

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

    for iteration in range(num_iterations):

        # Paths
        evaluations_subfolder = f"{directory}/iteration_{iteration}/evaluations"
        evaluations_test_set_json = f"{evaluations_subfolder}/evaluation_test_set_iteration_{iteration}.json"

        with open(evaluations_test_set_json, "r") as jsonFile:

            # Get the .json content
            data = json.load(jsonFile)

            # Get the evaluation metrics
            evaluations_mean = data["evaluations_mean"]
            evaluations_std = data["evaluations_std"]

            # Append to lists
            DICE.append(evaluations_mean["DICE_mean"])
            IoU.append(evaluations_mean["IoU_mean"])
            Hausdorff_distance.append(evaluations_mean["Hausdorff_distance_mean"])
            num_connected_components.append(evaluations_mean["num_connected_components_mean"])
            centerline_DICE.append(evaluations_mean["combined_centerline_dice_score_mean"])
            weighted_centerline_DICE.append(evaluations_mean["weighted_centerline_dice_score_mean"])
            entropy.append(evaluations_mean["entropy_mean"])

            DICE_std.append(evaluations_std["DICE_std"])
            IoU_std.append(evaluations_std["IoU_std"])
            Hausdorff_distance_std.append(evaluations_std["Hausdorff_distance_std"])
            num_connected_components_std.append(evaluations_std["num_connected_components_std"])
            centerline_DICE_std.append(evaluations_std["combined_centerline_dice_score_std"])
            weighted_centerline_DICE_std.append(evaluations_std["weighted_centerline_dice_score_std"])
            entropy_std.append(evaluations_std["entropy_std"])

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

def plot_evaluation_metric(all_evaluations, evaluation_metric, ax, title = None, y_label = None):

    # Get the evaluation metric across iterations for each selection strategy
    worst = all_evaluations["worst"][evaluation_metric][0]
    worst_std = all_evaluations["worst"][evaluation_metric][1]

    best = all_evaluations["best"][evaluation_metric][0]
    best_std = all_evaluations["best"][evaluation_metric][1]

    middle = all_evaluations["middle"][evaluation_metric][0]
    middle_std = all_evaluations["middle"][evaluation_metric][1]

    random = all_evaluations["random"][evaluation_metric][0]
    random_std = all_evaluations["random"][evaluation_metric][1]

    uncertainty = all_evaluations["uncertainty"][evaluation_metric][0]
    uncertainty_std = all_evaluations["uncertainty"][evaluation_metric][1]

    full_dataset = all_evaluations["full_dataset"][evaluation_metric][0]
    full_dataset_std = all_evaluations["full_dataset"][evaluation_metric][1]
    
    # Repeat the value of last iteration for all iterations
    full_dataset = [full_dataset[0]] * num_iterations        
    full_dataset_std = [full_dataset_std[0]] * num_iterations 

    # Samples in training set across iterations
    num_samples_dataset = 884 - 100  # Exluding the test set
    # num_samples_training = [5, 15, 25, 35, 45]
    # num_samples_training = [5, 30, 55, 80, 105]
    num_samples_training = [5, 30, 55, 80, 105, 130]
    # num_samples_training = [5, 30, 55, 80, 105, 130, 155, 180]
    percent_samples_training = [num_samples_training[i] / num_samples_dataset * 100 for i in range(len(num_samples_training))]

    # Combine num_samples_training and percent_samples_training into a list of strings
    xticks_labels = [f"{num_samples_training[i]} \n ({percent_samples_training[i]:.1f}%)" for i in range(len(num_samples_training))]

    # Plot mean of evaluation metric
    ax.plot(worst, label="LWOV", color='red') #, marker='o')
    ax.plot(best, label="HWOV", color='green') #, marker='o')
    ax.plot(middle, label="MWOV", color='orange') #, marker='o')
    ax.plot(random, label="Random", color='blue') #, marker='o')
    ax.plot(uncertainty, label="Entropy", color='purple')
    ax.plot(full_dataset, label="Full dataset", linestyle = '--', color='grey')

    # Plot standard deviation of evaluation metric as shaded area
    if not evaluation_metric in ["Hausdorff distance", "Number of connected components"]:
        ax.fill_between(range(num_iterations), np.array(worst) - np.array(worst_std), np.array(worst) + np.array(worst_std), color='red', alpha=0.1)
        ax.fill_between(range(num_iterations), np.array(best) - np.array(best_std), np.array(best) + np.array(best_std), color='green', alpha=0.1)
        ax.fill_between(range(num_iterations), np.array(middle) - np.array(middle_std), np.array(middle) + np.array(middle_std), color='orange', alpha=0.1)
        ax.fill_between(range(num_iterations), np.array(random) - np.array(random_std), np.array(random) + np.array(random_std), color='blue', alpha=0.1)
        ax.fill_between(range(num_iterations), np.array(uncertainty) - np.array(uncertainty_std), np.array(uncertainty) + np.array(uncertainty_std), color='purple', alpha=0.1)
        ax.fill_between(range(num_iterations), np.array(full_dataset) - np.array(full_dataset_std), np.array(full_dataset) + np.array(full_dataset_std), color='grey', alpha=0.1)
    
    x_ticks = list(np.arange(0, num_iterations))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(xticks_labels)
    ax.set_xlabel("Number(%) of annotated samples")
    
    if y_label:
        ax.set_ylabel(f"{y_label} (mean)")
    else:
        ax.set_ylabel(f"{evaluation_metric} (mean)")

    ax.legend()

    if title:
        ax.set_title(f"{title}")
    else:
        ax.set_title(f"{evaluation_metric}")

def plot_all_evaluation_metrics(all_evaluations, evaluation_metrics):
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    axs = axs.flatten()

    for i, evaluation_metric in enumerate(evaluation_metrics):
        plot_evaluation_metric(all_evaluations, evaluation_metric, axs[i])

    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    # plt.tight_layout()
    # plt.savefig("../Figures/results/all_evaluation_metrics_v6.png")
    plt.savefig("/zhome/cd/e/145569/Documents/special-course/all_evaluation_metrics_884samples_v3.png")
    # plt.show()

def plot_single_evaluation_metric(all_evaluations, evaluation_metric, title, y_label):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_evaluation_metric(all_evaluations, evaluation_metric, ax, title, y_label)
    plt.tight_layout() 
    plt.savefig(f"/zhome/cd/e/145569/Documents/special-course/{y_label}_884samples_v4.png")
    plt.show()

if __name__ == "__main__":

    # Directories
    # experiments_dir = r"C:/Users/annae/OneDrive - Danmarks Tekniske Universitet/Speciale/Specialkursus/experiments"
    experiments_dir = r"/work3/s193396"

    num_iterations = 6

    all_experiments = [
                       "experiment_worst_884samples_v3",
                       "experiment_best_884samples_v3",
                       "experiment_middle_884samples_v3",
                       "experiment_random_884samples_v3",
                       "experiment_uncertainty_884samples_v3",
                       "experiment_full_dataset_884samples_v1",
                       ]
    
    all_selections = [
                      "worst",
                      "best",
                      "middle",
                      "random",
                      "uncertainty",
                      "full_dataset"
                      ]
    
    all_metrics = [
                    "DICE",
                    "Centerline DICE",
                    "Hausdorff distance",
                    "IoU",
                    "Weighted centerline DICE",
                    "Number of connected components",
                ]

    # Dictionary for all evaluations
    # Keys are the selection strategy, either "worst", "best", "random", "uncertainty" or "full_dataset"
    all_evaluations = {}
                
    for experiment, selection in zip(all_experiments, all_selections):

        # Get evaluations for the experiment   
        evaluations = get_evaluations_from_json(experiments_dir, experiment, num_iterations)

        # Append to dictionary
        all_evaluations[selection] = evaluations

    # Plot all evaluation metrics
    # plot_all_evaluation_metrics(all_evaluations, all_metrics)

    # Plot a single evaluation metric
    plot_single_evaluation_metric(all_evaluations, "Weighted centerline DICE", title = "Weighted overlap", y_label = "WOV")