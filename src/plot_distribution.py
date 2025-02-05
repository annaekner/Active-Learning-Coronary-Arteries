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

    directory = f"{experiments_dir}/{experiment}/iterations/iteration_0/evaluations"
    filename = f"{directory}/evaluation_unlabeled_set_iteration_0.json"

    with open(filename, "r") as jsonFile:
        data = json.load(jsonFile)

    # Initialize an empty list to store the WOV scores
    WOV_scores = []

    # Iterate through each image index and extract the score
    for img_index in data.keys():
        score = data[img_index].get('weighted_centerline_dice_score')

        if score is not None:
            WOV_scores.append(score)
    
    # Check the length of the list
    print(f"Number of WOV scores: {len(WOV_scores)}")

    return WOV_scores

def plot_distribution(WOV_scores):

    # Plot the distribution of the WOV scores
    plt.figure(figsize=(10, 5))
    sns.histplot(WOV_scores, kde=True, bins=40)
    plt.title("Distribution of weighted overlap scores")
    plt.xlim(0, 1)
    plt.xlabel("WOV")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"/zhome/cd/e/145569/Documents/special-course/WOV_distribution_884samples_v4.png")
    plt.show()

if __name__ == "__main__":

    experiments_dir = r"/work3/s193396"
    experiment = "experiment_middle_884samples_v3"

    # Get the evaluations from the .json file
    WOV_scores = get_evaluations_from_json(experiments_dir, experiment)

    # Plot the distribution of the WOV scores
    plot_distribution(WOV_scores)