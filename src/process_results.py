import hydra
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")
sns.set(font_scale=1.)

# Set up logging
log = logging.getLogger(__name__)

def get_results_onechannel():
    """ 
    Results from using just one channel (i.e. without the centerline distance map)
    """

    # Dictionaries for selection methods
    worst = {"train": {
             # 3 samples (img: 1, 2, 3)
             "iteration_0": [0.9540],

             # 6 samples (extra img: 16, 17, 21)
             "iteration_1": [0.9336],

             # 9 samples (extra img: 5, 3, 2)
             "iteration_2": [0.9274],

             # 12 samples (extra img: 18, 13, 20)
             "iteration_3": [0.9246],

             # 15 samples (extra img: 4, 6, 7)
             "iteration_4": [0.9185]
             }, 
             
             "test": {
             # 17 samples
             "iteration_0": [0.7715, 0.9387, 0.9156, 0.8404, 0.6322, 0.8798, 0.6141, 0.7652, 0.7509, 0.9656, 0.4863, 0.5095, 0.8217, 0.8517, 0.7977, 0.5231, 0.8866],
             "iteration_0_retraining": [0.4863, 0.5095, 0.5231],

             # 14 samples
             "iteration_1": [0.8517, 0.9354, 0.9061, 0.7565, 0.7769, 0.8921, 0.7200, 0.8631, 0.7637, 0.9614, 0.8523, 0.9077, 0.8600, 0.8767],
             "iteration_1_retraining": [0.72, 0.7565, 0.7637],

             # 11 samples
             "iteration_2": [0.8214, 0.9463, 0.9172, 0.6589, 0.8992, 0.9076, 0.9526, 0.8726, 0.9132, 0.8873, 0.9593],
             "iteration_2_retraining": [0.6589, 0.8214, 0.8992],

             # 8 samples
             "iteration_3": [0.9371, 0.9336, 0.7558, 0.9153, 0.9300, 0.9098, 0.8672, 0.9442],
             "iteration_3_retraining": [0.93, 0.7558, 0.8672],

             # 5 samples
             "iteration_4": [0.9402, 0.9723, 0.9073, 0.9023, 0.9522],
             "iteration_4_retraining": [0.9023, 0.9073, 0.9402]
             }, }
    
    best = {"train": {
             "iteration_0": [],
             "iteration_1": [],
             "iteration_2": [],
             "iteration_3": [],
             "iteration_4": []
             }, 
             
             "test": {
             "iteration_0": [],
             "iteration_0_retraining": [],
             "iteration_1": [],
             "iteration_1_retraining": [],
             "iteration_2": [],
             "iteration_2_retraining": [],
             "iteration_3": [],
             "iteration_3_retraining": [],
             "iteration_4": [],
             "iteration_4_retraining": []
             }, }
    
    random = {"train": {
             "iteration_0": [],
             "iteration_1": [],
             "iteration_2": [],
             "iteration_3": [],
             "iteration_4": []
             }, 
             
             "test": {
             "iteration_0": [],
             "iteration_0_retraining": [],
             "iteration_1": [],
             "iteration_1_retraining": [],
             "iteration_2": [],
             "iteration_2_retraining": [],
             "iteration_3": [],
             "iteration_3_retraining": [],
             "iteration_4": [],
             "iteration_4_retraining": []
             }, }
    
    return worst, best, random

def get_arrays_for_plot(results):
    # Train scores (mean DICE)
    iteration_0_train = results["train"]["iteration_0"][0]
    iteration_1_train = results["train"]["iteration_1"][0]
    iteration_2_train = results["train"]["iteration_2"][0]
    iteration_3_train = results["train"]["iteration_3"][0]
    iteration_4_train = results["train"]["iteration_4"][0]

    iteration_all_train = [iteration_0_train, iteration_1_train, iteration_2_train, iteration_3_train, iteration_4_train]

    # Test scores (DICE for all samples)
    iteration_0_test_all = results["test"]["iteration_0"]
    iteration_1_test_all = results["test"]["iteration_1"]
    iteration_2_test_all = results["test"]["iteration_2"]
    iteration_3_test_all = results["test"]["iteration_3"]
    iteration_4_test_all = results["test"]["iteration_4"]

    # Test scores (mean DICE)
    iteration_0_test = np.mean(np.array(iteration_0_test_all))
    iteration_1_test = np.mean(np.array(iteration_1_test_all))
    iteration_2_test = np.mean(np.array(iteration_2_test_all))
    iteration_3_test = np.mean(np.array(iteration_3_test_all))
    iteration_4_test = np.mean(np.array(iteration_4_test_all))

    iteration_all_test = [iteration_0_test, iteration_1_test, iteration_2_test, iteration_3_test, iteration_4_test]
    iteration_all_test = [float(value) for value in iteration_all_test]

    # Variance of test scores
    iteration_0_test_var = np.var(np.array(iteration_0_test_all))
    iteration_1_test_var = np.var(np.array(iteration_1_test_all))
    iteration_2_test_var = np.var(np.array(iteration_2_test_all))
    iteration_3_test_var = np.var(np.array(iteration_3_test_all))
    iteration_4_test_var = np.var(np.array(iteration_4_test_all))

    iteration_all_test_var = [iteration_0_test_var, iteration_1_test_var, iteration_2_test_var, iteration_3_test_var, iteration_4_test_var]
    iteration_all_test_var = [float(value) for value in iteration_all_test_var]

    return iteration_all_train, iteration_all_test, iteration_all_test_var

def plot_train_test_scores(worst, best, random):
    # "worst"
    worst_iteration_all_train, worst_iteration_all_test, worst_iteration_all_test_var = get_arrays_for_plot(worst)

    # "best"
    best_iteration_all_train, best_iteration_all_test, best_iteration_all_test_var = get_arrays_for_plot(worst) # TODO: Change

    # "random"
    random_iteration_all_train, random_iteration_all_test, random_iteration_all_test_var = get_arrays_for_plot(worst) # TODO: Change

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    # "worst"
    ax[0].errorbar(x=[0, 1, 2, 3, 4], y=worst_iteration_all_test, yerr=worst_iteration_all_test_var, fmt='none', color='grey', capsize=5, label="Test (variance)")
    ax[0].plot(worst_iteration_all_train, label="Train", marker='o', linestyle='--', color='red')
    ax[0].plot(worst_iteration_all_test, label="Test", marker='o', color='red')
    # ax[0].set_ylim([])
    ax[0].set_xticks([0, 1, 2, 3, 4], ["0", "1", "2", "3", "4"])
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("DICE (mean)")
    ax[0].set_title("Selection strategy 'worst'")
    ax[0].legend()

    # "best"
    ax[1].errorbar(x=[0, 1, 2, 3, 4], y=best_iteration_all_test, yerr=best_iteration_all_test_var, fmt='none', color='grey', capsize=5, label="Test (variance)")
    ax[1].plot(best_iteration_all_train, label="Train", marker='o', linestyle='--', color='green')
    ax[1].plot(best_iteration_all_test, label="Test", marker='o', color='green')
    # ax[1].set_ylim([])
    ax[1].set_xticks([0, 1, 2, 3, 4], ["0", "1", "2", "3", "4"])
    # ax[1].set_yticklabels([])
    ax[1].set_xlabel("Iteration")
    ax[1].set_title("Selection strategy 'best'")
    ax[1].legend()

    # "random"
    ax[2].errorbar(x=[0, 1, 2, 3, 4], y=random_iteration_all_test, yerr=random_iteration_all_test_var, fmt='none', color='grey', capsize=5, label="Test (variance)")
    ax[2].plot(random_iteration_all_train, label="Train", marker='o', linestyle='--', color='blue')
    ax[2].plot(random_iteration_all_test, label="Test", marker='o', color='blue')
    # ax[2].set_ylim([])
    ax[2].set_xticks([0, 1, 2, 3, 4], ["0", "1", "2", "3", "4"])
    # ax[2].set_yticklabels([])
    ax[2].set_xlabel("Iteration")
    ax[2].set_title("Selection strategy 'random'")
    ax[2].legend()

    plt.suptitle("Model performance throughout iterations")
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    worst, best, random = get_results_onechannel()
    plot_train_test_scores(worst, best, random)