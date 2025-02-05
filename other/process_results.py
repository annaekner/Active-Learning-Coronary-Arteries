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
             "iteration_0": [0.9580],
             "iteration_1": [0.9408],
             "iteration_2": [0.9349],
             "iteration_3": [0.9267],
             "iteration_4": [0.9259]
             }, 
             
             "test": {
             "iteration_0": [0.7778, 0.9440, 0.8477, 0.7644, 0.6599, 0.8966, 0.6522, 0.7478, 0.9019, 0.9655, 0.6648, 0.5265, 0.8200, 0.7899, 0.8485, 0.3945, 0.9043],
             "iteration_0_retraining": [0.9655, 0.944, 0.9019],

             "iteration_1": [0.8644, 0.9126, 0.8889, 0.5641, 0.8304, 0.6190, 0.8229, 0.6844, 0.6820, 0.8123, 0.8561, 0.8861, 0.6455, 0.8610],
             "iteration_1_retraining": [0.9126, 0.8889, 0.8861],

             "iteration_2": [0.8274, 0.6778, 0.8546, 0.6223, 0.7802, 0.7150, 0.7789, 0.8182, 0.8378, 0.4721, 0.9485],
             "iteration_2_retraining": [0.6778, 0.8378, 0.9485],

             "iteration_3": [0.9079, 0.8805, 0.7030, 0.7975, 0.7526, 0.8185, 0.8717, 0.6310],
             "iteration_3_retraining": [0.8805, 0.9079, 0.8717],
             
             "iteration_4": [0.6738, 0.8355, 0.5840, 0.8767, 0.7075],
             "iteration_4_retraining": [0.8767, 0.8355, 0.7075]
             }, }
    
    random = {"train": {
             "iteration_0": [0.9538],
             "iteration_1": [0.9411],
             "iteration_2": [0.9330],
             "iteration_3": [0.9293],
             "iteration_4": [0.9206]
             }, 
             
             "test": {
             "iteration_0": [0.7565, 0.9466, 0.8591, 0.8341, 0.6371, 0.8348, 0.5787, 0.7417, 0.8067, 0.9496, 0.6017, 0.5358, 0.8024, 0.8491, 0.8065, 0.3425, 0.8940],
             "iteration_0_retraining": [0.6017, 0.8067, 0.8024],

             "iteration_1": [0.8688, 0.9375, 0.9218, 0.8591, 0.7398, 0.9339, 0.6339, 0.7729, 0.9690, 0.6929, 0.8740, 0.8159, 0.5404, 0.9084],
             "iteration_1_retraining": [0.969, 0.7729, 0.874],

             "iteration_2": [0.8703, 0.9337, 0.9253, 0.9289, 0.6679, 0.9113, 0.6806, 0.7308, 0.8778, 0.3981, 0.9564],
             "iteration_2_retraining": [0.6806, 0.9113, 0.7308],

             "iteration_3": [0.9153, 0.9350, 0.9183, 0.9526, 0.6296, 0.8487, 0.7321, 0.9481],
             "iteration_3_retraining": [0.6296, 0.9481, 0.8487],

             "iteration_4": [0.9327, 0.9288, 0.9213, 0.9667, 0.7435],
             "iteration_4_retraining": [0.9667, 0.7435, 0.9213]
             }, }
    
    return worst, best, random

def get_results_twochannel():
    """ 
    Results from using two channels (i.e. with the centerline distance map)
    """
    # Dictionaries for selection methods
    worst = {"train": {
             "iteration_0": [0.9448],
             "iteration_1": [0.9317],
             "iteration_2": [0.9225],
             "iteration_3": [0.9233],
             "iteration_4": [0.9206]
             }, 
             
             "test": {
             "iteration_0": [0.9625, 0.9607, 0.9519, 0.9617, 0.9367, 0.9696, 0.9775, 0.9195, 0.9723, 0.9565, 0.9304, 0.9522, 0.9916, 0.9746, 0.9588, 0.9689, 0.9707],
             "iteration_0_retraining": [0.9195, 0.9304, 0.9367],

             "iteration_1": [0.9552, 0.9662, 0.9446, 0.9710, 0.9740, 0.9583, 0.9755, 0.9765, 0.9417, 0.9875, 0.9529, 0.9537, 0.9648, 0.9554],
             "iteration_1_retraining": [0.9417, 0.9446, 0.9529],

             "iteration_2": [0.9622, 0.9747, 0.9713, 0.9649, 0.9238, 0.9724, 0.9883, 0.9664, 0.9781, 0.9791, 0.9670],
             "iteration_2_retraining": [0.9238, 0.9622, 0.9649],

             "iteration_3": [0.9718, 0.9758, 0.9693, 0.9922, 0.9833, 0.9641, 0.9896, 0.9780],
             "iteration_3_retraining": [0.9641, 0.9693, 0.9718],

             "iteration_4": [0.9805, 0.9923, 0.9665, 0.9931, 0.9708],
             "iteration_4_retraining": [0.9665, 0.9708, 0.9805],
             }, }
    
    best = {"train": {
             "iteration_0": [0.9472],
             "iteration_1": [0.9315],
             "iteration_2": [0.9238],
             "iteration_3": [0.9196],
             "iteration_4": [0.9228]
             }, 
             
             "test": {
             "iteration_0": [0.9521, 0.9746, 0.9622, 0.9471, 0.9321, 0.9610, 0.9727, 0.8983, 0.9692, 0.9686, 0.8969, 0.9258, 0.9916, 0.9565, 0.9588, 0.9718, 0.9483],
             "iteration_0_retraining": [0.9916, 0.9746, 0.9727],

             "iteration_1": [0.9340, 0.9724, 0.9854, 0.9401, 0.9657, 0.9595, 0.9755, 0.9724, 0.8986, 0.9649, 0.9600, 0.9472, 0.9752, 0.9632],
             "iteration_1_retraining": [0.9854, 0.9755, 0.9752],

             "iteration_2": [0.9552, 0.9654, 0.9259, 0.9698, 0.9633, 0.9844, 0.9248, 0.9618, 0.9748, 0.9669, 0.9781],
             "iteration_2_retraining": [0.9844, 0.9781, 0.9748],

             "iteration_3": [0.9690, 0.9655, 0.9447, 0.9697, 0.9590, 0.9185, 0.9355, 0.9809],
             "iteration_3_retraining": [0.9809, 0.9697, 0.969],

             "iteration_4": [0.9793, 0.9263, 0.9547, 0.9302, 0.9351],
             "iteration_4_retraining": [0.9793, 0.9547, 0.9351],
             }, }
    
    random = {"train": {
             "iteration_0": [0.9485],
             "iteration_1": [0.9358],
             "iteration_2": [0.9271],
             "iteration_3": [0.9221],
             "iteration_4": [0.9205]
             }, 
             
             "test": {
             "iteration_0": [0.9589, 0.9777, 0.9691, 0.9565, 0.9330, 0.9655, 0.9541, 0.9153, 0.9756, 0.9844, 0.9011, 0.9713, 0.9916, 0.9856, 0.9474, 0.9755, 0.9522],
             "iteration_0_retraining": [0.9011, 0.9756, 0.9916],

             "iteration_1": [0.9588, 0.9775, 0.9585, 0.9522, 0.9395, 0.9828, 0.9682, 0.9757, 0.9922, 0.9256, 0.9711, 0.9584, 0.9790, 0.9744],
             "iteration_1_retraining": [0.9922, 0.9757, 0.9711],

             "iteration_2": [0.9552, 0.9859, 0.9585, 0.9665, 0.9539, 0.9698, 0.9633, 0.9448, 0.9780, 0.9755, 0.9745],
             "iteration_2_retraining": [0.9633, 0.9698, 0.9448],

             "iteration_3": [0.9549, 0.9749, 0.9549, 0.9758, 0.9541, 0.9417, 0.9862, 0.9779],
             "iteration_3_retraining": [0.9541, 0.9779, 0.9417],

             "iteration_4": [0.9586, 0.9746, 0.9724, 0.9809, 0.9792],
             "iteration_4_retraining": [0.9809, 0.9792, 0.9724],
             }, }
    
    return worst, best, random

def get_arrays_for_plot(results, plot_type):
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

    iteration_all_test = [iteration_0_test_all, iteration_1_test_all, iteration_2_test_all, iteration_3_test_all, iteration_4_test_all]
    iteration_all_test_flat = [item for sublist in iteration_all_test for item in sublist]

    for i in range(5):
        iteration_all_test[i] = iteration_all_test[i] + [None]*(17-len(iteration_all_test[i]))

    # Re-training test scores (DICE for re-trained samples)
    iteration_0_test_retraining = results["test"]["iteration_0_retraining"]
    iteration_1_test_retraining = results["test"]["iteration_1_retraining"]
    iteration_2_test_retraining = results["test"]["iteration_2_retraining"]
    iteration_3_test_retraining = results["test"]["iteration_3_retraining"]
    iteration_4_test_retraining = results["test"]["iteration_4_retraining"]

    iteration_all_retraining = [iteration_0_test_retraining, iteration_1_test_retraining, iteration_2_test_retraining, iteration_3_test_retraining, iteration_4_test_retraining]

    # Test scores (mean DICE)
    iteration_0_test = np.mean(np.array(iteration_0_test_all))
    iteration_1_test = np.mean(np.array(iteration_1_test_all))
    iteration_2_test = np.mean(np.array(iteration_2_test_all))
    iteration_3_test = np.mean(np.array(iteration_3_test_all))
    iteration_4_test = np.mean(np.array(iteration_4_test_all))

    iteration_all_test_mean = [iteration_0_test, iteration_1_test, iteration_2_test, iteration_3_test, iteration_4_test]
    iteration_all_test_mean = [float(value) for value in iteration_all_test_mean]

    # Variance of test scores
    iteration_0_test_var = np.var(np.array(iteration_0_test_all))
    iteration_1_test_var = np.var(np.array(iteration_1_test_all))
    iteration_2_test_var = np.var(np.array(iteration_2_test_all))
    iteration_3_test_var = np.var(np.array(iteration_3_test_all))
    iteration_4_test_var = np.var(np.array(iteration_4_test_all))

    iteration_all_test_var = [iteration_0_test_var, iteration_1_test_var, iteration_2_test_var, iteration_3_test_var, iteration_4_test_var]
    iteration_all_test_var = [float(value) for value in iteration_all_test_var]

    if plot_type == "mean":
        return iteration_all_train, iteration_all_test_mean, iteration_all_test_var
    elif plot_type == "retraining":
        return iteration_all_test, iteration_all_retraining
    elif plot_type == "histogram":
        return iteration_all_test_flat

def plot_train_test_scores(worst, best, random, ylim, subplots=False):
    # "worst"
    worst_iteration_all_train, worst_iteration_all_test, worst_iteration_all_test_var = get_arrays_for_plot(worst, plot_type = "mean")

    # "best"
    best_iteration_all_train, best_iteration_all_test, best_iteration_all_test_var = get_arrays_for_plot(best, plot_type = "mean")

    # "random"
    random_iteration_all_train, random_iteration_all_test, random_iteration_all_test_var = get_arrays_for_plot(random, plot_type = "mean")

    if not subplots:
        plt.figure(figsize=(10, 6))

        # "worst"
        plt.errorbar(x=[0, 1, 2, 3, 4], y=worst_iteration_all_test, yerr=worst_iteration_all_test_var, fmt='none', color='grey', capsize=5, label="Worst - Unlabeled (variance)")
        plt.plot(worst_iteration_all_train, label="Worst - Validation", marker='o', linestyle='--', color='red')
        plt.plot(worst_iteration_all_test, label="Worst - Unlabeled", marker='o', color='red')

        # "best"
        plt.errorbar(x=[0, 1, 2, 3, 4], y=best_iteration_all_test, yerr=best_iteration_all_test_var, fmt='none', color='grey', capsize=5, label="Best - Unlabeled (variance)")
        plt.plot(best_iteration_all_train, label="Best - Validation", marker='o', linestyle='--', color='green')
        plt.plot(best_iteration_all_test, label="Best - Unlabeled", marker='o', color='green')

        # "random"
        plt.errorbar(x=[0, 1, 2, 3, 4], y=random_iteration_all_test, yerr=random_iteration_all_test_var, fmt='none', color='grey', capsize=5, label="Random - Unlabeled (variance)")
        plt.plot(random_iteration_all_train, label="Random - Validation", marker='o', linestyle='--', color='blue')
        plt.plot(random_iteration_all_test, label="Random - Unlabeled", marker='o', color='blue')

        plt.xticks([0, 1, 2, 3, 4], ["0", "1", "2", "3", "4"])
        plt.xlabel("Iteration")
        plt.ylabel("DICE (mean)")
        plt.title("Model performance throughout iterations")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if subplots:
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))

        # "worst"
        ax[0].errorbar(x=[0, 1, 2, 3, 4], y=worst_iteration_all_test, yerr=worst_iteration_all_test_var, fmt='none', color='grey', capsize=5, label="Unlabeled (variance)")
        ax[0].plot(worst_iteration_all_train, label="Validation", marker='o', linestyle='--', color='red')
        ax[0].plot(worst_iteration_all_test, label="Unlabeled", marker='o', color='red')
        ax[0].set_ylim(ylim)
        ax[0].set_xticks([0, 1, 2, 3, 4], ["0", "1", "2", "3", "4"])
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("DICE (mean)")
        ax[0].set_title("Selection strategy 'worst'")
        ax[0].legend()

        # "best"
        ax[1].errorbar(x=[0, 1, 2, 3, 4], y=best_iteration_all_test, yerr=best_iteration_all_test_var, fmt='none', color='grey', capsize=5, label="Unlabeled (variance)")
        ax[1].plot(best_iteration_all_train, label="Validation", marker='o', linestyle='--', color='green')
        ax[1].plot(best_iteration_all_test, label="Unlabeled", marker='o', color='green')
        ax[1].set_ylim(ylim)
        ax[1].set_xticks([0, 1, 2, 3, 4], ["0", "1", "2", "3", "4"])
        ax[1].set_yticklabels([])
        ax[1].set_xlabel("Iteration")
        ax[1].set_title("Selection strategy 'best'")
        ax[1].legend()

        # "random"
        ax[2].errorbar(x=[0, 1, 2, 3, 4], y=random_iteration_all_test, yerr=random_iteration_all_test_var, fmt='none', color='grey', capsize=5, label="Unlabeled (variance)")
        ax[2].plot(random_iteration_all_train, label="Validation", marker='o', linestyle='--', color='blue')
        ax[2].plot(random_iteration_all_test, label="Unlabeled", marker='o', color='blue')
        ax[2].set_ylim(ylim)
        ax[2].set_xticks([0, 1, 2, 3, 4], ["0", "1", "2", "3", "4"])
        ax[2].set_yticklabels([])
        ax[2].set_xlabel("Iteration")
        ax[2].set_title("Selection strategy 'random'")
        ax[2].legend()

        plt.suptitle("Model performance throughout iterations")
        plt.tight_layout()
        plt.show()

def plot_samples_selected_for_retraining(worst, best, random, ylim, subplots=False):
    # "worst"
    worst_iteration_all_test, worst_iteration_all_retraining = get_arrays_for_plot(worst, plot_type = "retraining")

    # "best"
    best_iteration_all_test, best_iteration_all_retraining = get_arrays_for_plot(best, plot_type = "retraining")

    # "random"
    random_iteration_all_test, random_iteration_all_retraining = get_arrays_for_plot(random, plot_type = "retraining")

    if subplots:
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))

        # "worst"
        # TODO: Need to hide/remove the re-train samples from the worst_iteration_all_test list
        ax[0].plot(worst_iteration_all_test, marker='o', color='red', linestyle = '', alpha=0.5) #, label = "Unlabeled samples")
        ax[0].plot(worst_iteration_all_retraining, marker='*', color='yellow', linestyle = '') #, label = "Retraining samples")
        ax[0].set_xticks([0, 1, 2, 3, 4], ["0", "1", "2", "3", "4"])
        ax[0].set_ylabel("DICE")
        ax[0].set_xlabel("Iteration")
        ax[0].set_title("Selection strategy 'worst'")

        handles = [plt.Line2D([0], [0], color='red', marker='o', markerfacecolor='red', linestyle='', markersize=6),
                   plt.Line2D([0], [0], color='yellow', marker='*', markerfacecolor='yellow', linestyle='', markersize=6)]
        labels = ['Unlabeled samples', 'Retraining samples']

        ax[0].legend(handles, labels, loc='lower right', ncol=1)
        # ax[0].legend()

        # "best"
        ax[1].plot(best_iteration_all_test, marker='o', color='green', linestyle = '', alpha=0.5) #, label = "Unlabeled samples")
        ax[1].plot(best_iteration_all_retraining, marker='*', color='yellow', linestyle = '') #, label = "Retraining samples")
        ax[1].set_xticks([0, 1, 2, 3, 4], ["0", "1", "2", "3", "4"])
        ax[1].set_yticklabels([])
        ax[1].set_xlabel("Iteration")
        ax[1].set_title("Selection strategy 'best'")
        handles = [plt.Line2D([0], [0], color='green', marker='o', markerfacecolor='green', linestyle='', markersize=6),
                   plt.Line2D([0], [0], color='yellow',marker='*', markerfacecolor='yellow',linestyle='', markersize=6)]
        labels = ['Unlabeled samples', 'Retraining samples']
        
        ax[1].legend(handles, labels, loc='lower right', ncol=1)
        # ax[1].legend()

        # "random"
        ax[2].plot(random_iteration_all_test, marker='o', color='blue', linestyle = '', alpha=0.5)
        ax[2].plot(random_iteration_all_retraining, marker='*', color='yellow', linestyle = '') #, label = "Retraining samples")
        ax[2].set_xticks([0, 1, 2, 3, 4], ["0", "1", "2", "3", "4"])
        ax[2].set_yticklabels([])
        ax[2].set_xlabel("Iteration")
        ax[2].set_title("Selection strategy 'random'")
        handles = [plt.Line2D([0], [0], color='blue',marker='o', markerfacecolor='blue', linestyle='', markersize=6),
                   plt.Line2D([0], [0], color='yellow',marker='*', markerfacecolor='yellow', linestyle='', markersize=6)]
        labels = ['Unlabeled samples', 'Retraining samples']
        
        ax[2].legend(handles, labels, loc='lower right', ncol=1)
        # ax[2].legend()

        plt.suptitle("Samples selected for retraining")
        plt.tight_layout()
        plt.show()

def plot_histogram_onechannel_and_twochannel(worst_twochannel, 
                                             best_twochannel, 
                                             random_twochannel, 
                                             worst_onechannel, 
                                             best_onechannel, 
                                             random_onechannel):

    # One channel
    iteration_all_test_flat_worst_onechannel = get_arrays_for_plot(worst_onechannel, plot_type = "histogram")
    iteration_all_test_flat_best_onechannel = get_arrays_for_plot(best_onechannel, plot_type = "histogram")
    iteration_all_test_flat_random_onechannel = get_arrays_for_plot(random_onechannel, plot_type = "histogram")

    combined_onechannel = [iteration_all_test_flat_worst_onechannel, iteration_all_test_flat_best_onechannel, iteration_all_test_flat_random_onechannel]
    combined_onechannel_flat = [item for sublist in combined_onechannel for item in sublist]

    # Two channels
    iteration_all_test_flat_worst_twochannel = get_arrays_for_plot(worst_twochannel, plot_type = "histogram")
    iteration_all_test_flat_best_twochannel = get_arrays_for_plot(best_twochannel, plot_type = "histogram")
    iteration_all_test_flat_random_twochannel = get_arrays_for_plot(random_twochannel, plot_type = "histogram")

    combined_twochannel = [iteration_all_test_flat_worst_twochannel, iteration_all_test_flat_best_twochannel, iteration_all_test_flat_random_twochannel]
    combined_twochannel_flat = [item for sublist in combined_twochannel for item in sublist]

    # Mean of each list
    onechannel_mean = np.mean(combined_onechannel_flat)
    twochannel_mean = np.mean(combined_twochannel_flat)

    # Define the bin edges
    bin_edges = np.linspace(min(combined_onechannel_flat), max(combined_twochannel_flat), num=101) 

    # Bin the sentiment scores
    counts_onechannel, bin_edges_onechannel = np.histogram(combined_onechannel_flat, bins = bin_edges)
    counts_twochannel, bin_edges_twochannel = np.histogram(combined_twochannel_flat, bins = bin_edges)

    # Plotting the sentiment scores
    plt.figure(figsize=(8, 4))
    plt.bar(bin_edges[:-1], counts_onechannel, width=bin_edges_onechannel[1] - bin_edges_onechannel[0], label = 'One input channel', color='C1', alpha = 0.5)
    plt.bar(bin_edges[:-1], counts_twochannel, width=bin_edges_twochannel[1] - bin_edges_twochannel[0], label = 'Two input channels', color='C4', alpha = 0.5)
    plt.axvline(x=onechannel_mean, color='C1', linestyle='--', label = 'Mean (one channel)')
    plt.axvline(x=twochannel_mean, color='C4', linestyle='--', label = 'Mean (two channels)')
    plt.xlabel("DICE")
    plt.ylabel("Number of samples")
    plt.title("Histogram of all unlabeled samples")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Experiment WITHOUT centerline distance map
    worst_onechannel, best_onechannel, random_onechannel = get_results_onechannel()
    # plot_train_test_scores(worst_onechannel, best_onechannel, random_onechannel, ylim = [0.72, 0.975], subplots=True)
    # plot_samples_selected_for_retraining(worst_onechannel, best_onechannel, random_onechannel, ylim = None, subplots=True)

    # Experiment WITH centerline distance map
    worst_twochannel, best_twochannel, random_twochannel = get_results_twochannel()
    # plot_train_test_scores(worst_twochannel, best_twochannel, random_twochannel, ylim = [0.9175, 0.985], subplots=True)
    # plot_samples_selected_for_retraining(worst_twochannel, best_twochannel, random_twochannel, ylim = None, subplots=True)

    plot_histogram_onechannel_and_twochannel(worst_twochannel, best_twochannel, random_twochannel, worst_onechannel, best_onechannel, random_onechannel)