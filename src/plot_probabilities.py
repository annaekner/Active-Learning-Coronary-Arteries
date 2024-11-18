import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")
sns.set(font_scale=1.)

def get_probabilities():

    # Base path
    base_path = r"C:/Users/annae/OneDrive - Danmarks Tekniske Universitet/Speciale/Specialkursus"

    # Path to .npz files
    worst_prediction_path = f"{base_path}/img41.npz"  # Combined DICE = 0.6265
    best_prediction_path = f"{base_path}/img57.npz"   # Combined DICE = 0.9746

    # Load .npz files into object
    worst_prediction_npz = np.load(worst_prediction_path)
    best_prediction_npz = np.load(best_prediction_path)

    # Get probabilities array 
    # NOTE: Assuming that the LAD probabilities are stored in the second index of the array
    worst_probabilities = worst_prediction_npz['probabilities'][1]
    best_probabilities = best_prediction_npz['probabilities'][1]

    # Flatten arrays
    worst_probabilities = worst_probabilities.flatten()
    best_probabilities = best_probabilities.flatten()

    return worst_probabilities, best_probabilities

def plot_probabilities():
    
    # Get probabilities
    worst_probabilities, best_probabilities = get_probabilities()

    # Compute mean
    worst_mean = np.mean(worst_probabilities)
    best_mean = np.mean(best_probabilities)

    # Define the bin edges
    min_value = min(min(worst_probabilities), min(best_probabilities))
    max_value = max(max(worst_probabilities), max(best_probabilities))

    print(f"worst_probabilities: min = {min(worst_probabilities)}, max = {max(worst_probabilities)}, mean = {worst_mean}")
    print(f"best_probabilities: min = {min(best_probabilities)}, max = {max(best_probabilities)}, mean = {best_mean}")

    bin_edges = np.linspace(min_value, max_value, num=200) 

    # Bin the probabilities
    counts_worst, bin_edges_worst = np.histogram(worst_probabilities, bins = bin_edges)
    counts_best, bin_edges_best = np.histogram(best_probabilities, bins = bin_edges)

    fraction_zero_probability_worst = counts_worst[0] / len(worst_probabilities)
    fraction_zero_probability_best = counts_best[0] / len(best_probabilities)
    fraction_one_probability_worst = counts_worst[-1] / len(worst_probabilities)
    fraction_one_probability_best = counts_best[-1] / len(best_probabilities)

    print(f"Total number of voxels: {len(worst_probabilities)} (worst prediction), {len(best_probabilities)} (best prediction)")
    print(f"Fraction of voxels with probability = 0: {fraction_zero_probability_worst:.4f} (worst prediction), {fraction_zero_probability_best:.4f} (best prediction)")
    print(f"Fraction of voxels with probability = 1: {fraction_one_probability_worst:.4f} (worst prediction), {fraction_one_probability_best:.4f} (best prediction)")

    # Exclude the bin with value = 0
    counts_worst = counts_worst[1:]
    counts_best = counts_best[1:]
    bin_edges = bin_edges[1:]
    bin_edges_worst = bin_edges_worst[1:]
    bin_edges_best = bin_edges_best[1:]

    # Exclude the bin with value = 1
    counts_worst = counts_worst[:-1]
    counts_best = counts_best[:-1]
    bin_edges = bin_edges[:-1]
    bin_edges_worst = bin_edges_worst[:-1]
    bin_edges_best = bin_edges_best[:-1]

    # Compute mean from binned data
    worst_mean = np.mean(bin_edges_worst)
    best_mean = np.mean(bin_edges_best)
    
    print("Counts worst:", counts_worst[:10])
    print("Counts best:", counts_best[:10])
    print("Bin edges worst:", bin_edges_worst[:10])
    print("Bin edges best:", bin_edges_best[:10])

    # Plotting the probabilities
    plt.figure(figsize=(8, 4))
    plt.bar(bin_edges[:-1], counts_worst, width=bin_edges_worst[1] - bin_edges_worst[0], label = 'Worst prediction (DICE = 0.6265)', color='C1', alpha = 0.5)
    plt.bar(bin_edges[:-1], counts_best, width=bin_edges_best[1] - bin_edges_best[0], label = 'Best prediction (DICE = 0.9746)', color='C4', alpha = 0.5)
    # plt.axvline(x=worst_mean, color='C1', linestyle='--', label = 'Mean (worst prediction)')
    # plt.axvline(x=best_mean, color='C4', linestyle='--', label = 'Mean (best prediction)')
    plt.xlabel("Probabilities")
    plt.ylabel("Number of voxels")
    plt.title("Histogram of probabilities (worst vs. best prediction)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_probabilities()