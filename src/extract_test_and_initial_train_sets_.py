import re
import os
import glob
import numpy as np

# Number of samples for the test set and initial training set
num_samples_test = 100
num_samples_initial_training = 5
seed = 0

# Paths
experiment_folder_path = "/work3/s193396/initial_884samples_full_dataset"
train_labels_path = f"{experiment_folder_path}/nnUNet_raw/Dataset001_Heart/labelsTr"
train_labels_filenames = [os.path.basename(x) for x in glob.glob(f"{train_labels_path}/*.nii.gz")]
train_labels_indices = sorted([int(re.search(r'img(\d+)\.nii\.gz', path).group(1)) for path in train_labels_filenames])

# -------------------------------- Extract test set -------------------------------- #
# Sample random image indices for the test set
test_img_indices = np.random.default_rng(seed = seed).choice(train_labels_indices, 
                                                             size = num_samples_test, 
                                                             replace = False)

test_img_indices = test_img_indices.tolist()
test_img_indices = sorted(test_img_indices)

# Path to the test image indices .txt file
test_img_indices_path = f"{experiment_folder_path}/test_img_indices.txt"

# Save to .txt file
with open(test_img_indices_path, "w") as file:
    for index in test_img_indices:

        # Write them as comma-separated values
        file.write(f"{index}\n")

# Load the .txt file
with open(test_img_indices_path, "r") as file:
    test_img_indices = [int(line.strip()) for line in file]
print(f"Number of test samples: {len(test_img_indices)}")

# -------------------------- Extract initial training set -------------------------- #
# Remove the test set from the training set
train_labels_indices_without_test_set = list(set(train_labels_indices) - set(test_img_indices))

# Sample random image indices for the initial training set
initial_training_img_indices = np.random.default_rng(seed = seed).choice(train_labels_indices_without_test_set, 
                                                                         size = num_samples_initial_training, 
                                                                         replace = False)

initial_training_img_indices = initial_training_img_indices.tolist()
initial_training_img_indices = sorted(initial_training_img_indices)

# Path to the initial training image indices .txt file
initial_training_img_indices_path = f"{experiment_folder_path}/initial_training_img_indices.txt"

# Save to .txt file
with open(initial_training_img_indices_path, "w") as file:
    for index in initial_training_img_indices:

        # Write them as comma-separated values
        file.write(f"{index}\n")

# Load the .txt file
with open(initial_training_img_indices_path, "r") as file:
    initial_training_img_indices = [int(line.strip()) for line in file]
print(f"Number of initial training samples: {len(initial_training_img_indices)}")

# Check that there is no overlap between the test set and the initial training set
assert len(set(test_img_indices).intersection(set(initial_training_img_indices))) == 0