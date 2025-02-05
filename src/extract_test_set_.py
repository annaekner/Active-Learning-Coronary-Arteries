import re
import os
import glob
import numpy as np

# Number of samples for the test set
num_samples_test = 10
seed = 0

# Paths
experiment_folder_path = "/work3/s193396/initial_2_onechannel_full_dataset"
train_labels_path = f"{experiment_folder_path}/nnUNet_raw/Dataset001_Heart/labelsTr"
train_labels_filenames = [os.path.basename(x) for x in glob.glob(f"{train_labels_path}/*.nii.gz")]
train_labels_indices = sorted([int(re.search(r'img(\d+)\.nii\.gz', path).group(1)) for path in train_labels_filenames])

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

# # Load the .txt file
# with open(test_img_indices_path, "r") as file:
#     test_img_indices = [int(line.strip()) for line in file]
#     print(test_img_indices)