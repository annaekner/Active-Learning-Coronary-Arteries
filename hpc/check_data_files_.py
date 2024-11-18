import os
import re
import json
import glob

def check_data_files(config, log):

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version
    dataset_name = config.dataset_settings.dataset_name
    num_channels = config.base_settings.num_channels

    data_raw_dir = config.data_raw.dir
    train_images_dir = config.data_raw.train_images_dir
    test_images_dir = config.data_raw.test_images_dir
    train_labels_dir = config.data_raw.train_labels_dir
    test_labels_dir = config.data_raw.test_labels_dir

    data_centerlines_dir = config.data_centerlines.dir

    dataset_json_nnUNetraw_path = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/dataset.json"

    # Number of samples in imagesTr, imagesTs, labelsTr, labelsTr
    train_images_path = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{train_images_dir}"
    test_images_path = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{test_images_dir}"
    train_labels_path = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{train_labels_dir}"
    test_labels_path = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{test_labels_dir}"
    centerlines_path = f"{base_dir}/{version}/{data_centerlines_dir}/{dataset_name}"

    train_images_filepaths = [os.path.basename(x) for x in glob.glob(f"{train_images_path}/*_0000.nii.gz")]
    train_labels_filepaths = [os.path.basename(x) for x in glob.glob(f"{train_labels_path}/*.nii.gz")]
    centerlines_indices_filepaths = os.listdir(centerlines_path)

    train_images_indices = sorted([int(re.search(r'img(\d+)_0000\.nii\.gz', path).group(1)) for path in train_images_filepaths])
    train_labels_indices = sorted([int(re.search(r'img(\d+)\.nii\.gz', path).group(1)) for path in train_labels_filepaths])
    centerlines_indices = sorted([int(re.search(r'img(\d+)_lad_centerline\.vtk', path).group(1)) for path in centerlines_indices_filepaths])

    num_train_images = len(glob.glob(f"{train_images_path}/*_0000.nii.gz"))
    num_test_images = len(glob.glob(f"{test_images_path}/*_0000.nii.gz"))
    num_train_labels = len(os.listdir(train_labels_path))
    num_test_labels = len(os.listdir(test_labels_path))
    num_centerlines = len(os.listdir(centerlines_path))

    # nnUNet_raw/dataset.json
    with open(dataset_json_nnUNetraw_path, "r+") as jsonFile:

        # Get the dataset.json content
        data = json.load(jsonFile)

        # Get the total number of samples in the dataset
        num_training_samples_json = data["numTraining"]

        # Get the number of channels in the dataset
        channel_names = data["channel_names"]
        num_channels_json = len(channel_names.keys())

    # Assert that everything looks fine
    assert num_train_images == num_training_samples_json
    assert num_train_images == num_train_labels
    assert num_train_images == num_centerlines
    assert num_test_images == 0
    assert num_test_labels == 0
    assert num_channels == num_channels_json
    assert train_images_indices == train_labels_indices
    assert train_images_indices == centerlines_indices

    log.info(f"----------------- Data files have successfully been checked ----------------")