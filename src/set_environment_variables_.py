import os

def set_environment_variables(config, log):

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version
    data_raw_dir = config.data_raw.dir
    data_preprocessed_dir = config.data_preprocessed.dir
    data_results_dir = config.data_results.dir

    # Set the nnUNet environment variables
    nnUNet_raw = f"{base_dir}/{version}/{data_raw_dir}"
    nnUNet_preprocessed = f"{base_dir}/{version}/{data_preprocessed_dir}"
    nnUNet_results = f"{base_dir}/{version}/{data_results_dir}"

    # Update the environment variables in the current process
    os.environ["nnUNet_raw"] = nnUNet_raw
    os.environ["nnUNet_preprocessed"] = nnUNet_preprocessed
    os.environ["nnUNet_results"] = nnUNet_results