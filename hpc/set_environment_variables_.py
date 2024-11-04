import os
import subprocess

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

    # Example:
    # export nnUNet_raw="/work3/s193396/v1/nnUNet_raw"
    # export nnUNet_preprocessed="/work3/s193396/v1/nnUNet_preprocessed"
    # export nnUNet_results="/work3/s193396/v1/nnUNet_results"

    # # Set the nnUNet environment variables
    # nnUNet_raw_command = f'export nnUNet_raw="{base_dir}/{version}/{data_raw_dir}"'
    # nnUNet_preprocessed_command = f'export nnUNet_preprocessed="{base_dir}/{version}/{data_preprocessed_dir}"'
    # nnUNet_results_command = f'export nnUNet_results="{base_dir}/{version}/{data_results_dir}"'
    
    # try:
    #     result = subprocess.run(nnUNet_raw_command, check=True, capture_output=True, text=True)
    #     result = subprocess.run(nnUNet_preprocessed_command, check=True, capture_output=True, text=True)
    #     result = subprocess.run(nnUNet_results_command, check=True, capture_output=True, text=True)

    # except subprocess.CalledProcessError as e:
    #     log.error(f"Command failed with error: {e.stderr}")