import subprocess

def train(config, log):

    # Configuration settings
    dataset_id = config.dataset_settings.dataset_id
    network_configuration = config.train_settings.network_configuration
    trainer = config.train_settings.trainer
    fold = config.train_settings.fold

    # Run the nnUNetv2_train command
    # EXAMPLE: nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainer_50epochs
    train_command = [
        "nnUNetv2_train",
        f"{dataset_id}",
        f"{network_configuration}",
        f"{fold}",
        "-tr", f"{trainer}"
    ]

    log.info(f"Train command: {train_command}")

    try:
        result = subprocess.run(train_command, check=True, capture_output=True, text=True, encoding='utf-8')
        log.info(f"Command output: {result.stdout}")
        log.info(f"Command error (if any): {result.stderr}")
        
    except subprocess.CalledProcessError as e:
        log.error(f"Command failed with error: {e.stderr}")