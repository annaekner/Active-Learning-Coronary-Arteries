import subprocess

def train(config, log, iteration):

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version
    dataset_name = config.dataset_settings.dataset_name
    dataset_id = config.dataset_settings.dataset_id

    data_iterations_dir = config.data_iterations.dir
    iterations_results_dir = config.data_iterations.results_dir

    network_configuration = config.train_settings.network_configuration
    trainer = config.train_settings.trainer
    fold = config.train_settings.fold
    finetuning = config.train_settings.finetuning


    # Run the nnUNetv2_train command
    # EXAMPLE: nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainer_50epochs

    # Finetuning: traing from scrath in first iteration, and finetune in all other iterations
    if finetuning: 

        # Path of the checkpoint to train from
        previous_iteration = iteration - 1
        checkpoint_filename = f"checkpoint_final.pth"
        checkpoint_path = f"{base_dir}/{version}/{data_iterations_dir}/iteration_{previous_iteration}/{iterations_results_dir}/{checkpoint_filename}"

        if iteration == 0:

            # Train without pre-trained weights
            train_command = [
                "nnUNetv2_train",
                f"{dataset_id}",
                f"{network_configuration}",
                f"{fold}",
                "-tr", f"{trainer}",
            ]

            log.info(f"Training without pre-trained weights")

        else:
            # Train with pre-trained weights
            train_command = [
                "nnUNetv2_train",
                f"{dataset_id}",
                f"{network_configuration}",
                f"{fold}",
                "-tr", f"{trainer}",
                "-pretrained_weights", f"{checkpoint_path}"  
            ]

            log.info(f"Training with pre-trained weights")
            log.info(f"Path to model checkpoint from previous iteration: '{checkpoint_path}'")
    
    # No finetuning: traing from scrath in every iteration
    elif not finetuning:
        
        # Train without pre-trained weights
        train_command = [
            "nnUNetv2_train",
            f"{dataset_id}",
            f"{network_configuration}",
            f"{fold}",
            "-tr", f"{trainer}" 
        ]

        log.info(f"Training without pre-trained weights")

    log.info(f"Train command: {train_command}")

    try:
        result = subprocess.run(train_command, check=True, capture_output=True, text=True, encoding='utf-8')
        log.info(f"Command output: {result.stdout}")
        log.info(f"Command error (if any): {result.stderr}")
        
    except subprocess.CalledProcessError as e:
        log.error(f"Command failed with error: {e.stderr}")