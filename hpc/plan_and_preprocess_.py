import subprocess

def plan_and_preprocess(config, log):

    # Configuration settings
    dataset_id = config.dataset_settings.dataset_id
    num_processors = config.train_settings.num_processors
    network_configuration = config.train_settings.network_configuration

    # Run the nnUNetv2_plan_and_preprocess command
    # EXAMPLE: nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres -np 4 --verify_dataset_integrity
    plan_and_preprocess_command = [
        "nnUNetv2_plan_and_preprocess",
        "-d", f"{dataset_id}",
        "-c", f"{network_configuration}",
        "-np", f"{num_processors}",
        "--verify_dataset_integrity"
    ]

    log.info(f"Plan and preprocess command: {plan_and_preprocess_command}")
    
    try:
        result = subprocess.run(plan_and_preprocess_command, check=True, capture_output=True, text=True, encoding='utf-8')
        log.info(f"Command output: {result.stdout}")
        log.info(f"Command error (if any): {result.stderr}")
        
    except subprocess.CalledProcessError as e:
        log.error(f"Command failed with error: {e.stderr}")