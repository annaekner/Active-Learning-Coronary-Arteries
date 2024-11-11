import subprocess

def predict(config, log):

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version
    iteration = config.base_settings.iteration
    dataset_name = config.dataset_settings.dataset_name
    dataset_id = config.dataset_settings.dataset_id

    network_configuration = config.train_settings.network_configuration
    trainer = config.train_settings.trainer
    fold = config.train_settings.fold

    data_raw_dir = config.data_raw.dir
    test_images_dir = config.data_raw.test_images_dir
    data_predicted_dir = config.data_predicted.dir

    # Input and output directory
    input_dir = f"{base_dir}/{version}/{data_raw_dir}/{dataset_name}/{test_images_dir}"
    output_dir = f"{base_dir}/{version}/{data_predicted_dir}/{dataset_name}/iteration_{iteration}"

    # Run the nnUNetv2_predict command
    # EXAMPLE: nnUNetv2_predict -i /work3/s193396/original/nnUNet_raw/Dataset001_Heart/imagesTs 
    #                           -o /work3/s193396/original/nnUNet_predictions/Dataset001_Heart/iteration_0/ 
    #                           -d 1 -c 3d_fullres -tr nnUNetTrainer_50epochs -f all

    predict_command = [
        "nnUNetv2_predict",
        "-i", f"{input_dir}",
        "-o", f"{output_dir}",
        "-d", f"{dataset_id}",
        "-c", f"{network_configuration}",
        "-tr", f"{trainer}",
        "-f", f"{fold}",
    ]

    log.info(f"Predict command: {predict_command}")

    try:
        result = subprocess.run(predict_command, check=True, capture_output=True, text=True, encoding='utf-8')
        log.info(f"Command output: {result.stdout}")
        log.info(f"Command error (if any): {result.stderr}")
        
    except subprocess.CalledProcessError as e:
        log.error(f"Command failed with error: {e.stderr}")