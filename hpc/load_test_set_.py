def load_test_set(config, log):

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version

    # Paths
    experiment_dir = f"{base_dir}/{version}"
    txt_filename = "test_img_indices.txt"
    txt_path = f"{experiment_dir}/{txt_filename}"
    
    # Load the .txt file
    with open(txt_path, "r") as file:

        test_img_indices = [int(line.strip()) for line in file]

    return test_img_indices