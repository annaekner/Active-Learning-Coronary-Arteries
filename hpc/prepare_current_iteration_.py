import os

def prepare_current_iteration(config, log, iteration):

    # Configuration settings
    base_dir = config.base_settings.base_dir
    version = config.base_settings.version
    dataset_name = config.dataset_settings.dataset_name

    data_iterations_dir = config.data_iterations.dir
    iterations_results_dir = config.data_iterations.results_dir
    iterations_predictions_dir = config.data_iterations.predictions_dir
    iterations_evaluations_dir = config.data_iterations.evaluations_dir

    log.info(f'------------------------- Prepare current iteration ------------------------')

    # Create a new folder in data_iterations_dir for current iteration
    current_iteration_folder = f"{base_dir}/{version}/{data_iterations_dir}/iteration_{iteration}"
    os.mkdir(current_iteration_folder)

    # Create subfolders for results, predictions, and evaluations
    os.mkdir(f"{current_iteration_folder}/{iterations_results_dir}")
    os.mkdir(f"{current_iteration_folder}/{iterations_predictions_dir}")
    os.mkdir(f"{current_iteration_folder}/{iterations_evaluations_dir}")

    log.info(f'Iteration: {iteration}')
    log.info(f'Results folder has been made: "~/iteration_{iteration}/{iterations_results_dir}"')
    log.info(f'Predictions folder has been made: "~/iteration_{iteration}/{iterations_predictions_dir}"')
    log.info(f'Evaluations folder has been made: "~/iteration_{iteration}/{iterations_evaluations_dir}"')
    log.info(f'----------------------------------------------------------------------------\n')