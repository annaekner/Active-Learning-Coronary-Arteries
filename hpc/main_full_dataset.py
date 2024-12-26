import hydra
import logging

from set_environment_variables_ import set_environment_variables
from check_data_files_ import check_data_files
from prepare_initial_training_full_dataset_ import prepare_initial_training_full_dataset
from load_test_set_ import load_test_set
from prepare_current_iteration_ import prepare_current_iteration
from plan_and_preprocess_ import plan_and_preprocess
from train_ import train
from predict_ import predict
from evaluate_test_set_ import evaluate_test_set
from evaluate_unlabeled_set_ import evaluate_unlabeled_set
from select_samples_for_retraining_ import select_samples_for_retraining
from prepare_next_iteration_full_dataset_ import prepare_next_iteration_full_dataset

# Set up logging
log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config_full_dataset_70samples_v2.yaml", version_base="1.3.2")
def main(config):

    # Export environment variables 
    set_environment_variables(config, log)

    # Check that the data files look fine
    check_data_files(config, log)

    # Load the test set image indices (consist across iterations and experiments)
    test_img_indices = load_test_set(config, log)

    # Prepare files for initial training
    prepare_initial_training_full_dataset(test_img_indices, config, log)

    # Number of iterations
    num_iterations = 5

    for iteration in range(num_iterations):

        # Run nnUNetv2_plan_and_preprocess command
        plan_and_preprocess(config, log)

        # ------------------------------------ STEP 1: Prepare current iteration ------------------------------------ #
        prepare_current_iteration(config, log, iteration)

        # ---------------------------------------------- STEP 2: Train ---------------------------------------------- #
        # Run nnUNetv2_train command
        train(config, log, iteration)

        # --------------------------------------------- STEP 3: Predict --------------------------------------------- #
        # Run nnUNetv2_predict command
        predict(config, log, iteration)

        # ---------------------------------------- STEP 4: Evaluate test set ---------------------------------------- #
        evaluation_metrics_test = evaluate_test_set(test_img_indices, config, log, iteration)

        # -------------------------------------- STEP 5: Evaluate unlabeled set ------------------------------------- #
        # evaluation_metrics_unlabeled = evaluate_unlabeled_set(test_img_indices, config, log, iteration)

        # ----------------------------------- STEP 5: Select samples for retraining --------------------------------- #
        # retraining = select_samples_for_retraining(evaluation_metrics_unlabeled, config, log, iteration)

        # -------------------------------------- STEP 6: Prepare next iteration ------------------------------------- #
        prepare_next_iteration_full_dataset(test_img_indices, config, log, iteration)

if __name__ == "__main__":
    main()

