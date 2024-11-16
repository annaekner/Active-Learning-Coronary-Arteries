import hydra
import logging

from set_environment_variables_ import set_environment_variables
from plan_and_preprocess_ import plan_and_preprocess
from train_ import train
from predict_ import predict
from evaluate_ import evaluate
from select_samples_for_retraining_ import select_samples_for_retraining
from move_files_ import move_files

# Set up logging
log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.3.2")
def main(config):

    # Export environment variables 
    set_environment_variables(config, log)

    # Number of iterations
    num_iterations = 5

    for iteration in range(num_iterations):

        log.info(f"Starting iteration {iteration}....")

        # --------------------------------------- STEP 1: Plan and preprocess --------------------------------------- #
        # Run nnUNetv2_plan_and_preprocess command
        plan_and_preprocess(config, log)

        # ---------------------------------------------- STEP 2: Train ---------------------------------------------- #
        # Run nnUNetv2_train command
        train(config, log)

        # --------------------------------------------- STEP 3: Predict --------------------------------------------- #
        # Run nnUNetv2_predict command
        predict(config, log, iteration)

        # -------------------------------------------- STEP 4: Evaluate --------------------------------------------- #
        evaluation_metrics_all = evaluate(config, log, iteration)

        # # ----------------------------------- STEP 5: Select samples for retraining ---------------------------------- #
        retraining = select_samples_for_retraining(evaluation_metrics_all, config, log)

        # -------------------------------------- STEP 6: Update and move files -------------------------------------- #
        move_files(retraining, config, log, iteration)

if __name__ == "__main__":
    main()

