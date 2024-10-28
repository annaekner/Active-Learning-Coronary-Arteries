import hydra
import logging
import subprocess

from plan_and_preprocess_ import plan_and_preprocess
from train_ import train
from predict_ import predict

# Set up logging
log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.3.2")
def main(config):

    # Number of iterations
    num_iterations = 1

    for iteration in range(num_iterations):

        # --------------------------------------- STEP 1: Plan and preprocess --------------------------------------- #
        # Run nnUNetv2_plan_and_preprocess command
        plan_and_preprocess(config, log)

        # ---------------------------------------------- STEP 2: Train ---------------------------------------------- #
        # Run nnUNetv2_train command
        train(config, log)

        # --------------------------------------------- STEP 3: Predict --------------------------------------------- #
        predict(config, log)

        # -------------------------------------------- STEP 4: Evaluate --------------------------------------------- #
        # ----------------------------------- STEP 5: Select sample for retraining ---------------------------------- #
        # -------------------------------------- STEP 6: Update and move files -------------------------------------- #

if __name__ == "__main__":
    main()