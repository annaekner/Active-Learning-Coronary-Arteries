#!/bin/bash
#BSUB -J it1_v5
#BSUB -q gpuv100
#BSUB -W 08:00
#BSUB -n 4
#BSUB -R "rusage[mem=3GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o ./batch_output/experiment_best_70samples_v5_it1_%J.out 
#BSUB -e ./batch_output/experiment_best_70samples_v5_it1_%J.err 

# Initialize conda environment
lscpu
nvidia-smi
source env_coronary_arteries.sh

# Main script
python -u ./src/main.py