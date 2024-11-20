#!/bin/bash
#BSUB -J aek
#BSUB -q gpuv100
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "rusage[mem=6GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o ./batch_output/experiment_worst_70samples_v1_%J.out 
#BSUB -e ./batch_output/experiment_worst_70samples_v1_%J.err 

# Initialize conda environment
lscpu
nvidia-smi
source env_coronary_arteries.sh

# Main script
python -u ./src/main.py