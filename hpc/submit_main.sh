#!/bin/bash
#BSUB -J aek
#BSUB -q gpuv100
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "rusage[mem=3GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o ./batch_output/python_%J.out 
#BSUB -e ./batch_output/python_%J.err 

# Initialize conda environment
lscpu
nvidia-smi
source env_coronary_arteries.sh

# Set environment variables
export nnUNet_raw="/work3/s193396/v1/nnUNet_raw"
export nnUNet_preprocessed="/work3/s193396/v1/nnUNet_preprocessed"
export nnUNet_results="/work3/s193396/v1/nnUNet_results"

# Main script
python -u ./src/main.py
