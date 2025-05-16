#!/bin/sh 
#BSUB -q gpuv100
#BSUB -J AL-VLM-ITL
#BSUB -n 4 
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 24:00 
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o logs/hpc/%J.out 
#BSUB -e logs/hpc/%J.err 

export HF_HOME="/work3/s204075/.cache"
export WANDB_DIR="/work3/s204075/.cache"

source .venv/bin/activate

python3 scripts/train_active_learning_overlap.py --config configs/active_learning_config.yaml