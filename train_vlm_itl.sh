#!/bin/sh 
#BSUB -q gpuv100
#BSUB -J FULL-VLM-ITL
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

python3 scripts/train_vlm_itl.py --config configs/vlm_itl_config.yaml