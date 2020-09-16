#!/usr/bin/env bash

#SBATCH --array=1-3
#SBATCH --time=50:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

srun python ${SLURM_ARRAY_TASK_ID}D.py
