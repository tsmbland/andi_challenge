#!/usr/bin/env bash

#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

srun python 1D.py
