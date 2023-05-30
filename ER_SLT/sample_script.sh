#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres gpu
#SBATCH --time=24:00:00
#SBATCH -o output_1a.txt
#SBATCH -e error_1a.txt

srun python3 sample_runs.py
